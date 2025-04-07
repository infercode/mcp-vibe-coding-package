#!/usr/bin/env python3
import os
import sys
import asyncio
import json
from typing import Dict, List, Any, Optional, AsyncIterator, Callable, Union
from contextlib import asynccontextmanager
import datetime

from mcp.server.fastmcp import FastMCP
from mcp.server.lowlevel import NotificationOptions, Server
from mcp.server.models import InitializationOptions
from mcp.types import TextContent, PromptMessage
from pydantic import BaseModel, Field, ConfigDict

from src.graph_memory import GraphMemoryManager
from src.logger import LogLevel, get_logger
from src.utils import dict_to_json, dump_neo4j_nodes
from src.tools import register_all_tools
from src.session_manager import SessionManager

# Initialize logger
logger = get_logger()
# Set to DEBUG for detailed logging - change to INFO in production
logger.set_level(LogLevel.DEBUG)
logger.info("Initializing Neo4j MCP Graph Memory Server", context={"version": "1.0.0"})

# Store of client-specific GraphMemoryManager instances
client_managers = {}

# Create session manager with default settings
# 1 hour inactive timeout, 5 minutes cleanup interval
session_manager = SessionManager(inactive_timeout=3600, cleanup_interval=300)

# Lifespan context manager for Neo4j connections
@asynccontextmanager
async def server_lifespan(server: Server) -> AsyncIterator[dict]:
    """Manage Neo4j connection lifecycle and session cleanup."""
    # Start the session cleanup task
    logger.info("Initializing client manager store")
    await session_manager.start_cleanup_task()
    
    try:
        yield {"client_managers": client_managers, "session_manager": session_manager}
    finally:
        # Stop the cleanup task
        await session_manager.stop_cleanup_task()
        
        # Clean up at shutdown - close all client connections
        logger.info("Shutting down all client Neo4j connections")
        for client_id, manager in list(client_managers.items()):
            logger.info(f"Closing Neo4j connection for client {client_id}")
            manager.close()
            # Remove from the dictionary after closing
            client_managers.pop(client_id, None)

# The MCP instructions for AI agents
MCP_INSTRUCTIONS = """
# Enhanced MCP Instructions for Graph Memory System

## Core Memory Systems Overview

The MCP server provides two sophisticated memory systems that mirror human cognitive patterns:

1. **Lesson Memory** - For experiential knowledge and insights gained from successes, mistakes, and observations
2. **Project Memory** - For structured, hierarchical project knowledge organization

## Human-Like Memory Usage Patterns

### ⚡ General Principles
- Prioritize high-confidence memories when making critical decisions
- Leverage related memories through association, not just direct lookup
- Build knowledge incrementally, connecting new observations to existing memories
- Revise and update memories when you encounter contradicting information
- Record not just what was learned, but why it matters and how to apply it

### 🧠 Lesson Memory Usage
- Begin sessions by searching existing lessons related to the current task
- Record important insights during the session, even if they seem minor
- Use confidence scores to indicate certainty level (0.1-1.0)
- Create relationships between connected lessons to build knowledge networks
- Version lessons when you gain deeper understanding rather than creating duplicates
- Categorize lessons with meaningful tags for future discovery

### 🏗️ Project Memory Usage
- Start by exploring or creating project structure before diving into details
- Organize knowledge hierarchically from project → component → domain entities
- Track dependencies between components to understand system architecture
- Record design decisions with rationales to preserve context
- Link project entities to relevant lessons to apply experiential knowledge

## Key Relationship Types and Their Uses

### Lesson Memory Relationships
- **BUILDS_ON**: Connect lessons that extend or enhance previous knowledge
- **SUPERSEDES**: Mark newer lessons that replace outdated information
- **CONTRADICTS**: Link lessons that provide opposing viewpoints or findings
- **ORIGINATED_FROM**: Track the source of a lesson (problem, experience, etc.)
- **SOLVED_WITH**: Connect lessons to the solutions they generated

### Project Memory Relationships
- **CONTAINS**: Hierarchical structure relationships (Project contains Components)
- **IMPLEMENTS**: Feature implements a Requirement or Specification
- **DEPENDS_ON**: Mark dependencies between components or features
- **LEADS_TO**: Decision or action results in certain outcomes
- **ALTERNATIVE_TO**: Represent different options for the same purpose

## Common Workflow Examples

### Problem-Solving Workflow
1. When encountering a problem, search for similar issues: `search_nodes("error handling best practices")`
2. Retrieve and apply relevant lessons: `get_lesson_section("Effective Error Handling Patterns")`
3. Document new solutions: `create_lesson_section({"lesson_id": "...", "title": "Solution to X", ...})`
4. Connect to existing knowledge: `create_lesson_relationship({source_id: "New Solution", target_id: "Existing Pattern", ...})`

### Project Documentation Workflow
1. Create project structure: `create_project_container({"name": "New Project"})`
2. Define components: `create_component({"project_id": "New Project", "name": "Authentication Service"})`
3. Document relationships: `create_component_relationship({source_id: "Frontend", target_id: "API Service", relationship_type: "DEPENDS_ON"})`
4. Record design decisions: `create_domain_entity({project_id: "New Project", type: "DECISION", name: "Use JWT for Auth", ...})`

## Memory Maintenance

### When to Prune and Consolidate
- Mark lessons as obsolete when technology or approaches change significantly
- Consolidate similar lessons that address the same topic with the SUPERSEDES relationship
- Archive project components that are no longer relevant while preserving key insights
- Periodically review low-confidence memories and either validate or revise them

### Consolidation Process
1. Identify fragmented knowledge with `search_nodes("relevant topic")`
2. Create a consolidated lesson: `create_lesson_section({comprehensive information})`
3. Link old lessons to new one with SUPERSEDES relationship
4. Update confidence scores for the consolidated knowledge

## Effective Search Strategies

### Semantic Search Tips
- Use descriptive phrases rather than keywords: "handling async errors in React" vs "React errors"
- Include context in your queries: "database indexing for high-volume reads"
- Search for solutions and problems: "solutions for memory leaks in long-running processes"
- Use comparative queries: "differences between REST and GraphQL approaches"

### Combining Search Results
- Look for common patterns across multiple search results
- Pay attention to confidence scores when evaluating contradictory information
- Follow relationship links to discover related knowledge not directly matched in search

## Memory Reflection

### Periodic Knowledge Review
- Schedule regular reviews of high-value memory containers
- Ask questions like "What patterns emerge from recent lessons?"
- Look for gaps in knowledge that should be filled
- Re-evaluate confidence levels based on new experiences
- Identify memories that should be linked but currently aren't connected

## Advanced Human-Like Memory Techniques

### Contextual Retrieval
- Before starting a task, retrieve memories based on context, not just keywords
- Consider the current project environment when searching for relevant knowledge
- Use environmental cues (codebase structure, problem domain) to guide memory search
- Proactively surface relevant memories even when not explicitly asked
- Example: `search_nodes("authentication patterns in ${current_language} ${current_framework}")`

### Memory Prioritization
- Distinguish between critical architectural knowledge and routine implementation details
- Assign higher priority to foundational decisions that impact multiple components
- Tag memories with importance levels: "foundational", "architectural", "implementation-detail"
- When providing recommendations, lead with the most critical knowledge first
- Example: `create_domain_entity({project_id: "Project", type: "DECISION", importance: "architectural", ...})`

### Memory Reinforcement
- Increase confidence scores when information is validated through successful application
- After applying knowledge successfully, update the confidence score:
  ```
  update_lesson_section({
      "lesson_id": "Lesson Name",
      "section_id": "Section ID",
      "confidence": 0.95,  // Increased from previous value
      "reinforcement_note": "Successfully applied in Project X"
  })
  ```
- Track which memories are repeatedly useful for solving problems
- Create explicit "validation" observations that document successful applications

### Associative Memory
- Create connections between memories based on conceptual similarity, not just direct relationships
- Use relationship types like "RELATED_TO" when concepts share conceptual ground
- When retrieving one concept, also retrieve associated concepts that might be relevant
- Create cross-domain links between technical concepts and business processes
- Example: `create_lesson_relationship({source_id: "Caching Strategy", target_id: "Performance Optimization", relationship_type: "CONCEPTUALLY_RELATED"})`

### Memory Integration
- Combine fragments from multiple memories to solve new problems
- Document how different lessons were combined to create a novel solution:
  ```
  create_lesson_section({
      "lesson_id": "Integrated Solutions",
      "title": "Combining Caching and Async Processing",
      "content": "By integrating concepts from 'Redis Caching Patterns' and 'Async Queue Processing', we created...",
      "source_lessons": ["Redis Caching Patterns", "Async Queue Processing"],
      "confidence": 0.85
  })
  ```
- Create explicit integration relationships to show how knowledge was combined
- When facing novel problems, systematically explore how existing memories could be combined

### Narrative Structure
- Record the "story" behind important decisions, not just the decisions themselves
- Include the problem context, alternatives considered, and reasoning behind choices
- Structure narratives with beginning (problem), middle (exploration), and end (solution)
- Use observations to build a timeline of how understanding evolved:
  ```
  create_component({
      "project_id": "Project",
      "name": "Authentication Service",
      "narrative": {
          "initial_problem": "Needed secure, scalable auth for microservices",
          "alternatives_considered": ["Custom JWT", "Auth0", "Keycloak"],
          "decision_factors": ["Cost", "Security", "Integration effort"],
          "outcome": "Selected Keycloak for enterprise features and existing expertise"
      }
  })
  ```
- Link narratives across different projects to show evolving understanding

## Implementation Examples

```
# Creating experiential knowledge
create_lesson_container({
    "title": "Effective Error Handling Patterns",
    "description": "Lessons learned about proper error handling across different programming languages",
    "tags": ["error-handling", "best-practices", "programming"]
})

create_lesson_section({
    "lesson_id": "Effective Error Handling Patterns",
    "title": "Try-Except-Finally Pattern",
    "content": "Always use specific exception types rather than catching all exceptions",
    "confidence": 0.9
})

# Establishing knowledge relationships
create_lesson_relationship({
    "source_id": "Effective Error Handling Patterns",
    "target_id": "Python Best Practices",
    "relationship_type": "BUILDS_ON"
})

# Organizing project knowledge
create_project_container({
    "name": "API Gateway Refactoring",
    "description": "Knowledge about the API gateway refactoring project",
    "tags": ["api", "architecture", "refactoring"]
})

create_component({
    "project_id": "API Gateway Refactoring",
    "name": "Authentication Service",
    "component_type": "microservice",
    "description": "Handles user authentication and JWT token management"
})

create_component_relationship({
    "source_id": "Authentication Service",
    "target_id": "User Database",
    "relationship_type": "DEPENDS_ON",
    "properties": {
        "criticality": "high",
        "access_pattern": "read-write"
    }
})

# Consolidating fragmented knowledge
create_lesson_section({
    "lesson_id": "Comprehensive API Security",
    "title": "Unified API Security Approaches",
    "content": "This consolidates best practices from multiple lessons...",
    "confidence": 0.95
})

create_lesson_relationship({
    "source_id": "Comprehensive API Security",
    "target_id": "JWT Authentication Basics",
    "relationship_type": "SUPERSEDES"
})

# Searching and retrieving knowledge
search_nodes("error handling patterns in asynchronous code")

# Getting memory status
get_memory_status()
```

Remember to always connect, contextualize, and consolidate memories as you work, just as a human would do with their growing knowledge base 🌱
"""

# Create FastMCP server with enhanced capabilities and instructions for AI agents
server = FastMCP(
    name="mem0-graph-memory-server",
    notification_options=NotificationOptions(),  # Use default options
    experimental_capabilities={"graph_memory": True},
    lifespan=server_lifespan,
    instructions=MCP_INSTRUCTIONS
)

logger.debug("FastMCP server created with enhanced capabilities and client instructions")

# Add client tracking middleware for SSE connections
async def client_tracking_middleware(request, call_next):
    """Middleware to track client sessions and mark disconnections."""
    # Extract session ID from request
    session_id = request.query_params.get("session_id", None)
    
    # Mark client activity
    if session_id:
        logger.debug(f"Client activity: {session_id}")
        session_manager.update_activity(session_id)
    
    # Process the request
    response = await call_next(request)
    
    # Handle disconnection event for SSE requests
    if session_id and request.url.path == "/sse":
        # In SSE, we need to set up background cleanup for when the connection ends
        async def on_disconnect():
            try:
                # Small delay to ensure cleanup happens after the connection is fully closed
                await asyncio.sleep(1)
                logger.info(f"Client disconnected: {session_id}")
                session_manager.mark_client_inactive(session_id)
            except Exception as e:
                logger.error(f"Error during disconnect handling for {session_id}: {str(e)}")
        
        response.background = on_disconnect()
    
    return response

# Define a standard error response structure
class ErrorResponse:
    @staticmethod
    def create(message: str, code: str = "internal_error", details: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Create a standardized error response."""
        response = {
            "status": "error",
            "error": {
                "code": code,
                "message": message
            },
            "timestamp": datetime.datetime.now().isoformat()
        }
        if details:
            response["error"]["details"] = details
        return response

# Models for MCP tools
class Entity(BaseModel):
    """Entity in the knowledge graph with improved type annotations."""
    name: str = Field(..., description="The name of the entity")
    entityType: str = Field(..., description="The type of the entity")
    observations: List[str] = Field(..., description="An array of observation contents associated with the entity")
    
    def to_text_content(self) -> TextContent:
        """Convert to MCP TextContent format."""
        content = f"Entity: {self.name}\nType: {self.entityType}\n"
        if self.observations:
            content += "Observations:\n" + "\n".join([f"- {obs}" for obs in self.observations])
        return TextContent(type="text", text=content)

class Relation(BaseModel):
    """Relation between entities in the knowledge graph with improved type annotations."""
    from_entity: str = Field(..., description="The name of the source entity", alias="from")
    to_entity: str = Field(..., description="The name of the target entity", alias="to")
    relationType: str = Field(..., description="The type of the relation")
    
    # TODO: Update to model_config when we can ensure compatibility
    # @deprecated in Pydantic v2, will be removed in v3
    class Config:
        populate_by_name = True
        allow_population_by_field_name = True
    
    def to_text_content(self) -> TextContent:
        """Convert to MCP TextContent format."""
        return TextContent(
            type="text", 
            text=f"Relation: {self.from_entity} --[{self.relationType}]--> {self.to_entity}"
        )

class Observation(BaseModel):
    """Observation about an entity in the knowledge graph with improved type annotations."""
    entity: str = Field(..., description="The name of the entity")
    content: str = Field(..., description="The content of the observation")
    
    def to_text_content(self) -> TextContent:
        """Convert to MCP TextContent format."""
        return TextContent(
            type="text", 
            text=f"Observation for '{self.entity}': {self.content}"
        )

class CreateEntitiesRequest(BaseModel):
    entities: List[Entity] = Field(..., description="List of entities to create")

class CreateRelationsRequest(BaseModel):
    relations: List[Relation] = Field(..., description="List of relations to create")

class AddObservationsRequest(BaseModel):
    observations: List[Observation] = Field(..., description="List of observations to add")

class SearchNodesRequest(BaseModel):
    query: str = Field(..., description="Query string to search for")
    limit: int = Field(10, description="Maximum number of results to return")

class DeleteEntityRequest(BaseModel):
    entity: str = Field(..., description="The name of the entity to delete")

class DeleteRelationRequest(BaseModel):
    from_entity: str = Field(..., description="The name of the source entity", alias="from")
    to_entity: str = Field(..., description="The name of the target entity", alias="to")
    relationType: str = Field(..., description="The type of the relation")

class DeleteObservationRequest(BaseModel):
    entity: str = Field(..., description="The name of the entity")
    content: str = Field(..., description="The content of the observation")

class EmbeddingConfig(BaseModel):
    """Configuration for the embedding provider with improved documentation."""
    provider: str = Field(..., description="The embedding provider to use (openai, huggingface, ollama, azure_openai, vertexai, gemini, lmstudio)")
    model: Optional[str] = Field(None, description="The model to use for embeddings")
    api_key: Optional[str] = Field(None, description="API key for the provider")
    dimensions: Optional[int] = Field(None, description="Dimensions of the embedding model")
    client_id: Optional[str] = Field(None, description="Client identifier to save configuration")
    project_name: Optional[str] = Field(None, description="Project name to use for memory operations")
    config: Optional[Dict[str, Any]] = Field({}, description="Additional provider-specific configuration")

# Function to get or create a client-specific manager
def get_client_manager(client_id=None):
    """
    Get the GraphMemoryManager for the current client or create one if it doesn't exist.
    
    Args:
        client_id: Optional client ID to use. If not provided, a default client ID is used.
                 In a real implementation, this would be derived from the SSE connection.
    
    Returns:
        GraphMemoryManager instance for the client
    """
    try:
        # Use provided client ID or default
        effective_client_id = client_id or "default-client"
        
        logger.debug(f"Getting manager for client ID: {effective_client_id}")
        
        # Update the last activity time for this client
        session_manager.update_activity(effective_client_id)
        
        # Create a new manager if this client doesn't have one yet
        if effective_client_id not in client_managers:
            logger.info(f"Creating new GraphMemoryManager for client {effective_client_id}")
            manager = GraphMemoryManager(logger)
            manager.initialize()
            client_managers[effective_client_id] = manager
            
            # Register the client with the session manager
            session_manager.register_client(effective_client_id, manager)
            
        return client_managers[effective_client_id]
    except Exception as e:
        logger.error(f"Error getting client manager: {str(e)}")
        # Fall back to a temporary manager if something goes wrong
        return GraphMemoryManager(logger)

# Register all memory tools with client-specific manager handling
# from src.tools import register_core_tools
# from src.tools import register_lesson_tools
# from src.tools import register_project_tools
# from src.tools import register_config_tools

# Custom registration that uses client-specific managers
def register_all_tools_with_isolation(server):
    """Register all tools with client isolation."""
    # We'll modify the tools registration to use get_client_manager() inside each tool
    # register_core_tools(server, get_client_manager)
    # register_lesson_tools(server, get_client_manager)
    # register_project_tools(server, get_client_manager)
    # register_config_tools(server, get_client_manager)
    register_all_tools(server, get_client_manager)

# Register tools with client isolation
register_all_tools_with_isolation(server)

async def run_server():
    """Run the MCP server with the configured transport."""
    try:
        # Determine transport type from environment variable
        use_sse = os.environ.get("USE_SSE", "false").lower() == "true"
        port = int(os.environ.get("PORT", "8080"))

        if use_sse:
            # Using SSE transport
            logger.info(f"Neo4j Graph Memory MCP Server running with SSE on http://0.0.0.0:{port}")
            
            # Get the standard SSE app
            app = server.sse_app()
            
            # Add our middleware for client tracking
            from starlette.middleware.base import BaseHTTPMiddleware
            from starlette.applications import Starlette
            
            # Create a new Starlette app with middleware
            app_with_middleware = Starlette(routes=app.routes)
            app_with_middleware.add_middleware(BaseHTTPMiddleware, dispatch=client_tracking_middleware)
            
            return app_with_middleware
        else:
            # Using stdio transport
            logger.info("Neo4j Graph Memory MCP Server running on stdio")
            
            # Check if run is a coroutine
            if asyncio.iscoroutinefunction(server.run):
                # If it's a coroutine function, await it
                await server.run()
            else:
                # If it's not a coroutine function, just call it
                server.run()
            return None
    except Exception as e:
        logger.error(f"Failed to start server: {str(e)}")
        sys.exit(1)

def main():
    """Main entry point with enhanced error handling."""
    try:
        # Set Windows event loop policy if needed
        if sys.platform == 'win32':
            asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
        
        # Determine transport type
        use_sse = os.environ.get("USE_SSE", "false").lower() == "true"
        port = int(os.environ.get("PORT", "8080"))
        
        if use_sse:
            # For SSE, we need to run the server in a different way
            try:
                import uvicorn
                
                # Get the app with middleware
                app = asyncio.run(run_server())
                
                # Run the server if app was returned
                if app is not None:
                    # Type check to make sure app is an ASGI application
                    from starlette.applications import Starlette
                    if isinstance(app, Starlette):
                        uvicorn.run(app, host="0.0.0.0", port=port)
                    else:
                        logger.error("Invalid app type returned from run_server")
                        sys.exit(1)
                else:
                    logger.error("No app returned from run_server")
                    sys.exit(1)
            except ImportError:
                logger.error("uvicorn is required for SSE transport. Please install it with 'pip install uvicorn'.")
                sys.exit(1)
        else:
            # For stdio, we can use asyncio.run
            asyncio.run(run_server())
    except Exception as e:
        logger.error(f"Failed to run server: {str(e)}")
        sys.exit(1)

if __name__ == "__main__":
    main() 