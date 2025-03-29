#!/usr/bin/env python3
import os
import sys
import asyncio
import json
from typing import Dict, List, Any, Optional, AsyncIterator, Callable
from contextlib import asynccontextmanager
import datetime

from mcp.server.fastmcp import FastMCP
from mcp.server.lowlevel import NotificationOptions, Server
from mcp.server.models import InitializationOptions
from mcp.types import TextContent, PromptMessage
from pydantic import BaseModel, Field

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

## Tool Usage Patterns

### For Exploration
1. First use `list_lesson_containers` or `list_project_containers` to see what exists
2. Use `search_nodes` with semantic queries to find relevant knowledge
3. Explore relationships between entities to discover connected information

### For Knowledge Creation
1. First check if similar knowledge exists to avoid duplication
2. Create containers before adding detailed entities
3. Always add meaningful observations to provide context
4. Establish relationships to connect new knowledge to existing memories

### For Knowledge Application
1. Retrieve relevant memories at the start of complex tasks
2. Reference specific memories when making recommendations
3. Update confidence scores when validating information
4. Create new lessons when encountering novel situations

## Implementation Examples

```python
# Creating experiential knowledge
manager.create_lesson_container({"title": "Effective Error Handling Patterns"})
manager.create_lesson_section({
    "lesson_id": "Effective Error Handling Patterns",
    "title": "Try-Except-Finally Pattern",
    "content": "Always use specific exception types rather than catching all exceptions",
    "confidence": 0.9
})

# Establishing knowledge relationships
manager.create_lesson_relationship({
    "source_id": "Effective Error Handling Patterns",
    "target_id": "Python Best Practices",
    "relationship_type": "BUILDS_ON"
})
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
from src.tools import register_core_tools
from src.tools import register_lesson_tools
from src.tools import register_project_tools
from src.tools import register_config_tools

# Custom registration that uses client-specific managers
def register_all_tools_with_isolation(server):
    """Register all tools with client isolation."""
    # We'll modify the tools registration to use get_client_manager() inside each tool
    register_core_tools(server, get_client_manager)
    register_lesson_tools(server, get_client_manager)
    register_project_tools(server, get_client_manager)
    register_config_tools(server, get_client_manager)

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