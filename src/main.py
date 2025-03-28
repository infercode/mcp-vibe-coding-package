#!/usr/bin/env python3
import os
import sys
import asyncio
import json
from typing import Dict, List, Any, Optional, AsyncIterator
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

# Initialize logger
logger = get_logger()
# Set to DEBUG for detailed logging - change to INFO in production
logger.set_level(LogLevel.DEBUG)
logger.info("Initializing Neo4j MCP Graph Memory Server", context={"version": "1.0.0"})

# Create memory graph manager
graph_manager = GraphMemoryManager(logger)

# Lifespan context manager for Neo4j connections
@asynccontextmanager
async def server_lifespan(server: Server) -> AsyncIterator[dict]:
    """Manage Neo4j connection lifecycle."""
    # Initialize at startup
    logger.info("Initializing Neo4j connection in lifespan")
    graph_manager.initialize()
    try:
        yield {"graph_manager": graph_manager}
    finally:
        # Clean up at shutdown
        logger.info("Shutting down Neo4j connection")
        # Call close without awaiting it
        graph_manager.close()

# Create FastMCP server with enhanced capabilities
server = FastMCP(
    name="mem0-graph-memory-server",
    notification_options=NotificationOptions(),  # Use default options
    experimental_capabilities={"graph_memory": True},
    lifespan=server_lifespan
)
logger.debug("FastMCP server created with enhanced capabilities")

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

# Register all memory tools from the tools package
register_all_tools(server, graph_manager)

async def run_server():
    """Run the MCP server with the configured transport."""
    try:
        # Determine transport type from environment variable
        use_sse = os.environ.get("USE_SSE", "false").lower() == "true"
        port = int(os.environ.get("PORT", "8080"))

        if use_sse:
            # Using SSE transport
            logger.info(f"Neo4j Graph Memory MCP Server running with SSE on http://0.0.0.0:{port}")
            # For SSE, return the app so it can be run separately
            return server.sse_app()
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
                from starlette.applications import Starlette
                from starlette.routing import Mount
                
                # Get the app from run_server directly without asyncio.run
                app = Starlette(
                    routes=[
                        Mount('/', app=server.sse_app()),
                    ],
                    debug=True
                )
                
                # Run uvicorn synchronously (not inside an async function)
                uvicorn.run(app, host="0.0.0.0", port=port)
            except ImportError:
                logger.error("Uvicorn not installed. Required for SSE mode.")
                sys.exit(1)
        else:
            # For stdio, run the server asynchronously
            asyncio.run(run_server())
    except KeyboardInterrupt:
        logger.info("Server stopped by user")
    except Exception as e:
        logger.error(f"Unhandled exception: {str(e)}", exc_info=True)
        sys.exit(1)

# Start the server when run directly
if __name__ == "__main__":
    main() 