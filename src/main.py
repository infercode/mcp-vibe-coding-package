#!/usr/bin/env python3
import os
import sys
import asyncio
from typing import Dict, List, Any

from mcp.server.fastmcp import FastMCP
from pydantic import BaseModel, Field

from src.graph_manager import GraphMemoryManager
from src.logger import LogLevel, get_logger

# Initialize logger
logger = get_logger()
logger.set_level(LogLevel.INFO)

# Create memory graph manager
graph_manager = GraphMemoryManager(logger)

# Create FastMCP server
server = FastMCP(name="mem0-graph-memory-server")

# Models for MCP tools
class Entity(BaseModel):
    name: str = Field(..., description="The name of the entity")
    entityType: str = Field(..., description="The type of the entity")
    observations: List[str] = Field(..., description="An array of observation contents associated with the entity")

class Relation(BaseModel):
    from_: str = Field(..., alias="from", description="The name of the entity where the relation starts")
    to: str = Field(..., description="The name of the entity where the relation ends")
    relationType: str = Field(..., description="The type of the relation")

class Observation(BaseModel):
    entityName: str = Field(..., description="The name of the entity to add the observations to")
    contents: List[str] = Field(..., description="An array of observation contents to add")

class CreateEntitiesRequest(BaseModel):
    entities: List[Entity] = Field(..., description="List of entities to create")

class CreateRelationsRequest(BaseModel):
    relations: List[Relation] = Field(..., description="List of relations to create")

class AddObservationsRequest(BaseModel):
    observations: List[Observation] = Field(..., description="List of observations to add")

class DeleteEntitiesRequest(BaseModel):
    entityNames: List[str] = Field(..., description="An array of entity names to delete")

class DeleteObservationsRequest(BaseModel):
    deletions: List[Observation] = Field(..., description="List of observations to delete")

class DeleteRelationsRequest(BaseModel):
    relations: List[Relation] = Field(..., description="List of relations to delete")

class SearchRequest(BaseModel):
    query: str = Field(..., description="The search query to match against entity names, types, and observation content")

class OpenNodesRequest(BaseModel):
    names: List[str] = Field(..., description="An array of entity names to retrieve")

# Register tools
@server.tool()
def create_entities(entities: List[Dict[str, Any]]) -> str:
    """Create multiple new entities in the knowledge graph"""
    result = graph_manager.create_entities(entities)
    return result

@server.tool()
def create_relations(relations: List[Dict[str, Any]]) -> str:
    """Create multiple new relations between entities in the knowledge graph. Relations should be in active voice"""
    result = graph_manager.create_relations(relations)
    return result

@server.tool()
def add_observations(observations: List[Dict[str, Any]]) -> str:
    """Add new observations to existing entities in the knowledge graph"""
    result = graph_manager.add_observations(observations)
    return result

@server.tool()
def delete_entities(entityNames: List[str]) -> str:
    """Delete multiple entities and their associated relations from the knowledge graph"""
    graph_manager.delete_entities(entityNames)
    return "Entities deleted successfully"

@server.tool()
def delete_observations(deletions: List[Dict[str, Any]]) -> str:
    """Delete specific observations from entities in the knowledge graph"""
    graph_manager.delete_observations(deletions)
    return "Observations deleted successfully"

@server.tool()
def delete_relations(relations: List[Dict[str, Any]]) -> str:
    """Delete multiple relations from the knowledge graph"""
    graph_manager.delete_relations(relations)
    return "Relations deleted successfully"

@server.tool()
def search_nodes(query: str) -> str:
    """Search for nodes in the knowledge graph based on a query"""
    result = graph_manager.search_nodes(query)
    return result

@server.tool()
def open_nodes(names: List[str]) -> str:
    """Open specific nodes in the knowledge graph by their names"""
    result = graph_manager.open_nodes(names)
    return result

def main():
    try:
        # Initialize graph manager
        graph_manager.initialize()

        # Determine transport type from environment variable
        use_sse = os.environ.get("USE_SSE", "false").lower() == "true"
        port = int(os.environ.get("PORT", "8080"))

        if use_sse:
            # Using SSE transport
            try:
                import uvicorn
                logger.info(f"Mem0 Graph Memory MCP Server running with SSE on http://0.0.0.0:{port}/sse")
                
                # Create ASGI app from server
                app = server.sse_app()
                
                # Run with uvicorn
                uvicorn.run(app, host="0.0.0.0", port=port)
            except ImportError:
                logger.error("Uvicorn not installed. Required for SSE mode.")
                sys.exit(1)
        else:
            # Using stdio transport
            logger.info("Mem0 Graph Memory MCP Server running on stdio")
            server.run()
    except Exception as e:
        logger.error(f"Failed to start server: {str(e)}")
        sys.exit(1)

if __name__ == "__main__":
    main() 