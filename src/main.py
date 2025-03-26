#!/usr/bin/env python3
import os
import sys
import asyncio
import json
from typing import Dict, List, Any, Optional
import datetime

from mcp.server.fastmcp import FastMCP
from pydantic import BaseModel, Field

from src.graph_manager import GraphMemoryManager
from src.logger import LogLevel, get_logger
from src.utils import dict_to_json

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
    from_entity: str = Field(..., description="The name of the source entity", alias="from")
    to_entity: str = Field(..., description="The name of the target entity", alias="to")
    relationType: str = Field(..., description="The type of the relation")
    
    class Config:
        populate_by_name = True
        allow_population_by_field_name = True

class Observation(BaseModel):
    entity: str = Field(..., description="The name of the entity")
    content: str = Field(..., description="The content of the observation")

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
    """Configuration for the embedding provider."""
    provider: str = Field(..., description="The embedding provider to use (openai, huggingface, ollama, azure_openai, vertexai, gemini, lmstudio)")
    model: Optional[str] = Field(None, description="The model to use for embeddings")
    api_key: Optional[str] = Field(None, description="API key for the provider")
    dimensions: Optional[int] = Field(None, description="Dimensions of the embedding model")
    client_id: Optional[str] = Field(None, description="Client identifier to save configuration")
    project_name: Optional[str] = Field(None, description="Project name to use for memory operations")
    config: Optional[Dict[str, Any]] = Field({}, description="Additional provider-specific configuration")

# Register tools
@server.tool()
def create_entities(entities: List[Dict[str, Any]]) -> str:
    """Create multiple new entities in the knowledge graph"""
    result = graph_manager.create_entities(entities)
    return result

@server.tool()
def create_relations(relations: List[Dict[str, Any]]) -> str:
    """Create multiple new relations in the knowledge graph"""
    result = graph_manager.create_relations(relations)
    return result

@server.tool()
def add_observations(observations: List[Dict[str, Any]]) -> str:
    """Add multiple observations to entities in the knowledge graph"""
    result = graph_manager.add_observations(observations)
    return result

@server.tool()
def search_nodes(query: str, limit: int = 10, user_id: Optional[str] = None) -> str:
    """
    Search for nodes in the knowledge graph.
    
    Args:
        query: Search query string
        limit: Maximum number of results to return (default: 10)
        user_id: Optional user/project ID to search in (defaults to current project)
    
    Returns:
        JSON string with search results
    """
    result = graph_manager.search_nodes(query, limit, user_id)
    return result

@server.tool()
def delete_entity(entity: str) -> str:
    """Delete an entity from the knowledge graph"""
    result = graph_manager.delete_entity(entity)
    return result

@server.tool()
def delete_relation(from_entity: str, to_entity: str, relationType: str) -> str:
    """Delete a relation from the knowledge graph"""
    result = graph_manager.delete_relation(from_entity, to_entity, relationType)
    return result

@server.tool()
def delete_observation(entity: str, content: str) -> str:
    """Delete an observation from the knowledge graph"""
    result = graph_manager.delete_observation(entity, content)
    return result

@server.tool()
def configure_embedding(config: Dict[str, Any]) -> str:
    """
    Configure the embedding provider for the knowledge graph and return the full configuration.
    
    This will:
    1. Update the memory manager configuration
    2. Set the project name for memory operations
    3. Reinitialize connections with the new configuration
    4. Return the complete configuration for the client to save locally
    
    Args:
        config: Configuration object with embedding provider details
        
    Returns:
        JSON string with the full configuration that can be saved by the client
    """
    try:
        # Extract client ID if provided
        client_id = config.pop("client_id", None)
        
        # Extract project name if provided
        project_name = config.pop("project_name", None)
        
        # Set project name if provided
        if project_name:
            graph_manager.set_project_name(project_name)
            
        # Convert flat config to mem0 expected structure
        provider = config.pop("provider", None)
        model = config.pop("model", None)
        api_key = config.pop("api_key", None)
        dimensions = config.pop("dimensions", None)
        additional_config = config.pop("config", {})
        
        if not provider:
            return dict_to_json({
                "status": "error",
                "message": "Missing required parameter: provider"
            })
            
        # Build proper configuration structure
        mem0_config = {
            "embedder": {
                "provider": provider,
                "config": {}
            }
        }
        
        # Add common parameters
        if model:
            mem0_config["embedder"]["config"]["model"] = model
        if dimensions:
            mem0_config["embedder"]["config"]["embedding_dims"] = dimensions
            
        # Add provider-specific parameters
        if provider == "openai":
            if api_key:
                mem0_config["embedder"]["config"]["api_key"] = api_key
            # Add other OpenAI specific configs from additional_config
            mem0_config["embedder"]["config"].update(additional_config)
            
        elif provider == "huggingface":
            if "model_kwargs" in additional_config:
                mem0_config["embedder"]["config"]["model_kwargs"] = additional_config["model_kwargs"]
                
        elif provider == "ollama":
            if "ollama_base_url" in additional_config:
                mem0_config["embedder"]["config"]["ollama_base_url"] = additional_config["ollama_base_url"]
                
        elif provider in ["azure", "azure_openai"]:
            azure_kwargs = {
                "api_version": additional_config.get("api_version", "2023-05-15")
            }
            
            if "azure_deployment" in additional_config:
                azure_kwargs["azure_deployment"] = additional_config["azure_deployment"]
            if "azure_endpoint" in additional_config:
                azure_kwargs["azure_endpoint"] = additional_config["azure_endpoint"]
            if api_key:
                azure_kwargs["api_key"] = api_key
            if "default_headers" in additional_config:
                azure_kwargs["default_headers"] = additional_config["default_headers"]
                
            mem0_config["embedder"]["config"]["azure_kwargs"] = azure_kwargs
            
        elif provider == "vertexai":
            if "vertex_credentials_json" in additional_config:
                mem0_config["embedder"]["config"]["vertex_credentials_json"] = additional_config["vertex_credentials_json"]
            if "memory_add_embedding_type" in additional_config:
                mem0_config["embedder"]["config"]["memory_add_embedding_type"] = additional_config["memory_add_embedding_type"]
            if "memory_update_embedding_type" in additional_config:
                mem0_config["embedder"]["config"]["memory_update_embedding_type"] = additional_config["memory_update_embedding_type"]
            if "memory_search_embedding_type" in additional_config:
                mem0_config["embedder"]["config"]["memory_search_embedding_type"] = additional_config["memory_search_embedding_type"]
                
        elif provider == "gemini":
            if api_key:
                mem0_config["embedder"]["config"]["api_key"] = api_key
                
        elif provider == "lmstudio":
            if "lmstudio_base_url" in additional_config:
                mem0_config["embedder"]["config"]["lmstudio_base_url"] = additional_config["lmstudio_base_url"]
        
        # Apply configuration to the memory manager
        result = graph_manager.apply_client_config(mem0_config)
        
        if result["status"] != "success":
            return dict_to_json(result)
            
        # Reinitialize the memory manager
        reinit_result = graph_manager.reinitialize()
        
        if reinit_result["status"] != "success":
            return dict_to_json(reinit_result)
            
        # Get the complete current configuration
        current_config = graph_manager.get_current_config()
        
        # Add Neo4j graph store config
        graph_store_config = {
            "graph_store": {
                "provider": "neo4j",
                "config": {
                    "url": graph_manager.neo4j_uri,
                    "username": graph_manager.neo4j_user,
                    "password": graph_manager.neo4j_password,
                    "database": graph_manager.neo4j_database
                }
            }
        }
        
        # Combine configurations
        full_config = {**graph_store_config, **mem0_config}
        
        # Add client ID if provided
        if client_id:
            full_config["client_id"] = client_id
            
        # Create the final response with instructions for the AI agent
        file_name = f"mcp_memory_config_{project_name or 'default'}.json"
        
        # Determine if embeddings are enabled
        embedding_status = "enabled" if graph_manager.embedding_enabled else "disabled"
        
        instructions = (
            f"IMPORTANT: Save this configuration to '{file_name}' in the root directory of your project. "
            f"This file will be used for future memory operations with the MCP server. "
            f"Embeddings are currently {embedding_status}. "
        )
        
        if not graph_manager.embedding_enabled:
            instructions += "Note that semantic search will not work until embeddings are configured."
        else:
            instructions += f"You should use this configuration whenever interacting with the memory graph for project '{graph_manager.default_user_id}'."
        
        response = {
            "status": "success",
            "message": f"Successfully configured embedding provider: {provider}",
            "provider": provider,
            "project_name": graph_manager.default_user_id,
            "embedding_enabled": graph_manager.embedding_enabled,
            "config": full_config,
            "instructions_for_agent": instructions,
            "file_name": file_name
        }
        
        # Return success result with configuration
        return dict_to_json(response)
        
    except Exception as e:
        logger.error(f"Error configuring embedding: {str(e)}")
        return dict_to_json({
            "status": "error",
            "message": f"Failed to configure embedding: {str(e)}"
        })

@server.tool()
def get_embedding_config() -> str:
    """
    Get the current embedding configuration.
    
    Returns:
        JSON string with the current configuration
    """
    try:
        # Get current configuration from memory manager
        config = graph_manager.get_current_config()
        
        return dict_to_json({
            "status": "success",
            "config": config
        })
        
    except Exception as e:
        logger.error(f"Error getting embedding configuration: {str(e)}")
        return dict_to_json({
            "status": "error",
            "message": f"Failed to get embedding configuration: {str(e)}"
        })

@server.tool()
def get_all_memories(user_id: Optional[str] = None) -> str:
    """
    Get all memories for a user/project.
    
    Args:
        user_id: Optional user/project identifier (defaults to current project)
        
    Returns:
        JSON string with all memories
    """
    try:
        return graph_manager.get_all_memories(user_id)
    except Exception as e:
        logger.error(f"Error getting all memories: {str(e)}")
        return dict_to_json({
            "status": "error",
            "message": f"Failed to get all memories: {str(e)}"
        })

@server.tool()
def set_project_name(project_name: str) -> str:
    """
    Set the project name for memory operations.
    
    This tool allows setting the current project name without needing to reconfigure embeddings.
    The project name is used as the default user_id for all memory operations when user_id is not explicitly provided.
    
    Args:
        project_name: The name of the project to use for memory operations
        
    Returns:
        JSON string with the result of the operation
    """
    try:
        if not project_name or not project_name.strip():
            return dict_to_json({
                "status": "error",
                "message": "Project name cannot be empty"
            })
            
        graph_manager.set_project_name(project_name)
        
        # Create instructions for the AI agent
        file_name = f"mcp_project_config_{project_name}.json"
        project_config = {
            "project_name": project_name,
            "timestamp": datetime.datetime.now().isoformat()
        }
        
        instructions = (
            f"IMPORTANT: Save this project configuration to '{file_name}' in the root directory of your project. "
            f"This file will help maintain project context between sessions. "
            f"You should use this project name whenever interacting with the memory graph."
        )
        
        return dict_to_json({
            "status": "success",
            "message": f"Project name set to: {project_name}",
            "project_name": project_name,
            "config": project_config,
            "instructions_for_agent": instructions,
            "file_name": file_name
        })
    except Exception as e:
        logger.error(f"Error setting project name: {str(e)}")
        return dict_to_json({
            "status": "error",
            "message": f"Failed to set project name: {str(e)}"
        })

@server.tool()
def delete_all_memories(user_id: Optional[str] = None) -> str:
    """
    Delete all memories for a user/project.
    
    Args:
        user_id: Optional user/project identifier (defaults to current project)
        
    Returns:
        JSON string with result information
    """
    try:
        return graph_manager.delete_all_memories(user_id)
    except Exception as e:
        logger.error(f"Error deleting all memories: {str(e)}")
        return dict_to_json({
            "status": "error",
            "message": f"Failed to delete all memories: {str(e)}"
        })

def main():
    try:
        # Initialize graph manager with default configuration
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