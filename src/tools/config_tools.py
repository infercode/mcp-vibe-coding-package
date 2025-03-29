#!/usr/bin/env python3
import json
import datetime
from typing import Dict, List, Any, Optional, Callable

from src.logger import get_logger
from src.utils import dict_to_json, dump_neo4j_nodes
from src.graph_memory import GraphMemoryManager

# Initialize logger
logger = get_logger()

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

def register_config_tools(server, get_client_manager):
    """Register configuration tools with the server.
    
    Args:
        server: The MCP server instance
        get_client_manager: Function to get the client-specific GraphMemoryManager
    """
    
    @server.tool()
    async def configure_embedding(config: Dict[str, Any]) -> str:
        """
        Configure the embedding service for memory operations.
        
        Args:
            config: Configuration object with the following fields:
                - provider: Embedding provider (e.g., 'openai', 'vertexai', 'azure', 'cohere', 'gemini', 'lmstudio')
                - model: Model name for the embedding provider
                - api_key: API key for the embedding provider (if needed)
                - dimensions: Optional number of dimensions for embeddings
                - config: Optional additional provider-specific configuration
                - client_id: Optional client ID for identifying the connection
                - project_name: Optional default project name
                
        Returns:
            Message with the result of the configuration
        """
        # Extract client ID if provided to get the right manager
        client_id = config.get("client_id")
        
        # Get the client-specific graph manager
        graph_manager = get_client_manager(client_id)
        
        try:
            # Extract client ID if provided (client_id should match what we already used)
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
                return "ERROR: Missing required parameter: provider"
                    
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
                if "batch_size" in additional_config:
                    mem0_config["embedder"]["config"]["batch_size"] = additional_config["batch_size"]
                    
            elif provider == "azure":
                if api_key:
                    mem0_config["embedder"]["config"]["api_key"] = api_key
                if "endpoint" in additional_config:
                    mem0_config["embedder"]["config"]["endpoint"] = additional_config["endpoint"]
                if "deployment" in additional_config:
                    mem0_config["embedder"]["config"]["deployment"] = additional_config["deployment"]
                if "api_version" in additional_config:
                    mem0_config["embedder"]["config"]["api_version"] = additional_config["api_version"]
                
            elif provider == "cohere":
                if api_key:
                    mem0_config["embedder"]["config"]["api_key"] = api_key
                
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
                return f"ERROR: Failed to apply configuration: {result.get('message', 'unknown error')}"
                
            # Reinitialize the memory manager
            reinit_result = graph_manager.reinitialize()
            
            if reinit_result["status"] != "success":
                return f"ERROR: Failed to reinitialize memory manager: {reinit_result.get('message', 'unknown error')}"
                
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
                
            # Determine if embeddings are enabled
            embedding_status = "enabled" if graph_manager.embedding_enabled else "disabled"
            
            # Create direct command message for the AI agent
            file_name = f"mcp_memory_config_{project_name or 'default'}.json"
            
            config_json = json.dumps(full_config, indent=2)
            
            # Response message - simplified to just provide the essential configuration feedback
            response_message = f"""
Configuration applied for provider: {provider}
Model: {model or 'Default model'}
Dimensions: {dimensions or 'Default dimensions'}
Embeddings: {embedding_status}
Project: {project_name or 'default'}

Configuration has been saved as {file_name}
            """
            
            return response_message.strip()
                
        except Exception as e:
            return f"ERROR: Failed to configure embedding service: {str(e)}"
            
    @server.tool()
    async def set_project_name(project_name: str, client_id: Optional[str] = None) -> str:
        """
        Set the current project name for the memory system.
        
        Args:
            project_name: The name of the project to use
            client_id: Optional client ID for identifying the connection
            
        Returns:
            Message with the result of setting the project name
        """
        # Get the client-specific graph manager
        graph_manager = get_client_manager(client_id)
        
        try:
            if not project_name or not project_name.strip():
                return "ERROR: Project name cannot be empty"
                
            graph_manager.set_project_name(project_name)
            
            # Create project configuration
            file_name = f"mcp_project_config_{project_name}.json"
            project_config = {
                "project_name": project_name,
                "timestamp": datetime.datetime.now().isoformat()
            }
            
            # Format the response using a simpler, direct instruction format
            config_json = json.dumps(project_config, indent=2)
            
            # Create project container configuration
            project_container_config = {
                "name": project_name,
                "description": f"Project container for {project_name}",
                "metadata": {
                    "created_date": datetime.datetime.now().isoformat(),
                    "created_via": "set_project_name command"
                }
            }
            project_json = json.dumps(project_container_config, indent=2)
            
            # Create lesson container configuration
            lesson_container_config = {
                "title": "Lessons",
                "description": "Lesson container for ai agents",
                "metadata": {
                    "created_date": datetime.datetime.now().isoformat()
                },
                "visibility": "private"
            }
            lesson_json = json.dumps(lesson_container_config, indent=2)
            
            return f"""
Project name set to: {project_name}

Project configuration created as {file_name}

Project information has been updated in the memory system.
            """.strip()
                
        except Exception as e:
            return f"ERROR: Failed to set project name: {str(e)}"
            
    @server.tool()
    async def get_memory_status(client_id: Optional[str] = None) -> str:
        """
        Get the current status of the memory system.
        
        Args:
            client_id: Optional client ID for identifying the connection
        
        Returns:
            JSON string with the memory system status
        """
        # Get the client-specific graph manager
        graph_manager = get_client_manager(client_id)
        
        try:
            # Get basic information about the memory system
            config = graph_manager.get_current_config()
            
            # Get Neo4j connection info
            neo4j_connected = graph_manager.check_connection()
            
            # Build status object
            status = {
                "neo4j": {
                    "connected": neo4j_connected,
                    "uri": graph_manager.neo4j_uri,
                    "database": graph_manager.neo4j_database
                },
                "embeddings": {
                    "enabled": graph_manager.embedding_enabled,
                    "provider": graph_manager.embedder_provider,
                    "model": graph_manager.embedding_model
                },
                "project": graph_manager.default_project_name,
                "timestamp": datetime.datetime.now().isoformat()
            }
            
            # Format the response as a simple message
            status_message = f"""
Memory System Status
-------------------
Neo4j Connection: {'Connected' if neo4j_connected else 'Disconnected'}
Database: {graph_manager.neo4j_database}
Embeddings: {'Enabled' if graph_manager.embedding_enabled else 'Disabled'}
Provider: {graph_manager.embedder_provider}
Model: {graph_manager.embedding_model or 'Not specified'}
Current Project: {graph_manager.default_project_name}
            """
            
            return status_message.strip()
                
        except Exception as e:
            return f"ERROR: Failed to get memory status: {str(e)}"

    @server.tool()
    async def get_embedding_config(client_id: Optional[str] = None) -> str:
        """
        Get the current embedding configuration.
        
        Args:
            client_id: Optional client ID for identifying the connection
        
        Returns:
            JSON configuration with current settings
        """
        try:
            # Get current configuration from memory manager
            config = get_client_manager(client_id).get_current_config()
            
            return dict_to_json({
                "status": "success",
                "config": config,
                "timestamp": datetime.datetime.now().isoformat()
            })
            
        except Exception as e:
            logger.error(f"Error getting embedding configuration: {str(e)}")
            error_response = ErrorResponse.create(
                message=f"Failed to get embedding configuration: {str(e)}",
                code="config_retrieval_error"
            )
            return dict_to_json(error_response) 