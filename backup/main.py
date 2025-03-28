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

from src.legacy_graph_manager import GraphMemoryManager
from src.logger import LogLevel, get_logger
from src.utils import dict_to_json, dump_neo4j_nodes

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

# Register tools with notification support
@server.tool()
async def create_entities(entities: List[Dict[str, Any]]) -> str:
    """
    Create multiple new entities in the knowledge graph.
    
    Each entity should have a name, entity type, and optional observations.
    The operation is transactional and will notify clients of changes.
    
    Args:
        entities: List of entity objects to create
            - name: Entity identifier
            - entityType: Type of entity (e.g., "Person", "Organization", "Component")
            - observations: List of facts or properties about the entity
    
    Returns:
        JSON response with operation result
    
    Example:
    ```
    [
      {
        "name": "Alice Smith",
        "entityType": "Person",
        "observations": ["Works as a software engineer", "Lives in San Francisco"]
      },
      {
        "name": "Google",
        "entityType": "Organization",
        "observations": ["Technology company founded in 1998"]
      }
    ]
    ```
    """
    logger.debug(f"Creating {len(entities)} entities", context={"entity_count": len(entities)})
    
    try:
        # Format entities to match the expected structure
        formatted_entities = []
        for entity in entities:
            # Extract the required fields or use defaults
            formatted_entity = {
                "name": entity.get("entity_id") or entity.get("name"),
                "entityType": entity.get("entity_type") or entity.get("entityType") or entity.get("type", "Unknown"),
                "observations": []
            }
            
            # Handle observations: could be in entity.observations or as separate metadata
            if "observations" in entity:
                formatted_entity["observations"] = entity["observations"]
            elif "content" in entity:
                formatted_entity["observations"] = [entity["content"]]
                
            # Add any metadata as observations if not already present
            if "metadata" in entity and isinstance(entity["metadata"], dict):
                for key, value in entity["metadata"].items():
                    content = f"{key}: {value}"
                    if content not in formatted_entity["observations"]:
                        formatted_entity["observations"].append(content)
            
            logger.debug(f"Formatted entity: {formatted_entity['name']}, type: {formatted_entity['entityType']}")
            formatted_entities.append(formatted_entity)
        
        # Pass the formatted entities to the graph manager
        result = graph_manager.create_entities(formatted_entities)
        
        # Process result 
        try:
            result_json = json.loads(result)
            
            logger.debug("Entities creation completed", context={"result": result[:100] + "..." if len(result) > 100 else result})
            return result
        except json.JSONDecodeError:
            logger.warn(f"Result not in JSON format: {result}")
            return result
            
    except Exception as e:
        logger.error(f"Error creating entities: {str(e)}", exc_info=True)
        error_response = ErrorResponse.create(
            message=f"Error creating entities: {str(e)}",
            code="entity_creation_error",
            details={"entity_count": len(entities)}
        )
        return dict_to_json(error_response)

@server.tool()
async def create_relations(relations: List[Dict[str, Any]]) -> str:
    """
    Create multiple new relations in the knowledge graph.
    
    Each relation connects a source entity to a target entity with a specific relationship type.
    The operation will notify clients of relation changes.
    
    Args:
        relations: List of relation objects
            - from: Source entity name
            - to: Target entity name
            - relationType: Type of relationship
    
    Returns:
        JSON response with operation result
    
    Example:
    ```
    [
      {
        "from": "Alice Smith",
        "to": "Google",
        "relationType": "WORKS_AT"
      },
      {
        "from": "Bob Jones",
        "to": "Alice Smith",
        "relationType": "KNOWS"
      }
    ]
    ```
    """
    logger.debug(f"Creating {len(relations)} relations", context={"relation_count": len(relations)})
    
    try:
        result = graph_manager.create_relations(relations)
        
        # Process result
        try:
            result_json = json.loads(result)
            logger.debug("Relations creation completed", context={"result": result[:100] + "..." if len(result) > 100 else result})
            return result
        except json.JSONDecodeError:
            logger.warn(f"Result not in JSON format: {result}")
            return result
            
    except Exception as e:
        logger.error(f"Error creating relations: {str(e)}", exc_info=True)
        error_response = ErrorResponse.create(
            message=f"Error creating relations: {str(e)}",
            code="relation_creation_error",
            details={"relation_count": len(relations)}
        )
        return dict_to_json(error_response)

@server.tool()
async def add_observations(observations: List[Dict[str, Any]]) -> str:
    """
    Add multiple observations to entities in the knowledge graph.
    
    Observations are facts or properties about an entity.
    The operation will notify clients of observation changes.
    
    Args:
        observations: List of observation objects
            - entity: Entity name the observation is about
            - content: The observation content
            - metadata: Optional metadata about the observation
    
    Returns:
        JSON response with operation result
    
    Example:
    ```
    [
      {
        "entity": "Alice Smith",
        "content": "Graduated from Stanford University in 2015"
      },
      {
        "entity": "Google",
        "content": "Headquartered in Mountain View, California"
      }
    ]
    ```
    """
    logger.debug(f"Adding {len(observations)} observations", context={"observation_count": len(observations)})
    
    # Stream initial acknowledgment
    yield dict_to_json({
        "status": "processing",
        "message": f"Adding {len(observations)} observations...",
        "timestamp": datetime.datetime.now().isoformat()
    })
    
    try:
        # Format observations to match the expected structure
        formatted_observations = []
        for observation in observations:
            # Extract the entity name or ID
            entity_name = observation.get("entity") or observation.get("entityName") or observation.get("entity_id")
            
            if not entity_name:
                logger.warn(f"Observation skipped due to missing entity reference: {observation}")
                continue
                
            # Extract the content
            content = observation.get("content")
            
            if not content:
                logger.warn(f"Observation skipped due to missing content: {observation}")
                continue
                
            # Format the observation
            formatted_observation = {
                "entityName": entity_name,
                "contents": [content] if isinstance(content, str) else content
            }
            
            # Add metadata if available
            if "metadata" in observation and isinstance(observation["metadata"], dict):
                metadata_str = ", ".join([f"{k}: {v}" for k, v in observation["metadata"].items()])
                if "category" in observation["metadata"]:
                    # Add category as a separate content item
                    category = observation["metadata"]["category"]
                    formatted_observation["contents"].append(f"Category: {category}")
            
            logger.debug(f"Formatted observation for entity: {entity_name}")
            formatted_observations.append(formatted_observation)
        
        # Stream progress for large operations
        if len(observations) > 10:
            yield dict_to_json({
                "status": "progress",
                "message": f"Processing {len(formatted_observations)} observations...",
                "complete": 25,
                "timestamp": datetime.datetime.now().isoformat()
            })
            
        # Pass the formatted observations to the graph manager
        result = graph_manager.add_observations(formatted_observations)
        
        # Stream completion status for large operations
        if len(observations) > 10:
            yield dict_to_json({
                "status": "progress",
                "message": f"Added {len(formatted_observations)} observations",
                "complete": 100,
                "timestamp": datetime.datetime.now().isoformat()
            })
            
        # Process result and notify about observation changes
        try:
            result_json = json.loads(result)
            
            # Send observation notifications if successful
            if result_json.get("status") == "success":
                # Access the server's underlying notification API
                ctx = server.request_context
                if ctx and hasattr(server, "_server"):
                    for obs in formatted_observations:
                        entity_name = obs.get("entityName")
                        contents = obs.get("contents", [])
                        for content in contents:
                            await server._server.send_notification(
                                "observation/changed",
                                {
                                    "entity": entity_name,
                                    "content": content,
                                    "action": "added"
                                }
                            )
            
            logger.debug("Observations added", context={"result": result[:100] + "..." if len(result) > 100 else result})
            return result
        except json.JSONDecodeError:
            logger.warn(f"Result not in JSON format: {result}")
            return result
            
    except Exception as e:
        logger.error(f"Error adding observations: {str(e)}", exc_info=True)
        error_response = ErrorResponse.create(
            message=f"Error adding observations: {str(e)}",
            code="observation_creation_error",
            details={"observation_count": len(observations)}
        )
        return dict_to_json(error_response)

@server.tool()
async def search_nodes(query: str, limit: int = 10, project_name: Optional[str] = None) -> str:
    """
    Search for nodes in the knowledge graph.
    
    This performs a semantic search when embeddings are enabled or a basic text search otherwise.
    Results are returned as a streaming response for improved client experience.
    
    Args:
        query: Search query string
        limit: Maximum number of results to return (default: 10)
        project_name: Optional project name to search in (defaults to current project)
    
    Returns:
        JSON string with search results
        
    Example:
    ```
    "Who works at Google?"
    ```
    
    With limit:
    ```
    {"query": "Who works at Google?", "limit": 5}
    ```
    """
    logger.debug(f"Searching nodes with query: '{query}'", context={
        "limit": limit,
        "project_name": project_name or graph_manager.default_project_name
    })
    
    # Stream initial acknowledgment
    yield dict_to_json({
        "status": "processing",
        "message": f"Searching for '{query}'...",
        "timestamp": datetime.datetime.now().isoformat()
    })
    
    try:
        # Ensure the limit is an integer
        try:
            limit = int(limit)
        except (ValueError, TypeError):
            logger.warn(f"Invalid limit value: {limit}, using default of 10")
            limit = 10
        
        # Clean the query
        cleaned_query = query.strip() if query else ""
        if not cleaned_query:
            logger.warn("Empty search query")
            error_response = ErrorResponse.create(
                message="Empty search query",
                code="empty_query_error",
                details={
                    "project": project_name or graph_manager.default_project_name
                }
            )
            error_response["results"] = []
            return dict_to_json(error_response)
        
        # Stream an update for long queries
        yield dict_to_json({
            "status": "progress",
            "message": f"Executing search for '{cleaned_query}'...",
            "complete": 50,
            "timestamp": datetime.datetime.now().isoformat()
        })
        
        # Search nodes
        result = graph_manager.search_nodes(cleaned_query, limit, project_name)
        
        # Parse result and add debugging info
        try:
            result_json = json.loads(result)
            result_count = len(result_json.get("results", []))
            logger.debug(f"Search completed, found {result_count} results", context={"result_count": result_count})
            
            # Add additional info for debugging
            if result_count == 0:
                # Try to get more diagnostic information
                all_entities = graph_manager.get_all_memories(project_name)
                try:
                    all_entities_json = json.loads(all_entities)
                    entity_count = len(all_entities_json.get("memories", []))
                    logger.debug(f"Database has {entity_count} total entities")
                    
                    # Add diagnostic info
                    result_json["diagnostic"] = {
                        "total_entities": entity_count,
                        "embeddings_enabled": graph_manager.embedding_enabled,
                        "timestamp": datetime.datetime.now().isoformat()
                    }
                except json.JSONDecodeError:
                    logger.warn(f"Could not parse get_all_memories result: {all_entities}")
        except json.JSONDecodeError:
            logger.warn(f"Could not parse search result as JSON: {result}")
        
        return result
    except Exception as e:
        logger.error(f"Error searching nodes: {str(e)}", exc_info=True)
        error_response = ErrorResponse.create(
            message=f"Error searching nodes: {str(e)}",
            code="search_error",
            details={
                "query": query,
                "limit": limit,
                "project": project_name or graph_manager.default_project_name
            }
        )
        return dict_to_json(error_response)

@server.tool()
async def delete_entity(entity: str) -> str:
    """
    Delete an entity from the knowledge graph.
    
    This removes the entity and all its observations and relationships.
    The operation will notify clients of entity changes.
    
    Args:
        entity: The name of the entity to delete
    
    Returns:
        JSON response with operation result
    
    Example:
    ```
    "Alice Smith"
    ```
    """
    logger.debug(f"Deleting entity: {entity}")
    
    # Stream initial acknowledgment
    yield dict_to_json({
        "status": "processing",
        "message": f"Deleting entity '{entity}'...",
        "timestamp": datetime.datetime.now().isoformat()
    })
    
    try:
        result = graph_manager.delete_entity(entity)
        
        # Send entity deletion notification
        try:
            result_json = json.loads(result)
            
            # Send deletion notification if successful
            if result_json.get("status") == "success":
                # Access the server's underlying notification API
                ctx = server.request_context
                if ctx and hasattr(server, "_server"):
                    await server._server.send_notification(
                        "entity/changed",
                        {"name": entity, "action": "deleted"}
                    )
            
            logger.debug(f"Entity deletion completed: {entity}")
            return result
        except json.JSONDecodeError:
            logger.warn(f"Result not in JSON format: {result}")
            return result
            
    except Exception as e:
        logger.error(f"Error deleting entity: {str(e)}", exc_info=True)
        error_response = ErrorResponse.create(
            message=f"Error deleting entity: {str(e)}",
            code="entity_deletion_error",
            details={"entity": entity}
        )
        return dict_to_json(error_response)

@server.tool()
async def delete_relation(from_entity: str, to_entity: str, relationType: str) -> str:
    """
    Delete a relation from the knowledge graph.
    
    Removes the specified relationship between two entities.
    The operation will notify clients of relation changes.
    
    Args:
        from_entity: The name of the source entity
        to_entity: The name of the target entity
        relationType: The type of the relation to delete
    
    Returns:
        JSON response with operation result
    
    Example:
    ```
    {
      "from_entity": "Alice Smith",
      "to_entity": "Google",
      "relationType": "WORKS_AT"
    }
    ```
    """
    logger.debug(f"Deleting relation: {from_entity} -{relationType}-> {to_entity}")
    
    # Stream initial acknowledgment
    yield dict_to_json({
        "status": "processing",
        "message": f"Deleting relation {from_entity} -{relationType}-> {to_entity}...",
        "timestamp": datetime.datetime.now().isoformat()
    })
    
    try:
        result = graph_manager.delete_relation(from_entity, to_entity, relationType)
        
        # Send relation deletion notification
        try:
            result_json = json.loads(result)
            
            # Send deletion notification if successful
            if result_json.get("status") == "success":
                # Access the server's underlying notification API
                ctx = server.request_context
                if ctx and hasattr(server, "_server"):
                    await server._server.send_notification(
                        "relation/changed",
                        {
                            "from": from_entity,
                            "to": to_entity,
                            "relationType": relationType,
                            "action": "deleted"
                        }
                    )
            
            logger.debug(f"Relation deletion completed")
            return result
        except json.JSONDecodeError:
            logger.warn(f"Result not in JSON format: {result}")
            return result
            
    except Exception as e:
        logger.error(f"Error deleting relation: {str(e)}", exc_info=True)
        error_response = ErrorResponse.create(
            message=f"Error deleting relation: {str(e)}",
            code="relation_deletion_error",
            details={
                "from_entity": from_entity,
                "to_entity": to_entity,
                "relationType": relationType
            }
        )
        return dict_to_json(error_response)

@server.tool()
async def delete_observation(entity: str, content: str) -> str:
    """
    Delete an observation from the knowledge graph.
    
    Removes a specific fact or property from an entity.
    The operation will notify clients of observation changes.
    
    Args:
        entity: The name of the entity
        content: The exact content of the observation to delete
    
    Returns:
        JSON response with operation result
    
    Example:
    ```
    {
      "entity": "Alice Smith",
      "content": "Lives in San Francisco"
    }
    ```
    """
    logger.debug(f"Deleting observation from entity: {entity}", context={"content_length": len(content)})
    
    # Stream initial acknowledgment
    yield dict_to_json({
        "status": "processing",
        "message": f"Deleting observation from entity '{entity}'...",
        "timestamp": datetime.datetime.now().isoformat()
    })
    
    try:
        result = graph_manager.delete_observation(entity, content)
        
        # Send observation deletion notification
        try:
            result_json = json.loads(result)
            
            # Send deletion notification if successful
            if result_json.get("status") == "success":
                # Access the server's underlying notification API
                ctx = server.request_context
                if ctx and hasattr(server, "_server"):
                    await server._server.send_notification(
                        "observation/changed",
                        {
                            "entity": entity,
                            "content": content,
                            "action": "deleted"
                        }
                    )
            
            logger.debug(f"Observation deletion completed")
            return result
        except json.JSONDecodeError:
            logger.warn(f"Result not in JSON format: {result}")
            return result
            
    except Exception as e:
        logger.error(f"Error deleting observation: {str(e)}", exc_info=True)
        error_response = ErrorResponse.create(
            message=f"Error deleting observation: {str(e)}",
            code="observation_deletion_error",
            details={
                "entity": entity,
                "content_length": len(content)
            }
        )
        return dict_to_json(error_response)

@server.tool()
async def configure_embedding(config: Dict[str, Any]) -> str:
    """
    Configure the embedding provider for the knowledge graph and return the full configuration.
    
    This will:
    1. Update the memory manager configuration
    2. Set the project name for memory operations
    3. Reinitialize connections with the new configuration
    4. Return the complete configuration for the client to save locally
    
    Args:
        config: Configuration object containing provider settings
            - provider: The embedding provider to use
            - model: Optional model name for the provider
            - api_key: Optional API key for authentication
            - dimensions: Optional embedding dimensions
            - project_name: Optional project name to use
            - client_id: Optional client identifier
            - config: Additional provider-specific settings
    
    Returns:
        JSON configuration with initialization status
    
    Example:
    ```
    {
      "provider": "openai",
      "model": "text-embedding-3-small",
      "api_key": "your-api-key",
      "dimensions": 1536,
      "project_name": "my-project"
    }
    ```
    
    Azure OpenAI example:
    ```
    {
      "provider": "azure_openai",
      "api_key": "your-api-key",
      "config": {
        "azure_deployment": "your-deployment-name",
        "azure_endpoint": "https://your-resource.openai.azure.com"
      }
    }
    ```
    """
    # Stream initial acknowledgment
    yield dict_to_json({
        "status": "processing",
        "message": f"Configuring embedding provider...",
        "timestamp": datetime.datetime.now().isoformat()
    })
    
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
            error_response = ErrorResponse.create(
                message="Missing required parameter: provider",
                code="missing_parameter",
                details={"parameter": "provider"}
            )
            return dict_to_json(error_response)
            
        # Stream progress update
        yield dict_to_json({
            "status": "progress",
            "message": f"Building configuration for provider: {provider}",
            "complete": 25,
            "timestamp": datetime.datetime.now().isoformat()
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
        
        # Stream progress update
        yield dict_to_json({
            "status": "progress",
            "message": f"Applying configuration to memory manager",
            "complete": 50,
            "timestamp": datetime.datetime.now().isoformat()
        })
        
        # Apply configuration to the memory manager
        result = graph_manager.apply_client_config(mem0_config)
        
        if result["status"] != "success":
            return dict_to_json(result)
            
        # Stream progress update
        yield dict_to_json({
            "status": "progress",
            "message": f"Reinitializing memory manager",
            "complete": 75,
            "timestamp": datetime.datetime.now().isoformat()
        })
            
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
            instructions += f"You should use this configuration whenever interacting with the memory graph for project '{graph_manager.default_project_name}'."
        
        response = {
            "status": "success",
            "message": f"Successfully configured embedding provider: {provider}",
            "provider": provider,
            "project_name": graph_manager.default_project_name,
            "embedding_enabled": graph_manager.embedding_enabled,
            "config": full_config,
            "instructions_for_agent": instructions,
            "file_name": file_name
        }
        
        # Return success result with configuration
        return dict_to_json(response)
        
    except Exception as e:
        logger.error(f"Error configuring embedding: {str(e)}")
        error_response = ErrorResponse.create(
            message=f"Failed to configure embedding: {str(e)}",
            code="embedding_configuration_error",
            details={"provider": config.get("provider")}
        )
        return dict_to_json(error_response)

@server.tool()
async def get_embedding_config() -> str:
    """
    Get the current embedding configuration.
    
    Returns a JSON string with the current configuration that can be used
    to understand the embedding provider settings or save to a config file.
    
    Returns:
        JSON configuration with current settings
    
    Example response:
    ```
    {
      "status": "success",
      "config": {
        "graph_store": {
          "provider": "neo4j",
          "config": {
            "url": "bolt://localhost:7687",
            "username": "neo4j",
            "password": "password",
            "database": "neo4j"
          }
        },
        "embedder": {
          "provider": "openai",
          "config": {
            "model": "text-embedding-3-small",
            "embedding_dims": 1536
          }
        }
      }
    }
    ```
    """
    try:
        # Get current configuration from memory manager
        config = graph_manager.get_current_config()
        
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

@server.tool()
async def get_all_memories(project_name: Optional[str] = None) -> str:
    """
    Get all memories (entities and their observations) for a user/project.
    
    Returns all entities and their observations, useful for understanding the 
    current state of the knowledge graph or for debugging.
    
    Args:
        project_name: Optional project name (defaults to current project)
    
    Returns:
        JSON string with all memories in the knowledge graph
        
    Example response:
    ```
    {
      "status": "success",
      "memories": [
        {
          "name": "Alice Smith",
          "entityType": "Person",
          "observations": [
            "Works as a software engineer",
            "Lives in San Francisco"
          ]
        },
        {
          "name": "Google",
          "entityType": "Organization",
          "observations": [
            "Technology company founded in 1998",
            "Headquartered in Mountain View, California"
          ]
        }
      ],
      "count": 2
    }
    ```
    """
    # Stream initial acknowledgment for potentially large result sets
    yield dict_to_json({
        "status": "processing",
        "message": f"Retrieving all memories for project: {project_name or graph_manager.default_project_name}",
        "timestamp": datetime.datetime.now().isoformat()
    })
    
    try:
        result = graph_manager.get_all_memories(project_name)
        
        # Try to check result size
        try:
            result_json = json.loads(result)
            memory_count = len(result_json.get("memories", []))
            
            # For large result sets, send a progress update
            if memory_count > 20:
                yield dict_to_json({
                    "status": "progress",
                    "message": f"Retrieved {memory_count} memories",
                    "count": memory_count,
                    "complete": 100,
                    "timestamp": datetime.datetime.now().isoformat()
                })
        except json.JSONDecodeError:
            pass
        
        return result
    except Exception as e:
        logger.error(f"Error getting all memories: {str(e)}")
        error_response = ErrorResponse.create(
            message=f"Failed to get all memories: {str(e)}",
            code="memory_retrieval_error",
            details={"project_name": project_name or graph_manager.default_project_name}
        )
        return dict_to_json(error_response)

@server.tool()
async def set_project_name(project_name: str) -> str:
    """
    Set the project name for memory operations.
    
    This tool establishes the context identifier for the current project in the memory system.
    All subsequent memory operations will be associated with this project name automatically.
    
    Args:
        project_name: A unique identifier for the project
            - Use lowercase with hyphens for multi-word names
            - Should be consistent across sessions for the same project
            - Must be 3-50 characters in length
    
    Returns:
        JSON response with project configuration data
    
    Effects:
        - Creates a dedicated memory namespace for the project if it doesn't exist
        - Enables retrieval of project-specific memories in future interactions
        - Links all new observations and entities to this project context
    
    When to use:
        - At the beginning of each new conversation about a project
        - When switching context between different projects
        - Before storing any project-specific knowledge
    
    Example:
    ```python
    # Set the current project context
    set_project_name("inventory-management-system")
    
    # Now all memory operations will use this context by default
    create_entities([{"name": "ProductService", "entityType": "Component"}])  # Will be associated with "inventory-management-system"
    ```
    """
    try:
        if not project_name or not project_name.strip():
            error_response = ErrorResponse.create(
                message="Project name cannot be empty",
                code="invalid_project_name"
            )
            return dict_to_json(error_response)
            
        graph_manager.set_project_name(project_name)
        
        # Create instructions for the AI agent
        file_name = f"mcp_project_config_{project_name}.json"
        project_config = {
            "project_name": project_name,
            "timestamp": datetime.datetime.now().isoformat()
        }
        
        # Send notification about project context change
        ctx = server.request_context
        if ctx and hasattr(server, "_server"):
            await server._server.send_notification(
                "project/changed",
                {"name": project_name, "action": "selected"}
            )
        
        return dict_to_json({
            "status": "success",
            "message": f"Project name set to: {project_name}",
            "project_name": project_name,
            "config": project_config,
            "actions": [
                {
                    "type": "write_file",
                    "path": file_name,
                    "content": project_config,
                    "description": "Save project configuration file to maintain context between sessions"
                },
                {
                    "type": "remember",
                    "data": {
                        "context": "project_name",
                        "value": project_name
                    },
                    "description": "Use this project name for all subsequent memory operations"
                }
            ]
        })
    except Exception as e:
        logger.error(f"Error setting project name: {str(e)}")
        error_response = ErrorResponse.create(
            message=f"Failed to set project name: {str(e)}",
            code="project_name_error"
        )
        return dict_to_json(error_response)

@server.tool()
async def delete_all_memories(project_name: Optional[str] = None) -> str:
    """
    Delete all memories for a user/project.
    
    This is a destructive operation that removes all entities, relationships,
    and observations from the knowledge graph. Use with caution.
    
    Args:
        project_name: Optional project name (defaults to current project)
    
    Returns:
        JSON response with deletion result
        
    Example response:
    ```
    {
      "status": "success",
      "message": "All memories deleted"
    }
    ```
    """
    # Stream initial acknowledgment
    yield dict_to_json({
        "status": "processing",
        "message": f"Preparing to delete all memories for project: {project_name or graph_manager.default_project_name}",
        "timestamp": datetime.datetime.now().isoformat()
    })
    
    try:
        # Get current memory count for reporting
        current_memories = graph_manager.get_all_memories(project_name)
        memory_count = 0
        try:
            current_json = json.loads(current_memories)
            memory_count = len(current_json.get("memories", []))
            
            # For large deletions, send an additional warning
            if memory_count > 10:
                yield dict_to_json({
                    "status": "warning",
                    "message": f"Deleting {memory_count} memories. This operation cannot be undone.",
                    "count": memory_count,
                    "timestamp": datetime.datetime.now().isoformat()
                })
        except json.JSONDecodeError:
            pass
        
        # Perform the deletion
        result = graph_manager.delete_all_memories(project_name)
        
        # Send notification about project memory reset
        ctx = server.request_context
        if ctx and hasattr(server, "_server"):
            await server._server.send_notification(
                "project/reset",
                {"name": project_name or graph_manager.default_project_name, "count": memory_count}
            )
        
        return result
    except Exception as e:
        logger.error(f"Error deleting all memories: {str(e)}")
        error_response = ErrorResponse.create(
            message=f"Failed to delete all memories: {str(e)}",
            code="memory_deletion_error",
            details={"project_name": project_name or graph_manager.default_project_name}
        )
        return dict_to_json(error_response)

@server.tool()
async def debug_dump_neo4j(limit: int = 100) -> str:
    """
    Dump Neo4j nodes and relationships for debugging.
    
    This tool is primarily intended for development and debugging.
    It returns a raw dump of nodes and relationships in the Neo4j database.
    
    Args:
        limit: Maximum number of nodes and relationships to return
    
    Returns:
        JSON dump of Neo4j database content
        
    Example:
    ```
    {"limit": 50}
    ```
    """
    logger.debug(f"Dumping Neo4j nodes and relationships (limit: {limit})")
    
    # Stream initial acknowledgment for potentially large result sets
    yield dict_to_json({
        "status": "processing",
        "message": f"Dumping Neo4j database (limit: {limit})",
        "timestamp": datetime.datetime.now().isoformat()
    })
    
    try:
        # Ensure graph manager is initialized
        graph_manager._ensure_initialized()
        
        if not graph_manager.neo4j_driver:
            error_response = ErrorResponse.create(
                message="Neo4j driver not initialized",
                code="driver_not_initialized"
            )
            return dict_to_json(error_response)
        
        # Stream progress update
        yield dict_to_json({
            "status": "progress",
            "message": "Querying Neo4j database...",
            "complete": 50,
            "timestamp": datetime.datetime.now().isoformat()
        })
        
        # Modify the queries to use the specified limit
        result = dump_neo4j_nodes(graph_manager.neo4j_driver, graph_manager.neo4j_database)
        
        # Add timestamp for debugging
        result["timestamp"] = datetime.datetime.now().isoformat()
        result["default_project_name"] = graph_manager.default_project_name
        result["embedding_enabled"] = graph_manager.embedding_enabled
        result["embedder_provider"] = graph_manager.embedder_provider
        
        logger.debug(f"Neo4j dump completed, found {len(result.get('nodes', []))} nodes and {len(result.get('relationships', []))} relationships")
        
        return dict_to_json(result)
    except Exception as e:
        logger.error(f"Error dumping Neo4j data: {str(e)}", exc_info=True)
        error_response = ErrorResponse.create(
            message=f"Error dumping Neo4j data: {str(e)}",
            code="neo4j_dump_error"
        )
        return dict_to_json(error_response)

async def run_server():
    """Run the MCP server with the configured transport."""
    try:
        # Determine transport type from environment variable
        use_sse = os.environ.get("USE_SSE", "false").lower() == "true"
        port = int(os.environ.get("PORT", "8080"))

        if use_sse:
            # Using SSE transport
            try:
                import uvicorn
                from starlette.applications import Starlette
                from starlette.routing import Mount
                
                # Create ASGI app from server with proper initialization options
                app = Starlette(
                    routes=[
                        Mount('/', app=server.sse_app()),
                    ],
                    debug=True
                )
                
                logger.info(f"Neo4j Graph Memory MCP Server running with SSE on http://0.0.0.0:{port}")
                
                # Run with uvicorn
                uvicorn.run(app, host="0.0.0.0", port=port)
            except ImportError:
                logger.error("Uvicorn not installed. Required for SSE mode.")
                sys.exit(1)
        else:
            # Using stdio transport
            logger.info("Neo4j Graph Memory MCP Server running on stdio")
            
            # Run with default settings
            await server.run()
    except Exception as e:
        logger.error(f"Failed to start server: {str(e)}")
        sys.exit(1)

def main():
    """Main entry point with enhanced error handling."""
    try:
        # Run the async server
        if sys.platform == 'win32':
            asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
        
        asyncio.run(run_server())
    except KeyboardInterrupt:
        logger.info("Server stopped by user")
    except Exception as e:
        logger.error(f"Unhandled exception: {str(e)}", exc_info=True)
        sys.exit(1)

if __name__ == "__main__":
    main() 