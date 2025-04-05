#!/usr/bin/env python3
"""
Core Memory Tools with Pydantic Integration

This module implements MCP tools for the core memory system using
Pydantic models for validation and serialization.
"""

import json
import datetime
from typing import Dict, List, Any, Optional, Union, cast

from src.logger import get_logger
from src.models.core_memory.models import (
    EntityCreate, EntitiesCreate, RelationshipCreate, RelationshipsCreate,
    EntityObservation, ObservationsCreate, SearchQuery, EntityDelete,
    RelationshipDelete, ObservationDelete, create_error_response,
    create_success_response, model_to_json, model_to_dict
)

# Initialize logger
logger = get_logger()

def register_core_tools(server, get_client_manager):
    """Register core memory tools with the server."""
    
    @server.tool()
    async def create_entities(entities: List[Dict[str, Any]], client_id: Optional[str] = None) -> str:
        """
        Create multiple new entities in the knowledge graph.
        
        Each entity should have a name, entity type, and optional observations.
        
        Args:
            entities: List of entity objects to create
            client_id: Optional client ID for identifying the connection
        
        Returns:
            JSON response with operation result
        """
        logger.debug(f"Creating {len(entities)} entities", context={"entity_count": len(entities)})
        
        try:
            # Convert input to Pydantic models for validation
            try:
                # Format entities to match the expected structure
                formatted_entities = []
                for entity_data in entities:
                    # Extract the required fields or use defaults
                    entity = {
                        "name": entity_data.get("entity_id") or entity_data.get("name"),
                        "entity_type": entity_data.get("entity_type") or entity_data.get("entityType") or entity_data.get("type", "Unknown"),
                        "observations": []
                    }
                    
                    # Handle observations: could be in entity.observations or as separate metadata
                    if "observations" in entity_data:
                        entity["observations"] = entity_data["observations"]
                    elif "content" in entity_data:
                        entity["observations"] = [entity_data["content"]]
                        
                    # Add any metadata
                    if "metadata" in entity_data and isinstance(entity_data["metadata"], dict):
                        entity["metadata"] = entity_data["metadata"]
                    
                    # Add client_id to metadata if provided
                    if client_id:
                        if "metadata" not in entity:
                            entity["metadata"] = {}
                        entity["metadata"]["client_id"] = client_id
                    
                    formatted_entities.append(entity)
                
                # Validate using Pydantic model
                entities_model = EntitiesCreate(entities=[EntityCreate(**entity) for entity in formatted_entities])
                
            except ValueError as e:
                logger.error(f"Validation error for entities: {str(e)}")
                error_response = create_error_response(
                    message=f"Invalid entity data: {str(e)}",
                    code="validation_error"
                )
                return model_to_json(error_response)
            
            # Get the client-specific graph manager
            client_graph_manager = get_client_manager(client_id)
            
            # Call the graph manager with the validated entities
            result = client_graph_manager.create_entities([model_to_dict(entity) for entity in entities_model.entities])
            
            # Parse the result
            try:
                if isinstance(result, str):
                    parsed_result = json.loads(result)
                else:
                    parsed_result = result
                
                if isinstance(parsed_result, dict) and parsed_result.get("status") == "error":
                    error_response = create_error_response(
                        message=parsed_result.get("message", "Unknown error"),
                        code="entity_creation_error"
                    )
                    return model_to_json(error_response)
                
                # Return the success response
                success_response = create_success_response(
                    message=f"Successfully created {len(entities)} entities",
                    data=parsed_result
                )
                return model_to_json(success_response)
                
            except json.JSONDecodeError:
                # If result is not valid JSON, return it as-is (legacy compatibility)
                return result
            
        except Exception as e:
            logger.error(f"Error creating entities: {str(e)}", exc_info=True)
            error_response = create_error_response(
                message=f"Error creating entities: {str(e)}",
                code="entity_creation_error",
                details={"entity_count": len(entities)}
            )
            return model_to_json(error_response)

    @server.tool()
    async def create_relations(relations: List[Dict[str, Any]], client_id: Optional[str] = None) -> str:
        """
        Create multiple new relations in the knowledge graph.
        
        Each relation connects a source entity to a target entity with a specific relationship type.
        
        Args:
            relations: List of relation objects
            client_id: Optional client ID for identifying the connection
        
        Returns:
            JSON response with operation result
        """
        logger.debug(f"Creating {len(relations)} relations", context={"relation_count": len(relations)})
        
        try:
            # Convert input to Pydantic models for validation
            try:
                formatted_relations = []
                for relation_data in relations:
                    # Normalize field names
                    relation = {
                        "from_entity": relation_data.get("from_entity") or relation_data.get("fromEntity") or relation_data.get("source"),
                        "to_entity": relation_data.get("to_entity") or relation_data.get("toEntity") or relation_data.get("target"),
                        "relationship_type": relation_data.get("relationship_type") or relation_data.get("relationType") or relation_data.get("type"),
                    }
                    
                    # Add optional fields if present
                    if "weight" in relation_data:
                        relation["weight"] = relation_data["weight"]
                        
                    # Add any metadata
                    if "metadata" in relation_data and isinstance(relation_data["metadata"], dict):
                        relation["metadata"] = relation_data["metadata"]
                    
                    # Add client_id to metadata if provided
                    if client_id:
                        if "metadata" not in relation or relation["metadata"] is None:
                            relation["metadata"] = {}
                        relation["metadata"]["client_id"] = client_id
                    
                    formatted_relations.append(relation)
                
                # Validate using Pydantic model
                relations_model = RelationshipsCreate(relationships=[RelationshipCreate(**relation) for relation in formatted_relations])
                
            except ValueError as e:
                logger.error(f"Validation error for relations: {str(e)}")
                error_response = create_error_response(
                    message=f"Invalid relation data: {str(e)}",
                    code="validation_error"
                )
                return model_to_json(error_response)
            
            # Get the client-specific graph manager
            client_graph_manager = get_client_manager(client_id)
            
            # Call the graph manager with the validated relations
            result = client_graph_manager.create_relations([model_to_dict(relation) for relation in relations_model.relationships])
            
            # Parse the result
            try:
                if isinstance(result, str):
                    parsed_result = json.loads(result)
                else:
                    parsed_result = result
                
                if isinstance(parsed_result, dict) and parsed_result.get("status") == "error":
                    error_response = create_error_response(
                        message=parsed_result.get("message", "Unknown error"),
                        code="relation_creation_error"
                    )
                    return model_to_json(error_response)
                
                # Return the success response
                success_response = create_success_response(
                    message=f"Successfully created {len(relations)} relations",
                    data=parsed_result
                )
                return model_to_json(success_response)
                
            except json.JSONDecodeError:
                # If result is not valid JSON, return it as-is (legacy compatibility)
                return result
            
        except Exception as e:
            logger.error(f"Error creating relations: {str(e)}", exc_info=True)
            error_response = create_error_response(
                message=f"Error creating relations: {str(e)}",
                code="relation_creation_error",
                details={"relation_count": len(relations)}
            )
            return model_to_json(error_response)

    @server.tool()
    async def add_observations(observations: List[Dict[str, Any]], client_id: Optional[str] = None) -> str:
        """
        Add multiple observations to entities in the knowledge graph.
        
        Observations are facts or properties about an entity.
        
        Args:
            observations: List of observation objects
            client_id: Optional client ID for identifying the connection
        
        Returns:
            JSON response with operation result
        """
        logger.debug(f"Adding {len(observations)} observations", context={"observation_count": len(observations)})
        
        try:
            # Convert input to Pydantic models for validation
            try:
                formatted_observations = []
                for obs_data in observations:
                    # Extract the entity name or ID
                    entity_name = obs_data.get("entity") or obs_data.get("entityName") or obs_data.get("entity_name") or obs_data.get("entity_id")
                    
                    if not entity_name:
                        logger.warn(f"Observation skipped due to missing entity reference: {obs_data}")
                        continue
                        
                    # Extract the content
                    content = obs_data.get("content") or obs_data.get("contents")
                    
                    if not content:
                        logger.warn(f"Observation skipped due to missing content: {obs_data}")
                        continue
                    
                    # Prepare the contents field as a list
                    contents_list = []
                    if isinstance(content, str):
                        contents_list = [content]
                    elif isinstance(content, list):
                        contents_list = content
                    else:
                        # Unexpected type, try to convert to string
                        try:
                            contents_list = [str(content)]
                        except:
                            logger.warn(f"Observation skipped due to invalid content type: {type(content)}")
                            continue
                        
                    # Format the observation
                    observation = {
                        "entity_name": entity_name,
                        "contents": contents_list
                    }
                    
                    # Add metadata if available
                    if "metadata" in obs_data and isinstance(obs_data["metadata"], dict):
                        observation["metadata"] = obs_data["metadata"]
                    
                    # Add client_id to metadata if provided
                    if client_id:
                        if "metadata" not in observation:
                            observation["metadata"] = {}
                        observation["metadata"]["client_id"] = client_id
                    
                    formatted_observations.append(observation)
                
                # Validate using Pydantic model
                observations_model = ObservationsCreate(observations=[EntityObservation(**obs) for obs in formatted_observations])
                
            except ValueError as e:
                logger.error(f"Validation error for observations: {str(e)}")
                error_response = create_error_response(
                    message=f"Invalid observation data: {str(e)}",
                    code="validation_error"
                )
                return model_to_json(error_response)
            
            # Get the client-specific graph manager
            client_graph_manager = get_client_manager(client_id)
            
            # Call the graph manager with the validated observations
            result = client_graph_manager.add_observations([model_to_dict(obs) for obs in observations_model.observations])
            
            # Parse the result
            try:
                if isinstance(result, str):
                    parsed_result = json.loads(result)
                else:
                    parsed_result = result
                
                if isinstance(parsed_result, dict) and parsed_result.get("status") == "error":
                    error_response = create_error_response(
                        message=parsed_result.get("message", "Unknown error"),
                        code="observation_creation_error"
                    )
                    return model_to_json(error_response)
                
                # Return the success response
                success_response = create_success_response(
                    message=f"Successfully added {len(observations)} observations",
                    data=parsed_result
                )
                return model_to_json(success_response)
                
            except json.JSONDecodeError:
                # If result is not valid JSON, return it as-is (legacy compatibility)
                return result
            
        except Exception as e:
            logger.error(f"Error adding observations: {str(e)}", exc_info=True)
            error_response = create_error_response(
                message=f"Error adding observations: {str(e)}",
                code="observation_creation_error",
                details={"observation_count": len(observations)}
            )
            return model_to_json(error_response)

    @server.tool()
    async def search_nodes(query: str, limit: int = 10, project_name: Optional[str] = None, client_id: Optional[str] = None) -> str:
        """
        Search for nodes in the knowledge graph.
        
        This performs a semantic search when embeddings are enabled or a basic text search otherwise.
        
        Args:
            query: Search query string
            limit: Maximum number of results to return (default: 10)
            project_name: Optional project name to search in (defaults to current project)
            client_id: Optional client ID for identifying the connection
        
        Returns:
            JSON string with search results
        """
        try:
            # Validate input using Pydantic model
            try:
                search_model = SearchQuery(
                    query=query,
                    limit=limit,
                    project_name=project_name
                )
            except ValueError as e:
                logger.error(f"Validation error for search query: {str(e)}")
                error_response = create_error_response(
                    message=f"Invalid search query: {str(e)}",
                    code="validation_error"
                )
                return model_to_json(error_response)
            
            # Get the client-specific graph manager
            client_graph_manager = get_client_manager(client_id)
            
            # Set project name if provided
            if search_model.project_name:
                client_graph_manager.set_project_name(search_model.project_name)
            
            # Log search parameters
            logger.debug(f"Searching nodes with query: '{search_model.query}'", context={
                "limit": search_model.limit,
                "project_name": search_model.project_name or client_graph_manager.default_project_name
            })
            
            # Perform the search
            result = client_graph_manager.search_nodes(
                search_model.query,
                search_model.limit
            )
            
            # Parse the result
            try:
                if isinstance(result, str):
                    parsed_result = json.loads(result)
                else:
                    parsed_result = result
                
                if isinstance(parsed_result, dict) and parsed_result.get("status") == "error":
                    error_response = create_error_response(
                        message=parsed_result.get("message", "Unknown error"),
                        code="search_error"
                    )
                    return model_to_json(error_response)
                
                # Extract search results
                if isinstance(parsed_result, dict) and "nodes" in parsed_result:
                    nodes = parsed_result["nodes"]
                    
                    # Return the success response
                    success_response = create_success_response(
                        message=f"Found {len(nodes)} results for query '{search_model.query}'",
                        data={"nodes": nodes}
                    )
                    return model_to_json(success_response)
                
                # If result has a different format, return it as-is
                return result
                
            except json.JSONDecodeError:
                # If result is not valid JSON, return it as-is (legacy compatibility)
                return result
            
        except Exception as e:
            logger.error(f"Error searching nodes: {str(e)}", exc_info=True)
            error_response = create_error_response(
                message=f"Error searching nodes: {str(e)}",
                code="search_error"
            )
            return model_to_json(error_response)

    @server.tool()
    async def delete_entity(entity: str, client_id: Optional[str] = None) -> str:
        """
        Delete an entity from the knowledge graph.
        
        Args:
            entity: Name or ID of the entity to delete
            client_id: Optional client ID for identifying the connection
        
        Returns:
            JSON response with operation result
        """
        try:
            # Validate input using Pydantic model
            try:
                entity_model = EntityDelete(entity_name=entity)
            except ValueError as e:
                logger.error(f"Validation error for entity deletion: {str(e)}")
                error_response = create_error_response(
                    message=f"Invalid entity name: {str(e)}",
                    code="validation_error"
                )
                return model_to_json(error_response)
            
            # Get the client-specific graph manager
            client_graph_manager = get_client_manager(client_id)
            
            # Delete the entity
            result = client_graph_manager.delete_entity(entity_model.entity_name)
            
            # Parse the result
            try:
                if isinstance(result, str):
                    parsed_result = json.loads(result)
                else:
                    parsed_result = result
                
                if isinstance(parsed_result, dict) and parsed_result.get("status") == "error":
                    error_response = create_error_response(
                        message=parsed_result.get("message", "Unknown error"),
                        code="entity_deletion_error"
                    )
                    return model_to_json(error_response)
                
                # Return the success response
                success_response = create_success_response(
                    message=f"Successfully deleted entity '{entity_model.entity_name}'",
                    data=parsed_result
                )
                return model_to_json(success_response)
                
            except json.JSONDecodeError:
                # If result is not valid JSON, return it as-is (legacy compatibility)
                return result
            
        except Exception as e:
            logger.error(f"Error deleting entity: {str(e)}", exc_info=True)
            error_response = create_error_response(
                message=f"Error deleting entity: {str(e)}",
                code="entity_deletion_error"
            )
            return model_to_json(error_response)

    @server.tool()
    async def delete_relation(from_entity: str, to_entity: str, relationship_type: str, client_id: Optional[str] = None) -> str:
        """
        Delete a relationship from the knowledge graph.
        
        Args:
            from_entity: Name or ID of the source entity
            to_entity: Name or ID of the target entity
            relationship_type: Type of the relationship to delete
            client_id: Optional client ID for identifying the connection
        
        Returns:
            JSON response with operation result
        """
        try:
            # Validate input using Pydantic model
            try:
                relationship_model = RelationshipDelete(
                    from_entity=from_entity,
                    to_entity=to_entity,
                    relationship_type=relationship_type
                )
            except ValueError as e:
                logger.error(f"Validation error for relationship deletion: {str(e)}")
                error_response = create_error_response(
                    message=f"Invalid relationship data: {str(e)}",
                    code="validation_error"
                )
                return model_to_json(error_response)
            
            # Get the client-specific graph manager
            client_graph_manager = get_client_manager(client_id)
            
            # Delete the relationship
            result = client_graph_manager.delete_relation(
                relationship_model.from_entity,
                relationship_model.to_entity,
                relationship_model.relationship_type
            )
            
            # Parse the result
            try:
                if isinstance(result, str):
                    parsed_result = json.loads(result)
                else:
                    parsed_result = result
                
                if isinstance(parsed_result, dict) and parsed_result.get("status") == "error":
                    error_response = create_error_response(
                        message=parsed_result.get("message", "Unknown error"),
                        code="relationship_deletion_error"
                    )
                    return model_to_json(error_response)
                
                # Return the success response
                success_response = create_success_response(
                    message=f"Successfully deleted relationship '{relationship_model.relationship_type}' from '{relationship_model.from_entity}' to '{relationship_model.to_entity}'",
                    data=parsed_result
                )
                return model_to_json(success_response)
                
            except json.JSONDecodeError:
                # If result is not valid JSON, return it as-is (legacy compatibility)
                return result
            
        except Exception as e:
            logger.error(f"Error deleting relationship: {str(e)}", exc_info=True)
            error_response = create_error_response(
                message=f"Error deleting relationship: {str(e)}",
                code="relationship_deletion_error"
            )
            return model_to_json(error_response)

    @server.tool()
    async def delete_observation(entity: str, content: str, client_id: Optional[str] = None) -> str:
        """
        Delete an observation from an entity in the knowledge graph.
        
        Args:
            entity: Name or ID of the entity
            content: Content of the observation to delete
            client_id: Optional client ID for identifying the connection
        
        Returns:
            JSON response with operation result
        """
        try:
            # Validate input using Pydantic model
            try:
                observation_model = ObservationDelete(
                    entity_name=entity,
                    content=content
                )
            except ValueError as e:
                logger.error(f"Validation error for observation deletion: {str(e)}")
                error_response = create_error_response(
                    message=f"Invalid observation data: {str(e)}",
                    code="validation_error"
                )
                return model_to_json(error_response)
            
            # Get the client-specific graph manager
            client_graph_manager = get_client_manager(client_id)
            
            # Delete the observation
            result = client_graph_manager.delete_observation(
                observation_model.entity_name,
                observation_model.content
            )
            
            # Parse the result
            try:
                if isinstance(result, str):
                    parsed_result = json.loads(result)
                else:
                    parsed_result = result
                
                if isinstance(parsed_result, dict) and parsed_result.get("status") == "error":
                    error_response = create_error_response(
                        message=parsed_result.get("message", "Unknown error"),
                        code="observation_deletion_error"
                    )
                    return model_to_json(error_response)
                
                # Return the success response
                success_response = create_success_response(
                    message=f"Successfully deleted observation from entity '{observation_model.entity_name}'",
                    data=parsed_result
                )
                return model_to_json(success_response)
                
            except json.JSONDecodeError:
                # If result is not valid JSON, return it as-is (legacy compatibility)
                return result
            
        except Exception as e:
            logger.error(f"Error deleting observation: {str(e)}", exc_info=True)
            error_response = create_error_response(
                message=f"Error deleting observation: {str(e)}",
                code="observation_deletion_error"
            )
            return model_to_json(error_response)

    @server.tool()
    async def get_all_memories(random_string: str = "", client_id: Optional[str] = None) -> str:
        """
        Get all memories from the knowledge graph.
        
        WARNING: This is a potentially expensive operation that should be used with caution.
        
        Args:
            random_string: Random string to confirm intentional use of this tool
            client_id: Optional client ID for identifying the connection
        
        Returns:
            JSON response with all memories
        """
        try:
            # Get the client-specific graph manager
            client_graph_manager = get_client_manager(client_id)
            
            # Get all memories
            result = client_graph_manager.get_all_memories(random_string)
            
            # Return the result as-is (legacy compatibility)
            return result
            
        except Exception as e:
            logger.error(f"Error getting all memories: {str(e)}", exc_info=True)
            error_response = create_error_response(
                message=f"Error getting all memories: {str(e)}",
                code="memory_retrieval_error"
            )
            return model_to_json(error_response)

    @server.tool()
    async def delete_all_memories(random_string: str = "", client_id: Optional[str] = None) -> str:
        """
        Delete all memories from the knowledge graph.
        
        WARNING: This is a destructive operation that should be used with extreme caution.
        
        Args:
            random_string: Random string to confirm intentional use of this tool
            client_id: Optional client ID for identifying the connection
        
        Returns:
            JSON response with operation result
        """
        try:
            # Get the client-specific graph manager
            client_graph_manager = get_client_manager(client_id)
            
            # Delete all memories
            result = client_graph_manager.delete_all_memories(random_string)
            
            # Return the result as-is (legacy compatibility)
            return result
            
        except Exception as e:
            logger.error(f"Error deleting all memories: {str(e)}", exc_info=True)
            error_response = create_error_response(
                message=f"Error deleting all memories: {str(e)}",
                code="memory_deletion_error"
            )
            return model_to_json(error_response)

    @server.tool()
    async def debug_dump_neo4j(limit: int = 100, client_id: Optional[str] = None) -> str:
        """
        Dump Neo4j database contents for debugging purposes.
        
        Args:
            limit: Maximum number of nodes to return (default: 100)
            client_id: Optional client ID for identifying the connection
        
        Returns:
            JSON response with Neo4j database contents
        """
        try:
            # Get the client-specific graph manager
            client_graph_manager = get_client_manager(client_id)
            
            # Dump Neo4j database contents
            result = client_graph_manager.debug_dump_neo4j(limit)
            
            # Return the result as-is (legacy compatibility)
            return result
            
        except Exception as e:
            logger.error(f"Error dumping Neo4j database: {str(e)}", exc_info=True)
            error_response = create_error_response(
                message=f"Error dumping Neo4j database: {str(e)}",
                code="debug_dump_error"
            )
            return model_to_json(error_response)
    
    # Return the registered tools
    return {
        "create_entities": create_entities,
        "create_relations": create_relations,
        "add_observations": add_observations,
        "search_nodes": search_nodes,
        "delete_entity": delete_entity,
        "delete_relation": delete_relation,
        "delete_observation": delete_observation,
        "get_all_memories": get_all_memories,
        "delete_all_memories": delete_all_memories,
        "debug_dump_neo4j": debug_dump_neo4j
    } 