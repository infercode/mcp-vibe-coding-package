#!/usr/bin/env python3
import json
import datetime
from typing import Dict, List, Any, Optional

from src.logger import get_logger
from src.utils import dict_to_json

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

def register_core_tools(server, graph_manager):
    """Register core memory tools with the server."""
    
    @server.tool()
    async def create_entities(entities: List[Dict[str, Any]]) -> str:
        """
        Create multiple new entities in the knowledge graph.
        
        Each entity should have a name, entity type, and optional observations.
        
        Args:
            entities: List of entity objects to create
        
        Returns:
            JSON response with operation result
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
            logger.debug("Entities creation completed", context={"result": result[:100] + "..." if len(result) > 100 else result})
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
        
        Args:
            relations: List of relation objects
        
        Returns:
            JSON response with operation result
        """
        logger.debug(f"Creating {len(relations)} relations", context={"relation_count": len(relations)})
        
        try:
            result = graph_manager.create_relations(relations)
            logger.debug("Relations creation completed", context={"result": result[:100] + "..." if len(result) > 100 else result})
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
        
        Args:
            observations: List of observation objects
        
        Returns:
            JSON response with operation result
        """
        logger.debug(f"Adding {len(observations)} observations", context={"observation_count": len(observations)})
        
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
            
            # Pass the formatted observations to the graph manager
            result = graph_manager.add_observations(formatted_observations)
            logger.debug("Observations added", context={"result": result[:100] + "..." if len(result) > 100 else result})
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
        
        Args:
            query: Search query string
            limit: Maximum number of results to return (default: 10)
            project_name: Optional project name to search in (defaults to current project)
        
        Returns:
            JSON string with search results
        """
        logger.debug(f"Searching nodes with query: '{query}'", context={
            "limit": limit,
            "project_name": project_name or graph_manager.default_project_name
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
        
        Args:
            entity: The name of the entity to delete
        
        Returns:
            JSON response with operation result
        """
        logger.debug(f"Deleting entity: {entity}")
        
        try:
            result = graph_manager.delete_entity(entity)
            logger.debug(f"Entity deletion completed: {entity}")
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
        
        Args:
            from_entity: The name of the source entity
            to_entity: The name of the target entity
            relationType: The type of the relation to delete
        
        Returns:
            JSON response with operation result
        """
        logger.debug(f"Deleting relation: {from_entity} -{relationType}-> {to_entity}")
        
        try:
            result = graph_manager.delete_relation(from_entity, to_entity, relationType)
            logger.debug(f"Relation deletion completed")
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
        
        Args:
            entity: The name of the entity
            content: The exact content of the observation to delete
        
        Returns:
            JSON response with operation result
        """
        logger.debug(f"Deleting observation from entity: {entity}", context={"content_length": len(content)})
        
        try:
            result = graph_manager.delete_observation(entity, content)
            logger.debug(f"Observation deletion completed")
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
    async def get_all_memories(random_string: str = "") -> str:
        """
        Get all memories for a user/project.
        
        Args:
            random_string: Dummy parameter (required by MCP interface but not used)
        
        Returns:
            JSON string with all memories in the knowledge graph
        """
        try:
            # Project name is taken from graph_manager.default_project_name internally
            result = graph_manager.get_all_memories(None)
            return result
        except Exception as e:
            logger.error(f"Error getting all memories: {str(e)}")
            error_response = ErrorResponse.create(
                message=f"Failed to get all memories: {str(e)}",
                code="memory_retrieval_error",
                details={"project_name": graph_manager.default_project_name}
            )
            return dict_to_json(error_response)

    @server.tool()
    async def delete_all_memories(random_string: str = "") -> str:
        """
        Delete all memories for a user/project.
        
        Args:
            random_string: Dummy parameter (required by MCP interface but not used)
        
        Returns:
            JSON response with deletion result
        """
        try:
            # Use default project name from graph manager
            result = graph_manager.delete_all_memories(None)
            return result
        except Exception as e:
            logger.error(f"Error deleting all memories: {str(e)}")
            error_response = ErrorResponse.create(
                message=f"Failed to delete all memories: {str(e)}",
                code="memory_deletion_error",
                details={"project_name": graph_manager.default_project_name}
            )
            return dict_to_json(error_response)

    @server.tool()
    async def debug_dump_neo4j(limit: int = 100) -> str:
        """
        Dump Neo4j nodes and relationships for debugging.
        
        Args:
            limit: Maximum number of nodes and relationships to return
        
        Returns:
            JSON dump of Neo4j database content
        """
        logger.debug(f"Dumping Neo4j nodes and relationships (limit: {limit})")
        
        try:
            # Ensure graph manager is initialized
            graph_manager._ensure_initialized()
            
            if not graph_manager.neo4j_driver:
                error_response = ErrorResponse.create(
                    message="Neo4j driver not initialized",
                    code="driver_not_initialized"
                )
                return dict_to_json(error_response)
            
            # Get the Neo4j dump
            from src.utils import dump_neo4j_nodes
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