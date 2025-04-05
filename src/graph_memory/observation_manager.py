from typing import Any, Dict, List, Optional, Union

from src.utils import dict_to_json, generate_id
from src.utils.neo4j_query_utils import sanitize_query_parameters
from src.graph_memory.base_manager import BaseManager

class ObservationManager:
    """Manager for observation CRUD operations in the knowledge graph."""
    
    def __init__(self, base_manager: BaseManager):
        """
        Initialize the observation manager.
        
        Args:
            base_manager: The base manager instance for database operations
        """
        self.base_manager = base_manager
        self.logger = base_manager.logger
    
    def add_observations(self, observations: List[Union[Dict, Any]]) -> str:
        """
        Add multiple observations to entities in the knowledge graph.
        
        Args:
            observations: List of observations to add
            
        Returns:
            JSON string with the added observations
        """
        try:
            self.base_manager.ensure_initialized()
            
            added_observations = []
            errors = []
            
            for observation in observations:
                # Process observation
                observation_dict = self._convert_to_dict(observation)
                
                # Extract observation information
                entity_name = observation_dict.get("entity", observation_dict.get("entityName", ""))
                content = observation_dict.get("content", "")
                observation_type = observation_dict.get("type", None)
                
                if entity_name and content:
                    # Add the observation in Neo4j
                    try:
                        observation_id = self._add_observation_to_entity(entity_name, content, observation_type)
                        
                        # Format for response
                        added_observation = {
                            "entity": entity_name,
                            "content": content
                        }
                        
                        if observation_type:
                            added_observation["type"] = observation_type
                            
                        if observation_id:
                            added_observation["id"] = observation_id
                        
                        added_observations.append(added_observation)
                        self.logger.info(f"Added observation to entity: {entity_name}")
                    except Exception as e:
                        error = {
                            "entity": entity_name,
                            "content": content,
                            "error": str(e)
                        }
                        errors.append(error)
                        self.logger.error(f"Failed to add observation to entity: {entity_name}: {str(e)}")
            
            result = {"added": added_observations}
            if errors:
                result["errors"] = errors
                
            return dict_to_json(result)
                
        except Exception as e:
            error_msg = f"Error adding observations: {str(e)}"
            self.logger.error(error_msg)
            return dict_to_json({"error": error_msg})
    
    def get_entity_observations(self, entity_name: str, observation_type: Optional[str] = None) -> str:
        """
        Get observations for an entity from the knowledge graph.
        
        Args:
            entity_name: The name of the entity to get observations for
            observation_type: Optional type of observations to filter by
            
        Returns:
            JSON string with the observations
        """
        try:
            self.base_manager.ensure_initialized()
            
            # Check if entity exists
            entity_query = """
            MATCH (e:Entity {name: $name})
            RETURN e
            """
            
            try:
                # Sanitize parameters
                sanitized_params = sanitize_query_parameters({"name": entity_name})
                
                # Execute read query with validation
                entity_records = self.base_manager.safe_execute_read_query(
                    entity_query,
                    sanitized_params
                )
                
                if not entity_records or len(entity_records) == 0:
                    return dict_to_json({"error": f"Entity '{entity_name}' not found"})
                
                # Build query based on filters
                query_parts = ["MATCH (e:Entity {name: $name})-[:HAS_OBSERVATION]->(o:Observation)"]
                params = {"name": entity_name}
                
                if observation_type:
                    query_parts.append("WHERE o.type = $type")
                    params["type"] = observation_type
                
                query = f"""
                {query_parts[0]}
                {' '.join(query_parts[1:])}
                RETURN o.id as id, o.content as content, o.type as type, 
                       o.created as created, o.lastUpdated as lastUpdated
                """
                
                # Sanitize query parameters
                sanitized_query_params = sanitize_query_parameters(params)
                
                # Execute read query with validation
                records = self.base_manager.safe_execute_read_query(
                    query,
                    sanitized_query_params
                )
                
                observations = []
                if records:
                    for record in records:
                        # Extract observation data
                        observation = {
                            "entity": entity_name,
                            "content": record.get("content")
                        }
                        
                        # Add optional fields if they exist
                        for field in ["id", "type", "created", "lastUpdated"]:
                            if record.get(field):
                                observation[field] = record.get(field)
                        
                        observations.append(observation)
                
                return dict_to_json({"observations": observations})
            except ValueError as e:
                error_msg = f"Query validation error: {str(e)}"
                self.logger.error(error_msg)
                return dict_to_json({"error": error_msg})
            
        except Exception as e:
            error_msg = f"Error retrieving observations: {str(e)}"
            self.logger.error(error_msg)
            return dict_to_json({"error": error_msg})
    
    def update_observation(self, entity_name: str, observation_id: str, content: str, 
                          observation_type: Optional[str] = None) -> str:
        """
        Update an observation in the knowledge graph.
        
        Args:
            entity_name: The name of the entity
            observation_id: The ID of the observation to update
            content: The new content for the observation
            observation_type: Optional new type for the observation
            
        Returns:
            JSON string with the updated observation
        """
        try:
            self.base_manager.ensure_initialized()
            
            try:
                # Check if observation exists
                query = """
                MATCH (e:Entity {name: $entity_name})-[:HAS_OBSERVATION]->(o:Observation {id: $observation_id})
                RETURN o
                """
                
                # Sanitize parameters
                check_params = {
                    "entity_name": entity_name, 
                    "observation_id": observation_id
                }
                sanitized_check_params = sanitize_query_parameters(check_params)
                
                # Execute read query with validation
                records = self.base_manager.safe_execute_read_query(
                    query,
                    sanitized_check_params
                )
                
                if not records or len(records) == 0:
                    return dict_to_json({
                        "error": f"Observation with ID '{observation_id}' not found for entity '{entity_name}'"
                    })
                
                # Build update query
                update_parts = ["o.content = $content", "o.lastUpdated = datetime()"]
                params = {
                    "entity_name": entity_name,
                    "observation_id": observation_id,
                    "content": content
                }
                
                if observation_type:
                    update_parts.append("o.type = $type")
                    params["type"] = observation_type
                
                update_query = f"""
                MATCH (e:Entity {{name: $entity_name}})-[:HAS_OBSERVATION]->(o:Observation {{id: $observation_id}})
                SET {', '.join(update_parts)}
                RETURN o
                """
                
                # Sanitize parameters
                sanitized_params = sanitize_query_parameters(params)
                
                # Execute write query with validation
                updated_records = self.base_manager.safe_execute_write_query(
                    update_query,
                    sanitized_params
                )
                
                if updated_records and len(updated_records) > 0:
                    observation = updated_records[0].get("o")
                    
                    if observation:
                        # Convert to dictionary
                        observation_dict = dict(observation.items())
                        observation_dict["entity"] = entity_name
                        
                        return dict_to_json({"observation": observation_dict})
                
                return dict_to_json({
                    "error": f"Failed to update observation"
                })
            except ValueError as e:
                error_msg = f"Query validation error: {str(e)}"
                self.logger.error(error_msg)
                return dict_to_json({"error": error_msg})
            
        except Exception as e:
            error_msg = f"Error updating observation: {str(e)}"
            self.logger.error(error_msg)
            return dict_to_json({"error": error_msg})
    
    def delete_observation(self, entity_name: str, observation_content: Optional[str] = None, 
                          observation_id: Optional[str] = None) -> str:
        """
        Delete an observation from the knowledge graph.
        
        Args:
            entity_name: The name of the entity
            observation_content: The content of the observation to delete (exact match)
            observation_id: The ID of the observation to delete
            
        Returns:
            JSON string with the result of the deletion
        """
        try:
            self.base_manager.ensure_initialized()
            
            if not observation_content and not observation_id:
                return dict_to_json({"error": "Must provide either observation_content or observation_id"})
            
            try:
                # Build delete query
                query_parts = ["MATCH (e:Entity {name: $entity_name})-[r:HAS_OBSERVATION]->(o:Observation)"]
                params = {"entity_name": entity_name}
                
                if observation_id:
                    query_parts.append("WHERE o.id = $observation_id")
                    params["observation_id"] = observation_id
                elif observation_content:
                    query_parts.append("WHERE o.content = $content")
                    params["content"] = observation_content
                
                delete_query = f"""
                {' '.join(query_parts)}
                DELETE r, o
                RETURN count(o) as deleted_count
                """
                
                # Sanitize parameters
                sanitized_params = sanitize_query_parameters(params)
                
                # Execute write query with validation
                records = self.base_manager.safe_execute_write_query(
                    delete_query,
                    sanitized_params
                )
                
                deleted_count = 0
                if records and len(records) > 0:
                    deleted_count = records[0].get("deleted_count", 0)
                
                if deleted_count > 0:
                    return dict_to_json({
                        "status": "success", 
                        "message": f"Deleted {deleted_count} observation(s) from entity '{entity_name}'"
                    })
                else:
                    return dict_to_json({
                        "status": "success", 
                        "message": f"No matching observations found for entity '{entity_name}'"
                    })
            except ValueError as e:
                error_msg = f"Query validation error: {str(e)}"
                self.logger.error(error_msg)
                return dict_to_json({"error": error_msg})
            
        except Exception as e:
            error_msg = f"Error deleting observation: {str(e)}"
            self.logger.error(error_msg)
            return dict_to_json({"error": error_msg})
    
    def _add_observation_to_entity(self, entity_name: str, content: str, 
                                 observation_type: Optional[str] = None) -> Optional[str]:
        """
        Add an observation to an entity.
        
        Args:
            entity_name: The name of the entity
            content: The content of the observation
            observation_type: Optional type of the observation
            
        Returns:
            The ID of the created observation, or None if creation failed
        """
        if not self.base_manager.neo4j_driver:
            self.logger.debug("Neo4j driver not initialized, skipping observation creation")
            return None
        
        if not content or not entity_name:
            return None
        
        try:
            # First check if entity exists
            entity_query = """
            MATCH (e:Entity {name: $name})
            RETURN e
            """
            
            # Sanitize parameters
            sanitized_entity_params = sanitize_query_parameters({"name": entity_name})
            
            # Execute read query with validation
            entity_records = self.base_manager.safe_execute_read_query(
                entity_query,
                sanitized_entity_params
            )
            
            if not entity_records or len(entity_records) == 0:
                raise ValueError(f"Entity '{entity_name}' not found")
            
            # Generate unique ID for observation
            observation_id = generate_id()
            
            # Build query
            query_parts = [
                "MATCH (e:Entity {name: $entity_name})",
                "CREATE (o:Observation {",
                "id: $id,",
                "content: $content,",
                "created: datetime()"
            ]
            
            params = {
                "entity_name": entity_name,
                "id": observation_id,
                "content": content
            }
            
            if observation_type:
                query_parts.append(",type: $type")
                params["type"] = observation_type
            
            query_parts.append("})")
            query_parts.append("CREATE (e)-[r:HAS_OBSERVATION]->(o)")
            query_parts.append("RETURN o")
            
            query = "\n".join(query_parts)
            
            # Sanitize parameters
            sanitized_params = sanitize_query_parameters(params)
            
            # Execute write query with validation
            self.base_manager.safe_execute_write_query(
                query,
                sanitized_params
            )
            
            return observation_id
            
        except ValueError as e:
            self.logger.error(f"Validation error adding observation to entity: {str(e)}")
            raise
        except Exception as e:
            self.logger.error(f"Error adding observation to entity: {str(e)}")
            raise
    
    def _convert_to_dict(self, observation: Any) -> Dict[str, Any]:
        """Convert an observation object to a dictionary."""
        if isinstance(observation, dict):
            return observation
        
        try:
            # Try to convert to dict if it has __dict__ attribute
            return observation.__dict__
        except (AttributeError, TypeError):
            # If not dict-like, try to get basic attributes
            result = {}
            
            # Common attributes to try
            for attr in ["entity", "entityName", "content", "type"]:
                try:
                    value = getattr(observation, attr)
                    if attr == "entityName":
                        result["entity"] = value
                    else:
                        result[attr] = value
                except (AttributeError, TypeError):
                    pass
            
            return result 