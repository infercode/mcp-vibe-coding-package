from typing import Any, Dict, List, Optional, Union, Tuple
import json
import uuid
from datetime import datetime

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
                metadata = observation_dict.get("metadata", None)
                confidence = observation_dict.get("confidence", None)
                
                if entity_name and content:
                    # Add the observation in Neo4j
                    try:
                        observation_id, created_timestamp = self._add_observation_to_entity(
                            entity_name, 
                            content, 
                            observation_type,
                            metadata,
                            confidence
                        )
                        
                        # Format for response
                        added_observation = {
                            "entity": entity_name,
                            "content": content,
                            "created": created_timestamp
                        }
                        
                        if observation_type:
                            added_observation["type"] = observation_type
                            
                        if observation_id:
                            added_observation["id"] = observation_id
                            
                        if metadata:
                            added_observation["metadata"] = metadata
                            
                        if confidence is not None:
                            added_observation["confidence"] = confidence
                        
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
                       o.created as created, o.lastUpdated as lastUpdated,
                       o.metadata as metadata, o.confidence as confidence
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
                        
                        # Handle metadata (stored as JSON string)
                        if record.get("metadata"):
                            try:
                                metadata_str = str(record.get("metadata"))
                                metadata = json.loads(metadata_str)
                                observation["metadata"] = metadata
                            except json.JSONDecodeError:
                                # If not valid JSON, store as is
                                observation["metadata"] = record.get("metadata")
                        
                        # Handle confidence
                        if record.get("confidence"):
                            try:
                                confidence_str = str(record.get("confidence"))
                                confidence = float(confidence_str)
                                observation["confidence"] = confidence
                            except ValueError:
                                # If not a valid float, store as is
                                observation["confidence"] = record.get("confidence")
                        
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
                
                # Update observation with new values
                update_parts = ["o.content = $content", "o.lastUpdated = datetime()"]
                update_params = {
                    "entity_name": entity_name,
                    "observation_id": observation_id,
                    "content": content
                }
                
                if observation_type:
                    update_parts.append("o.type = $type")
                    update_params["type"] = observation_type
                
                update_query = f"""
                MATCH (e:Entity {{name: $entity_name}})-[:HAS_OBSERVATION]->(o:Observation {{id: $observation_id}})
                SET {', '.join(update_parts)}
                RETURN o, o.lastUpdated as lastUpdated
                """
                
                # Sanitize parameters
                sanitized_update_params = sanitize_query_parameters(update_params)
                
                # Execute write query with validation
                updated_records = self.base_manager.safe_execute_write_query(
                    update_query,
                    sanitized_update_params
                )
                
                # Extract updated time from results
                last_updated = None
                if updated_records and len(updated_records) > 0:
                    last_updated = updated_records[0].get("lastUpdated")
                
                # Format response
                observation = {
                    "entity": entity_name,
                    "id": observation_id,
                    "content": content,
                    "lastUpdated": last_updated
                }
                
                if observation_type:
                    observation["type"] = observation_type
                
                return dict_to_json({"observation": observation})
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
                                 observation_type: Optional[str] = None,
                                 metadata: Optional[Dict[str, Any]] = None,
                                 confidence: Optional[float] = None) -> Tuple[Optional[str], Optional[str]]:
        """
        Add an observation to an entity.
        
        Args:
            entity_name: The name of the entity
            content: The content of the observation
            observation_type: Optional type of the observation
            metadata: Optional metadata for the observation
            confidence: Optional confidence score (0-1)
            
        Returns:
            Tuple of (observation_id, created_timestamp)
            
        Raises:
            Exception: If the entity does not exist or there is an error adding the observation
        """
        # Validate entity exists
        check_query = """
        MATCH (e:Entity {name: $name})
        RETURN e
        """
        
        sanitized_params = sanitize_query_parameters({"name": entity_name})
        
        # Execute read query with validation
        entity_records = self.base_manager.safe_execute_read_query(
            check_query,
            sanitized_params
        )
        
        if not entity_records or len(entity_records) == 0:
            self.logger.error(f"Validation error adding observation to entity: Entity '{entity_name}' not found")
            raise Exception(f"Entity '{entity_name}' not found")
        
        # Generate a unique ID for the observation
        observation_id = str(uuid.uuid4())
        
        # Current timestamp
        timestamp = datetime.now().isoformat()
        
        # Handle metadata
        metadata_json = None
        if metadata:
            try:
                metadata_json = json.dumps(metadata)
            except (TypeError, json.JSONDecodeError):
                self.logger.error(f"Could not encode metadata for observation: {metadata}")
        
        # Create observation node
        params = {
            "entity_name": entity_name,
            "observation_id": observation_id,
            "content": content,
            "created": timestamp,
            "lastUpdated": timestamp
        }
        
        if observation_type:
            params["type"] = observation_type
            
        if metadata_json:
            params["metadata"] = metadata_json
            
        if confidence is not None:
            params["confidence"] = str(confidence)
        
        # Building the SET clause dynamically for optional parameters
        set_clauses = [
            "o.id = $observation_id",
            "o.content = $content",
            "o.created = $created",
            "o.lastUpdated = $lastUpdated"
        ]
        
        if observation_type:
            set_clauses.append("o.type = $type")
            
        if metadata_json:
            set_clauses.append("o.metadata = $metadata")
            
        if confidence is not None:
            set_clauses.append("o.confidence = $confidence")
        
        # Create observation and link to entity
        query = f"""
        MATCH (e:Entity {{name: $entity_name}})
        CREATE (o:Observation)
        SET {', '.join(set_clauses)}
        CREATE (e)-[r:HAS_OBSERVATION]->(o)
        RETURN o.id as id, o.created as created
        """
        
        # Sanitize query parameters
        sanitized_query_params = sanitize_query_parameters(params)
        
        # Execute write query with validation
        records = self.base_manager.safe_execute_write_query(
            query,
            sanitized_query_params
        )
        
        if records and len(records) > 0:
            # Get observation_id and timestamp
            result_id = records[0].get("id")
            result_created = records[0].get("created")
            return result_id, result_created
        
        return None, None
    
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

    def get_observations(self, entity_name: str, observation_type: Optional[str] = None) -> str:
        """
        Get observations for an entity.
        
        Args:
            entity_name: Name of the entity to get observations for
            observation_type: Optional type of observations to retrieve
            
        Returns:
            JSON string with the observations
        """
        try:
            self.base_manager.ensure_initialized()
            
            # Build query based on filters
            query_parts = []
            params = {"entity_name": entity_name}
            
            # Match entity and its observations
            query_parts.append("(e:Entity {name: $entity_name})-[:HAS_OBSERVATION]->(o:Observation)")
            
            if observation_type:
                query_parts.append("o.type = $observation_type")
                params["observation_type"] = observation_type
            
            query = f"""
            MATCH {query_parts[0]}
            {' WHERE ' + ' AND '.join(query_parts[1:]) if len(query_parts) > 1 else ''}
            RETURN o.content as content, o.type as type, o.metadata as metadata,
                   o.confidence as confidence, o.created as created, 
                   o.lastUpdated as lastUpdated, id(o) as id
            """
            
            try:
                # Use safe_execute_read_query for validation
                records = self.base_manager.safe_execute_read_query(
                    query,
                    params
                )
                
                observations = []
                if records:
                    for record in records:
                        observation = {
                            "content": record.get("content", ""),
                            "type": record.get("type", "GENERAL"),
                            "id": str(record.get("id", ""))
                        }
                        
                        # Add optional properties if they exist
                        if record.get("metadata"):
                            try:
                                metadata_str = str(record.get("metadata"))
                                metadata = json.loads(metadata_str)
                                observation["metadata"] = metadata
                            except json.JSONDecodeError:
                                # If not valid JSON, store as is
                                observation["metadata"] = record.get("metadata")
                        
                        if record.get("confidence") is not None:
                            try:
                                confidence_str = str(record.get("confidence"))
                                confidence = float(confidence_str)
                                observation["confidence"] = confidence
                            except ValueError:
                                # If not a valid float, store as is
                                observation["confidence"] = record.get("confidence")
                        
                        # Add timestamps
                        if record.get("created"):
                            observation["created"] = record.get("created")
                        
                        if record.get("lastUpdated"):
                            observation["lastUpdated"] = record.get("lastUpdated")
                        
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