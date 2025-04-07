from typing import Any, Dict, List, Optional, Union, Tuple
import json
import logging
from datetime import datetime

from src.utils import dict_to_json
from src.utils.neo4j_query_utils import sanitize_query_parameters
from src.graph_memory.base_manager import BaseManager

class EntityManager:
    """Manager for entity CRUD operations in the knowledge graph."""
    
    def __init__(self, base_manager: BaseManager):
        """
        Initialize the entity manager.
        
        Args:
            base_manager: The base manager instance for database operations
        """
        self.base_manager = base_manager
        self.logger = base_manager.logger
    
    def create_entities(self, entities: List[Union[Dict, Any]]) -> str:
        """
        Create entities in the knowledge graph.
        
        Args:
            entities: List of entities to create
            
        Returns:
            JSON string with the created entities
        """
        try:
            self.base_manager.ensure_initialized()
            created_entities = []
            errors = []
            
            for entity_dict in entities:
                try:
                    # Get entity data from dictionary
                    entity_name = entity_dict.get("name", "")
                    
                    # Handle type field, which could be either "type" or "entityType"
                    entity_type = entity_dict.get("entityType", entity_dict.get("type", ""))
                    
                    if not entity_name:
                        raise ValueError("Entity name is required")
                    
                    if not entity_type:
                        raise ValueError("Entity type is required")
                    
                    # Get observations if any
                    observations = []
                    if "observations" in entity_dict:
                        observations = entity_dict["observations"]
                    
                    # Create entity in Neo4j
                    self._create_entity_in_neo4j(entity_name, entity_type, observations)
                    
                    # Add all other properties
                    additional_properties = {}
                    for key, value in entity_dict.items():
                        if key not in ["name", "entityType", "type", "observations"]:
                            additional_properties[key] = value
                    
                    if additional_properties:
                        update_query = """
                        MATCH (e:Entity {name: $name})
                        SET e += $properties
                        """
                        
                        self.base_manager.safe_execute_write_query(
                            update_query,
                            {"name": entity_name, "properties": additional_properties}
                        )
                    
                    # Generate and store embedding if enabled
                    if self.base_manager.embedding_enabled:
                        # Generate description for embedding
                        description = f"{entity_name} is a {entity_type}."
                        if observations:
                            description += " " + " ".join(observations)
                            
                        # Generate embedding
                        embedding = self.base_manager.generate_embedding(description)
                        
                        # Store embedding with entity
                        if embedding:
                            self._update_entity_embedding(entity_name, embedding)
                    
                    # Create a standardized entity object for the response
                    created_entity = {
                        "name": entity_name,
                        "type": entity_type,
                        "entityType": entity_type
                    }
                    
                    # Include observations in the response if they exist
                    if observations:
                        created_entity["observations"] = observations
                    
                    # Include additional properties in the response
                    for key, value in additional_properties.items():
                        created_entity[key] = value
                    
                    created_entities.append(created_entity)
                    self.logger.info(f"Created entity: {entity_name}")
                except Exception as e:
                    errors.append({
                        "name": entity_dict.get("name", "unknown"),
                        "error": str(e)
                    })
                    self.logger.error(f"Error creating entity: {entity_dict.get('name', 'unknown')}: {str(e)}")
            
            response = {"created": created_entities}
            if errors:
                response["errors"] = errors
            
            return dict_to_json(response)
            
        except Exception as e:
            error_msg = f"Error creating entities: {str(e)}"
            self.logger.error(error_msg)
            return dict_to_json({"error": error_msg})
    
    def get_entity(self, entity_name: str) -> str:
        """
        Get an entity from the knowledge graph.
        
        Args:
            entity_name: The name of the entity to retrieve
            
        Returns:
            JSON string with the entity information
        """
        try:
            self.base_manager.ensure_initialized()
            
            query = """
            MATCH (e:Entity {name: $name})
            RETURN e
            """
            
            # Use safe_execute_read_query for validation
            records = self.base_manager.safe_execute_read_query(
                query,
                {"name": entity_name}
            )
            
            if records and len(records) > 0:
                entity = records[0].get("e")
                
                if entity:
                    # Convert entity to dictionary
                    entity_dict = dict(entity.items())
                    
                    # Ensure both 'type' and 'entityType' fields exist
                    if "entityType" in entity_dict and "type" not in entity_dict:
                        entity_dict["type"] = entity_dict["entityType"]
                    elif "type" in entity_dict and "entityType" not in entity_dict:
                        entity_dict["entityType"] = entity_dict["type"]
                    
                    # If we still don't have type fields, add default type
                    if "type" not in entity_dict:
                        entity_dict["type"] = "Entity"
                        entity_dict["entityType"] = "Entity"
                    
                    # Get observations
                    observations = self._get_entity_observations(entity_name)
                    entity_dict["observations"] = observations
                    
                    return dict_to_json({"entity": entity_dict})
            
            return dict_to_json({"error": f"Entity '{entity_name}' not found"})
            
        except ValueError as e:
            error_msg = f"Query validation error: {str(e)}"
            self.logger.error(error_msg)
            return dict_to_json({"error": error_msg})
        except Exception as e:
            error_msg = f"Error retrieving entity: {str(e)}"
            self.logger.error(error_msg)
            return dict_to_json({"error": error_msg})
    
    def update_entity(self, entity_name: str, updates: Dict[str, Any]) -> str:
        """
        Update an entity in the knowledge graph.
        
        Args:
            entity_name: The name of the entity to update
            updates: Dictionary of property updates
            
        Returns:
            JSON string with the updated entity
        """
        try:
            self.base_manager.ensure_initialized()
            
            # Check if entity exists
            query = """
            MATCH (e:Entity {name: $name})
            RETURN e
            """
            
            # Use safe_execute_read_query for validation
            records = self.base_manager.safe_execute_read_query(
                query,
                {"name": entity_name}
            )
            
            if not records or len(records) == 0:
                return dict_to_json({"error": f"Entity '{entity_name}' not found"})
            
            # Build update query
            update_set_clauses = []
            params = {"name": entity_name}
            
            for key, value in updates.items():
                if key not in ["name", "observations"]:  # Don't update name or observations here
                    update_set_clauses.append(f"e.{key} = ${key}")
                    params[key] = value
            
            if update_set_clauses:
                update_query = f"""
                MATCH (e:Entity {{name: $name}})
                SET {', '.join(update_set_clauses)}
                RETURN e
                """
                
                # Use safe_execute_write_query for validation
                self.base_manager.safe_execute_write_query(
                    update_query,
                    params
                )
            
            # Handle observations separately if they exist in updates
            if "observations" in updates:
                self._update_entity_observations(entity_name, updates["observations"])
            
            # If entity type changed, update embedding
            if "entityType" in updates and self.base_manager.embedding_enabled:
                # Get entity with updated data
                updated_entity_query = """
                MATCH (e:Entity {name: $name})
                RETURN e
                """
                
                # Use safe_execute_read_query for validation
                updated_records = self.base_manager.safe_execute_read_query(
                    updated_entity_query,
                    {"name": entity_name}
                )
                
                if updated_records and len(updated_records) > 0:
                    entity = updated_records[0].get("e")
                    
                    if entity:
                        entity_dict = dict(entity.items())
                        entity_type = entity_dict.get("entityType", "")
                        
                        # Get observations
                        observations = self._get_entity_observations(entity_name)
                        
                        # Generate new embedding
                        description = f"{entity_name} is a {entity_type}."
                        if observations:
                            description += " " + " ".join(observations)
                        
                        embedding = self.base_manager.generate_embedding(description)
                        
                        if embedding:
                            self._update_entity_embedding(entity_name, embedding)
            
            return self.get_entity(entity_name)
            
        except ValueError as e:
            error_msg = f"Query validation error: {str(e)}"
            self.logger.error(error_msg)
            return dict_to_json({"error": error_msg})
        except Exception as e:
            error_msg = f"Error updating entity: {str(e)}"
            self.logger.error(error_msg)
            return dict_to_json({"error": error_msg})
    
    def delete_entity(self, entity_name: str) -> str:
        """
        Delete an entity from the knowledge graph.
        
        Args:
            entity_name: The name of the entity to delete
            
        Returns:
            JSON string with the result of the deletion
        """
        try:
            self.base_manager.ensure_initialized()
            
            # Delete entity relationships and observations
            delete_query = """
            MATCH (e:Entity {name: $name})
            OPTIONAL MATCH (e)-[r]-()
            DELETE r
            WITH e
            MATCH (e)-[r2:HAS_OBSERVATION]->(o:Observation)
            DELETE r2, o
            WITH e
            DELETE e
            RETURN count(e) as deleted_count
            """
            
            # Use safe_execute_write_query for validation
            records = self.base_manager.safe_execute_write_query(
                delete_query,
                {"name": entity_name}
            )
            
            deleted_count = 0
            if records and len(records) > 0:
                deleted_count = records[0].get("deleted_count", 0)
            
            if deleted_count > 0:
                return dict_to_json({"status": "success", "message": f"Entity '{entity_name}' deleted"})
            else:
                return dict_to_json({"status": "success", "message": f"Entity '{entity_name}' not found"})
            
        except ValueError as e:
            error_msg = f"Query validation error: {str(e)}"
            self.logger.error(error_msg)
            return dict_to_json({"error": error_msg})
        except Exception as e:
            error_msg = f"Error deleting entity: {str(e)}"
            self.logger.error(error_msg)
            return dict_to_json({"error": error_msg})
    
    def _create_entity_in_neo4j(self, name: str, entity_type: str, observations: Optional[List[str]] = None) -> None:
        """
        Create an entity in Neo4j.
        
        Args:
            name: The name of the entity
            entity_type: The type of the entity
            observations: Optional list of observation strings
            
        Raises:
            Exception: If there is an error creating the entity
        """
        # Create the entity node
        query = """
        MERGE (e:Entity {name: $name})
        SET e.entityType = $entity_type,
            e.type = $entity_type,
            e.created = datetime(),
            e.lastUpdated = datetime()
        RETURN e
        """
        
        try:
            # Sanitize parameters
            sanitized_params = sanitize_query_parameters({
                "name": name,
                "entity_type": entity_type
            })
            
            # Execute write query with validation
            self.base_manager.safe_execute_write_query(
                query,
                sanitized_params
            )
            
            # Add observations if provided
            if observations:
                for observation in observations:
                    self._add_observation_to_entity(name, observation)
        
        except Exception as e:
            self.logger.error(f"Error creating entity '{name}': {str(e)}")
            raise
    
    def _add_observation_to_entity(self, entity_name: str, observation_content: str) -> None:
        """Add an observation to an entity."""
        if not observation_content or not entity_name:
            return
        
        try:
            # Create observation query with uniqueness constraint
            query = """
            MATCH (e:Entity {name: $entity_name})
            MERGE (o:Observation {content: $content})
            MERGE (e)-[r:HAS_OBSERVATION]->(o)
            RETURN o
            """
            
            # Use safe_execute_write_query for validation
            self.base_manager.safe_execute_write_query(
                query,
                {"entity_name": entity_name, "content": observation_content}
            )
            
        except Exception as e:
            self.logger.error(f"Error adding observation to entity: {str(e)}")
    
    def _update_entity_embedding(self, entity_name: str, embedding: List[float]) -> None:
        """
        Update the embedding for an entity.
        
        Args:
            entity_name: The name of the entity
            embedding: The embedding vector
        """
        if not self.base_manager.neo4j_driver:
            return
            
        try:
            # Update entity with embedding
            query = """
            MATCH (e:Entity {name: $name})
            SET e.embedding = $embedding
            """
            
            # Use safe_execute_write_query for validation
            self.base_manager.safe_execute_write_query(
                query,
                {"name": entity_name, "embedding": embedding}
            )
            
            self.logger.info(f"Updated embedding for entity: {entity_name}")
        except Exception as e:
            self.logger.error(f"Error updating entity embedding: {str(e)}")
            # Don't raise - we can still use basic functionality without embeddings
    
    def _get_entity_observations(self, entity_name: str) -> List[str]:
        """Get all observations for an entity."""
        if not self.base_manager.neo4j_driver:
            return []
        
        try:
            query = """
            MATCH (e:Entity {name: $name})-[:HAS_OBSERVATION]->(o:Observation)
            RETURN o.content as content
            """
            
            # Use safe_execute_read_query for validation
            records = self.base_manager.safe_execute_read_query(
                query,
                {"name": entity_name}
            )
            
            observations = []
            if records:
                for record in records:
                    content = record.get("content")
                    if content:
                        observations.append(content)
            
            return observations
            
        except Exception as e:
            self.logger.error(f"Error retrieving entity observations: {str(e)}")
            return []
    
    def _update_entity_observations(self, entity_name: str, observations: List[str]) -> None:
        """Update observations for an entity."""
        if not self.base_manager.neo4j_driver:
            return
        
        try:
            # First, delete existing observations
            delete_query = """
            MATCH (e:Entity {name: $name})-[r:HAS_OBSERVATION]->(o:Observation)
            DELETE r, o
            """
            
            # Use safe_execute_write_query for validation
            self.base_manager.safe_execute_write_query(
                delete_query,
                {"name": entity_name}
            )
            
            # Then add new observations
            for observation in observations:
                self._add_observation_to_entity(entity_name, observation)
                
        except Exception as e:
            self.logger.error(f"Error updating entity observations: {str(e)}")
    
    def _convert_to_dict(self, entity: Any) -> Dict[str, Any]:
        """Convert an entity object to a dictionary."""
        if isinstance(entity, dict):
            # Verify all values are serializable
            for key, value in entity.items():
                if isinstance(value, (list, tuple)):
                    # Check items in lists/tuples
                    for item in value:
                        if not self._is_serializable(item):
                            raise ValueError(f"Entity has non-serializable value in list for '{key}': {type(item)}")
                elif not self._is_serializable(value):
                    raise ValueError(f"Entity has non-serializable value for '{key}': {type(value)}")
            return entity
        
        try:
            # Try to convert to dict if it has __dict__ attribute
            result = entity.__dict__
            # Verify all values are serializable
            for key, value in result.items():
                if isinstance(value, (list, tuple)):
                    # Check items in lists/tuples
                    for item in value:
                        if not self._is_serializable(item):
                            raise ValueError(f"Entity has non-serializable value in list for '{key}': {type(item)}")
                elif not self._is_serializable(value):
                    raise ValueError(f"Entity has non-serializable value for '{key}': {type(value)}")
            return result
        except (AttributeError, TypeError):
            # If not dict-like, try to get basic attributes
            result = {}
            
            # Common attributes to try
            for attr in ["name", "entityType", "observations"]:
                try:
                    value = getattr(entity, attr)
                    if not self._is_serializable(value):
                        raise ValueError(f"Entity has non-serializable value for '{attr}': {type(value)}")
                    result[attr] = value
                except (AttributeError, TypeError):
                    pass
            
            return result
            
    def _is_serializable(self, value: Any) -> bool:
        """Check if a value is serializable for Neo4j."""
        if value is None:
            return True
        # Neo4j accepts these primitive types
        if isinstance(value, (str, int, float, bool)):
            return True
        # Lists and arrays are fine if their items are serializable
        if isinstance(value, (list, tuple)):
            return all(self._is_serializable(item) for item in value)
        # Small dictionaries with serializable values are ok
        if isinstance(value, dict):
            return all(isinstance(k, str) and self._is_serializable(v) for k, v in value.items())
        # Reject anything else
        return False 