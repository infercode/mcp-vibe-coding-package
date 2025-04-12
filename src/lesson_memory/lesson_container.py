from typing import Any, Dict, List, Optional, Union
import time
import datetime
import json
import logging

from src.utils import dict_to_json, generate_id
from src.graph_memory.base_manager import BaseManager
from src.models.lesson_memory import (
    LessonContainerCreate, LessonContainerUpdate, 
    ContainerResponse, ErrorResponse, SuccessResponse,
    SearchResponse, EntityResponse
)
from src.models.lesson_responses import (
    create_container_response, create_lesson_error_response,
    create_search_response, create_entity_response
)
from src.models.responses import model_to_json, create_success_response

class LessonContainer:
    """
    Container for lesson entities and relationships.
    Manages the lifecycle and organization of lesson knowledge.
    There should only be one lesson container in the system, always named 'Lessons'.
    """
    
    CONTAINER_NAME = "Lessons"  # Constant for the single container name
    
    def __init__(self, base_manager: BaseManager):
        """
        Initialize a lesson container.
        
        Args:
            base_manager: Base manager for graph operations
        """
        self.base_manager = base_manager
        self.logger = logging.getLogger(__name__)
    
    def create_container(self, description: Optional[str] = None,
                      metadata: Optional[Dict[str, Any]] = None) -> str:
        """
        Create the lesson container in the knowledge graph if it doesn't exist.
        Only one lesson container named 'Lessons' should exist in the system.
        
        Args:
            description: Optional description
            metadata: Optional metadata dictionary
        
        Returns:
            JSON string with the created container
        """
        try:
            self.base_manager.ensure_initialized()
            
            # Check if a container already exists
            check_query = """
            MATCH (c:LessonContainer)
            RETURN count(c) as container_count
            """
            
            count_records = self.base_manager.safe_execute_read_query(check_query, {})
            
            if count_records and len(count_records) > 0:
                container_count = count_records[0].get("container_count", 0)
                if container_count > 0:
                    error_msg = f"A lesson container already exists. Only one lesson container named '{self.CONTAINER_NAME}' is allowed in the system."
                    self.logger.error(error_msg)
                    return dict_to_json({"error": error_msg})
            
            # Build container properties
            container_props = {
                "name": self.CONTAINER_NAME,
                "description": description or ""
            }
            
            if metadata:
                # Add metadata as properties
                for key, value in metadata.items():
                    if key not in container_props:
                        container_props[key] = value
            
            # Create container
            query = """
            CREATE (c:LessonContainer $props)
            SET c.created = datetime(), c.lastUpdated = datetime()
            RETURN c
            """
            
            # Execute query with validated parameters
            try:
                records = self.base_manager.safe_execute_write_query(
                    query, 
                    {"props": container_props}
                )
                
                if records and len(records) > 0:
                    container_node = records[0]["c"]
                    container_dict = dict(container_node.items())
                    
                    # Convert Neo4j DateTime objects to ISO format strings
                    for key, value in container_dict.items():
                        if hasattr(value, 'iso_format'):
                            container_dict[key] = value.iso_format()
                    
                    self.logger.info(f"Created lesson container: {self.CONTAINER_NAME}")
                    return dict_to_json({
                        "container": container_dict
                    })
                else:
                    return dict_to_json({
                        "error": "Failed to create container"
                    })
                
            except Exception as e:
                error_msg = f"Error creating lesson container: {str(e)}"
                self.logger.error(error_msg)
                return dict_to_json({"error": error_msg})
                
        except Exception as e:
            error_msg = f"Error creating lesson container: {str(e)}"
            self.logger.error(error_msg)
            return dict_to_json({"error": error_msg})
    
    def get_container(self) -> str:
        """
        Get the lesson container. There should only be one lesson container
        in the Neo4j instance named 'Lessons'.
        
        Returns:
            JSON string with the container data
        """
        try:
            self.logger.info("Starting get_container method")
            self.base_manager.ensure_initialized()
            
            # First check if there are multiple containers
            check_query = """
            MATCH (c:LessonContainer)
            RETURN count(c) as container_count
            """
            
            self.logger.info("Executing container count query")
            count_records = self.base_manager.safe_execute_read_query(check_query, {})
            
            self.logger.info(f"Container count query results: {count_records}")
            
            if count_records and len(count_records) > 0:
                container_count = count_records[0].get("container_count", 0)
                self.logger.info(f"Found {container_count} lesson containers")
                
                if container_count > 1:
                    error_msg = "Multiple lesson containers found. There should only be one lesson container in the system."
                    self.logger.error(error_msg)
                    return dict_to_json({"error": error_msg})
                elif container_count == 0:
                    self.logger.info("No lesson containers found in the database")
                    return dict_to_json({
                        "status": "error",
                        "error": f"Lesson container '{self.CONTAINER_NAME}' not found",
                        "code": "container_not_found"
                    })
            
            # Query to get container
            query = """
            MATCH (c:LessonContainer {name: $name})
            RETURN c
            """
            
            self.logger.info(f"Executing container retrieval query for '{self.CONTAINER_NAME}'")
            # Execute query
            records = self.base_manager.safe_execute_read_query(
                query, 
                {"name": self.CONTAINER_NAME}
            )
            
            self.logger.info(f"Container retrieval results: {records}")
            
            if records and len(records) > 0:
                container_node = records[0].get("c")
                if container_node:
                    self.logger.info(f"Found container node: {container_node}")
                    # Convert node to dictionary
                    container_dict = dict(container_node.items())
                    
                    # Convert Neo4j DateTime objects to ISO format strings
                    for key, value in container_dict.items():
                        if hasattr(value, 'iso_format'):
                            container_dict[key] = value.iso_format()
                    
                    self.logger.info(f"Processed container data: {container_dict}")
                    
                    # Create a success response with container data
                    response = {
                        "status": "success",
                        "message": f"Found container '{self.CONTAINER_NAME}'",
                        "container": container_dict
                    }
                    return dict_to_json(response)
                else:
                    self.logger.warning(f"Record returned but no 'c' field: {records}")
            else:
                self.logger.warning(f"No records returned for container query")
            
            # Container not found
            self.logger.warning(f"Container '{self.CONTAINER_NAME}' not found")
            return dict_to_json({
                "status": "error",
                "error": f"Lesson container '{self.CONTAINER_NAME}' not found",
                "code": "container_not_found"
            })
            
        except Exception as e:
            error_msg = f"Error retrieving lesson container: {str(e)}"
            self.logger.error(error_msg)
            return dict_to_json({"error": error_msg})
    
    def update_container(self, updates: Dict[str, Any]) -> str:
        """
        Update the lesson container.
        
        Args:
            updates: Dictionary of property updates
            
        Returns:
            JSON string with the updated container
        """
        try:
            self.base_manager.ensure_initialized()
            
            # Create Pydantic model for validation
            try:
                # Strip any fields that shouldn't be updated
                valid_updates = {}
                for key, value in updates.items():
                    if key not in ["id", "name", "created"]:
                        valid_updates[key] = value
                
                update_model = LessonContainerUpdate(**valid_updates)
            except Exception as e:
                error_msg = f"Validation error for container updates: {str(e)}"
                self.logger.error(error_msg)
                error_response = create_lesson_error_response(
                    message=error_msg,
                    code="validation_error"
                )
                return model_to_json(error_response)
            
            # Check if container exists
            container = self._get_container()
            if not container:
                error_response = create_lesson_error_response(
                    message=f"Lesson container '{self.CONTAINER_NAME}' not found",
                    code="container_not_found"
                )
                return model_to_json(error_response)
            
            # Build update query
            set_parts = []
            params = {"name": self.CONTAINER_NAME}
            
            for key, value in valid_updates.items():
                # Skip core properties that shouldn't be updated
                if key in ["id", "name", "created"]:
                    continue
                
                # Handle tags specially for neo4j compatibility
                if key == "tags" and isinstance(value, list):
                    set_parts.append(f"c.{key} = ${key}")
                    params[key] = json.dumps(value)
                    continue
                
                # Handle other properties
                set_parts.append(f"c.{key} = ${key}")
                params[key] = value
            
            # Add lastUpdated timestamp
            set_parts.append("c.lastUpdated = datetime()")
            
            if not set_parts:
                # No valid updates to apply
                response = create_container_response(
                    container_data=container,
                    message=f"No updates provided for container '{self.CONTAINER_NAME}'"
                )
                return model_to_json(response)
            
            # Build and execute update query
            update_query = f"""
            MATCH (c:LessonContainer {{name: $name}})
            SET {', '.join(set_parts)}
            RETURN c
            """
            
            # Use safe_execute_write_query for validation (write operation)
            update_records = self.base_manager.safe_execute_write_query(
                update_query,
                params
            )
            
            if update_records and len(update_records) > 0:
                updated_container = update_records[0].get("c")
                if updated_container:
                    container_dict = dict(updated_container.items())
                    self.logger.info(f"Updated lesson container: {self.CONTAINER_NAME}")
                    
                    response = create_container_response(
                        container_data=container_dict,
                        message=f"Lesson container '{self.CONTAINER_NAME}' updated successfully"
                    )
                    return model_to_json(response)
            
            error_response = create_lesson_error_response(
                message=f"Failed to update lesson container '{self.CONTAINER_NAME}'",
                code="container_update_error"
            )
            return model_to_json(error_response)
                
        except Exception as e:
            error_msg = f"Error updating lesson container: {str(e)}"
            self.logger.error(error_msg)
            error_response = create_lesson_error_response(
                message=error_msg,
                code="container_update_error"
            )
            return model_to_json(error_response)
    
    def delete_container(self, delete_entities: bool = False) -> str:
        """
        Delete the lesson container.
        
        Args:
            delete_entities: Whether to delete entities in the container
            
        Returns:
            JSON string with the deletion result
        """
        try:
            self.base_manager.ensure_initialized()
            
            # Check if container exists
            container = self._get_container()
            if not container:
                error_response = create_lesson_error_response(
                    message=f"Lesson container '{self.CONTAINER_NAME}' not found",
                    code="container_not_found"
                )
                return model_to_json(error_response)
            
            # Check if container has entities
            entity_count_query = """
            MATCH (c:LessonContainer {name: $name})-[:CONTAINS]->(e:Entity)
            RETURN count(e) as entity_count
            """
            
            # Use safe_execute_read_query for validation (read-only operation)
            count_records = self.base_manager.safe_execute_read_query(
                entity_count_query,
                {"name": self.CONTAINER_NAME}
            )
            
            entity_count = 0
            if count_records and len(count_records) > 0:
                entity_count = count_records[0].get("entity_count", 0)
            
            # If container has entities and delete_entities is False, prevent deletion
            if entity_count > 0 and not delete_entities:
                error_response = create_lesson_error_response(
                    message=f"Cannot delete container '{self.CONTAINER_NAME}' with {entity_count} entities. Set delete_entities=True to force deletion.",
                    code="container_not_empty",
                    details={"entity_count": entity_count}
                )
                return model_to_json(error_response)
            
            # Delete all entities in the container if requested
            if delete_entities and entity_count > 0:
                # Delete all observations of entities in the container
                delete_observations_query = """
                MATCH (c:LessonContainer {name: $name})-[:CONTAINS]->(e:Entity)-[:HAS_OBSERVATION]->(o:Observation)
                DETACH DELETE o
                """
                
                # Use safe_execute_write_query for validation (write operation)
                self.base_manager.safe_execute_write_query(
                    delete_observations_query,
                    {"name": self.CONTAINER_NAME}
                )
                
                # Delete all relationships between entities in the container
                delete_relationships_query = """
                MATCH (c:LessonContainer {name: $name})-[:CONTAINS]->(e1:Entity)
                MATCH (e1)-[r]->(e2:Entity)
                WHERE NOT type(r) = 'HAS_OBSERVATION'
                DELETE r
                """
                
                # Use safe_execute_write_query for validation (write operation)
                self.base_manager.safe_execute_write_query(
                    delete_relationships_query,
                    {"name": self.CONTAINER_NAME}
                )
                
                # Delete all container-entity relationships
                delete_container_relations_query = """
                MATCH (c:LessonContainer {name: $name})-[r:CONTAINS]->(e:Entity)
                DELETE r
                """
                
                # Use safe_execute_write_query for validation (write operation)
                self.base_manager.safe_execute_write_query(
                    delete_container_relations_query,
                    {"name": self.CONTAINER_NAME}
                )
                
                # Delete all entities
                delete_entities_query = """
                MATCH (c:LessonContainer {name: $name})-[:CONTAINS]->(e:Entity)
                DETACH DELETE e
                """
                
                # Use safe_execute_write_query for validation (write operation)
                self.base_manager.safe_execute_write_query(
                    delete_entities_query,
                    {"name": self.CONTAINER_NAME}
                )
            
            # Delete the container
            delete_container_query = """
            MATCH (c:LessonContainer {name: $name})
            DETACH DELETE c
            """
            
            # Use safe_execute_write_query for validation (write operation)
            self.base_manager.safe_execute_write_query(
                delete_container_query,
                {"name": self.CONTAINER_NAME}
            )
            
            self.logger.info(f"Deleted lesson container: {self.CONTAINER_NAME}")
            
            success_response = create_success_response(
                message=f"Lesson container '{self.CONTAINER_NAME}' deleted successfully",
                data={"deleted_entities": entity_count if delete_entities else 0}
            )
            return model_to_json(success_response)
                
        except Exception as e:
            error_msg = f"Error deleting lesson container: {str(e)}"
            self.logger.error(error_msg)
            error_response = create_lesson_error_response(
                message=error_msg,
                code="container_deletion_error"
            )
            return model_to_json(error_response)
    
    def list_containers(self, filter_tags: Optional[List[str]] = None, 
                      name_filter: Optional[str] = None,
                      limit: int = 100) -> str:
        """
        List all lesson containers, optionally filtered by tags or name.
        
        Args:
            filter_tags: Optional list of tags to filter by
            name_filter: Optional name substring to filter by
            limit: Maximum number of containers to return
            
        Returns:
            JSON string with the list of containers
        """
        try:
            self.base_manager.ensure_initialized()
            
            # Build query parts
            query_parts = ["MATCH (c:LessonContainer)"]
            where_clauses = []
            params = {}  # Remove type constraint by not pre-assigning limit
            params["limit"] = min(limit, 1000)  # Cap limit for safety
            
            # Filter by name
            if name_filter:
                where_clauses.append("c.name CONTAINS $name_filter")
                params["name_filter"] = name_filter
            
            # Filter by tags
            if filter_tags and len(filter_tags) > 0:
                # Tags are stored as JSON strings in Neo4j
                # This makes filtering more complex
                tag_conditions = []
                for i, tag in enumerate(filter_tags):
                    tag_param = f"tag_{i}"
                    tag_conditions.append(f"c.tags CONTAINS $" + tag_param)
                    params[tag_param] = tag
                
                if tag_conditions:
                    where_clauses.append("(" + " OR ".join(tag_conditions) + ")")
            
            # Add WHERE clause if there are filters
            if where_clauses:
                query_parts.append("WHERE " + " AND ".join(where_clauses))
            
            # Complete the query
            query_parts.append("RETURN c")
            query_parts.append("ORDER BY c.name")
            query_parts.append("LIMIT $limit")
            
            query = "\n".join(query_parts)
            
            # Use safe_execute_read_query for validation (read-only operation)
            records = self.base_manager.safe_execute_read_query(
                query,
                params
            )
            
            # Process results
            containers = []
            if records:
                for record in records:
                    container = record.get("c")
                    if container:
                        container_dict = dict(container.items())
                        
                        # Parse tags from JSON string to list
                        if "tags" in container_dict and isinstance(container_dict["tags"], str):
                            try:
                                container_dict["tags"] = json.loads(container_dict["tags"])
                            except:
                                # If parsing fails, keep as string
                                pass
                        
                        containers.append(container_dict)
            
            # Return results directly instead of using create_search_response which has incorrect params
            return dict_to_json({
                "containers": containers,
                "count": len(containers),
                "message": f"Retrieved {len(containers)} lesson containers"
            })
                
        except Exception as e:
            error_msg = f"Error listing lesson containers: {str(e)}"
            self.logger.error(error_msg)
            error_response = create_lesson_error_response(
                message=error_msg,
                code="container_list_error"
            )
            return model_to_json(error_response)
    
    def _get_container(self) -> Optional[Dict[str, Any]]:
        """
        Internal method to get the lesson container.
        
        Returns:
            Container dict or None if not found
        """
        try:
            query = """
            MATCH (c:LessonContainer {name: $name})
            RETURN c
            """
            
            # Use safe_execute_read_query for validation (read-only operation)
            records = self.base_manager.safe_execute_read_query(
                query,
                {"name": self.CONTAINER_NAME}
            )
            
            if records and len(records) > 0:
                container = records[0].get("c")
                if container:
                    container_dict = dict(container.items())
                    
                    # Parse tags from JSON string to list
                    if "tags" in container_dict and isinstance(container_dict["tags"], str):
                        try:
                            container_dict["tags"] = json.loads(container_dict["tags"])
                        except:
                            # If parsing fails, keep as string
                            pass
                    
                    return container_dict
                    
            return None
            
        except Exception as e:
            self.logger.error(f"Error in _get_container: {str(e)}")
            return None 