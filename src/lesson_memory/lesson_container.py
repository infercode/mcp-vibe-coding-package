from typing import Any, Dict, List, Optional, Union
import time
import datetime
import json

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
    """
    
    def __init__(self, base_manager: BaseManager):
        """
        Initialize a lesson container.
        
        Args:
            base_manager: Base manager for graph operations
        """
        self.base_manager = base_manager
        self.logger = base_manager.logger
    
    def create_container(self, name: str, description: Optional[str] = None,
                      metadata: Optional[Dict[str, Any]] = None) -> str:
        """
        Create a new lesson container in the knowledge graph.
        
        Args:
            name: Unique name for the container
            description: Optional description
            metadata: Optional metadata dictionary
        
        Returns:
            JSON string with the created container
        """
        try:
            self.base_manager.ensure_initialized()
            
            # Create Pydantic model for validation
            try:
                container_data = {
                    "name": name,
                    "description": description
                }
                
                if metadata:
                    container_data["metadata"] = metadata
                
                container_model = LessonContainerCreate(**container_data)
            except Exception as e:
                error_msg = f"Validation error for container data: {str(e)}"
                self.logger.error(error_msg)
                error_response = create_lesson_error_response(
                    message=error_msg,
                    code="validation_error"
                )
                return model_to_json(error_response)
            
            # Check if container already exists
            existing = self._get_container_by_name(name)
            if existing:
                error_response = create_lesson_error_response(
                    message=f"Lesson container '{name}' already exists",
                    code="container_exists",
                    details={"container": existing}
                )
                return model_to_json(error_response)
            
            # Generate unique ID
            container_id = generate_id()
            
            # Create query
            query_parts = [
                "CREATE (c:LessonContainer {",
                "id: $id,",
                "name: $name,",
                "created: datetime()"
            ]
            
            params = {
                "id": container_id,
                "name": container_model.name
            }
            
            if container_model.description:
                query_parts.append(",description: $description")
                params["description"] = container_model.description
            
            # Handle metadata
            if container_model.metadata:
                for key, value in container_model.metadata.items():
                    if key not in ["id", "name", "created", "lastUpdated", "description"]:
                        query_parts.append(f",{key}: ${key}")
                        params[key] = value
            
            # Add tags if provided
            if container_model.tags and len(container_model.tags) > 0:
                query_parts.append(",tags: $tags")
                # Convert tags list to JSON string for Neo4j compatibility
                params["tags"] = json.dumps(container_model.tags)
            
            query_parts.append("})")
            query_parts.append("RETURN c")
            
            query = "\n".join(query_parts)
            
            records, _ = self.base_manager.safe_execute_query(
                query, 
                params
            )
            
            if records and len(records) > 0:
                container = records[0].get("c")
                if container:
                    container_dict = dict(container.items())
                    self.logger.info(f"Created lesson container: {name}")
                    
                    response = create_container_response(
                        container_data=container_dict,
                        message=f"Lesson container '{name}' created successfully"
                    )
                    return model_to_json(response)
            
            error_response = create_lesson_error_response(
                message="Failed to create lesson container",
                code="container_creation_error"
            )
            return model_to_json(error_response)
                
        except Exception as e:
            error_msg = f"Error creating lesson container: {str(e)}"
            self.logger.error(error_msg)
            error_response = create_lesson_error_response(
                message=error_msg,
                code="container_creation_error"
            )
            return model_to_json(error_response)
    
    def get_container(self, container_name: str) -> str:
        """
        Get a lesson container by name.
        
        Args:
            container_name: Name of the container
            
        Returns:
            JSON string with the container
        """
        try:
            self.base_manager.ensure_initialized()
            
            container = self._get_container_by_name(container_name)
            if container:
                response = create_container_response(
                    container_data=container,
                    message=f"Retrieved lesson container '{container_name}'"
                )
                return model_to_json(response)
            
            error_response = create_lesson_error_response(
                message=f"Lesson container '{container_name}' not found",
                code="container_not_found"
            )
            return model_to_json(error_response)
            
        except Exception as e:
            error_msg = f"Error retrieving lesson container: {str(e)}"
            self.logger.error(error_msg)
            error_response = create_lesson_error_response(
                message=error_msg,
                code="container_retrieval_error"
            )
            return model_to_json(error_response)
    
    def update_container(self, container_name: str, updates: Dict[str, Any]) -> str:
        """
        Update a lesson container.
        
        Args:
            container_name: Name of the container to update
            updates: Dictionary of property updates
            
        Returns:
            JSON string with the updated container
        """
        try:
            self.base_manager.ensure_initialized()
            
            # Create Pydantic model for validation
            try:
                container_update = LessonContainerUpdate(
                    container_name=container_name,
                    updates=updates
                )
            except Exception as e:
                error_msg = f"Validation error for container update: {str(e)}"
                self.logger.error(error_msg)
                error_response = create_lesson_error_response(
                    message=error_msg,
                    code="validation_error"
                )
                return model_to_json(error_response)
            
            # Check if container exists
            container = self._get_container_by_name(container_name)
            if not container:
                error_response = create_lesson_error_response(
                    message=f"Lesson container '{container_name}' not found",
                    code="container_not_found"
                )
                return model_to_json(error_response)
            
            # Build update query
            update_set_clauses = ["c.lastUpdated = datetime()"]
            params = {"name": container_name}
            
            for key, value in container_update.updates.items():
                if key not in ["id", "name", "created"]:  # Don't update these fields
                    if key == "tags" and isinstance(value, list):
                        # Handle tags specially for Neo4j compatibility
                        update_set_clauses.append(f"c.{key} = ${key}")
                        params[key] = json.dumps(value)
                    else:
                        update_set_clauses.append(f"c.{key} = ${key}")
                        params[key] = value
            
            if len(update_set_clauses) > 1:  # Only proceed if there are updates beyond lastUpdated
                update_query = f"""
                MATCH (c:LessonContainer {{name: $name}})
                SET {', '.join(update_set_clauses)}
                RETURN c
                """
                
                records, _ = self.base_manager.safe_execute_query(
                    update_query,
                    params
                )
                
                if records and len(records) > 0:
                    container = records[0].get("c")
                    if container:
                        container_dict = dict(container.items())
                        self.logger.info(f"Updated lesson container: {container_name}")
                        
                        response = create_container_response(
                            container_data=container_dict,
                            message=f"Lesson container '{container_name}' updated successfully"
                        )
                        return model_to_json(response)
            
            error_response = create_lesson_error_response(
                message="Failed to update lesson container",
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
    
    def delete_container(self, container_name: str, delete_entities: bool = False) -> str:
        """
        Delete a lesson container and optionally its entities.
        
        Args:
            container_name: Name of the container to delete
            delete_entities: Whether to delete contained entities
            
        Returns:
            JSON string with the deletion result
        """
        try:
            self.base_manager.ensure_initialized()
            
            # Check if container exists
            container = self._get_container_by_name(container_name)
            if not container:
                # Return success but inform that container wasn't found
                response = create_success_response(
                    message=f"Lesson container '{container_name}' not found, nothing to delete"
                )
                return model_to_json(response)
            
            # Initialize entity count
            entity_count = 0
            
            # If delete_entities is True, delete all entities in the container
            if delete_entities:
                # Get entities in the container
                query = """
                MATCH (c:LessonContainer {name: $container_name})-[:CONTAINS]->(e)
                RETURN count(e) as entity_count
                """
                
                count_records, _ = self.base_manager.safe_execute_query(
                    query,
                    {"container_name": container_name}
                )
                
                if count_records and len(count_records) > 0:
                    entity_count = count_records[0].get("entity_count", 0)
                
                # Delete entities and their relationships
                delete_query = """
                MATCH (c:LessonContainer {name: $container_name})-[:CONTAINS]->(e)
                OPTIONAL MATCH (e)-[r]-()
                DELETE e, r
                """
                
                _, summary = self.base_manager.safe_execute_query(
                    delete_query,
                    {"container_name": container_name}
                )
                
                self.logger.info(f"Deleted {entity_count} entities from container '{container_name}'")
            
            # Now delete the container
            container_query = """
            MATCH (c:LessonContainer {name: $container_name})
            OPTIONAL MATCH (c)-[r]-()
            DELETE c, r
            """
            
            _, summary = self.base_manager.safe_execute_query(
                container_query,
                {"container_name": container_name}
            )
            
            self.logger.info(f"Deleted lesson container: {container_name}")
            
            # Create success response
            response = create_success_response(
                message=f"Lesson container '{container_name}' deleted successfully",
                data={
                    "deleted_entities": entity_count if delete_entities else 0,
                    "deleted_container": True
                }
            )
            return model_to_json(response)
                
        except Exception as e:
            error_msg = f"Error deleting lesson container: {str(e)}"
            self.logger.error(error_msg)
            error_response = create_lesson_error_response(
                message=error_msg,
                code="container_deletion_error"
            )
            return model_to_json(error_response)
    
    def list_containers(self, limit: int = 100, 
                      sort_by: str = "created", 
                      sort_desc: bool = True) -> str:
        """
        List all lesson containers.
        
        Args:
            limit: Maximum number of containers to return
            sort_by: Property to sort by
            sort_desc: Whether to sort in descending order
            
        Returns:
            JSON string with the list of containers
        """
        try:
            self.base_manager.ensure_initialized()
            
            # Validate parameters
            if limit <= 0:
                limit = 100
            
            valid_sort_fields = ["name", "created", "lastUpdated"]
            if sort_by not in valid_sort_fields:
                sort_by = "created"  # Default to created date
            
            # Build order direction
            order_dir = "DESC" if sort_desc else "ASC"
            
            # Build query
            query = f"""
            MATCH (c:LessonContainer)
            RETURN c
            ORDER BY c.{sort_by} {order_dir}
            LIMIT $limit
            """
            
            records, _ = self.base_manager.safe_execute_query(
                query,
                {"limit": limit}
            )
            
            # Process results
            containers = []
            if records:
                for record in records:
                    container = record.get("c")
                    if container:
                        container_dict = dict(container.items())
                        containers.append(container_dict)
            
            # Create search response
            response = create_search_response(
                results=containers,
                total_count=len(containers),
                query_params={
                    "limit": limit,
                    "sort_by": sort_by,
                    "sort_desc": sort_desc
                },
                message=f"Found {len(containers)} lesson containers"
            )
            return model_to_json(response)
                
        except Exception as e:
            error_msg = f"Error listing lesson containers: {str(e)}"
            self.logger.error(error_msg)
            error_response = create_lesson_error_response(
                message=error_msg,
                code="container_listing_error"
            )
            return model_to_json(error_response)
    
    def add_entity_to_container(self, container_name: str, entity_name: str) -> str:
        """
        Add an entity to a lesson container.
        
        Args:
            container_name: Name of the container
            entity_name: Name of the entity to add
            
        Returns:
            JSON string with the result
        """
        try:
            self.base_manager.ensure_initialized()
            
            # Check if container exists
            container = self._get_container_by_name(container_name)
            if not container:
                error_response = create_lesson_error_response(
                    message=f"Lesson container '{container_name}' not found",
                    code="container_not_found"
                )
                return model_to_json(error_response)
            
            # Check if entity exists
            entity_query = """
            MATCH (e)
            WHERE e.name = $entity_name
            RETURN e
            LIMIT 1
            """
            
            entity_records, _ = self.base_manager.safe_execute_query(
                entity_query,
                {"entity_name": entity_name}
            )
            
            if not entity_records or len(entity_records) == 0:
                error_response = create_lesson_error_response(
                    message=f"Entity '{entity_name}' not found",
                    code="entity_not_found"
                )
                return model_to_json(error_response)
            
            entity = None
            entity_node = entity_records[0].get("e") if entity_records and len(entity_records) > 0 else None
            if entity_node:
                entity = dict(entity_node.items())
            
            # Check if relationship already exists
            check_query = """
            MATCH (c:LessonContainer {name: $container_name})-[r:CONTAINS]->(e {name: $entity_name})
            RETURN count(r) as rel_count
            """
            
            check_records, _ = self.base_manager.safe_execute_query(
                check_query,
                {
                    "container_name": container_name,
                    "entity_name": entity_name
                }
            )
            
            rel_count = 0
            if check_records and len(check_records) > 0:
                rel_count = check_records[0].get("rel_count", 0)
            
            if rel_count > 0:
                # Relationship already exists
                response = create_success_response(
                    message=f"Entity '{entity_name}' is already in container '{container_name}'",
                    data={"entity": entity, "container": container}
                )
                return model_to_json(response)
            
            # Create relationship
            create_query = """
            MATCH (c:LessonContainer {name: $container_name})
            MATCH (e {name: $entity_name})
            CREATE (c)-[r:CONTAINS]->(e)
            SET e.lastUpdated = datetime()
            RETURN e
            """
            
            create_records, _ = self.base_manager.safe_execute_query(
                create_query,
                {
                    "container_name": container_name,
                    "entity_name": entity_name
                }
            )
            
            if create_records and len(create_records) > 0:
                updated_entity = create_records[0].get("e")
                if updated_entity:
                    entity_dict = dict(updated_entity.items())
                    self.logger.info(f"Added entity '{entity_name}' to container '{container_name}'")
                    
                    response = create_entity_response(
                        entity_data=entity_dict,
                        message=f"Entity '{entity_name}' added to container '{container_name}'"
                    )
                    return model_to_json(response)
            
            error_response = create_lesson_error_response(
                message=f"Failed to add entity '{entity_name}' to container '{container_name}'",
                code="container_update_error"
            )
            return model_to_json(error_response)
                
        except Exception as e:
            error_msg = f"Error adding entity to container: {str(e)}"
            self.logger.error(error_msg)
            error_response = create_lesson_error_response(
                message=error_msg,
                code="container_update_error"
            )
            return model_to_json(error_response)
    
    def remove_entity_from_container(self, container_name: str, entity_name: str) -> str:
        """
        Remove an entity from a lesson container.
        
        Args:
            container_name: Name of the container
            entity_name: Name of the entity to remove
            
        Returns:
            JSON string with the result
        """
        try:
            self.base_manager.ensure_initialized()
            
            # Check if container exists
            container = self._get_container_by_name(container_name)
            if not container:
                error_response = create_lesson_error_response(
                    message=f"Lesson container '{container_name}' not found",
                    code="container_not_found"
                )
                return model_to_json(error_response)
            
            # Check if entity exists and is in the container
            check_query = """
            MATCH (c:LessonContainer {name: $container_name})-[r:CONTAINS]->(e {name: $entity_name})
            RETURN e, count(r) as rel_count
            """
            
            check_records, _ = self.base_manager.safe_execute_query(
                check_query,
                {
                    "container_name": container_name,
                    "entity_name": entity_name
                }
            )
            
            rel_count = 0
            entity = None
            
            if check_records and len(check_records) > 0:
                rel_count = check_records[0].get("rel_count", 0)
                entity_node = check_records[0].get("e")
                if entity_node:
                    entity = dict(entity_node.items())
            
            if rel_count == 0:
                # Relationship doesn't exist
                response = create_success_response(
                    message=f"Entity '{entity_name}' is not in container '{container_name}'",
                    data={"container": container}
                )
                return model_to_json(response)
            
            # Delete relationship
            delete_query = """
            MATCH (c:LessonContainer {name: $container_name})-[r:CONTAINS]->(e {name: $entity_name})
            DELETE r
            RETURN e
            """
            
            delete_records, _ = self.base_manager.safe_execute_query(
                delete_query,
                {
                    "container_name": container_name,
                    "entity_name": entity_name
                }
            )
            
            self.logger.info(f"Removed entity '{entity_name}' from container '{container_name}'")
            
            if delete_records and len(delete_records) > 0:
                entity_node = delete_records[0].get("e")
                if entity_node:
                    entity_dict = dict(entity_node.items())
                    response = create_entity_response(
                        entity_data=entity_dict,
                        message=f"Entity '{entity_name}' removed from container '{container_name}'"
                    )
                    return model_to_json(response)
            
            # Return success even if we don't have the entity details
            response = create_success_response(
                message=f"Entity '{entity_name}' removed from container '{container_name}'",
                data={"entity_name": entity_name, "container_name": container_name}
            )
            return model_to_json(response)
                
        except Exception as e:
            error_msg = f"Error removing entity from container: {str(e)}"
            self.logger.error(error_msg)
            error_response = create_lesson_error_response(
                message=error_msg,
                code="container_update_error"
            )
            return model_to_json(error_response)
    
    def get_container_entities(self, container_name: str, 
                             entity_type: Optional[str] = None,
                             limit: int = 100) -> str:
        """
        Get entities in a container.
        
        Args:
            container_name: Name of the container
            entity_type: Optional filter by entity type
            limit: Maximum number of entities to return
            
        Returns:
            JSON string with the entities
        """
        try:
            self.base_manager.ensure_initialized()
            
            # Validate parameters
            if limit <= 0:
                limit = 100
            
            # Check if container exists
            container = self._get_container_by_name(container_name)
            if not container:
                error_response = create_lesson_error_response(
                    message=f"Lesson container '{container_name}' not found",
                    code="container_not_found"
                )
                return model_to_json(error_response)
            
            # Build query with optional type filter
            query_parts = [
                "MATCH (c:LessonContainer {name: $container_name})-[:CONTAINS]->(e)"
            ]
            
            params = {
                "container_name": container_name,
                "limit": limit
            }
            
            if entity_type:
                query_parts.append("WHERE e.type = $entity_type")
                params["entity_type"] = entity_type
            
            query_parts.append("RETURN e")
            query_parts.append("ORDER BY e.name")
            query_parts.append("LIMIT $limit")
            
            query = "\n".join(query_parts)
            
            records, _ = self.base_manager.safe_execute_query(
                query,
                params
            )
            
            # Process results
            entities = []
            if records:
                for record in records:
                    entity_node = record.get("e")
                    if entity_node:
                        entity_dict = dict(entity_node.items())
                        entities.append(entity_dict)
            
            # Build search response
            response = create_search_response(
                results=entities,
                total_count=len(entities),
                query_params={
                    "container_name": container_name,
                    "entity_type": entity_type,
                    "limit": limit
                },
                message=f"Found {len(entities)} entities in container '{container_name}'"
            )
            return model_to_json(response)
                
        except Exception as e:
            error_msg = f"Error getting container entities: {str(e)}"
            self.logger.error(error_msg)
            error_response = create_lesson_error_response(
                message=error_msg,
                code="entity_retrieval_error"
            )
            return model_to_json(error_response)
    
    def _get_container_by_name(self, container_name: str) -> Optional[Dict[str, Any]]:
        """
        Get a container by name.
        
        Args:
            container_name: Name of the container
            
        Returns:
            Container dictionary or None if not found
        """
        try:
            query = """
            MATCH (c:LessonContainer {name: $name})
            RETURN c
            LIMIT 1
            """
            
            records, _ = self.base_manager.safe_execute_query(
                query,
                {"name": container_name}
            )
            
            if records and len(records) > 0:
                container_node = records[0].get("c")
                if container_node:
                    return dict(container_node.items())
            
            return None
            
        except Exception as e:
            self.logger.error(f"Error in _get_container_by_name: {str(e)}")
            return None 