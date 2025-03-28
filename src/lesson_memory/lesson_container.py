from typing import Any, Dict, List, Optional, Union
import time
import datetime

from src.utils import dict_to_json, generate_id
from src.graph_memory.base_manager import BaseManager

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
            
            # Check if container already exists
            existing = self._get_container_by_name(name)
            if existing:
                return dict_to_json({
                    "error": f"Lesson container '{name}' already exists",
                    "container": existing
                })
            
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
                "name": name
            }
            
            if description:
                query_parts.append(",description: $description")
                params["description"] = description
            
            # Handle metadata
            if metadata:
                for key, value in metadata.items():
                    if key not in ["id", "name", "created", "lastUpdated", "description"]:
                        query_parts.append(f",{key}: ${key}")
                        params[key] = value
            
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
                    return dict_to_json({"container": container_dict})
            
            return dict_to_json({"error": "Failed to create lesson container"})
                
        except Exception as e:
            error_msg = f"Error creating lesson container: {str(e)}"
            self.logger.error(error_msg)
            return dict_to_json({"error": error_msg})
    
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
                return dict_to_json({"container": container})
            
            return dict_to_json({"error": f"Lesson container '{container_name}' not found"})
            
        except Exception as e:
            error_msg = f"Error retrieving lesson container: {str(e)}"
            self.logger.error(error_msg)
            return dict_to_json({"error": error_msg})
    
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
            
            # Check if container exists
            container = self._get_container_by_name(container_name)
            if not container:
                return dict_to_json({"error": f"Lesson container '{container_name}' not found"})
            
            # Build update query
            update_set_clauses = ["c.lastUpdated = datetime()"]
            params = {"name": container_name}
            
            for key, value in updates.items():
                if key not in ["id", "name", "created"]:  # Don't update these fields
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
                        return dict_to_json({"container": container_dict})
            
            return dict_to_json({"error": "Failed to update lesson container"})
                
        except Exception as e:
            error_msg = f"Error updating lesson container: {str(e)}"
            self.logger.error(error_msg)
            return dict_to_json({"error": error_msg})
    
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
                return dict_to_json({
                    "status": "success",
                    "message": f"Lesson container '{container_name}' not found"
                })
            
            # If delete_entities is True, delete all entities in the container
            if delete_entities:
                delete_entities_query = """
                MATCH (c:LessonContainer {name: $name})-[:CONTAINS]->(e:Entity)
                OPTIONAL MATCH (e)-[r]-()
                DELETE r
                WITH e
                OPTIONAL MATCH (e)-[ro:HAS_OBSERVATION]->(o:Observation)
                DELETE ro, o
                WITH e
                DELETE e
                """
                
                self.base_manager.safe_execute_query(
                    delete_entities_query,
                    {"name": container_name}
                )
                
                self.logger.info(f"Deleted entities in lesson container: {container_name}")
                
            # Delete all relationships between the container and its entities
            delete_relations_query = """
            MATCH (c:LessonContainer {name: $name})-[r:CONTAINS]->()
            DELETE r
            """
            
            self.base_manager.safe_execute_query(
                delete_relations_query,
                {"name": container_name}
            )
            
            # Delete the container
            delete_container_query = """
            MATCH (c:LessonContainer {name: $name})
            DELETE c
            RETURN count(c) as deleted_count
            """
            
            records, _ = self.base_manager.safe_execute_query(
                delete_container_query,
                {"name": container_name}
            )
            
            deleted_count = 0
            if records and len(records) > 0:
                deleted_count = records[0].get("deleted_count", 0)
            
            if deleted_count > 0:
                self.logger.info(f"Deleted lesson container: {container_name}")
                return dict_to_json({
                    "status": "success",
                    "message": f"Lesson container '{container_name}' deleted"
                })
            
            return dict_to_json({
                "error": f"Failed to delete lesson container '{container_name}'"
            })
                
        except Exception as e:
            error_msg = f"Error deleting lesson container: {str(e)}"
            self.logger.error(error_msg)
            return dict_to_json({"error": error_msg})
    
    def list_containers(self, limit: int = 100, 
                      sort_by: str = "created", 
                      sort_desc: bool = True) -> str:
        """
        List all lesson containers.
        
        Args:
            limit: Maximum number of containers to return
            sort_by: Field to sort by (e.g., created, name)
            sort_desc: Whether to sort in descending order
            
        Returns:
            JSON string with the list of containers
        """
        try:
            self.base_manager.ensure_initialized()
            
            # Validate sort field
            valid_sort_fields = ["id", "name", "created", "lastUpdated", "description"]
            if sort_by not in valid_sort_fields:
                sort_by = "created"  # Default to created if invalid
                
            # Build order clause
            order_direction = "DESC" if sort_desc else "ASC"
            order_clause = f"ORDER BY c.{sort_by} {order_direction}"
            
            # Build query
            query = f"""
            MATCH (c:LessonContainer)
            RETURN c
            {order_clause}
            LIMIT $limit
            """
            
            records, _ = self.base_manager.safe_execute_query(
                query,
                {"limit": min(limit, 1000)}  # Cap limit for safety
            )
            
            containers = []
            if records:
                for record in records:
                    container = record.get("c")
                    if container:
                        container_dict = dict(container.items())
                        containers.append(container_dict)
            
            return dict_to_json({"containers": containers})
                
        except Exception as e:
            error_msg = f"Error listing lesson containers: {str(e)}"
            self.logger.error(error_msg)
            return dict_to_json({"error": error_msg})
    
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
                return dict_to_json({"error": f"Lesson container '{container_name}' not found"})
            
            # Check if entity exists
            entity_query = """
            MATCH (e:Entity {name: $entity_name})
            RETURN e
            """
            
            entity_records, _ = self.base_manager.safe_execute_query(
                entity_query,
                {"entity_name": entity_name}
            )
            
            if not entity_records or len(entity_records) == 0:
                return dict_to_json({"error": f"Entity '{entity_name}' not found"})
            
            # Check if relationship already exists
            relation_query = """
            MATCH (c:LessonContainer {name: $container_name})-[r:CONTAINS]->(e:Entity {name: $entity_name})
            RETURN r
            """
            
            relation_records, _ = self.base_manager.safe_execute_query(
                relation_query,
                {"container_name": container_name, "entity_name": entity_name}
            )
            
            if relation_records and len(relation_records) > 0:
                return dict_to_json({
                    "status": "success",
                    "message": f"Entity '{entity_name}' is already in container '{container_name}'"
                })
            
            # Create relationship
            create_relation_query = """
            MATCH (c:LessonContainer {name: $container_name})
            MATCH (e:Entity {name: $entity_name})
            CREATE (c)-[r:CONTAINS {added: datetime()}]->(e)
            RETURN r
            """
            
            self.base_manager.safe_execute_query(
                create_relation_query,
                {"container_name": container_name, "entity_name": entity_name}
            )
            
            self.logger.info(f"Added entity '{entity_name}' to lesson container '{container_name}'")
            return dict_to_json({
                "status": "success",
                "message": f"Entity '{entity_name}' added to container '{container_name}'"
            })
                
        except Exception as e:
            error_msg = f"Error adding entity to container: {str(e)}"
            self.logger.error(error_msg)
            return dict_to_json({"error": error_msg})
    
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
            
            # Delete relationship
            delete_relation_query = """
            MATCH (c:LessonContainer {name: $container_name})-[r:CONTAINS]->(e:Entity {name: $entity_name})
            DELETE r
            RETURN count(r) as deleted_count
            """
            
            records, _ = self.base_manager.safe_execute_query(
                delete_relation_query,
                {"container_name": container_name, "entity_name": entity_name}
            )
            
            deleted_count = 0
            if records and len(records) > 0:
                deleted_count = records[0].get("deleted_count", 0)
            
            if deleted_count > 0:
                self.logger.info(f"Removed entity '{entity_name}' from lesson container '{container_name}'")
                return dict_to_json({
                    "status": "success",
                    "message": f"Entity '{entity_name}' removed from container '{container_name}'"
                })
            
            return dict_to_json({
                "status": "success",
                "message": f"Entity '{entity_name}' was not in container '{container_name}'"
            })
                
        except Exception as e:
            error_msg = f"Error removing entity from container: {str(e)}"
            self.logger.error(error_msg)
            return dict_to_json({"error": error_msg})
    
    def get_container_entities(self, container_name: str, 
                             entity_type: Optional[str] = None,
                             limit: int = 100) -> str:
        """
        Get all entities in a lesson container.
        
        Args:
            container_name: Name of the container
            entity_type: Optional entity type filter
            limit: Maximum number of entities to return
            
        Returns:
            JSON string with the entities
        """
        try:
            self.base_manager.ensure_initialized()
            
            # Check if container exists
            container = self._get_container_by_name(container_name)
            if not container:
                return dict_to_json({"error": f"Lesson container '{container_name}' not found"})
            
            # Build query based on filters
            query_parts = ["MATCH (c:LessonContainer {name: $container_name})-[:CONTAINS]->(e:Entity)"]
            params = {"container_name": container_name}
            
            if entity_type:
                query_parts.append("WHERE e.entityType = $entity_type")
                params["entity_type"] = entity_type
            
            query_parts.append("RETURN e")
            query_parts.append("LIMIT $limit")
            
            query = "\n".join(query_parts)
            
            records, _ = self.base_manager.safe_execute_query(
                query,
                {**params, "limit": min(limit, 1000)}  # Cap limit for safety
            )
            
            entities = []
            if records:
                for record in records:
                    entity = record.get("e")
                    if entity:
                        entity_dict = dict(entity.items())
                        entities.append(entity_dict)
            
            return dict_to_json({"entities": entities})
                
        except Exception as e:
            error_msg = f"Error retrieving container entities: {str(e)}"
            self.logger.error(error_msg)
            return dict_to_json({"error": error_msg})
    
    def _get_container_by_name(self, container_name: str) -> Optional[Dict[str, Any]]:
        """
        Helper method to get a container by name.
        
        Args:
            container_name: Name of the container
            
        Returns:
            Container dictionary or None if not found
        """
        try:
            query = """
            MATCH (c:LessonContainer {name: $name})
            RETURN c
            """
            
            records, _ = self.base_manager.safe_execute_query(
                query,
                {"name": container_name}
            )
            
            if records and len(records) > 0:
                container = records[0].get("c")
                if container:
                    return dict(container.items())
            
            return None
                
        except Exception as e:
            self.logger.error(f"Error retrieving container by name: {str(e)}")
            return None 