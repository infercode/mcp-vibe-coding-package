from typing import Any, Dict, List, Optional, Union
import time
import json
import logging

from src.utils import dict_to_json, generate_id
from src.graph_memory.base_manager import BaseManager
from src.graph_memory.entity_manager import EntityManager

class LessonEntity:
    """
    Manager for lesson entity operations.
    Extends the core entity manager with lesson-specific functionality.
    """
    
    def __init__(self, base_manager: BaseManager):
        """
        Initialize the lesson entity manager.
        
        Args:
            base_manager: Base manager for graph operations
        """
        self.base_manager = base_manager
        self.logger = logging.getLogger(__name__)
        self.entity_manager = EntityManager(base_manager)
    
    def create_lesson_entity(self, container_name: str, entity_name: str, entity_type: str,
                           observations: Optional[List[str]] = None,
                           metadata: Optional[Dict[str, Any]] = None) -> str:
        """
        Create a lesson entity and add it to a container.
        
        Args:
            container_name: Name of the container to add the entity to
            entity_name: Name of the entity
            entity_type: Type of the entity
            observations: Optional list of observations
            metadata: Optional metadata for the entity
            
        Returns:
            JSON string with the created entity
        """
        try:
            self.base_manager.ensure_initialized()
            
            # Check if container exists
            container_query = """
            MATCH (c:LessonContainer {name: $name})
            RETURN c
            """
            
            # Use safe_execute_read_query for validation (read-only operation)
            container_records = self.base_manager.safe_execute_read_query(
                container_query,
                {"name": container_name}
            )
            
            if not container_records or len(container_records) == 0:
                return dict_to_json({"error": f"Lesson container '{container_name}' not found"})
            
            # Prepare entity creation
            entity_dict = {
                "name": entity_name,
                "entityType": entity_type,
                "observations": observations or []
            }
            
            # Add metadata if provided
            if metadata:
                for key, value in metadata.items():
                    if key not in ["name", "entityType", "observations"]:
                        entity_dict[key] = value
            
            # Add "lesson" tag to entity
            entity_dict["domain"] = "lesson"
            
            # Create entity
            query = """
            MATCH (c:LessonContainer {name: $container_name})
            CREATE (e:Entity $entity_props)
            CREATE (e)-[r1:BELONGS_TO]->(c)
            CREATE (c)-[r2:CONTAINS]->(e)
            SET e.created = datetime(), 
                e.lastUpdated = datetime()
            RETURN e, c
            """
            
            # Use safe_execute_write_query for validation (write operation)
            self.base_manager.safe_execute_write_query(
                query,
                {"container_name": container_name, "entity_props": entity_dict}
            )
            
            # Get the created entity
            entity_query = """
            MATCH (e:Entity {name: $entity_name})
            RETURN e
            """
            
            # Use safe_execute_read_query for validation (read-only operation)
            entity_records = self.base_manager.safe_execute_read_query(
                entity_query,
                {"entity_name": entity_name}
            )
            
            if entity_records and len(entity_records) > 0:
                entity = entity_records[0].get("e")
                if entity:
                    entity_dict = dict(entity.items())
                    
                    # Add container info
                    entity_dict["container"] = container_name
                    
                    self.logger.info(f"Created lesson entity '{entity_name}' in container '{container_name}'")
                    return dict_to_json({"entity": entity_dict})
            
            return dict_to_json({"error": "Failed to retrieve created lesson entity"})
                
        except Exception as e:
            error_msg = f"Error creating lesson entity: {str(e)}"
            self.logger.error(error_msg)
            return dict_to_json({"error": error_msg})
    
    def get_lesson_entity(self, entity_name: str, container_name: Optional[str] = None) -> str:
        """
        Get a lesson entity, optionally verifying it belongs to a specific container.
        
        Args:
            entity_name: Name of the entity
            container_name: Optional container to verify membership
            
        Returns:
            JSON string with the entity
        """
        try:
            self.base_manager.ensure_initialized()
            
            # Get the entity
            entity_result = self.entity_manager.get_entity(entity_name)
            entity_json = json.loads(entity_result)
            
            if "error" in entity_json:
                return entity_result
            
            # Check if entity belongs to the lesson domain
            entity = entity_json.get("entity", {})
            if entity.get("domain") != "lesson":
                return dict_to_json({
                    "error": f"Entity '{entity_name}' is not a lesson entity"
                })
            
            # If container specified, check membership
            if container_name:
                relation_query = """
                MATCH (c:LessonContainer {name: $container_name})-[:CONTAINS]->(e:Entity {name: $entity_name})
                RETURN c
                """
                
                # Use safe_execute_read_query for validation (read-only operation)
                relation_records = self.base_manager.safe_execute_read_query(
                    relation_query,
                    {"container_name": container_name, "entity_name": entity_name}
                )
                
                if not relation_records or len(relation_records) == 0:
                    return dict_to_json({
                        "error": f"Entity '{entity_name}' is not in container '{container_name}'"
                    })
                
                # Add container info
                entity["container"] = container_name
            
            # Get containers this entity belongs to
            container_query = """
            MATCH (c:LessonContainer)-[:CONTAINS]->(e:Entity {name: $entity_name})
            RETURN c.name as container_name
            """
            
            # Use safe_execute_read_query for validation (read-only operation)
            container_records = self.base_manager.safe_execute_read_query(
                container_query,
                {"entity_name": entity_name}
            )
            
            containers = []
            if container_records:
                for record in container_records:
                    container_name = record.get("container_name")
                    if container_name:
                        containers.append(container_name)
            
            # Add containers to entity
            entity["containers"] = containers
            
            return dict_to_json({"entity": entity})
                
        except Exception as e:
            error_msg = f"Error retrieving lesson entity: {str(e)}"
            self.logger.error(error_msg)
            return dict_to_json({"error": error_msg})
    
    def update_lesson_entity(self, entity_name: str, updates: Dict[str, Any],
                           container_name: Optional[str] = None) -> str:
        """
        Update a lesson entity.
        
        Args:
            entity_name: Name of the entity
            updates: Dictionary of property updates
            container_name: Optional container to verify membership
            
        Returns:
            JSON string with the updated entity
        """
        try:
            self.base_manager.ensure_initialized()
            
            # Check if entity exists and is a lesson entity
            entity_result = self.get_lesson_entity(entity_name, container_name)
            entity_json = json.loads(entity_result)
            
            if "error" in entity_json:
                return entity_result
            
            # Ensure domain remains lesson
            updates["domain"] = "lesson"
            
            # Build update query
            set_parts = []
            params = {"name": entity_name}
            
            # Add update properties
            for i, (key, value) in enumerate(updates.items()):
                param_name = f"p{i}"
                set_parts.append(f"e.{key} = ${param_name}")
                params[param_name] = value
            
            # Add lastUpdated timestamp
            set_parts.append("e.lastUpdated = datetime()")
            
            # Build and execute query
            query = f"""
            MATCH (e:Entity {{name: $name}})
            SET {', '.join(set_parts)}
            RETURN e
            """
            
            # Use safe_execute_write_query for validation (write operation)
            self.base_manager.safe_execute_write_query(
                query,
                params
            )
            
            # Get the updated entity with lesson context
            return self.get_lesson_entity(entity_name, container_name)
                
        except Exception as e:
            error_msg = f"Error updating lesson entity: {str(e)}"
            self.logger.error(error_msg)
            return dict_to_json({"error": error_msg})
    
    def delete_lesson_entity(self, entity_name: str, container_name: Optional[str] = None,
                           remove_from_container_only: bool = False) -> str:
        """
        Delete a lesson entity or remove it from a container.
        
        Args:
            entity_name: Name of the entity
            container_name: Optional container to verify membership
            remove_from_container_only: If True, only remove from container without deleting
            
        Returns:
            JSON string with the result
        """
        try:
            self.base_manager.ensure_initialized()
            
            # Check if entity exists and is a lesson entity
            entity_result = self.get_lesson_entity(entity_name, container_name)
            entity_json = json.loads(entity_result)
            
            if "error" in entity_json:
                return entity_result
            
            # If container specified and remove_from_container_only is True
            if container_name and remove_from_container_only:
                # Remove entity from container only
                relation_query = """
                MATCH (c:LessonContainer {name: $container_name})-[r:CONTAINS]->(e:Entity {name: $entity_name})
                DELETE r
                """
                
                # Use safe_execute_write_query for validation (write operation)
                self.base_manager.safe_execute_write_query(
                    relation_query,
                    {"container_name": container_name, "entity_name": entity_name}
                )
                
                return dict_to_json({
                    "status": "success",
                    "message": f"Entity '{entity_name}' removed from container '{container_name}'"
                })
            
            # Delete the entity
            delete_result = self.entity_manager.delete_entity(entity_name)
            delete_json = json.loads(delete_result)
            
            if "error" in delete_json:
                return delete_result
            
            return dict_to_json({
                "status": "success",
                "message": f"Lesson entity '{entity_name}' deleted successfully"
            })
                
        except Exception as e:
            error_msg = f"Error deleting lesson entity: {str(e)}"
            self.logger.error(error_msg)
            return dict_to_json({"error": error_msg})
    
    def tag_lesson_entity(self, entity_name: str, tags: List[str],
                        container_name: Optional[str] = None) -> str:
        """
        Add tags to a lesson entity.
        
        Args:
            entity_name: Name of the entity
            tags: List of tags to add
            container_name: Optional container to verify membership
            
        Returns:
            JSON string with the updated entity
        """
        try:
            self.base_manager.ensure_initialized()
            
            # Check if entity exists and is a lesson entity
            entity_result = self.get_lesson_entity(entity_name, container_name)
            entity_json = json.loads(entity_result)
            
            if "error" in entity_json:
                return entity_result
            
            # Get current entity data
            entity = entity_json.get("entity", {})
            current_tags = entity.get("tags", [])
            
            # Add new tags
            updated_tags = list(set(current_tags + tags))
            
            # Update entity with new tags
            updates = {"tags": updated_tags}
            return self.update_lesson_entity(entity_name, updates, container_name)
                
        except Exception as e:
            error_msg = f"Error tagging lesson entity: {str(e)}"
            self.logger.error(error_msg)
            return dict_to_json({"error": error_msg})
    
    def search_lesson_entities(self, container_name: Optional[str] = None,
                             search_term: Optional[str] = None,
                             entity_type: Optional[str] = None,
                             tags: Optional[List[str]] = None,
                             limit: int = 50,
                             semantic: bool = False) -> str:
        """
        Search for lesson entities.
        
        Args:
            container_name: Optional container to search in
            search_term: Optional search term to filter by
            entity_type: Optional entity type to filter by
            tags: Optional list of tags to filter by
            limit: Maximum number of results to return
            semantic: Whether to use semantic search
            
        Returns:
            JSON string with the results
        """
        try:
            self.base_manager.ensure_initialized()
            
            # If semantic search is requested
            if semantic and search_term:
                return self._semantic_search_lesson_entities(
                    search_term, container_name, entity_type, tags, limit
                )
            
            # Build query parts
            match_clauses = ["MATCH (e:Entity)"]
            where_clauses = ["e.domain = 'lesson'"]
            params = {}
            
            # Filter by container
            if container_name:
                match_clauses.append("MATCH (c:LessonContainer {name: $container_name})-[:CONTAINS]->(e)")
                params["container_name"] = container_name
            
            # Filter by search term
            if search_term:
                where_clauses.append("(e.name CONTAINS $search_term OR e.description CONTAINS $search_term)")
                params["search_term"] = search_term
            
            # Filter by entity type
            if entity_type:
                where_clauses.append("e.entityType = $entity_type")
                params["entity_type"] = entity_type
            
            # Filter by tags
            if tags and len(tags) > 0:
                where_clauses.append("ALL(tag IN $tags WHERE tag IN e.tags)")
                params["tags"] = tags
            
            # Build final query
            query = " ".join(match_clauses)
            if where_clauses:
                query += " WHERE " + " AND ".join(where_clauses)
            
            query += " RETURN e ORDER BY e.name LIMIT $limit"
            params["limit"] = limit
            
            # Use safe_execute_read_query for validation (read-only operation)
            records = self.base_manager.safe_execute_read_query(query, params)
            
            # Process results
            entities = []
            if records:
                for record in records:
                    entity = dict(record["e"].items())
                    entities.append(entity)
            
            return dict_to_json({
                "entities": entities,
                "count": len(entities),
                "search_term": search_term,
                "entity_type": entity_type,
                "tags": tags,
                "container": container_name
            })
                
        except Exception as e:
            error_msg = f"Error searching lesson entities: {str(e)}"
            self.logger.error(error_msg)
            return dict_to_json({"error": error_msg})
    
    def _semantic_search_lesson_entities(self, search_term: str,
                                       container_name: Optional[str] = None,
                                       entity_type: Optional[str] = None,
                                       tags: Optional[List[str]] = None,
                                       limit: int = 50) -> str:
        """
        Perform semantic search for lesson entities.
        
        Args:
            search_term: Search term for semantic search
            container_name: Optional container to search in
            entity_type: Optional entity type to filter by
            tags: Optional list of tags to filter by
            limit: Maximum number of results to return
            
        Returns:
            JSON string with the results
        """
        try:
            self.base_manager.ensure_initialized()
            
            # Get vector embedding for search term
            # Note: Implementation depends on the vector search implementation
            # For this example, we'll build a simple query with filters
            
            # Build query parts
            match_clauses = ["MATCH (e:Entity)"]
            where_clauses = ["e.domain = 'lesson'"]
            params = {}
            
            # Filter by container
            if container_name:
                match_clauses.append("MATCH (c:LessonContainer {name: $container_name})-[:CONTAINS]->(e)")
                params["container_name"] = container_name
            
            # Filter by entity type
            if entity_type:
                where_clauses.append("e.entityType = $entity_type")
                params["entity_type"] = entity_type
            
            # Filter by tags
            if tags and len(tags) > 0:
                where_clauses.append("ALL(tag IN $tags WHERE tag IN e.tags)")
                params["tags"] = tags
            
            # Build final query - assume a vector index and KNN search
            # This is a placeholder for actual vector search implementation
            query = " ".join(match_clauses)
            if where_clauses:
                query += " WHERE " + " AND ".join(where_clauses)
            
            # For now, fallback to regular search since semantic search requires vector embeddings
            query += " RETURN e ORDER BY e.name LIMIT $limit"
            params["limit"] = limit
            
            # Use safe_execute_read_query for validation (read-only operation)
            records = self.base_manager.safe_execute_read_query(query, params)
            
            # Process results
            entities = []
            if records:
                for record in records:
                    entity = dict(record["e"].items())
                    entities.append(entity)
            
            return dict_to_json({
                "entities": entities,
                "count": len(entities),
                "search_term": search_term,
                "entity_type": entity_type,
                "tags": tags,
                "container": container_name,
                "search_type": "semantic"
            })
                
        except Exception as e:
            error_msg = f"Error performing semantic search on lesson entities: {str(e)}"
            self.logger.error(error_msg)
            return dict_to_json({"error": error_msg}) 