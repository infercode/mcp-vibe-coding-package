from typing import Any, Dict, List, Optional, Union
import time
import json

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
        self.logger = base_manager.logger
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
            
            container_records, _ = self.base_manager.safe_execute_query(
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
            entity_result = self.entity_manager.create_entities([entity_dict])
            entity_json = json.loads(entity_result)
            
            if "error" in entity_json:
                return entity_result
            
            # Add entity to container
            relation_query = """
            MATCH (c:LessonContainer {name: $container_name})
            MATCH (e:Entity {name: $entity_name})
            MERGE (c)-[r:CONTAINS {added: datetime()}]->(e)
            RETURN r
            """
            
            self.base_manager.safe_execute_query(
                relation_query,
                {"container_name": container_name, "entity_name": entity_name}
            )
            
            # Get the created entity
            entity_query = """
            MATCH (e:Entity {name: $entity_name})
            RETURN e
            """
            
            entity_records, _ = self.base_manager.safe_execute_query(
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
                
                relation_records, _ = self.base_manager.safe_execute_query(
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
            
            container_records, _ = self.base_manager.safe_execute_query(
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
            
            # Update the entity
            update_result = self.entity_manager.update_entity(entity_name, updates)
            update_json = json.loads(update_result)
            
            if "error" in update_json:
                return update_result
            
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
            
            # If remove_from_container_only and container_name is provided
            if remove_from_container_only and container_name:
                relation_query = """
                MATCH (c:LessonContainer {name: $container_name})-[r:CONTAINS]->(e:Entity {name: $entity_name})
                DELETE r
                RETURN count(r) as deleted_count
                """
                
                records, _ = self.base_manager.safe_execute_query(
                    relation_query,
                    {"container_name": container_name, "entity_name": entity_name}
                )
                
                deleted_count = 0
                if records and len(records) > 0:
                    deleted_count = records[0].get("deleted_count", 0)
                
                if deleted_count > 0:
                    self.logger.info(f"Removed entity '{entity_name}' from container '{container_name}'")
                    return dict_to_json({
                        "status": "success",
                        "message": f"Entity '{entity_name}' removed from container '{container_name}'"
                    })
                
                return dict_to_json({
                    "error": f"Failed to remove entity '{entity_name}' from container '{container_name}'"
                })
            
            # Delete the entity completely
            delete_result = self.entity_manager.delete_entity(entity_name)
            delete_json = json.loads(delete_result)
            
            if "error" in delete_json:
                return delete_result
            
            self.logger.info(f"Deleted lesson entity: {entity_name}")
            return dict_to_json({
                "status": "success",
                "message": f"Lesson entity '{entity_name}' deleted"
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
            
            # Get existing tags
            entity = entity_json.get("entity", {})
            existing_tags = entity.get("tags", [])
            
            # Merge tags without duplicates
            if isinstance(existing_tags, str):
                existing_tags = [existing_tags]
            elif not isinstance(existing_tags, list):
                existing_tags = []
            
            merged_tags = list(set(existing_tags + tags))
            
            # Update entity with merged tags
            updates = {"tags": merged_tags}
            
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
        Search for lesson entities with various filters.
        
        Args:
            container_name: Optional container to search within
            search_term: Optional search term to match entity names
            entity_type: Optional entity type filter
            tags: Optional list of tags to filter by
            limit: Maximum number of results
            semantic: Whether to use semantic search
            
        Returns:
            JSON string with the matching entities
        """
        try:
            self.base_manager.ensure_initialized()
            
            # Base query parts
            query_parts = []
            params = {}
            
            # Container filter
            if container_name:
                query_parts.append("MATCH (c:LessonContainer {name: $container_name})-[:CONTAINS]->(e:Entity)")
                params["container_name"] = container_name
            else:
                query_parts.append("MATCH (e:Entity)")
            
            # Domain filter (always restrict to lesson domain)
            query_parts.append("WHERE e.domain = 'lesson'")
            
            # Name search
            if search_term:
                query_parts.append("AND e.name CONTAINS $search_term")
                params["search_term"] = search_term
            
            # Entity type filter
            if entity_type:
                query_parts.append("AND e.entityType = $entity_type")
                params["entity_type"] = entity_type
            
            # Tags filter
            if tags and len(tags) > 0:
                tags_conditions = []
                for i, tag in enumerate(tags):
                    tag_param = f"tag_{i}"
                    tags_conditions.append(f"$" + tag_param + " IN e.tags")
                    params[tag_param] = tag
                
                if tags_conditions:
                    query_parts.append("AND (" + " OR ".join(tags_conditions) + ")")
            
            # Complete query
            query_parts.append("RETURN e")
            query_parts.append(f"LIMIT {min(limit, 1000)}")  # Cap limit for safety
            
            query = "\n".join(query_parts)
            
            # If semantic search is requested and search_term is provided, handle differently
            if semantic and search_term and self.base_manager.embedding_enabled:
                return self._semantic_search_lesson_entities(
                    search_term, container_name, entity_type, tags, limit
                )
            
            # Execute standard query
            records, _ = self.base_manager.safe_execute_query(
                query,
                params
            )
            
            # Process results
            entities = []
            if records:
                for record in records:
                    entity = record.get("e")
                    if entity:
                        entity_dict = dict(entity.items())
                        entities.append(entity_dict)
            
            return dict_to_json({"entities": entities})
                
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
            search_term: Search term to match semantically
            container_name: Optional container to search within
            entity_type: Optional entity type filter
            tags: Optional list of tags to filter by
            limit: Maximum number of results
            
        Returns:
            JSON string with the matching entities
        """
        try:
            # Generate embedding for search term
            embedding = self.base_manager.generate_embedding(search_term)
            if not embedding:
                self.logger.error(f"Failed to generate embedding for search term: {search_term}")
                return dict_to_json({"error": "Failed to generate embedding for search term"})
            
            # Build vector search query with filters
            vector_query_parts = [
                "CALL db.index.vector.queryNodes('entity_embedding', $k, $embedding)"
            ]
            
            filter_conditions = ["node.domain = 'lesson'"]
            params = {
                "k": min(limit * 2, 1000),  # Get more results for filtering
                "embedding": embedding
            }
            
            # Entity type filter
            if entity_type:
                filter_conditions.append("node.entityType = $entity_type")
                params["entity_type"] = entity_type
            
            # Container filter
            if container_name:
                vector_query_parts.append("YIELD node, score")
                vector_query_parts.append("WITH node, score")
                vector_query_parts.append("MATCH (c:LessonContainer {name: $container_name})-[:CONTAINS]->(node)")
                params["container_name"] = container_name
            else:
                vector_query_parts.append("YIELD node, score")
            
            # Add WHERE clause with filters
            if filter_conditions:
                vector_query_parts.append("WHERE " + " AND ".join(filter_conditions))
            
            # Tags filter - add separately since they might need array contains logic
            if tags and len(tags) > 0:
                tags_conditions = []
                for i, tag in enumerate(tags):
                    tag_param = f"tag_{i}"
                    tags_conditions.append(f"$" + tag_param + " IN node.tags")
                    params[tag_param] = tag
                
                if tags_conditions:
                    if filter_conditions:
                        vector_query_parts.append("AND (" + " OR ".join(tags_conditions) + ")")
                    else:
                        vector_query_parts.append("WHERE " + " OR ".join(tags_conditions))
            
            # Complete the query
            vector_query_parts.append("RETURN node as e, score")
            vector_query_parts.append(f"LIMIT {min(limit, 100)}")
            
            vector_query = "\n".join(vector_query_parts)
            
            # Execute query
            records, _ = self.base_manager.safe_execute_query(
                vector_query,
                params
            )
            
            # Process results
            entities = []
            if records:
                for record in records:
                    entity = record.get("e")
                    score = record.get("score")
                    
                    if entity:
                        entity_dict = dict(entity.items())
                        entity_dict["similarity"] = score
                        entities.append(entity_dict)
            
            return dict_to_json({"entities": entities})
                
        except Exception as e:
            error_msg = f"Error in semantic search for lesson entities: {str(e)}"
            self.logger.error(error_msg)
            return dict_to_json({"error": error_msg}) 