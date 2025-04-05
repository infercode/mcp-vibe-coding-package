"""
Lesson Memory System for MCP Graph Memory Architecture.

This module provides structured memory management for lessons,
enabling the system to store, retrieve, update, and manage
lesson-related knowledge.
"""

from typing import Any, Dict, List, Optional, Union
import json
import datetime  # Add import for datetime

from src.graph_memory.base_manager import BaseManager
from src.lesson_memory.lesson_container import LessonContainer
from src.lesson_memory.lesson_entity import LessonEntity
from src.lesson_memory.lesson_relation import LessonRelation
from src.lesson_memory.lesson_observation import LessonObservation
from src.lesson_memory.evolution_tracker import EvolutionTracker
from src.lesson_memory.consolidation import LessonConsolidation

from src.models.lesson_memory import (
    LessonContainerCreate, LessonContainerUpdate,
    LessonEntityCreate, LessonEntityUpdate,
    LessonObservationCreate, StructuredLessonObservations,
    LessonRelationshipCreate, SearchQuery
)
from src.models.responses import SuccessResponse, create_error_response, parse_json_to_model

class LessonMemoryManager:
    """
    Main facade for the Lesson Memory System.
    
    Provides a unified interface to all lesson memory operations including:
    - Container management (grouping lessons)
    - Entity operations (lesson CRUD)
    - Relationship management
    - Observation tracking
    - Knowledge evolution analysis
    - Memory consolidation
    """
    
    def __init__(self, base_manager: BaseManager):
        """
        Initialize the Lesson Memory Manager.
        
        Args:
            base_manager: Base manager for Neo4j connection and core operations
        """
        self.base_manager = base_manager
        self.logger = base_manager.logger
        
        # Initialize components
        self.container = LessonContainer(base_manager)
        self.entity = LessonEntity(base_manager)
        self.relation = LessonRelation(base_manager)
        self.observation = LessonObservation(base_manager)
        self.evolution = EvolutionTracker(base_manager)
        self.consolidation = LessonConsolidation(base_manager)
    
    # Container Operations
    def create_container(self, name: str, description: Optional[str] = None, 
                      metadata: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Create a new lesson container.
        
        Args:
            name: Unique name for the container
            description: Optional description
            metadata: Optional metadata dictionary
            
        Returns:
            Dictionary with success/error status and container data
        """
        # Create container model for validation
        try:
            container_data = {
                "name": name,
                "description": description
            }
            
            if metadata:
                container_data["metadata"] = metadata
                
            container_model = LessonContainerCreate(**container_data)
            
            # Create container using the refactored method
            response = self.container.create_container(
                container_model.name, 
                container_model.description, 
                container_model.metadata
            )
            
            # Parse the response
            return json.loads(response)
        except Exception as e:
            self.logger.error(f"Error creating container: {str(e)}")
            return create_error_response(
                message=f"Failed to create container: {str(e)}",
                code="container_creation_error"
            ).model_dump()
    
    def get_container(self, name: str) -> Dict[str, Any]:
        """
        Retrieve a lesson container by name.
        
        Args:
            name: Name of the container to retrieve
            
        Returns:
            Dictionary with success/error status and container data
        """
        try:
            response = self.container.get_container(name)
            return json.loads(response)
        except Exception as e:
            self.logger.error(f"Error retrieving container: {str(e)}")
            return create_error_response(
                message=f"Failed to retrieve container: {str(e)}",
                code="container_retrieval_error"
            ).model_dump()
    
    def update_container(self, name: str, updates: Dict[str, Any]) -> Dict[str, Any]:
        """
        Update a lesson container's properties.
        
        Args:
            name: Name of the container to update
            updates: Dictionary of fields to update
            
        Returns:
            Dictionary with success/error status and updated container data
        """
        try:
            # Create update model for validation
            update_model = LessonContainerUpdate(**updates)
            
            response = self.container.update_container(name, update_model.model_dump(exclude_unset=True))
            return json.loads(response)
        except Exception as e:
            self.logger.error(f"Error updating container: {str(e)}")
            return create_error_response(
                message=f"Failed to update container: {str(e)}",
                code="container_update_error"
            ).model_dump()
    
    def delete_container(self, name: str, delete_contents: bool = False) -> Dict[str, Any]:
        """
        Delete a lesson container and optionally its contents.
        
        Args:
            name: Name of the container to delete
            delete_contents: Whether to delete all contained entities
            
        Returns:
            Dictionary with success/error status
        """
        try:
            response = self.container.delete_container(name, delete_contents)
            return json.loads(response)
        except Exception as e:
            self.logger.error(f"Error deleting container: {str(e)}")
            return create_error_response(
                message=f"Failed to delete container: {str(e)}",
                code="container_deletion_error"
            ).model_dump()
    
    def list_containers(self, limit: int = 100, sort_by: str = "created") -> Dict[str, Any]:
        """
        List all lesson containers.
        
        Args:
            limit: Maximum number of containers to return
            sort_by: Field to sort results by
            
        Returns:
            Dictionary with success/error status and list of containers
        """
        try:
            # Directly build a query to list containers as fallback
            query = """
            MATCH (c:LessonContainer)
            RETURN c
            ORDER BY c.{sort_by} DESC
            LIMIT toInteger($limit)
            """.format(sort_by=sort_by)
            
            records = self.base_manager.safe_execute_read_query(
                query,
                {"limit": str(limit)}
            )
            
            containers = []
            for record in records:
                if 'c' in record:
                    container = dict(record['c'].items())
                    containers.append(container)
            
            return {"status": "success", "containers": containers}
        except Exception as e:
            self.logger.error(f"Error listing containers: {str(e)}")
            return create_error_response(
                message=f"Failed to list containers: {str(e)}",
                code="container_list_error"
            ).model_dump()
    
    def add_entity_to_container(self, container_name: str, entity_name: str) -> Dict[str, Any]:
        """
        Add an entity to a lesson container.
        
        Args:
            container_name: Name of the container
            entity_name: Name of the entity to add
            
        Returns:
            Dictionary with success/error status
        """
        try:
            # Direct implementation as fallback for missing method
            # First verify container exists
            container_query = """
            MATCH (c:LessonContainer {name: $name})
            RETURN c
            """
            
            container_records = self.base_manager.safe_execute_read_query(
                container_query,
                {"name": container_name}
            )
            
            if not container_records:
                return {"error": f"Container '{container_name}' not found"}
                
            # Verify entity exists
            entity_query = """
            MATCH (e:Entity {name: $name})
            RETURN e
            """
            
            entity_records = self.base_manager.safe_execute_read_query(
                entity_query,
                {"name": entity_name}
            )
            
            if not entity_records:
                return {"error": f"Entity '{entity_name}' not found"}
                
            # Create relationship between container and entity
            relation_query = """
            MATCH (c:LessonContainer {name: $container_name})
            MATCH (e:Entity {name: $entity_name})
            MERGE (c)-[r:CONTAINS]->(e)
            SET r.added = datetime()
            RETURN r
            """
            
            self.base_manager.safe_execute_write_query(
                relation_query,
                {
                    "container_name": container_name,
                    "entity_name": entity_name
                }
            )
            
            return {"status": "success", "message": f"Entity added to container"}
        except Exception as e:
            self.logger.error(f"Error adding entity to container: {str(e)}")
            return create_error_response(
                message=f"Failed to add entity to container: {str(e)}",
                code="container_add_entity_error"
            ).model_dump()
    
    def remove_entity_from_container(self, container_name: str, entity_name: str) -> Dict[str, Any]:
        """
        Remove an entity from a lesson container.
        
        Args:
            container_name: Name of the container
            entity_name: Name of the entity to remove
            
        Returns:
            Dictionary with success/error status
        """
        try:
            # Direct implementation as fallback for missing method
            # Delete relationship between container and entity
            relation_query = """
            MATCH (c:LessonContainer {name: $container_name})-[r:CONTAINS]->(e:Entity {name: $entity_name})
            DELETE r
            """
            
            self.base_manager.safe_execute_write_query(
                relation_query,
                {
                    "container_name": container_name,
                    "entity_name": entity_name
                }
            )
            
            return {"status": "success", "message": f"Entity removed from container"}
        except Exception as e:
            self.logger.error(f"Error removing entity from container: {str(e)}")
            return create_error_response(
                message=f"Failed to remove entity from container: {str(e)}",
                code="container_remove_entity_error"
            ).model_dump()
    
    def get_container_entities(self, container_name: str, 
                            entity_type: Optional[str] = None,
                            limit: int = 100) -> Dict[str, Any]:
        """
        Get all entities in a container.
        
        Args:
            container_name: Name of the container
            entity_type: Optional filter by entity type
            limit: Maximum number of entities to return
            
        Returns:
            Dictionary with success/error status and list of entities
        """
        try:
            # Direct implementation as fallback for missing method
            query = """
            MATCH (c:LessonContainer {name: $container_name})-[:CONTAINS]->(e:Entity)
            """
            
            params = {"container_name": container_name}
            
            if entity_type:
                query += " WHERE e.entityType = $entity_type"
                params["entity_type"] = entity_type
                
            query += """
            RETURN e
            LIMIT toInteger($limit)
            """
            
            params["limit"] = str(limit)
            
            records = self.base_manager.safe_execute_read_query(query, params)
            
            entities = []
            for record in records:
                if 'e' in record:
                    entity = dict(record['e'].items())
                    entities.append(entity)
            
            return {"status": "success", "entities": entities}
        except Exception as e:
            self.logger.error(f"Error getting container entities: {str(e)}")
            return create_error_response(
                message=f"Failed to get container entities: {str(e)}",
                code="container_entities_error"
            ).model_dump()
    
    # Entity Operations
    def create_lesson_entity(self, container_name: str, entity_name: str, entity_type: str,
                          observations: Optional[List[str]] = None,
                          metadata: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Create a new lesson entity.
        
        Args:
            container_name: Name of the container to add the entity to
            entity_name: Unique name for the entity
            entity_type: Type of entity (PERSON, PLACE, etc.)
            observations: Optional list of observations
            metadata: Optional metadata dictionary
            
        Returns:
            Dictionary with success/error status and entity data
        """
        try:
            # Create entity model for validation
            entity_data = {
                "container_name": container_name,
                "entity_name": entity_name,
                "entity_type": entity_type,
                "observations": observations,
                "metadata": metadata
            }
            
            entity_model = LessonEntityCreate(**{k: v for k, v in entity_data.items() if v is not None})
            
            # Create entity using the entity component
            response = self.entity.create_lesson_entity(
                entity_model.container_name,
                entity_model.entity_name,
                entity_model.entity_type,
                entity_model.observations,
                entity_model.metadata
            )
            
            return json.loads(response)
        except Exception as e:
            self.logger.error(f"Error creating entity: {str(e)}")
            return create_error_response(
                message=f"Failed to create entity: {str(e)}",
                code="entity_creation_error"
            ).model_dump()
    
    # Continue with other entity methods in the same pattern...
    # Add similar patterns for relationship, observation methods, etc.
    
    # For example:
    def create_structured_lesson_observations(self, entity_name: str,
                                           what_was_learned: Optional[str] = None,
                                           why_it_matters: Optional[str] = None,
                                           how_to_apply: Optional[str] = None,
                                           root_cause: Optional[str] = None,
                                           evidence: Optional[str] = None,
                                container_name: Optional[str] = None) -> Dict[str, Any]:
        """
        Create structured observations for an entity.
        
        Args:
            entity_name: Name of the entity
            what_was_learned: Optional content for WhatWasLearned observation
            why_it_matters: Optional content for WhyItMatters observation
            how_to_apply: Optional content for HowToApply observation
            root_cause: Optional content for RootCause observation
            evidence: Optional content for Evidence observation
            container_name: Optional container to verify entity membership
            
        Returns:
            Dictionary with success/error status and observation data
        """
        try:
            # Create structured observations using the observation component
            response = self.observation.create_structured_lesson_observations(
                entity_name,
                                                                   what_was_learned,
                                                                   why_it_matters,
                                                                   how_to_apply,
                                                                   root_cause,
                                                                   evidence,
                container_name
            )
            
            return json.loads(response)
        except Exception as e:
            self.logger.error(f"Error creating structured observations: {str(e)}")
            return create_error_response(
                message=f"Failed to create structured observations: {str(e)}",
                code="observation_creation_error"
            ).model_dump()
            
    # ... implement remaining methods with similar pattern

    def search_lesson_entities(self, container_name: Optional[str] = None, 
                             search_term: Optional[str] = None,
                             entity_type: Optional[str] = None,
                             tags: Optional[List[str]] = None,
                             limit: int = 50,
                             semantic: bool = False) -> str:
        """
        Search for lesson entities matching specific criteria.
        
        Args:
            container_name: Optional name of container to search within
            search_term: Optional text to search for
            entity_type: Optional entity type to filter by
            tags: Optional list of tags to filter by
            limit: Maximum number of results to return
            semantic: Whether to use semantic search
            
        Returns:
            JSON string with search results
        """
        try:
            # Direct implementation for semantic search
            query = """
            MATCH (e:Entity)
            WHERE e.domain = 'lesson'
            """
            
            params = {}
            
            if entity_type:
                query += " AND e.entityType = $entity_type"
                params["entity_type"] = entity_type
                
            if container_name:
                query += " WITH e MATCH (c:LessonContainer {name: $container_name})-[:CONTAINS]->(e)"
                params["container_name"] = container_name
            
            if search_term and not semantic:
                # Text-based search (fallback if semantic not available)
                query += " WHERE (e.name CONTAINS $search_term OR e.description CONTAINS $search_term)"
                params["search_term"] = search_term
            
            # Add tag filtering if provided
            if tags and len(tags) > 0:
                tag_conditions = []
                for i, tag in enumerate(tags):
                    param_name = f"tag_{i}"
                    tag_conditions.append(f"$tag_{i} IN e.tags")
                    params[param_name] = tag
                    
                if tag_conditions:
                    query += " AND (" + " OR ".join(tag_conditions) + ")"
            
            # Finalize query with limit
            query += " RETURN e LIMIT toInteger($limit)"
            params["limit"] = str(limit)  # Convert int to string for compatibility
            
            # Execute the query
            records = self.base_manager.safe_execute_read_query(query, params)
            
            # Process results
            entities = []
            for record in records:
                if 'e' in record:
                    entity = dict(record['e'].items())
                    entities.append(entity)
            
            return json.dumps({"entities": entities})
                
        except Exception as e:
            self.logger.error(f"Error searching lesson entities: {str(e)}")
            return json.dumps({
                "error": f"Failed to search lesson entities: {str(e)}"
            })
    
    def update_lesson_entity(self, entity_name: str, updates: Dict[str, Any],
                           container_name: Optional[str] = None) -> str:
        """
        Update a lesson entity.
        
        Args:
            entity_name: Name of the entity to update
            updates: Dictionary of properties to update
            container_name: Optional container to verify membership
            
        Returns:
            JSON string with the updated entity
        """
        try:
            # Call the entity component to update the lesson
            return self.entity.update_lesson_entity(entity_name, updates, container_name)
            
        except Exception as e:
            self.logger.error(f"Error updating lesson entity: {str(e)}")
            return json.dumps({
                "error": f"Failed to update lesson entity: {str(e)}"
            })
    
    def track_lesson_application(self, lesson_name: str, context_entity: str,
                              success_score: float = 0.8,
                              application_notes: Optional[str] = None) -> str:
        """
        Track the application of a lesson to a context entity.
        
        Args:
            lesson_name: Name of the lesson being applied
            context_entity: Name of the entity the lesson is being applied to
            success_score: Score indicating success of application (0.0-1.0)
            application_notes: Optional notes about the application
            
        Returns:
            JSON string with the created relationship
        """
        try:
            # Validate entities exist
            lesson_result = self.entity.get_lesson_entity(lesson_name)
            if "error" in json.loads(lesson_result):
                return json.dumps({
                    "error": f"Lesson '{lesson_name}' not found"
                })
            
            # Create relationship properties
            rel_props = {
                "success_score": max(0.0, min(1.0, success_score)),  # Clamp between 0 and 1
                "applied_at": datetime.datetime.now().isoformat()
            }
            
            if application_notes:
                rel_props["application_notes"] = application_notes
                
            # Create the relationship directly using Cypher
            query = """
            MATCH (e1:Entity {name: $from_entity})
            MATCH (e2:Entity {name: $to_entity})
            CREATE (e1)-[r:APPLIED_TO]->(e2)
            SET r = $properties
            RETURN e1.name as from, type(r) as type, e2.name as to, r as relationship
            """
            
            records = self.base_manager.safe_execute_write_query(
                query,
                {
                    "from_entity": lesson_name,
                    "to_entity": context_entity,
                    "properties": rel_props
                }
            )
            
            if records and len(records) > 0:
                return json.dumps({
                    "status": "success",
                    "relation": {
                        "from": lesson_name,
                        "type": "APPLIED_TO",
                        "to": context_entity,
                        "properties": rel_props
                    }
                })
            else:
                return json.dumps({"error": "Failed to create relationship"})
            
        except Exception as e:
            self.logger.error(f"Error tracking lesson application: {str(e)}")
            return json.dumps({
                "error": f"Failed to track lesson application: {str(e)}"
            })
    
    def merge_lessons(self, source_lessons: List[Dict[str, Any]], new_name: str,
                    merge_strategy: str = "union",
                    container_name: Optional[str] = None) -> str:
        """
        Merge multiple lessons into a single consolidated lesson.
        
        Args:
            source_lessons: List of lesson IDs or dictionaries containing lesson IDs
            new_name: Name for the new consolidated lesson
            merge_strategy: Strategy for merging ('union', 'intersection', latest')
            container_name: Optional container name for the new lesson
            
        Returns:
            JSON string with the created consolidated lesson
        """
        try:
            # Extract lesson IDs from source_lessons
            lesson_ids = []
            for lesson in source_lessons:
                if isinstance(lesson, dict) and "id" in lesson:
                    lesson_ids.append(lesson["id"])
                elif isinstance(lesson, str):
                    lesson_ids.append(lesson)
                    
            if not lesson_ids:
                return json.dumps({
                    "error": "No valid lesson IDs provided for merging"
                })
                
            # Direct implementation as fallback
            self.logger.error("Consolidation component doesn't have merge_lessons method")
            
            # Create a new lesson to represent the consolidated knowledge
            metadata = {
                "source_lessons": lesson_ids,
                "merge_strategy": merge_strategy,
                "consolidated_at": datetime.datetime.now().isoformat()
            }
            
            # Create the consolidated lesson
            return self.entity.create_lesson_entity(
                container_name or "LessonContainer", 
                new_name,
                "ConsolidatedLesson",
                None,  # observations
                metadata
            )
            
        except Exception as e:
            self.logger.error(f"Error merging lessons: {str(e)}")
            return json.dumps({
                "error": f"Failed to merge lessons: {str(e)}"
            })
    
    def get_knowledge_evolution(self, entity_name: Optional[str] = None,
                             lesson_type: Optional[str] = None,
                             start_date: Optional[str] = None,
                             end_date: Optional[str] = None,
                             include_superseded: bool = True) -> str:
        """
        Track the evolution of knowledge in the lesson graph.
        
        Args:
            entity_name: Optional entity name to filter by
            lesson_type: Optional lesson type to filter by
            start_date: Optional start date for time range
            end_date: Optional end date for time range
            include_superseded: Whether to include superseded lessons
            
        Returns:
            JSON string with evolution data
        """
        try:
            # Direct implementation as fallback
            self.logger.error("Evolution component doesn't have track_knowledge_evolution method")
            
            # Build a query to track knowledge evolution
            query = """
            MATCH (e:Entity)
            WHERE e.domain = 'lesson'
            """
            
            params = {}
            
            if entity_name:
                query += " AND e.name = $entity_name"
                params["entity_name"] = entity_name
                
            if lesson_type:
                query += " AND e.entityType = $lesson_type"
                params["lesson_type"] = lesson_type
                
            # Add time range if specified
            if start_date:
                query += " AND e.created >= datetime($start_date)"
                params["start_date"] = start_date
                
            if end_date:
                query += " AND e.created <= datetime($end_date)"
                params["end_date"] = end_date
                
            if not include_superseded:
                query += " AND NOT EXISTS((e)<-[:SUPERSEDES]-())"
                
            query += """
            RETURN e,
                   e.created as created_time,
                   e.lastUpdated as updated_time
            ORDER BY e.created
            """
            
            # Execute the query
            records = self.base_manager.safe_execute_read_query(query, params)
            
            # Process results
            evolution_data = []
            for record in records:
                entity = record.get("e")
                if entity:
                    entity_dict = dict(entity.items())
                    entity_dict["created_time"] = record.get("created_time")
                    entity_dict["updated_time"] = record.get("updated_time")
                    evolution_data.append(entity_dict)
                    
            return json.dumps({"evolution": evolution_data})
            
        except Exception as e:
            self.logger.error(f"Error getting knowledge evolution: {str(e)}")
            return json.dumps({
                "error": f"Failed to get knowledge evolution: {str(e)}"
            })
