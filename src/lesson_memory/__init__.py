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
from src.models.responses import SuccessResponse, create_error_response, parse_json_to_model, create_success_response
from pydantic import ValidationError

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
    def create_lesson_container(self, description: Optional[str] = None, metadata: Optional[Dict[str, Any]] = None) -> str:
        """
        Create a new lesson container. The container name is always 'Lessons' and there is only one instance of it.
        
        Args:
            description: Optional description of the container
            metadata: Optional metadata dictionary with additional properties
            
        Returns:
            JSON string with success/error status and container data
        """
        try:
            # Create container model for validation
            container_data = {
                "description": description,
                "metadata": metadata
            }
            container_model = LessonContainerCreate(**{k: v for k, v in container_data.items() if v is not None})
            
            # Create container using the container component
            return self.container.create_container(
                description=container_model.description,
                metadata=container_model.metadata
            )

        except ValidationError as ve:
            self.logger.error(f"Validation error creating lesson container: {str(ve)}")
            return json.dumps({
                "error": f"Invalid container data: {str(ve)}",
                "code": "container_validation_error"
            })
        except Exception as e:
            self.logger.error(f"Error creating lesson container: {str(e)}")
            return json.dumps({
                "error": f"Failed to create lesson container: {str(e)}",
                "code": "container_creation_error"
            })
    
    def get_lesson_container(self) -> str:
        """
        Retrieve a lesson container.
                
        Returns:
            JSON response with lesson container data
        """
        try:
            # Call the container component directly
            response_json = self.container.get_container()
            
            # Parse the JSON response to a SuccessResponse model
            response = parse_json_to_model(response_json, SuccessResponse)
            return json.dumps(response.model_dump(), default=str)
            
        except Exception as e:
            self.logger.error(f"Error retrieving lesson container: {str(e)}")
            error_response = create_error_response(
                message=f"Failed to retrieve lesson container: {str(e)}",
                code="container_retrieval_error"
            )
            try:
                return json.dumps(error_response.model_dump(), default=str)
            except TypeError:
                # If JSON serialization fails due to non-serializable objects, use str
                return json.dumps({
                    "error": f"Failed to retrieve lesson container: {str(e)}",
                    "code": "container_retrieval_error"
                })
    
    def update_lesson_container(self, name: str, updates: Dict[str, Any]) -> str:
        """
        Update a lesson container's properties.
        
        Args:
            name: Name of the container to update
            updates: Dictionary of fields to update
            
        Returns:
            JSON string with success/error status and updated container data
        """
        try:
            # Create update model for validation
            update_data = {
                "container_name": name,
                "updates": updates
            }
            update_model = LessonContainerUpdate(**update_data)
            
            # The container.update_container expects the updates
            response = self.container.update_container(update_model.updates)
            return response
        except ValidationError as ve:
            self.logger.error(f"Validation error updating lesson container: {str(ve)}")
            error_response = create_error_response(
                message=f"Invalid container update data: {str(ve)}",
                code="container_validation_error"
            )
            try:
                return json.dumps(error_response.model_dump(), default=str)
            except TypeError:
                # If JSON serialization fails due to non-serializable objects, use str
                return json.dumps({
                    "error": f"Invalid container update data: {str(ve)}",
                    "code": "container_validation_error"
                })
        except Exception as e:
            self.logger.error(f"Error updating lesson container: {str(e)}")
            error_response = create_error_response(
                message=f"Failed to update lesson container: {str(e)}",
                code="container_update_error"
            )
            # Convert to JSON string, handling datetime objects
            try:
                return json.dumps(error_response.model_dump(), default=str)
            except TypeError:
                # If JSON serialization fails due to non-serializable objects, use str
                return json.dumps({
                    "error": f"Failed to update lesson container: {str(e)}",
                    "code": "container_update_error"
                })
    
    def delete_lesson_container(self, delete_contents: bool = False) -> str:
        """
        Delete a lesson container and optionally its contents.
        
        Args:
            delete_contents: Whether to delete all contained entities
            
        Returns:
            JSON string with success/error status
        """
        try:
            response = self.container.delete_container(delete_contents)
            return response
        except Exception as e:
            self.logger.error(f"Error deleting lesson container: {str(e)}")
            error_response = create_error_response(
                message=f"Failed to delete lesson container: {str(e)}",
                code="container_deletion_error"
            )
            try:
                return json.dumps(error_response.model_dump(), default=str)
            except TypeError:
                # If JSON serialization fails due to non-serializable objects, use str
                return json.dumps({
                    "error": f"Failed to delete lesson container: {str(e)}",
                    "code": "container_deletion_error"
                })
    
    def list_lesson_containers(self, limit: int = 100, sort_by: str = "created") -> Dict[str, Any]:
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
            
            # Use the helper function from responses module
            success_response = create_success_response(
                message="Successfully retrieved lesson containers",
                data={"containers": containers}
            )
            return success_response.model_dump()
        except Exception as e:
            self.logger.error(f"Error listing containers: {str(e)}")
            return create_error_response(
                message=f"Failed to list containers: {str(e)}",
                code="container_list_error"
            ).model_dump()
    
    def add_entity_to_lesson_container(self, container_name: str, entity_name: str) -> Dict[str, Any]:
        """
        Add an entity to a lesson container.
        
        Args:
            container_name: Name of the container
            entity_name: Name of the entity to add
            
        Returns:
            Dictionary with success/error status
        """
        try:
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
                return create_error_response(
                    message=f"Container '{container_name}' not found",
                    code="container_not_found"
                ).model_dump()
                
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
                return create_error_response(
                    message=f"Entity '{entity_name}' not found",
                    code="entity_not_found"
                ).model_dump()
                
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
            
            return create_success_response(
                message=f"Entity '{entity_name}' added to container '{container_name}'",
                data={"entity_name": entity_name, "container_name": container_name}
            ).model_dump()
        except Exception as e:
            self.logger.error(f"Error adding entity to container: {str(e)}")
            return create_error_response(
                message=f"Failed to add entity to container: {str(e)}",
                code="container_add_entity_error"
            ).model_dump()
    
    def remove_entity_from_lesson_container(self, container_name: str, entity_name: str) -> Dict[str, Any]:
        """
        Remove an entity from a lesson container.
        
        Args:
            container_name: Name of the container
            entity_name: Name of the entity to remove
            
        Returns:
            Dictionary with success/error status
        """
        try:
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
            
            return create_success_response(
                message=f"Entity '{entity_name}' removed from container '{container_name}'",
                data={"entity_name": entity_name, "container_name": container_name}
            ).model_dump()
        except Exception as e:
            self.logger.error(f"Error removing entity from container: {str(e)}")
            return create_error_response(
                message=f"Failed to remove entity from container: {str(e)}",
                code="container_remove_entity_error"
            ).model_dump()
    
    def get_lesson_container_entities(self, container_name: str, 
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
                return create_error_response(
                    message=f"Container '{container_name}' not found",
                    code="container_not_found"
                ).model_dump()
                
            # Query for entities in the container
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
            
            return create_success_response(
                message=f"Retrieved {len(entities)} entities from container '{container_name}'",
                data={"container_name": container_name, "entities": entities}
            ).model_dump()
        except Exception as e:
            self.logger.error(f"Error getting container entities: {str(e)}")
            return create_error_response(
                message=f"Failed to get container entities: {str(e)}",
                code="container_entities_error"
            ).model_dump()
    
    # Entity Operations
    def create_lesson_entity(
            self,
            container_name: str,
            entity_name: str, entity_type: str,
            observations: Optional[List[str]] = None,
            metadata: Optional[Dict[str, Any]] = None
        ) -> Dict[str, Any]:
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
            # Create entity model for validation, only providing non-None values
            entity_data: Dict[str, Any] = {
                "container_name": container_name,
                "entity_name": entity_name,
                "entity_type": entity_type
            }
            
            if observations is not None:
                entity_data["observations"] = observations
            if metadata is not None:
                entity_data["metadata"] = metadata
            
            entity_model = LessonEntityCreate(**entity_data)
            
            # Create entity using the entity component
            response = self.entity.create_lesson_entity(
                entity_model.container_name,
                entity_model.entity_name,
                entity_model.entity_type,
                entity_model.observations,
                entity_model.metadata
            )
            
            return json.loads(response)
        except ValidationError as ve:
            self.logger.error(f"Validation error creating lesson entity: {str(ve)}")
            return create_error_response(
                message=f"Invalid entity data: {str(ve)}",
                code="entity_validation_error"
            ).model_dump()
        except Exception as e:
            self.logger.error(f"Error creating entity: {str(e)}")
            return create_error_response(
                message=f"Failed to create entity: {str(e)}",
                code="entity_creation_error"
            ).model_dump()
    
    # Continue with other entity methods in the same pattern...
    # Add similar patterns for relationship, observation methods, etc.
    
    # For example:
    def create_structured_lesson_observations(
            self, entity_name: str,
            what_was_learned: Optional[str] = None,
            why_it_matters: Optional[str] = None,
            how_to_apply: Optional[str] = None,
            root_cause: Optional[str] = None,
            evidence: Optional[str] = None,
            container_name: Optional[str] = None
        ) -> Dict[str, Any]:
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
            # Create structured observations model for validation
            observations_data = {
                "entity_name": entity_name,
                "container_name": container_name
            }
            
            if what_was_learned is not None:
                observations_data["what_was_learned"] = what_was_learned
            if why_it_matters is not None:
                observations_data["why_it_matters"] = why_it_matters
            if how_to_apply is not None:
                observations_data["how_to_apply"] = how_to_apply
            if root_cause is not None:
                observations_data["root_cause"] = root_cause
            if evidence is not None:
                observations_data["evidence"] = evidence
                
            observations_model = StructuredLessonObservations(**observations_data)
            
            # Create structured observations using the observation component
            response = self.observation.create_structured_lesson_observations(
                observations_model.entity_name,
                observations_model.what_was_learned,
                observations_model.why_it_matters,
                observations_model.how_to_apply,
                observations_model.root_cause,
                observations_model.evidence,
                observations_model.container_name
            )
            
            return json.loads(response)
        except ValidationError as ve:
            self.logger.error(f"Validation error creating structured observations: {str(ve)}")
            return create_error_response(
                message=f"Invalid structured observations data: {str(ve)}",
                code="observation_validation_error"
            ).model_dump()
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
            # Create search query model for validation
            search_data = {
                "search_term": search_term,
                "limit": limit,
                "semantic": semantic
            }
            if container_name:
                search_data["container_name"] = container_name
            if entity_type:
                search_data["entity_type"] = entity_type
            if tags:
                search_data["tags"] = tags
                
            search_model = SearchQuery(**search_data)
            
            # Direct implementation for semantic search
            query = """
            MATCH (e:Entity)
            WHERE e.domain = 'lesson'
            """
            
            params = {}
            
            if search_model.entity_type:
                query += " AND e.entityType = $entity_type"
                params["entity_type"] = search_model.entity_type
                
            if search_model.container_name:
                query += " WITH e MATCH (c:LessonContainer {name: $container_name})-[:CONTAINS]->(e)"
                params["container_name"] = search_model.container_name
            
            if search_model.search_term and not search_model.semantic:
                # Text-based search (fallback if semantic not available)
                query += " WHERE (e.name CONTAINS $search_term OR e.description CONTAINS $search_term)"
                params["search_term"] = search_model.search_term
            
            # Add tag filtering if provided
            if search_model.tags and len(search_model.tags) > 0:
                tag_conditions = []
                for i, tag in enumerate(search_model.tags):
                    param_name = f"tag_{i}"
                    tag_conditions.append(f"$tag_{i} IN e.tags")
                    params[param_name] = tag
                    
                if tag_conditions:
                    query += " AND (" + " OR ".join(tag_conditions) + ")"
            
            # Finalize query with limit
            query += " RETURN e LIMIT toInteger($limit)"
            params["limit"] = str(search_model.limit)  # Convert int to string for compatibility
            
            # Execute the query
            records = self.base_manager.safe_execute_read_query(query, params)
            
            # Process results
            entities = []
            for record in records:
                if 'e' in record:
                    entity = dict(record['e'].items())
                    entities.append(entity)
            
            search_description = f"'{search_model.search_term}'" if search_model.search_term else "all entities"
            if search_model.container_name:
                search_description += f" in container '{search_model.container_name}'"
            
            response = create_success_response(
                message=f"Found {len(entities)} entities matching {search_description}",
                data={
                    "entities": entities,
                    "query": search_data,
                    "count": len(entities)
                }
            )
            
            return json.dumps(response.model_dump(), default=str)
                
        except ValidationError as ve:
            self.logger.error(f"Validation error searching lesson entities: {str(ve)}")
            return json.dumps({
                "error": f"Invalid search parameters: {str(ve)}"
            })
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
            # Create update model for validation
            update_data: Dict[str, Any] = {
                "entity_name": entity_name,
                "updates": updates
            }
            
            if container_name is not None:
                update_data["container_name"] = container_name
                
            update_model = LessonEntityUpdate(**update_data)
            
            # Call the entity component to update the lesson
            return self.entity.update_lesson_entity(
                update_model.entity_name, 
                update_model.updates,
                update_model.container_name
            )
            
        except ValidationError as ve:
            self.logger.error(f"Validation error updating lesson entity: {str(ve)}")
            return json.dumps({
                "error": f"Invalid entity update data: {str(ve)}"
            })
        except Exception as e:
            self.logger.error(f"Error updating lesson entity: {str(e)}")
            return json.dumps({
                "error": f"Failed to update lesson entity: {str(e)}"
            })
    
    def track_lesson_application(
            self,
            lesson_name: str,
            context_entity: str,
            success_score: float = 0.8,
            application_notes: Optional[str] = None
        ) -> str:
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
    
    def merge_lessons(
            self,
            source_lessons: List[Dict[str, Any]],
            new_name: str,
            merge_strategy: str = "union",
            container_name: Optional[str] = None
        ) -> str:
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
    
    def get_lesson_knowledge_evolution(self, entity_name: Optional[str] = None,
                             lesson_type: Optional[str] = None,
                             start_date: Optional[str] = None,
                             end_date: Optional[str] = None,
                             include_superseded: bool = True) -> str:
        """
        Track the evolution of knowledge in the lesson graph.
        
        Args:
            entity_name: Optional entity name to filter by
            lesson_type: Optional lesson type to filter by
            start_date: Optional start date for time range (ISO format)
            end_date: Optional end date for time range (ISO format)
            include_superseded: Whether to include superseded lessons
            
        Returns:
            JSON string with evolution data
        """
        try:
            # Delegate to the evolution component
            response_json = self.evolution.get_knowledge_evolution(
                entity_name, lesson_type, start_date, end_date, include_superseded
            )
            
            # Parse and enhance the response
            response_data = json.loads(response_json)
            
            # If it's already an error response, just return it
            if "error" in response_data:
                return response_json
                
            # Create a standardized success response
            lesson_count = response_data.get("count", 0)
            entity_filter = f" for '{entity_name}'" if entity_name else ""
            
            return json.dumps(create_success_response(
                message=f"Retrieved evolution data for {lesson_count} lessons{entity_filter}",
                data=response_data
            ).model_dump(), default=str)
            
        except Exception as e:
            self.logger.error(f"Error getting knowledge evolution: {str(e)}")
            return json.dumps(create_error_response(
                message=f"Failed to get knowledge evolution: {str(e)}",
                code="knowledge_evolution_error"
            ).model_dump(), default=str)

    # Missing observation methods
    def add_lesson_observation(self, observation_data: Dict[str, Any]) -> str:
        """
        Add an observation to a lesson entity.
        
        Args:
            observation_data: Dictionary containing:
                - entity_name: Name of the entity to add observation to
                - content: Content of the observation
                - observation_type: Type of observation
                - container_name: Optional container to verify entity membership
                
        Returns:
            JSON string with the created observation
        """
        try:
            # Create observation model for validation
            observation_model = LessonObservationCreate(**observation_data)
            
            # Delegate to the observation component
            entity_name = observation_model.entity_name
            content = observation_model.content
            observation_type = observation_model.observation_type
            
            # Default confidence and properties since they aren't in the model
            confidence = observation_data.get("confidence", 1.0)
            properties = observation_data.get("metadata", {})
            
            return self.observation.add_lesson_observation(
                entity_name,
                content,
                observation_type,
                confidence,
                properties
            )
        except ValidationError as ve:
            self.logger.error(f"Validation error adding lesson observation: {str(ve)}")
            return json.dumps({
                "error": f"Invalid observation data: {str(ve)}"
            })
        except Exception as e:
            self.logger.error(f"Error adding lesson observation: {str(e)}")
            return json.dumps({
                "error": f"Failed to add lesson observation: {str(e)}"
            })
    
    def get_lesson_observations(self, entity_name: str, observation_type: Optional[str] = None) -> str:
        """
        Get observations for a lesson entity.
        
        Args:
            entity_name: Name of the entity
            observation_type: Optional type of observation to filter by
            container_name: Optional container to verify entity membership
            
        Returns:
            JSON string with the entity's observations
        """
        try:
            # Delegate to the observation component
            return self.observation.get_lesson_observations(
                entity_name,
                observation_type
            )
        except Exception as e:
            self.logger.error(f"Error getting lesson observations: {str(e)}")
            return json.dumps({
                "error": f"Failed to get lesson observations: {str(e)}"
            })
    
    def update_lesson_observation(
            self,
            observation_id: str,
            content: str, observation_type: Optional[str] = None
        ) -> str:
        """
        Update an observation for a lesson entity.
        
        Args:
            observation_id: ID of the observation to update
            content: New content for the observation
            observation_type: Optional new type for the observation
            
        Returns:
            JSON string with the updated observation
        """
        try:
            # Delegate to the observation component
            updates = {
                "content": content
            }
            if observation_type:
                updates["type"] = observation_type
                
            return self.observation.update_lesson_observation(
                observation_id,
                updates
            )
        except Exception as e:
            self.logger.error(f"Error updating lesson observation: {str(e)}")
            return json.dumps({
                "error": f"Failed to update lesson observation: {str(e)}"
            })
    
    def delete_lesson_observation(self, observation_id: str) -> str:
        """
        Delete an observation from a lesson entity.
        
        Args:
            entity_name: Name of the entity
            observation_id: ID of the observation to delete
            
        Returns:
            JSON string with operation result
        """
        try:
            # Delegate to the observation component
            return self.observation.delete_lesson_observation(
                observation_id
            )
        except Exception as e:
            self.logger.error(f"Error deleting lesson observation: {str(e)}")
            return json.dumps({
                "error": f"Failed to delete lesson observation: {str(e)}"
            })
    
    # Missing relation methods
    def create_lesson_relationship(self, relationship_data: Dict[str, Any]) -> str:
        """
        Create a relationship between lesson entities.
        
        Args:
            relationship_data: Dictionary containing:
                - source_name: Name of the source entity
                - target_name: Name of the target entity
                - relationship_type: Type of relationship
                - properties: Optional properties for the relationship
                
        Returns:
            JSON string with the created relationship
        """
        try:
            # Create relationship model for validation
            relationship_model = LessonRelationshipCreate(**relationship_data)
            
            # Get container name from the data or use default
            container_name = relationship_data.get("container_name", "Lessons")
            
            # Delegate to the relation component
            return self.relation.create_lesson_relation(
                container_name,
                relationship_model.source_name,
                relationship_model.target_name,
                relationship_model.relationship_type,
                relationship_model.properties or {}
            )
        except ValidationError as ve:
            self.logger.error(f"Validation error creating relationship: {str(ve)}")
            return json.dumps({
                "error": f"Invalid relationship data: {str(ve)}"
            })
        except Exception as e:
            self.logger.error(f"Error creating relationship: {str(e)}")
            return json.dumps({
                "error": f"Failed to create relationship: {str(e)}"
            })
    
    def get_lesson_relationships(
            self,
            entity_name: str,
            direction: str = "both",
            relationship_type: Optional[str] = None
        ) -> str:
        """
        Get relationships for a lesson entity.
        
        Args:
            entity_name: Name of the entity
            direction: Direction of relationships ('outgoing', 'incoming', or 'both')
            relationship_type: Optional type of relationship to filter by
            
        Returns:
            JSON string with the entity's relationships
        """
        try:
            # Delegate to the relation component
            return self.relation.get_lesson_relations(
                entity_name=entity_name,
                relation_type=relationship_type,
                direction=direction
            )
        except Exception as e:
            self.logger.error(f"Error getting relationships: {str(e)}")
            return json.dumps({
                "error": f"Failed to get relationships: {str(e)}"
            })
    
    def delete_lesson_relationship(
            self,
            source_name: str,
            target_name: str,
            relationship_type: str,
            container_name: str = "Lessons"
        ) -> str:
        """
        Delete a relationship between lesson entities.
        
        Args:
            source_name: Name of the source entity
            target_name: Name of the target entity
            relationship_type: Type of relationship to delete
            container_name: Container name
            
        Returns:
            JSON string with operation result
        """
        try:
            # Delegate to the relation component
            return self.relation.delete_lesson_relation(
                container_name,
                source_name,
                target_name,
                relationship_type
            )
        except Exception as e:
            self.logger.error(f"Error deleting relationship: {str(e)}")
            return json.dumps({
                "error": f"Failed to delete relationship: {str(e)}"
            })
    
    # Improved evolution methods
    def track_lesson_supersession(
            self,
            old_lesson: str,
            new_lesson: str, 
            reason: Optional[str] = None
        ) -> str:
        """
        Track when a new lesson supersedes an older one.
        
        Args:
            old_lesson: Name of the lesson being superseded
            new_lesson: Name of the new lesson
            reason: Optional reason for the supersession
            
        Returns:
            JSON string with operation result
        """
        try:
            # Use the specialized supersedes method if available
            if hasattr(self.relation, "create_supersedes_relation"):
                return self.relation.create_supersedes_relation(
                    new_lesson,
                    old_lesson
                )
            
            # Create relationship properties
            rel_props = {
                "reason": reason or "Knowledge updated",
                "superseded_at": datetime.datetime.now().isoformat()
            }
            
            # Create relationship using the relation component
            return self.create_lesson_relationship({
                "container_name": "Lessons",
                "source_name": new_lesson,
                "target_name": old_lesson,
                "relationship_type": "SUPERSEDES",
                "properties": rel_props
            })
        except Exception as e:
            self.logger.error(f"Error tracking lesson supersession: {str(e)}")
            return json.dumps({
                "error": f"Failed to track lesson supersession: {str(e)}"
            })
    
    # Enhanced search capabilities
    def search_lesson_semantic(self, query: str, limit: int = 10, 
                     container_name: Optional[str] = None) -> str:
        """
        Search for lessons using semantic similarity.
        
        Args:
            query: The search query text
            limit: Maximum number of results to return
            container_name: Optional container to search within
            
        Returns:
            JSON string with search results
        """
        try:
            # Create search query model for validation
            search_data = {
                "search_term": query,
                "limit": limit,
                "semantic": True
            }
            if container_name:
                search_data["container_name"] = container_name
                
            search_model = SearchQuery(**search_data)
            
            # Fallback to text-based search with semantic flag
            # (the underlying implementation may use semantic search if available)
            return self.search_lesson_entities(
                container_name=search_model.container_name,
                search_term=search_model.search_term,
                limit=search_model.limit,
                semantic=True
            )
        except ValidationError as ve:
            self.logger.error(f"Validation error performing semantic lesson search: {str(ve)}")
            return json.dumps({
                "error": f"Invalid search parameters: {str(ve)}"
            })
        except Exception as e:
            self.logger.error(f"Error performing semantic lesson search: {str(e)}")
            return json.dumps({
                "error": f"Failed to perform semantic lesson search: {str(e)}"
            })

    def check_lesson_observation_completeness(self, entity_name: str) -> str:
        """
        Check which structured observation types are present for a lesson entity.
        
        Args:
            entity_name: Name of the entity to check
            
        Returns:
            JSON string with completeness assessment including:
            - Which standard observation types are present
            - Completeness score
            - Missing observation types
        """
        try:
            # Delegate to the observation component's existing method
            response_json = self.observation.check_observation_completeness(entity_name)
            
            # Parse and enhance the response if needed
            response_data = json.loads(response_json)
            
            # If it's already an error response, just return it
            if "error" in response_data:
                return response_json
                
            # Create a standardized success response
            return json.dumps(create_success_response(
                message=f"Lesson observation completeness check for '{entity_name}'",
                data=response_data
            ).model_dump(), default=str)
            
        except Exception as e:
            self.logger.error(f"Error checking lesson observation completeness: {str(e)}")
            return json.dumps(create_error_response(
                message=f"Failed to check lesson observation completeness: {str(e)}",
                code="observation_completeness_error"
            ).model_dump(), default=str)

    # Consolidation Methods
    def identify_similar_lessons(self, min_similarity: float = 0.7, 
                              entity_type: Optional[str] = None,
                              max_results: int = 20) -> str:
        """
        Identify clusters of similar lessons based on semantic similarity.
        
        Args:
            min_similarity: Minimum similarity threshold (0.0-1.0)
            entity_type: Optional specific lesson type to filter by
            max_results: Maximum number of similarity pairs to return
            
        Returns:
            JSON string with similar lesson pairs
        """
        try:
            # Delegate to the consolidation component
            response_json = self.consolidation.identify_similar_lessons(
                min_similarity, entity_type, max_results
            )
            
            # Parse and enhance the response
            response_data = json.loads(response_json)
            
            # If it's already an error response, just return it
            if "error" in response_data:
                return response_json
                
            # Create a standardized success response
            return json.dumps(create_success_response(
                message=f"Found {response_data.get('similar_pair_count', 0)} similar lesson pairs",
                data=response_data
            ).model_dump(), default=str)
            
        except Exception as e:
            self.logger.error(f"Error identifying similar lessons: {str(e)}")
            return json.dumps(create_error_response(
                message=f"Failed to identify similar lessons: {str(e)}",
                code="similarity_analysis_error"
            ).model_dump(), default=str)
            
    def suggest_lesson_consolidations(self, threshold: float = 0.8, max_suggestions: int = 10) -> str:
        """
        Suggest lessons that could be consolidated based on similarity.
        
        Args:
            threshold: Similarity threshold for suggestions (0.0-1.0)
            max_suggestions: Maximum number of suggestions to return
            
        Returns:
            JSON string with consolidation suggestions
        """
        try:
            # Delegate to the consolidation component
            response_json = self.consolidation.suggest_consolidations(
                threshold, max_suggestions
            )
            
            # Parse and enhance the response
            response_data = json.loads(response_json)
            
            # If it's already an error response, just return it
            if "error" in response_data:
                return response_json
                
            # Create a standardized success response
            return json.dumps(create_success_response(
                message=f"Found {response_data.get('suggestion_count', 0)} consolidation suggestions",
                data=response_data
            ).model_dump(), default=str)
            
        except Exception as e:
            self.logger.error(f"Error suggesting lesson consolidations: {str(e)}")
            return json.dumps(create_error_response(
                message=f"Failed to suggest lesson consolidations: {str(e)}",
                code="consolidation_suggestion_error"
            ).model_dump(), default=str)
            
    def cleanup_superseded_lessons(self, older_than_days: int = 30, 
                                 min_confidence: float = 0.0,
                                 dry_run: bool = True) -> str:
        """
        Clean up lessons that have been superseded and are older than a given threshold.
        
        Args:
            older_than_days: Only include lessons older than this many days
            min_confidence: Only include lessons with confidence >= this value
            dry_run: If True, only report what would be done without making changes
            
        Returns:
            JSON string with cleanup details
        """
        try:
            # Delegate to the consolidation component
            response_json = self.consolidation.cleanup_superseded_lessons(
                older_than_days, min_confidence, dry_run
            )
            
            # Parse and enhance the response
            response_data = json.loads(response_json)
            
            # If it's already an error response, just return it
            if "error" in response_data:
                return response_json
                
            # Create a standardized success response
            action = "would be deleted" if dry_run else "deleted"
            count = response_data.get("count", 0) if dry_run else response_data.get("deleted_count", 0)
            
            return json.dumps(create_success_response(
                message=f"{count} superseded lessons {action}",
                data=response_data
            ).model_dump(), default=str)
            
        except Exception as e:
            self.logger.error(f"Error cleaning up superseded lessons: {str(e)}")
            return json.dumps(create_error_response(
                message=f"Failed to clean up superseded lessons: {str(e)}",
                code="cleanup_error"
            ).model_dump(), default=str)

    # Evolution Tracker Methods
    def get_lesson_confidence_evolution(self, entity_name: str) -> str:
        """
        Track how confidence in knowledge has changed over time.
        
        Args:
            entity_name: Name of the entity to track
            
        Returns:
            JSON string with confidence evolution data
        """
        try:
            # Delegate to the evolution component
            response_json = self.evolution.get_confidence_evolution(entity_name)
            
            # Parse and enhance the response
            response_data = json.loads(response_json)
            
            # If it's already an error response, just return it
            if "error" in response_data:
                return response_json
                
            # Create a standardized success response
            return json.dumps(create_success_response(
                message=f"Confidence evolution for '{entity_name}'",
                data=response_data
            ).model_dump(), default=str)
            
        except Exception as e:
            self.logger.error(f"Error retrieving lesson confidence evolution: {str(e)}")
            return json.dumps(create_error_response(
                message=f"Failed to retrieve lesson confidence evolution: {str(e)}",
                code="confidence_evolution_error"
            ).model_dump(), default=str)
    
    def get_lesson_application_impact(self, entity_name: str) -> str:
        """
        Analyze the impact of lesson application over time.
        
        Args:
            entity_name: Name of the entity to analyze
            
        Returns:
            JSON string with application impact data
        """
        try:
            # Delegate to the evolution component
            response_json = self.evolution.get_lesson_application_impact(entity_name)
            
            # Parse and enhance the response
            response_data = json.loads(response_json)
            
            # If it's already an error response, just return it
            if "error" in response_data:
                return response_json
                
            # Create a standardized success response
            application_count = response_data.get("application_count", 0)
            return json.dumps(create_success_response(
                message=f"Analyzed {application_count} applications of lesson '{entity_name}'",
                data=response_data
            ).model_dump(), default=str)
            
        except Exception as e:
            self.logger.error(f"Error analyzing lesson application impact: {str(e)}")
            return json.dumps(create_error_response(
                message=f"Failed to analyze lesson application impact: {str(e)}",
                code="application_impact_error"
            ).model_dump(), default=str)
    
    def get_lesson_learning_progression(self, entity_name: str, max_depth: int = 3) -> str:
        """
        Analyze the learning progression by tracking superseded versions.
        
        Args:
            entity_name: Name of the entity to analyze
            max_depth: Maximum depth of superseded relationships to traverse
            
        Returns:
            JSON string with learning progression data
        """
        try:
            # Delegate to the evolution component
            response_json = self.evolution.get_learning_progression(entity_name, max_depth)
            
            # Parse and enhance the response
            response_data = json.loads(response_json)
            
            # If it's already an error response, just return it
            if "error" in response_data:
                return response_json
                
            # Create a standardized success response
            version_count = response_data.get("version_count", 0)
            return json.dumps(create_success_response(
                message=f"Analyzed learning progression for '{entity_name}' with {version_count} versions",
                data=response_data
            ).model_dump(), default=str)
            
        except Exception as e:
            self.logger.error(f"Error analyzing lesson learning progression: {str(e)}")
            return json.dumps(create_error_response(
                message=f"Failed to analyze lesson learning progression: {str(e)}",
                code="learning_progression_error"
            ).model_dump(), default=str)

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
            # Delegate to the entity component
            response_json = self.entity.tag_lesson_entity(entity_name, tags, container_name)
            
            # Parse and enhance the response
            response_data = json.loads(response_json)
            
            # If it's already an error response, just return it
            if "error" in response_data:
                return response_json
                
            # Create a standardized success response
            tag_count = len(tags)
            container_info = f" in container '{container_name}'" if container_name else ""
            
            return json.dumps(create_success_response(
                message=f"Added {tag_count} tags to entity '{entity_name}'{container_info}",
                data=response_data
            ).model_dump(), default=str)
            
        except Exception as e:
            self.logger.error(f"Error tagging lesson entity: {str(e)}")
            return json.dumps(create_error_response(
                message=f"Failed to tag lesson entity: {str(e)}",
                code="tag_entity_error"
            ).model_dump(), default=str)

    def update_lesson_relation(self, container_name: str, from_entity: str, to_entity: str,
                           relation_type: str, updates: Dict[str, Any]) -> str:
        """
        Update properties of a relationship between lesson entities.
        
        Args:
            container_name: Name of the container
            from_entity: Name of the source entity
            to_entity: Name of the target entity
            relation_type: Type of the relationship
            updates: Dictionary of property updates
            
        Returns:
            JSON string with the updated relationship
        """
        try:
            # Delegate to the relation component
            response_json = self.relation.update_lesson_relation(
                container_name, from_entity, to_entity, relation_type, updates
            )
            
            # Parse and enhance the response
            response_data = json.loads(response_json)
            
            # If it's already an error response, just return it
            if "error" in response_data:
                return response_json
                
            # Create a standardized success response
            return json.dumps(create_success_response(
                message=f"Updated {relation_type} relation from '{from_entity}' to '{to_entity}'",
                data=response_data
            ).model_dump(), default=str)
            
        except Exception as e:
            self.logger.error(f"Error updating lesson relation: {str(e)}")
            return json.dumps(create_error_response(
                message=f"Failed to update lesson relation: {str(e)}",
                code="relation_update_error"
            ).model_dump(), default=str)
    
    def get_lesson_knowledge_graph(self, container_name: str, depth: int = 2) -> str:
        """
        Get a complete knowledge graph of all entities and relationships in a lesson container.
        
        Args:
            container_name: Name of the container
            depth: Maximum relationship depth to include (1-3)
            
        Returns:
            JSON string with nodes and relationships comprising the knowledge graph
        """
        try:
            # Validate depth parameter
            if depth < 1:
                depth = 1
            elif depth > 3:
                depth = 3  # Cap at 3 for performance reasons
                
            # Delegate to the relation component
            response_json = self.relation.get_lesson_knowledge_graph(container_name, depth)
            
            # Parse and enhance the response
            response_data = json.loads(response_json)
            
            # If it's already an error response, just return it
            if "error" in response_data:
                return response_json
                
            # Get counts for summary
            node_count = len(response_data.get("nodes", []))
            relationship_count = len(response_data.get("relationships", []))
            
            # Create a standardized success response
            return json.dumps(create_success_response(
                message=f"Retrieved knowledge graph with {node_count} nodes and {relationship_count} relationships",
                data=response_data
            ).model_dump(), default=str)
            
        except Exception as e:
            self.logger.error(f"Error retrieving lesson knowledge graph: {str(e)}")
            return json.dumps(create_error_response(
                message=f"Failed to retrieve lesson knowledge graph: {str(e)}",
                code="knowledge_graph_error"
            ).model_dump(), default=str)
