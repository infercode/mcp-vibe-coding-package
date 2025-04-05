"""
Lesson Memory System for MCP Graph Memory Architecture.

This module provides structured memory management for lessons,
enabling the system to store, retrieve, update, and manage
lesson-related knowledge.
"""

from typing import Any, Dict, List, Optional, Union
import json

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
            response = self.container.list_containers(limit, sort_by)
            return json.loads(response)
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
            response = self.container.add_entity_to_container(container_name, entity_name)
            return json.loads(response)
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
            response = self.container.remove_entity_from_container(container_name, entity_name)
            return json.loads(response)
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
            response = self.container.get_container_entities(container_name, entity_type, limit)
            return json.loads(response)
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
