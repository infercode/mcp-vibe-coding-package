"""
Lesson Memory System for MCP Graph Memory Architecture.

This module provides structured memory management for lessons,
enabling the system to store, retrieve, update, and manage
lesson-related knowledge.
"""

from typing import Any, Dict, List, Optional, Union

from src.graph_memory.base_manager import BaseManager
from src.lesson_memory.lesson_container import LessonContainer
from src.lesson_memory.lesson_entity import LessonEntity
from src.lesson_memory.lesson_relation import LessonRelation
from src.lesson_memory.lesson_observation import LessonObservation
from src.lesson_memory.evolution_tracker import EvolutionTracker
from src.lesson_memory.consolidation import LessonConsolidation

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
                      metadata: Optional[Dict[str, Any]] = None) -> str:
        """Create a new lesson container."""
        return self.container.create_container(name, description, metadata)
    
    def get_container(self, name: str) -> str:
        """Retrieve a lesson container by name."""
        return self.container.get_container(name)
    
    def update_container(self, name: str, updates: Dict[str, Any]) -> str:
        """Update a lesson container's properties."""
        return self.container.update_container(name, updates)
    
    def delete_container(self, name: str, delete_contents: bool = False) -> str:
        """Delete a lesson container and optionally its contents."""
        return self.container.delete_container(name, delete_contents)
    
    def list_containers(self, sort_by: str = "name", limit: int = 100) -> str:
        """List all lesson containers."""
        return self.container.list_containers(sort_by, limit)
    
    def add_entity_to_container(self, container_name: str, entity_name: str) -> str:
        """Add an entity to a lesson container."""
        return self.container.add_entity_to_container(container_name, entity_name)
    
    def remove_entity_from_container(self, container_name: str, entity_name: str) -> str:
        """Remove an entity from a lesson container."""
        return self.container.remove_entity_from_container(container_name, entity_name)
    
    def get_container_entities(self, container_name: str, 
                            entity_type: Optional[str] = None) -> str:
        """Get all entities in a container."""
        return self.container.get_container_entities(container_name, entity_type)
    
    # Lesson Entity Operations
    def create_lesson_entity(self, name: str, content: str, 
                          entity_type: str = "Lesson",
                          container_name: Optional[str] = None,
                          context: Optional[str] = None,
                          confidence: float = 0.5,
                          metadata: Optional[Dict[str, Any]] = None,
                          observations: Optional[Dict[str, Any]] = None) -> str:
        """Create a new lesson entity."""
        return self.entity.create_lesson_entity(name, content, entity_type,
                                            container_name, context, confidence,
                                            metadata, observations)
    
    def get_lesson_entity(self, entity_name: str, 
                       container_name: Optional[str] = None) -> str:
        """Retrieve a lesson entity."""
        return self.entity.get_lesson_entity(entity_name, container_name)
    
    def update_lesson_entity(self, entity_name: str, 
                          updates: Dict[str, Any],
                          container_name: Optional[str] = None) -> str:
        """Update a lesson entity."""
        return self.entity.update_lesson_entity(entity_name, updates, container_name)
    
    def delete_lesson_entity(self, entity_name: str, 
                          container_name: Optional[str] = None,
                          only_remove_from_container: bool = False) -> str:
        """Delete a lesson entity or remove it from a container."""
        return self.entity.delete_lesson_entity(entity_name, container_name, 
                                            only_remove_from_container)
    
    def tag_lesson_entity(self, entity_name: str, tags: List[str],
                       container_name: Optional[str] = None) -> str:
        """Add tags to a lesson entity."""
        return self.entity.tag_lesson_entity(entity_name, tags, container_name)
    
    def search_lesson_entities(self, 
                            container_name: Optional[str] = None,
                            search_term: Optional[str] = None,
                            entity_type: Optional[str] = None,
                            tags: Optional[List[str]] = None,
                            use_semantic_search: bool = False) -> str:
        """Search for lesson entities."""
        return self.entity.search_lesson_entities(container_name, search_term,
                                              entity_type, tags, use_semantic_search)
    
    # Lesson Relation Operations
    def create_lesson_relation(self, from_entity: str, to_entity: str, 
                            relation_type: str,
                            container_name: Optional[str] = None,
                            properties: Optional[Dict[str, Any]] = None) -> str:
        """Create a relationship between lesson entities."""
        return self.relation.create_lesson_relation(from_entity, to_entity,
                                                relation_type, container_name, properties)
    
    def get_lesson_relations(self, entity_name: Optional[str] = None,
                          relation_type: Optional[str] = None,
                          container_name: Optional[str] = None,
                          direction: str = "both") -> str:
        """Get lesson relationships."""
        return self.relation.get_lesson_relations(entity_name, relation_type,
                                               container_name, direction)
    
    def update_lesson_relation(self, from_entity: str, to_entity: str,
                            relation_type: str,
                            updates: Dict[str, Any]) -> str:
        """Update a lesson relationship."""
        return self.relation.update_lesson_relation(from_entity, to_entity,
                                                relation_type, updates)
    
    def delete_lesson_relation(self, from_entity: str, to_entity: str,
                            relation_type: str) -> str:
        """Delete a lesson relationship."""
        return self.relation.delete_lesson_relation(from_entity, to_entity, relation_type)
    
    def get_lesson_knowledge_graph(self, container_name: str, 
                                depth: int = 2) -> str:
        """Get a knowledge graph for a lesson container."""
        return self.relation.get_lesson_knowledge_graph(container_name, depth)
    
    def create_supersedes_relation(self, new_version: str, 
                                old_version: str) -> str:
        """Create a supersedes relationship between lesson versions."""
        return self.relation.create_supersedes_relation(new_version, old_version)
    
    def track_lesson_application(self, lesson_name: str, target_context: str,
                              success_score: float,
                              application_notes: Optional[str] = None) -> str:
        """Track the application of a lesson to a specific context."""
        return self.relation.track_lesson_application(lesson_name, target_context,
                                                  success_score, application_notes)
    
    # Lesson Observation Operations
    def add_lesson_observation(self, entity_name: str, 
                            observation_type: str,
                            content: str,
                            container_name: Optional[str] = None) -> str:
        """Add a structured observation to a lesson entity."""
        return self.observation.add_lesson_observation(entity_name, observation_type,
                                                   content, container_name)
    
    def get_lesson_observations(self, entity_name: str,
                             observation_type: Optional[str] = None,
                             container_name: Optional[str] = None) -> str:
        """Get observations for a lesson entity."""
        return self.observation.get_lesson_observations(entity_name, observation_type, container_name)
    
    def update_lesson_observation(self, entity_name: str,
                               observation_id: str,
                               content: str,
                               observation_type: Optional[str] = None) -> str:
        """Update a lesson observation."""
        return self.observation.update_lesson_observation(entity_name, observation_id,
                                                       content, observation_type)
    
    def delete_lesson_observation(self, entity_name: str,
                               observation_id: str) -> str:
        """Delete a lesson observation."""
        return self.observation.delete_lesson_observation(entity_name, observation_id)
    
    def create_structured_lesson_observations(self, entity_name: str,
                                           observations: Dict[str, str],
                                           container_name: Optional[str] = None) -> str:
        """Create all structured observations for a lesson entity at once."""
        return self.observation.create_structured_lesson_observations(entity_name, observations, container_name)
    
    def check_observation_completeness(self, entity_name: str) -> str:
        """Check which structured observation types are present for a lesson entity."""
        return self.observation.check_observation_completeness(entity_name)
    
    # Evolution Tracking Operations
    def get_knowledge_evolution(self, entity_name: Optional[str] = None,
                             lesson_type: Optional[str] = None,
                             start_date: Optional[Union[str, float]] = None,
                             end_date: Optional[Union[str, float]] = None,
                             include_superseded: bool = True) -> str:
        """Track how knowledge has evolved over time."""
        return self.evolution.get_knowledge_evolution(entity_name, lesson_type,
                                                 start_date, end_date, include_superseded)
    
    def get_confidence_evolution(self, entity_name: str) -> str:
        """Track how confidence has evolved for a lesson over time."""
        return self.evolution.get_confidence_evolution(entity_name)
    
    def get_lesson_application_impact(self, entity_name: str) -> str:
        """Analyze the impact of lesson applications on success metrics."""
        return self.evolution.get_lesson_application_impact(entity_name)
    
    def get_learning_progression(self, entity_name: str, max_depth: int = 3) -> str:
        """Analyze the learning progression path for a lesson."""
        return self.evolution.get_learning_progression(entity_name, max_depth)
    
    # Consolidation Operations
    def identify_similar_lessons(self, min_similarity: float = 0.7,
                              entity_type: Optional[str] = None,
                              max_results: int = 20) -> str:
        """Identify clusters of similar lessons based on semantic similarity."""
        return self.consolidation.identify_similar_lessons(min_similarity, entity_type, max_results)
    
    def merge_lessons(self, source_lessons: List[Dict[str, Any]],
                   new_name: str,
                   merge_strategy: str = "latest",
                   container_name: Optional[str] = None) -> str:
        """Merge multiple related lessons into a consolidated lesson."""
        return self.consolidation.merge_lessons(source_lessons, new_name, 
                                             merge_strategy, container_name)
    
    def suggest_consolidations(self, threshold: float = 0.8, 
                            max_suggestions: int = 10) -> str:
        """Suggest groups of lessons that could be consolidated based on similarity."""
        return self.consolidation.suggest_consolidations(threshold, max_suggestions)
    
    def cleanup_superseded_lessons(self, older_than_days: int = 30,
                                min_confidence: float = 0.0,
                                dry_run: bool = True) -> str:
        """Archive superseded lessons that are no longer needed."""
        return self.consolidation.cleanup_superseded_lessons(older_than_days, 
                                                         min_confidence, dry_run)
