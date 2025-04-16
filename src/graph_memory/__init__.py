"""
Graph Memory Module

This module provides functionality for interacting with the knowledge graph.
It exposes a set of managers for different graph memory operations, as well
as a facade class that maintains backward compatibility with the original API.
"""

from typing import Any, Dict, List, Optional, Union, cast
import json
import time
import os
import logging
import sys
import uuid
import re
from contextlib import contextmanager
from datetime import datetime
from pydantic import BaseModel, Field

from src.graph_memory.base_manager import BaseManager
from src.graph_memory.entity_manager import EntityManager
from src.graph_memory.relation_manager import RelationManager
from src.graph_memory.observation_manager import ObservationManager
from src.graph_memory.search_manager import SearchManager
from src.graph_memory.embedding_adapter import EmbeddingAdapter
from src.utils import dict_to_json

# Import specialized memory managers
from src.lesson_memory import LessonMemoryManager
from src.project_memory import ProjectMemoryManager

# Import error and success models
from src.models.response_models import ErrorDetail, ErrorResponse, SuccessResponse

__all__ = [
    'BaseManager',
    'EntityManager',
    'RelationManager',
    'ObservationManager',
    'SearchManager',
    'EmbeddingAdapter',
    'GraphMemoryManager'
]

class LessonContext:
    """
    Context manager for batch lesson memory operations.
    
    Provides a simplified interface for creating and manipulating
    lesson memory entities with a shared context.
    """
    
    def __init__(self, lesson_memory, container_name=None):
        """
        Initialize a lesson context.
        
        Args:
            lesson_memory: LessonMemory manager instance
            container_name: Optional container name to use for operations
        """
        from typing import Optional, Any
        from src.models.response_models import LessonContextModel
        
        self.lesson_memory = lesson_memory
        self.container_name = container_name
        self.logger = lesson_memory.logger if hasattr(lesson_memory, "logger") else None
        # Storage for the Pydantic context model
        self._context_model: Optional[LessonContextModel] = None
    
    def create_container(self, **kwargs) -> str:
        """
        Create a new lesson container.
        
        Args:
            **kwargs: Arguments for container creation:
                - description: Optional description
                - metadata: Optional metadata dictionary
                
        Returns:
            JSON response with created container
        """
        try:
            return self.lesson_memory.lesson_operation(
                operation_type="create_container",
                **kwargs
            )
        except Exception as e:
            if self.logger:
                self.logger.error(f"Error creating container in lesson context: {str(e)}")
            return json.dumps({
                "status": "error",
                "error": f"Failed to create container: {str(e)}",
                "code": "context_container_creation_error"
            })
    
    def get_container(self) -> str:
        """
        Get the lesson container.
        
        Returns:
            JSON response with container data
        """
        try:
            return self.lesson_memory.lesson_operation(
                operation_type="get_container"
            )
        except Exception as e:
            if self.logger:
                self.logger.error(f"Error getting container in lesson context: {str(e)}")
            return json.dumps({
                "status": "error",
                "error": f"Failed to get container: {str(e)}",
                "code": "context_container_get_error"
            })
    
    def list_containers(self, limit: int = 100, sort_by: str = "created") -> str:
        """
        List all lesson containers.
        
        Args:
            limit: Maximum number of containers to return
            sort_by: Field to sort results by
            
        Returns:
            JSON response with list of containers
        """
        try:
            return self.lesson_memory.lesson_operation(
                operation_type="list_containers",
                limit=limit,
                sort_by=sort_by
            )
        except Exception as e:
            if self.logger:
                self.logger.error(f"Error listing containers in lesson context: {str(e)}")
            return json.dumps({
                "status": "error",
                "error": f"Failed to list containers: {str(e)}",
                "code": "context_container_list_error"
            })
    
    def container_exists(self, container_name: str = "Lessons") -> str:
        """
        Check if a lesson container exists.
        
        Args:
            container_name: Name of the container to check
            
        Returns:
            JSON response with existence status
        """
        try:
            return self.lesson_memory.lesson_operation(
                operation_type="container_exists",
                container_name=container_name
            )
        except Exception as e:
            if self.logger:
                self.logger.error(f"Error checking container existence in lesson context: {str(e)}")
            return json.dumps({
                "status": "error",
                "error": f"Failed to check container existence: {str(e)}",
                "code": "context_container_check_error"
            })
    
    def create(self, name: str, lesson_type: str, **kwargs) -> str:
        """
        Create a lesson within this context.
        
        Args:
            name: Name of the lesson
            lesson_type: Type of the lesson
            **kwargs: Additional parameters
        
        Returns:
            JSON response with created lesson
        """
        # Merge container_name into kwargs if not explicitly provided
        if "container_name" not in kwargs:
            kwargs["container_name"] = self.container_name
            
        observations = kwargs.pop("observations", None)
        metadata = kwargs.pop("metadata", None)
            
        result = self.lesson_memory.create_lesson_entity(
            kwargs["container_name"], name, lesson_type, observations, metadata
        )
        
        # Ensure string return
        if isinstance(result, str):
            return result
        else:
            return json.dumps(result)
    
    def observe(self, entity_name: str, **kwargs) -> str:
        """
        Add observations to a lesson within this context.
        
        Args:
            entity_name: Name of the entity to add observations to
            **kwargs: Observation fields (what_was_learned, why_it_matters, etc.)
        
        Returns:
            JSON response with observation results
        """
        # Merge context information into kwargs
        kwargs["entity_name"] = entity_name
        if "container_name" not in kwargs:
            kwargs["container_name"] = self.container_name
            
        result = self.lesson_memory.create_structured_lesson_observations(**kwargs)
        
        # Ensure string return
        if isinstance(result, str):
            return result
        else:
            return json.dumps(result)
    
    def relate(self, source_name: str, target_name: str, relationship_type: str, **kwargs) -> str:
        """
        Create a relationship between lessons within this context.
        
        Args:
            source_name: Name of the source entity
            target_name: Name of the target entity
            relationship_type: Type of relationship
            **kwargs: Additional parameters
        
        Returns:
            JSON response with relationship data
        """
        # Build relationship data
        relationship_data = {
            "source_name": source_name,
            "target_name": target_name,
            "relationship_type": relationship_type,
            "container_name": self.container_name
        }
        
        # Add any additional properties
        if "properties" in kwargs:
            relationship_data["properties"] = kwargs["properties"]
            
        result = self.lesson_memory.create_lesson_relationship(relationship_data)
        
        # Ensure string return
        if isinstance(result, str):
            return result
        else:
            return json.dumps(result)
    
    def search(self, query: Optional[str] = None, **kwargs) -> str:
        """
        Search for lessons within this context.
        
        Args:
            query: Search query text
            **kwargs: Additional parameters
        
        Returns:
            JSON response with search results
        """
        # Set container context if not explicitly provided
        if "container_name" not in kwargs:
            kwargs["container_name"] = self.container_name
            
        # Set defaults
        kwargs.setdefault("limit", 50)
        kwargs.setdefault("semantic", True)
        
        if query is not None:
            kwargs["search_term"] = query
            
        # Use semantic search if enabled
        if kwargs.get("semantic", False):
            try:
                search_term = kwargs.get("search_term", "")
                if search_term is None:
                    search_term = ""
                    
                result = self.lesson_memory.search_lesson_semantic(
                    query=search_term,
                    limit=kwargs.get("limit", 50),
                    container_name=kwargs.get("container_name")
                )
                
                # Ensure string return
                if isinstance(result, str):
                    return result
                else:
                    return json.dumps(result)
            except Exception:
                # Fall back to standard search
                pass
                
        # Use standard search
        result = self.lesson_memory.search_lesson_entities(
            container_name=kwargs.get("container_name"),
            search_term=kwargs.get("search_term"),
            entity_type=kwargs.get("entity_type"),
            tags=kwargs.get("tags"),
            limit=kwargs.get("limit", 50),
            semantic=False
        )
        
        # Ensure string return
        if isinstance(result, str):
            return result
        else:
            return json.dumps(result)
    
    def track(self, lesson_name: str, context_entity: str, **kwargs) -> str:
        """
        Track application of a lesson to a context entity.
        
        Args:
            lesson_name: Name of the lesson
            context_entity: Name of the target entity
            **kwargs: Additional parameters
        
        Returns:
            JSON response with tracking results
        """
        success_score = kwargs.get("success_score", 0.8)
        application_notes = kwargs.get("application_notes")
        
        result = self.lesson_memory.track_lesson_application(
            lesson_name=lesson_name,
            context_entity=context_entity,
            success_score=success_score,
            application_notes=application_notes
        )
        
        # Ensure string return
        if isinstance(result, str):
            return result
        else:
            return json.dumps(result)
    
    def consolidate(self, source_lessons: List[str], new_name: str, **kwargs) -> str:
        """
        Consolidate multiple lessons into a single lesson.
        
        Args:
            source_lessons: List of lesson names or IDs to consolidate
            new_name: Name for the new consolidated lesson
            **kwargs: Additional parameters
                - merge_strategy: Strategy for merging ('union', 'intersection', 'latest')
        
        Returns:
            JSON response with consolidated lesson
        """
        # Set container context
        kwargs.setdefault("container_name", self.container_name)
        merge_strategy = kwargs.get("merge_strategy", "union")
        
        # Convert source_lessons to proper format if needed
        if isinstance(source_lessons, str):
            source_lessons = [source_lessons]
                
        processed_sources = []
        for lesson in source_lessons:
            if isinstance(lesson, dict):
                processed_sources.append(lesson)
            else:
                processed_sources.append({"id": lesson})
        
        # Merge the lessons
        result = self.lesson_memory.merge_lessons(
            source_lessons=processed_sources,
            new_name=new_name,
            merge_strategy=merge_strategy,
            container_name=kwargs.get("container_name")
        )
        
        # Ensure string return
        if isinstance(result, str):
            return result
        else:
            return json.dumps(result)
    
    def evolve(self, old_lesson: str, new_lesson: str, **kwargs) -> str:
        """
        Track evolution of lesson knowledge when a new lesson supersedes an older one.
        
        Args:
            old_lesson: Name of the lesson being superseded
            new_lesson: Name of the new lesson
            **kwargs: Additional parameters
                - reason: Optional reason for the evolution
        
        Returns:
            JSON response with relationship data
        """
        reason = kwargs.get("reason")
        
        # Track the supersession
        result = self.lesson_memory.track_lesson_supersession(
            old_lesson=old_lesson,
            new_lesson=new_lesson,
            reason=reason
        )
        
        # Ensure string return
        if isinstance(result, str):
            return result
        else:
            return json.dumps(result)
    
    def update(self, entity_name: str, updates: Dict[str, Any]) -> str:
        """
        Update a lesson entity within this context.
        
        Args:
            entity_name: Name of the entity to update
            updates: Dictionary of fields to update
        
        Returns:
            JSON response with updated entity
        """
        result = self.lesson_memory.update_lesson_entity(
            entity_name=entity_name,
            updates=updates,
            container_name=self.container_name
        )
        
        # Ensure string return
        if isinstance(result, str):
            return result
        else:
            return json.dumps(result)

class ProjectContext:
    """
    Context manager for batch project memory operations.
    """
    
    def __init__(self, project_memory, project_name=None):
        """
        Initialize a project context.
        
        Args:
            project_memory: ProjectMemory manager instance
            project_name: Optional project name to use for operations
        """
        from typing import Optional, Any
        from src.models.response_models import ProjectContextModel
        
        self.project_memory = project_memory
        self.project_name = project_name
        self.logger = project_memory.logger if hasattr(project_memory, "logger") else None
        # Storage for the Pydantic context model
        self._context_model: Optional[ProjectContextModel] = None
    
    def create_project(self, name: str, **kwargs) -> str:
        """
        Create a project with the current context.
        
        Args:
            name: Name of the project
            **kwargs: Additional parameters including description, metadata, and tags
        
        Returns:
            JSON response with created project
        """
        # Prepare project data
        project_data = {
            "name": name
        }
        
        # Add optional parameters if provided
        if "description" in kwargs:
            project_data["description"] = kwargs["description"]
        if "metadata" in kwargs:
            project_data["metadata"] = kwargs["metadata"]
        if "tags" in kwargs:
            project_data["tags"] = kwargs["tags"]
            
        result = self.project_memory.create_project_container(project_data)
        
        # Ensure string return
        if isinstance(result, str):
            return result
        else:
            return json.dumps(result)
    
    def create_component(self, name: str, component_type: str, domain_name: str, **kwargs) -> str:
        """
        Create a component within this project context.
        
        Args:
            name: Name of the component
            component_type: Type of the component
            domain_name: Name of the domain
            **kwargs: Additional parameters
        
        Returns:
            JSON response with created component
        """
        # Extract optional parameters with defaults
        description = kwargs.pop("description", None)
        content = kwargs.pop("content", None)
        metadata = kwargs.pop("metadata", None)
        
        result = self.project_memory.create_project_component(
            name=name,
            component_type=component_type,
            domain_name=domain_name,
            container_name=self.project_name,
            description=description,
            content=content,
            metadata=metadata
        )
        
        # Ensure string return
        if isinstance(result, str):
            return result
        else:
            return json.dumps(result)
    
    def create_domain(self, name: str, **kwargs) -> str:
        """
        Create a domain within this project context.
        
        Args:
            name: Name of the domain
            **kwargs: Additional parameters
        
        Returns:
            JSON response with created domain
        """
        # Extract optional parameters with defaults
        description = kwargs.pop("description", None)
        properties = kwargs.pop("properties", None)
        
        result = self.project_memory.create_project_domain(
            name=name,
            container_name=self.project_name,
            description=description,
            properties=properties
        )
        
        # Ensure string return
        if isinstance(result, str):
            return result
        else:
            return json.dumps(result)
    
    def create_domain_entity(self, name: str, entity_type: str, **kwargs) -> str:
        """
        Create a domain entity within this project context.
        
        Args:
            name: Name of the entity
            entity_type: Type of entity (DECISION, REQUIREMENT, CONSTRAINT, etc.)
            **kwargs: Additional parameters
                - domain_name: Name of the domain (required)
                - description: Optional description of the entity
                - properties: Optional dictionary of properties
        
        Returns:
            JSON response with created entity
        """
        # Extract optional parameters
        domain_name = kwargs.pop("domain_name", None)
        description = kwargs.pop("description", None)
        properties = kwargs.pop("properties", None)
        
        # Domain name is required for domain entities
        if not domain_name:
            error_msg = "Missing domain_name for domain entity creation"
            return json.dumps({
                "status": "error",
                "error": error_msg,
                "code": "missing_domain_name"
            })
        
        # Create the domain entity
        result = self.project_memory.add_entity_to_project_domain(
            entity_name=name,
            entity_type=entity_type,
            domain_name=domain_name,
            container_name=self.project_name,
            properties=properties or {"description": description} if description else None
        )
        
        # Ensure string return
        if isinstance(result, str):
            return result
        else:
            return json.dumps(result)
    
    def relate(self, source_name: str, target_name: str, relation_type: str, **kwargs) -> str:
        """
        Create a relationship between entities within this project context.
        
        Args:
            source_name: Name of the source entity
            target_name: Name of the target entity
            relation_type: Type of relationship to create
              - DEPENDS_ON: Dependency relationship
              - IMPLEMENTS: Implementation relationship
              - CONTAINS: Containment relationship
              - USES: Usage relationship
              - EXTENDS: Extension relationship
            **kwargs: Additional parameters
                 
        Returns:
            JSON response with operation results
            
        Required parameters:
            - source_name: Source entity identifier
            - target_name: Target entity identifier
            - relation_type: Type of relationship
            
        Optional parameters:
            - domain_name: Domain name if entities are in the same domain
            - entity_type: Type of entities ('component', 'domain', 'dependency')
            - properties: Dictionary with additional relationship attributes
            
        Response format:
            All operations return a JSON string with at minimum:
            - status: "success" or "error"
            - message or error: Description of result or error
            - relationship: Relationship data if successful
        """
        try:
            # Extract optional parameters with defaults
            entity_type = kwargs.pop("entity_type", "component").lower()
            domain_name = kwargs.pop("domain_name", None)
            properties = kwargs.pop("properties", None)
            
            # Determine which relationship creation method to use based on entity_type
            if entity_type == "domain":
                # Create relationship between domains
                result = self.project_memory.create_project_domain_relationship(
                    from_domain=source_name,
                    to_domain=target_name,
                    container_name=self.project_name,
                    relation_type=relation_type,
                    properties=properties
                )
            elif entity_type == "component" and domain_name:
                # Create relationship between components in the same domain
                result = self.project_memory.create_project_component_relationship(
                    from_component=source_name,
                    to_component=target_name,
                    domain_name=domain_name,
                    container_name=self.project_name,
                    relation_type=relation_type,
                    properties=properties
                )
            elif entity_type == "dependency" and domain_name:
                # Create a dependency relationship between components
                result = self.project_memory.create_project_dependency(
                    from_component=source_name,
                    to_component=target_name,
                    domain_name=domain_name,
                    container_name=self.project_name,
                    dependency_type=relation_type,
                    properties=properties
                )
            else:
                # Handle unsupported entity type or missing domain_name
                error_msg = "Unsupported entity type or missing domain_name"
                return json.dumps({
                    "status": "error",
                    "error": error_msg,
                    "code": "invalid_entity_type"
                })
            
            # Ensure string return
            if isinstance(result, str):
                return result
            else:
                return json.dumps(result)
                
        except Exception as e:
            return json.dumps({
                "status": "error",
                "error": f"Failed to create entity relationship: {str(e)}",
                "code": "relationship_creation_error"
            })
    
    def search(self, query: str, **kwargs) -> str:
        """
        Search for entities within this project context.
        
        Args:
            query: Search query text
            **kwargs: Additional parameters
        
        Returns:
            JSON response with search results
        """
        # Extract optional parameters with defaults
        entity_types = kwargs.pop("entity_types", None)
        limit = kwargs.pop("limit", 10)
        semantic = kwargs.pop("semantic", False)
        domain_name = kwargs.pop("domain_name", None)
        
        # Determine which search method to use
        if semantic:
            # Use semantic search
            result = self.project_memory.semantic_search_project(
                search_term=query,
                container_name=self.project_name,
                entity_types=entity_types,
                limit=limit
            )
        else:
            # Use regular search
            result = self.project_memory.search_project_entities(
                search_term=query,
                container_name=self.project_name,
                entity_types=entity_types,
                limit=limit,
                semantic=False
            )
        
        # If domain filtering is needed, parse and filter results
        if domain_name and isinstance(result, str):
            try:
                result_data = json.loads(result)
                if "data" in result_data and "entities" in result_data["data"]:
                    # Filter entities by domain
                    filtered_entities = []
                    for entity in result_data["data"]["entities"]:
                        # Check if entity belongs to specified domain
                        if entity.get("domain") == domain_name:
                            filtered_entities.append(entity)
                    
                    # Replace entities with filtered list
                    result_data["data"]["entities"] = filtered_entities
                    result_data["data"]["total_count"] = len(filtered_entities)
                    return json.dumps(result_data)
                
                return result
            except json.JSONDecodeError:
                # If parsing fails, return original result
                return result
        
        # Ensure string return
        if isinstance(result, str):
            return result
        else:
            return json.dumps(result)
    
    def get_structure(self, **kwargs) -> str:
        """
        Get the structure of this project.
        
        Args:
            **kwargs: Additional parameters
        
        Returns:
            JSON response with project structure
        """
        # Extract optional parameters with defaults
        include_domains = kwargs.pop("include_domains", True)
        include_components = kwargs.pop("include_components", True)
        include_relationships = kwargs.pop("include_relationships", True)
        domain_name = kwargs.pop("domain_name", None)
        
        # Build the query parameters
        query_params = {
            "project_id": self.project_name,
            "include_domains": include_domains,
            "include_components": include_components,
            "include_relationships": include_relationships
        }
        
        if domain_name:
            query_params["domain_name"] = domain_name
            
        # Get project status which includes structure information
        result = self.project_memory.get_project_status(self.project_name)
        
        # Ensure string return
        if isinstance(result, str):
            return result
        else:
            return json.dumps(result)
    
    def add_observation(self, entity_name: str, content: str, **kwargs) -> str:
        """
        Add an observation to an entity within this project context.
        
        Args:
            entity_name: Name of the entity
            content: Content of the observation
            **kwargs: Additional parameters
        
        Returns:
            JSON response with observation data
        """
        # Extract optional parameters with defaults
        observation_type = kwargs.pop("observation_type", "general")
        
        # Build the observation data
        observation_data = {
            "entity_name": entity_name,
            "content": content,
            "observation_type": observation_type
        }
        
        # Add an observation
        result = self.project_memory.add_observations([observation_data])
        
        # Ensure string return
        if isinstance(result, str):
            return result
        else:
            return json.dumps(result)
    
    def update(self, entity_name: str, updates: Dict[str, Any], **kwargs) -> str:
        """
        Update an entity within this project context.
        
        Args:
            entity_name: Name of the entity to update
            updates: Dictionary of updates to apply
            **kwargs: Additional parameters
        
        Returns:
            JSON response with updated entity
        """
        # Extract optional parameters with defaults
        entity_type = kwargs.pop("entity_type", "component").lower()
        domain_name = kwargs.pop("domain_name", None)
        
        # Determine which update method to use based on entity_type
        if entity_type == "project":
            # Update project container
            result = self.project_memory.update_project_container(entity_name, updates)
        elif entity_type == "domain":
            # Update domain
            result = self.project_memory.update_project_domain(
                name=entity_name,
                container_name=self.project_name,
                updates=updates
            )
        elif entity_type == "component" and domain_name:
            # Update component
            result = self.project_memory.update_project_component(
                name=entity_name,
                container_name=self.project_name,
                updates=updates,
                domain_name=domain_name
            )
        else:
            # Handle unsupported entity type or missing domain_name for components
            error_msg = f"Unsupported entity type '{entity_type}' or missing domain_name for component update"
            return json.dumps({
                "status": "error",
                "error": error_msg,
                "code": "invalid_update_parameters"
            })
        
        # Ensure string return
        if isinstance(result, str):
            return result
        else:
            return json.dumps(result)
    
    def delete_entity(self, entity_name: str, **kwargs) -> str:
        """
        Delete an entity within this project context.
        
        Args:
            entity_name: Name of the entity to delete
            **kwargs: Additional parameters
                - entity_type: Type of entity (project, domain, component, observation)
                - domain_name: Domain name (required for components)
                - delete_contents: Whether to delete contained entities (default: False)
                - observation_id: ID of observation to delete (for observation entity_type)
        
        Returns:
            JSON response with deletion result
        """
        # Extract optional parameters with defaults
        entity_type = kwargs.pop("entity_type", "component").lower()
        domain_name = kwargs.pop("domain_name", None)
        delete_contents = kwargs.pop("delete_contents", False)
        observation_id = kwargs.pop("observation_id", None)
        
        # Build deletion parameters
        delete_params = {
            "entity_name": entity_name,
            "entity_type": entity_type
        }
        
        # Add container name for all operations
        delete_params["container_name"] = self.project_name
        
        # Add optional parameters if provided
        if domain_name:
            delete_params["domain_name"] = domain_name
        if delete_contents:
            delete_params["delete_contents"] = delete_contents
        if observation_id:
            delete_params["observation_id"] = observation_id
            
        # Use the appropriate deletion method
        result = self.project_memory.delete_entity(entity_name, entity_type=entity_type, **kwargs)
        
        # Ensure string return
        if isinstance(result, str):
            return result
        else:
            return json.dumps(result)
    
    def delete_relationship(self, source_name: str, target_name: str, relationship_type: str, **kwargs) -> str:
        """
        Delete a relationship between entities within this project context.
        
        Args:
            source_name: Name of the source entity
            target_name: Name of the target entity
            relationship_type: Type of relationship to delete
            **kwargs: Additional parameters
                - domain_name: Domain name (required for component relationships)
        
        Returns:
            JSON response with deletion result
        """
        # Extract optional parameters with defaults
        domain_name = kwargs.pop("domain_name", None)
        
        # For dependencies and component relationships, domain name is required
        if not domain_name:
            error_msg = "Missing domain_name for relationship deletion"
            return json.dumps({
                "status": "error",
                "error": error_msg,
                "code": "missing_domain_name"
            })
            
        # Delete the relationship
        result = self.project_memory.delete_project_dependency(
            from_component=source_name,
            to_component=target_name,
            domain_name=domain_name,
            container_name=self.project_name,
            dependency_type=relationship_type
        )
        
        # Ensure string return
        if isinstance(result, str):
            return result
        else:
            return json.dumps(result)

class GraphMemoryManager:
    """
    Facade to maintain backward compatibility with original API.
    
    This class combines functionality from all the specialized managers
    to provide a unified interface that matches the original GraphManager.
    """
    
    def __init__(self, logger=None, embedding_api_key=None, embedding_model=None, neo4j_uri=None, 
                neo4j_username=None, neo4j_password=None, database=None, embedding_index_name=None):
        """
        Initialize the Graph Memory Manager.
        
        Args:
            logger: Optional logger instance
            embedding_api_key: Optional API key for embedding service
            embedding_model: Optional embedding model name
            neo4j_uri: Optional Neo4j URI
            neo4j_username: Optional Neo4j username
            neo4j_password: Optional Neo4j password
            database: Optional Neo4j database name
            embedding_index_name: Optional name for the embedding index
        """
        # Initialize base manager
        self.base_manager = BaseManager(logger=logger)
        
        # Store configuration values for later use in initialize
        self._neo4j_uri = neo4j_uri
        self._neo4j_username = neo4j_username
        self._neo4j_password = neo4j_password
        self._database = database
        self._embedding_index_name = embedding_index_name
        
        # Store logger for consistent logging
        self.logger = logger
        
        # Detect operating mode (SSE or STDIO)
        self._is_sse_mode = self._detect_sse_mode()
        
        # Configure embedding based on mode
        if self._is_sse_mode:
            # In SSE mode, we default to disabled but will load from config files later
            if self.logger:
                self.logger.info("Running in SSE mode, will load embedding config from files")
            self.embedding_enabled = False
            self.embedder_provider = "openai"  # Default, will be overridden by config
        else:
            # In STDIO mode, check environment variables
            if self.logger:
                self.logger.info("Running in STDIO mode, using environment variables")
            env_embedder_provider = os.environ.get("EMBEDDER_PROVIDER", "none").lower()
            self.embedding_enabled = env_embedder_provider != "none"
            self.embedder_provider = env_embedder_provider if env_embedder_provider and env_embedder_provider != "none" else "openai"
        
        # Initialize embedding adapter - will be configured in initialize
        self.embedding_adapter = EmbeddingAdapter(logger=logger)
        
        # Store embedding configuration for later use
        self.embedding_api_key = embedding_api_key
        self.embedding_model = embedding_model
        
        # Initialize specialized managers - ensure proper object instantiation
        self.entity_manager = EntityManager(self.base_manager)
        self.relation_manager = RelationManager(self.base_manager)
        self.observation_manager = ObservationManager(self.base_manager)
        self.search_manager = SearchManager(self.base_manager)
        
        # Initialize specialized memory systems
        self.lesson_memory = LessonMemoryManager(self.base_manager)
        self.project_memory = ProjectMemoryManager(self.base_manager)
        
        # Required attributes for backward compatibility
        self.default_project_name = "default"
        self.neo4j_uri = neo4j_uri or "bolt://localhost:7687"
        self.neo4j_user = neo4j_username or "neo4j"
        self.neo4j_password = neo4j_password or "password"
        self.neo4j_database = database or "neo4j"
        self.neo4j_driver = None
        
        # Client-specific state tracking
        self._client_id = None
        self._client_projects = {}
    
    def _standardize_response(self, result_json: Union[str, Dict[str, Any]], success_message: str, error_code: str) -> str:
        """
        Standardize API responses for consistency across the system.
        
        Args:
            result_json: JSON result string or dict from a manager or operation
            success_message: Message to include in successful responses
            error_code: Error code to use for error responses
            
        Returns:
            Standardized JSON response string
        """
        try:
            # Parse the result JSON
            if isinstance(result_json, str):
                try:
                    result_data = json.loads(result_json)
                except json.JSONDecodeError as e:
                    if hasattr(self, 'logger') and self.logger:
                        self.logger.error(f"Error parsing JSON: {str(e)}, original: {result_json}")
                    error = ErrorDetail(
                        code="json_parse_error",
                        message=f"Invalid JSON response: {str(e)}",
                        details=None
                    )
                    error_response = ErrorResponse(
                        status="error",
                        timestamp=datetime.now(),
                        error=error
                    )
                    return error_response.model_dump_json()
            else:
                # Handle case where result is already a dict
                result_data = result_json
            
            # If the result already has a Pydantic-style error structure, return it directly
            if isinstance(result_data, dict) and result_data.get("status") == "error" and "error" in result_data:
                if isinstance(result_json, str):
                    return result_json
                else:
                    # Convert dict to JSON if needed
                    return json.dumps(result_data, default=str)
                
            # If result contains an "error" key, convert to proper error response
            if isinstance(result_data, dict) and "error" in result_data:
                details = None
                # Extract additional details if they exist
                if "details" in result_data:
                    details = result_data["details"]
                elif "context" in result_data:
                    details = result_data["context"]
                
                error = ErrorDetail(
                    code=error_code,
                    message=result_data["error"],
                    details=details
                )
                error_response = ErrorResponse(
                    status="error",
                    timestamp=datetime.now(),
                    error=error
                )
                return error_response.model_dump_json()
                
            # Otherwise, it's a success response - create proper Pydantic response
            # Create a standard response with just the message and timestamp
            success_response = SuccessResponse(
                status="success",
                message=success_message,
                timestamp=datetime.now(),
                data=result_data  # Include all result data
            )
            
            # Return as JSON with proper datetime handling
            return success_response.model_dump_json()
            
        except Exception as e:
            # Something went wrong while processing the response
            if hasattr(self, 'logger') and self.logger:
                self.logger.error(f"Error standardizing response: {str(e)}")
                
            error = ErrorDetail(
                code="response_processing_error",
                message=f"Error processing response: {str(e)}",
                details={"original_response": str(result_json)[:1000]} if result_json else None
            )
            error_response = ErrorResponse(
                status="error",
                timestamp=datetime.now(),
                error=error
            )
            return error_response.model_dump_json()
    
    def _detect_sse_mode(self) -> bool:
        """
        Detect if running in SSE mode by checking for common indicators.
        
        Returns:
            True if in SSE mode, False for STDIO mode
        """
        import os
        
        # Check for presence of environment variable that would only be in STDIO mode
        stdio_indicators = ["MCP_STDIO_MODE", "STDIO_MODE"]
        for indicator in stdio_indicators:
            if os.environ.get(indicator, "").lower() in ("1", "true", "yes"):
                return False
                
        # Default to SSE mode if no clear indicators
        return True
    
    def initialize(self, client_id=None) -> bool:
        """
        Initialize connections to Neo4j and embedding services.
        
        Args:
            client_id: Optional client ID for client isolation
            
        Returns:
            True if initialization successful, False otherwise
        """
        # Track client ID for isolation
        self._client_id = client_id
        
        # Use different initialization approaches based on mode
        if self._is_sse_mode:
            return self._initialize_sse_mode(client_id)
        else:
            return self._initialize_stdio_mode(client_id)
    
    def _initialize_sse_mode(self, client_id=None) -> bool:
        """
        Initialize in SSE mode with default settings.
        
        In SSE mode, we start with default values and wait for configuration 
        to be provided by the client through the get_unified_config tool.
        
        Args:
            client_id: Optional client ID for client isolation
            
        Returns:
            True if initialization successful, False otherwise
        """
        try:
            # In SSE mode, we start with defaults and wait for config from client
            if self.logger:
                self.logger.info("Initializing in SSE mode with defaults - waiting for client configuration")
            
            # Store client ID for later use
            if client_id:
                self._client_id = client_id
            
            # Initialize base manager for Neo4j connection - critical for all operations
            try:
                if not self.base_manager.initialize():
                    if self.logger:
                        self.logger.error("Failed to initialize base manager")
                    return False
            except Exception as e:
                if self.logger:
                    self.logger.error(f"Failed to initialize base manager: {str(e)}")
                return False
            
            # Update forwarded properties from base manager
            self._update_properties_from_base_manager()
            
            # Ensure all manager object references are proper instances
            if not self._ensure_managers_initialized():
                return False
                
            # Client isolation: Store client-specific project memory if client_id is provided
            if client_id and client_id not in self._client_projects:
                self._client_projects[client_id] = self.project_memory
                
            return True
            
        except Exception as e:
            if self.logger:
                self.logger.error(f"Error during SSE initialization: {str(e)}")
            return False
    
    def _initialize_stdio_mode(self, client_id=None) -> bool:
        """
        Initialize in STDIO mode using environment variables.
        
        Args:
            client_id: Optional client ID for client isolation
            
        Returns:
            True if initialization successful, False otherwise
        """
        # Override configuration if provided in constructor
        import os
        
        if self._neo4j_uri:
            os.environ["NEO4J_URI"] = self._neo4j_uri
        
        if self._neo4j_username:
            os.environ["NEO4J_USER"] = self._neo4j_username
            
        if self._neo4j_password:
            os.environ["NEO4J_PASSWORD"] = self._neo4j_password
            
        if self._database:
            os.environ["NEO4J_DATABASE"] = self._database
            
        # Check if embeddings disabled via environment variable (takes precedence)
        env_embedder_provider = os.environ.get("EMBEDDER_PROVIDER", "").lower()
        if env_embedder_provider == "none":
            # Explicitly disabled via environment
            self.embedding_enabled = False
            if self.logger:
                self.logger.info("Embeddings explicitly disabled via EMBEDDER_PROVIDER=none")
        
        # Handle embedding initialization when enabled
        if self.embedding_enabled:
            # Try to initialize embedding adapter
            adapter_success = self.embedding_adapter.init_embedding_manager(
                api_key=self.embedding_api_key,
                model_name=self.embedding_model
            )
            
            if not adapter_success:
                # Log failure but continue without embeddings
                self.embedding_enabled = False
                if self.logger:
                    self.logger.error("Failed to initialize embedding manager, continuing without embeddings")
        else:
            # Log that embeddings are skipped
            if self.logger:
                self.logger.info("Embeddings disabled - skipping embedding adapter initialization")
        
        # Initialize base manager for Neo4j connection - critical for all operations
        try:
            if not self.base_manager.initialize():
                if self.logger:
                    self.logger.error("Failed to initialize base manager")
                return False
        except Exception as e:
            if self.logger:
                self.logger.error(f"Failed to initialize base manager: {str(e)}")
            return False
        
        # Update forwarded properties from base manager
        self._update_properties_from_base_manager()
        
        # Ensure all manager object references are proper instances
        if not self._ensure_managers_initialized():
            return False
        
        # Client isolation: Store client-specific project memory if client_id is provided
        if client_id and client_id not in self._client_projects:
            self._client_projects[client_id] = self.project_memory
            
        return True
    
    def _update_properties_from_base_manager(self):
        """Update forwarded properties from base manager."""
        self.neo4j_uri = getattr(self.base_manager, "neo4j_uri", "bolt://localhost:7687")
        self.neo4j_user = getattr(self.base_manager, "neo4j_user", "neo4j")
        self.neo4j_password = getattr(self.base_manager, "neo4j_password", "password")
        self.neo4j_database = getattr(self.base_manager, "neo4j_database", "neo4j")
        self.embedder_provider = getattr(self.base_manager, "embedder_provider", "openai")
        self.neo4j_driver = getattr(self.base_manager, "neo4j_driver", None)
    
    def _ensure_managers_initialized(self) -> bool:
        """Ensure all manager objects are properly initialized."""
        if not isinstance(self.project_memory, object) or callable(self.project_memory):
            if self.logger:
                self.logger.warn("Project memory manager not properly initialized, recreating")
            try:
                self.project_memory = ProjectMemoryManager(self.base_manager)
            except Exception as e:
                if self.logger:
                    self.logger.error(f"Failed to re-initialize project memory manager: {str(e)}")
                return False
        return True
    
    def close(self) -> None:
        """Close connections to Neo4j and clean up resources."""
        self.base_manager.close()
    
    def check_connection(self) -> bool:
        """
        Check if the Neo4j connection is working.
        
        Returns:
            True if connection is working, False otherwise
        """
        # Implement connection check directly
        if not self.neo4j_driver:
            return False
            
        try:
            # Simple query to test connection
            result = self.neo4j_driver.execute_query(
                "RETURN 'Connection test' as message",
                database_=self.neo4j_database
            )
            return True
        except Exception as e:
            if self.logger:
                self.logger.error(f"Connection check failed: {str(e)}")
            return False
    
    # Entity Management
    
    def create_entity(self, entity_data: Dict[str, Any]) -> str:
        """
        Create an entity.
        
        Args:
            entity_data: Dictionary with the entity data
            
        Returns:
            JSON string with the result
        """
        try:
            self._ensure_initialized()
            
            # Associate with current project if not specified
            if "project" not in entity_data and self.default_project_name:
                entity_data["project"] = self.default_project_name
                
            # Use the entity manager to create the entity
            result = self.entity_manager.create_entities([entity_data])
            
            # Standardize the response
            return self._standardize_response(
                result_json=result,
                success_message="Entity created successfully",
                error_code="entity_creation_error"
            )
        except Exception as e:
            if self.logger:
                self.logger.error(f"Error creating entity: {str(e)}")
                
            error = ErrorDetail(
                code="entity_creation_error",
                message=f"Failed to create entity: {str(e)}",
                details=None
            )
            error_response = ErrorResponse(
                status="error",
                timestamp=datetime.now(),
                error=error
            )
            return error_response.model_dump_json()
    
    def create_entities(self, entities: List[Dict[str, Any]]) -> str:
        """
        Create multiple entities.
        
        Args:
            entities: List of dictionaries with entity data
            
        Returns:
            JSON string with the result
        """
        try:
            self._ensure_initialized()
            
            # Associate with current project if not specified
            if self.default_project_name:
                for entity in entities:
                    if "project" not in entity:
                        entity["project"] = self.default_project_name
                        
            # Use the entity manager to create the entities
            result = self.entity_manager.create_entities(entities)
            
            # Standardize the response
            return self._standardize_response(
                result_json=result,
                success_message=f"Created {len(entities)} entities",
                error_code="entity_creation_error"
            )
        except Exception as e:
            if self.logger:
                self.logger.error(f"Error creating entities: {str(e)}")
                
            error = ErrorDetail(
                code="entity_creation_error",
                message=f"Failed to create entities: {str(e)}",
                details={"entity_count": len(entities)}
            )
            error_response = ErrorResponse(
                status="error",
                timestamp=datetime.now(),
                error=error
            )
            return error_response.model_dump_json()
    
    def get_entity(self, entity_name: str) -> str:
        """
        Get an entity from the knowledge graph.
        
        Args:
            entity_name: The name of the entity to retrieve
            
        Returns:
            JSON string with the entity information
        """
        try:
            self._ensure_initialized()
            
            # Use the entity manager to get the entity
            result = self.entity_manager.get_entity(entity_name)
            
            # Standardize the response
            return self._standardize_response(
                result_json=result,
                success_message=f"Retrieved entity: {entity_name}",
                error_code="entity_retrieval_error"
            )
        except Exception as e:
            if self.logger:
                self.logger.error(f"Error retrieving entity: {str(e)}")
                
            error = ErrorDetail(
                code="entity_retrieval_error",
                message=f"Failed to retrieve entity: {str(e)}",
                details={"entity_name": entity_name}
            )
            error_response = ErrorResponse(
                status="error",
                timestamp=datetime.now(),
                error=error
            )
            return error_response.model_dump_json()
    
    def update_entity(self, entity_name: str, updates: Dict[str, Any]) -> str:
        """
        Update an entity in the knowledge graph.
        
        Args:
            entity_name: The name of the entity
            updates: The updates to apply
            
        Returns:
            JSON string with the updated entity
        """
        return self.entity_manager.update_entity(entity_name, updates)
    
    def delete_entity(self, entity_name: str) -> str:
        """
        Delete an entity from the knowledge graph.
        
        Args:
            entity_name: The name of the entity
            
        Returns:
            JSON string with the result of the deletion
        """
        return self.entity_manager.delete_entity(entity_name)
    
    # Relation Management
    
    def create_relationship(self, relationship_data: Dict[str, Any]) -> str:
        """
        Create a new relationship in the knowledge graph.
        
        Args:
            relationship_data: Relationship information
            
        Returns:
            JSON response with operation result
        """
        self._ensure_initialized()
        return self.relation_manager.create_relationship(relationship_data)
        
    def create_relations(self, relations: List[Dict[str, Any]]) -> str:
        """
        Create multiple new relationships in the knowledge graph.
        
        Args:
            relations: List of relation objects
            
        Returns:
            JSON response with operation result
        """
        self._ensure_initialized()
        return self.relation_manager.create_relations(relations)
    
    def create_relationships(self, relationships: List[Dict[str, Any]]) -> str:
        """
        Create multiple new relationships in the knowledge graph.
        This is an alias for create_relations to maintain API compatibility.
        
        Args:
            relationships: List of relationship objects
            
        Returns:
            JSON response with operation result
        """
        self._ensure_initialized()
        return self.relation_manager.create_relations(relationships)
    
    def get_relationships(self, entity_name: str, relation_type: Optional[str] = None) -> str:
        """
        Get relationships for an entity from the knowledge graph.
        
        Args:
            entity_name: The name of the entity
            relation_type: Optional type of relationship to filter by
            
        Returns:
            JSON string with the relationships
        """
        self._ensure_initialized()
        return self.relation_manager.get_relations(entity_name, relation_type)
    
    def update_relation(self, from_entity: str, to_entity: str, relation_type: str, updates: Dict[str, Any]) -> str:
        """
        Update a relation in the knowledge graph.
        
        Args:
            from_entity: The name of the source entity
            to_entity: The name of the target entity
            relation_type: The type of the relation
            updates: The updates to apply
            
        Returns:
            JSON string with the updated relation
        """
        return self.relation_manager.update_relation(from_entity, to_entity, relation_type, updates)
    
    def delete_relation(self, from_entity: str, to_entity: str, relation_type: Optional[str] = None) -> str:
        """
        Delete a relation from the knowledge graph.
        
        Args:
            from_entity: The name of the source entity
            to_entity: The name of the target entity
            relation_type: Optional type of the relation
            
        Returns:
            JSON string with the result of the deletion
        """
        # Convert None to empty string if needed for compatibility
        relation_type_str = relation_type if relation_type is not None else ""
        return self.relation_manager.delete_relation(from_entity, to_entity, relation_type_str)
    
    # Observation Management
    
    def add_observations(self, observations: List[Dict[str, Any]]) -> str:
        """
        Add observations to entities.
        
        Args:
            observations: List of observations to add
            
        Returns:
            JSON string with result
        """
        self._ensure_initialized()
        return self.observation_manager.add_observations(observations)
    
    def get_observations(self, entity_name: str, observation_type: Optional[str] = None) -> str:
        """
        Get observations for an entity from the knowledge graph.
        
        Args:
            entity_name: The name of the entity
            observation_type: Optional type of observation to filter by
            
        Returns:
            JSON string with the observations
        """
        self._ensure_initialized()
        return self.observation_manager.get_entity_observations(entity_name, observation_type)
    
    def update_observation(self, entity_name: str, observation_id: str, content: str, 
                          observation_type: Optional[str] = None) -> str:
        """
        Update an observation in the knowledge graph.
        
        Args:
            entity_name: The name of the entity
            observation_id: The ID of the observation
            content: The new content for the observation
            observation_type: Optional new type for the observation
            
        Returns:
            JSON string with the updated observation
        """
        return self.observation_manager.update_observation(entity_name, observation_id, content, observation_type)
    
    def delete_observation(self, entity_name: str, observation_content: Optional[str] = None, 
                          observation_id: Optional[str] = None) -> str:
        """
        Delete an observation from the knowledge graph.
        
        Args:
            entity_name: The name of the entity
            observation_content: The content of the observation to delete
            observation_id: The ID of the observation to delete
            
        Returns:
            JSON string with the result of the deletion
        """
        return self.observation_manager.delete_observation(entity_name, observation_content, observation_id)
    
    # Search Functionality
    
    def search_nodes(self, query: str, limit: int = 10, entity_types: Optional[List[str]] = None, 
                   semantic: bool = True) -> str:
        """
        Search for nodes in the knowledge graph.
        
        Args:
            query: The search query
            limit: Maximum number of results to return
            entity_types: List of entity types to filter by
            semantic: Whether to use semantic search (requires embeddings)
            
        Returns:
            JSON string with search results
        """
        try:
            self._ensure_initialized()
            
            # Use the search manager to search for entities
            result = self.search_manager.search_entities(query, limit, entity_types, semantic)
            
            # Standardize the response
            return self._standardize_response(
                result_json=result,
                success_message=f"Search completed for: {query}",
                error_code="search_error"
            )
        except Exception as e:
            if self.logger:
                self.logger.error(f"Error searching nodes: {str(e)}")
                
            error = ErrorDetail(
                code="search_error",
                message=f"Failed to search nodes: {str(e)}",
                details={
                    "query": query,
                    "limit": limit,
                    "entity_types": entity_types,
                    "semantic": semantic
                }
            )
            error_response = ErrorResponse(
                status="error",
                timestamp=datetime.now(),
                error=error
            )
            return error_response.model_dump_json()
    
    def query_knowledge_graph(self, cypher_query: str, params: Optional[Dict[str, Any]] = None) -> str:
        """
        Execute a custom Cypher query against the knowledge graph.
        
        Args:
            cypher_query: The Cypher query to execute
            params: Optional parameters for the Cypher query
            
        Returns:
            JSON string with the query results
        """
        return self.search_manager.query_knowledge_graph(cypher_query, params)
    
    def search_entity_neighborhoods(self, entity_name: str, max_depth: int = 2, max_nodes: int = 50) -> str:
        """
        Search for entity neighborhoods (entity graph exploration).
        
        Args:
            entity_name: The name of the entity to start from
            max_depth: Maximum relationship depth to traverse
            max_nodes: Maximum number of nodes to return
            
        Returns:
            JSON string with the neighborhood graph
        """
        return self.search_manager.search_entity_neighborhoods(entity_name, max_depth, max_nodes)
    
    def find_paths_between_entities(self, from_entity: str, to_entity: str, max_depth: int = 4) -> str:
        """
        Find all paths between two entities in the knowledge graph.
        
        Args:
            from_entity: The name of the starting entity
            to_entity: The name of the target entity
            max_depth: Maximum relationship depth to traverse
            
        Returns:
            JSON string with all paths found
        """
        return self.search_manager.find_paths_between_entities(from_entity, to_entity, max_depth)
    
    # Embedding Functionality
    
    def get_embedding(self, text: str) -> Optional[List[float]]:
        """
        Get embedding for text.
        
        Args:
            text: The text to get embedding for
            
        Returns:
            List of float values representing the embedding vector, or None if failed
        """
        return self.embedding_adapter.get_embedding(text)
    
    def similarity_score(self, embedding1: List[float], embedding2: List[float]) -> float:
        """
        Calculate cosine similarity between two embeddings.
        
        Args:
            embedding1: First embedding vector
            embedding2: Second embedding vector
            
        Returns:
            Cosine similarity score (0-1)
        """
        return self.embedding_adapter.similarity_score(embedding1, embedding2)
        
    # Additional methods for backward compatibility with original GraphManager
    
    def set_project_name(self, project_name: str, client_id: Optional[str] = None) -> None:
        """
        Set the default project name for memory operations.
        
        Args:
            project_name: The project name to use
            client_id: Optional client ID for isolation
        """
        # Store client ID for future reference if provided
        if client_id:
            self._client_id = client_id
            
        self.base_manager.set_project_name(project_name)
        self.default_project_name = self.base_manager.default_project_name
    
    def search_entities(self, query: str, limit: int = 10, project_name: Optional[str] = None) -> str:
        """
        Search for entities in the knowledge graph.
        
        Args:
            query: The search query
            limit: Maximum number of results to return
            project_name: Optional project name to scope the search
            
        Returns:
            JSON string with search results
        """
        # Set project context if provided
        if project_name:
            self.set_project_name(project_name)
        
        return self.search_manager.search_entities(query, limit, semantic=True)
    
    def get_all_memories(self) -> str:
        """
        Get all entities in the knowledge graph.
            
        Returns:
            JSON string with all entities
        """
        try:
            # Build query
            query = """
                MATCH (e:Entity)
                RETURN e
                ORDER BY e.name
                """
            
            # Execute query with appropriate parameters
            records = self.base_manager.safe_execute_read_query(query)
            
            # Process results
            entities = []
            if records:
                for record in records:
                    entity = record.get("e")
                    if entity:
                        # Convert Neo4j node to dict
                        entity_dict = dict(entity.items())
                        entity_dict["id"] = entity.id
                        entities.append(entity_dict)
            
            return dict_to_json({"memories": entities})
            
        except Exception as e:
            if self.logger:
                self.logger.error(f"Error getting all memories: {str(e)}")
            return dict_to_json({"error": f"Failed to get all memories: {str(e)}"})

    def delete_all_memories(self, project_name: Optional[str] = None) -> str:
        """
        Delete all memories in the knowledge graph.
        
        Args:
            project_name: Optional project name to scope the deletion
            
        Returns:
            JSON string with operation result
        """
        if project_name:
            self.set_project_name(project_name)
        
        try:
            # Delete all nodes and relationships
            query = """
            MATCH (n)
            DETACH DELETE n
            """
            
            self.base_manager.safe_execute_write_query(query)
            
            return dict_to_json({
                "status": "success",
                "message": "All memories deleted successfully",
                "project": self.default_project_name
            })
            
        except Exception as e:
            if self.logger:
                self.logger.error(f"Error deleting all memories: {str(e)}")
            return dict_to_json({"error": f"Failed to delete all memories: {str(e)}"})
    
    def apply_client_config(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """
        Apply client configuration.
        
        Args:
            config: Configuration to apply
            
        Returns:
            Dictionary with status of configuration application
        """
        try:
            # Store client ID for isolation if provided
            client_id = config.get("client_id")
            if client_id:
                self._client_id = client_id
            
            # Apply project name if present
            if "project_name" in config:
                project_name = config["project_name"]
                self.default_project_name = project_name
                self.base_manager.set_project_name(project_name)
            
            # Apply Neo4j configuration if present
            if "neo4j" in config:
                neo4j_config = config["neo4j"]
                neo4j_changed = False
                
                if "uri" in neo4j_config:
                    self.neo4j_uri = neo4j_config["uri"]
                    neo4j_changed = True
                if "username" in neo4j_config:
                    self.neo4j_user = neo4j_config["username"]
                    neo4j_changed = True
                if "password" in neo4j_config:
                    # Ensure password is properly handled as a string to preserve special characters
                    password = neo4j_config["password"]
                    if password is not None:
                        self.neo4j_password = str(password)
                    else:
                        self.neo4j_password = ""
                    neo4j_changed = True
                if "database" in neo4j_config:
                    self.neo4j_database = neo4j_config["database"]
                    neo4j_changed = True
                
                # Reinitialize Neo4j connection if configuration changed
                if neo4j_changed:
                    if self.logger:
                        self.logger.info("Neo4j configuration changed, reinitializing connection")
                        if "password" in neo4j_config:
                            self.logger.debug(f"Using password with length: {len(self.neo4j_password)} characters")
                    # Close existing connection safely
                    try:
                        self.close()
                    except Exception as e:
                        if self.logger:
                            self.logger.warn(f"Error closing existing connection (can be ignored for initial setup): {str(e)}")
                    
                    # Initialize with new settings
                    self.base_manager.neo4j_uri = self.neo4j_uri
                    self.base_manager.neo4j_user = self.neo4j_user
                    self.base_manager.neo4j_password = self.neo4j_password
                    self.base_manager.neo4j_database = self.neo4j_database
                    
                    # The initialize method returns a boolean now
                    try:
                        if not self.base_manager.initialize():
                            if self.logger:
                                self.logger.error("Failed to initialize Neo4j with new configuration")
                            return {"status": "error", "message": "Failed to initialize Neo4j with new configuration"}
                        # Update forwarded properties from base manager
                        self._update_properties_from_base_manager()
                    except Exception as e:
                        if self.logger:
                            self.logger.error(f"Failed to initialize Neo4j with new configuration: {str(e)}")
                        return {"status": "error", "message": f"Failed to initialize Neo4j with new configuration: {str(e)}"}
            
            # Apply embedding configuration if present
            if "embeddings" in config:
                embedding_config = config.get("embeddings", {})
                self.embedding_enabled = embedding_config.get("enabled", False)
                
                if self.embedding_enabled:
                    provider = embedding_config.get("provider", "openai")
                    model = embedding_config.get("model", "text-embedding-ada-002")
                    api_key = embedding_config.get("api_key")
                    
                    self.embedder_provider = provider
                    self.embedding_model = model
                    
                    # Initialize embedding adapter if enabled and API key provided
                    if api_key:
                        self.embedding_api_key = api_key
                        adapter_success = self.embedding_adapter.init_embedding_manager(
                            api_key=api_key,
                            model_name=model
                        )
                        
                        if not adapter_success:
                            # Log failure but continue without embeddings
                            self.embedding_enabled = False
                            if self.logger:
                                self.logger.error("Failed to initialize embedding manager with client configuration")
                            return {"status": "warning", "message": "Failed to initialize embedding manager"}
                    elif self.logger:
                        self.logger.warning("Embeddings enabled but no API key provided")
                elif self.logger:
                    self.logger.info("Embeddings disabled by client configuration")
            
            return {"status": "success", "message": "Configuration applied successfully"}
            
        except Exception as e:
            if self.logger:
                self.logger.error(f"Error applying client config: {str(e)}")
            return {"status": "error", "message": f"Failed to apply configuration: {str(e)}"}
    
    def reinitialize(self) -> bool:
        """
        Reinitialize the memory manager.
        
        Returns:
            True if reinitialization successful, False otherwise
        """
        try:
            # Close existing connections
            self.close()
            
            # Reinitialize
            success = self.initialize()
            
            if success:
                return True
            else:
                return False
                
        except Exception as e:
            if self.logger:
                self.logger.error(f"Error reinitializing memory manager: {str(e)}")
            return False
    
    def get_current_config(self) -> Dict[str, Any]:
        """
        Get the current configuration.
        
        Returns:
            Dictionary with current configuration
        """
        config = {
            "project_name": self.default_project_name,
            "client_id": self._client_id,
            "timestamp": datetime.now().isoformat(),
            "neo4j": {
                "uri": self.neo4j_uri,
                "username": self.neo4j_user,
                "password": self.neo4j_password,  # Return actual password
                "database": self.neo4j_database
            }
        }
        
        # Only include detailed embedding configuration when enabled
        if self.embedding_enabled:
            config["embeddings"] = {
                "provider": self.embedder_provider,
                "model": self.embedding_model,
                "dimensions": getattr(self.embedding_adapter.embedding_manager, "dimensions", 0),
                "enabled": True
            }
        else:
            # Just indicate embeddings are disabled
            config["embeddings"] = {
                "enabled": False
            }
        
        return config
    
    def _ensure_initialized(self) -> None:
        """Ensure the memory manager is initialized."""
        self.base_manager.ensure_initialized()
    
    # Lesson Memory System methods
    
    def lesson_operation(self, operation_type: str, **kwargs) -> str:
        """
        Manage lesson memory with a unified interface
        
        This tool provides a simplified interface to the Lesson Memory System,
        allowing AI agents to store and retrieve experiential knowledge in a
        structured way.
        
        Args:
            operation_type: The type of operation to perform
              - create: Create a new lesson
              - create_container: Create a lesson container
              - get_container: Get the lesson container
              - list_containers: List all lesson containers
              - container_exists: Check if a container exists
              - observe: Add structured observations to a lesson
              - relate: Create relationships between lessons
              - search: Find relevant lessons
              - track: Track lesson application
              - consolidate: Combine related lessons
              - evolve: Track lesson knowledge evolution
              - update: Update existing lessons
            **kwargs: Operation-specific parameters
                
        Returns:
            JSON response with operation results
        """
        try:
            self._ensure_initialized()
            
            # Map operation types to handler methods
            operation_handlers = {
                # Container operations
                "create_container": self._handle_lesson_container_creation,
                "get_container": self._handle_get_lesson_container,
                "list_containers": self._handle_list_lesson_containers,
                "container_exists": self._handle_container_exists,
                
                # Lesson operations
                "create": self._handle_lesson_creation,
                "observe": self._handle_lesson_observation,
                "relate": self._handle_lesson_relationship,
                "search": self._handle_lesson_search,
                "track": self._handle_lesson_tracking,
                "consolidate": self._handle_lesson_consolidation,
                "evolve": self._handle_lesson_evolution,
                "update": self._handle_lesson_update,
            }
            
            # Check if operation type is valid
            if operation_type not in operation_handlers:
                error = ErrorDetail(
                    code="unknown_operation",
                    message=f"Unknown operation type: {operation_type}",
                    details={"available_operations": list(operation_handlers.keys())}
                )
                error_response = ErrorResponse(
                    status="error",
                    timestamp=datetime.now(),
                    error=error
                )
                return error_response.model_dump_json()
            
            # Get the appropriate handler and call it with the provided arguments
            handler = operation_handlers[operation_type]
            result = handler(**kwargs)
            
            # Standardize the response
            return self._standardize_response(
                result_json=result,
                success_message=f"Lesson operation '{operation_type}' completed successfully",
                error_code=f"lesson_{operation_type}_error"
            )
                
        except Exception as e:
            # Log and report the error
            if self.logger:
                self.logger.error(f"Error in lesson operation '{operation_type}': {str(e)}")
            
            error = ErrorDetail(
                code=f"lesson_{operation_type}_error",
                message=f"Error in lesson operation '{operation_type}': {str(e)}",
                details={"operation_args": kwargs}
            )
            error_response = ErrorResponse(
                status="error",
                timestamp=datetime.now(),
                error=error
            )
            return error_response.model_dump_json()
    
    def _handle_lesson_container_creation(self, **kwargs) -> str:
        """
        Handle creation of a lesson container.
        
        Args:
            **kwargs: Additional parameters
                - description: Optional description of the container
                - metadata: Optional metadata dictionary for the container
        
        Returns:
            JSON response string with created container data
        """
        # Initialize variables that may be used in error handling
        description = kwargs.get("description", None)
        metadata = kwargs.get("metadata", None)
        
        try:
            # Validate required fields
            if not hasattr(self, "lesson_memory") or not self.lesson_memory:
                error = ErrorDetail(
                    code="lesson_memory_not_initialized",
                    message="Lesson memory system not initialized",
                    details=None
                )
                error_response = ErrorResponse(
                    status="error",
                    timestamp=datetime.now(),
                    error=error
                )
                return error_response.model_dump_json()
            
            # Extract optional parameters
            description = kwargs.pop("description", None)
            metadata = kwargs.pop("metadata", None)
            
            # Create the container 
            result = self.lesson_memory.create_lesson_container(
                description=description,
                metadata=metadata
            )
            
            # Return standardized response
            return self._standardize_response(
                result_json=result,
                success_message="Lesson container created successfully",
                error_code="container_creation_error"
            )
            
        except Exception as e:
            if self.logger:
                self.logger.error(f"Error creating lesson container: {str(e)}")
            
            error = ErrorDetail(
                code="container_creation_error",
                message=f"Failed to create lesson container: {str(e)}",
                details={"description": description, "metadata_keys": list(metadata.keys()) if metadata else None}
            )
            error_response = ErrorResponse(
                status="error",
                timestamp=datetime.now(),
                error=error
            )
            return error_response.model_dump_json()
    
    def _handle_lesson_creation(self, name: str, lesson_type: str, **kwargs) -> str:
        """
        Handle lesson creation with proper defaults and project context.
        
        Args:
            name: Name of the lesson to create
            lesson_type: Type of lesson entity
            **kwargs: Additional parameters
                - container_name: Optional container name (default: "Lessons")
                - observations: Optional list of observations
                - metadata: Optional metadata dictionary
        
        Returns:
            JSON response string with created lesson data
        """
        # Initialize variables that may be used in error handling
        container = kwargs.get("container_name", "Lessons")
        observations = kwargs.get("observations", None)
        metadata = kwargs.get("metadata", None)
        
        try:
            # Validate required parameters
            if not name or not name.strip():
                error = ErrorDetail(
                    code="missing_lesson_name",
                    message="Lesson name is required",
                    details=None
                )
                error_response = ErrorResponse(
                    status="error",
                    timestamp=datetime.now(),
                    error=error
                )
                return error_response.model_dump_json()
                
            if not lesson_type or not lesson_type.strip():
                error = ErrorDetail(
                    code="missing_lesson_type",
                    message="Lesson type is required",
                    details=None
                )
                error_response = ErrorResponse(
                    status="error",
                    timestamp=datetime.now(),
                    error=error
                )
                return error_response.model_dump_json()
            
            # Validate lesson memory system
            if not hasattr(self, "lesson_memory") or not self.lesson_memory:
                error = ErrorDetail(
                    code="lesson_memory_not_initialized",
                    message="Lesson memory system not initialized",
                    details=None
                )
                error_response = ErrorResponse(
                    status="error",
                    timestamp=datetime.now(),
                    error=error
                )
                return error_response.model_dump_json()
            
            # Extract optional parameters with defaults
            container = kwargs.pop("container_name", "Lessons")
            observations = kwargs.pop("observations", None)
            metadata = kwargs.pop("metadata", None)
            
            # Create the lesson entity
            result = self.lesson_memory.create_lesson_entity(
                container, name, lesson_type, observations, metadata
            )
            
            # Return standardized response
            return self._standardize_response(
                result_json=result,
                success_message=f"Lesson '{name}' created successfully",
                error_code="lesson_creation_error"
            )
            
        except Exception as e:
            if self.logger:
                self.logger.error(f"Error creating lesson: {str(e)}")
            
            error = ErrorDetail(
                code="lesson_creation_error",
                message=f"Failed to create lesson: {str(e)}",
                details={
                    "name": name,
                    "type": lesson_type,
                    "container": container,
                    "has_observations": observations is not None,
                    "has_metadata": metadata is not None
                }
            )
            error_response = ErrorResponse(
                status="error",
                timestamp=datetime.now(),
                error=error
            )
            return error_response.model_dump_json()
    
    def _handle_lesson_observation(self, entity_name: str, **kwargs) -> str:
        """
        Handle adding structured observations to a lesson.
        
        Args:
            entity_name: Name of the entity to add observations to
            **kwargs: Additional parameters including observation fields:
                - what_was_learned: Optional factual knowledge
                - why_it_matters: Optional importance explanation
                - how_to_apply: Optional application guidance
                - root_cause: Optional underlying causes
                - evidence: Optional examples and data
                - container_name: Optional container name
                
        Returns:
            JSON response string with observation results
        """
        try:
            # Pre-initialize result in case of error
            result = None
            
            # Use the entity_name and pass all other kwargs directly
            kwargs["entity_name"] = entity_name
            
            # Create the structured observations
            result = self.lesson_memory.create_structured_lesson_observations(**kwargs)
            
            # Use standardized response
            return self._standardize_response(
                result_json=result,
                success_message=f"Successfully added observation to lesson '{entity_name}'",
                error_code="observation_error"
            )
            
        except Exception as e:
            if self.logger:
                self.logger.error(f"Error adding lesson observations: {str(e)}")
            
            error = ErrorDetail(
                code="observation_error",
                message=f"Failed to add lesson observations: {str(e)}",
                details={"entity_name": entity_name}
            )
            error_response = ErrorResponse(
                status="error",
                timestamp=datetime.now(),
                error=error
            )
            return error_response.model_dump_json()
    
    def _handle_lesson_relationship(self, **kwargs) -> str:
        """
        Handle creating relationships between lessons.
        
        Args:
            **kwargs: Parameters including:
                - source_name: Name of the source entity
                - target_name: Name of the target entity
                - relationship_type: Type of relationship
                - properties: Optional relationship properties
                - container_name: Optional container name
                
        Returns:
            JSON response string with relationship data
        """
        try:
            # Pre-initialize result in case of error
            result = None
            
            # Validate required parameters
            required_params = ["source_name", "target_name", "relationship_type"]
            for param in required_params:
                if param not in kwargs:
                    raise ValueError(f"Missing required parameter: {param}")
            
            source_name = kwargs.get("source_name")
            target_name = kwargs.get("target_name")
            relationship_type = kwargs.get("relationship_type")
            
            # Create the relationship
            result = self.lesson_memory.create_lesson_relationship(kwargs)
            
            # Use standardized response
            return self._standardize_response(
                result_json=result,
                success_message=f"Successfully created {relationship_type} relationship from '{source_name}' to '{target_name}'",
                error_code="relationship_error"
            )
            
        except Exception as e:
            if self.logger:
                self.logger.error(f"Error creating lesson relationship: {str(e)}")
            
            # Extract parameters for error context if available
            details = {
                "source_name": kwargs.get("source_name", "unknown"),
                "target_name": kwargs.get("target_name", "unknown"),
                "relationship_type": kwargs.get("relationship_type", "unknown")
            }
            
            error = ErrorDetail(
                code="relationship_error",
                message=f"Failed to create lesson relationship: {str(e)}",
                details=details
            )
            error_response = ErrorResponse(
                status="error",
                timestamp=datetime.now(),
                error=error
            )
            return error_response.model_dump_json()
    
    def _handle_lesson_search(self, query: Optional[str] = None, **kwargs) -> str:
        """
        Handle searching for lessons.
        
        Args:
            query: Optional search query text
            **kwargs: Additional parameters including:
                - container_name: Optional container to search within
                - entity_type: Optional entity type to filter by
                - tags: Optional tags to filter by
                - limit: Maximum number of results (default: 50)
                - semantic: Whether to use semantic search (default: True)
                
        Returns:
            JSON response string with search results
        """
        try:
            # Pre-initialize result in case of error
            result = None
            semantic_used = False
            
            # Set defaults
            kwargs.setdefault("limit", 50)
            kwargs.setdefault("semantic", True)
            
            # If query is provided, add it to kwargs
            if query is not None:
                kwargs["search_term"] = query
            
            # Use semantic search if specified and available
            if kwargs.get("semantic", False):
                try:
                    # Default to empty string if search_term is None
                    search_term = kwargs.get("search_term", "")
                    if search_term is None:
                        search_term = ""
                    
                    result = self.lesson_memory.search_lesson_semantic(
                        query=search_term,
                        limit=kwargs.get("limit", 50),
                        container_name=kwargs.get("container_name")
                    )
                    semantic_used = True
                    
                except Exception as semantic_error:
                    # Fall back to standard search if semantic search fails
                    if self.logger:
                        self.logger.warning(f"Semantic search failed, falling back to standard search: {str(semantic_error)}")
                    semantic_used = False
            
            # Use standard search (either as primary method or fallback)
            if not semantic_used or result is None:
                result = self.lesson_memory.search_lesson_entities(
                    container_name=kwargs.get("container_name"),
                    search_term=kwargs.get("search_term"),
                    entity_type=kwargs.get("entity_type"),
                    tags=kwargs.get("tags"),
                    limit=kwargs.get("limit", 50),
                    semantic=False  # We already tried semantic search above if enabled
                )
            
            # Prepare search context for the success message
            search_context = f"'{query}'" if query else "all lessons"
            container_context = f" in container '{kwargs.get('container_name')}'" if kwargs.get('container_name') else ""
            search_type = "semantic" if semantic_used else "standard"
            
            # Use standardized response
            return self._standardize_response(
                result_json=result,
                success_message=f"Successfully performed {search_type} search for {search_context}{container_context}",
                error_code="search_error"
            )
            
        except Exception as e:
            if self.logger:
                self.logger.error(f"Error searching lessons: {str(e)}")
            
            # Prepare error details
            details = {
                "query": query,
                "container_name": kwargs.get("container_name", "default"),
                "limit": kwargs.get("limit", 50),
                "semantic": kwargs.get("semantic", True)
            }
            
            error = ErrorDetail(
                code="search_error",
                message=f"Failed to search lessons: {str(e)}",
                details=details
            )
            error_response = ErrorResponse(
                status="error",
                timestamp=datetime.now(),
                error=error
            )
            return error_response.model_dump_json()
    
    def _handle_lesson_tracking(self, lesson_name: str, context_entity: str, **kwargs) -> str:
        """
        Handle tracking lesson application to context entities.
        
        Args:
            lesson_name: Name of the lesson being applied
            context_entity: Name of the entity the lesson is being applied to
            **kwargs: Additional parameters including:
                - success_score: Score indicating success of application (0.0-1.0, default: 0.8)
                - application_notes: Optional notes about the application
                
        Returns:
            JSON response string with tracking results
        """
        try:
            # Pre-initialize result in case of error
            result = None
            
            # Get optional parameters with defaults
            success_score = kwargs.get("success_score", 0.8)
            application_notes = kwargs.get("application_notes")
            
            # Track the lesson application
            result = self.lesson_memory.track_lesson_application(
                lesson_name=lesson_name,
                context_entity=context_entity,
                success_score=success_score,
                application_notes=application_notes
            )
            
            # Use standardized response
            score_text = f" with success score {success_score}" if success_score is not None else ""
            return self._standardize_response(
                result_json=result,
                success_message=f"Successfully tracked application of lesson '{lesson_name}' to '{context_entity}'{score_text}",
                error_code="tracking_error"
            )
            
        except Exception as e:
            if self.logger:
                self.logger.error(f"Error tracking lesson application: {str(e)}")
            
            # Prepare error details
            details = {
                "lesson_name": lesson_name,
                "context_entity": context_entity,
                "success_score": kwargs.get("success_score", 0.8)
            }
            
            error = ErrorDetail(
                code="tracking_error",
                message=f"Failed to track lesson application: {str(e)}",
                details=details
            )
            error_response = ErrorResponse(
                status="error",
                timestamp=datetime.now(),
                error=error
            )
            return error_response.model_dump_json()
    
    def _handle_lesson_consolidation(self, source_lessons: List[str], new_name: str, **kwargs) -> str:
        """
        Handle consolidating multiple lessons into a single consolidated lesson.
        
        Args:
            source_lessons: List of lesson IDs or names to merge
            new_name: Name for the new consolidated lesson
            **kwargs: Additional parameters including:
                - merge_strategy: Strategy for merging ('union', 'intersection', 'latest', default: 'union')
                - container_name: Optional container name for the new lesson
                
        Returns:
            JSON response string with the consolidated lesson
        """
        try:
            # Pre-initialize result in case of error
            result = None
            
            # Get optional parameters with defaults
            merge_strategy = kwargs.get("merge_strategy", "union")
            container_name = kwargs.get("container_name")
            
            # Convert source_lessons to list if it's a single string
            if isinstance(source_lessons, str):
                source_lessons = [source_lessons]
                
            # Handle both list of strings and list of dicts
            processed_sources = []
            for lesson in source_lessons:
                if isinstance(lesson, dict):
                    processed_sources.append(lesson)
                else:
                    processed_sources.append({"id": lesson})
            
            # Merge the lessons
            result = self.lesson_memory.merge_lessons(
                source_lessons=processed_sources,
                new_name=new_name,
                merge_strategy=merge_strategy,
                container_name=container_name
            )
            
            # Format source lesson names for message
            source_names = []
            for lesson in source_lessons:
                if isinstance(lesson, dict) and "name" in lesson:
                    source_names.append(lesson.get("name"))
                elif isinstance(lesson, dict) and "id" in lesson:
                    source_names.append(lesson.get("id"))
                else:
                    source_names.append(str(lesson))
            
            source_text = ", ".join(f"'{name}'" for name in source_names)
            
            # Use standardized response
            return self._standardize_response(
                result_json=result,
                success_message=f"Successfully consolidated lessons [{source_text}] into new lesson '{new_name}' using {merge_strategy} strategy",
                error_code="consolidation_error"
            )
            
        except Exception as e:
            if self.logger:
                self.logger.error(f"Error consolidating lessons: {str(e)}")
            
            # Prepare error details
            details = {
                "source_lessons": [str(lesson) for lesson in source_lessons],
                "new_name": new_name,
                "merge_strategy": kwargs.get("merge_strategy", "union"),
                "container_name": kwargs.get("container_name")
            }
            
            error = ErrorDetail(
                code="consolidation_error",
                message=f"Failed to consolidate lessons: {str(e)}",
                details=details
            )
            error_response = ErrorResponse(
                status="error",
                timestamp=datetime.now(),
                error=error
            )
            return error_response.model_dump_json()
    
    def _handle_lesson_evolution(self, old_lesson: str, new_lesson: str, **kwargs) -> str:
        """
        Handle tracking when a new lesson supersedes an older one.
        
        Args:
            old_lesson: Name of the lesson being superseded
            new_lesson: Name of the new lesson
            **kwargs: Additional parameters including:
                - reason: Optional reason for the supersession
                
        Returns:
            JSON response string with the created relationship
        """
        try:
            # Pre-initialize result in case of error
            result = None
            
            # Get optional parameters
            reason = kwargs.get("reason")
            
            # Track the supersession
            result = self.lesson_memory.track_lesson_supersession(
                old_lesson=old_lesson,
                new_lesson=new_lesson,
                reason=reason
            )
            
            # Use standardized response
            reason_text = f" (Reason: {reason})" if reason else ""
            return self._standardize_response(
                result_json=result,
                success_message=f"Successfully tracked evolution from '{old_lesson}' to '{new_lesson}'{reason_text}",
                error_code="evolution_error"
            )
            
        except Exception as e:
            if self.logger:
                self.logger.error(f"Error tracking lesson evolution: {str(e)}")
            
            # Prepare error details
            details = {
                "old_lesson": old_lesson,
                "new_lesson": new_lesson,
                "reason": kwargs.get("reason")
            }
            
            error = ErrorDetail(
                code="evolution_error",
                message=f"Failed to track lesson evolution: {str(e)}",
                details=details
            )
            error_response = ErrorResponse(
                status="error",
                timestamp=datetime.now(),
                error=error
            )
            return error_response.model_dump_json()
    
    def _handle_lesson_update(self, entity_name: str, updates: Dict[str, Any], **kwargs) -> str:
        """
        Handle updating an existing lesson entity.
        
        Args:
            entity_name: Name of the entity to update
            updates: Dictionary of fields to update
            **kwargs: Additional parameters including:
                - container_name: Optional container name to verify entity membership
                
        Returns:
            JSON response string with the updated entity
        """
        try:
            # Pre-initialize result in case of error
            result = None
            
            # Get optional parameters
            container_name = kwargs.get("container_name")
            
            # Update the lesson entity
            result = self.lesson_memory.update_lesson_entity(
                entity_name=entity_name,
                updates=updates,
                container_name=container_name
            )
            
            # Format update fields for message
            update_fields = list(updates.keys())
            update_text = ", ".join([f"'{field}'" for field in update_fields[:3]])
            if len(update_fields) > 3:
                update_text += f" and {len(update_fields) - 3} more fields"
            
            # Use standardized response
            container_text = f" in container '{container_name}'" if container_name else ""
            return self._standardize_response(
                result_json=result,
                success_message=f"Successfully updated lesson '{entity_name}'{container_text} with fields: {update_text}",
                error_code="update_error"
            )
            
        except Exception as e:
            if self.logger:
                self.logger.error(f"Error updating lesson: {str(e)}")
            
            # Prepare error details
            details = {
                "entity_name": entity_name,
                "update_fields": list(updates.keys()),
                "container_name": kwargs.get("container_name")
            }
            
            error = ErrorDetail(
                code="update_error",
                message=f"Failed to update lesson: {str(e)}",
                details=details
            )
            error_response = ErrorResponse(
                status="error",
                timestamp=datetime.now(),
                error=error
            )
            return error_response.model_dump_json()
    
    def _handle_get_lesson_container(self, **kwargs) -> str:
        """
        Get the lesson container.
        
        Args:
            **kwargs: Not used
            
        Returns:
            JSON string with container data
        """
        try:
            # Ensure lesson memory manager is available
            if not hasattr(self, "lesson_memory") or not self.lesson_memory:
                if self.logger:
                    self.logger.error("Lesson memory manager not initialized")
                return json.dumps({
                    "status": "error",
                    "error": "Lesson memory manager not initialized",
                    "code": "lesson_memory_not_initialized"
                })
                
            # Call the get_lesson_container method
            if self.logger:
                self.logger.info("Calling get_lesson_container method")
                
            result = self.lesson_memory.get_lesson_container()
            
            if self.logger:
                self.logger.info(f"Result from get_lesson_container: {result}")
                
            # Parse the result to handle when no container exists
            if isinstance(result, str):
                result_data = json.loads(result)
            else:
                result_data = result
                
            if self.logger:
                self.logger.info(f"Parsed result data: {result_data}")
                
            # Check if the result is empty or has null data (indicating no container)
            if "error" in result_data:
                # Check if this is actually a "not found" error or a real error
                error_msg = result_data.get("error", "").lower()
                if self.logger:
                    self.logger.info(f"Error message in result: {error_msg}")
                    
                if "not found" in error_msg or "doesn't exist" in error_msg:
                    # This is specifically a container not found error
                    if self.logger:
                        self.logger.info("Container not found based on error message")
                    return json.dumps({
                        "status": "error",
                        "error": "No lesson container found",
                        "code": "container_not_found",
                        "exists": False
                    })
                else:
                    # This is some other error
                    if self.logger:
                        self.logger.info(f"Other error detected: {error_msg}")
                    return json.dumps({
                        "status": "error",
                        "error": result_data.get("error", "Unknown error"),
                        "code": "container_error"
                    })
            elif result_data.get("container"):
                # Container exists and has data
                if self.logger:
                    self.logger.info(f"Container found: {result_data.get('container')}")
                return json.dumps({
                    "status": "success",
                    "data": result_data,
                    "exists": True
                })
            elif result_data.get("status") == "success":
                # Status is success but no container field
                if self.logger:
                    self.logger.info("Success status but no container field")
                return json.dumps({
                    "status": "success",
                    "data": result_data,
                    "exists": True
                })
            
            # Handle empty or unexpected response format
            if self.logger:
                self.logger.warning(f"Unexpected response format: {result_data}")
            return json.dumps({
                "status": "error",
                "error": "No lesson container found or unexpected response format",
                "code": "container_not_found",
                "exists": False
            })
                
        except Exception as e:
            if self.logger:
                self.logger.error(f"Error getting lesson container: {str(e)}")
            return json.dumps({
                "status": "error",
                "error": f"Failed to get lesson container: {str(e)}",
                "code": "container_get_error"
            })
    
    def _handle_list_lesson_containers(self, limit: int = 100, sort_by: str = "created", **kwargs) -> str:
        """
        List all lesson containers.
        
        Args:
            limit: Maximum number of containers to return
            sort_by: Field to sort results by
            **kwargs: Additional arguments (not used)
            
        Returns:
            JSON string with list of containers
        """
        try:
            # Ensure lesson memory manager is available
            if not hasattr(self, "lesson_memory") or not self.lesson_memory:
                if self.logger:
                    self.logger.error("Lesson memory manager not initialized")
                return json.dumps({
                    "status": "error",
                    "error": "Lesson memory manager not initialized",
                    "code": "lesson_memory_not_initialized"
                })
                
            # Call the list_lesson_containers method
            result = self.lesson_memory.list_lesson_containers(limit, sort_by)
            
            if self.logger:
                self.logger.info(f"Result from list_lesson_containers: {result}")
                
            # Parse the result 
            if isinstance(result, str):
                try:
                    result_data = json.loads(result)
                except json.JSONDecodeError:
                    if self.logger:
                        self.logger.error(f"Invalid JSON response: {result}")
                    result_data = result
            else:
                result_data = result
                
            if self.logger:
                self.logger.info(f"Parsed result data: {result_data}")
                
            # Process the result to ensure consistent response format
            if isinstance(result_data, dict):
                # Check if this is a success response
                if result_data.get("status") == "success":
                    # Extract containers from data field if present
                    containers_data = result_data.get("data", {}).get("containers", [])
                    
                    # Return standardized response
                    return json.dumps({
                        "status": "success",
                        "message": result_data.get("message", "Successfully listed containers"),
                        "containers": containers_data
                    }, default=str)
                    
                # Check if we have an error response
                elif result_data.get("status") == "error" or "error" in result_data:
                    error_msg = result_data.get("error", result_data.get("message", "Unknown error"))
                    if self.logger:
                        self.logger.error(f"Error in list containers response: {error_msg}")
                    return json.dumps({
                        "status": "error",
                        "error": error_msg,
                        "code": result_data.get("code", "container_list_error")
                    })
                    
            # Fall back to returning the raw result if format is unexpected
            if isinstance(result, str):
                return result
            else:
                return json.dumps(result_data, default=str)
                
        except Exception as e:
            if self.logger:
                self.logger.error(f"Error listing lesson containers: {str(e)}")
            return json.dumps({
                "status": "error",
                "error": f"Failed to list lesson containers: {str(e)}",
                "code": "container_list_error"
            })
    
    def _handle_container_exists(self, container_name: str = "Lessons", **kwargs) -> str:
        """
        Check if a lesson container exists.
        
        Args:
            container_name: Name of the container to check
            **kwargs: Additional arguments (not used)
            
        Returns:
            JSON string with existence status
        """
        try:
            # Ensure lesson memory manager is available
            if not hasattr(self, "lesson_memory") or not self.lesson_memory:
                return json.dumps({
                    "status": "error",
                    "error": "Lesson memory manager not initialized",
                    "code": "lesson_memory_not_initialized"
                })
            
            # We'll try two approaches:
            # 1. First, direct Cypher query to check for container existence
            exists_query = """
            MATCH (c:LessonContainer {name: $name})
            RETURN count(c) > 0 as exists
            """
            
            try:
                # Direct database query to check container existence
                records = self.lesson_memory.base_manager.safe_execute_read_query(
                    exists_query,
                    {"name": container_name}
                )
                
                if records and len(records) > 0 and records[0].get("exists", False) is True:
                    # Container exists based on direct query
                    return json.dumps({
                        "status": "success",
                        "exists": True,
                        "container_name": container_name,
                        "message": f"Container '{container_name}' exists"
                    })
            except Exception as e:
                # If this fails, we'll try the second approach
                if self.logger:
                    self.logger.warning(f"Direct query for container existence failed: {str(e)}")
            
            # 2. Fallback: Use get_lesson_container method to check existence
            result = self.lesson_memory.get_lesson_container()
            
            # Parse the result
            if isinstance(result, str):
                result_data = json.loads(result)
            else:
                result_data = result
                
            # Check for container existence
            exists = False
            
            # First check if the status is success
            if result_data.get("status") == "success":
                exists = True
            # Next look for container data
            elif "container" in result_data:
                exists = True
            # Check if there's no explicit error about container not found
            elif "error" in result_data:
                error_msg = result_data.get("error", "").lower()
                exists = not ("not found" in error_msg or "doesn't exist" in error_msg)
            
            # Return a standardized response
            return json.dumps({
                "status": "success",
                "exists": exists,
                "container_name": container_name,
                "message": f"Container '{container_name}' {'exists' if exists else 'does not exist'}"
            })
                
        except Exception as e:
            if self.logger:
                self.logger.error(f"Error checking container existence: {str(e)}")
            return json.dumps({
                "status": "error",
                "error": f"Failed to check container existence: {str(e)}",
                "code": "container_check_error"
            })

    @contextmanager
    def lesson_context(self, project_name: Optional[str] = None, container_name: Optional[str] = None):
        """
        Create a context for batch lesson memory operations.
        
        This tool returns a context object that can be used for multiple
        lesson operations with shared project and container context.
        
        Args:
            project_name: Optional. Project name to set as context
            container_name: Optional. Container name to use for operations (defaults to "Lessons")
                
        Returns:
            JSON response with context information that includes:
            - status: "success" or "error"
            - message or error: Description of result or error
            - context: Context object with project_name, container_name, created_at timestamp,
                      available operations, and usage instructions
        
        Response structure:
            ```json
            {
                "status": "success",
                "message": "Lesson memory context created for project 'ProjectName' and container 'ContainerName'",
                "context": {
                    "project_name": "ProjectName",
                    "container_name": "ContainerName",
                    "created_at": "2023-07-15T10:30:45.123456",
                    "operations_available": ["create_container", "get_container", "list_containers", "container_exists", "create", "observe", "relate", "search", "track", "update", "consolidate", "evolve"],
                    "usage": "Use this context information with any lesson memory operation by including it in the operation's context parameter"
                }
            }
            ```
             
        The returned context object has these methods:
            - create_container(): Create a new lesson container
            - get_container(): Get the lesson container
            - list_containers(): List all lesson containers
            - container_exists(): Check if a container exists
            - create(): Create a new lesson
            - observe(): Add structured observations to lessons
            - relate(): Create relationships between lessons
            - search(): Find relevant lessons
            - track(): Track lesson application
            - consolidate(): Combine related lessons
            - update(): Update existing lessons
            
        Example:
            ```
            # Create a context for a specific project and container
            context = @lesson_memory_context({
                "project_name": "E-commerce Refactoring",
                "container_name": "PerformanceLessons"
            })
            
            # Use the context with another tool
            result = @lesson_memory_tool({
                "operation_type": "search",
                "query": "database optimization patterns",
                "context": context["context"]  # Pass the context object from the response
            })
            
            # Using container operations
            with memory.lesson_context(project_name="Project1", container_name="LessonContainer") as context:
                # Check if container exists
                container_exists = context.container_exists("LessonContainer")
                # Get container details
                container_details = context.get_container()
                # Create a new lesson
                context.create(name="Lesson1", lesson_type="BestPractice")
            ```
        """
        from src.models.response_models import LessonContextModel, LessonContextResponse
        
        self._ensure_initialized()
        
        # Save current project
        original_project = self.default_project_name
        
        try:
            # Set project context if provided
            if project_name is not None:
                self.set_project_name(project_name)
            
            # Create context model for tracking
            context_model = LessonContextModel(
                container_name=container_name,
                project_name=project_name or self.default_project_name,
                created_at=datetime.now()
            )
            
            # Log context creation
            if self.logger:
                self.logger.debug(f"Created lesson context with project={context_model.project_name}, container={context_model.container_name}")
            
            # Create and yield context helper
            context = LessonContext(self, container_name)
            
            # Attach context model to the context object for reference
            context._context_model = context_model
            
            yield context
            
        finally:
            # Restore original project context
            if project_name is not None and original_project != project_name:
                # Make sure we don't pass None
                self.set_project_name(original_project or "")
    
    # Project Memory System methods
    
    def project_operation(self, operation_type: str, **kwargs) -> str:
        """
        Manage project memory with a unified interface.
        
        This method provides a simplified interface to the Project Memory System,
        allowing for structured organization of project knowledge.
        
        Args:
            operation_type: The type of operation to perform
              - create_project: Create a new project
              - create_component: Create a component within a project
              - create_domain: Create a domain within a project
              - create_domain_entity: Create an entity within a domain
              - relate: Create relationship between entities
              - search: Search for project entities
              - get_structure: Get project structure
              - add_observation: Add observation to an entity
              - update: Update entity properties
              - delete_entity: Delete an entity
              - delete_relationship: Delete a relationship
            **kwargs: Operation-specific parameters
        
        Returns:
            JSON response with operation results
        """
        # Initialize variables that may be used in error handling
        project_id = kwargs.get("project_id", kwargs.get("container_name", self.default_project_name))
        
        try:
            self._ensure_initialized()
            
            # Map operation types to handler methods
            operation_handlers = {
                # Container operations
                "create_project": self._handle_project_creation,
                
                # Component operations
                "create_component": self._handle_component_creation,
                # Replace _handle_domain_creation with _handle_domain_entity_creation 
                "create_domain": self._handle_domain_entity_creation,  
                "create_domain_entity": self._handle_domain_entity_creation,
                
                # Relationship operations
                "relate": self._handle_entity_relationship,
                
                # Search operations
                "search": self._handle_project_search,
                "get_structure": self._handle_structure_retrieval,
                
                # Observation operations
                "add_observation": self._handle_add_observation,
                
                # Update operations
                "update": self._handle_entity_update,
                
                # Delete operations
                "delete_entity": self._handle_entity_deletion,
                "delete_relationship": self._handle_relationship_deletion,
            }
            
            # Check if operation type is valid
            if operation_type not in operation_handlers:
                error = ErrorDetail(
                    code="unknown_operation",
                    message=f"Unknown project operation type: {operation_type}",
                    details={"available_operations": list(operation_handlers.keys())}
                )
                error_response = ErrorResponse(
                    status="error",
                    timestamp=datetime.now(),
                    error=error
                )
                return error_response.model_dump_json()
            
            # Get the handler for this operation
            handler = operation_handlers[operation_type]
            
            # Special handling for different operation types
            if operation_type == "create_project":
                # Validate project name
                if "name" not in kwargs:
                    error = ErrorDetail(
                        code="missing_project_name",
                        message="Project name is required for create_project operation",
                        details=None
                    )
                    error_response = ErrorResponse(
                        status="error",
                        timestamp=datetime.now(),
                        error=error
                    )
                    return error_response.model_dump_json()
                    
                # Call the handler with name parameter
                result = handler(name=kwargs.pop("name"), **kwargs)
                
            elif operation_type in ["create_component", "create_domain_entity"]:
                # Validate required parameters
                if "name" not in kwargs:
                    error = ErrorDetail(
                        code="missing_entity_name",
                        message=f"Entity name is required for {operation_type} operation",
                        details=None
                    )
                    error_response = ErrorResponse(
                        status="error",
                        timestamp=datetime.now(),
                        error=error
                    )
                    return error_response.model_dump_json()
                    
                # For component creation, require component_type
                if operation_type == "create_component" and "component_type" not in kwargs:
                    error = ErrorDetail(
                        code="missing_component_type",
                        message="Component type is required for create_component operation",
                        details=None
                    )
                    error_response = ErrorResponse(
                        status="error",
                        timestamp=datetime.now(),
                        error=error
                    )
                    return error_response.model_dump_json()
                
                # Add project_id if not provided
                if "project_id" not in kwargs:
                    kwargs["project_id"] = project_id
                    
                # Call the appropriate handler
                result = handler(**kwargs)
                
            elif operation_type == "relate":
                # Validate relationship parameters
                if "source_name" not in kwargs or "target_name" not in kwargs or "relation_type" not in kwargs:
                    error = ErrorDetail(
                        code="missing_relationship_parameters",
                        message="source_name, target_name, and relation_type are required for relate operation",
                        details={"provided_params": list(kwargs.keys())}
                    )
                    error_response = ErrorResponse(
                        status="error",
                        timestamp=datetime.now(),
                        error=error
                    )
                    return error_response.model_dump_json()
                    
                # Call the relationship handler
                result = handler(
                    source_name=kwargs.pop("source_name"),
                    target_name=kwargs.pop("target_name"),
                    relation_type=kwargs.pop("relation_type"),
                    **kwargs
                )
                
            else:
                # For other operations, just call the handler directly
                result = handler(**kwargs)
            
            # Standardize the response
            return self._standardize_response(
                result_json=result,
                success_message=f"Project operation '{operation_type}' completed successfully",
                error_code=f"project_{operation_type}_error"
            )
            
        except Exception as e:
            # Log and report the error
            if self.logger:
                self.logger.error(f"Error in project operation '{operation_type}': {str(e)}")
            
            error = ErrorDetail(
                code=f"project_{operation_type}_error",
                message=f"Error in project operation '{operation_type}': {str(e)}",
                details={
                    "project_id": project_id,
                    "operation_args": kwargs
                }
            )
            error_response = ErrorResponse(
                status="error",
                timestamp=datetime.now(),
                error=error
            )
            return error_response.model_dump_json()
    
    def _handle_project_creation(self, name: str, **kwargs) -> str:
        """
        Handle project container creation.
        
        Args:
            name: Name of the project to create
            **kwargs: Additional parameters
                - description: Optional description of the project
                - metadata: Optional metadata dictionary
                - tags: Optional list of tags for the project
        
        Returns:
            JSON response string with created project data
        """
        # Initialize variables that may be used in error handling
        description = kwargs.get("description", None)
        metadata = kwargs.get("metadata", None)
        tags = kwargs.get("tags", None)
        
        try:
            # Validate required parameters
            if not name or not name.strip():
                error = ErrorDetail(
                    code="missing_project_name",
                    message="Project name is required",
                    details=None
                )
                error_response = ErrorResponse(
                    status="error",
                    timestamp=datetime.now(),
                    error=error
                )
                return error_response.model_dump_json()
            
            # Validate project memory system
            if not hasattr(self, "project_memory") or not self.project_memory:
                error = ErrorDetail(
                    code="project_memory_not_initialized",
                    message="Project memory system not initialized",
                    details=None
                )
                error_response = ErrorResponse(
                    status="error",
                    timestamp=datetime.now(),
                    error=error
                )
                return error_response.model_dump_json()
            
            # Prepare project data
            project_data = {
                "name": name
            }
            
            # Add optional parameters if provided
            if description is not None:
                project_data["description"] = description
            if metadata is not None:
                project_data["metadata"] = metadata
            if tags is not None:
                project_data["tags"] = tags
                
            # Create the project
            result = self.project_memory.create_project_container(project_data)
            
            # Return standardized response
            return self._standardize_response(
                result_json=result,
                success_message=f"Project '{name}' created successfully",
                error_code="project_creation_error"
            )
            
        except Exception as e:
            if self.logger:
                self.logger.error(f"Error creating project container: {str(e)}")
            
            error = ErrorDetail(
                code="project_creation_error",
                message=f"Failed to create project: {str(e)}",
                details={
                    "name": name,
                    "has_description": description is not None,
                    "has_metadata": metadata is not None,
                    "has_tags": tags is not None
                }
            )
            error_response = ErrorResponse(
                status="error",
                timestamp=datetime.now(),
                error=error
            )
            return error_response.model_dump_json()
    
    def _handle_component_creation(self, name: str, component_type: str, project_id: str, **kwargs) -> str:
        """
        Create a component within a project
        
        This method creates a functional component (service, library, or module) that represents 
        an architectural building block of the project's system.
        
        Args:
            name: Name of the component to create
            component_type: Type of component to create
              - MICROSERVICE: Independent service with its own deployment
              - LIBRARY: Reusable code package
              - MODULE: Component within a larger system
              - API: Interface for other components
              - UI: User interface component
            project_id: ID or name of the project container
            **kwargs: Additional parameters
                
        Returns:
            JSON response with operation results
            
        Required parameters:
            - name: Component name
            - component_type: Component classification/category
            - project_id: Project container identifier
            
        Optional parameters:
            - domain_name: Name of the domain within the project
            - description: Description of the component
            - content: Content or notes about the component
            - metadata: Dictionary with additional attributes
            
        Response format:
            All operations return a JSON string with at minimum:
            - status: "success" or "error"
            - message or error: Description of result or error
            - component: Component data if successful

        Example:
            ```
            # Create a microservice component
            result = self._handle_component_creation(
                name="Authentication Service",
                component_type="MICROSERVICE",
                project_id="E-commerce Platform",
                domain_name="Backend",
                description="Handles user authentication and authorization"
            )
            ```
        """
        # Pre-initialize variables that may be used in error handling
        domain_name = None
        description = None
        content = None
        metadata = None
        
        try:
            # Extract optional parameters with defaults
            domain_name = kwargs.pop("domain_name", None)
            description = kwargs.pop("description", None)
            content = kwargs.pop("content", None)
            metadata = kwargs.pop("metadata", None)
            
            # Create the component using project_memory manager
            result = self.project_memory.create_project_component(
                name=name,
                component_type=component_type,
                domain_name=domain_name,
                container_name=project_id,
                description=description,
                content=content,
                metadata=metadata
            )
            
            # Prepare context for success message
            domain_text = f" in domain '{domain_name}'" if domain_name else ""
            
            # Use standardized response
            return self._standardize_response(
                result_json=result,
                success_message=f"Successfully created {component_type} component '{name}'{domain_text} in project '{project_id}'",
                error_code="component_creation_error"
            )
            
        except Exception as e:
            if self.logger:
                self.logger.error(f"Error creating component: {str(e)}")
            
            # Prepare error details
            details = {
                "name": name,
                "component_type": component_type,
                "project_id": project_id,
                "domain_name": kwargs.get("domain_name"),
                "has_description": description is not None,
                "has_content": content is not None,
                "has_metadata": metadata is not None
            }
            
            error = ErrorDetail(
                code="component_creation_error",
                message=f"Failed to create component: {str(e)}",
                details=details
            )
            error_response = ErrorResponse(
                status="error",
                timestamp=datetime.now(),
                error=error
            )
            return error_response.model_dump_json()
    
    def _handle_domain_entity_creation(self, name: str, entity_type: str, project_id: str, **kwargs) -> str:
        """
        Create a domain entity within a project
        
        This method creates a knowledge entity (decision, requirement, or constraint) that 
        represents project-specific knowledge rather than functional components.
        
        Args:
            name: Name of the domain entity to create
            entity_type: Type of domain entity to create
              - DECISION: Architectural or design decision
              - REQUIREMENT: System requirement or specification
              - CONSTRAINT: Limitation or boundary condition
              - PROCESS: Business or development process
              - DOMAIN: Knowledge domain within the project
            project_id: ID or name of the project container
            **kwargs: Additional parameters
                
        Returns:
            JSON response with operation results
            
        Required parameters:
            - name: Domain entity name
            - entity_type: Entity classification/category
            - project_id: Project container identifier
            
        Optional parameters:
            - description: Description of the entity
            - properties: Dictionary with additional attributes
            
        Response format:
            All operations return a JSON string with at minimum:
            - status: "success" or "error"
            - message or error: Description of result or error
            - entity: Domain entity data if successful

        Example:
            ```
            # Create a decision entity
            result = self._handle_domain_entity_creation(
                name="Use JWT for Auth",
                entity_type="DECISION",
                project_id="E-commerce Platform",
                description="Decision to use JWT for authentication"
            )
            ```
        """
        try:
            # Extract optional parameters with defaults
            description = kwargs.pop("description", None)
            properties = kwargs.pop("properties", None)
            
            if entity_type.lower() == 'domain':
                # Create a domain
                result = self.project_memory.create_project_domain(
                    name=name,
                    container_name=project_id,
                    description=description,
                    properties=properties
                )
            else:
                # Create a domain entity
                # This is a placeholder until we add more specific domain entity types
                # For now, we'll create a domain with the entity type as metadata
                domain_properties = properties or {}
                domain_properties["entity_type"] = entity_type
                
                result = self.project_memory.create_project_domain(
                    name=name,
                    container_name=project_id,
                    description=description,
                    properties=domain_properties
                )
            
            # Handle different return types (future-proof)
            if isinstance(result, str):
                return result
            else:
                return json.dumps(result)
            
        except Exception as e:
            if self.logger:
                self.logger.error(f"Error creating domain entity: {str(e)}")
            return json.dumps({
                "status": "error",
                "error": f"Failed to create domain entity: {str(e)}",
                "code": "domain_entity_creation_error"
            })
    
    def _handle_entity_relationship(self, source_name: str, target_name: str, relation_type: str, **kwargs) -> str:
        """
        Create a relationship between project entities
        
        This method establishes connections between different entities within a project,
        such as dependencies between components or relationships between domain entities.
        
        Args:
            source_name: Name of the source entity
            target_name: Name of the target entity
            relation_type: Type of relationship to create
              - DEPENDS_ON: Dependency relationship
              - IMPLEMENTS: Implementation relationship
              - CONTAINS: Containment relationship
              - USES: Usage relationship
              - EXTENDS: Extension relationship
            **kwargs: Additional parameters
                
        Returns:
            JSON response with operation results
            
        Required parameters:
            - source_name: Source entity identifier
            - target_name: Target entity identifier
            - relation_type: Type of relationship
            - project_id: Project container identifier
            
        Optional parameters:
            - domain_name: Domain name if entities are in the same domain
            - entity_type: Type of entities ('component', 'domain', 'dependency')
            - properties: Dictionary with additional relationship attributes
            
        Response format:
            All operations return a JSON string with at minimum:
            - status: "success" or "error"
            - message or error: Description of result or error
            - relationship: Relationship data if successful

        Example:
            ```
            # Create a dependency relationship between components
            result = self._handle_entity_relationship(
                source_name="Frontend",
                target_name="Authentication Service",
                relation_type="DEPENDS_ON",
                project_id="E-commerce Platform",
                domain_name="Architecture",
                entity_type="component"
            )
            ```
        """
        try:
            # Extract required parameters
            project_id = kwargs.pop("project_id", None)
            if not project_id:
                raise ValueError("project_id is required for entity relationships")
                
            # Extract optional parameters
            domain_name = kwargs.pop("domain_name", None)
            entity_type = kwargs.pop("entity_type", "component").lower()
            properties = kwargs.pop("properties", None)
            
            # Determine which relationship creation method to use based on entity_type
            if entity_type == "domain":
                # Create relationship between domains
                result = self.project_memory.create_project_domain_relationship(
                    from_domain=source_name,
                    to_domain=target_name,
                    container_name=project_id,
                    relation_type=relation_type,
                    properties=properties
                )
            elif entity_type == "component" and domain_name:
                # Create relationship between components in the same domain
                result = self.project_memory.create_project_component_relationship(
                    from_component=source_name,
                    to_component=target_name,
                    domain_name=domain_name,
                    container_name=project_id,
                    relation_type=relation_type,
                    properties=properties
                )
            elif entity_type == "dependency" and domain_name:
                # Create a dependency relationship between components
                result = self.project_memory.create_project_dependency(
                    from_component=source_name,
                    to_component=target_name,
                    domain_name=domain_name,
                    container_name=project_id,
                    dependency_type=relation_type,
                    properties=properties
                )
            else:
                # Handle unsupported entity type or missing domain_name
                error_msg = "Unsupported entity type or missing domain_name"
                if self.logger:
                    self.logger.error(error_msg)
                return json.dumps({
                    "status": "error",
                    "error": error_msg,
                    "code": "invalid_entity_type"
                })
            
            # Handle different return types (future-proof)
            if isinstance(result, str):
                return result
            else:
                return json.dumps(result)
            
        except Exception as e:
            if self.logger:
                self.logger.error(f"Error creating entity relationship: {str(e)}")
            return json.dumps({
                "status": "error",
                "error": f"Failed to create entity relationship: {str(e)}",
                "code": "relationship_creation_error"
            })
    
    def _handle_project_search(self, query: str, project_id: str, **kwargs) -> str:
        """
        Search for entities within a project
        
        This method searches for components, domains, and other entities within a project
        using keywords, semantic matching, or entity type filtering.
        
        Args:
            query: The search term
            project_id: ID or name of the project container
            **kwargs: Additional parameters
                
        Returns:
            JSON response with operation results
            
        Required parameters:
            - query: Search term or phrase
            - project_id: Project container identifier
            
        Optional parameters:
            - entity_types: List of entity types to filter by (e.g., ['component', 'domain', 'decision'])
            - limit: Maximum number of results to return (default: 10)
            - semantic: Boolean flag to enable semantic search (default: False)
            - domain_name: Domain name to limit search scope
            
        Response format:
            All operations return a JSON string with at minimum:
            - status: "success" or "error"
            - message or error: Description of result or error
            - results: List of matching entities if successful

        Example:
            ```
            # Search for authentication-related components
            result = self._handle_project_search(
                query="authentication",
                project_id="E-commerce Platform",
                entity_types=["component"],
                limit=5,
                semantic=True
            )
            ```
        """
        # Pre-initialize variables that may be used in error handling
        entity_types = None
        limit = 10
        semantic = False
        domain_name = None
        result = None
        
        try:
            # Extract optional parameters with defaults
            entity_types = kwargs.pop("entity_types", None)
            limit = kwargs.pop("limit", 10)
            semantic = kwargs.pop("semantic", False)
            domain_name = kwargs.pop("domain_name", None)
            
            # Determine which search method to use
            if domain_name and semantic:
                # Semantic search within a specific domain 
                # (This is a placeholder - no direct domain-scoped semantic search in current API)
                # For now, we'll use semantic search and filter results by domain
                result_json = self.project_memory.semantic_search_project(
                    search_term=query,
                    container_name=project_id,
                    entity_types=entity_types,
                    limit=limit
                )
                
                # Parse results to filter by domain
                try:
                    result_data = json.loads(result_json)
                    if "data" in result_data and "entities" in result_data["data"]:
                        # Filter entities by domain
                        filtered_entities = []
                        for entity in result_data["data"]["entities"]:
                            # Check if entity belongs to specified domain
                            if entity.get("domain") == domain_name:
                                filtered_entities.append(entity)
                        
                        # Replace entities with filtered list
                        result_data["data"]["entities"] = filtered_entities
                        result_data["data"]["total_count"] = len(filtered_entities)
                        result = result_data
                    else:
                        result = result_data
                except json.JSONDecodeError:
                    # If parsing fails, return original result
                    result = result_json
                
            elif domain_name:
                # Regular search within a specific domain
                # (This is a placeholder - API may need to be extended)
                # For now, we'll use project-wide search and filter results by domain
                result_json = self.project_memory.search_project_entities(
                    search_term=query,
                    container_name=project_id,
                    entity_types=entity_types,
                    limit=limit,
                    semantic=False
                )
                
                # Parse results to filter by domain
                try:
                    result_data = json.loads(result_json)
                    if "data" in result_data and "entities" in result_data["data"]:
                        # Filter entities by domain
                        filtered_entities = []
                        for entity in result_data["data"]["entities"]:
                            # Check if entity belongs to specified domain
                            if entity.get("domain") == domain_name:
                                filtered_entities.append(entity)
                        
                        # Replace entities with filtered list
                        result_data["data"]["entities"] = filtered_entities
                        result_data["data"]["total_count"] = len(filtered_entities)
                        result = result_data
                    else:
                        result = result_data
                except json.JSONDecodeError:
                    # If parsing fails, return original result
                    result = result_json
                
            elif semantic:
                # Project-wide semantic search
                result = self.project_memory.semantic_search_project(
                    search_term=query,
                    container_name=project_id,
                    entity_types=entity_types,
                    limit=limit
                )
            else:
                # Project-wide regular search
                result = self.project_memory.search_project_entities(
                    search_term=query,
                    container_name=project_id,
                    entity_types=entity_types,
                    limit=limit,
                    semantic=False
                )
            
            # Prepare context for success message
            entity_types_text = ""
            if entity_types:
                if isinstance(entity_types, list):
                    entity_types_text = f" for {', '.join(entity_types)} entities"
                else:
                    entity_types_text = f" for {entity_types} entities"
                    
            domain_text = f" in domain '{domain_name}'" if domain_name else ""
            search_type = "semantic" if semantic else "standard"
            
            # Use standardized response
            return self._standardize_response(
                result_json=result,
                success_message=f"Successfully performed {search_type} search for '{query}'{entity_types_text}{domain_text} in project '{project_id}'",
                error_code="project_search_error"
            )
            
        except Exception as e:
            if self.logger:
                self.logger.error(f"Error searching project: {str(e)}")
            
            # Prepare error details
            details = {
                "query": query,
                "project_id": project_id,
                "entity_types": entity_types,
                "limit": limit,
                "semantic": semantic,
                "domain_name": domain_name
            }
            
            error = ErrorDetail(
                code="project_search_error",
                message=f"Failed to search project: {str(e)}",
                details=details
            )
            error_response = ErrorResponse(
                status="error",
                timestamp=datetime.now(),
                error=error
            )
            return error_response.model_dump_json()
    
    def _handle_structure_retrieval(self, project_id: str, **kwargs) -> str:
        """
        Retrieve the structure of a project
        
        This method returns the hierarchical structure of a project, including all components,
        domains, entities, and their relationships, suitable for visualization or navigation.
        
        Args:
            project_id: ID or name of the project container
            **kwargs: Additional parameters
                
        Returns:
            JSON response with operation results
            
        Required parameters:
            - project_id: Project container identifier
            
        Optional parameters:
            - include_components: Boolean flag to include components (default: True)
            - include_domains: Boolean flag to include domains (default: True)
            - include_relationships: Boolean flag to include relationships (default: True)
            - max_depth: Maximum depth for relationship traversal (default: 3)
            
        Response format:
            All operations return a JSON string with at minimum:
            - status: "success" or "error"
            - message or error: Description of result or error
            - structure: Project structure data if successful

        Example:
            ```
            # Get the complete structure of a project
            result = self._handle_structure_retrieval(
                project_id="E-commerce Platform",
                include_relationships=True,
                max_depth=4
            )
            ```
        """
        try:
            # Extract optional parameters with defaults
            include_components = kwargs.pop("include_components", True)
            include_domains = kwargs.pop("include_domains", True)
            include_relationships = kwargs.pop("include_relationships", True)
            max_depth = kwargs.pop("max_depth", 3)
            
            # Start building the structure with project container info
            result = {}
            
            # Get project container info
            container_result_json = self.project_memory.get_project_container(project_id)
            try:
                container_data = json.loads(container_result_json)
                if "data" in container_data and "container" in container_data["data"]:
                    result["project"] = container_data["data"]["container"]
                else:
                    # If we can't get container data, return error
                    return json.dumps({
                        "status": "error",
                        "error": f"Project '{project_id}' not found",
                        "code": "project_not_found"
                    })
            except json.JSONDecodeError:
                # If parsing fails, use a simplified approach
                result["project"] = {"name": project_id}
            
            # Get domains if requested
            if include_domains:
                if max_depth > 0:
                    # Get all domains
                    domains_result_json = self.project_memory.list_project_domains(project_id)
                    try:
                        domains_data = json.loads(domains_result_json)
                        if "data" in domains_data and "domains" in domains_data["data"]:
                            result["domains"] = domains_data["data"]["domains"]
                        else:
                            result["domains"] = []
                    except json.JSONDecodeError:
                        result["domains"] = []
                
                # Get domains recursively
                for domain in result["domains"]:
                    domain_name = domain.get("name")
                    if domain_name:
                        domain_result_json = self.project_memory.get_project_domain(domain_name, project_id)
                        try:
                            domain_data = json.loads(domain_result_json)
                            if "data" in domain_data and "domain" in domain_data["data"]:
                                result["domains"].append(domain_data["data"]["domain"])
                            else:
                                result["domains"].append({"name": domain_name})
                        except json.JSONDecodeError:
                            result["domains"].append({"name": domain_name})
            
            # Get components if requested
            if include_components:
                result["components"] = []
                
                # Domains to iterate over
                domains_to_check = []
                if "domains" in result:
                    domains_to_check = [d.get("name") for d in result["domains"] if d.get("name")]
                
                # Get components for each domain
                for d_name in domains_to_check:
                    components_result_json = self.project_memory.list_project_components(d_name, project_id)
                    try:
                        components_data = json.loads(components_result_json)
                        if "data" in components_data and "components" in components_data["data"]:
                            # Add domain name to each component for reference
                            for component in components_data["data"]["components"]:
                                component["domain"] = d_name
                                result["components"].append(component)
                    except json.JSONDecodeError:
                        pass  # Skip if we can't parse
            
            # Get relationships if requested (and we have components)
            if include_relationships and "components" in result and result["components"]:
                result["relationships"] = []
                
                # For each component, get its dependencies
                for component in result["components"]:
                    comp_name = component.get("name")
                    if not comp_name:
                        continue
                        
                    domain = component.get("domain")
                    if not domain:
                        continue
                    
                    # Get outgoing dependencies
                    deps_result_json = self.project_memory.get_project_dependencies(
                        comp_name, domain, project_id, "outgoing"
                    )
                    
                    try:
                        deps_data = json.loads(deps_result_json)
                        if "data" in deps_data and "dependencies" in deps_data["data"]:
                            for dep in deps_data["data"]["dependencies"]:
                                result["relationships"].append(dep)
                    except json.JSONDecodeError:
                        pass  # Skip if we can't parse
            
            # Add project status information
            status_result_json = self.project_memory.get_project_status(project_id)
            try:
                status_data = json.loads(status_result_json)
                if "data" in status_data:
                    result["status"] = status_data["data"]
            except json.JSONDecodeError:
                pass  # Skip if we can't parse
            
            return json.dumps({
                "status": "success",
                "message": f"Retrieved structure for project '{project_id}'",
                "data": result
            })
            
        except Exception as e:
            if self.logger:
                self.logger.error(f"Error retrieving project structure: {str(e)}")
            return json.dumps({
                "status": "error",
                "error": f"Failed to retrieve project structure: {str(e)}",
                "code": "structure_retrieval_error"
            })
    
    def _handle_add_observation(self, entity_name: str, content: str, **kwargs) -> str:
        """
        Add an observation to a project entity
        
        This method adds structured observations or notes to project entities to
        capture decisions, insights, or other contextual information.
        
        Args:
            entity_name: Name of the entity
            content: Content of the observation
            **kwargs: Additional parameters
                
        Returns:
            JSON response with operation results
            
        Required parameters:
            - entity_name: Entity identifier to attach observation to
            - content: Text content of the observation
            
        Optional parameters:
            - project_id: Project container identifier (required if entity_name isn't unique)
            - observation_type: Classification of the observation (e.g., 'DECISION', 'NOTE', 'ISSUE')
            - entity_type: Type of entity ('component', 'domain', 'project')
            - domain_name: Domain name if entity is a component
            
        Response format:
            All operations return a JSON string with at minimum:
            - status: "success" or "error"
            - message or error: Description of result or error
            - observation: Created observation data if successful

        Example:
            ```
            # Add a decision observation to a component
            result = self._handle_add_observation(
                entity_name="Authentication Service",
                content="Decided to use JWT tokens with 1-hour expiration for security",
                project_id="E-commerce Platform",
                observation_type="DECISION"
            )
            ```
        """
        try:
            # Extract optional parameters with defaults
            project_id = kwargs.pop("project_id", None)
            observation_type = kwargs.pop("observation_type", "general")
            entity_type = kwargs.pop("entity_type", "component").lower()
            domain_name = kwargs.pop("domain_name", None)
            
            # Build the observation data
            observation_data = {
                "content": content,
                "type": observation_type,
                "timestamp": datetime.now().isoformat()
            }
            
            # Use the appropriate method based on entity_type to add the observation
            if entity_type == "component":
                # Add observation to a component
                if project_id:
                    # Use the observation_manager to add an observation to the component
                    result = self._add_entity_observation_generic(
                        entity_name=entity_name,
                        observation_data=observation_data,
                        project_id=project_id,
                        entity_type="Component",
                        domain_name=domain_name
                    )
                else:
                    # No project context, use entity name only (less reliable)
                    result = self._add_entity_observation_generic(
                        entity_name=entity_name,
                        observation_data=observation_data
                    )
            elif entity_type == "domain":
                # Add observation to a domain
                if project_id:
                    # Use the observation_manager to add an observation to the domain
                    result = self._add_entity_observation_generic(
                        entity_name=entity_name,
                        observation_data=observation_data,
                        project_id=project_id,
                        entity_type="Domain"
                    )
                else:
                    # No project context, use entity name only (less reliable)
                    result = self._add_entity_observation_generic(
                        entity_name=entity_name,
                        observation_data=observation_data
                    )
            elif entity_type == "project":
                # Add observation to a project container
                container_name = entity_name if entity_name else project_id
                if container_name:
                    result = self._add_entity_observation_generic(
                        entity_name=container_name,
                        observation_data=observation_data,
                        entity_type="ProjectContainer"
                    )
                else:
                    return json.dumps({
                        "status": "error",
                        "error": "Project name is required",
                        "code": "missing_project_name"
                    })
            else:
                # Fallback to generic observation for unknown entity types
                result = self._add_entity_observation_generic(
                    entity_name=entity_name,
                    observation_data=observation_data,
                    project_id=project_id
                )
            
            # Return the result
            return result
            
        except Exception as e:
            if self.logger:
                self.logger.error(f"Error adding observation: {str(e)}")
            return json.dumps({
                "status": "error",
                "error": f"Failed to add observation: {str(e)}",
                "code": "observation_error"
            })
    
    def _add_entity_observation_generic(self, entity_name: str, observation_data: Dict[str, Any], 
                                       project_id: Optional[str] = None, entity_type: Optional[str] = None,
                                       domain_name: Optional[str] = None) -> str:
        """Helper method for adding observations to generic entities when specific entity type is unknown."""
        try:
            # Query to match the entity
            match_clause = "MATCH (e) WHERE e.name = $entity_name"
            params = {"entity_name": entity_name}
            
            # Add entity type filter if available
            if entity_type:
                match_clause += f" AND e:{entity_type}"
            
            # Add project filter if available
            if project_id:
                match_clause += " AND (e.project = $project_id OR EXISTS((e)<-[:CONTAINS]-(:ProjectContainer {name: $project_id})))"
                params["project_id"] = project_id
                
            # Add domain filter if available
            if domain_name and entity_type == "Component":
                match_clause += " AND EXISTS((e)<-[:CONTAINS]-(:Domain {name: $domain_name}))"
                params["domain_name"] = domain_name
            
            # First find the entity
            entity_query = f"{match_clause} RETURN e"
            entity_result = self.base_manager.safe_execute_read_query(entity_query, params)
            
            if not entity_result or len(entity_result) == 0:
                return json.dumps({
                    "status": "error",
                    "error": f"Entity '{entity_name}' not found",
                    "code": "entity_not_found"
                })
            
            # Create the observation
            create_query = f"""
            {match_clause}
            CREATE (o:Observation {{
                content: $content,
                type: $type,
                created: $timestamp,
                id: randomUUID()
            }})
            CREATE (e)-[:HAS_OBSERVATION]->(o)
            RETURN o
            """
            
            create_params = {
                **params,
                "content": observation_data["content"],
                "type": observation_data["type"],
                "timestamp": observation_data["timestamp"]
            }
            
            # Use safe_execute_write_query to safely execute the write query
            observation_result = self.base_manager.safe_execute_write_query(create_query, create_params)
            
            # Check if the observation was created
            if observation_result and len(observation_result) > 0:
                observation_record = observation_result[0]
                if observation_record and 'o' in observation_record:
                    observation = observation_record['o']
                    return json.dumps({
                        "status": "success",
                        "message": f"Observation added to '{entity_name}'",
                        "observation": {
                            "id": observation.get('id'),
                            "content": observation.get('content'),
                            "type": observation.get('type'),
                            "created": observation.get('created')
                        }
                    })
            
            # Observation could not be created
            return json.dumps({
                "status": "error",
                "error": f"Observation could not be created for entity '{entity_name}'",
                "code": "observation_creation_error"
            })
            
        except Exception as e:
            if self.logger:
                self.logger.error(f"Error in generic observation: {str(e)}")
            return json.dumps({
                "status": "error",
                "error": f"Failed to add generic observation: {str(e)}",
                "code": "observation_error"
            })
    
    def _handle_entity_update(self, entity_name: str, updates: Dict[str, Any], **kwargs) -> str:
        """
        Update an existing project entity
        
        This method applies updates to the attributes of existing project entities,
        including components, domains, and other project-related entities.
        
        Args:
            entity_name: Name of the entity to update
            updates: Dictionary of updates to apply
            **kwargs: Additional parameters
                
        Returns:
            JSON response with operation results
            
        Required parameters:
            - entity_name: Entity identifier to update
            - updates: Dictionary of field-value pairs to update
            
        Optional parameters:
            - project_id: Project container identifier (required if entity_name isn't unique)
            - entity_type: Type of entity ('component', 'domain', 'project')
            - domain_name: Domain name if entity is a component
            
        Response format:
            All operations return a JSON string with at minimum:
            - status: "success" or "error"
            - message or error: Description of result or error
            - entity: Updated entity data if successful

        Example:
            ```
            # Update a component description
            result = self._handle_entity_update(
                entity_name="Authentication Service",
                updates={"description": "Updated service using OAuth2 protocol"},
                project_id="E-commerce Platform",
                entity_type="component"
            )
            ```
        """
        # Pre-initialize variables that may be used in error handling
        entity_type = "component"
        domain_name = None
        project_id = None
        result = None
        
        try:
            # Extract optional parameters with defaults
            entity_type = kwargs.pop("entity_type", "component").lower()
            domain_name = kwargs.pop("domain_name", None)
            project_id = kwargs.pop("project_id", None)
            
            # Determine which update method to use based on entity_type
            if entity_type == "project":
                # Update project container
                result = self.project_memory.update_project_container(entity_name, updates)
            elif entity_type == "domain":
                # Update domain
                result = self.project_memory.update_project_domain(
                    name=entity_name,
                    container_name=project_id,
                    updates=updates
                )
            elif entity_type == "component" and domain_name:
                # Update component
                result = self.project_memory.update_project_component(
                    name=entity_name,
                    container_name=project_id,
                    updates=updates,
                    domain_name=domain_name
                )
            else:
                # Handle unsupported entity type or missing domain_name for components
                error_msg = f"Unsupported entity type '{entity_type}' or missing domain_name for component update"
                if self.logger:
                    self.logger.error(error_msg)
                
                error = ErrorDetail(
                    code="invalid_update_parameters",
                    message=error_msg,
                    details={
                        "entity_type": entity_type,
                        "has_domain_name": domain_name is not None,
                        "entity_name": entity_name
                    }
                )
                error_response = ErrorResponse(
                    status="error",
                    timestamp=datetime.now(),
                    error=error
                )
                return error_response.model_dump_json()
            
            # Format update fields for message
            update_fields = list(updates.keys())
            update_text = ", ".join([f"'{field}'" for field in update_fields[:3]])
            if len(update_fields) > 3:
                update_text += f" and {len(update_fields) - 3} more fields"
                
            # Create context for success message
            project_context = f" in project '{project_id}'" if project_id else ""
            domain_context = f" in domain '{domain_name}'" if domain_name else ""
            
            # Use standardized response
            return self._standardize_response(
                result_json=result,
                success_message=f"Successfully updated {entity_type} '{entity_name}'{project_context}{domain_context} with fields: {update_text}",
                error_code="entity_update_error"
            )
            
        except Exception as e:
            if self.logger:
                self.logger.error(f"Error updating entity: {str(e)}")
            
            # Prepare error details
            details = {
                "entity_name": entity_name,
                "entity_type": entity_type,
                "project_id": project_id,
                "domain_name": domain_name,
                "update_fields": list(updates.keys())
            }
            
            error = ErrorDetail(
                code="entity_update_error",
                message=f"Failed to update entity: {str(e)}",
                details=details
            )
            error_response = ErrorResponse(
                status="error",
                timestamp=datetime.now(),
                error=error
            )
            return error_response.model_dump_json()
    
    def _handle_entity_deletion(self, entity_name: str, entity_type: str, **kwargs) -> str:
        """
        Delete a project entity (project, domain, component, or observation)
        
        Args:
            entity_name: Name of the entity to delete
            entity_type: Type of entity to delete ('project', 'domain', 'component', or 'observation')
            **kwargs: Additional parameters
                
        Returns:
            JSON response with operation results
            
        Required parameters:
            - entity_name: Name of the entity to delete
            - entity_type: Type of entity to delete
            
        Optional parameters:
            - container_name: Project container name (required for domain and component)
            - domain_name: Domain name (required for component)
            - delete_contents: Boolean flag to delete contents when deleting projects or domains
            - observation_content: Content of observation to delete (alternative to observation_id)
            - observation_id: ID of the observation (alternative to content)
            
        Response format:
            All operations return a JSON string with at minimum:
            - status: "success" or "error"
            - message or error: Description of result or error

        Example:
            ```
            # Delete a project
            result = self._handle_entity_deletion(
                entity_name="E-commerce Platform",
                entity_type="project",
                delete_contents=True
            )
            
            # Delete a domain
            result = self._handle_entity_deletion(
                entity_name="Backend",
                entity_type="domain",
                container_name="E-commerce Platform",
                delete_contents=True
            )
            
            # Delete a component
            result = self._handle_entity_deletion(
                entity_name="Authentication Service",
                entity_type="component",
                container_name="E-commerce Platform",
                domain_name="Backend"
            )
            
            # Delete an observation
            result = self._handle_entity_deletion(
                entity_name="Security Policy",
                entity_type="observation",
                observation_content="Security policy needs updating"
            )
            ```
        """
        try:
            # Extract optional parameters with defaults
            container_name = kwargs.pop("container_name", None)
            domain_name = kwargs.pop("domain_name", None)
            delete_contents = kwargs.pop("delete_contents", False)
            observation_content = kwargs.pop("observation_content", None)
            observation_id = kwargs.pop("observation_id", None)
            
            # Determine which deletion method to use based on entity_type
            if entity_type.lower() == "project":
                # Delete project container
                if not self.project_memory:
                    error_msg = "Project memory manager not initialized"
                    if self.logger:
                        self.logger.error(error_msg)
                    return json.dumps({
                        "status": "error",
                        "error": error_msg,
                        "code": "memory_not_initialized"
                    })
                
                result = self.project_memory.delete_project_container(
                    name=entity_name,
                    delete_contents=delete_contents
                )
                
            elif entity_type.lower() == "domain":
                # Delete domain
                if not container_name:
                    error_msg = "Container name is required for domain deletion"
                    if self.logger:
                        self.logger.error(error_msg)
                    return json.dumps({
                        "status": "error",
                        "error": error_msg,
                        "code": "missing_container_name"
                    })
                
                result = self.project_memory.delete_project_domain(
                    name=entity_name,
                    container_name=container_name,
                    delete_components=delete_contents
                )
                
            elif entity_type.lower() == "component":
                # Delete component
                if not container_name and not domain_name:
                    error_msg = "Container name or domain name is required for component deletion"
                    if self.logger:
                        self.logger.error(error_msg)
                    return json.dumps({
                        "status": "error",
                        "error": error_msg,
                        "code": "missing_container_or_domain"
                    })
                
                result = self.project_memory.delete_project_component(
                    component_id=entity_name,
                    domain_name=domain_name,
                    container_name=container_name
                )
                
            elif entity_type.lower() == "observation":
                # Delete observation
                result = self.delete_observation(
                    entity_name=entity_name,
                    observation_content=observation_content,
                    observation_id=observation_id
                )
                
            else:
                # Handle unsupported entity type
                error_msg = f"Unsupported entity type: {entity_type}"
                if self.logger:
                    self.logger.error(error_msg)
                return json.dumps({
                    "status": "error",
                    "error": error_msg,
                    "code": "invalid_entity_type"
                })
            
            # Handle different return types (future-proof)
            if isinstance(result, str):
                return result
            else:
                return json.dumps(result)
            
        except Exception as e:
            if self.logger:
                self.logger.error(f"Error deleting entity: {str(e)}")
            return json.dumps({
                "status": "error",
                "error": f"Failed to delete entity: {str(e)}",
                "code": "entity_deletion_error"
            })

    def _handle_relationship_deletion(self, source_name: str, target_name: str, relationship_type: str, **kwargs) -> str:
        """
        Delete a relationship between entities within a project
        
        Args:
            source_name: Name of the source entity
            target_name: Name of the target entity
            relationship_type: Type of relationship to delete
            **kwargs: Additional parameters
                
        Returns:
            JSON response with operation results
            
        Required parameters:
            - source_name: Source entity identifier
            - target_name: Target entity identifier
            - relationship_type: Type of relationship to delete
            
        Optional parameters:
            - domain_name: Domain name if entities are components in a domain
            - container_name: Project container name (required for domain components)
            - project_id: Alternative name for container_name
            
        Response format:
            All operations return a JSON string with at minimum:
            - status: "success" or "error"
            - message or error: Description of result or error

        Example:
            ```
            # Delete a dependency relationship
            result = self._handle_relationship_deletion(
                source_name="Authentication Service",
                target_name="User Database",
                relationship_type="DEPENDS_ON",
                container_name="E-commerce Platform",
                domain_name="Backend"
            )
            ```
        """
        try:
            # Extract parameters with fallbacks
            container_name = kwargs.pop("container_name", kwargs.pop("project_id", None))
            domain_name = kwargs.pop("domain_name", None)
            
            if not container_name:
                error_msg = "container_name or project_id is required for relationship deletion"
                if self.logger:
                    self.logger.error(error_msg)
                return json.dumps({
                    "status": "error",
                    "error": error_msg,
                    "code": "missing_container_name"
                })
            
            # Delete relationship using the project dependency manager
            # This handles component relationships, domain relationships are handled separately
            result = self.project_memory.delete_project_dependency(
                from_component=source_name,
                to_component=target_name,
                domain_name=domain_name,
                container_name=container_name,
                dependency_type=relationship_type
            )
            
            # Handle different return types (future-proof)
            if isinstance(result, str):
                return result
            else:
                return json.dumps(result)
            
        except Exception as e:
            if self.logger:
                self.logger.error(f"Error deleting relationship: {str(e)}")
            return json.dumps({
                "status": "error",
                "error": f"Failed to delete relationship: {str(e)}",
                "code": "relationship_deletion_error"
            })

    def project_context(self, project_name: Optional[str] = None):
        """
        Create a context for batch project memory operations.
        
        This tool returns a context object that can be used for multiple
        project operations with shared project context.
        
        Args:
            project_name: Optional. Project name to set as context
                
        Returns:
            JSON response with context information that includes:
            - status: "success" or "error"
            - message or error: Description of result or error
            - context: Context object with project_name, created_at timestamp,
                      available operations, and usage instructions
        
        Response structure:
            ```json
            {
                "status": "success",
                "message": "Project memory context created for project 'ProjectName'",
                "context": {
                    "project_name": "ProjectName",
                    "created_at": "2023-07-15T10:30:45.123456",
                    "operations_available": ["create_component", "create_domain_entity", "relate", "search", "get_structure", "add_observation", "update", "delete_entity", "delete_relationship"],
                    "usage": "Use this context information with any project memory operation by including it in the operation's context parameter"
                }
            }
            ```
             
        The returned context object has these methods:
            - create_project(): Create a new project
            - create_component(): Create a component within a project
            - create_domain(): Create a domain within a project
            - create_domain_entity(): Create an entity within a domain
            - relate(): Create relationship between entities 
            - search(): Search for project entities
            - get_structure(): Get project structure
            - add_observation(): Add observation to an entity
            - update(): Update entity properties
            - delete_entity(): Delete an entity
            - delete_relationship(): Delete a relationship
            
        Example:
            ```
            # Use as a context manager for a specific project
            with project_context("E-commerce Platform") as context:
                # Perform operations using the context
                component = context.create_component({
                    "name": "Authentication Service",
                    "component_type": "microservice"
                })
                
                # All operations use the same project context
                entity = context.create_domain_entity({
                    "name": "User",
                    "entity_type": "DATA_MODEL"
                })
            
            # Original project context is automatically restored here
            ```
        """
        from src.models.response_models import ProjectContextModel, ProjectContextResponse
        
        self._ensure_initialized()
        
        # Save current project
        original_project = self.default_project_name
        
        try:
            # Set project context if provided
            if project_name:
                self.set_project_name(project_name)
            
            # Create context model for tracking
            context_model = ProjectContextModel(
                project_name=project_name or self.default_project_name,
                created_at=datetime.now()
            )
            
            # Log context creation
            if self.logger:
                self.logger.debug(f"Created project context with project={context_model.project_name}")
                
            # Create and yield context helper
            context = ProjectContext(self.project_memory, project_name)
            
            # Attach context model to the context object for reference
            context._context_model = context_model
            
            yield context
            
        finally:
            # Restore original project context
            self.set_project_name(original_project)
    
    