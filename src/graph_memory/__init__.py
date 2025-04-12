"""
Graph Memory Module

This module provides functionality for interacting with the knowledge graph.
It exposes a set of managers for different graph memory operations, as well
as a facade class that maintains backward compatibility with the original API.
"""

from typing import Any, Dict, List, Optional, Union, cast
import json
import datetime
import time
import os
import logging
import sys
import uuid
import re
from contextlib import contextmanager

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
    """Helper class for context-bound lesson operations."""
    
    def __init__(self, lesson_memory, container_name=None):
        """
        Initialize with lesson memory manager and container context.
        
        Args:
            lesson_memory: The LessonMemoryManager instance
            container_name: Optional container name to use as context
        """
        self.lesson_memory = lesson_memory
        self.container_name = container_name or "Lessons"
    
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
    """Helper class for context-bound project operations."""
    
    def __init__(self, project_memory, project_name=None):
        """
        Initialize with project memory manager and project context.
        
        Args:
            project_memory: The ProjectMemoryManager instance
            project_name: Optional project name to use as context
        """
        self.project_memory = project_memory
        self.project_name = project_name or "default"
    
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
    
    def relate(self, source_name: str, target_name: str, relation_type: str, **kwargs) -> str:
        """
        Create a relationship between entities within this project context.
        
        Args:
            source_name: Name of the source entity
            target_name: Name of the target entity
            relation_type: Type of relationship
            **kwargs: Additional parameters including entity_type and domain_name
        
        Returns:
            JSON response with relationship data
        """
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
        self._ensure_initialized()
        
        # Associate with current project if not specified
        if "project" not in entity_data and self.default_project_name:
            entity_data["project"] = self.default_project_name
            
        # Use the entity manager to create the entity
        return self.entity_manager.create_entities([entity_data])
    
    def create_entities(self, entities: List[Dict[str, Any]]) -> str:
        """
        Create multiple entities.
        
        Args:
            entities: List of dictionaries with entity data
            
        Returns:
            JSON string with the result
        """
        self._ensure_initialized()
        
        # Associate with current project if not specified
        if self.default_project_name:
            for entity in entities:
                if "project" not in entity:
                    entity["project"] = self.default_project_name
                    
        # Use the entity manager to create the entities
        return self.entity_manager.create_entities(entities)
    
    def get_entity(self, entity_name: str) -> str:
        """
        Get an entity from the knowledge graph.
        
        Args:
            entity_name: The name of the entity to retrieve
            
        Returns:
            JSON string with the entity information
        """
        self._ensure_initialized()
        return self.entity_manager.get_entity(entity_name)
    
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
        self._ensure_initialized()
        return self.search_manager.search_entities(query, limit, entity_types, semantic)
    
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
            "timestamp": datetime.datetime.now().isoformat(),
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
        Single entry point for lesson operations.
        
        Args:
            operation_type: Type of operation to perform
            **kwargs: Operation-specific parameters
            
        Returns:
            JSON response string
        """
        self._ensure_initialized()
        
        operations = {
            "create": self._handle_lesson_creation,
            "observe": self._handle_lesson_observation,
            "relate": self._handle_lesson_relationship,
            "search": self._handle_lesson_search,
            "track": self._handle_lesson_tracking,
            "consolidate": self._handle_lesson_consolidation,
            "evolve": self._handle_lesson_evolution,
            "update": self._handle_lesson_update,
        }
        
        if operation_type not in operations:
            error_msg = f"Unknown operation type: {operation_type}"
            if self.logger:
                self.logger.error(error_msg)
            return json.dumps({"status": "error", "error": error_msg, "code": "invalid_operation"})
            
        try:
            return operations[operation_type](**kwargs)
        except Exception as e:
            if self.logger:
                self.logger.error(f"Error in lesson operation '{operation_type}': {str(e)}")
            return json.dumps({
                "status": "error", 
                "error": f"Operation failed: {str(e)}", 
                "code": "operation_error"
            })
    
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
        try:
            # Extract optional parameters with defaults
            container = kwargs.pop("container_name", "Lessons")
            observations = kwargs.pop("observations", None)
            metadata = kwargs.pop("metadata", None)
            
            # Create the lesson entity - ensure string return type
            result = self.lesson_memory.create_lesson_entity(
                container, name, lesson_type, observations, metadata
            )
            
            # Handle different return types (future-proof)
            if isinstance(result, str):
                return result
            else:
                return json.dumps(result)
            
        except Exception as e:
            if self.logger:
                self.logger.error(f"Error creating lesson: {str(e)}")
            return json.dumps({
                "status": "error",
                "error": f"Failed to create lesson: {str(e)}",
                "code": "lesson_creation_error"
            })
            
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
            # Use the entity_name and pass all other kwargs directly
            kwargs["entity_name"] = entity_name
            
            # Create the structured observations - ensure string return type
            result = self.lesson_memory.create_structured_lesson_observations(**kwargs)
            
            # Handle different return types (future-proof)
            if isinstance(result, str):
                return result
            else:
                return json.dumps(result)
            
        except Exception as e:
            if self.logger:
                self.logger.error(f"Error adding lesson observations: {str(e)}")
            return json.dumps({
                "status": "error",
                "error": f"Failed to add lesson observations: {str(e)}",
                "code": "observation_error"
            })
    
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
            # Validate required parameters
            required_params = ["source_name", "target_name", "relationship_type"]
            for param in required_params:
                if param not in kwargs:
                    raise ValueError(f"Missing required parameter: {param}")
            
            # Create the relationship
            result = self.lesson_memory.create_lesson_relationship(kwargs)
            
            # Handle different return types (future-proof)
            if isinstance(result, str):
                return result
            else:
                return json.dumps(result)
            
        except Exception as e:
            if self.logger:
                self.logger.error(f"Error creating lesson relationship: {str(e)}")
            return json.dumps({
                "status": "error",
                "error": f"Failed to create lesson relationship: {str(e)}",
                "code": "relationship_error"
            })
        
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
                    
                    # Handle different return types (future-proof)
                    if isinstance(result, str):
                        return result
                    else:
                        return json.dumps(result)
                        
                except Exception as semantic_error:
                    # Fall back to standard search if semantic search fails
                    if self.logger:
                        self.logger.warning(f"Semantic search failed, falling back to standard search: {str(semantic_error)}")
            
            # Use standard search (either as primary method or fallback)
            result = self.lesson_memory.search_lesson_entities(
                container_name=kwargs.get("container_name"),
                search_term=kwargs.get("search_term"),
                entity_type=kwargs.get("entity_type"),
                tags=kwargs.get("tags"),
                limit=kwargs.get("limit", 50),
                semantic=False  # We already tried semantic search above if enabled
            )
            
            # Handle different return types (future-proof)
            if isinstance(result, str):
                return result
            else:
                return json.dumps(result)
            
        except Exception as e:
            if self.logger:
                self.logger.error(f"Error searching lessons: {str(e)}")
            return json.dumps({
                "status": "error",
                "error": f"Failed to search lessons: {str(e)}",
                "code": "search_error"
            })
        
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
            
            # Handle different return types (future-proof)
            if isinstance(result, str):
                return result
            else:
                return json.dumps(result)
            
        except Exception as e:
            if self.logger:
                self.logger.error(f"Error tracking lesson application: {str(e)}")
            return json.dumps({
                "status": "error",
                "error": f"Failed to track lesson application: {str(e)}",
                "code": "tracking_error"
            })
        
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
            
            # Handle different return types (future-proof)
            if isinstance(result, str):
                return result
            else:
                return json.dumps(result)
            
        except Exception as e:
            if self.logger:
                self.logger.error(f"Error consolidating lessons: {str(e)}")
            return json.dumps({
                "status": "error",
                "error": f"Failed to consolidate lessons: {str(e)}",
                "code": "consolidation_error"
            })
        
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
            # Get optional parameters
            reason = kwargs.get("reason")
            
            # Track the supersession
            result = self.lesson_memory.track_lesson_supersession(
                old_lesson=old_lesson,
                new_lesson=new_lesson,
                reason=reason
            )
            
            # Handle different return types (future-proof)
            if isinstance(result, str):
                return result
            else:
                return json.dumps(result)
            
        except Exception as e:
            if self.logger:
                self.logger.error(f"Error tracking lesson evolution: {str(e)}")
            return json.dumps({
                "status": "error",
                "error": f"Failed to track lesson evolution: {str(e)}",
                "code": "evolution_error"
            })
        
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
            # Get optional parameters
            container_name = kwargs.get("container_name")
            
            # Update the lesson entity
            result = self.lesson_memory.update_lesson_entity(
                entity_name=entity_name,
                updates=updates,
                container_name=container_name
            )
            
            # Handle different return types (future-proof)
            if isinstance(result, str):
                return result
            else:
                return json.dumps(result)
            
        except Exception as e:
            if self.logger:
                self.logger.error(f"Error updating lesson: {str(e)}")
            return json.dumps({
                "status": "error",
                "error": f"Failed to update lesson: {str(e)}",
                "code": "update_error"
            })

    @contextmanager
    def lesson_context(self, project_name: Optional[str] = None, container_name: Optional[str] = None):
        """
        Context manager for batch lesson operations with proper context.
        
        Args:
            project_name: Optional project name to set as context
            container_name: Optional container name to use
            
        Yields:
            LessonContext object with bound methods
        
        Example:
            ```python
            with graph_manager.lesson_context("ProjectX", "EngineeringLessons") as lessons:
                lessons.create("CachingStrategy", "BestPractice")
                lessons.observe("CachingStrategy", 
                               what_was_learned="Redis outperforms in-memory for distributed systems",
                               how_to_apply="Use for session data across load-balanced servers")
                lessons.relate("CachingStrategy", "AuthSystem", "APPLIES_TO")
            ```
        """
        self._ensure_initialized()
        
        # Save current project
        original_project = self.default_project_name
        
        try:
            # Set project context if provided
            if project_name:
                self.set_project_name(project_name)
                
            # Create and yield context helper
            context = LessonContext(self.lesson_memory, container_name)
            yield context
            
        finally:
            # Restore original project context
            self.set_project_name(original_project)
    
    # Project Memory System methods
    
    def project_operation(self, operation_type: str, **kwargs) -> str:
        """
        Single entry point for project memory operations.
        
        Args:
            operation_type: Type of operation to perform
            **kwargs: Operation-specific parameters
            
        Returns:
            JSON response string
        """
        self._ensure_initialized()
        
        operations = {
            "create_project": self._handle_project_creation,
            "create_component": self._handle_component_creation,
            "create_domain_entity": self._handle_domain_entity_creation,
            "relate_entities": self._handle_entity_relationship,
            "search": self._handle_project_search,
            "get_structure": self._handle_structure_retrieval,
            "add_observation": self._handle_add_observation,
            "update": self._handle_entity_update,
        }
        
        if operation_type not in operations:
            error_msg = f"Unknown operation type: {operation_type}"
            if self.logger:
                self.logger.error(error_msg)
            return json.dumps({"status": "error", "error": error_msg, "code": "invalid_operation"})
            
        try:
            return operations[operation_type](**kwargs)
        except Exception as e:
            if self.logger:
                self.logger.error(f"Error in project operation '{operation_type}': {str(e)}")
            return json.dumps({
                "status": "error", 
                "error": f"Operation failed: {str(e)}", 
                "code": "operation_error"
            })
    
    def _handle_project_creation(self, name: str, **kwargs) -> str:
        """
        Handle project creation with proper defaults.
        
        Args:
            name: Name of the project to create
            **kwargs: Additional parameters
                - description: Optional description of the project
                - metadata: Optional metadata dictionary
                - tags: Optional list of tags
        
        Returns:
            JSON response string with created project data
        """
        try:
            # Extract optional parameters with defaults
            description = kwargs.pop("description", None)
            metadata = kwargs.pop("metadata", None)
            tags = kwargs.pop("tags", None)
            
            # Prepare project data
            project_data = {
                "name": name
            }
            
            if description:
                project_data["description"] = description
            if metadata:
                project_data["metadata"] = metadata
            if tags:
                project_data["tags"] = tags
            
            # Create the project container
            result = self.project_memory.create_project_container(project_data)
            
            # Handle different return types (future-proof)
            if isinstance(result, str):
                return result
            else:
                return json.dumps(result)
            
        except Exception as e:
            if self.logger:
                self.logger.error(f"Error creating project: {str(e)}")
            return json.dumps({
                "status": "error",
                "error": f"Failed to create project: {str(e)}",
                "code": "project_creation_error"
            })
    
    def _handle_component_creation(self, name: str, component_type: str, project_id: str, domain_name: str, **kwargs) -> str:
        """
        Handle component creation within a project domain.
        
        Args:
            name: Name of the component to create
            component_type: Type of component (e.g., 'Service', 'Module', 'Feature')
            project_id: Name of the project container
            domain_name: Name of the domain within the project
            **kwargs: Additional parameters
                - description: Optional description of the component
                - content: Optional content of the component
                - metadata: Optional metadata dictionary
        
        Returns:
            JSON response string with created component data
        """
        try:
            # Extract optional parameters with defaults
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
            
            # Handle different return types (future-proof)
            if isinstance(result, str):
                return result
            else:
                return json.dumps(result)
            
        except Exception as e:
            if self.logger:
                self.logger.error(f"Error creating component: {str(e)}")
            return json.dumps({
                "status": "error",
                "error": f"Failed to create component: {str(e)}",
                "code": "component_creation_error"
            })
    
    def _handle_domain_entity_creation(self, name: str, entity_type: str, project_id: str, **kwargs) -> str:
        """
        Handle domain entity creation within a project.
        
        Args:
            name: Name of the domain entity to create
            entity_type: Type of domain entity (e.g., 'Domain', 'Decision', 'Requirement')
            project_id: Name of the project container
            **kwargs: Additional parameters
                - description: Optional description
                - properties: Optional properties dictionary
        
        Returns:
            JSON response string with created domain entity data
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
        Handle creation of relationships between project entities.
        
        Args:
            source_name: Name of the source entity
            target_name: Name of the target entity
            relation_type: Type of relationship to create
            **kwargs: Additional parameters
                - project_id: Name of the project container
                - domain_name: Optional domain name if entities are in the same domain
                - entity_type: Optional type of entities ('component' or 'domain')
                - properties: Optional properties dictionary for the relationship
        
        Returns:
            JSON response string with created relationship data
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
        Handle searching for entities within a project.
        
        Args:
            query: The search term
            project_id: Name of the project container to search within
            **kwargs: Additional parameters
                - entity_types: Optional list of entity types to filter by
                - limit: Optional maximum number of results to return (default: 10)
                - semantic: Optional flag to use semantic search (default: False)
                - domain_name: Optional domain name to limit search scope
        
        Returns:
            JSON response string with search results
        """
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
                        return json.dumps(result_data)
                    
                    return result_json
                except json.JSONDecodeError:
                    # If parsing fails, return original result
                    return result_json
                
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
                        return json.dumps(result_data)
                    
                    return result_json
                except json.JSONDecodeError:
                    # If parsing fails, return original result
                    return result_json
                
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
            
            # Handle different return types (future-proof)
            if isinstance(result, str):
                return result
            else:
                return json.dumps(result)
            
        except Exception as e:
            if self.logger:
                self.logger.error(f"Error searching project: {str(e)}")
            return json.dumps({
                "status": "error",
                "error": f"Failed to search project: {str(e)}",
                "code": "search_error"
            })
    
    def _handle_structure_retrieval(self, project_id: str, **kwargs) -> str:
        """
        Handle retrieval of project structure.
        
        Args:
            project_id: Name of the project container
            **kwargs: Additional parameters
                - include_domains: Optional flag to include domains (default: True)
                - include_components: Optional flag to include components (default: True)
                - include_relationships: Optional flag to include relationships (default: True)
                - domain_name: Optional domain name to limit the scope
        
        Returns:
            JSON response string with project structure
        """
        try:
            # Extract optional parameters with defaults
            include_domains = kwargs.pop("include_domains", True)
            include_components = kwargs.pop("include_components", True)
            include_relationships = kwargs.pop("include_relationships", True)
            domain_name = kwargs.pop("domain_name", None)
            
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
                if domain_name:
                    # Get a specific domain
                    domain_result_json = self.project_memory.get_project_domain(domain_name, project_id)
                    try:
                        domain_data = json.loads(domain_result_json)
                        if "data" in domain_data and "domain" in domain_data["data"]:
                            result["domains"] = [domain_data["data"]["domain"]]
                        else:
                            result["domains"] = []
                    except json.JSONDecodeError:
                        result["domains"] = []
                else:
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
            
            # Get components if requested
            if include_components:
                result["components"] = []
                
                # Domains to iterate over
                domains_to_check = []
                if domain_name:
                    domains_to_check = [domain_name]
                elif "domains" in result:
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
    
    def _handle_add_observation(self, entity_name: str, content: str, project_id: str, **kwargs) -> str:
        """
        Handle adding an observation to a project entity.
        
        Args:
            entity_name: Name of the entity to add observation to
            content: Content of the observation
            project_id: Name of the project container
            **kwargs: Additional parameters
                - observation_type: Optional type of observation (default: "general")
                - domain_name: Optional domain name if entity is a component
                - entity_type: Optional type of entity ('component' or 'domain')
        
        Returns:
            JSON response string with the added observation
        """
        try:
            # Extract optional parameters with defaults
            observation_type = kwargs.pop("observation_type", "general")
            domain_name = kwargs.pop("domain_name", None)
            entity_type = kwargs.pop("entity_type", "component").lower()
            
            # Build the observation data
            observation_data = {
                "entity_name": entity_name,
                "content": content,
                "observation_type": observation_type
            }
            
            # Add an observation using the appropriate method
            # Note: ProjectMemoryManager doesn't have a direct method for adding observations
            # We need to use the base GraphMemory observation mechanisms
            
            # First, ensure the entity exists
            query = """
            MATCH (e:Entity {name: $entity_name})
            RETURN e
            """
            
            records = self.base_manager.safe_execute_read_query(
                query,
                {"entity_name": entity_name}
            )
            
            if not records or len(records) == 0:
                return json.dumps({
                    "status": "error",
                    "error": f"Entity '{entity_name}' not found",
                    "code": "entity_not_found"
                })
            
            # Add the observation to the entity
            result = self.observation_manager.add_observations([observation_data])
            
            # Handle different return types (future-proof)
            if isinstance(result, str):
                return result
            else:
                return json.dumps(result)
            
        except Exception as e:
            if self.logger:
                self.logger.error(f"Error adding observation: {str(e)}")
            return json.dumps({
                "status": "error",
                "error": f"Failed to add observation: {str(e)}",
                "code": "observation_error"
            })
    
    def _handle_entity_update(self, entity_name: str, updates: Dict[str, Any], project_id: str, **kwargs) -> str:
        """
        Handle updating a project entity.
        
        Args:
            entity_name: Name of the entity to update
            updates: Dictionary of updates to apply
            project_id: Name of the project container
            **kwargs: Additional parameters
                - entity_type: Optional type of entity ('component', 'domain', or 'project')
                - domain_name: Optional domain name if entity is a component
        
        Returns:
            JSON response string with the updated entity
        """
        try:
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
                return json.dumps({
                    "status": "error",
                    "error": error_msg,
                    "code": "invalid_update_parameters"
                })
            
            # Handle different return types (future-proof)
            if isinstance(result, str):
                return result
            else:
                return json.dumps(result)
            
        except Exception as e:
            if self.logger:
                self.logger.error(f"Error updating entity: {str(e)}")
            return json.dumps({
                "status": "error",
                "error": f"Failed to update entity: {str(e)}",
                "code": "entity_update_error"
            })
    
    @contextmanager
    def project_context(self, project_name: Optional[str] = None):
        """
        Context manager for batch project operations with proper context.
        
        Args:
            project_name: Optional project name to set as context
            
        Yields:
            ProjectContext object with bound methods
        
        Example:
            ```python
            with graph_manager.project_context("MyProject") as project:
                project.create_domain("Authentication")
                project.create_component("AuthService", "Microservice", "Authentication",
                                      description="Handles user authentication")
                project.relate("AuthService", "UserDatabase", "DEPENDS_ON", 
                             entity_type="component", domain_name="Authentication")
            ```
        """
        self._ensure_initialized()
        
        # Save current project
        original_project = self.default_project_name
        
        try:
            # Set project context if provided
            if project_name:
                self.set_project_name(project_name)
                
            # Create and yield context helper
            context = ProjectContext(self.project_memory, project_name)
            yield context
            
        finally:
            # Restore original project context
            self.set_project_name(original_project)

