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
    
    def get_all_memories(self, project_name: Optional[str] = None) -> str:
        """
        Get all entities in the knowledge graph.
        
        Args:
            project_name: Optional project name to scope the query
            
        Returns:
            JSON string with all entities
        """
        if project_name:
            self.set_project_name(project_name)
        
        try:
            # Query all entities
            query = """
            MATCH (e:Entity)
            RETURN e
            ORDER BY e.name
            """
            
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
        Delete all entities in the knowledge graph.
        
        Args:
            project_name: Optional project name to scope the deletion
            
        Returns:
            JSON string with the result of the deletion
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
    
    # Lesson Memory System methods for backward compatibility
    
    def create_lesson_container(self, lesson_data: Dict[str, Any]) -> str:
        """
        Create a new lesson container.
        
        Args:
            lesson_data: Dictionary containing lesson container data
            
        Returns:
            JSON string with the created container information
        """
        try:
            self._ensure_initialized()
            
            # Extract basic container information
            title = lesson_data.get("title", "Untitled Lesson")
            description = lesson_data.get("description", "")
            
            # Delegate to the lesson container component via lesson_memory facade
            response = self.lesson_memory.container.create_container(
                title, 
                description,
                metadata=lesson_data.get("metadata")
            )
            
            # Parse the response
            result = json.loads(response)
            
            # Return response with "status":"success" format for consistency with tests
            if "error" not in result:
                return json.dumps({
                    "status": "success",
                    "container": result.get("container", result)
                })
                
            return json.dumps({
                "status": "error",
                "error": result.get("error", "Failed to create lesson container")
            })
            
        except Exception as e:
            if self.logger:
                self.logger.error(f"Error creating lesson container: {str(e)}")
            return json.dumps({
                "status": "error",
                "error": f"Failed to create lesson container: {str(e)}"
            })
    
    def create_lesson(self, name: str, problem_description: str, **kwargs) -> str:
        """
        Create a new lesson entity.
        
        Args:
            name: The name/identifier for the lesson
            problem_description: Description of the problem the lesson addresses
            **kwargs: Additional parameters for lesson creation
            
        Returns:
            JSON string with the created lesson information
        """
        try:
            self._ensure_initialized()
            
            # Extract parameters from kwargs with defaults
            container_name = kwargs.get("container_name", "LessonContainer")
            solution = kwargs.get("solution", "")
            context = kwargs.get("context", "")
            tags = kwargs.get("tags", [])
            related_entities = kwargs.get("related_entities", [])
            source_reference = kwargs.get("source_reference", "")
            
            # Create metadata dictionary
            metadata = {
                "problem_description": problem_description,
                "solution": solution,
                "context": context,
                "tags": tags,
                "source_reference": source_reference
            }
            
            # Create observations dictionary
            observations = {}
            if problem_description:
                observations["problem"] = {
                    "content": problem_description,
                    "type": "problem_description"
                }
            if solution:
                observations["solution"] = {
                    "content": solution,
                    "type": "solution"
                }
            if context:
                observations["context"] = {
                    "content": context,
                    "type": "context"
                }
            
            # Create the lesson entity using the LessonMemoryManager facade
            # The correct parameter order is container_name, entity_name, entity_type, observations, metadata
            return json.dumps(self.lesson_memory.create_lesson_entity(
                container_name,
                name,
                "Lesson",
                list(observations.values()) if observations else None,
                metadata
            ))
            
        except Exception as e:
            if self.logger:
                self.logger.error(f"Error creating lesson: {str(e)}")
            return dict_to_json({"error": f"Failed to create lesson: {str(e)}"})
    
    def get_lessons(self, **kwargs) -> str:
        """
        Get lessons from the knowledge graph.
        
        Args:
            **kwargs: Search parameters for lessons
            
        Returns:
            JSON string with matching lessons
        """
        try:
            self._ensure_initialized()
            
            # Extract default container name
            container_name = kwargs.get("container_name", "LessonContainer")
            search_term = kwargs.get("search_term", None)
            entity_type = kwargs.get("entity_type", "Lesson")
            
            # Use the lesson entity search method
            result = self.lesson_memory.entity.search_lesson_entities(
                container_name=container_name,
                search_term=search_term,
                entity_type=entity_type,
                tags=kwargs.get("tags"),
                limit=kwargs.get("limit", 50),
                semantic=kwargs.get("semantic", False)
            )
            
            return result
            
        except Exception as e:
            if self.logger:
                self.logger.error(f"Error getting lessons: {str(e)}")
            return json.dumps({"error": f"Failed to get lessons: {str(e)}"})
    
    def update_lesson(self, lesson_name: str, **kwargs) -> str:
        """
        Update a lesson's properties.
        
        Args:
            lesson_name: Name of the lesson to update
            **kwargs: Properties to update
            
        Returns:
            JSON string with the updated lesson
        """
        try:
            self._ensure_initialized()
            
            # Extract optional container_name
            container_name = kwargs.pop("container_name", "LessonContainer")
            
            # Update the lesson entity
            return self.lesson_memory.entity.update_lesson_entity(
                lesson_name, 
                kwargs,
                container_name
            )
            
        except Exception as e:
            if self.logger:
                self.logger.error(f"Error updating lesson: {str(e)}")
            return json.dumps({"error": f"Failed to update lesson: {str(e)}"})
    
    def apply_lesson_to_context(self, lesson_name: str, context_entity: str, **kwargs) -> str:
        """
        Apply a lesson to a context entity.
        
        Args:
            lesson_name: Name of the lesson to apply
            context_entity: Entity to apply the lesson to
            **kwargs: Additional application parameters
            
        Returns:
            JSON string with the result of the application
        """
        try:
            self._ensure_initialized()
            
            # Extract parameters
            success_score = kwargs.get("success_score", 0.8)
            application_notes = kwargs.get("application_notes", "")
            
            # Create a relationship between the lesson and context entity
            from_entity = lesson_name
            to_entity = context_entity
            relation_type = "APPLIED_TO"
            
            properties = {
                "success_score": success_score,
                "application_notes": application_notes,
                "applied_at": kwargs.get("applied_at", time.strftime("%Y-%m-%dT%H:%M:%SZ"))
            }
            
            # Create the relationship using the relation manager
            result = self.relation_manager.create_relations([{
                "from": from_entity,
                "to": to_entity,
                "relationType": relation_type,
                **properties
            }])
            
            return result
            
        except Exception as e:
            if self.logger:
                self.logger.error(f"Error applying lesson: {str(e)}")
            return json.dumps({"error": f"Failed to apply lesson: {str(e)}"})
    
    def extract_potential_lessons(self, **kwargs) -> str:
        """
        Extract potential lessons from provided content.
        
        Args:
            **kwargs: Content to extract lessons from
            
        Returns:
            JSON string with potential lessons
        """
        try:
            # Ensure we're initialized
            self._ensure_initialized()
            
            # In actual implementation, this would delegate to LessonMemoryManager
            # However, lesson_memory.py doesn't appear to have this specific method
            # So we'd need to implement a proper delegation when it becomes available
            
            # For now, return empty result
            return dict_to_json({"potential_lessons": [], "status": "success"})
            
        except Exception as e:
            if self.logger:
                self.logger.error(f"Error extracting lessons: {str(e)}")
            return dict_to_json({"error": f"Failed to extract lessons: {str(e)}"})
    
    def consolidate_related_lessons(self, lesson_ids: List[str], **kwargs) -> str:
        """
        Consolidate related lessons into a new lesson.
        
        Args:
            lesson_ids: List of lesson IDs to consolidate
            **kwargs: Consolidation parameters
            
        Returns:
            JSON string with the consolidated lesson
        """
        try:
            self._ensure_initialized()
            
            # Extract parameters
            new_name = kwargs.get("new_name", f"Consolidated_Lesson_{int(time.time())}")
            merge_strategy = kwargs.get("merge_strategy", "union")
            source_lessons = lesson_ids
            
            # Create a new lesson to represent the consolidated knowledge
            new_lesson_data = {
                "name": new_name,
                "entityType": "Lesson",
                "problem_description": kwargs.get("problem_description", "Consolidated from multiple related lessons"),
                "source_lessons": source_lessons,
                "merge_strategy": merge_strategy
            }
            
            # Create the new lesson
            result = json.loads(self.create_entities([new_lesson_data]))
            
            if "created" not in result or len(result["created"]) == 0:
                return json.dumps({"error": "Failed to create consolidated lesson"})
            
            # Create relationships from source lessons to the new lesson
            for lesson_id in source_lessons:
                relation_data = {
                    "from": lesson_id,
                    "to": new_name,
                    "relationType": "CONSOLIDATED_INTO"
                }
                
                self.relation_manager.create_relations([relation_data])
            
            # Return the new consolidated lesson
            return json.dumps({
                "consolidated_lesson": new_name,
                "source_lessons": source_lessons,
                "strategy": merge_strategy
            })
            
        except Exception as e:
            if self.logger:
                self.logger.error(f"Error consolidating lessons: {str(e)}")
            return json.dumps({"error": f"Failed to consolidate lessons: {str(e)}"})
    
    def get_knowledge_evolution(self, **kwargs) -> str:
        """
        Track the evolution of knowledge in the lesson graph.
        
        Args:
            **kwargs: Parameters for evolution tracking
            
        Returns:
            JSON string with evolution data
        """
        try:
            self._ensure_initialized()
            
            # Extract parameters
            entity_name = kwargs.get("entity_name")
            lesson_type = kwargs.get("lesson_type")
            start_date = kwargs.get("start_date")
            end_date = kwargs.get("end_date")
            
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
            
            query += """
            RETURN e,
                   e.created as created_time,
                   e.lastUpdated as updated_time
            ORDER BY e.created
            """
            
            # Execute the query
            results = self.base_manager.safe_execute_read_query(query, params)
            
            # Process results
            evolution_data = []
            for record in results:
                entity = record.get("e")
                if entity:
                    entity_dict = dict(entity.items())
                    entity_dict["created_time"] = record.get("created_time")
                    entity_dict["updated_time"] = record.get("updated_time")
                    evolution_data.append(entity_dict)
            
            return json.dumps({"evolution": evolution_data})
            
        except Exception as e:
            if self.logger:
                self.logger.error(f"Error getting knowledge evolution: {str(e)}")
            return json.dumps({"error": f"Failed to get knowledge evolution: {str(e)}"})
    
    def query_across_contexts(self, query_text: str, **kwargs) -> str:
        """
        Query across multiple contexts.
        
        Args:
            query_text: The text to query for
            **kwargs: Additional parameters
            
        Returns:
            JSON string with query results
        """
        try:
            # Ensure we're initialized
            self._ensure_initialized()
            
            # This would be implemented with cross-context searches
            # utilizing both lesson_memory and project_memory systems
            
            return dict_to_json({"results": [], "status": "success"})
            
        except Exception as e:
            if self.logger:
                self.logger.error(f"Error querying across contexts: {str(e)}")
            return dict_to_json({"error": f"Failed to query across contexts: {str(e)}"})
    
    # Project Memory System methods
    
    def create_project_container(self, project_data: Dict[str, Any]) -> str:
        """
        Create a project container.
        
        Args:
            project_data: Dictionary with project container data
            
        Returns:
            JSON string with the created container
        """
        try:
            self._ensure_initialized()
            
            # Store client_id in project_data for isolation
            if self._client_id:
                if "metadata" not in project_data:
                    project_data["metadata"] = {}
                project_data["metadata"]["client_id"] = self._client_id
            
            # Delegate to the project memory manager
            result = self.get_client_project_memory().create_project_container(project_data)
            
            # Ensure we return a string and add status field if not already present
            if isinstance(result, dict):
                if "status" not in result and "error" not in result:
                    result["status"] = "success"
                return json.dumps(result)
                
            # If already a JSON string, parse and add status field if needed
            parsed_result = json.loads(result)
            if "status" not in parsed_result and "error" not in parsed_result:
                parsed_result["status"] = "success"
            return json.dumps(parsed_result)
            
        except Exception as e:
            if self.logger:
                self.logger.error(f"Error creating project container: {str(e)}")
            return json.dumps({
                "status": "error",
                "error": f"Failed to create project container: {str(e)}"
            })
    
    def get_project_container(self, project_id: str) -> str:
        """
        Get a project container.
        
        Args:
            project_id: The ID of the project
            
        Returns:
            JSON string with the project container
        """
        result = self.get_client_project_memory().get_project_container(project_id)
        # Ensure we return a string
        if isinstance(result, dict):
            return json.dumps(result)
        return result
    
    def update_project_container(self, project_id: str, updates: Dict[str, Any]) -> str:
        """
        Update a project container.
        
        Args:
            project_id: The ID of the project
            updates: Dictionary with updates
            
        Returns:
            JSON string with the result
        """
        result = self.get_client_project_memory().update_project_container(project_id, updates)
        # Ensure we return a string
        if isinstance(result, dict):
            return json.dumps(result)
        return result
    
    def delete_project_container(self, project_id: str) -> str:
        """
        Delete a project container.
        
        Args:
            project_id: The ID of the project
            
        Returns:
            JSON string with the result
        """
        result = self.get_client_project_memory().delete_project_container(project_id, False)
        # Ensure we return a string
        if isinstance(result, dict):
            return json.dumps(result)
        return result
    
    def list_project_containers(self) -> str:
        """
        List project containers.
        
        Returns:
            JSON string with the project containers
        """
        try:
            # Get the result from the project memory manager
            result = self.get_client_project_memory().list_project_containers("name", 100)
            
            # Ensure we return a string
            if isinstance(result, dict):
                return json.dumps(result)
            return result
        except Exception as e:
            if self.logger:
                self.logger.error(f"Error listing project containers: {str(e)}")
            return json.dumps({"error": f"Failed to list project containers: {str(e)}"})
    
    def get_project_status(self, container_name: str) -> str:
        """
        Get a summary of the project container status.
        
        Args:
            container_name: Name of the project container
            
        Returns:
            JSON string with the project status
        """
        try:
            self._ensure_initialized()
            result = self.project_memory.get_project_status(container_name)
            if isinstance(result, dict):
                return dict_to_json(result)
            return result
        except Exception as e:
            if self.logger:
                self.logger.error(f"Error getting project status: {str(e)}")
            return dict_to_json({"error": f"Failed to get project status: {str(e)}"})

    # Domain management methods
    def create_domain(self, name: str, container_name: str, 
                    description: Optional[str] = None,
                    properties: Optional[Dict[str, Any]] = None) -> str:
        """
        Create a new domain within a project container.
        
        Args:
            name: Name of the domain
            container_name: Name of the project container
            description: Optional description of the domain
            properties: Optional additional properties
            
        Returns:
            JSON string with the created domain
        """
        try:
            self._ensure_initialized()
            result = self.project_memory.create_domain(name, container_name, description, properties)
            if isinstance(result, dict):
                return dict_to_json(result)
            return result
        except Exception as e:
            if self.logger:
                self.logger.error(f"Error creating domain: {str(e)}")
            return dict_to_json({"error": f"Failed to create domain: {str(e)}"})
    
    def get_domain(self, name: str, container_name: str) -> str:
        """
        Retrieve a domain from a project container.
        
        Args:
            name: Name of the domain
            container_name: Name of the project container
            
        Returns:
            JSON string with the domain details
        """
        try:
            self._ensure_initialized()
            result = self.project_memory.get_domain(name, container_name)
            if isinstance(result, dict):
                return dict_to_json(result)
            return result
        except Exception as e:
            if self.logger:
                self.logger.error(f"Error getting domain: {str(e)}")
            return dict_to_json({"error": f"Failed to get domain: {str(e)}"})
    
    def update_domain(self, name: str, container_name: str, 
                    updates: Dict[str, Any]) -> str:
        """
        Update a domain's properties.
        
        Args:
            name: Name of the domain
            container_name: Name of the project container
            updates: Dictionary of properties to update
            
        Returns:
            JSON string with the updated domain
        """
        try:
            self._ensure_initialized()
            result = self.project_memory.update_domain(name, container_name, updates)
            if isinstance(result, dict):
                return dict_to_json(result)
            return result
        except Exception as e:
            if self.logger:
                self.logger.error(f"Error updating domain: {str(e)}")
            return dict_to_json({"error": f"Failed to update domain: {str(e)}"})
    
    def delete_domain(self, name: str, container_name: str, 
                    delete_components: bool = False) -> str:
        """
        Delete a domain from a project container.
        
        Args:
            name: Name of the domain
            container_name: Name of the project container
            delete_components: If True, delete all components belonging to the domain
            
        Returns:
            JSON string with the deletion result
        """
        try:
            self._ensure_initialized()
            result = self.project_memory.delete_domain(name, container_name, delete_components)
            if isinstance(result, dict):
                return dict_to_json(result)
            return result
        except Exception as e:
            if self.logger:
                self.logger.error(f"Error deleting domain: {str(e)}")
            return dict_to_json({"error": f"Failed to delete domain: {str(e)}"})
    
    def list_domains(self, container_name: str, sort_by: str = "name", 
                   limit: int = 100) -> str:
        """
        List all domains in a project container.
        
        Args:
            container_name: Name of the project container
            sort_by: Field to sort by ('name', 'created', or 'lastUpdated')
            limit: Maximum number of domains to return
            
        Returns:
            JSON string with the list of domains
        """
        try:
            self._ensure_initialized()
            result = self.project_memory.list_domains(container_name, sort_by, limit)
            if isinstance(result, dict):
                return dict_to_json(result)
            return result
        except Exception as e:
            if self.logger:
                self.logger.error(f"Error listing domains: {str(e)}")
            return dict_to_json({"error": f"Failed to list domains: {str(e)}"})
    
    # Additional dependency management methods
    def get_dependencies(self, component_name: str, domain_name: str, 
                      container_name: str, direction: str = "outgoing",
                      dependency_type: Optional[str] = None) -> str:
        """
        Get dependencies for a component.
        
        Args:
            component_name: Name of the component
            domain_name: Name of the domain
            container_name: Name of the project container
            direction: Direction of dependencies ('outgoing', 'incoming', or 'both')
            dependency_type: Optional dependency type to filter by
            
        Returns:
            JSON string with the dependencies
        """
        try:
            self._ensure_initialized()
            result = self.project_memory.get_dependencies(
                component_name, domain_name, container_name, direction, dependency_type
            )
            if isinstance(result, dict):
                return dict_to_json(result)
            return result
        except Exception as e:
            if self.logger:
                self.logger.error(f"Error getting dependencies: {str(e)}")
            return dict_to_json({"error": f"Failed to get dependencies: {str(e)}"})
    
    def delete_dependency(self, from_component: str, to_component: str,
                       domain_name: str, container_name: str,
                       dependency_type: str) -> str:
        """
        Delete a dependency relationship between components.
        
        Args:
            from_component: Name of the source component
            to_component: Name of the target component
            domain_name: Name of the domain
            container_name: Name of the project container
            dependency_type: Type of dependency to delete
            
        Returns:
            JSON string with the result
        """
        try:
            self._ensure_initialized()
            result = self.project_memory.delete_dependency(
                from_component, to_component, domain_name, container_name, dependency_type
            )
            if isinstance(result, dict):
                return dict_to_json(result)
            return result
        except Exception as e:
            if self.logger:
                self.logger.error(f"Error deleting dependency: {str(e)}")
            return dict_to_json({"error": f"Failed to delete dependency: {str(e)}"})
    
    def analyze_dependency_graph(self, domain_name: str, container_name: str) -> str:
        """
        Analyze the dependency graph for a domain.
        
        Args:
            domain_name: Name of the domain
            container_name: Name of the project container
            
        Returns:
            JSON string with the dependency analysis
        """
        try:
            self._ensure_initialized()
            result = self.project_memory.analyze_dependency_graph(domain_name, container_name)
            if isinstance(result, dict):
                return dict_to_json(result)
            return result
        except Exception as e:
            if self.logger:
                self.logger.error(f"Error analyzing dependency graph: {str(e)}")
            return dict_to_json({"error": f"Failed to analyze dependency graph: {str(e)}"})
    
    def find_path(self, from_component: str, to_component: str,
               domain_name: str, container_name: str,
               max_depth: int = 5) -> str:
        """
        Find dependency paths between two components.
        
        Args:
            from_component: Name of the source component
            to_component: Name of the target component
            domain_name: Name of the domain
            container_name: Name of the project container
            max_depth: Maximum path depth to search
            
        Returns:
            JSON string with the dependency paths
        """
        try:
            self._ensure_initialized()
            result = self.project_memory.find_path(
                from_component, to_component, domain_name, container_name, max_depth
            )
            if isinstance(result, dict):
                return dict_to_json(result)
            return result
        except Exception as e:
            if self.logger:
                self.logger.error(f"Error finding path: {str(e)}")
            return dict_to_json({"error": f"Failed to find path: {str(e)}"})
    
    def import_dependencies_from_code(self, dependencies: List[Dict[str, Any]],
                                  domain_name: str, container_name: str) -> str:
        """
        Import dependencies detected from code analysis.
        
        Args:
            dependencies: List of dependencies, each with from_component, to_component, and dependency_type
            domain_name: Name of the domain
            container_name: Name of the project container
            
        Returns:
            JSON string with the import result
        """
        try:
            self._ensure_initialized()
            result = self.project_memory.import_dependencies_from_code(
                dependencies, domain_name, container_name
            )
            if isinstance(result, dict):
                return dict_to_json(result)
            return result
        except Exception as e:
            if self.logger:
                self.logger.error(f"Error importing dependencies: {str(e)}")
            return dict_to_json({"error": f"Failed to import dependencies: {str(e)}"})
    
    # Additional version management methods
    def get_version(self, component_name: str, domain_name: str,
                 container_name: str, version_number: Optional[str] = None) -> str:
        """
        Get a specific version of a component.
        
        Args:
            component_name: Name of the component
            domain_name: Name of the domain
            container_name: Name of the project container
            version_number: Optional version number (latest if not specified)
            
        Returns:
            JSON string with the version details
        """
        try:
            self._ensure_initialized()
            result = self.project_memory.get_version(
                component_name, domain_name, container_name, version_number
            )
            if isinstance(result, dict):
                return dict_to_json(result)
            return result
        except Exception as e:
            if self.logger:
                self.logger.error(f"Error getting version: {str(e)}")
            return dict_to_json({"error": f"Failed to get version: {str(e)}"})
    
    def list_versions(self, component_name: str, domain_name: str,
                   container_name: str, limit: int = 10) -> str:
        """
        List all versions of a component.
        
        Args:
            component_name: Name of the component
            domain_name: Name of the domain
            container_name: Name of the project container
            limit: Maximum number of versions to return
            
        Returns:
            JSON string with the list of versions
        """
        try:
            self._ensure_initialized()
            result = self.project_memory.list_versions(
                component_name, domain_name, container_name, limit
            )
            if isinstance(result, dict):
                return dict_to_json(result)
            return result
        except Exception as e:
            if self.logger:
                self.logger.error(f"Error listing versions: {str(e)}")
            return dict_to_json({"error": f"Failed to list versions: {str(e)}"})
    
    def get_version_history(self, component_name: str, domain_name: str,
                         container_name: str, include_content: bool = False) -> str:
        """
        Get the version history of a component.
        
        Args:
            component_name: Name of the component
            domain_name: Name of the domain
            container_name: Name of the project container
            include_content: Whether to include content in the version history
            
        Returns:
            JSON string with the version history
        """
        try:
            self._ensure_initialized()
            result = self.project_memory.get_version_history(
                component_name, domain_name, container_name, include_content
            )
            if isinstance(result, dict):
                return dict_to_json(result)
            return result
        except Exception as e:
            if self.logger:
                self.logger.error(f"Error getting version history: {str(e)}")
            return dict_to_json({"error": f"Failed to get version history: {str(e)}"})
    
    def compare_versions(self, component_name: str, domain_name: str,
                      container_name: str, version1: str, version2: str) -> str:
        """
        Compare two versions of a component.
        
        Args:
            component_name: Name of the component
            domain_name: Name of the domain
            container_name: Name of the project container
            version1: First version number
            version2: Second version number
            
        Returns:
            JSON string with the comparison result
        """
        try:
            self._ensure_initialized()
            result = self.project_memory.compare_versions(
                component_name, domain_name, container_name, version1, version2
            )
            if isinstance(result, dict):
                return dict_to_json(result)
            return result
        except Exception as e:
            if self.logger:
                self.logger.error(f"Error comparing versions: {str(e)}")
            return dict_to_json({"error": f"Failed to compare versions: {str(e)}"})
    
    def tag_version(self, component_name: str, domain_name: str,
                 container_name: str, version_number: str,
                 tag_name: str, tag_description: Optional[str] = None) -> str:
        """
        Add a tag to a specific version of a component.
        
        Args:
            component_name: Name of the component
            domain_name: Name of the domain
            container_name: Name of the project container
            version_number: Version number to tag
            tag_name: Name of the tag
            tag_description: Optional description of the tag
            
        Returns:
            JSON string with the result
        """
        try:
            self._ensure_initialized()
            result = self.project_memory.tag_version(
                component_name, domain_name, container_name, 
                version_number, tag_name, tag_description
            )
            if isinstance(result, dict):
                return dict_to_json(result)
            return result
        except Exception as e:
            if self.logger:
                self.logger.error(f"Error tagging version: {str(e)}")
            return dict_to_json({"error": f"Failed to tag version: {str(e)}"})
    
    def sync_with_version_control(self, component_name: str, domain_name: str,
                               container_name: str,
                               commit_data: List[Dict[str, Any]]) -> str:
        """
        Synchronize component versions with version control system data.
        
        Args:
            component_name: Name of the component
            domain_name: Name of the domain
            container_name: Name of the project container
            commit_data: List of commit data
            
        Returns:
            JSON string with the sync result
        """
        try:
            self._ensure_initialized()
            result = self.project_memory.sync_with_version_control(
                component_name, domain_name, container_name, commit_data
            )
            if isinstance(result, dict):
                return dict_to_json(result)
            return result
        except Exception as e:
            if self.logger:
                self.logger.error(f"Error syncing with version control: {str(e)}")
            return dict_to_json({"error": f"Failed to sync with version control: {str(e)}"})

    # Component management methods
    def create_component(self, name: str, component_type: str, domain_name: str,
                       container_name: str, description: Optional[str] = None,
                       content: Optional[str] = None,
                       metadata: Optional[Dict[str, Any]] = None) -> str:
        """
        Create a component within a project.
        
        Args:
            name: Name of the component
            component_type: Type of component (e.g., "Module", "Class", "Function")
            domain_name: Name of the domain
            container_name: Name of the project container
            description: Optional description
            content: Optional content for the component
            metadata: Optional additional metadata
            
        Returns:
            JSON string with operation result
        """
        try:
            self._ensure_initialized()
            result = self.project_memory.create_component(
                name, component_type, domain_name, container_name,
                description, content, metadata
            )
            if isinstance(result, dict):
                return dict_to_json(result)
            return result
        except Exception as e:
            if self.logger:
                self.logger.error(f"Error creating component: {str(e)}")
            return dict_to_json({"error": f"Failed to create component: {str(e)}"})
    
    def get_component(self, name: str, domain_name: str, container_name: str) -> str:
        """
        Get a component from a project.
        
        Args:
            name: Name of the component
            domain_name: Name of the domain
            container_name: Name of the project container
            
        Returns:
            JSON string with the component
        """
        try:
            self._ensure_initialized()
            result = self.project_memory.get_component(name, domain_name, container_name)
            if isinstance(result, dict):
                return dict_to_json(result)
            return result
        except Exception as e:
            if self.logger:
                self.logger.error(f"Error getting component: {str(e)}")
            return dict_to_json({"error": f"Failed to get component: {str(e)}"})
    
    def update_component(self, name: str, container_name: str, updates: Dict[str, Any], domain_name: Optional[str] = None) -> str:
        """
        Update a component's properties.
        
        Args:
            name: Name of the component
            container_name: Name of the project container
            updates: Dictionary of properties to update
            domain_name: Optional name of the domain
            
        Returns:
            JSON string with the updated component
        """
        try:
            self._ensure_initialized()
            # The project_memory.update_component expects: name, container_name, updates, domain_name
            result = self.project_memory.update_component(name, container_name, updates, domain_name)
            if isinstance(result, dict):
                return dict_to_json(result)
            return result
        except Exception as e:
            if self.logger:
                self.logger.error(f"Error updating component: {str(e)}")
            return dict_to_json({"error": f"Failed to update component: {str(e)}"})
    
    def delete_component(self, component_id: str, domain_name: Optional[str] = None, container_name: Optional[str] = None) -> str:
        """
        Delete a component from a domain.
        
        Args:
            component_id: ID or name of the component to delete
            domain_name: Optional name of the domain
            container_name: Optional name of the project container
            
        Returns:
            JSON string with the deletion result
        """
        try:
            self._ensure_initialized()
            result = self.project_memory.delete_component(component_id, domain_name, container_name)
            if isinstance(result, dict):
                return dict_to_json(result)
            return result
        except Exception as e:
            if self.logger:
                self.logger.error(f"Error deleting component: {str(e)}")
            return dict_to_json({"error": f"Failed to delete component: {str(e)}"})
    
    def list_components(self, domain_name: str, container_name: str, 
                      component_type: Optional[str] = None,
                      sort_by: str = "name", limit: int = 100) -> str:
        """
        List all components in a domain.
        
        Args:
            domain_name: Name of the domain
            container_name: Name of the project container
            component_type: Optional type of components to filter by
            sort_by: Field to sort by ('name', 'created', or 'lastUpdated')
            limit: Maximum number of components to return
            
        Returns:
            JSON string with the components
        """
        try:
            self._ensure_initialized()
            result = self.project_memory.list_components(
                domain_name, container_name, component_type, sort_by, limit
            )
            if isinstance(result, dict):
                return dict_to_json(result)
            return result
        except Exception as e:
            if self.logger:
                self.logger.error(f"Error listing components: {str(e)}")
            return dict_to_json({"error": f"Failed to list components: {str(e)}"})
    
    def create_component_relationship(self, from_component: str, to_component: str,
                                   domain_name: str, container_name: str,
                                   relation_type: str, 
                                   properties: Optional[Dict[str, Any]] = None) -> str:
        """
        Create a relationship between two components.
        
        Args:
            from_component: Name of the source component
            to_component: Name of the target component
            domain_name: Name of the domain
            container_name: Name of the project container
            relation_type: Type of relationship
            properties: Optional additional properties
            
        Returns:
            JSON string with the created relationship
        """
        try:
            self._ensure_initialized()
            result = self.project_memory.create_component_relationship(
                from_component, to_component, domain_name, container_name,
                relation_type, properties
            )
            if isinstance(result, dict):
                return dict_to_json(result)
            return result
        except Exception as e:
            if self.logger:
                self.logger.error(f"Error creating component relationship: {str(e)}")
            return dict_to_json({"error": f"Failed to create component relationship: {str(e)}"})
    
    # Domain entity relationship methods
    def add_entity_to_domain(self, entity_name: str, entity_type: str,
                          domain_name: str, container_name: str,
                          properties: Optional[Dict[str, Any]] = None) -> str:
        """
        Add an entity to a domain.
        
        Args:
            entity_name: Name of the entity
            entity_type: Type of entity
            domain_name: Name of the domain
            container_name: Name of the project container
            properties: Optional additional properties
            
        Returns:
            JSON string with the result
        """
        try:
            self._ensure_initialized()
            # Note: The ProjectMemoryManager expects only domain_name, container_name, entity_name
            # But our implementation expects more parameters, so we need to adapt
            result = self.project_memory.add_entity_to_domain(
                domain_name, container_name, entity_name
            )
            if isinstance(result, dict):
                return dict_to_json(result)
            return result
        except Exception as e:
            if self.logger:
                self.logger.error(f"Error adding entity to domain: {str(e)}")
            return dict_to_json({"error": f"Failed to add entity to domain: {str(e)}"})
    
    def remove_entity_from_domain(self, entity_name: str, entity_type: str,
                               domain_name: str, container_name: str) -> str:
        """
        Remove an entity from a domain.
        
        Args:
            entity_name: Name of the entity
            entity_type: Type of entity
            domain_name: Name of the domain
            container_name: Name of the project container
            
        Returns:
            JSON string with the result
        """
        try:
            self._ensure_initialized()
            # Note: The ProjectMemoryManager expects only domain_name, container_name, entity_name
            # But our implementation expects more parameters, so we need to adapt
            result = self.project_memory.remove_entity_from_domain(
                domain_name, container_name, entity_name
            )
            if isinstance(result, dict):
                return dict_to_json(result)
            return result
        except Exception as e:
            if self.logger:
                self.logger.error(f"Error removing entity from domain: {str(e)}")
            return dict_to_json({"error": f"Failed to remove entity from domain: {str(e)}"})
    
    def get_domain_entities(self, domain_name: str, container_name: str,
                         entity_type: Optional[str] = None) -> str:
        """
        Get all entities in a domain.
        
        Args:
            domain_name: Name of the domain
            container_name: Name of the project container
            entity_type: Optional entity type to filter by
            
        Returns:
            JSON string with the entities
        """
        try:
            self._ensure_initialized()
            result = self.project_memory.get_domain_entities(
                domain_name, container_name, entity_type
            )
            if isinstance(result, dict):
                return dict_to_json(result)
            return result
        except Exception as e:
            if self.logger:
                self.logger.error(f"Error getting domain entities: {str(e)}")
            return dict_to_json({"error": f"Failed to get domain entities: {str(e)}"})
    
    def create_domain_relationship(self, from_domain: str, to_domain: str,
                                container_name: str, relation_type: str,
                                properties: Optional[Dict[str, Any]] = None) -> str:
        """
        Create a relationship between two domains.
        
        Args:
            from_domain: Name of the source domain
            to_domain: Name of the target domain
            container_name: Name of the project container
            relation_type: Type of relationship
            properties: Optional additional properties
            
        Returns:
            JSON string with the created relationship
        """
        try:
            self._ensure_initialized()
            result = self.project_memory.create_domain_relationship(
                from_domain, to_domain, container_name, relation_type, properties
            )
            if isinstance(result, dict):
                return dict_to_json(result)
            return result
        except Exception as e:
            if self.logger:
                self.logger.error(f"Error creating domain relationship: {str(e)}")
            return dict_to_json({"error": f"Failed to create domain relationship: {str(e)}"})
    
    # Dependency management methods
    def create_dependency(self, from_component: str, to_component: str, 
                        domain_name: str, container_name: str,
                        dependency_type: str,
                        properties: Optional[Dict[str, Any]] = None) -> str:
        """
        Create a dependency between components.
        
        Args:
            from_component: Source component name
            to_component: Target component name
            domain_name: Name of the domain
            container_name: Name of the project container
            dependency_type: Type of dependency
            properties: Optional additional properties
            
        Returns:
            JSON string with operation result
        """
        try:
            self._ensure_initialized()
            result = self.project_memory.create_dependency(
                from_component, to_component, domain_name, container_name,
                dependency_type, properties
            )
            if isinstance(result, dict):
                return dict_to_json(result)
            return result
        except Exception as e:
            if self.logger:
                self.logger.error(f"Error creating dependency: {str(e)}")
            return dict_to_json({"error": f"Failed to create dependency: {str(e)}"})
    
    # Version management methods
    def create_version(self, component_name: str, domain_name: str,
                     container_name: str, version_number: str,
                     commit_hash: Optional[str] = None,
                     content: Optional[str] = None,
                     changes: Optional[str] = None,
                     metadata: Optional[Dict[str, Any]] = None) -> str:
        """
        Create a new version for a component.
        
        Args:
            component_name: Name of the component
            domain_name: Name of the domain
            container_name: Name of the project container
            version_number: Version number or identifier
            commit_hash: Optional commit hash from version control
            content: Optional content of the component at this version
            changes: Optional description of changes from previous version
            metadata: Optional additional metadata
            
        Returns:
            JSON string with operation result
        """
        try:
            self._ensure_initialized()
            result = self.project_memory.create_version(
                component_name, domain_name, container_name, version_number,
                commit_hash, content, changes, metadata
            )
            if isinstance(result, dict):
                return dict_to_json(result)
            return result
        except Exception as e:
            if self.logger:
                self.logger.error(f"Error creating version: {str(e)}")
            return dict_to_json({"error": f"Failed to create version: {str(e)}"})

    def get_client_project_memory(self):
        """
        Get the appropriate project memory manager for the current client.
        
        Returns:
            The correct ProjectMemoryManager for this client
        """
        if self._client_id and self._client_id in self._client_projects:
            return self._client_projects[self._client_id]
        return self.project_memory

    def get_project_entities(self, project_name: str) -> str:
        """
        Get all entities in a project.
        
        Args:
            project_name: Name of the project
            
        Returns:
            JSON string with the entities
        """
        self._ensure_initialized()
        
        try:
            # Temporarily set project name for context
            original_project = self.default_project_name
            self.set_project_name(project_name)
            
            project_memory = self.get_client_project_memory()
            entities = []
            
            # First try: Get entities using project_container.get_container_components
            if hasattr(project_memory, "project_container") and hasattr(project_memory.project_container, "get_container_components"):
                try:
                    result_json = project_memory.project_container.get_container_components(project_name)
                    result = json.loads(result_json)
                    
                    if "error" not in result and result.get("components", []):
                        # Transform to standard format with entities array
                        result["entities"] = result.pop("components", [])
                        self.set_project_name(original_project)
                        return json.dumps(result)
                except Exception:
                    # Silently continue to fallback methods
                    pass
            
            # Method 1: Find entities with project property matching project name
            query1 = """
            MATCH (e:Entity)
            WHERE e.project = $project_name
            RETURN DISTINCT e
            ORDER BY e.name
            """
            
            records1 = self.base_manager.safe_execute_read_query(
                query1,
                {"project_name": project_name}
            )
            
            # Process the results from query1
            for record in records1:
                if 'e' in record:
                    entity = dict(record['e'].items())
                    # Convert any DateTime objects to strings
                    for key, value in entity.items():
                        if hasattr(value, 'iso_format'):  # Check if it's a datetime-like object
                            entity[key] = value.iso_format()
                    entities.append(entity)
            
            # Method 2: Find entities with names starting with project name prefix
            query2 = """
            MATCH (e:Entity)
            WHERE e.name STARTS WITH $name_prefix
            RETURN DISTINCT e
            ORDER BY e.name
            """
            
            records2 = self.base_manager.safe_execute_read_query(
                query2,
                {"name_prefix": project_name + "_"}
            )
            
            # Process the results from query2
            for record in records2:
                if 'e' in record:
                    entity = dict(record['e'].items())
                    # Convert any DateTime objects to strings
                    for key, value in entity.items():
                        if hasattr(value, 'iso_format'):  # Check if it's a datetime-like object
                            entity[key] = value.iso_format()
                    # Check if already added
                    if not any(e.get('name') == entity.get('name') for e in entities):
                        entities.append(entity)
            
            # Method 3: Find entities with container property matching project name
            query3 = """
            MATCH (e:Entity)
            WHERE e.container = $project_name
            RETURN DISTINCT e
            ORDER BY e.name
            """
            
            records3 = self.base_manager.safe_execute_read_query(
                query3,
                {"project_name": project_name}
            )
            
            # Process the results from query3
            for record in records3:
                if 'e' in record:
                    entity = dict(record['e'].items())
                    # Convert any DateTime objects to strings
                    for key, value in entity.items():
                        if hasattr(value, 'iso_format'):  # Check if it's a datetime-like object
                            entity[key] = value.iso_format()
                    # Check if already added
                    if not any(e.get('name') == entity.get('name') for e in entities):
                        entities.append(entity)
            
            # Reset original project context
            self.set_project_name(original_project)
            
            return json.dumps({
                "entities": entities,
                "project": project_name
            })
                
        except Exception as e:
            if self.logger:
                self.logger.error(f"Error getting project entities: {str(e)}")
            return json.dumps({
                "error": f"Failed to get project entities: {str(e)}",
                "entities": []
            })

    def get_lesson_container(self, container_id: str) -> str:
        """
        Retrieve a lesson container by ID or name.
        
        Args:
            container_id: ID or name of the container to retrieve
            
        Returns:
            JSON string with the container information
        """
        try:
            self._ensure_initialized()
            
            # Delegate to the lesson container component via lesson_memory facade
            response = self.lesson_memory.container.get_container(container_id)
            
            # Parse the response
            result = json.loads(response)
            
            # Return response with "status":"success" format for consistency with tests
            if "error" not in result:
                return json.dumps({
                    "status": "success",
                    "container": result.get("container", result)
                })
                
            return json.dumps({
                "status": "error",
                "error": result.get("error", f"Lesson container '{container_id}' not found")
            })
            
        except Exception as e:
            if self.logger:
                self.logger.error(f"Error retrieving lesson container: {str(e)}")
            return json.dumps({
                "status": "error",
                "error": f"Failed to retrieve lesson container: {str(e)}"
            })

    def get_entity_observations(self, entity_name: str, observation_type: Optional[str] = None) -> str:
        """
        Get observations for an entity from the knowledge graph.
        This is an alias for get_observations to maintain backward compatibility.
        
        Args:
            entity_name: The name of the entity
            observation_type: Optional type of observation to filter by
            
        Returns:
            JSON string with the observations
        """
        return self.get_observations(entity_name, observation_type)

    def get_entities(self, query: Optional[str] = None, limit: int = 10) -> List[Dict[str, Any]]:
        """
        Get entities from the knowledge graph.
        
        Args:
            query: Optional search query to filter entities
            limit: Maximum number of entities to return
            
        Returns:
            List of entity dictionaries
        """
        self._ensure_initialized()
        
        try:
            # If query is provided, search for entities
            if query:
                search_result = json.loads(self.search_nodes(query, limit))
                if "results" in search_result:
                    return search_result["results"]
                return []
            
            # Otherwise, get all entities
            query = """
            MATCH (e:Entity)
            RETURN e
            LIMIT $limit
            """
            
            records = self.base_manager.safe_execute_read_query(
                query,
                {"limit": limit}
            )
            
            entities = []
            if records:
                for record in records:
                    # Handle different record types
                    entity = None
                    if isinstance(record, dict) and "e" in record:
                        # Dictionary access for dict type records
                        entity = record["e"]
                    elif hasattr(record, "get") and callable(record.get):
                        # Use get method for Neo4j Record objects
                        entity = record.get("e")
                    
                    if entity:
                        # Handle different entity types
                        if hasattr(entity, "items") and callable(entity.items):
                            # Convert Neo4j Node to dictionary
                            entities.append(dict(entity.items()))
                        elif isinstance(entity, dict):
                            # Already a dictionary
                            entities.append(entity)
            
            return entities
        except Exception as e:
            if self.logger and hasattr(self.logger, "error") and callable(self.logger.error):
                self.logger.error(f"Error retrieving entities: {str(e)}")
            return []

    def add_observation(self, observation: Dict[str, Any]) -> str:
        """
        Add a single observation to an entity.
        This is a convenience method that wraps add_observations.
        
        Args:
            observation: Dictionary with observation data
            
        Returns:
            JSON string with the result
        """
        self._ensure_initialized()
        # Call add_observations with a list containing the single observation
        return self.observation_manager.add_observations([observation])
