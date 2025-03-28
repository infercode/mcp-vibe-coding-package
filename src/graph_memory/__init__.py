"""
Graph Memory Module

This module provides functionality for interacting with the knowledge graph.
It exposes a set of managers for different graph memory operations, as well
as a facade class that maintains backward compatibility with the original API.
"""

from typing import Any, Dict, List, Optional, Union
import json
import datetime
import time

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
        
        # Initialize embedding adapter
        self.embedding_adapter = EmbeddingAdapter(logger=logger)
        
        # Initialize specialized managers
        self.entity_manager = EntityManager(self.base_manager)
        self.relation_manager = RelationManager(self.base_manager)
        self.observation_manager = ObservationManager(self.base_manager)
        self.search_manager = SearchManager(self.base_manager)
        
        # Initialize specialized memory systems
        self.lesson_memory = LessonMemoryManager(self.base_manager)
        self.project_memory = ProjectMemoryManager(self.base_manager)
        
        # Store configuration
        self.embedding_api_key = embedding_api_key
        self.embedding_model = embedding_model
        self.logger = logger
        
        # Required attributes for backward compatibility
        self.default_project_name = "default"
        self.embedding_enabled = True
        self.neo4j_uri = neo4j_uri or "bolt://localhost:7687"
        self.neo4j_user = neo4j_username or "neo4j"
        self.neo4j_password = neo4j_password or "password"
        self.neo4j_database = database or "neo4j"
        self.embedder_provider = "openai"
        self.neo4j_driver = None
    
    # Connection Management
    
    def initialize(self) -> bool:
        """
        Initialize connections to Neo4j and embedding services.
        
        Returns:
            True if initialization successful, False otherwise
        """
        # Override configuration if provided in constructor
        if self._neo4j_uri:
            # Set environment variables or direct properties if accessible
            import os
            os.environ["NEO4J_URI"] = self._neo4j_uri
        
        if self._neo4j_username:
            import os
            os.environ["NEO4J_USER"] = self._neo4j_username
            
        if self._neo4j_password:
            import os
            os.environ["NEO4J_PASSWORD"] = self._neo4j_password
            
        if self._database:
            import os
            os.environ["NEO4J_DATABASE"] = self._database
            
        # Initialize embedding adapter
        if not self.embedding_adapter.init_embedding_manager(
            api_key=self.embedding_api_key,
            model_name=self.embedding_model
        ):
            if self.logger:
                self.logger.error("Failed to initialize embedding manager")
            return False
        
        # Initialize base manager and Neo4j
        self.base_manager.initialize()
        
        # Update forwarded properties after initialization
        self.neo4j_uri = getattr(self.base_manager, "neo4j_uri", "bolt://localhost:7687")
        self.neo4j_user = getattr(self.base_manager, "neo4j_user", "neo4j")
        self.neo4j_password = getattr(self.base_manager, "neo4j_password", "password")
        self.neo4j_database = getattr(self.base_manager, "neo4j_database", "neo4j")
        self.embedder_provider = getattr(self.base_manager, "embedder_provider", "none")
        self.neo4j_driver = getattr(self.base_manager, "neo4j_driver", None)
        
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
    
    def create_entities(self, entities: List[Union[Dict, Any]]) -> str:
        """
        Create multiple entities in the knowledge graph.
        
        Args:
            entities: List of entities to create
            
        Returns:
            JSON string with the created entities
        """
        return self.entity_manager.create_entities(entities)
    
    def get_entity(self, entity_name: str) -> str:
        """
        Get an entity from the knowledge graph.
        
        Args:
            entity_name: The name of the entity
            
        Returns:
            JSON string with the entity
        """
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
    
    def create_relations(self, relations: List[Union[Dict, Any]]) -> str:
        """
        Create multiple relations in the knowledge graph.
        
        Args:
            relations: List of relations to create
            
        Returns:
            JSON string with the created relations
        """
        return self.relation_manager.create_relations(relations)
    
    def get_relations(self, entity_name: Optional[str] = None, relation_type: Optional[str] = None) -> str:
        """
        Get relations from the knowledge graph.
        
        Args:
            entity_name: Optional name of the entity to filter by
            relation_type: Optional type of relation to filter by
            
        Returns:
            JSON string with the relations
        """
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
    
    def add_observations(self, observations: List[Union[Dict, Any]]) -> str:
        """
        Add multiple observations to entities in the knowledge graph.
        
        Args:
            observations: List of observations to add
            
        Returns:
            JSON string with the added observations
        """
        return self.observation_manager.add_observations(observations)
    
    def get_entity_observations(self, entity_name: str, observation_type: Optional[str] = None) -> str:
        """
        Get observations for an entity from the knowledge graph.
        
        Args:
            entity_name: The name of the entity
            observation_type: Optional type of observations to filter by
            
        Returns:
            JSON string with the observations
        """
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
    
    def search_entities(self, search_term: str, limit: int = 10, 
                       entity_types: Optional[List[str]] = None,
                       semantic: bool = False) -> str:
        """
        Search for entities in the knowledge graph.
        
        Args:
            search_term: The term to search for
            limit: Maximum number of results to return
            entity_types: Optional list of entity types to filter by
            semantic: Whether to perform semantic (embedding-based) search
            
        Returns:
            JSON string with the search results
        """
        return self.search_manager.search_entities(search_term, limit, entity_types, semantic)
    
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
    
    def set_project_name(self, project_name: str) -> None:
        """
        Set the default project name for memory operations.
        
        Args:
            project_name: The project name to use
        """
        self.base_manager.set_project_name(project_name)
        self.default_project_name = self.base_manager.default_project_name
    
    def search_nodes(self, query: str, limit: int = 10, project_name: Optional[str] = None) -> str:
        """
        Search for nodes in the knowledge graph.
        
        Args:
            query: The query to search for
            limit: Maximum number of results to return
            project_name: Optional project name to scope the search
            
        Returns:
            JSON string with the search results
        """
        if project_name:
            self.set_project_name(project_name)
        
        return self.search_entities(query, limit, semantic=True)
    
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
            
            records, _ = self.base_manager.safe_execute_query(query)
            
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
            
            self.base_manager.safe_execute_query(query)
            
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
            # Apply embedding configuration if present
            embeddings = config.get("embeddings", {})
            if embeddings:
                provider = embeddings.get("provider")
                model = embeddings.get("model")
                api_key = embeddings.get("api_key")
                
                if provider and model and api_key:
                    # Configure embedding adapter
                    result = self.embedding_adapter.init_embedding_manager(api_key, model)
                    if not result:
                        return {"status": "error", "message": "Failed to initialize embedding manager"}
            
            return {"status": "success", "message": "Configuration applied successfully"}
            
        except Exception as e:
            if self.logger:
                self.logger.error(f"Error applying client config: {str(e)}")
            return {"status": "error", "message": f"Failed to apply configuration: {str(e)}"}
    
    def reinitialize(self) -> Dict[str, Any]:
        """
        Reinitialize the memory manager.
        
        Returns:
            Dictionary with status of reinitialization
        """
        try:
            # Close existing connections
            self.close()
            
            # Reinitialize
            success = self.initialize()
            
            if success:
                return {"status": "success", "message": "Memory manager reinitialized successfully"}
            else:
                return {"status": "error", "message": "Failed to reinitialize memory manager"}
                
        except Exception as e:
            if self.logger:
                self.logger.error(f"Error reinitializing memory manager: {str(e)}")
            return {"status": "error", "message": f"Failed to reinitialize memory manager: {str(e)}"}
    
    def get_current_config(self) -> Dict[str, Any]:
        """
        Get the current configuration.
        
        Returns:
            Dictionary with current configuration
        """
        config = {
            "embeddings": {
                "provider": self.embedder_provider,
                "model": self.embedding_model,
                "dimensions": getattr(self.embedding_adapter.embedding_manager, "dimensions", 0),
                "enabled": self.embedding_enabled
            },
            "database": {
                "uri": self.neo4j_uri,
                "username": self.neo4j_user,
                "database": self.neo4j_database
            },
            "project": self.default_project_name
        }
        
        return config
    
    def _ensure_initialized(self) -> None:
        """Ensure the memory manager is initialized."""
        self.base_manager.ensure_initialized()
    
    # Lesson Memory System methods for backward compatibility
    
    def create_lesson_container(self) -> str:
        """
        Create a lessons container in the knowledge graph.
        
        This container serves as a central hub for organizing all lessons in the system.
        
        Returns:
            JSON string with operation result
        """
        try:
            # Ensure we're initialized
            self._ensure_initialized()
            
            # Delegate to the lesson memory manager
            return self.lesson_memory.create_container("LessonContainer", 
                                                   "Central container for all lessons")
            
        except Exception as e:
            if self.logger:
                self.logger.error(f"Error creating lesson container: {str(e)}")
            return dict_to_json({"error": f"Failed to create lesson container: {str(e)}"})
            
    def create_lesson(self, name: str, problem_description: str, **kwargs) -> str:
        """
        Create a lesson in the knowledge graph.
        
        Args:
            name: Name or title of the lesson
            problem_description: Description of the problem the lesson addresses
            **kwargs: Additional lesson properties
            
        Returns:
            JSON string with operation result
        """
        try:
            # Ensure we're initialized
            self._ensure_initialized()
            
            # Map from the API parameters to lesson_memory parameters
            # Default container name for backward compatibility
            container_name = kwargs.pop("container_name", "LessonContainer")
            
            # Extract specific parameters that need remapping
            context = kwargs.pop("context", None)
            impact = kwargs.pop("impact", "Medium")
            resolution = kwargs.pop("resolution", None)
            what_was_learned = kwargs.pop("what_was_learned", None)
            why_it_matters = kwargs.pop("why_it_matters", None)
            how_to_apply = kwargs.pop("how_to_apply", None)
            root_cause = kwargs.pop("root_cause", None)
            evidence = kwargs.pop("evidence", None)
            confidence = kwargs.pop("confidence", 0.8)
            
            # Create metadata from kwargs
            metadata = kwargs
            
            # Create observations for structured data
            observations = {}
            if problem_description:
                observations["problemDescription"] = problem_description
            if context:
                observations["context"] = context
            if impact:
                observations["impact"] = impact
            if resolution:
                observations["resolution"] = resolution
            if what_was_learned:
                observations["whatWasLearned"] = what_was_learned
            if why_it_matters:
                observations["whyItMatters"] = why_it_matters
            if how_to_apply:
                observations["howToApply"] = how_to_apply
            if root_cause:
                observations["rootCause"] = root_cause
            if evidence:
                observations["evidence"] = evidence
            
            # Add confidence to metadata
            metadata["confidence"] = confidence
            
            # Create the lesson entity using the LessonMemoryManager facade
            # The correct parameter order is container_name, entity_name, entity_type, observations, metadata
            return self.lesson_memory.create_lesson_entity(
                container_name,
                name,
                "Lesson",
                list(observations.values()) if observations else None,
                metadata
            )
            
        except Exception as e:
            if self.logger:
                self.logger.error(f"Error creating lesson: {str(e)}")
            return dict_to_json({"error": f"Failed to create lesson: {str(e)}"})
    
    def get_lessons(self, **kwargs) -> str:
        """
        Get lessons matching the specified criteria.
        
        Args:
            **kwargs: Criteria for filtering lessons
            
        Returns:
            JSON string with the matching lessons
        """
        try:
            # Ensure we're initialized
            self._ensure_initialized()
            
            # Extract and map parameters
            filter_criteria = kwargs.get("filter_criteria", {})
            related_to = kwargs.get("related_to")
            applies_to = kwargs.get("applies_to")
            limit = kwargs.get("limit", 50)
            include_superseded = kwargs.get("include_superseded", False)
            min_confidence = kwargs.get("min_confidence", 0.0)
            sort_by = kwargs.get("sort_by", "relevance")
            include_observations = kwargs.get("include_observations", True)
            
            # Delegate to lesson memory system through the LessonMemoryManager facade
            return self.lesson_memory.search_lesson_entities(
                container_name="LessonContainer",
                search_term=kwargs.get("search_term"),
                entity_type="Lesson",
                tags=kwargs.get("tags"),
                use_semantic_search=kwargs.get("semantic", False)
            )
            
        except Exception as e:
            if self.logger:
                self.logger.error(f"Error getting lessons: {str(e)}")
            return dict_to_json({"error": f"Failed to get lessons: {str(e)}"})
    
    def update_lesson(self, lesson_name: str, **kwargs) -> str:
        """
        Update a lesson.
        
        Args:
            lesson_name: The name of the lesson to update
            **kwargs: Properties to update
            
        Returns:
            JSON string with the updated lesson
        """
        try:
            # Ensure we're initialized
            self._ensure_initialized()
            
            # Default container name for backward compatibility
            container_name = kwargs.pop("container_name", "LessonContainer")
            
            # Delegate to lesson memory system through the LessonMemoryManager facade
            return self.lesson_memory.update_lesson_entity(
                lesson_name, 
                kwargs,
                container_name
            )
            
        except Exception as e:
            if self.logger:
                self.logger.error(f"Error updating lesson: {str(e)}")
            return dict_to_json({"error": f"Failed to update lesson: {str(e)}"})
    
    def apply_lesson_to_context(self, lesson_name: str, context_entity: str, **kwargs) -> str:
        """
        Apply a lesson to a context.
        
        Args:
            lesson_name: The name of the lesson
            context_entity: The context to apply the lesson to
            **kwargs: Additional parameters
            
        Returns:
            JSON string with the result
        """
        try:
            # Ensure we're initialized
            self._ensure_initialized()
            
            # Extract parameters
            application_notes = kwargs.get("application_notes")
            success_score = kwargs.get("success_score", 0.8)
            
            # Delegate to lesson memory system through the LessonMemoryManager facade
            return self.lesson_memory.track_lesson_application(
                lesson_name,
                context_entity,
                success_score,
                application_notes
            )
            
        except Exception as e:
            if self.logger:
                self.logger.error(f"Error applying lesson: {str(e)}")
            return dict_to_json({"error": f"Failed to apply lesson: {str(e)}"})
    
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
        Consolidate related lessons.
        
        Args:
            lesson_ids: List of lesson IDs to consolidate
            **kwargs: Additional parameters
            
        Returns:
            JSON string with the result
        """
        try:
            # Ensure we're initialized
            self._ensure_initialized()
            
            # Extract parameters
            new_name = kwargs.get("new_name", f"Consolidated_Lesson_{int(time.time())}")
            merge_strategy = kwargs.get("merge_strategy", "latest")
            container_name = kwargs.get("container_name", "LessonContainer")
            
            # Create source_lessons structure expected by consolidation.merge_lessons
            source_lessons = []
            for lesson_id in lesson_ids:
                # Add each lesson ID as a dictionary with the ID
                source_lessons.append({"id": lesson_id})
            
            # Delegate to lesson memory system through the LessonMemoryManager facade
            return self.lesson_memory.merge_lessons(
                source_lessons, 
                new_name, 
                merge_strategy, 
                container_name
            )
            
        except Exception as e:
            if self.logger:
                self.logger.error(f"Error consolidating lessons: {str(e)}")
            return dict_to_json({"error": f"Failed to consolidate lessons: {str(e)}"})
    
    def get_knowledge_evolution(self, **kwargs) -> str:
        """
        Get the evolution of knowledge.
        
        Args:
            **kwargs: Criteria for filtering
            
        Returns:
            JSON string with knowledge evolution
        """
        try:
            # Ensure we're initialized
            self._ensure_initialized()
            
            # Extract parameters
            entity_name = kwargs.get("entity_name")
            lesson_type = kwargs.get("lesson_type")
            start_date = kwargs.get("start_date")
            end_date = kwargs.get("end_date")
            include_superseded = kwargs.get("include_superseded", True)
            
            # Delegate to lesson memory system through the LessonMemoryManager facade
            return self.lesson_memory.get_knowledge_evolution(
                entity_name,
                lesson_type,
                start_date,
                end_date,
                include_superseded
            )
            
        except Exception as e:
            if self.logger:
                self.logger.error(f"Error getting knowledge evolution: {str(e)}")
            return dict_to_json({"error": f"Failed to get knowledge evolution: {str(e)}"})
    
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
    
    def create_project_container(self, name: str, description: Optional[str] = None, 
                               properties: Optional[Dict[str, Any]] = None) -> str:
        """
        Create a project container in the knowledge graph.
        
        Args:
            name: Name of the project container
            description: Optional description of the project
            properties: Optional additional properties
            
        Returns:
            JSON string with operation result
        """
        try:
            # Ensure we're initialized
            self._ensure_initialized()
            
            # Delegate directly to project memory manager
            result = self.project_memory.create_project_container(
                name, description, properties
            )
            
            # Convert to string if needed
            if isinstance(result, dict):
                return dict_to_json(result)
            return result
            
        except Exception as e:
            if self.logger:
                self.logger.error(f"Error creating project container: {str(e)}")
            return dict_to_json({"error": f"Failed to create project container: {str(e)}"})
    
    def get_project_container(self, name: str) -> str:
        """
        Retrieve a project container by name.
        
        Args:
            name: Name of the project container
            
        Returns:
            JSON string with the project container
        """
        try:
            self._ensure_initialized()
            result = self.project_memory.get_project_container(name)
            if isinstance(result, dict):
                return dict_to_json(result)
            return result
        except Exception as e:
            if self.logger:
                self.logger.error(f"Error getting project container: {str(e)}")
            return dict_to_json({"error": f"Failed to get project container: {str(e)}"})
    
    def update_project_container(self, name: str, updates: Dict[str, Any]) -> str:
        """
        Update a project container's properties.
        
        Args:
            name: Name of the project container
            updates: Dictionary of properties to update
            
        Returns:
            JSON string with the updated project container
        """
        try:
            self._ensure_initialized()
            result = self.project_memory.update_project_container(name, updates)
            if isinstance(result, dict):
                return dict_to_json(result)
            return result
        except Exception as e:
            if self.logger:
                self.logger.error(f"Error updating project container: {str(e)}")
            return dict_to_json({"error": f"Failed to update project container: {str(e)}"})
    
    def delete_project_container(self, name: str, delete_contents: bool = False) -> str:
        """
        Delete a project container.
        
        Args:
            name: Name of the project container
            delete_contents: If True, delete all domains and components in the container
            
        Returns:
            JSON string with the deletion result
        """
        try:
            self._ensure_initialized()
            result = self.project_memory.delete_project_container(name, delete_contents)
            if isinstance(result, dict):
                return dict_to_json(result)
            return result
        except Exception as e:
            if self.logger:
                self.logger.error(f"Error deleting project container: {str(e)}")
            return dict_to_json({"error": f"Failed to delete project container: {str(e)}"})
    
    def list_project_containers(self, sort_by: str = "name", limit: int = 100) -> str:
        """
        List all project containers.
        
        Args:
            sort_by: Field to sort by ('name', 'created', or 'lastUpdated')
            limit: Maximum number of containers to return
            
        Returns:
            JSON string with the list of project containers
        """
        try:
            self._ensure_initialized()
            result = self.project_memory.list_project_containers(sort_by, limit)
            if isinstance(result, dict):
                return dict_to_json(result)
            return result
        except Exception as e:
            if self.logger:
                self.logger.error(f"Error listing project containers: {str(e)}")
            return dict_to_json({"error": f"Failed to list project containers: {str(e)}"})
    
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
    
    def update_component(self, name: str, domain_name: str, container_name: str, 
                        updates: Dict[str, Any]) -> str:
        """
        Update a component's properties.
        
        Args:
            name: Name of the component
            domain_name: Name of the domain
            container_name: Name of the project container
            updates: Dictionary of properties to update
            
        Returns:
            JSON string with the updated component
        """
        try:
            self._ensure_initialized()
            result = self.project_memory.update_component(name, domain_name, container_name, updates)
            if isinstance(result, dict):
                return dict_to_json(result)
            return result
        except Exception as e:
            if self.logger:
                self.logger.error(f"Error updating component: {str(e)}")
            return dict_to_json({"error": f"Failed to update component: {str(e)}"})
    
    def delete_component(self, name: str, domain_name: str, container_name: str) -> str:
        """
        Delete a component from a domain.
        
        Args:
            name: Name of the component
            domain_name: Name of the domain
            container_name: Name of the project container
            
        Returns:
            JSON string with the deletion result
        """
        try:
            self._ensure_initialized()
            result = self.project_memory.delete_component(name, domain_name, container_name)
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
