"""
Graph Memory Module

This module provides functionality for interacting with the knowledge graph.
It exposes a set of managers for different graph memory operations, as well
as a facade class that maintains backward compatibility with the original API.
"""

from typing import Any, Dict, List, Optional, Union

from src.utils import dict_to_json
from src.graph_memory.base_manager import BaseManager
from src.graph_memory.entity_manager import EntityManager
from src.graph_memory.relation_manager import RelationManager
from src.graph_memory.observation_manager import ObservationManager
from src.graph_memory.search_manager import SearchManager
from src.graph_memory.embedding_adapter import EmbeddingAdapter

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
        self.base_manager = BaseManager(
            logger=logger,
            neo4j_uri=neo4j_uri,
            neo4j_username=neo4j_username,
            neo4j_password=neo4j_password,
            database=database,
            embedding_index_name=embedding_index_name
        )
        
        # Initialize embedding adapter
        self.embedding_adapter = EmbeddingAdapter(logger=logger)
        
        # Initialize specialized managers
        self.entity_manager = EntityManager(self.base_manager)
        self.relation_manager = RelationManager(self.base_manager)
        self.observation_manager = ObservationManager(self.base_manager)
        self.search_manager = SearchManager(self.base_manager)
        
        # Store configuration
        self.embedding_api_key = embedding_api_key
        self.embedding_model = embedding_model
        self.logger = logger
    
    # Connection Management
    
    def initialize(self) -> bool:
        """
        Initialize connections to Neo4j and embedding services.
        
        Returns:
            True if initialization successful, False otherwise
        """
        # Initialize embedding adapter
        if not self.embedding_adapter.init_embedding_manager(
            api_key=self.embedding_api_key,
            model_name=self.embedding_model
        ):
            if self.logger:
                self.logger.error("Failed to initialize embedding manager")
            return False
        
        # Initialize base manager and Neo4j
        return self.base_manager.initialize(self.embedding_adapter.embedding_manager)
    
    def close(self) -> None:
        """Close connections to Neo4j and clean up resources."""
        self.base_manager.close()
    
    def check_connection(self) -> bool:
        """
        Check if the Neo4j connection is working.
        
        Returns:
            True if connection is working, False otherwise
        """
        return self.base_manager.check_connection()
    
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
        return self.relation_manager.delete_relation(from_entity, to_entity, relation_type)
    
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
