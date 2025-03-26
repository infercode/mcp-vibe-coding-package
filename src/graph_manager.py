import json
import os
from typing import Any, Dict, List, Optional, Union, cast, TYPE_CHECKING

# No imports in TYPE_CHECKING block - we'll use string literals for type annotations

# Try to import mem0 with fallbacks for when types aren't available during IDE linting
try:
    from mem0 import Memory  # type: ignore
except ImportError:
    # Stubs for type checking
    class Memory:
        @classmethod
        def from_config(cls, config_dict=None):
            return cls()
            
        def __init__(self, config=None):
            pass

        def search(self, **kwargs):
            return {"results": []}
            
        def add(self, **kwargs):
            pass
            
        def get_all(self, **kwargs):
            return []
            
        def delete_all(self, **kwargs):
            pass

# Import Neo4j driver for direct operations
try:
    from neo4j import GraphDatabase, RoutingControl  # type: ignore
except ImportError:
    # Stubs for type checking when Neo4j driver is not available
    class GraphDatabase:
        @staticmethod
        def driver(*args, **kwargs):
            return None
    
    class RoutingControl:
        READ = "READ"
        WRITE = "WRITE"

from src.logger import Logger, get_logger
from src.types import Entity, KnowledgeGraph, Observation, Relation
from src.utils import dict_to_json, extract_error, generate_id


class GraphMemoryManager:
    """Memory manager for the knowledge graph using mem0ai with graph capabilities."""

    def __init__(self, logger: Optional[Logger] = None):
        """
        Initialize the memory manager.
        
        Args:
            logger: Optional logger instance
        """
        self.logger = logger or get_logger()
        self.initialized = False
        self.memory = None
        self.neo4j_driver = None
        
        # Get config from environment variables
        self.neo4j_uri = os.environ.get("NEO4J_URI", "bolt://localhost:7687")
        self.neo4j_user = os.environ.get("NEO4J_USER", "neo4j")
        self.neo4j_password = os.environ.get("NEO4J_PASSWORD", "password")
        self.neo4j_database = os.environ.get("NEO4J_DATABASE", "neo4j")
        
        # Get embedder provider configuration
        self.embedder_provider = os.environ.get("EMBEDDER_PROVIDER", "openai").lower()
        
        # Initialize embedding model configuration based on provider
        if self.embedder_provider == "openai":
            self.embedding_model = os.environ.get("EMBEDDING_MODEL", "text-embedding-3-small")
            self.openai_api_key = os.environ.get("OPENAI_API_KEY", "")
            self.openai_api_base = os.environ.get("OPENAI_API_BASE", "")
            self.embedding_dims = int(os.environ.get("EMBEDDING_DIMS", "1536"))
        elif self.embedder_provider == "huggingface":
            self.huggingface_model = os.environ.get("HUGGINGFACE_MODEL", "sentence-transformers/all-mpnet-base-v2")
            self.huggingface_model_kwargs = json.loads(os.environ.get("HUGGINGFACE_MODEL_KWARGS", '{"device":"cpu"}'))
            self.embedding_dims = int(os.environ.get("EMBEDDING_DIMS", "768"))
        elif self.embedder_provider == "ollama":
            self.ollama_model = os.environ.get("OLLAMA_MODEL", "llama2")
            self.ollama_base_url = os.environ.get("OLLAMA_BASE_URL", "http://localhost:11434")
            self.embedding_dims = int(os.environ.get("EMBEDDING_DIMS", "4096"))
        elif self.embedder_provider == "azure":
            self.azure_api_key = os.environ.get("AZURE_API_KEY", "")
            self.azure_endpoint = os.environ.get("AZURE_ENDPOINT", "")
            self.azure_deployment = os.environ.get("AZURE_DEPLOYMENT", "")
            self.embedding_dims = int(os.environ.get("EMBEDDING_DIMS", "1536"))
        elif self.embedder_provider == "lmstudio":
            self.lmstudio_base_url = os.environ.get("LMSTUDIO_BASE_URL", "http://localhost:1234")
            self.embedding_dims = int(os.environ.get("EMBEDDING_DIMS", "4096"))
        
        # Default user ID for MCP context where user ID isn't explicitly provided
        self.default_user_id = "default_user"

    def initialize(self) -> None:
        """Initialize the memory manager and connect to the graph database."""
        if self.initialized:
            return

        try:
            # Configure Neo4j graph store
            graph_store_config = {
                "graph_store": {
                    "provider": "neo4j",
                    "config": {
                        "url": self.neo4j_uri,
                        "username": self.neo4j_user,
                        "password": self.neo4j_password,
                        "database": self.neo4j_database
                    }
                }
            }
            
            # Configure embedding based on provider
            embedding_config = self._get_embedding_config()
            
            # Combine configurations
            config = {**graph_store_config, **embedding_config}
            
            # Initialize mem0 memory
            self.memory = Memory.from_config(config_dict=config)
            
            # Initialize direct Neo4j driver for operations not supported by mem0
            self.neo4j_driver = GraphDatabase.driver(
                self.neo4j_uri,
                auth=(self.neo4j_user, self.neo4j_password)
            )
            
            # Test the Neo4j connection
            self._test_neo4j_connection()
            
            self.initialized = True
            self.logger.info(f"Memory graph manager initialized successfully with {self.embedder_provider} embedder")
        except Exception as e:
            self.logger.error(f"Failed to initialize memory graph manager: {str(e)}")
            raise e
            
    def _get_embedding_config(self) -> Dict[str, Any]:
        """Get the embedding configuration based on the selected provider."""
        if self.embedder_provider == "openai":
            config = {
                "embedder": {
                    "provider": "openai",
                    "config": {
                        "api_key": self.openai_api_key,
                        "model": self.embedding_model
                    }
                }
            }
            # Add API base if provided
            if hasattr(self, "openai_api_base") and self.openai_api_base:
                config["embedder"]["config"]["api_base"] = self.openai_api_base
            return config
            
        elif self.embedder_provider == "huggingface":
            return {
                "embedder": {
                    "provider": "huggingface",
                    "config": {
                        "model": self.huggingface_model,
                        "model_kwargs": self.huggingface_model_kwargs
                    }
                }
            }
            
        elif self.embedder_provider == "ollama":
            return {
                "embedder": {
                    "provider": "ollama",
                    "config": {
                        "model": self.ollama_model,
                        "ollama_base_url": self.ollama_base_url
                    }
                }
            }
            
        elif self.embedder_provider == "azure":
            return {
                "embedder": {
                    "provider": "azure",
                    "config": {
                        "api_key": self.azure_api_key,
                        "azure_kwargs": {
                            "api_version": "2023-05-15",
                            "azure_deployment": self.azure_deployment,
                            "azure_endpoint": self.azure_endpoint
                        }
                    }
                }
            }
            
        elif self.embedder_provider == "lmstudio":
            return {
                "embedder": {
                    "provider": "lmstudio",
                    "config": {
                        "lmstudio_base_url": self.lmstudio_base_url
                    }
                }
            }
            
        # Default to OpenAI if provider is not recognized
        self.logger.warn(f"Unknown embedder provider: {self.embedder_provider}. Falling back to OpenAI.")
        return {
            "embedder": {
                "provider": "openai",
                "config": {
                    "api_key": os.environ.get("OPENAI_API_KEY", ""),
                    "model": "text-embedding-3-small"
                }
            }
        }

    def _test_neo4j_connection(self) -> None:
        """Test the Neo4j connection by running a simple query."""
        if not self.neo4j_driver:
            return
        
        try:
            result_data = self.neo4j_driver.execute_query(
                "RETURN 'Connection successful' AS message",
                database_=self.neo4j_database
            )
            records = result_data[0] if result_data and len(result_data) > 0 else []
            message = records[0]["message"] if records and len(records) > 0 else "Unknown"
            self.logger.info(f"Neo4j direct connection test: {message}")
        except Exception as e:
            self.logger.error(f"Neo4j connection test failed: {str(e)}")
            # Don't raise - we can still use mem0 even if direct connection fails

    def close(self) -> None:
        """Close the Neo4j driver connection."""
        if self.neo4j_driver:
            self.neo4j_driver.close()
            self.logger.info("Neo4j driver closed")

    def create_entities(self, entities: List[Union[Dict, Any]]) -> str:
        """
        Create multiple entities in the knowledge graph.
        
        Args:
            entities: List of entities to create
            
        Returns:
            JSON string with the created entities
        """
        try:
            self._ensure_initialized()
            
            created_entities = []
            
            for entity in entities:
                # Process entity for mem0 format
                entity_dict = self._convert_to_dict(entity)
                
                # Create memory string from entity
                entity_name = entity_dict.get("name", "")
                entity_type = entity_dict.get("entityType", "")
                observations = entity_dict.get("observations", [])
                
                if entity_name and entity_type:
                    memory_str = f"{entity_name} is a {entity_type}."
                    
                    # Add base entity information with metadata
                    if self.memory:
                        self.memory.add(
                            memory=memory_str, 
                            user_id=self.default_user_id,
                            metadata={
                                "type": "entity",
                                "entity_name": entity_name,
                                "entity_type": entity_type
                            }
                        )
                    
                    # Add observations if present
                    if observations:
                        for obs in observations:
                            # Add each observation as a separate memory with metadata
                            memory_obs = f"{entity_name} {obs}"
                            if self.memory:
                                self.memory.add(
                                    memory=memory_obs, 
                                    user_id=self.default_user_id,
                                    metadata={
                                        "type": "observation",
                                        "entity_name": entity_name,
                                        "entity_type": entity_type
                                    }
                                )
                                
                    # Also create entity directly in Neo4j for operations not supported by mem0
                    if self.neo4j_driver:
                        self._create_entity_in_neo4j(entity_name, entity_type, observations)
                        
                    created_entities.append(entity_dict)
                    self.logger.info(f"Created entity: {entity_name}")
            
            return dict_to_json({"created": created_entities})
        except Exception as e:
            error_msg = f"Error creating entities: {str(e)}"
            self.logger.error(error_msg)
            return dict_to_json({"error": error_msg})

    def _create_entity_in_neo4j(self, name: str, entity_type: str, observations: List[str]) -> None:
        """Create an entity directly in Neo4j."""
        if not self.neo4j_driver:
            return
        
        try:
            # Create the entity node
            query = """
            MERGE (e:Entity {name: $name})
            SET e.entityType = $entity_type
            RETURN e
            """
            self.neo4j_driver.execute_query(
                query,
                name=name,
                entity_type=entity_type,
                database_=self.neo4j_database
            )
            
            # Add observations if present
            if observations:
                for observation in observations:
                    obs_query = """
                    MATCH (e:Entity {name: $name})
                    MERGE (o:Observation {content: $content})
                    MERGE (e)-[:HAS_OBSERVATION]->(o)
                    """
                    self.neo4j_driver.execute_query(
                        obs_query,
                        name=name,
                        content=observation,
                        database_=self.neo4j_database
                    )
                    
            self.logger.info(f"Created entity in Neo4j: {name}")
        except Exception as e:
            self.logger.error(f"Error creating entity in Neo4j: {str(e)}")
            # Don't raise - we can still use mem0 even if direct Neo4j operations fail

    def create_relations(self, relations: List[Union[Dict, Any]]) -> str:
        """
        Create multiple relations between entities in the knowledge graph.
        
        Args:
            relations: List of relations to create
            
        Returns:
            JSON string with the created relations
        """
        try:
            self._ensure_initialized()
            
            created_relations = []
            
            for relation in relations:
                # Process relation for mem0 format
                relation_dict = self._convert_to_dict(relation)
                
                # Map from_ to from for Pydantic models
                from_entity = relation_dict.get("from")
                if from_entity is None and "from_" in relation_dict:
                    from_entity = relation_dict["from_"]
                
                to_entity = relation_dict.get("to", "")
                relation_type = relation_dict.get("relationType", "")
                
                if from_entity and to_entity and relation_type:
                    # Create a memory string for the relation
                    memory_str = f"{from_entity} {relation_type} {to_entity}."
                    
                    # Add relation as a memory with metadata
                    if self.memory:
                        self.memory.add(
                            memory=memory_str, 
                            user_id=self.default_user_id,
                            metadata={
                                "type": "relation",
                                "from_entity": from_entity,
                                "to_entity": to_entity,
                                "relation_type": relation_type
                            }
                        )
                    
                    # Also create relation directly in Neo4j for operations not supported by mem0
                    if self.neo4j_driver:
                        self._create_relation_in_neo4j(from_entity, to_entity, relation_type)
                        
                    created_relations.append(relation_dict)
                    self.logger.info(f"Created relation: {from_entity} {relation_type} {to_entity}")
            
            return dict_to_json({"created": created_relations})
        except Exception as e:
            error_msg = f"Error creating relations: {str(e)}"
            self.logger.error(error_msg)
            return dict_to_json({"error": error_msg})

    def _create_relation_in_neo4j(self, from_entity: str, to_entity: str, relation_type: str) -> None:
        """Create a relation directly in Neo4j."""
        if not self.neo4j_driver:
            return
        
        try:
            query = """
            MERGE (from:Entity {name: $from_name})
            MERGE (to:Entity {name: $to_name})
            MERGE (from)-[r:RELATIONSHIP {type: $relation_type}]->(to)
            RETURN from, to, r
            """
            self.neo4j_driver.execute_query(
                query,
                from_name=from_entity,
                to_name=to_entity,
                relation_type=relation_type,
                database_=self.neo4j_database
            )
            self.logger.info(f"Created relation in Neo4j: {from_entity} {relation_type} {to_entity}")
        except Exception as e:
            self.logger.error(f"Error creating relation in Neo4j: {str(e)}")
            # Don't raise - we can still use mem0 even if direct Neo4j operations fail

    def add_observations(self, observations: List[Union[Dict, Any]]) -> str:
        """
        Add observations to existing entities.
        
        Args:
            observations: List of observations to add
            
        Returns:
            JSON string with the result
        """
        try:
            self._ensure_initialized()
            
            added_observations = []
            
            for observation in observations:
                # Process observation for mem0 format
                observation_dict = self._convert_to_dict(observation)
                
                entity_name = observation_dict.get("entityName", "")
                contents = observation_dict.get("contents", [])
                
                if entity_name and contents:
                    for content in contents:
                        # Create memory string for observation
                        memory_str = f"{entity_name} {content}"
                        
                        # Add observation as memory with metadata
                        if self.memory:
                            self.memory.add(
                                memory=memory_str, 
                                user_id=self.default_user_id,
                                metadata={
                                    "type": "observation",
                                    "entity_name": entity_name
                                }
                            )
                        
                        # Also add observation directly in Neo4j
                        if self.neo4j_driver:
                            self._add_observation_in_neo4j(entity_name, content)
                            
                    added_observations.append(observation_dict)
                    self.logger.info(f"Added observations to {entity_name}")
            
            return dict_to_json({"added": added_observations})
        except Exception as e:
            error_msg = f"Error adding observations: {str(e)}"
            self.logger.error(error_msg)
            return dict_to_json({"error": error_msg})

    def _add_observation_in_neo4j(self, entity_name: str, content: str) -> None:
        """Add an observation directly in Neo4j."""
        if not self.neo4j_driver:
            return
        
        try:
            query = """
            MATCH (e:Entity {name: $entity_name})
            MERGE (o:Observation {content: $content})
            MERGE (e)-[:HAS_OBSERVATION]->(o)
            """
            self.neo4j_driver.execute_query(
                query,
                entity_name=entity_name,
                content=content,
                database_=self.neo4j_database
            )
            self.logger.info(f"Added observation in Neo4j for {entity_name}")
        except Exception as e:
            self.logger.error(f"Error adding observation in Neo4j: {str(e)}")
            # Don't raise - we can still use mem0 even if direct Neo4j operations fail

    def delete_entities(self, entity_names: List[str]) -> None:
        """
        Delete entities from the knowledge graph.
        
        Args:
            entity_names: List of entity names to delete
        """
        try:
            self._ensure_initialized()
            
            # Using direct Neo4j connection for selective deletion
            if self.neo4j_driver:
                for entity_name in entity_names:
                    self._delete_entity_in_neo4j(entity_name)
                
                self.logger.info(f"Deleted entities from Neo4j: {entity_names}")
            else:
                # Fallback warning for when Neo4j driver is not available
                self.logger.warn("Selective entity deletion requires Neo4j driver")
                self.logger.info(f"Attempted to delete entities: {entity_names}")
        except Exception as e:
            error_msg = f"Error deleting entities: {str(e)}"
            self.logger.error(error_msg)
            raise e

    def _delete_entity_in_neo4j(self, entity_name: str) -> None:
        """Delete an entity directly in Neo4j."""
        if not self.neo4j_driver:
            return
        
        try:
            # Delete the entity and its observations
            query = """
            MATCH (e:Entity {name: $entity_name})
            OPTIONAL MATCH (e)-[:HAS_OBSERVATION]->(o:Observation)
            DETACH DELETE e, o
            """
            self.neo4j_driver.execute_query(
                query,
                entity_name=entity_name,
                database_=self.neo4j_database
            )
            self.logger.info(f"Deleted entity from Neo4j: {entity_name}")
        except Exception as e:
            self.logger.error(f"Error deleting entity in Neo4j: {str(e)}")
            raise e

    def delete_observations(self, deletions: List[Union[Dict, Any]]) -> None:
        """
        Delete observations from entities.
        
        Args:
            deletions: List of observations to delete
        """
        try:
            self._ensure_initialized()
            
            # Using direct Neo4j connection for selective deletion
            if self.neo4j_driver:
                for deletion in deletions:
                    deletion_dict = self._convert_to_dict(deletion)
                    entity_name = deletion_dict.get("entityName", "")
                    contents = deletion_dict.get("contents", [])
                    
                    if entity_name and contents:
                        for content in contents:
                            self._delete_observation_in_neo4j(entity_name, content)
                
                self.logger.info("Deleted observations from Neo4j")
            else:
                # Fallback warning for when Neo4j driver is not available
                self.logger.warn("Selective observation deletion requires Neo4j driver")
                self.logger.info("Attempted to delete observations")
        except Exception as e:
            error_msg = f"Error deleting observations: {str(e)}"
            self.logger.error(error_msg)
            raise e

    def _delete_observation_in_neo4j(self, entity_name: str, content: str) -> None:
        """Delete an observation directly in Neo4j."""
        if not self.neo4j_driver:
            return
        
        try:
            query = """
            MATCH (e:Entity {name: $entity_name})-[r:HAS_OBSERVATION]->(o:Observation {content: $content})
            DELETE r, o
            """
            self.neo4j_driver.execute_query(
                query,
                entity_name=entity_name,
                content=content,
                database_=self.neo4j_database
            )
            self.logger.info(f"Deleted observation from Neo4j for {entity_name}")
        except Exception as e:
            self.logger.error(f"Error deleting observation in Neo4j: {str(e)}")
            raise e

    def delete_relations(self, relations: List[Union[Dict, Any]]) -> None:
        """
        Delete relations from the knowledge graph.
        
        Args:
            relations: List of relations to delete
        """
        try:
            self._ensure_initialized()
            
            # Using direct Neo4j connection for selective deletion
            if self.neo4j_driver:
                for relation in relations:
                    relation_dict = self._convert_to_dict(relation)
                    
                    # Map from_ to from for Pydantic models
                    from_entity = relation_dict.get("from")
                    if from_entity is None and "from_" in relation_dict:
                        from_entity = relation_dict["from_"]
                    
                    to_entity = relation_dict.get("to", "")
                    relation_type = relation_dict.get("relationType", "")
                    
                    if from_entity and to_entity and relation_type:
                        self._delete_relation_in_neo4j(from_entity, to_entity, relation_type)
                
                self.logger.info("Deleted relations from Neo4j")
            else:
                # Fallback warning for when Neo4j driver is not available
                self.logger.warn("Selective relation deletion requires Neo4j driver")
                self.logger.info("Attempted to delete relations")
        except Exception as e:
            error_msg = f"Error deleting relations: {str(e)}"
            self.logger.error(error_msg)
            raise e

    def _delete_relation_in_neo4j(self, from_entity: str, to_entity: str, relation_type: str) -> None:
        """Delete a relation directly in Neo4j."""
        if not self.neo4j_driver:
            return
        
        try:
            query = """
            MATCH (from:Entity {name: $from_name})-[r:RELATIONSHIP {type: $relation_type}]->(to:Entity {name: $to_name})
            DELETE r
            """
            self.neo4j_driver.execute_query(
                query,
                from_name=from_entity,
                to_name=to_entity,
                relation_type=relation_type,
                database_=self.neo4j_database
            )
            self.logger.info(f"Deleted relation from Neo4j: {from_entity} {relation_type} {to_entity}")
        except Exception as e:
            self.logger.error(f"Error deleting relation in Neo4j: {str(e)}")
            raise e

    def search_nodes(self, query: str) -> str:
        """
        Search for nodes in the knowledge graph.
        
        Args:
            query: Search query
            
        Returns:
            JSON string with search results
        """
        try:
            self._ensure_initialized()
            
            # Search using mem0's search method
            if self.memory:
                results = self.memory.search(
                    query=query, 
                    limit=10,
                    user_id=self.default_user_id
                )
            else:
                results = {"results": []}
            
            # Format results
            formatted_results = []
            for result in results.get("results", []):
                # Extract entity information from memory content
                memory = result.get("memory", "")
                metadata = result.get("metadata", {})
                
                # Extract entity information from metadata if available
                entity_name = metadata.get("entity_name", "")
                entity_type = metadata.get("entity_type", "")
                
                # If metadata doesn't have entity info, try to parse from memory content
                if not entity_name:
                    parts = memory.split(" ")
                    entity_name = parts[0] if parts else ""
                
                # Combine memory information into an entity
                entity = {
                    "name": entity_name,
                    "entityType": entity_type or "Unknown",
                    "observations": [memory]
                }
                
                formatted_result = {
                    "entity": entity,
                    "score": result.get("score", 0.0)
                }
                formatted_results.append(formatted_result)
                
            search_response = {
                "results": formatted_results,
                "query": query
            }
            
            return dict_to_json(search_response)
        except Exception as e:
            error_msg = f"Error searching nodes: {str(e)}"
            self.logger.error(error_msg)
            return dict_to_json({"error": error_msg, "query": query})

    def open_nodes(self, names: List[str]) -> str:
        """
        Retrieve specific nodes by name.
        
        Args:
            names: List of entity names to retrieve
            
        Returns:
            JSON string with the retrieved entities
        """
        try:
            self._ensure_initialized()
            
            # Retrieve entities by name
            results = []
            
            for name in names:
                entity = None
                
                # Try to get entity directly from Neo4j first if available
                if self.neo4j_driver:
                    entity = self._get_entity_from_neo4j(name)
                
                # Fall back to mem0 search if Neo4j didn't return a result
                if not entity and self.memory:
                    # Search for the specific entity name
                    results_data = self.memory.search(
                        query=f"Find information about {name}", 
                        user_id=self.default_user_id,
                        limit=10
                    )
                    
                    # Extract entity information
                    memories = []
                    entity_type = "Unknown"
                    
                    for result in results_data.get("results", []):
                        memory = result.get("memory", "")
                        metadata = result.get("metadata", {})
                        
                        if metadata.get("entity_type"):
                            entity_type = metadata.get("entity_type")
                            
                        if memory:
                            memories.append(memory)
                    
                    if memories:
                        entity = {
                            "name": name,
                            "entityType": entity_type,
                            "observations": memories
                        }
                
                if entity:
                    results.append(entity)
            
            return dict_to_json({"entities": results})
        except Exception as e:
            error_msg = f"Error opening nodes: {str(e)}"
            self.logger.error(error_msg)
            return dict_to_json({"error": error_msg})

    def _get_entity_from_neo4j(self, entity_name: str) -> Optional[Dict[str, Any]]:
        """Get an entity directly from Neo4j."""
        if not self.neo4j_driver:
            return None
        
        try:
            query = """
            MATCH (e:Entity {name: $entity_name})
            OPTIONAL MATCH (e)-[:HAS_OBSERVATION]->(o:Observation)
            RETURN e.name as name, e.entityType as entityType, COLLECT(o.content) as observations
            """
            
            # Execute the query and handle the result format properly
            result_data = self.neo4j_driver.execute_query(
                query,
                entity_name=entity_name,
                database_=self.neo4j_database
            )
            
            # Extract records from the result
            records = result_data[0] if result_data and len(result_data) >= 1 else []
            
            if records and len(records) > 0:
                record = records[0]
                return {
                    "name": record["name"],
                    "entityType": record["entityType"],
                    "observations": record["observations"] if record["observations"] else []
                }
            
            return None
        except Exception as e:
            self.logger.error(f"Error getting entity from Neo4j: {str(e)}")
            return None

    def _convert_to_dict(self, obj: Any) -> Dict:
        """
        Convert an object to a dictionary, handling Pydantic models.
        
        Args:
            obj: Object to convert
            
        Returns:
            Dictionary representation of the object
        """
        if hasattr(obj, "model_dump"):
            # For Pydantic models
            try:
                return obj.model_dump(by_alias=True)
            except Exception:
                # Fallback method
                return {k: getattr(obj, k) for k in dir(obj) 
                       if not k.startswith('_') and not callable(getattr(obj, k))}
        elif isinstance(obj, dict):
            # For dictionaries
            return dict(obj)
        else:
            # For other objects, try to convert to dict
            try:
                return dict(obj)
            except (ValueError, TypeError):
                return {k: getattr(obj, k) for k in dir(obj)
                       if not k.startswith('_') and not callable(getattr(obj, k))}

    def _ensure_initialized(self) -> None:
        """Ensure the memory manager is initialized."""
        if not self.initialized:
            self.initialize()
            
    def __del__(self):
        """Destructor to ensure Neo4j driver is closed."""
        try:
            self.close()
        except:
            # Ignore errors during cleanup
            pass 