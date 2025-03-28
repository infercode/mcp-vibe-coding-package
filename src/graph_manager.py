import os
import json
import time
import traceback
import datetime
from typing import List, Dict, Any, Optional, Union, cast, TYPE_CHECKING
from typing_extensions import LiteralString
from neo4j import GraphDatabase

from src.types import Entity, KnowledgeGraph, Observation, Relation
from src.utils import dict_to_json, extract_error, generate_id
from src.embedding_manager import LiteLLMEmbeddingManager
from src.logger import Logger, get_logger

"""
NOTE: Due to type checking requirements in the Neo4j Python driver, query strings passed to
the execute_query method must be of type LiteralString (not just regular str).
To handle this properly:
1. Use the _safe_execute_query helper method to execute all Neo4j queries
2. The helper method:
   - Casts query strings to LiteralString to satisfy type checking
   - Handles float parameters by converting them to strings
   - Provides consistent error handling

This approach avoids linter errors related to string types and parameter type mismatches.
"""


class GraphMemoryManager:
    """Memory manager for the knowledge graph using Neo4j."""

    def __init__(self, logger: Optional[Logger] = None):
        """
        Initialize the memory manager.
        
        Args:
            logger: Optional logger instance
        """
        self.logger = logger or get_logger()
        self.initialized = False
        self.neo4j_driver = None
        
        # Default project name for memory operations if none is provided
        # This will be overridden when the AI agent provides a project name
        self.default_project_name = "default-project"
        
        self.logger.info(f"Using initial default project name: {self.default_project_name}")
        
        # Get config from environment variables
        self.neo4j_uri = os.environ.get("NEO4J_URI", "bolt://localhost:7687")
        self.neo4j_user = os.environ.get("NEO4J_USER", "neo4j")
        self.neo4j_password = os.environ.get("NEO4J_PASSWORD", "password")
        self.neo4j_database = os.environ.get("NEO4J_DATABASE", "neo4j")
        
        # Initialize embedding manager
        self.embedding_manager = LiteLLMEmbeddingManager(self.logger)
        
        # Check if embeddings should be enabled
        self.embedder_provider = os.environ.get("EMBEDDER_PROVIDER", "none").lower()
        
        # Flag to track if embeddings are enabled
        self.embedding_enabled = self.embedder_provider != "none"
        
        if not self.embedding_enabled:
            self.logger.info("Embeddings disabled - operating in basic mode without vector search")
            return
            
        # Configure embedding manager based on provider
        embedding_config = {
            "provider": self.embedder_provider,
            "model": "",
            "api_key": "",
            "api_base": "",
            "dimensions": 0,
            "additional_params": {}
        }
        
        # Initialize embedding model configuration based on provider
        if self.embedder_provider == "openai":
            embedding_config["model"] = os.environ.get("EMBEDDING_MODEL", "text-embedding-3-small")
            embedding_config["api_key"] = os.environ.get("OPENAI_API_KEY", "")
            embedding_config["api_base"] = os.environ.get("OPENAI_API_BASE", "")
            embedding_config["dimensions"] = int(os.environ.get("EMBEDDING_DIMS", "1536"))
            
            # Check if API key is provided
            if not embedding_config["api_key"]:
                self.logger.warn("OpenAI API key not provided. Embeddings will not work until configured.")
                self.embedding_enabled = False
        elif self.embedder_provider == "huggingface":
            embedding_config["model"] = os.environ.get("HUGGINGFACE_MODEL", "sentence-transformers/all-mpnet-base-v2")
            embedding_config["api_key"] = os.environ.get("HUGGINGFACE_API_KEY", "")
            embedding_config["dimensions"] = int(os.environ.get("EMBEDDING_DIMS", "768"))
            embedding_config["additional_params"] = json.loads(os.environ.get("HUGGINGFACE_MODEL_KWARGS", '{"device":"cpu"}'))
        elif self.embedder_provider == "ollama":
            embedding_config["model"] = os.environ.get("OLLAMA_MODEL", "llama2")
            embedding_config["api_base"] = os.environ.get("OLLAMA_BASE_URL", "http://localhost:11434")
            embedding_config["dimensions"] = int(os.environ.get("EMBEDDING_DIMS", "4096"))
        elif self.embedder_provider in ["azure", "azure_openai"]:
            # Support both "azure" and "azure_openai" for backward compatibility
            embedding_config["model"] = os.environ.get("AZURE_MODEL", "text-embedding-3-small")
            embedding_config["api_key"] = os.environ.get("EMBEDDING_AZURE_OPENAI_API_KEY", os.environ.get("AZURE_API_KEY", ""))
            embedding_config["api_base"] = os.environ.get("EMBEDDING_AZURE_ENDPOINT", os.environ.get("AZURE_ENDPOINT", ""))
            embedding_config["dimensions"] = int(os.environ.get("EMBEDDING_DIMS", "1536"))
            
            additional_params = {}
            additional_params["api_version"] = os.environ.get("EMBEDDING_AZURE_API_VERSION", "2023-05-15")
            additional_params["deployment"] = os.environ.get("EMBEDDING_AZURE_DEPLOYMENT", os.environ.get("AZURE_DEPLOYMENT", ""))
            
            embedding_config["additional_params"] = additional_params
            
            # Add custom headers if provided
            azure_headers_env = os.environ.get("EMBEDDING_AZURE_DEFAULT_HEADERS", "{}")
            try:
                azure_headers = json.loads(azure_headers_env)
                if azure_headers:
                    embedding_config["additional_params"]["default_headers"] = azure_headers
            except json.JSONDecodeError:
                self.logger.warn(f"Failed to parse EMBEDDING_AZURE_DEFAULT_HEADERS as JSON: {azure_headers_env}")
            
            # Check if required Azure config is provided
            if not (embedding_config["api_key"] and embedding_config["additional_params"]["deployment"] and embedding_config["api_base"]):
                self.logger.warn("Required Azure OpenAI configuration missing. Embeddings will not work until configured.")
                self.embedding_enabled = False
        elif self.embedder_provider == "lmstudio":
            embedding_config["model"] = "lmstudio/embedding"
            embedding_config["api_base"] = os.environ.get("LMSTUDIO_BASE_URL", "http://localhost:1234")
            embedding_config["dimensions"] = int(os.environ.get("EMBEDDING_DIMS", "4096"))
        elif self.embedder_provider == "vertexai":
            embedding_config["model"] = os.environ.get("VERTEXAI_MODEL", "text-embedding-004")
            embedding_config["dimensions"] = int(os.environ.get("EMBEDDING_DIMS", "256"))
            
            additional_params = {}
            additional_params["vertex_credentials_json"] = os.environ.get("GOOGLE_APPLICATION_CREDENTIALS", "")
            additional_params["project"] = os.environ.get("VERTEX_PROJECT", "")
            additional_params["location"] = os.environ.get("VERTEX_LOCATION", "us-central1")
            additional_params["embedding_type"] = os.environ.get("VERTEXAI_MEMORY_ADD_EMBEDDING_TYPE", "RETRIEVAL_DOCUMENT")
            
            embedding_config["additional_params"] = additional_params
            
            # Check if credentials file is provided
            if not embedding_config["additional_params"]["vertex_credentials_json"]:
                self.logger.warn("Google credentials not provided. Embeddings will not work until configured.")
                self.embedding_enabled = False
        elif self.embedder_provider == "gemini":
            embedding_config["model"] = os.environ.get("GEMINI_MODEL", "models/text-embedding-004")
            embedding_config["api_key"] = os.environ.get("GOOGLE_API_KEY", "")
            embedding_config["dimensions"] = int(os.environ.get("EMBEDDING_DIMS", "768"))
            
            # Check if API key is provided
            if not embedding_config["api_key"]:
                self.logger.warn("Google API key not provided. Embeddings will not work until configured.")
                self.embedding_enabled = False
        elif self.embedder_provider == "mistral":
            embedding_config["model"] = os.environ.get("MISTRAL_MODEL", "mistral-embed")
            embedding_config["api_key"] = os.environ.get("MISTRAL_API_KEY", "")
            embedding_config["dimensions"] = int(os.environ.get("EMBEDDING_DIMS", "1024"))
            
            # Check if API key is provided
            if not embedding_config["api_key"]:
                self.logger.warn("Mistral API key not provided. Embeddings will not work until configured.")
                self.embedding_enabled = False
        else:
            # Unknown provider
            self.logger.warn(f"Unknown embedder provider: {self.embedder_provider}. Embeddings will be disabled.")
            self.embedding_enabled = False
        
        # Configure the embedding manager
        if self.embedding_enabled:
            result = self.embedding_manager.configure(embedding_config)
            if result["status"] != "success":
                self.logger.warn(f"Failed to configure embedding manager: {result['message']}")
                self.embedding_enabled = False
            else:
                self.logger.info(f"Embedding manager configured successfully: {result['message']}")

    def set_project_name(self, project_name: str) -> None:
        """
        Set the default project name for memory operations.
        
        Args:
            project_name: The project name to use for memory operations
        """
        if project_name and project_name.strip():
            self.default_project_name = project_name.strip()
            self.logger.info(f"Updated default project name to: {self.default_project_name}")
        else:
            self.logger.warn("Cannot set empty project name")

    def initialize(self) -> None:
        """Initialize the memory manager and connect to the graph database."""
        if self.initialized:
            return

        try:
            # Initialize Neo4j driver with connection pooling parameters
            self.neo4j_driver = GraphDatabase.driver(
                self.neo4j_uri,
                auth=(self.neo4j_user, self.neo4j_password),
                # Connection pooling parameters
                max_connection_lifetime=30 * 60,  # 30 minutes
                max_connection_pool_size=50,      # Max 50 connections in the pool
                connection_acquisition_timeout=60, # 60 seconds timeout for acquiring a connection
                keep_alive=True                   # Keep connections alive
            )
            
            # Test the Neo4j connection with retry
            self._test_neo4j_connection_with_retry(max_retries=3)
            
            # Setup vector index if embeddings are enabled
            if self.embedding_enabled:
                self._setup_vector_index()
                self.logger.info(f"Memory graph manager initialized successfully with {self.embedder_provider} embedder")
            else:
                self.logger.info("Memory graph manager initialized successfully without embeddings")
                
            self.initialized = True
            
        except Exception as e:
            self.logger.error(f"Failed to initialize memory manager: {str(e)}")
            raise e
            
    def _test_neo4j_connection_with_retry(self, max_retries=3, initial_delay=1.0):
        """Test Neo4j connection with exponential backoff retry mechanism."""
        if not self.neo4j_driver:
            self.logger.debug("Neo4j driver not initialized, skipping connection test")
            return
            
        import time
        
        delay = initial_delay
        last_exception = None
        
        for retry in range(max_retries):
            try:
                self.logger.debug(f"Connection attempt {retry + 1}/{max_retries}")
                self._test_neo4j_connection()
                self.logger.debug(f"Connection successful on attempt {retry + 1}")
                return  # Success, exit the retry loop
            except Exception as e:
                last_exception = e
                self.logger.warn(f"Neo4j connection attempt {retry + 1}/{max_retries} failed: {str(e)}")
                
                if retry < max_retries - 1:  # Don't sleep after the last attempt
                    self.logger.info(f"Retrying in {delay:.1f} seconds...")
                    time.sleep(delay)
                    delay *= 2  # Exponential backoff
        
        # If we're here, all retries failed
        self.logger.error(f"Failed to connect to Neo4j after {max_retries} attempts")
        if last_exception:
            raise last_exception

    def _setup_vector_index(self) -> None:
        """Setup vector index in Neo4j if it doesn't exist."""
        if not self.neo4j_driver or not self.embedding_enabled:
            return
            
        try:
            # Check if vector index exists
            index_query = """
            SHOW INDEXES
            YIELD name, type
            WHERE type = 'VECTOR'
            RETURN count(*) > 0 AS exists
            """
            
            result = self.neo4j_driver.execute_query(
                index_query,
                database_=self.neo4j_database
            )
            
            index_exists = result[0][0]["exists"] if result and result[0] else False
            
            if not index_exists:
                # Create vector index for entity nodes with embeddings
                create_index_query = """
                CREATE VECTOR INDEX entity_embedding IF NOT EXISTS
                FOR (e:Entity)
                ON e.embedding
                OPTIONS {indexConfig: {
                    `vector.dimensions`: $dimensions,
                    `vector.similarity_function`: 'cosine'
                }}
                """
                
                # Get dimensions from embedding manager config
                embedding_dimensions = self.embedding_manager.dimensions
                if not embedding_dimensions:
                    embedding_dimensions = 1536  # Use a default value if not set
                
                self.neo4j_driver.execute_query(
                    create_index_query,
                    dimensions=embedding_dimensions,
                    database_=self.neo4j_database
                )
                
                self.logger.info(f"Created vector index with {embedding_dimensions} dimensions")
            else:
                self.logger.info("Vector index already exists")
        except Exception as e:
            self.logger.error(f"Error setting up vector index: {str(e)}")
            # Don't fail initialization if index creation fails

    def _test_neo4j_connection(self) -> None:
        """Test the Neo4j connection by running a simple query."""
        if not self.neo4j_driver:
            self.logger.debug("Neo4j driver not initialized, skipping connection test")
            return
        
        try:
            self.logger.debug("Testing Neo4j connection with simple query")
            result_data = self.neo4j_driver.execute_query(
                "RETURN 'Connection successful' AS message",
                database_=self.neo4j_database
            )
            records = result_data[0] if result_data and len(result_data) > 0 else []
            message = records[0]["message"] if records and len(records) > 0 else "Unknown"
            self.logger.info(f"Neo4j direct connection test: {message}")
            
            # Additional connection status check
            driver_info = self.neo4j_driver.get_server_info()
            self.logger.info(f"Connected to Neo4j Server: {driver_info.agent} (Version: {driver_info.protocol_version})")
            
            # Log connection pool metrics - commented out due to potential API incompatibility
            # Some Neo4j driver versions don't support get_metrics()
            self.logger.debug("Neo4j connection pool metrics not available or skipped")
            
        except Exception as e:
            error_message = str(e)
            if "Authentication failed" in error_message:
                self.logger.error(f"Neo4j authentication failed. Please check your credentials: {error_message}")
            elif "Connection refused" in error_message:
                self.logger.error(f"Neo4j connection refused. Is the database running? {error_message}")
            elif "Failed to establish connection" in error_message:
                self.logger.error(f"Failed to establish Neo4j connection. Check network and server: {error_message}")
            else:
                self.logger.error(f"Neo4j connection test failed: {error_message}")
            
            # Don't raise - we can still use basic functionality without direct connection
            
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
                # Process entity
                entity_dict = self._convert_to_dict(entity)
                
                # Extract entity information
                entity_name = entity_dict.get("name", "")
                entity_type = entity_dict.get("entityType", "")
                observations = entity_dict.get("observations", [])
                
                if entity_name and entity_type:
                    # Create the entity in Neo4j
                    self._create_entity_in_neo4j(entity_name, entity_type, observations)
                    
                    # Add entity with embedding if embeddings are enabled
                    if self.embedding_enabled:
                        # Generate description for embedding
                        description = f"{entity_name} is a {entity_type}."
                        if observations:
                            description += " " + " ".join(observations)
                            
                        # Generate embedding
                        embedding = self._generate_embedding(description)
                        
                        # Store embedding with entity
                        if embedding:
                            self._update_entity_embedding(entity_name, embedding)
                    
                    created_entities.append(entity_dict)
                    self.logger.info(f"Created entity: {entity_name}")
            
            return dict_to_json({"created": created_entities})
                
        except Exception as e:
            error_msg = f"Error creating entities: {str(e)}"
            self.logger.error(error_msg)
            return dict_to_json({"error": error_msg})
            
    def _update_entity_embedding(self, entity_name: str, embedding: List[float]) -> None:
        """
        Update the embedding for an entity.
        
        Args:
            entity_name: The name of the entity
            embedding: The embedding vector
        """
        if not self.neo4j_driver:
            return
            
        try:
            # Update entity with embedding
            query = """
            MATCH (e:Entity {name: $name})
            SET e.embedding = $embedding
            """
            
            self.neo4j_driver.execute_query(
                query,
                name=entity_name,
                embedding=embedding,
                database_=self.neo4j_database
            )
            
            self.logger.info(f"Updated embedding for entity: {entity_name}")
        except Exception as e:
            self.logger.error(f"Error updating entity embedding: {str(e)}")
            # Don't raise - we can still use basic functionality without embeddings

    def _create_entity_in_neo4j(self, name: str, entity_type: str, observations: List[str]) -> None:
        """Create an entity directly in Neo4j."""
        if not self.neo4j_driver:
            self.logger.debug("Neo4j driver not initialized, skipping entity creation")
            return
        
        try:
            # Create the entity node
            self.logger.debug(f"Creating entity node in Neo4j: {name}", context={"entity_type": entity_type})
            query = """
            MERGE (e:Entity {name: $name})
            SET e.entityType = $entity_type
            RETURN e
            """
            result = self.neo4j_driver.execute_query(
                query,
                name=name,
                entity_type=entity_type,
                database_=self.neo4j_database
            )
            
            result_summary = result[1] if len(result) > 1 else None
            self.logger.debug("Entity node creation result", context={
                "counters": str(result_summary.counters) if result_summary else "No summary",
                "entity": name
            })
            
            # Add observations if present
            if observations:
                self.logger.debug(f"Adding {len(observations)} observations to entity: {name}")
                for idx, observation in enumerate(observations):
                    self.logger.debug(f"Adding observation {idx+1}/{len(observations)}", 
                                     context={"entity": name, "content_length": len(observation)})
                    obs_query = """
                    MATCH (e:Entity {name: $name})
                    MERGE (o:Observation {content: $content})
                    MERGE (e)-[:HAS_OBSERVATION]->(o)
                    """
                    obs_result = self.neo4j_driver.execute_query(
                        obs_query,
                        name=name,
                        content=observation,
                        database_=self.neo4j_database
                    )
                    
                    obs_summary = obs_result[1] if len(obs_result) > 1 else None
                    self.logger.debug(f"Observation {idx+1} creation result", context={
                        "counters": str(obs_summary.counters) if obs_summary else "No summary"
                    })
                    
            self.logger.info(f"Created entity in Neo4j: {name}")
        except Exception as e:
            self.logger.error(f"Error creating entity in Neo4j: {str(e)}", context={"entity": name, "entity_type": entity_type}, exc_info=True)
            # Don't raise - we can still use mem0 even if direct Neo4j operations fail
            
    def create_relations(self, relations: List[Union[Dict, Any]]) -> str:
        """
        Create multiple relations in the knowledge graph.
        
        Args:
            relations: List of relations to create
            
        Returns:
            JSON string with the created relations
        """
        try:
            self._ensure_initialized()
            
            created_relations = []
            
            for relation in relations:
                # Process relation
                relation_dict = self._convert_to_dict(relation)
                
                # Map from_ to from for Pydantic models
                from_entity = relation_dict.get("from")
                if from_entity is None and "from_" in relation_dict:
                    from_entity = relation_dict["from_"]
                
                to_entity = relation_dict.get("to", "")
                relation_type = relation_dict.get("relationType", "")
                
                if from_entity and to_entity and relation_type:
                    # Create relation directly in Neo4j
                    self._create_relation_in_neo4j(from_entity, to_entity, relation_type)
                    
                    # Add relation to created_relations
                    created_relation = {
                        "from": from_entity,
                        "to": to_entity,
                        "relationType": relation_type
                    }
                    created_relations.append(created_relation)
                    self.logger.info(f"Created relation: {from_entity} -{relation_type}-> {to_entity}")
            
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
                # Process observation
                observation_dict = self._convert_to_dict(observation)
                
                entity_name = observation_dict.get("entityName", "")
                contents = observation_dict.get("contents", [])
                
                if entity_name and contents:
                    for content in contents:
                        # Add observation directly to Neo4j
                        self._add_observation_in_neo4j(entity_name, content)
                    
                    # Update embedding if enabled
                    if self.embedding_enabled and self.neo4j_driver:
                        # Get all observations for the entity
                        query = """
                        MATCH (e:Entity {name: $name})<-[:DESCRIBES]-(o:Observation)
                        RETURN e.entityType as entityType, collect(o.content) as observations
                        """
                        
                        result = self.neo4j_driver.execute_query(
                            query,
                            name=entity_name,
                            database_=self.neo4j_database
                        )
                        
                        if result and result[0] and result[0][0]:
                            entity_type = result[0][0]["entityType"] or "Unknown"
                            all_observations = result[0][0]["observations"]
                            
                            # Generate description for embedding
                            description = f"{entity_name} is a {entity_type}."
                            if all_observations:
                                description += " " + " ".join(all_observations)
                                
                            # Generate embedding
                            embedding = self._generate_embedding(description)
                            
                            # Store embedding with entity
                            if embedding:
                                self._update_entity_embedding(entity_name, embedding)
                        
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
            MATCH (e:Entity {name: $entity_name})-[r:HAS_OBSERVATION]->(o:Observation {content: $obs_content})
            DELETE r, o
            """
            self.neo4j_driver.execute_query(
                query,
                entity_name=entity_name,
                obs_content=content,
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

    def search_nodes(self, query: str, limit: int = 10, project_name: Optional[str] = None) -> str:
        """
        Search for nodes in the knowledge graph.
        
        Args:
            query: Search query
            limit: Maximum number of results to return
            project_name: Optional project identifier (defaults to current project)
            
        Returns:
            JSON string with search results
        """
        try:
            self._ensure_initialized()
            
            # Use default project name if not provided
            project_name = project_name or self.default_project_name
            
            # Check if Neo4j driver is available
            if not self.neo4j_driver:
                self.logger.warn("Neo4j driver not available for search")
                return dict_to_json({
                    "results": [],
                    "query": query,
                    "project": project_name,
                    "message": "Neo4j driver not available"
                })
            
            # Split query into tokens for better search
            search_tokens = [token for token in query.strip().split() if token]
            # Use original query if no tokens or just one token
            use_token_search = len(search_tokens) > 1
                
            formatted_results = []
            
            # Check if embeddings are enabled for semantic search
            if self.embedding_enabled:
                # Generate embedding for query
                query_embedding = self._generate_embedding(query)
                
                if query_embedding:
                    # Use vector search to find similar entities
                    vector_query = """
                    MATCH (e:Entity)
                    WHERE e.embedding IS NOT NULL
                    WITH e, vector.similarity.cosine(e.embedding, $query_embedding) AS score
                    WHERE score > 0.7
                    RETURN e.name AS name, e.entityType AS entityType, score
                    ORDER BY score DESC
                    LIMIT $limit
                    """
                    
                    vector_results = self.neo4j_driver.execute_query(
                        vector_query,
                        query_embedding=query_embedding,
                        limit=limit,
                        database_=self.neo4j_database
                    )
                    
                    # Process vector search results
                    records = vector_results[0] if vector_results and len(vector_results) > 0 else []
                    for record in records:
                        entity_name = record["name"]
                        entity_type = record["entityType"]
                        score = record["score"]
                        
                        # Get observations for this entity
                        entity_data = self._get_entity_from_neo4j(entity_name)
                        
                        if entity_data:
                            formatted_result = {
                                "entity": entity_data,
                                "score": score
                            }
                            formatted_results.append(formatted_result)
            else:
                # Fallback to basic text search if embeddings are not enabled
                self.logger.info("Using basic text search (embeddings disabled)")
                
                # Use basic text matching in Cypher with correct relationship pattern
                if use_token_search:
                    # Use parameterized query for multi-token search
                    text_query = """
                    MATCH (e:Entity)
                    WHERE any(token IN $search_tokens WHERE 
                         toLower(e.name) CONTAINS toLower(token) OR 
                         toLower(e.entityType) CONTAINS toLower(token))
                    WITH e
                    OPTIONAL MATCH (e)-[:HAS_OBSERVATION]->(o:Observation)
                    WITH e, collect(o.content) as observations
                    RETURN e.name as name, e.entityType as entityType, observations
                    LIMIT $limit
                    """
                    
                    # Execute with search_tokens parameter
                    text_results = self.neo4j_driver.execute_query(
                        text_query,
                        search_tokens=search_tokens,
                        limit=limit,
                        database_=self.neo4j_database
                    )
                else:
                    # Use single-token search
                    text_query = """
                    MATCH (e:Entity)
                    WHERE toLower(e.name) CONTAINS toLower($search_text) 
                       OR toLower(e.entityType) CONTAINS toLower($search_text)
                    WITH e
                    OPTIONAL MATCH (e)-[:HAS_OBSERVATION]->(o:Observation)
                    WITH e, collect(o.content) as observations
                    RETURN e.name as name, e.entityType as entityType, observations
                    LIMIT $limit
                    """
                    
                    # Execute with search_text parameter
                    text_results = self.neo4j_driver.execute_query(
                        text_query,
                        search_text=query,
                        limit=limit,
                        database_=self.neo4j_database
                    )
                
                # Process text search results
                records = text_results[0] if text_results and len(text_results) > 0 else []
                
                # If no results from entity name/type, try searching through observations
                if not records:
                    self.logger.info("No results in entity names/types, searching in observations")
                    
                    if use_token_search:
                        # Use parameterized query for multi-token observation search
                        obs_query = """
                        MATCH (e:Entity)-[:HAS_OBSERVATION]->(o:Observation)
                        WHERE any(token IN $search_tokens WHERE toLower(o.content) CONTAINS toLower(token))
                        WITH e, collect(DISTINCT o.content) as observations
                        RETURN DISTINCT e.name as name, e.entityType as entityType, observations
                        LIMIT $limit
                        """
                        
                        # Execute with search_tokens parameter
                        obs_results = self.neo4j_driver.execute_query(
                            obs_query,
                            search_tokens=search_tokens,
                            limit=limit,
                            database_=self.neo4j_database
                        )
                    else:
                        # Single-token observation search
                        obs_query = """
                        MATCH (e:Entity)-[:HAS_OBSERVATION]->(o:Observation)
                        WHERE toLower(o.content) CONTAINS toLower($search_text)
                        WITH e, collect(o.content) as observations
                        RETURN e.name as name, e.entityType as entityType, observations
                        LIMIT $limit
                        """
                        
                        obs_results = self.neo4j_driver.execute_query(
                            obs_query,
                            search_text=query,
                            limit=limit,
                            database_=self.neo4j_database
                        )
                    
                    records = obs_results[0] if obs_results and len(obs_results) > 0 else []
                
                for record in records:
                    entity_name = record["name"]
                    entity_type = record["entityType"]
                    observations = record["observations"]
                    
                    # Calculate basic relevance score
                    score = 1.0  # Default score
                    
                    # Boost score based on matches in name and type
                    name_match = query.lower() in entity_name.lower()
                    type_match = query.lower() in (entity_type or "").lower()
                    
                    if name_match:
                        score += 0.5  # Boost for name match
                    if type_match:
                        score += 0.3  # Boost for type match
                        
                    # For multi-token searches, count matching tokens
                    if use_token_search:
                        token_matches = 0
                        for token in search_tokens:
                            if token.lower() in entity_name.lower():
                                token_matches += 1
                            if token.lower() in (entity_type or "").lower():
                                token_matches += 1
                        
                        # Observation token matches
                        obs_matches = 0
                        for obs in observations:
                            for token in search_tokens:
                                if token.lower() in obs.lower():
                                    obs_matches += 1
                        
                        # Calculate token match percentage and add to score
                        if len(search_tokens) > 0:
                            token_match_percentage = token_matches / (len(search_tokens) * 2)  # x2 for name and type
                            score += token_match_percentage * 0.3
                            
                            obs_match_percentage = min(1.0, obs_matches / len(search_tokens))
                            score += obs_match_percentage * 0.2
                    
                    entity = {
                        "name": entity_name,
                        "entityType": entity_type or "Unknown",
                        "observations": observations,
                        "project": project_name
                    }
                    
                    formatted_result = {
                        "entity": entity,
                        "score": score
                    }
                    formatted_results.append(formatted_result)
                
                # Sort results by score in descending order
                formatted_results.sort(key=lambda x: x["score"], reverse=True)
                
                # Limit to requested number of results
                formatted_results = formatted_results[:limit]
            
            # If no results from either method, provide a message
            if not formatted_results:
                self.logger.info(f"No results found for query: {query}")
                
            search_response = {
                "results": formatted_results,
                "query": query,
                "project": project_name
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
                # Get entity directly from Neo4j
                entity = self._get_entity_from_neo4j(name)
                
                if entity:
                    results.append(entity)
                else:
                    self.logger.info(f"Entity '{name}' not found")
            
            return dict_to_json({"entities": results})
                
        except Exception as e:
            error_msg = f"Error retrieving nodes: {str(e)}"
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

    def _convert_to_dict(self, obj: Any) -> Dict[str, Any]:
        """
        Convert an object to a dictionary.
        
        Args:
            obj: Object to convert
            
        Returns:
            Dictionary representation of the object
        """
        if isinstance(obj, dict):
            return obj
        elif hasattr(obj, "dict") and callable(getattr(obj, "dict")):
            return obj.dict()
        elif hasattr(obj, "__dict__"):
            return obj.__dict__
        else:
            return {"value": str(obj)}

    def _ensure_initialized(self) -> None:
        """Ensure the memory manager is initialized."""
        if not self.initialized:
            self.initialize()
            
    def get_all_memories(self, project_name: Optional[str] = None) -> str:
        """
        Get all entities and their observations from the knowledge graph.
        
        Args:
            project_name: Optional project identifier (currently not used with Neo4j implementation)
            
        Returns:
            JSON string with all memories
        """
        try:
            self._ensure_initialized()
            
            if not self.neo4j_driver:
                return dict_to_json({
                    "status": "error",
                    "message": "Neo4j driver not initialized"
                })
                
            # Retrieve all entities and observations from Neo4j
            query = """
            MATCH (e:Entity)
            OPTIONAL MATCH (e)<-[:DESCRIBES]-(o:Observation)
            WITH e, collect(o.content) as observations
            RETURN e.name as name, e.entityType as entityType, observations
            """
            
            results = self.neo4j_driver.execute_query(
                query, 
                database_=self.neo4j_database
            )
            
            memories = []
            
            records = results[0] if results and len(results) > 0 else []
            for record in records:
                entity_name = record["name"]
                entity_type = record["entityType"]
                observations = record["observations"]
                
                # Format as entity with observations
                entity = {
                    "name": entity_name,
                    "entityType": entity_type or "Unknown",
                    "observations": observations
                }
                
                memories.append(entity)
            
            return dict_to_json({
                "status": "success",
                "memories": memories,
                "count": len(memories)
            })
                
        except Exception as e:
            error_msg = f"Error getting all memories: {str(e)}"
            self.logger.error(error_msg)
            return dict_to_json({"error": error_msg})
            
    def delete_all_memories(self, project_name: Optional[str] = None) -> str:
        """
        Delete all entities and observations in the knowledge graph.
        
        Args:
            project_name: Optional project identifier (currently not used with Neo4j implementation)
            
        Returns:
            JSON string with result information
        """
        try:
            self._ensure_initialized()
            
            if not self.neo4j_driver:
                return dict_to_json({
                    "status": "error",
                    "message": "Neo4j driver not initialized"
                })
                
            # Delete all relationships and nodes using the Neo4j driver
            query = """
            MATCH (n)
            DETACH DELETE n
            """
            
            self.neo4j_driver.execute_query(
                query,
                database_=self.neo4j_database
            )
            
            self.logger.info("Deleted all nodes and relationships from Neo4j")
            
            return dict_to_json({
                "status": "success",
                "message": "All memories deleted"
            })
                
        except Exception as e:
            error_msg = f"Error deleting all memories: {str(e)}"
            self.logger.error(error_msg)
            return dict_to_json({"error": error_msg})
            
    def __del__(self):
        """Destructor to ensure Neo4j driver is closed."""
        try:
            self.close()
        except:
            # Ignore errors during cleanup
            pass 

    def configure_embeddings(self, config: Dict[str, Any]) -> Dict[str, str]:
        """
        Configure the embedding provider with the provided settings.
        
        Args:
            config: Dictionary containing the embedder configuration
            
        Returns:
            Dictionary with status and message
        """
        try:
            if not config or "provider" not in config:
                return {"status": "error", "message": "Invalid configuration: missing provider"}
            
            provider = config.get("provider", "").lower()
            provider_config = config.get("config", {})
            
            # Update the embedder provider
            self.embedder_provider = provider
            
            # Enable or disable embeddings based on provider
            self.embedding_enabled = provider != "none"
            
            if not self.embedding_enabled:
                self.logger.info("Embeddings disabled by configuration")
                return {"status": "success", "message": "Embeddings disabled"}
                
            # Prepare embedding configuration for the manager
            embedding_config = {
                "provider": provider,
                "model": provider_config.get("model", ""),
                "dimensions": provider_config.get("embedding_dims", 0),
                "api_key": provider_config.get("api_key", ""),
                "api_base": provider_config.get("api_base", ""),
                "additional_params": {}
            }
            
            # Provider-specific configuration adjustments
            if provider == "openai":
                # OpenAI uses model string without provider prefix
                if embedding_config["model"].startswith("openai/"):
                    embedding_config["model"] = embedding_config["model"].replace("openai/", "")
                
                # Default to text-embedding-3-small if not specified
                if not embedding_config["model"]:
                    embedding_config["model"] = "text-embedding-3-small"
                    
            elif provider == "huggingface":
                # Add any model_kwargs as additional_params
                if "model_kwargs" in provider_config:
                    embedding_config["additional_params"] = provider_config["model_kwargs"]
                
                # Ensure the model has huggingface/ prefix if needed
                if embedding_config["model"] and not embedding_config["model"].startswith("huggingface/"):
                    embedding_config["model"] = f"huggingface/{embedding_config['model']}"
                    
                # Set input_type parameter if provided
                if "input_type" in provider_config:
                    embedding_config["additional_params"]["input_type"] = provider_config["input_type"]
                    
            elif provider == "ollama":
                # Map ollama_base_url to api_base if needed
                if "ollama_base_url" in provider_config and not embedding_config["api_base"]:
                    embedding_config["api_base"] = provider_config["ollama_base_url"]
                
                # Ensure the model has ollama/ prefix if needed
                if embedding_config["model"] and not embedding_config["model"].startswith("ollama/"):
                    embedding_config["model"] = f"ollama/{embedding_config['model']}"
                    
            elif provider in ["azure", "azure_openai"]:
                # Handle azure-specific configuration
                azure_kwargs = provider_config.get("azure_kwargs", {})
                
                # Set deployment as required parameter
                deployment = None
                if "azure_deployment" in azure_kwargs:
                    deployment = azure_kwargs["azure_deployment"]
                elif "deployment" in azure_kwargs:
                    deployment = azure_kwargs["deployment"]
                elif "deployment_name" in provider_config:
                    deployment = provider_config["deployment_name"]
                
                # Azure requires a deployment name
                if deployment:
                    embedding_config["additional_params"]["deployment"] = deployment
                
                # Model format for Azure is different - use the deployment name
                if deployment and not embedding_config["model"]:
                    embedding_config["model"] = f"azure/{deployment}"
                
                # Azure API version
                api_version = azure_kwargs.get("api_version") or provider_config.get("api_version")
                if api_version:
                    embedding_config["additional_params"]["api_version"] = api_version
                
                # Azure endpoint/api_base
                if "azure_endpoint" in azure_kwargs and not embedding_config["api_base"]:
                    embedding_config["api_base"] = azure_kwargs["azure_endpoint"]
                elif "api_base" in azure_kwargs and not embedding_config["api_base"]:
                    embedding_config["api_base"] = azure_kwargs["api_base"]
                
                # Azure custom headers
                if "default_headers" in azure_kwargs:
                    embedding_config["additional_params"]["default_headers"] = azure_kwargs["default_headers"]
                    
            elif provider == "vertexai" or provider == "vertex_ai":
                # Vertex AI specific configuration
                # Set default location if not provided
                location = provider_config.get("location", "us-central1")
                embedding_config["additional_params"]["location"] = location
                
                # Set project if provided
                if "project" in provider_config:
                    embedding_config["additional_params"]["project"] = provider_config["project"]
                    
                # Handle credentials
                if "vertex_credentials_json" in provider_config:
                    embedding_config["additional_params"]["vertex_credentials_json"] = provider_config["vertex_credentials_json"]
                
                # Ensure model has correct format for VertexAI
                if embedding_config["model"]:
                    if not embedding_config["model"].startswith("vertex_ai/"):
                        embedding_config["model"] = f"vertex_ai/{embedding_config['model']}"
                else:
                    # Default model if not specified
                    embedding_config["model"] = "vertex_ai/textembedding-gecko"
                    
            elif provider == "lmstudio":
                # Map lmstudio_base_url to api_base
                if "lmstudio_base_url" in provider_config and not embedding_config["api_base"]:
                    embedding_config["api_base"] = provider_config["lmstudio_base_url"]
                
                # Ensure model is set with proper format
                if not embedding_config["model"]:
                    embedding_config["model"] = "lmstudio/embedding"
                elif not embedding_config["model"].startswith("lmstudio/"):
                    embedding_config["model"] = f"lmstudio/{embedding_config['model']}"
                    
            elif provider == "mistral":
                # Ensure model has correct format for Mistral
                if not embedding_config["model"]:
                    embedding_config["model"] = "mistral/mistral-embed"
                elif not embedding_config["model"].startswith("mistral/"):
                    embedding_config["model"] = f"mistral/{embedding_config['model']}"
            
            elif provider == "gemini":
                # Ensure model has correct format for Gemini
                if not embedding_config["model"]:
                    embedding_config["model"] = "gemini/text-embedding-004"
                elif not embedding_config["model"].startswith("gemini/"):
                    embedding_config["model"] = f"gemini/{embedding_config['model']}"
                                 
            # Configure the embedding manager with our prepared config
            result = self.embedding_manager.configure(embedding_config)
            if result["status"] != "success":
                self.logger.warn(f"Failed to configure embedding manager: {result['message']}")
                self.embedding_enabled = False
                return result
                
            self.logger.info(f"Embedding provider configured: {provider} with model {embedding_config['model']}")
            return {"status": "success", "message": f"Embedding provider configured: {provider}"}
        except Exception as e:
            error_msg = f"Error configuring embeddings: {str(e)}"
            self.logger.error(error_msg)
            return {"status": "error", "message": error_msg}
            
    def get_current_config(self) -> Dict[str, Any]:
        """
        Get the current embedding configuration.
        
        Returns:
            The current embedding configuration
        """
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
        
        embedding_config = {}
        
        if self.embedding_enabled:
            # Get configuration from the embedding manager
            manager_config = self.embedding_manager.get_config()
            
            embedding_config = {
                "embedder": {
                    "provider": manager_config.get("provider", "none"),
                    "config": {
                        "model": manager_config.get("model", ""),
                        "embedding_dims": manager_config.get("dimensions", 0)
                    }
                }
            }
            
            # Include any additional non-sensitive parameters
            if "additional_params" in manager_config:
                for key, value in manager_config.get("additional_params", {}).items():
                    if "key" not in key.lower() and "password" not in key.lower() and "credentials" not in key.lower():
                        embedding_config["embedder"]["config"][key] = value
        
        return {**graph_store_config, **embedding_config}

    def delete_entity(self, entity: str) -> str:
        """
        Delete an entity from the knowledge graph.
        
        Args:
            entity: Name of the entity to delete
            
        Returns:
            JSON string with result information
        """
        try:
            self._ensure_initialized()
            
            # Remove entity from the graph using direct Neo4j connection
            if self.neo4j_driver:
                # First delete all relationships involving this entity
                relations_query = """
                MATCH (e:Entity {name: $name})-[r]-()
                DELETE r
                """
                self.neo4j_driver.execute_query(
                    relations_query,
                    name=entity,
                    database_=self.neo4j_database
                )
                
                # Then delete all associated observations
                observations_query = """
                MATCH (e:Entity {name: $name})-[:HAS_OBSERVATION]->(o:Observation)
                DETACH DELETE o
                """
                self.neo4j_driver.execute_query(
                    observations_query,
                    name=entity,
                    database_=self.neo4j_database
                )
                
                # Finally delete the entity itself
                entity_query = """
                MATCH (e:Entity {name: $name})
                DETACH DELETE e
                """
                self.neo4j_driver.execute_query(
                    entity_query,
                    name=entity,
                    database_=self.neo4j_database
                )
                
                self.logger.info(f"Deleted entity: {entity}")
                
                # Since mem0 itself doesn't have a direct delete by name function,
                # we'll handle this through the Neo4j interface
                return dict_to_json({
                    "status": "success",
                    "message": f"Entity '{entity}' deleted successfully"
                })
            else:
                raise Exception("Neo4j driver not initialized")
                
        except Exception as e:
            error_msg = f"Error deleting entity: {str(e)}"
            self.logger.error(error_msg)
            return dict_to_json({"error": error_msg})
            
    def delete_relation(self, from_entity: str, to_entity: str, relationType: str) -> str:
        """
        Delete a specific relation from the knowledge graph.
        
        Args:
            from_entity: Name of the source entity
            to_entity: Name of the target entity
            relationType: Type of the relation
            
        Returns:
            JSON string with result information
        """
        try:
            self._ensure_initialized()
            
            # Delete relation using direct Neo4j connection
            if self.neo4j_driver:
                query = """
                MATCH (a:Entity {name: $from_name})-[r:HAS_RELATION]->(rel:Relation {type: $rel_type})-[r2:RELATES_TO]->(b:Entity {name: $to_name})
                DELETE r, rel, r2
                """
                self.neo4j_driver.execute_query(
                    query,
                    from_name=from_entity,
                    to_name=to_entity,
                    rel_type=relationType,
                    database_=self.neo4j_database
                )
                
                self.logger.info(f"Deleted relation: {from_entity} -{relationType}-> {to_entity}")
                
                return dict_to_json({
                    "status": "success", 
                    "message": f"Relation '{from_entity} -{relationType}-> {to_entity}' deleted successfully"
                })
            else:
                raise Exception("Neo4j driver not initialized")
                
        except Exception as e:
            error_msg = f"Error deleting relation: {str(e)}"
            self.logger.error(error_msg)
            return dict_to_json({"error": error_msg})
            
    def delete_observation(self, entity: str, content: str) -> str:
        """
        Delete a specific observation from an entity.
        
        Args:
            entity: Name of the entity
            content: Content of the observation to delete
            
        Returns:
            JSON string with result information
        """
        try:
            self._ensure_initialized()
            
            # Delete observation using direct Neo4j connection
            if self.neo4j_driver:
                query = """
                MATCH (e:Entity {name: $entity_name})-[r:HAS_OBSERVATION]->(o:Observation {content: $obs_content})
                DELETE r, o
                """
                self.neo4j_driver.execute_query(
                    query,
                    entity_name=entity,
                    obs_content=content,
                    database_=self.neo4j_database
                )
                
                self.logger.info(f"Deleted observation: '{content}' from entity {entity}")
                
                return dict_to_json({
                    "status": "success",
                    "message": f"Observation deleted successfully from entity {entity}"
                })
            else:
                raise Exception("Neo4j driver not initialized")
                
        except Exception as e:
            error_msg = f"Error deleting observation: {str(e)}"
            self.logger.error(error_msg)
            return dict_to_json({"error": error_msg})

    def _generate_embedding(self, text: str) -> Optional[List[float]]:
        """
        Generate an embedding for the given text using the configured provider.
        
        Args:
            text: The text to generate an embedding for
            
        Returns:
            A list of floats representing the embedding vector, or None if embedding generation fails
        """
        if not self.embedding_enabled:
            self.logger.debug("Embeddings are disabled, skipping embedding generation")
            return None
            
        try:
            self.logger.debug(f"Generating embedding for text: '{text[:50]}...' ({len(text)} chars)")
            embedding = self.embedding_manager.generate_embedding(text)
            
            if embedding:
                self.logger.debug(f"Successfully generated embedding with {len(embedding)} dimensions")
                return embedding
            else:
                self.logger.warn("Failed to generate embedding")
                return None
        except Exception as e:
            self.logger.error(f"Error generating embedding: {str(e)}", exc_info=True)
            return None 

    def apply_client_config(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """
        Apply client-provided configuration for embedding provider.
        This method updates the instance variables but does not reinitialize the memory.
        Call reinitialize() after this to apply the changes.
        
        Args:
            config: A dictionary containing embedding configuration
            
        Returns:
            Dictionary with status information
        """
        try:
            # Extract the embedder configuration
            embedder_config = config.get("embedder", {})
            
            if not embedder_config:
                return {"status": "error", "message": "Missing embedder configuration"}
            
            # Use our new configure_embeddings method
            result = self.configure_embeddings({
                "provider": embedder_config.get("provider", "none"),
                "config": embedder_config.get("config", {})
            })
            
            if result["status"] != "success":
                return result
            
            return {
                "status": "success", 
                "message": f"Applied configuration for {self.embedder_provider}",
                "provider": self.embedder_provider,
                "embedding_enabled": self.embedding_enabled
            }
            
        except Exception as e:
            self.logger.error(f"Error applying client configuration: {str(e)}")
            return {"status": "error", "message": f"Failed to apply configuration: {str(e)}"}
    
    def reinitialize(self) -> Dict[str, Any]:
        """
        Reinitialize the memory manager with the current configuration.
        This should be called after apply_client_config() to apply the changes.
        
        In SSE mode, this method will perform a non-disruptive reinitialization
        that preserves existing connections for other clients.
        
        Returns:
            Dictionary with status information
        """
        try:
            # Check if we're running in SSE mode
            use_sse = os.environ.get("USE_SSE", "").lower() in ["true", "1", "yes"]
            
            if use_sse:
                # Non-disruptive reinitialization for SSE mode
                self.logger.info("Performing non-disruptive reinitialization in SSE mode")
                
                # Store current driver temporarily if it exists
                current_driver = self.neo4j_driver
                
                # Create a new driver with the current configuration
                try:
                    # Create a new driver instance with the updated configuration
                    new_driver = GraphDatabase.driver(
                        self.neo4j_uri,
                        auth=(self.neo4j_user, self.neo4j_password),
                        # Connection pooling parameters
                        max_connection_lifetime=30 * 60,  # 30 minutes
                        max_connection_pool_size=50,      # Max 50 connections in the pool
                        connection_acquisition_timeout=60, # 60 seconds timeout for acquiring a connection
                        keep_alive=True                   # Keep connections alive
                    )
                    
                    # Test the new connection
                    test_result = new_driver.execute_query(
                        "RETURN 'Connection successful' AS message",
                        database_=self.neo4j_database
                    )
                    
                    # If we got here, the new connection is working
                    self.logger.info("New Neo4j connection successful")
                    
                    # Update the driver instance
                    self.neo4j_driver = new_driver
                    
                    # Close the old driver only after the new one is successfully established
                    if current_driver:
                        self.logger.debug("Closing old Neo4j driver")
                        current_driver.close()
                        
                except Exception as e:
                    # If creating the new driver fails, keep using the old one
                    self.logger.error(f"Failed to create new Neo4j driver: {str(e)}")
                    self.neo4j_driver = current_driver
                    raise e
                
                # Setup vector index if embeddings are enabled
                if self.embedding_enabled:
                    self._setup_vector_index()
                
                self.initialized = True
                
            else:
                # Standard reinitialization for stdio mode - can disrupt connections
                self.logger.info("Performing standard reinitialization in stdio mode")
                
                # Close any existing connections
                self.close()
                
                # Reset initialization flag
                self.initialized = False
                
                # Reinitialize with new configuration
                self.initialize()
            
            return {
                "status": "success",
                "message": f"Reinitialized memory manager with {self.embedder_provider} embedder",
                "provider": self.embedder_provider
            }
            
        except Exception as e:
            error_msg = f"Failed to reinitialize memory manager: {str(e)}"
            self.logger.error(error_msg)
            return {"status": "error", "message": error_msg}

    def create_lesson_container(self) -> str:
        """
        Create the global lessons container if it doesn't exist yet.
        
        Returns:
            JSON string with result information
        """
        try:
            self._ensure_initialized()
            
            if not self.neo4j_driver:
                return dict_to_json({
                    "status": "error",
                    "message": "Neo4j driver not initialized"
                })
            
            query = """
            MERGE (lc:MemoryContainer {name: "Lessons", type: "LessonsContainer"})
            ON CREATE SET lc.created = datetime(), 
                          lc.description = "Global container for all lessons learned"
            RETURN lc
            """
            
            result = self.neo4j_driver.execute_query(
                query,
                database_=self.neo4j_database
            )
            
            self.logger.info("Created or found Lessons Container in Neo4j")
            
            return dict_to_json({
                "status": "success",
                "message": "Lessons container created or found successfully"
            })
            
        except Exception as e:
            error_info = extract_error(e)
            self.logger.error(f"Error creating lessons container: {error_info}")
            return dict_to_json({
                "status": "error",
                "message": f"Failed to create lessons container: {error_info}"
            })

    def create_lesson(
        self,
        name: str,
        problem_description: str,
        context: Optional[str] = None,
        impact: str = "Medium",
        resolution: Optional[str] = None,
        what_was_learned: Optional[str] = None,
        why_it_matters: Optional[str] = None,
        how_to_apply: Optional[str] = None,
        root_cause: Optional[str] = None,
        evidence: Optional[str] = None,
        originated_from: Optional[str] = None,
        solved_with: Optional[str] = None,
        prevents: Optional[str] = None,
        builds_on: Optional[str] = None,
        applies_to: Optional[str] = None,
        confidence: float = 0.8,
        source: str = "Manual"
    ) -> str:
        """
        Create a comprehensive lesson with all relevant information.
        
        Args:
            name: Unique identifier for the lesson
            problem_description: Description of the problem that led to the lesson
            context: Where/when the lesson was encountered
            impact: Severity/impact of the problem (High/Medium/Low)
            resolution: Summary of how the problem was resolved
            what_was_learned: Factual knowledge gained
            why_it_matters: Consequences and benefits
            how_to_apply: Step-by-step application guidance
            root_cause: Underlying causes of the problem
            evidence: Concrete examples demonstrating the lesson
            originated_from: Entity the lesson originated from
            solved_with: Pattern/technique/tool that solved the problem
            prevents: Problem/anti-pattern the lesson helps prevent
            builds_on: Prior lesson this one builds upon
            applies_to: Domain/technology/context where lesson applies
            confidence: Confidence level (0.0-1.0) in the lesson's validity
            source: Source of the lesson (Manual, Automated, Inferred)
            
        Returns:
            JSON string with result information
        """
        try:
            self._ensure_initialized()
            
            if not self.neo4j_driver:
                return dict_to_json({
                    "status": "error",
                    "message": "Neo4j driver not initialized"
                })
            
            # First, ensure the Lessons container exists
            container_result = self.create_lesson_container()
            container_data = json.loads(container_result)
            if container_data.get("status") != "success":
                self.logger.error(f"Failed to create lessons container: {container_data.get('message')}")
                return dict_to_json({
                    "status": "error",
                    "message": f"Failed to create lesson due to container error: {container_data.get('message')}"
                })
            
            # Create the lesson entity
            lesson_query = """
            MERGE (l:Entity:Lesson {name: $name})
            ON CREATE SET 
                l.entityType = 'Lesson',
                l.created = datetime(),
                l.problemDescription = $problem_description,
                l.impact = $impact,
                l.status = 'Active',
                l.version = 1,
                l.confidence = $confidence,
                l.source = $source,
                l.lastRefreshed = datetime()
            """
            
            params = {
                "name": name,
                "problem_description": problem_description,
                "impact": impact,
                "confidence": confidence,
                "source": source
            }
            
            # Add optional parameters
            if context:
                lesson_query += ", l.context = $context"
                params["context"] = context
                
            if resolution:
                lesson_query += ", l.resolution = $resolution"
                params["resolution"] = resolution
                
            lesson_query += " RETURN l"
            
            # Create the lesson
            result = self.neo4j_driver.execute_query(
                lesson_query,
                parameters_=params,
                database_=self.neo4j_database
            )
            
            # Add lesson to the Lessons container
            container_query = """
            MATCH (lc:MemoryContainer {name: 'Lessons', type: 'LessonsContainer'})
            MATCH (l:Lesson {name: $name})
            MERGE (lc)-[r:CONTAINS]->(l)
            ON CREATE SET r.since = datetime()
            RETURN lc, l
            """
            
            self.neo4j_driver.execute_query(
                container_query,
                name=name,
                database_=self.neo4j_database
            )
            
            # Add observations if provided
            observations = []
            
            if what_was_learned:
                observations.append({
                    "type": "WhatWasLearned",
                    "content": what_was_learned
                })
                
            if why_it_matters:
                observations.append({
                    "type": "WhyItMatters",
                    "content": why_it_matters
                })
                
            if how_to_apply:
                observations.append({
                    "type": "HowToApply",
                    "content": how_to_apply
                })
                
            if root_cause:
                observations.append({
                    "type": "RootCause",
                    "content": root_cause
                })
                
            if evidence:
                observations.append({
                    "type": "Evidence",
                    "content": evidence
                })
            
            # Add all observations
            for obs in observations:
                obs_query = """
                MATCH (l:Lesson {name: $name})
                CREATE (o:Observation {
                    id: $obs_id,
                    type: $obs_type,
                    content: $obs_content,
                    created: datetime()
                })
                CREATE (l)-[r:HAS_OBSERVATION]->(o)
                RETURN o
                """
                
                self.neo4j_driver.execute_query(
                    obs_query,
                    name=name,
                    obs_id=generate_id(),
                    obs_type=obs["type"],
                    obs_content=obs["content"],
                    database_=self.neo4j_database
                )
            
            # Create relationships if provided
            if originated_from:
                self._create_relationship(name, originated_from, "ORIGINATED_FROM")
                
            if solved_with:
                self._create_relationship(name, solved_with, "SOLVED_WITH")
                
            if prevents:
                self._create_relationship(name, prevents, "PREVENTS")
                
            if builds_on:
                self._create_relationship(name, builds_on, "BUILDS_ON")
                
            if applies_to:
                self._create_relationship(name, applies_to, "APPLIES_TO")
            
            self.logger.info(f"Created lesson: {name}")
            
            return dict_to_json({
                "status": "success",
                "message": f"Lesson '{name}' created successfully",
                "data": {
                    "name": name,
                    "version": 1,
                    "status": "Active",
                    "observation_count": len(observations),
                    "relationship_count": sum(1 for r in [originated_from, solved_with, prevents, builds_on, applies_to] if r)
                }
            })
            
        except Exception as e:
            error_info = extract_error(e)
            self.logger.error(f"Error creating lesson: {error_info}")
            return dict_to_json({
                "status": "error",
                "message": f"Failed to create lesson: {error_info}"
            })

    def _create_relationship(self, from_entity: str, to_entity: str, relationship_type: str) -> None:
        """
        Create a relationship between two entities.
        
        Args:
            from_entity: Source entity name
            to_entity: Target entity name
            relationship_type: Type of relationship
        """
        if not self.neo4j_driver:
            return
            
        # Ensure the target entity exists
        create_entity_query = """
        MERGE (e:Entity {name: $name})
        ON CREATE SET e.created = datetime()
        RETURN e
        """
        
        self.neo4j_driver.execute_query(
            create_entity_query,
            name=to_entity,
            database_=self.neo4j_database
        )
        
        # Create the relationship
        rel_query = """
        MATCH (from:Entity {name: $from_name})
        MATCH (to:Entity {name: $to_name})
        CALL apoc.merge.relationship(from, $rel_type, {since: datetime()}, {}, to)
        YIELD rel
        RETURN rel
        """
        
        try:
            self.neo4j_driver.execute_query(
                rel_query,
                from_name=from_entity,
                to_name=to_entity,
                rel_type=relationship_type,
                database_=self.neo4j_database
            )
        except Exception as e:
            # Fall back to basic relationship creation if APOC is not available
            self.logger.warn(f"APOC procedure not available, using basic relationship creation: {str(e)}")
            
            basic_rel_query = """
            MATCH (from:Entity {name: $from_name})
            MATCH (to:Entity {name: $to_name})
            MERGE (from)-[r:$rel_type]->(to)
            ON CREATE SET r.since = datetime()
            RETURN r
            """
            
            self.neo4j_driver.execute_query(
                basic_rel_query,
                from_name=from_entity,
                to_name=to_entity,
                rel_type=relationship_type,
                database_=self.neo4j_database
            ) 

    def get_lessons(
        self,
        filter_criteria: Optional[Dict[str, Any]] = None,
        related_to: Optional[str] = None,
        applies_to: Optional[str] = None,
        limit: int = 50,
        include_superseded: bool = False,
        min_confidence: float = 0.0,
        sort_by: str = "relevance",
        include_observations: bool = True
    ) -> str:
        """
        Retrieve lessons based on various criteria.
        
        Args:
            filter_criteria: Dictionary of property filters to apply
            related_to: Entity name that lessons should be related to
            applies_to: Domain/context that lessons should apply to
            limit: Maximum number of lessons to return
            include_superseded: Whether to include superseded lesson versions
            min_confidence: Minimum confidence score for returned lessons
            sort_by: Field to sort results by ("relevance", "date", "confidence")
            include_observations: Whether to include lesson observations
            
        Returns:
            JSON string with lesson results
        """
        try:
            self._ensure_initialized()
            
            if not self.neo4j_driver:
                return dict_to_json({
                    "status": "error",
                    "message": "Neo4j driver not initialized"
                })
            
            # Base query starts with the lessons container
            base_query = """
            MATCH (lc:MemoryContainer {name: 'Lessons', type: 'LessonsContainer'})-[:CONTAINS]->(l:Lesson)
            WHERE l.confidence >= $min_confidence
            """
            
            params: Dict[str, Any] = {
                "min_confidence": min_confidence
            }
            
            # Add status filter
            if not include_superseded:
                base_query += " AND (l.status = 'Active' OR l.status IS NULL)"
            
            # Add any additional filter criteria
            if filter_criteria:
                for key, value in filter_criteria.items():
                    if key and value:
                        safe_key = key.replace(".", "_") # Sanitize key for parameter naming
                        base_query += f" AND l.{key} = ${safe_key}"
                        params[safe_key] = value
            
            # Add relationship filters
            rel_conditions = []
            
            if related_to:
                rel_conditions.append("""
                EXISTS (
                    MATCH (l)-[rel]-(related:Entity {name: $related_to})
                    RETURN rel
                )
                """)
                params["related_to"] = related_to
            
            if applies_to:
                rel_conditions.append("""
                EXISTS (
                    MATCH (l)-[:APPLIES_TO]->(domain:Entity {name: $applies_to})
                    RETURN domain
                )
                """)
                params["applies_to"] = applies_to
            
            # Add relationship conditions to query
            for condition in rel_conditions:
                base_query += f" AND {condition}"
            
            # Add sorting
            if sort_by == "date":
                base_query += " ORDER BY l.created DESC"
            elif sort_by == "confidence":
                base_query += " ORDER BY l.confidence DESC"
            else:  # Default to relevance/lastRefreshed
                base_query += " ORDER BY l.lastRefreshed DESC"
            
            # Add limit
            base_query += " LIMIT $limit"
            params["limit"] = limit
            
            # Complete the query with lesson return
            if include_observations:
                query = base_query + """
                OPTIONAL MATCH (l)-[:HAS_OBSERVATION]->(o:Observation)
                RETURN l, collect(o) as observations
                """
            else:
                query = base_query + """
                RETURN l
                """
            
            # Execute the query
            result = self._safe_execute_query(
                query,
                parameters_=params,
                database_=self.neo4j_database
            )
            
            # Process results
            lessons = []
            records = result[0] if result and len(result) > 0 else []
            
            for record in records:
                lesson_node = record["l"]
                
                # Convert Neo4j node to Python dict
                lesson_data = {k: v for k, v in lesson_node.items()}
                
                # Process datetime objects
                for key, value in lesson_data.items():
                    if hasattr(value, 'iso_format'):  # Check if it's a datetime
                        lesson_data[key] = value.iso_format()
                
                # Add observations if included
                if include_observations and "observations" in record:
                    observations = []
                    for obs in record["observations"]:
                        if obs:  # Ensure observation is not None
                            obs_data = {k: v for k, v in obs.items()}
                            # Process datetime objects in observations
                            for key, value in obs_data.items():
                                if hasattr(value, 'iso_format'):
                                    obs_data[key] = value.iso_format()
                            observations.append(obs_data)
                    
                    lesson_data["observations"] = observations
                
                lessons.append(lesson_data)
            
            self.logger.info(f"Retrieved {len(lessons)} lessons matching criteria")
            
            return dict_to_json({
                "status": "success",
                "count": len(lessons),
                "lessons": lessons
            })
            
        except Exception as e:
            error_info = extract_error(e)
            self.logger.error(f"Error retrieving lessons: {error_info}")
            return dict_to_json({
                "status": "error",
                "message": f"Failed to retrieve lessons: {error_info}"
            })

    def update_lesson(
        self,
        lesson_name: str,
        updated_properties: Optional[Dict[str, Any]] = None,
        updated_observations: Optional[Dict[str, str]] = None,
        new_relationships: Optional[Dict[str, List[str]]] = None,
        update_confidence: bool = True
    ) -> str:
        """
        Update a lesson, creating a new version that supersedes the old one.
        
        Args:
            lesson_name: Name of the lesson to update
            updated_properties: Dictionary of properties to update
            updated_observations: Dictionary of observations to update (type -> content)
            new_relationships: Dictionary of relationships to add (type -> list of target entities)
            update_confidence: Whether to adjust confidence based on repeated reinforcement
            
        Returns:
            JSON string with result information
        """
        try:
            self._ensure_initialized()
            
            if not self.neo4j_driver:
                return dict_to_json({
                    "status": "error",
                    "message": "Neo4j driver not initialized"
                })
            
            # First, get the existing lesson
            old_lesson_query = """
            MATCH (l:Lesson {name: $name})
            WHERE l.status = 'Active' OR l.status IS NULL
            RETURN l
            """
            
            result = self.neo4j_driver.execute_query(
                old_lesson_query,
                name=lesson_name,
                database_=self.neo4j_database
            )
            
            records = result[0] if result and len(result) > 0 else []
            
            if not records:
                return dict_to_json({
                    "status": "error",
                    "message": f"Lesson '{lesson_name}' not found or not active"
                })
            
            old_lesson = records[0]["l"]
            
            # Get the current version number
            current_version = old_lesson.get("version", 1)
            new_version = current_version + 1
            
            # Create the new version with incremented version number
            # Start by copying properties from the old lesson
            new_properties = {k: v for k, v in old_lesson.items() if k != "status"}
            new_properties["version"] = new_version
            new_properties["created"] = datetime.datetime.now()
            new_properties["lastRefreshed"] = datetime.datetime.now()
            
            # Apply updated properties if provided
            if updated_properties:
                for key, value in updated_properties.items():
                    if key not in ["name", "version", "created"]:  # Protect key fields
                        new_properties[key] = value
            
            # Adjust confidence if requested
            if update_confidence and "confidence" in new_properties:
                # Increase confidence slightly with each reinforcement, capped at 0.99
                current_confidence = float(new_properties["confidence"])
                new_confidence = min(current_confidence + 0.05, 0.99)
                new_properties[f"confidence"] = new_confidence
            
            # Create the new lesson version
            new_lesson_name = f"{lesson_name}_v{new_version}"
            
            create_new_version_query = """
            CREATE (l:Entity:Lesson {name: $name})
            SET l += $properties
            RETURN l
            """
            
            self.neo4j_driver.execute_query(
                create_new_version_query,
                name=new_lesson_name,
                properties=new_properties,
                database_=self.neo4j_database
            )
            
            # Create SUPERSEDES relationship between new and old versions
            supersedes_query = """
            MATCH (new:Lesson {name: $new_name})
            MATCH (old:Lesson {name: $old_name})
            MERGE (new)-[r:SUPERSEDES]->(old)
            ON CREATE SET r.since = datetime()
            RETURN r
            """
            
            self.neo4j_driver.execute_query(
                supersedes_query,
                new_name=new_lesson_name,
                old_name=lesson_name,
                database_=self.neo4j_database
            )
            
            # Update status of old version to "Superseded"
            update_status_query = """
            MATCH (l:Lesson {name: $name})
            SET l.status = 'Superseded'
            """
            
            self.neo4j_driver.execute_query(
                update_status_query,
                name=lesson_name,
                database_=self.neo4j_database
            )
            
            # Add the new lesson to the Lessons container
            container_query = """
            MATCH (lc:MemoryContainer {name: 'Lessons', type: 'LessonsContainer'})
            MATCH (l:Lesson {name: $name})
            MERGE (lc)-[r:CONTAINS]->(l)
            ON CREATE SET r.since = datetime()
            RETURN lc, l
            """
            
            self.neo4j_driver.execute_query(
                container_query,
                name=new_lesson_name,
                database_=self.neo4j_database
            )
            
            # Copy existing observations from old lesson to new one
            copy_observations_query = """
            MATCH (old:Lesson {name: $old_name})-[:HAS_OBSERVATION]->(o:Observation)
            MATCH (new:Lesson {name: $new_name})
            CREATE (new)-[:HAS_OBSERVATION]->(o)
            RETURN count(o) as observation_count
            """
            
            copy_result = self.neo4j_driver.execute_query(
                copy_observations_query,
                old_name=lesson_name,
                new_name=new_lesson_name,
                database_=self.neo4j_database
            )
            
            observation_count = 0
            if copy_result and copy_result[0]:
                observation_count = copy_result[0][0].get("observation_count", 0)
            
            # Update observations if provided
            if updated_observations:
                for obs_type, content in updated_observations.items():
                    # First, try to find existing observation of this type
                    find_obs_query = """
                    MATCH (l:Lesson {name: $name})-[:HAS_OBSERVATION]->(o:Observation)
                    WHERE o.type = $type
                    RETURN o
                    """
                    
                    obs_result = self.neo4j_driver.execute_query(
                        find_obs_query,
                        name=new_lesson_name,
                        type=obs_type,
                        database_=self.neo4j_database
                    )
                    
                    obs_records = obs_result[0] if obs_result and len(obs_result[0]) > 0 else []
                    
                    if obs_records:
                        # Update existing observation
                        update_obs_query = """
                        MATCH (l:Lesson {name: $name})-[:HAS_OBSERVATION]->(o:Observation)
                        WHERE o.type = $type
                        SET o.content = $content, o.updated = datetime()
                        RETURN o
                        """
                        
                        self.neo4j_driver.execute_query(
                            update_obs_query,
                            name=new_lesson_name,
                            type=obs_type,
                            content=content,
                            database_=self.neo4j_database
                        )
                    else:
                        # Create new observation
                        new_obs_query = """
                        MATCH (l:Lesson {name: $name})
                        CREATE (o:Observation {
                            id: $obs_id,
                            type: $type,
                            content: $content,
                            created: datetime()
                        })
                        CREATE (l)-[r:HAS_OBSERVATION]->(o)
                        RETURN o
                        """
                        
                        self.neo4j_driver.execute_query(
                            new_obs_query,
                            name=new_lesson_name,
                            obs_id=generate_id(),
                            type=obs_type,
                            content=content,
                            database_=self.neo4j_database
                        )
                        
                        observation_count += 1
            
            # Copy existing relationships (except SUPERSEDES) from old lesson to new one
            copy_relations_query = """
            MATCH (old:Lesson {name: $old_name})-[r]->(target)
            WHERE type(r) <> 'SUPERSEDES' AND type(r) <> 'HAS_OBSERVATION'
            MATCH (new:Lesson {name: $new_name})
            CALL apoc.create.relationship(new, type(r), {since: datetime()}, target)
            YIELD rel
            RETURN count(rel) as relation_count
            """
            
            try:
                # Try to use APOC procedure first
                copy_rel_result = self.neo4j_driver.execute_query(
                    copy_relations_query,
                    old_name=lesson_name,
                    new_name=new_lesson_name,
                    database_=self.neo4j_database
                )
                
                relation_count = 0
                if copy_rel_result and copy_rel_result[0]:
                    relation_count = copy_rel_result[0][0].get("relation_count", 0)
                    
            except Exception as e:
                # Fall back to simpler relationship copying if APOC is not available
                self.logger.warn(f"APOC procedure not available for relationship copying: {str(e)}")
                
                # Get existing relationships
                get_rels_query = """
                MATCH (old:Lesson {name: $old_name})-[r]->(target)
                WHERE type(r) <> 'SUPERSEDES' AND type(r) <> 'HAS_OBSERVATION'
                RETURN type(r) as rel_type, target.name as target_name
                """
                
                rels_result = self.neo4j_driver.execute_query(
                    get_rels_query,
                    old_name=lesson_name,
                    database_=self.neo4j_database
                )
                
                relation_count = 0
                if rels_result and rels_result[0]:
                    for record in rels_result[0]:
                        rel_type = record.get("rel_type")
                        target_name = record.get("target_name")
                        
                        if rel_type and target_name:
                            # Create the relationship
                            self._create_relationship(new_lesson_name, target_name, rel_type)
                            relation_count += 1
            
            # Add new relationships if provided
            if new_relationships:
                for rel_type, targets in new_relationships.items():
                    for target in targets:
                        self._create_relationship(new_lesson_name, target, rel_type)
                        relation_count += 1
            
            self.logger.info(f"Updated lesson '{lesson_name}' to version {new_version} as '{new_lesson_name}'")
            
            return dict_to_json({
                "status": "success",
                "message": f"Lesson '{lesson_name}' updated successfully to version {new_version}",
                "data": {
                    "name": new_lesson_name,
                    "old_name": lesson_name,
                    "version": new_version,
                    "status": "Active",
                    "observation_count": observation_count,
                    "relationship_count": relation_count
                }
            })
            
        except Exception as e:
            error_info = extract_error(e)
            self.logger.error(f"Error updating lesson: {error_info}")
            return dict_to_json({
                "status": "error",
                "message": f"Failed to update lesson: {error_info}"
            })

    def apply_lesson_to_context(
        self,
        lesson_name: str,
        context_entity: str,
        application_notes: Optional[str] = None,
        success_score: Optional[float] = None
    ) -> str:
        """
        Record that a lesson was applied to a specific context.
        
        Args:
            lesson_name: Name of the lesson that was applied
            context_entity: Name of the entity where the lesson was applied
            application_notes: Optional notes about the application
            success_score: Optional score (0.0-1.0) indicating how successful the application was
            
        Returns:
            JSON string with result information
        """
        try:
            self._ensure_initialized()
            
            if not self.neo4j_driver:
                return dict_to_json({
                    "status": "error",
                    "message": "Neo4j driver not initialized"
                })
            
            # First, ensure the lesson exists
            lesson_query = """
            MATCH (l:Lesson {name: $name})
            WHERE l.status = 'Active' OR l.status IS NULL
            RETURN l
            """
            
            lesson_result = self.neo4j_driver.execute_query(
                lesson_query,
                name=lesson_name,
                database_=self.neo4j_database
            )
            
            lesson_records = lesson_result[0] if lesson_result and len(lesson_result[0]) > 0 else []
            
            if not lesson_records:
                return dict_to_json({
                    "status": "error",
                    "message": f"Active lesson '{lesson_name}' not found"
                })
            
            # Next, ensure the context entity exists
            context_query = """
            MERGE (e:Entity {name: $name})
            ON CREATE SET e.created = datetime()
            RETURN e
            """
            
            self.neo4j_driver.execute_query(
                context_query,
                name=context_entity,
                database_=self.neo4j_database
            )
            
            # Create application relationship with metadata
            application_query = """
            MATCH (l:Lesson {name: $lesson_name})
            MATCH (c:Entity {name: $context_name})
            MERGE (l)-[r:APPLIED_TO]->(c)
            ON CREATE SET r.first_applied = datetime()
            SET r.last_applied = datetime(),
                r.application_count = COALESCE(r.application_count, 0) + 1
            """
            
            # Add optional parameters if provided
            params = {
                "lesson_name": lesson_name,
                "context_name": context_entity
            }
            
            if application_notes:
                application_query += ", r.notes = $notes"
                params["notes"] = application_notes
                
            if success_score is not None:
                application_query += ", r.success_score = $score"
                params["score"] = str(success_score)  # Convert float to string
                
            application_query += " RETURN r"
            
            # Create the relationship
            application_result = self._safe_execute_query(
                application_query,
                parameters_=params,
                database_=self.neo4j_database
            )
            
            # Update lesson's lastRefreshed and relevance score
            update_lesson_query = """
            MATCH (l:Lesson {name: $name})
            SET l.lastRefreshed = datetime()
            """
            
            # If success score provided, adjust relevance score
            if success_score is not None:
                update_lesson_query += """
                , l.relevanceScore = CASE 
                    WHEN l.relevanceScore IS NULL THEN $score
                    ELSE (l.relevanceScore * 0.7) + ($score * 0.3)
                  END
                """
            
            update_lesson_query += " RETURN l"
            
            update_params = {
                "name": lesson_name
            }
            
            if success_score is not None:
                update_params["score"] = str(success_score)  # Convert float to string
                
            self._safe_execute_query(
                update_lesson_query,
                parameters_=update_params,
                database_=self.neo4j_database
            )
            
            self.logger.info(f"Applied lesson '{lesson_name}' to context '{context_entity}'")
            
            return dict_to_json({
                "status": "success",
                "message": f"Lesson '{lesson_name}' applied to '{context_entity}' successfully",
                "data": {
                    "lesson_name": lesson_name,
                    "context_entity": context_entity,
                    "application_time": datetime.datetime.now().isoformat()
                }
            })
            
        except Exception as e:
            error_info = extract_error(e)
            self.logger.error(f"Error applying lesson to context: {error_info}")
            return dict_to_json({
                "status": "error",
                "message": f"Failed to apply lesson to context: {error_info}"
            })

    def extract_potential_lessons(
        self,
        conversation_text: Optional[str] = None,
        code_diff: Optional[str] = None,
        issue_description: Optional[str] = None,
        error_logs: Optional[str] = None,
        min_confidence: float = 0.6
    ) -> str:
        """
        Analyze various inputs to automatically extract potential lessons.
        
        Args:
            conversation_text: Text of a conversation to analyze
            code_diff: Code changes to analyze
            issue_description: Description of an issue to analyze
            error_logs: Error logs to analyze
            min_confidence: Minimum confidence threshold for extracted lessons
            
        Returns:
            JSON string with extracted lesson candidates and confidence scores
        """
        try:
            self._ensure_initialized()
            
            if not self.neo4j_driver:
                return dict_to_json({
                    "status": "error",
                    "message": "Neo4j driver not initialized"
                })
            
            # Collect all input sources
            sources = []
            if conversation_text:
                sources.append(("conversation", conversation_text))
            if code_diff:
                sources.append(("code_diff", code_diff))
            if issue_description:
                sources.append(("issue", issue_description))
            if error_logs:
                sources.append(("error_logs", error_logs))
                
            if not sources:
                return dict_to_json({
                    "status": "error",
                    "message": "No input sources provided for lesson extraction"
                })
            
            # In a production system, this would use NLP or LLM to extract lessons
            # For this implementation, we'll use a simplified rule-based approach
            
            extracted_lessons = []
            
            for source_type, source_text in sources:
                # Look for problem-solution patterns in the text
                lessons_from_source = self._extract_lessons_from_text(source_type, source_text, min_confidence)
                extracted_lessons.extend(lessons_from_source)
            
            # Filter lessons by confidence threshold
            filtered_lessons = [lesson for lesson in extracted_lessons if lesson["confidence"] >= min_confidence]
            
            # Sort by confidence score
            filtered_lessons.sort(key=lambda x: x["confidence"], reverse=True)
            
            self.logger.info(f"Extracted {len(filtered_lessons)} potential lessons with confidence >= {min_confidence}")
            
            return dict_to_json({
                "status": "success",
                "count": len(filtered_lessons),
                "extracted_lessons": filtered_lessons
            })
            
        except Exception as e:
            error_info = extract_error(e)
            self.logger.error(f"Error extracting potential lessons: {error_info}")
            return dict_to_json({
                "status": "error",
                "message": f"Failed to extract potential lessons: {error_info}"
            })
    
    def _extract_lessons_from_text(self, source_type: str, text: str, min_confidence: float) -> List[Dict[str, Any]]:
        """
        Extract potential lessons from a text source.
        
        Args:
            source_type: Type of the source (conversation, code_diff, issue, error_logs)
            text: The text to analyze
            min_confidence: Minimum confidence threshold
            
        Returns:
            List of extracted lesson candidates with metadata
        """
        lessons = []
        
        # Split the text into paragraphs for analysis
        paragraphs = text.split("\n\n")
        
        # Common problem indicators
        problem_indicators = [
            "error", "bug", "issue", "problem", "fail", "exception",
            "incorrect", "unexpected", "wrong", "broken", "crash",
            "critical", "warning"
        ]
        
        # Common solution indicators
        solution_indicators = [
            "fix", "solv", "resolv", "solution", "implement",
            "correct", "update", "change", "modify", "adjustment",
            "approach", "technique", "pattern", "best practice"
        ]
        
        # Common lesson indicators
        lesson_indicators = [
            "learn", "lesson", "takeaway", "insight", "understand",
            "realize", "discover", "found out", "important to note"
        ]
        
        # Analyze each paragraph for problem-solution patterns
        for i, paragraph in enumerate(paragraphs):
            paragraph = paragraph.lower()
            
            # Calculate indicator presence scores
            problem_score = sum(1 for indicator in problem_indicators if indicator in paragraph)
            solution_score = sum(1 for indicator in solution_indicators if indicator in paragraph)
            lesson_score = sum(1 for indicator in lesson_indicators if indicator in paragraph)
            
            # Check context in surrounding paragraphs
            context_paragraphs = []
            if i > 0:
                context_paragraphs.append(paragraphs[i-1])
            if i < len(paragraphs) - 1:
                context_paragraphs.append(paragraphs[i+1])
            
            context_text = " ".join(context_paragraphs).lower()
            context_problem_score = sum(1 for indicator in problem_indicators if indicator in context_text)
            context_solution_score = sum(1 for indicator in solution_indicators if indicator in context_text)
            
            # Calculate a confidence score based on indicators
            # This is a simplified heuristic - a real system would use more sophisticated NLP
            base_confidence = 0.0
            
            # If it has both problem and solution aspects, it's likely a lesson
            if problem_score > 0 and solution_score > 0:
                base_confidence = 0.5 + (min(problem_score, solution_score) * 0.05)
            
            # If it explicitly mentions being a lesson, increase confidence
            if lesson_score > 0:
                base_confidence += (lesson_score * 0.1)
            
            # Context from surrounding paragraphs adds confidence
            if context_problem_score > 0 or context_solution_score > 0:
                base_confidence += 0.1
            
            # Cap at 0.95 - leaving room for human verification
            confidence = min(base_confidence, 0.95)
            
            # If confidence is above threshold, extract a potential lesson
            if confidence >= min_confidence:
                # Generate a suitable lesson name (simplified)
                words = paragraph.split()
                # Take first 5-10 significant words for the name
                name_words = [word for word in words[:20] 
                              if len(word) > 3 and word not in ["with", "that", "this", "from", "when", "what"]][:5]
                
                if not name_words:
                    name_words = ["Extracted", "Lesson"]
                
                lesson_name = "Lesson_" + "_".join(name_words).title().replace(".", "").replace(",", "")
                
                # Extract problem and solution parts (very simplified)
                problem = None
                solution = None
                
                for sentence in paragraph.split("."):
                    sentence = sentence.strip()
                    if not sentence:
                        continue
                        
                    sentence_problem_score = sum(1 for indicator in problem_indicators if indicator in sentence.lower())
                    sentence_solution_score = sum(1 for indicator in solution_indicators if indicator in sentence.lower())
                    
                    if sentence_problem_score > sentence_solution_score and not problem:
                        problem = sentence
                    elif sentence_solution_score > sentence_problem_score and not solution:
                        solution = sentence
                
                # Add extracted lesson
                lesson = {
                    "name": lesson_name,
                    "source_type": source_type,
                    "confidence": confidence,
                    "problem_description": problem or paragraph[:100],
                    "resolution": solution or "",
                    "raw_text": paragraph,
                    "context": "\n".join(context_paragraphs)
                }
                
                # Add source-specific metadata
                if source_type == "code_diff":
                    lesson["impact"] = "Medium"  # Default for code changes
                elif source_type == "error_logs":
                    lesson["impact"] = "High"  # Errors typically have higher impact
                else:
                    lesson["impact"] = "Low"  # Default for other sources
                
                lessons.append(lesson)
        
        return lessons
    
    def consolidate_related_lessons(
        self,
        lesson_ids: List[str],
        new_name: Optional[str] = None,
        strategy: str = "merge",
        confidence_handling: str = "max"
    ) -> str:
        """
        Combine several related lessons into a more comprehensive one.
        
        Args:
            lesson_ids: List of lessons to consolidate
            new_name: Name for consolidated lesson (auto-generated if None)
            strategy: How to combine content ('merge', 'summarize')
            confidence_handling: How to set confidence ('max', 'avg', 'weighted')
            
        Returns:
            JSON string with result information
        """
        try:
            self._ensure_initialized()
            
            if not self.neo4j_driver:
                return dict_to_json({
                    "status": "error",
                    "message": "Neo4j driver not initialized"
                })
                
            if not lesson_ids or len(lesson_ids) < 2:
                return dict_to_json({
                    "status": "error",
                    "message": "At least two lesson IDs are required for consolidation"
                })
            
            # Retrieve all lessons to consolidate
            lessons_query = """
            MATCH (l:Lesson)
            WHERE l.name IN $lesson_ids
            AND (l.status = 'Active' OR l.status IS NULL)
            RETURN l
            """
            
            lessons_result = self.neo4j_driver.execute_query(
                lessons_query,
                lesson_ids=lesson_ids,
                database_=self.neo4j_database
            )
            
            lessons_records = lessons_result[0] if lessons_result and len(lessons_result[0]) > 0 else []
            
            if not lessons_records:
                return dict_to_json({
                    "status": "error",
                    "message": "No active lessons found with the provided IDs"
                })
                
            if len(lessons_records) < len(lesson_ids):
                missing_count = len(lesson_ids) - len(lessons_records)
                self.logger.warn(f"{missing_count} lessons not found or not active")
            
            # Extract lesson data from records
            lessons_data = []
            for record in lessons_records:
                lesson = record["l"]
                lesson_data = {k: v for k, v in lesson.items()}
                lessons_data.append(lesson_data)
            
            # Generate a name for the consolidated lesson if not provided
            if not new_name:
                # Use common words from existing lesson names
                lesson_names = [data.get("name", "").split("_") for data in lessons_data]
                flattened_names = [word for name_parts in lesson_names for word in name_parts]
                common_words = [word for word in flattened_names if flattened_names.count(word) > 1 and len(word) > 3]
                
                if common_words:
                    new_name = "Consolidated_" + "_".join(sorted(set(common_words)))[:50]
                else:
                    # If no common words, use a generic name with timestamp
                    timestamp = datetime.datetime.now().strftime("%Y%m%d%H%M")
                    new_name = f"Consolidated_Lesson_{timestamp}"
            
            # Calculate confidence for the consolidated lesson
            confidences = [float(data.get("confidence", 0.8)) for data in lessons_data]
            if confidence_handling == "max":
                calculated_confidence = max(confidences)
            elif confidence_handling == "avg":
                calculated_confidence = sum(confidences) / len(confidences)
            elif confidence_handling == "weighted":
                # Weight by version number (more refined lessons count more)
                versions = [int(data.get("version", 1)) for data in lessons_data]
                calculated_confidence = sum(c * v for c, v in zip(confidences, versions)) / sum(versions)
            else:
                calculated_confidence = 0.8  # Default
            
            # Combine content based on strategy
            problem_descriptions = [data.get("problemDescription", "") for data in lessons_data if data.get("problemDescription")]
            resolutions = [data.get("resolution", "") for data in lessons_data if data.get("resolution")]
            contexts = [data.get("context", "") for data in lessons_data if data.get("context")]
            
            if strategy == "merge":
                # Combine all content with delimiters
                combined_description = "\n\n".join(f"• {desc}" for desc in problem_descriptions)
                combined_resolution = "\n\n".join(f"• {res}" for res in resolutions)
                combined_context = ", ".join(set(contexts))
            else:  # summarize strategy - just take the first one for now
                # In a production system, this would use NLP to create a true summary
                combined_description = problem_descriptions[0] if problem_descriptions else ""
                combined_resolution = resolutions[0] if resolutions else ""
                combined_context = contexts[0] if contexts else ""
            
            # Determine impact (highest impact wins)
            impact_order = {"High": 3, "Medium": 2, "Low": 1}
            impacts = [data.get("impact", "Medium") for data in lessons_data]
            highest_impact = max(impacts, key=lambda x: impact_order.get(x, 0))
            
            # Create the consolidated lesson
            create_lesson_query = """
            CREATE (l:Entity:Lesson {
              name: $name,
              entityType: 'Lesson',
              context: $context,
              problemDescription: $problem_description,
              impact: $impact,
              resolution: $resolution,
              status: 'Active',
              version: 1,
              confidence: $confidence,
              source: 'Consolidated',
              created: datetime(),
              lastRefreshed: datetime()
            })
            RETURN l
            """
            
            create_params = {
                "name": new_name,
                "context": combined_context,
                "problem_description": combined_description,
                "impact": highest_impact,
                "resolution": combined_resolution,
                "confidence": calculated_confidence
            }
            
            self.neo4j_driver.execute_query(
                create_lesson_query,
                parameters_=create_params,
                database_=self.neo4j_database
            )
            
            # Add the consolidated lesson to the Lessons container
            container_query = """
            MATCH (lc:MemoryContainer {name: 'Lessons', type: 'LessonsContainer'})
            MATCH (l:Lesson {name: $name})
            MERGE (lc)-[r:CONTAINS]->(l)
            ON CREATE SET r.since = datetime()
            RETURN lc, l
            """
            
            self.neo4j_driver.execute_query(
                container_query,
                name=new_name,
                database_=self.neo4j_database
            )
            
            # Create consolidation relationships
            for lesson_id in lesson_ids:
                consolidation_query = """
                MATCH (master:Lesson {name: $master_name})
                MATCH (component:Lesson {name: $component_name})
                MERGE (master)-[r:CONSOLIDATES]->(component)
                ON CREATE SET r.since = datetime()
                RETURN r
                """
                
                try:
                    self.neo4j_driver.execute_query(
                        consolidation_query,
                        master_name=new_name,
                        component_name=lesson_id,
                        database_=self.neo4j_database
                    )
                except Exception as e:
                    self.logger.warn(f"Failed to create consolidation relationship for {lesson_id}: {str(e)}")
            
            # Set component lessons as consolidated
            update_status_query = """
            MATCH (master:Lesson {name: $master_name})-[:CONSOLIDATES]->(component:Lesson)
            SET component.status = 'Consolidated'
            RETURN count(component) as updated_count
            """
            
            status_result = self.neo4j_driver.execute_query(
                update_status_query,
                master_name=new_name,
                database_=self.neo4j_database
            )
            
            updated_count = 0
            if status_result and status_result[0]:
                updated_count = status_result[0][0].get("updated_count", 0)
            
            # Collect all observations from component lessons
            consolidated_observations = {}
            
            for lesson_id in lesson_ids:
                observations_query = """
                MATCH (l:Lesson {name: $name})-[:HAS_OBSERVATION]->(o:Observation)
                RETURN o.type as type, o.content as content
                """
                
                obs_result = self.neo4j_driver.execute_query(
                    observations_query,
                    name=lesson_id,
                    database_=self.neo4j_database
                )
                
                if obs_result and obs_result[0]:
                    for record in obs_result[0]:
                        obs_type = record.get("type")
                        content = record.get("content")
                        
                        if obs_type and content:
                            if obs_type not in consolidated_observations:
                                consolidated_observations[obs_type] = []
                            consolidated_observations[obs_type].append(content)
            
            # Add consolidated observations to the master lesson
            for obs_type, contents in consolidated_observations.items():
                if strategy == "merge":
                    # Combine all contents with delimiters
                    combined_content = "\n\n".join(f"• {content}" for content in contents)
                else:  # summarize
                    # Just take the first one in this simplified implementation
                    combined_content = contents[0] if contents else ""
                
                obs_query = """
                MATCH (l:Lesson {name: $name})
                CREATE (o:Observation {
                    id: $obs_id,
                    type: $type,
                    content: $content,
                    created: datetime()
                })
                CREATE (l)-[r:HAS_OBSERVATION]->(o)
                RETURN o
                """
                
                self.neo4j_driver.execute_query(
                    obs_query,
                    name=new_name,
                    obs_id=generate_id(),
                    type=obs_type,
                    content=combined_content,
                    database_=self.neo4j_database
                )
            
            # Migrate relevant relationships from component lessons to master lesson
            relationship_types = ["APPLIES_TO", "SOLVED_WITH", "PREVENTS", "BUILDS_ON", "ORIGINATED_FROM"]
            migrated_relationships = 0
            
            for rel_type in relationship_types:
                migrate_rels_query = f"""
                MATCH (master:Lesson {{name: $master_name}})
                MATCH (master)-[:CONSOLIDATES]->(component:Lesson)-[r:{rel_type}]->(target)
                WHERE NOT EXISTS((master)-[:{rel_type}]->(target))
                WITH master, target, collect(component.name) as sources
                MERGE (master)-[new_rel:{rel_type}]->(target)
                ON CREATE SET new_rel.sources = sources,
                               new_rel.since = datetime()
                RETURN count(new_rel) as rel_count
                """
                
                migrate_rels_query_str = cast(LiteralString, migrate_rels_query)  # Cast to LiteralString
                copy_rel_result = self._safe_execute_query(
                    migrate_rels_query,
                    parameters_={"master_name": new_name},
                    database_=self.neo4j_database
                )
                
                if copy_rel_result and copy_rel_result[0]:
                    migrated_relationships += copy_rel_result[0][0].get("rel_count", 0)
            
            self.logger.info(f"Consolidated {len(lessons_records)} lessons into '{new_name}'")
            
            return dict_to_json({
                "status": "success",
                "message": f"Successfully consolidated {len(lessons_records)} lessons",
                "data": {
                    "name": new_name,
                    "source_lessons": lesson_ids,
                    "confidence": calculated_confidence,
                    "observation_types": list(consolidated_observations.keys()),
                    "relationships_migrated": migrated_relationships
                }
            })
            
        except Exception as e:
            error_info = extract_error(e)
            self.logger.error(f"Error consolidating lessons: {error_info}")
            return dict_to_json({
                "status": "error",
                "message": f"Failed to consolidate lessons: {error_info}"
            })

    def get_knowledge_evolution(
        self,
        entity_name: Optional[str] = None,
        lesson_type: Optional[str] = None,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
        include_superseded: bool = True
    ) -> str:
        """
        Track how knowledge has evolved over time.
        
        Args:
            entity_name: Entity name to filter lessons by (can be partial)
            lesson_type: Type of lessons to include
            start_date: Start date for timeline (ISO format YYYY-MM-DD)
            end_date: End date for timeline (ISO format YYYY-MM-DD)
            include_superseded: Whether to include superseded lesson versions
            
        Returns:
            JSON string with timeline of lesson evolution
        """
        try:
            self._ensure_initialized()
            
            if not self.neo4j_driver:
                return dict_to_json({
                    "status": "error",
                    "message": "Neo4j driver not initialized"
                })
            
            # Build the query based on provided filters
            query = """
            MATCH (lc:MemoryContainer {name: 'Lessons', type: 'LessonsContainer'})-[:CONTAINS]->(l:Lesson)
            """
            
            # Initialize parameters
            params = {}
            
            # Add entity name filter if provided
            if entity_name:
                query += """
                WHERE (l.name CONTAINS $entity_name 
                       OR l.context CONTAINS $entity_name
                       OR EXISTS((l)-[:ORIGINATED_FROM]->(:Entity {name: $entity_name}))
                       OR EXISTS((l)-[:APPLIES_TO]->(:Entity {name: $entity_name})))
                """
                params["entity_name"] = entity_name
            
            # Add lesson type filter if provided
            if lesson_type:
                if entity_name:  # If we already have a WHERE clause
                    query += " AND "
                else:
                    query += " WHERE "
                
                query += "(l.entityType = $lesson_type OR l.type = $lesson_type)"
                params["lesson_type"] = lesson_type
            
            # Add date range filters if provided
            date_filters = []
            
            if start_date:
                date_filters.append("l.created >= datetime($start_date)")
                params["start_date"] = start_date
                
            if end_date:
                date_filters.append("l.created <= datetime($end_date)")
                params["end_date"] = end_date
            
            if date_filters:
                if entity_name or lesson_type:  # If we already have a WHERE clause
                    query += " AND "
                else:
                    query += " WHERE "
                
                query += "(" + " AND ".join(date_filters) + ")"
            
            # Exclude superseded if requested
            if not include_superseded:
                if entity_name or lesson_type or date_filters:  # If we already have a WHERE clause
                    query += " AND "
                else:
                    query += " WHERE "
                
                query += "(l.status = 'Active' OR l.status IS NULL)"
            
            # Add versioning relationships to capture evolution
            query += """
            OPTIONAL MATCH version_path = (l)-[:SUPERSEDES*]->(older:Lesson)
            OPTIONAL MATCH application_path = (l)-[applied:APPLIED_TO]->(context:Entity)
            OPTIONAL MATCH consolidation_path = (master:Lesson)-[:CONSOLIDATES]->(l)
            
            WITH l,
                 collect(older) as older_versions,
                 collect(DISTINCT {context: context.name, date: applied.last_applied, count: applied.application_count}) as applications,
                 collect(DISTINCT master.name) as consolidated_into
            
            // Calculate evolution metrics
            WITH l,
                 older_versions,
                 applications,
                 consolidated_into,
                 size(older_versions) as version_count,
                 CASE WHEN l.confidence IS NOT NULL THEN l.confidence ELSE 0.0 END as current_confidence
                 
            RETURN l.name as name,
                   l.created as created,
                   l.lastRefreshed as last_refreshed,
                   l.status as status,
                   l.version as version,
                   current_confidence as confidence,
                   CASE WHEN l.relevanceScore IS NOT NULL THEN l.relevanceScore ELSE 0.0 END as relevance,
                   applications,
                   version_count,
                   [v in older_versions | {name: v.name, created: v.created, version: v.version, confidence: v.confidence}] as version_history,
                   consolidated_into,
                   l.context as context
            ORDER BY l.created
            """
            
            # Execute the query
            result = self._safe_execute_query(
                query,
                parameters_=params,
                database_=self.neo4j_database
            )
            
            # Process results
            records = result[0] if result and len(result) > 0 else []
            
            if not records:
                return dict_to_json({
                    "status": "success",
                    "message": "No knowledge evolution data found for the given parameters",
                    "data": {
                        "timeline": []
                    }
                })
            
            # Process records to build the evolution timeline
            timeline = []
            
            for record in records:
                # Convert Neo4j data to Python dictionary
                lesson_data = {}
                
                for key in record.keys():
                    value = record[key]
                    
                    # Handle datetime objects
                    if hasattr(value, 'iso_format'):
                        lesson_data[key] = value.iso_format()
                    # Handle non-serializable objects
                    elif not isinstance(value, (dict, list, str, int, float, bool, type(None))):
                        lesson_data[key] = str(value)
                    else:
                        lesson_data[key] = value
                
                # Calculate evolution metrics
                if "version_history" in lesson_data and lesson_data["version_history"]:
                    # Convert datetimes in version history
                    for version in lesson_data["version_history"]:
                        if "created" in version and hasattr(version["created"], 'iso_format'):
                            version["created"] = version["created"].iso_format()
                    
                    # Calculate confidence evolution
                    if "confidence" in lesson_data and lesson_data["version_history"][0].get("confidence") is not None:
                        initial_confidence = float(lesson_data["version_history"][0].get("confidence", 0))
                        current_confidence = float(lesson_data.get("confidence", 0))
                        lesson_data["confidence_evolution"] = current_confidence - initial_confidence
                
                # Filter out empty applications
                if "applications" in lesson_data:
                    lesson_data["applications"] = [app for app in lesson_data["applications"] 
                                                 if app.get("context") is not None]
                    
                    # Convert datetimes in applications
                    for app in lesson_data["applications"]:
                        if "date" in app and hasattr(app["date"], 'iso_format'):
                            app["date"] = app["date"].iso_format()
                
                timeline.append(lesson_data)
            
            # Add timeline events for key changes
            events = []
            
            for lesson in timeline:
                # Add creation event
                events.append({
                    "type": "creation",
                    "date": lesson.get("created"),
                    "lesson": lesson.get("name"),
                    "version": lesson.get("version", 1),
                    "confidence": lesson.get("confidence")
                })
                
                # Add version events
                for version in lesson.get("version_history", []):
                    events.append({
                        "type": "version_update",
                        "date": version.get("created"),
                        "lesson": version.get("name"),
                        "version": version.get("version", 1),
                        "confidence": version.get("confidence")
                    })
                
                # Add application events
                for app in lesson.get("applications", []):
                    events.append({
                        "type": "application",
                        "date": app.get("date"),
                        "lesson": lesson.get("name"),
                        "context": app.get("context"),
                        "application_count": app.get("count")
                    })
                
                # Add consolidation events
                for master in lesson.get("consolidated_into", []):
                    events.append({
                        "type": "consolidation",
                        "date": lesson.get("last_refreshed"),  # Approximation
                        "from_lesson": lesson.get("name"),
                        "to_lesson": master
                    })
            
            # Sort events by date
            events.sort(key=lambda x: x.get("date", ""))
            
            self.logger.info(f"Retrieved knowledge evolution timeline with {len(timeline)} lessons and {len(events)} events")
            
            return dict_to_json({
                "status": "success",
                "message": f"Retrieved knowledge evolution timeline with {len(timeline)} lessons",
                "data": {
                    "lessons": timeline,
                    "events": events
                }
            })
            
        except Exception as e:
            error_info = extract_error(e)
            self.logger.error(f"Error retrieving knowledge evolution: {error_info}")
            return dict_to_json({
                "status": "error",
                "message": f"Failed to retrieve knowledge evolution: {error_info}"
            })

    def query_across_contexts(
        self,
        query_text: str,
        containers: Optional[List[str]] = None,
        confidence_threshold: float = 0.0,
        relevance_threshold: float = 0.0,
        limit_per_container: int = 10
    ) -> str:
        """
        Execute unified search queries across different memory containers.
        
        Args:
            query_text: Search query text
            containers: List of container names to include (all if None)
            confidence_threshold: Minimum confidence level for returned results
            relevance_threshold: Minimum relevance score for returned results
            limit_per_container: Maximum results to return per container
            
        Returns:
            JSON string with unified results across containers
        """
        try:
            self._ensure_initialized()
            
            if not self.neo4j_driver:
                return dict_to_json({
                    "status": "error",
                    "message": "Neo4j driver not initialized"
                })
            
            # If no containers specified, get all available containers
            if not containers:
                containers_query = """
                MATCH (c:MemoryContainer)
                RETURN c.name as name
                """
                
                containers_result = self.neo4j_driver.execute_query(
                    containers_query,
                    database_=self.neo4j_database
                )
                
                containers = []
                if containers_result and containers_result[0]:
                    for record in containers_result[0]:
                        container_name = record.get("name")
                        if container_name:
                            containers.append(container_name)
                
                if not containers:
                    containers = ["Lessons"]  # Default to Lessons container if none found
            
            # Build the cross-container query
            query = """
            // Find entities in specified containers
            MATCH (container:MemoryContainer)-[:CONTAINS]->(entity)
            WHERE container.name IN $containers
            
            // Apply confidence threshold
            AND (entity.confidence >= $confidence_threshold OR entity.confidence IS NULL)
            
            // Apply relevance threshold if available
            AND (entity.relevanceScore >= $relevance_threshold OR entity.relevanceScore IS NULL)
            
            // Text matching in entity properties and observations
            AND (
                entity.name CONTAINS $query_text
                OR entity.problemDescription CONTAINS $query_text
                OR entity.resolution CONTAINS $query_text
                OR entity.context CONTAINS $query_text
                OR EXISTS(
                    (entity)-[:HAS_OBSERVATION]->(o:Observation)
                    WHERE o.content CONTAINS $query_text
                )
            )
            
            // Get observations for each matching entity
            OPTIONAL MATCH (entity)-[:HAS_OBSERVATION]->(obs:Observation)
            
            // Collect results by container
            WITH container.name as container_name,
                 entity,
                 collect(obs) as observations
                 
            // Calculate relevance score - prefer explicit scores but provide defaults
            WITH container_name,
                 entity,
                 observations,
                 CASE
                     WHEN entity.relevanceScore IS NOT NULL THEN entity.relevanceScore
                     WHEN entity.confidence IS NOT NULL THEN entity.confidence
                     ELSE 0.5
                 END as calculated_relevance
            
            // Return results with relevance ranking
            RETURN container_name,
                   entity.name as entity_name,
                   entity.entityType as entity_type,
                   calculated_relevance as relevance,
                   CASE WHEN entity.confidence IS NOT NULL THEN entity.confidence ELSE 0.5 END as confidence,
                   [o IN observations | {type: o.type, content: o.content}] as observations,
                   entity.created as created,
                   entity.lastRefreshed as last_refreshed
            
            // Order by relevance and limit results
            ORDER BY calculated_relevance DESC, entity.lastRefreshed DESC
            LIMIT $total_limit
            """
            
            # Set up parameters
            params = {
                "containers": containers,
                "query_text": query_text,
                "confidence_threshold": confidence_threshold,
                "relevance_threshold": relevance_threshold,
                "total_limit": limit_per_container * len(containers)
            }
            
            # Execute the query
            result = self._safe_execute_query(
                query,
                parameters_=params,
                database_=self.neo4j_database
            )
            
            # Process results
            records = result[0] if result and len(result) > 0 else []
            
            if not records:
                return dict_to_json({
                    "status": "success",
                    "message": "No results found across the specified containers",
                    "data": {
                        "containers": containers,
                        "results": []
                    }
                })
            
            # Organize results by container
            container_results = {}
            
            for record in records:
                container_name = record.get("container_name")
                
                if container_name not in container_results:
                    container_results[container_name] = []
                
                # Convert Neo4j data to Python dictionary
                result_data = {}
                
                for key in record.keys():
                    if key == "container_name":
                        continue
                        
                    value = record[key]
                    
                    # Handle datetime objects
                    if hasattr(value, 'iso_format'):
                        result_data[key] = value.iso_format()
                    # Handle non-serializable objects
                    elif not isinstance(value, (dict, list, str, int, float, bool, type(None))):
                        result_data[key] = str(value)
                    else:
                        result_data[key] = value
                
                container_results[container_name].append(result_data)
                
            # Limit results per container
            for container, results in container_results.items():
                container_results[container] = results[:limit_per_container]
            
            # Count total results
            total_results = sum(len(results) for results in container_results.values())
            
            self.logger.info(f"Cross-container query found {total_results} results across {len(container_results)} containers")
            
            return dict_to_json({
                "status": "success",
                "message": f"Found {total_results} results across {len(container_results)} containers",
                "data": {
                    "query": query_text,
                    "containers": containers,
                    "results_by_container": container_results,
                    "total_results": total_results
                }
            })
            
        except Exception as e:
            error_info = extract_error(e)
            self.logger.error(f"Error in cross-container query: {error_info}")
            return dict_to_json({
                "status": "error",
                "message": f"Failed to execute cross-container query: {error_info}"
            })

    def _safe_execute_query(self, query: str, **kwargs):
        """
        Helper method to safely execute Neo4j queries with the correct type handling.
        
        Args:
            query: The query string to execute
            **kwargs: Additional parameters to pass to execute_query
            
        Returns:
            Query result from Neo4j driver, processed to be more consistent for record access
        """
        if not self.neo4j_driver:
            self.logger.error("Neo4j driver not initialized")
            return None
            
        # Cast the query string to LiteralString to satisfy Neo4j driver's typing requirements
        typed_query = cast(LiteralString, query)
        
        # Handle float values in parameters to prevent type errors
        if 'parameters_' in kwargs and kwargs['parameters_']:
            params = kwargs['parameters_']
            for key, value in params.items():
                if isinstance(value, float):
                    params[key] = str(value)
        
        try:
            result = self.neo4j_driver.execute_query(typed_query, **kwargs)
            
            # Process the Neo4j result to make it more consistent and accessible
            # Neo4j driver returns a tuple with (records, summary, keys)
            if result and len(result) > 0:
                records = result[0]  # Get the records part
                
                # Return the records directly - they have both index access and attribute access
                return records
            else:
                return []
                
        except Exception as e:
            self.logger.error(f"Error executing Neo4j query: {str(e)}")
            return []
    
    # Project Memory System Implementation
    
    def create_project_container(
        self,
        project_name: str,
        description: Optional[str] = None,
        start_date: Optional[str] = None,
        team: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
        confidence: float = 0.9,
        tags: Optional[List[str]] = None
    ) -> str:
        """
        Create a project container for organizing project knowledge.
        
        Args:
            project_name: Name of the project
            description: Optional description of the project
            start_date: Optional start date in ISO format
            team: Optional team name or description
            metadata: Optional additional metadata as key-value pairs
            confidence: Confidence level for this container (0.0-1.0)
            tags: Optional list of tags for categorization
            
        Returns:
            JSON string with result information
        """
        try:
            self._ensure_initialized()
            
            if not self.neo4j_driver:
                return dict_to_json({
                    "status": "error",
                    "message": "Neo4j driver not initialized"
                })
            
            # Create project container
            container_query = """
            MERGE (pc:MemoryContainer {name: $project_name, type: "ProjectContainer"})
            ON CREATE SET pc.created = datetime(),
                          pc.description = $description,
                          pc.startDate = $start_date,
                          pc.team = $team,
                          pc.confidence = $confidence,
                          pc.tags = $tags,
                          pc.lastUpdated = datetime()
            RETURN pc
            """
            
            container_params = {
                "project_name": project_name,
                "description": description or f"Knowledge container for {project_name}",
                "start_date": start_date,
                "team": team,
                "confidence": confidence,
                "tags": tags
            }
            
            self._safe_execute_query(
                container_query,
                parameters_=container_params,
                database_=self.neo4j_database
            )
            
            # Create project entity
            project_query = """
            MERGE (p:Entity {name: $project_name, entityType: "Project"})
            ON CREATE SET p.created = datetime(),
                          p.description = $description,
                          p.startDate = $start_date,
                          p.team = $team,
                          p.confidence = $confidence,
                          p.status = "Active",
                          p.lastUpdated = datetime()
            RETURN p
            """
            
            project_params = {
                "project_name": project_name,
                "description": description or f"Project: {project_name}",
                "start_date": start_date,
                "team": team,
                "confidence": confidence
            }
            
            self._safe_execute_query(
                project_query,
                parameters_=project_params,
                database_=self.neo4j_database
            )
            
            # Link project to container
            link_query = """
            MATCH (pc:MemoryContainer {name: $project_name, type: "ProjectContainer"})
            MATCH (p:Entity {name: $project_name, entityType: "Project"})
            MERGE (pc)-[r:CONTAINS]->(p)
            ON CREATE SET r.since = datetime()
            RETURN p, pc
            """
            
            self._safe_execute_query(
                link_query,
                project_name=project_name,
                database_=self.neo4j_database
            )
            
            # Add metadata if provided
            if metadata and isinstance(metadata, dict):
                self._add_metadata_to_project(project_name, metadata)
            
            self.logger.info(f"Created project container and entity for: {project_name}")
            
            return dict_to_json({
                "status": "success",
                "message": f"Project container '{project_name}' created successfully",
                "data": {
                    "project_name": project_name,
                    "created": datetime.datetime.now().isoformat()
                }
            })
            
        except Exception as e:
            error_info = extract_error(e)
            self.logger.error(f"Error creating project container: {error_info}")
            return dict_to_json({
                "status": "error",
                "message": f"Failed to create project container: {error_info}"
            })
    
    def _add_metadata_to_project(self, project_name: str, metadata: Dict[str, Any]) -> None:
        """
        Add metadata to a project as properties.
        
        Args:
            project_name: Name of the project
            metadata: Dictionary of metadata to add
        """
        try:
            # Filter out None values and convert to properties
            filtered_metadata = {k: v for k, v in metadata.items() if v is not None}
            
            if not filtered_metadata:
                return
                
            # Build dynamic property setting
            props = []
            params = {"project_name": project_name}
            
            for i, (key, value) in enumerate(filtered_metadata.items()):
                param_name = f"meta_{i}"
                props.append(f"p.{key} = ${param_name}")
                params[param_name] = value
            
            if not props:
                return
                
            # Update project entity with metadata
            metadata_query = f"""
            MATCH (p:Entity {{name: $project_name, entityType: "Project"}})
            SET {', '.join(props)},
                p.lastUpdated = datetime()
            RETURN p
            """
            
            self._safe_execute_query(
                metadata_query,
                parameters_=params,
                database_=self.neo4j_database
            )
            
        except Exception as e:
            self.logger.error(f"Error adding metadata to project: {str(e)}")
            
    def add_domain(
        self,
        project_name: str,
        domain_name: str,
        description: Optional[str] = None,
        purpose: Optional[str] = None,
        confidence: float = 0.9,
        status: str = "Active",
        metadata: Optional[Dict[str, Any]] = None
    ) -> str:
        """
        Add a domain (logical subdivision) to a project.
        
        Args:
            project_name: Name of the project
            domain_name: Name of the domain to add
            description: Optional description of the domain
            purpose: Optional purpose or function of this domain
            confidence: Confidence level for this domain (0.0-1.0)
            status: Status of the domain (Active, Planned, Deprecated)
            metadata: Optional additional metadata as key-value pairs
            
        Returns:
            JSON string with result information
        """
        try:
            self._ensure_initialized()
            
            if not self.neo4j_driver:
                return dict_to_json({
                    "status": "error",
                    "message": "Neo4j driver not initialized"
                })
            
            # Create domain entity
            domain_query = """
            MERGE (d:Entity {name: $domain_name, entityType: "Domain"})
            ON CREATE SET d.created = datetime(),
                          d.description = $description,
                          d.purpose = $purpose,
                          d.confidence = $confidence,
                          d.status = $status,
                          d.lastUpdated = datetime()
            RETURN d
            """
            
            domain_params = {
                "domain_name": domain_name,
                "description": description or f"Domain: {domain_name}",
                "purpose": purpose,
                "confidence": confidence,
                "status": status
            }
            
            self._safe_execute_query(
                domain_query,
                parameters_=domain_params,
                database_=self.neo4j_database
            )
            
            # Link to project entity
            project_link_query = """
            MATCH (p:Entity {name: $project_name, entityType: "Project"})
            MATCH (d:Entity {name: $domain_name, entityType: "Domain"})
            MERGE (p)-[r:CONTAINS]->(d)
            ON CREATE SET r.since = datetime(),
                          r.confidence = $confidence
            RETURN p, d
            """
            
            project_link_params = {
                "project_name": project_name,
                "domain_name": domain_name,
                "confidence": confidence
            }
            
            project_result = self._safe_execute_query(
                project_link_query,
                parameters_=project_link_params,
                database_=self.neo4j_database
            )
            
            # Check if the project exists
            if not project_result or not project_result[0]:
                self.logger.warn(f"Project '{project_name}' not found when adding domain")
                return dict_to_json({
                    "status": "error",
                    "message": f"Project '{project_name}' not found. Please create the project first."
                })
            
            # Link to project container
            container_link_query = """
            MATCH (pc:MemoryContainer {name: $project_name, type: "ProjectContainer"})
            MATCH (d:Entity {name: $domain_name, entityType: "Domain"})
            MERGE (pc)-[r:CONTAINS]->(d)
            ON CREATE SET r.since = datetime(),
                          r.confidence = $confidence
            RETURN pc, d
            """
            
            container_link_params = {
                "project_name": project_name,
                "domain_name": domain_name,
                "confidence": confidence
            }
            
            container_result = self._safe_execute_query(
                container_link_query,
                parameters_=container_link_params,
                database_=self.neo4j_database
            )
            
            # Check if the project container exists
            if not container_result or not container_result[0]:
                self.logger.warn(f"Project container '{project_name}' not found when adding domain")
                
                # Create project container if it doesn't exist
                container_creation = self.create_project_container(project_name)
                container_data = json.loads(container_creation)
                
                if container_data.get("status") != "success":
                    self.logger.error(f"Failed to create missing project container: {container_data.get('message')}")
                    return dict_to_json({
                        "status": "warning",
                        "message": f"Domain created but project container could not be created: {container_data.get('message')}",
                        "data": {
                            "domain_name": domain_name,
                            "project_name": project_name
                        }
                    })
                    
                # Try linking again
                self._safe_execute_query(
                    container_link_query,
                    parameters_=container_link_params,
                    database_=self.neo4j_database
                )
            
            # Add metadata if provided
            if metadata and isinstance(metadata, dict):
                self._add_metadata_to_domain(domain_name, metadata)
            
            self.logger.info(f"Added domain '{domain_name}' to project '{project_name}'")
            
            return dict_to_json({
                "status": "success",
                "message": f"Domain '{domain_name}' added to project '{project_name}' successfully",
                "data": {
                    "project_name": project_name,
                    "domain_name": domain_name,
                    "created": datetime.datetime.now().isoformat()
                }
            })
            
        except Exception as e:
            error_info = extract_error(e)
            self.logger.error(f"Error adding domain to project: {error_info}")
            return dict_to_json({
                "status": "error",
                "message": f"Failed to add domain to project: {error_info}"
            })
    
    def _add_metadata_to_domain(self, domain_name: str, metadata: Dict[str, Any]) -> None:
        """
        Add metadata to a domain as properties.
        
        Args:
            domain_name: Name of the domain
            metadata: Dictionary of metadata to add
        """
        try:
            # Filter out None values and convert to properties
            filtered_metadata = {k: v for k, v in metadata.items() if v is not None}
            
            if not filtered_metadata:
                return
                
            # Build dynamic property setting
            props = []
            params = {"domain_name": domain_name}
            
            for i, (key, value) in enumerate(filtered_metadata.items()):
                param_name = f"meta_{i}"
                props.append(f"d.{key} = ${param_name}")
                params[param_name] = value
            
            if not props:
                return
                
            # Update domain entity with metadata
            metadata_query = f"""
            MATCH (d:Entity {{name: $domain_name, entityType: "Domain"}})
            SET {', '.join(props)},
                d.lastUpdated = datetime()
            RETURN d
            """
            
            self._safe_execute_query(
                metadata_query,
                parameters_=params,
                database_=self.neo4j_database
            )
            
        except Exception as e:
            self.logger.error(f"Error adding metadata to domain: {str(e)}")
    
    def add_component(
        self,
        project_name: str,
        component_name: str,
        domain_name: Optional[str] = None,
        component_type: Optional[str] = None,
        responsibility: Optional[str] = None,
        status: str = "Planned",
        confidence: float = 0.8,
        maturity: str = "Emerging",
        dependencies: Optional[List[Dict[str, Any]]] = None,
        evolution_history: Optional[List[Dict[str, Any]]] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> str:
        """
        Add a technical component to a project or domain.
        
        Args:
            project_name: Name of the project
            component_name: Name of the component to add
            domain_name: Optional domain to which the component belongs
            component_type: Optional type of component (e.g., "Frontend", "Service", "Database")
            responsibility: Optional description of what the component does
            status: Status of the component (Planned, InDevelopment, Active, Deprecated)
            confidence: Confidence level for this component (0.0-1.0)
            maturity: Maturity level (Emerging, Stable, Mature)
            dependencies: Optional list of component dependencies with format 
                          [{"name": "ComponentName", "criticality": "High", "type": "Requires"}]
            evolution_history: Optional list of predecessor components with format
                              [{"predecessor_name": "OldComponent", "date": "2023-01-01", 
                                "change_type": "Refactor", "significance": "Major"}]
            metadata: Optional additional metadata as key-value pairs
            
        Returns:
            JSON string with result information
        """
        try:
            self._ensure_initialized()
            
            if not self.neo4j_driver:
                return dict_to_json({
                    "status": "error",
                    "message": "Neo4j driver not initialized"
                })
            
            # Create component entity
            component_query = """
            MERGE (c:Entity {name: $component_name, entityType: "Component"})
            ON CREATE SET c.created = datetime(),
                          c.type = $component_type,
                          c.responsibility = $responsibility,
                          c.status = $status,
                          c.confidence = $confidence,
                          c.maturity = $maturity,
                          c.lastUpdated = datetime()
            RETURN c
            """
            
            component_params = {
                "component_name": component_name,
                "component_type": component_type,
                "responsibility": responsibility or f"Component: {component_name}",
                "status": status,
                "confidence": confidence,
                "maturity": maturity
            }
            
            self._safe_execute_query(
                component_query,
                parameters_=component_params,
                database_=self.neo4j_database
            )
            
            # Link to domain if provided
            if domain_name:
                domain_link_query = """
                MATCH (d:Entity {name: $domain_name, entityType: "Domain"})
                MATCH (c:Entity {name: $component_name, entityType: "Component"})
                MERGE (d)-[r:CONTAINS]->(c)
                ON CREATE SET r.since = datetime(),
                              r.confidence = $confidence
                RETURN d, c
                """
                
                domain_link_params = {
                    "domain_name": domain_name,
                    "component_name": component_name,
                    "confidence": confidence
                }
                
                domain_result = self._safe_execute_query(
                    domain_link_query,
                    parameters_=domain_link_params,
                    database_=self.neo4j_database
                )
                
                # Check if the domain exists
                if not domain_result or not domain_result[0]:
                    self.logger.warn(f"Domain '{domain_name}' not found when adding component")
                    return dict_to_json({
                        "status": "error",
                        "message": f"Domain '{domain_name}' not found. Please create the domain first."
                    })
            
            # Link to project container
            container_link_query = """
            MATCH (pc:MemoryContainer {name: $project_name, type: "ProjectContainer"})
            MATCH (c:Entity {name: $component_name, entityType: "Component"})
            MERGE (pc)-[r:CONTAINS]->(c)
            ON CREATE SET r.since = datetime(),
                          r.confidence = $confidence
            RETURN pc, c
            """
            
            container_link_params = {
                "project_name": project_name,
                "component_name": component_name,
                "confidence": confidence
            }
            
            container_result = self._safe_execute_query(
                container_link_query,
                parameters_=container_link_params,
                database_=self.neo4j_database
            )
            
            # Check if the project container exists
            if not container_result or not container_result[0]:
                self.logger.warn(f"Project container '{project_name}' not found when adding component")
                
                # Create project container if it doesn't exist
                container_creation = self.create_project_container(project_name)
                container_data = json.loads(container_creation)
                
                if container_data.get("status") != "success":
                    self.logger.error(f"Failed to create missing project container: {container_data.get('message')}")
                    return dict_to_json({
                        "status": "warning",
                        "message": f"Component created but project container could not be created: {container_data.get('message')}",
                        "data": {
                            "component_name": component_name,
                            "project_name": project_name
                        }
                    })
                    
                # Try linking again
                self._safe_execute_query(
                    container_link_query,
                    parameters_=container_link_params,
                    database_=self.neo4j_database
                )
            
            # Create dependencies if provided
            if dependencies and isinstance(dependencies, list):
                self._create_component_dependencies(component_name, dependencies)
            
            # Record evolution history if provided
            if evolution_history and isinstance(evolution_history, list):
                self._record_component_evolution(component_name, evolution_history)
            
            # Add metadata if provided
            if metadata and isinstance(metadata, dict):
                self._add_metadata_to_component(component_name, metadata)
            
            self.logger.info(f"Added component '{component_name}' to project '{project_name}'")
            
            result_data = {
                "project_name": project_name,
                "component_name": component_name,
                "created": datetime.datetime.now().isoformat()
            }
            
            if domain_name:
                result_data["domain_name"] = domain_name
            
            return dict_to_json({
                "status": "success",
                "message": f"Component '{component_name}' added successfully",
                "data": result_data
            })
            
        except Exception as e:
            error_info = extract_error(e)
            self.logger.error(f"Error adding component to project: {error_info}")
            return dict_to_json({
                "status": "error",
                "message": f"Failed to add component to project: {error_info}"
            })
    
    def _create_component_dependencies(self, component_name: str, dependencies: List[Dict[str, Any]]) -> None:
        """
        Create dependency relationships between components.
        
        Args:
            component_name: Name of the component
            dependencies: List of dependencies with format:
                         [{"name": "ComponentName", "criticality": "High", "type": "Requires"}]
        """
        try:
            for dependency in dependencies:
                dep_name = dependency.get("name")
                
                if not dep_name:
                    self.logger.warn("Skipping dependency with missing name")
                    continue
                
                # Get dependency properties
                criticality = dependency.get("criticality", "Medium")
                dep_type = dependency.get("type", "Requires")
                confidence = dependency.get("confidence", 0.8)
                
                # Create dependency relationship
                dependency_query = """
                MATCH (c:Entity {name: $component_name, entityType: "Component"})
                MATCH (dep:Entity {name: $dep_name, entityType: "Component"})
                MERGE (c)-[r:DEPENDS_ON]->(dep)
                ON CREATE SET r.created = datetime(),
                              r.criticality = $criticality,
                              r.type = $dep_type,
                              r.confidence = $confidence
                ON MATCH SET r.lastUpdated = datetime(),
                             r.criticality = $criticality,
                             r.type = $dep_type,
                             r.confidence = $confidence
                RETURN c, r, dep
                """
                
                dependency_params = {
                    "component_name": component_name,
                    "dep_name": dep_name,
                    "criticality": criticality,
                    "dep_type": dep_type,
                    "confidence": confidence
                }
                
                result = self._safe_execute_query(
                    dependency_query,
                    parameters_=dependency_params,
                    database_=self.neo4j_database
                )
                
                if not result or not result[0]:
                    self.logger.warn(f"Could not create dependency to '{dep_name}' - component may not exist")
                else:
                    self.logger.debug(f"Created dependency from '{component_name}' to '{dep_name}'")
                
        except Exception as e:
            self.logger.error(f"Error creating component dependencies: {str(e)}")
    
    def _record_component_evolution(self, component_name: str, evolution_history: List[Dict[str, Any]]) -> None:
        """
        Record the evolution history of a component.
        
        Args:
            component_name: Name of the component
            evolution_history: List of predecessor components with format:
                              [{"predecessor_name": "OldComponent", "date": "2023-01-01", 
                                "change_type": "Refactor", "significance": "Major"}]
        """
        try:
            for history_item in evolution_history:
                predecessor_name = history_item.get("predecessor_name")
                
                if not predecessor_name:
                    self.logger.warn("Skipping evolution record with missing predecessor name")
                    continue
                
                # Get evolution properties
                date_str = history_item.get("date")
                change_type = history_item.get("change_type", "Refactor")
                significance = history_item.get("significance", "Medium")
                confidence = history_item.get("confidence", 0.8)
                
                # Validate and format date
                date_param = None
                if date_str:
                    try:
                        # Convert to Neo4j datetime format
                        date_param = datetime.datetime.fromisoformat(date_str).isoformat()
                    except ValueError:
                        self.logger.warn(f"Invalid date format: {date_str}, using current date")
                        date_param = datetime.datetime.now().isoformat()
                else:
                    date_param = datetime.datetime.now().isoformat()
                
                # Create evolution relationship
                evolution_query = """
                MATCH (c:Entity {name: $component_name, entityType: "Component"})
                MATCH (pred:Entity {name: $predecessor_name})
                MERGE (c)-[r:EVOLVED_FROM]->(pred)
                ON CREATE SET r.date = datetime($date),
                              r.changeType = $change_type,
                              r.significance = $significance,
                              r.confidence = $confidence
                ON MATCH SET r.lastUpdated = datetime(),
                             r.changeType = $change_type,
                             r.significance = $significance,
                             r.confidence = $confidence
                RETURN c, r, pred
                """
                
                evolution_params = {
                    "component_name": component_name,
                    "predecessor_name": predecessor_name,
                    "date": date_param,
                    "change_type": change_type,
                    "significance": significance,
                    "confidence": confidence
                }
                
                result = self._safe_execute_query(
                    evolution_query,
                    parameters_=evolution_params,
                    database_=self.neo4j_database
                )
                
                if not result or not result[0]:
                    self.logger.warn(f"Could not create evolution link to '{predecessor_name}' - entity may not exist")
                else:
                    self.logger.debug(f"Recorded evolution from '{predecessor_name}' to '{component_name}'")
                
        except Exception as e:
            self.logger.error(f"Error recording component evolution: {str(e)}")
    
    def _add_metadata_to_component(self, component_name: str, metadata: Dict[str, Any]) -> None:
        """
        Add metadata to a component as properties.
        
        Args:
            component_name: Name of the component
            metadata: Dictionary of metadata to add
        """
        try:
            # Filter out None values and convert to properties
            filtered_metadata = {k: v for k, v in metadata.items() if v is not None}
            
            if not filtered_metadata:
                return
                
            # Build dynamic property setting
            props = []
            params = {"component_name": component_name}
            
            for i, (key, value) in enumerate(filtered_metadata.items()):
                param_name = f"meta_{i}"
                props.append(f"c.{key} = ${param_name}")
                params[param_name] = value
            
            if not props:
                return
                
            # Update component entity with metadata
            metadata_query = f"""
            MATCH (c:Entity {{name: $component_name, entityType: "Component"}})
            SET {', '.join(props)},
                c.lastUpdated = datetime()
            RETURN c
            """
            
            self._safe_execute_query(
                metadata_query,
                parameters_=params,
                database_=self.neo4j_database
            )
            
        except Exception as e:
            self.logger.error(f"Error adding metadata to component: {str(e)}")
    
    def add_feature(
        self,
        project_name: str,
        feature_name: str,
        domain_name: Optional[str] = None,
        description: Optional[str] = None,
        priority: str = "Medium",
        status: str = "Planned",
        requirements: Optional[List[Dict[str, Any]]] = None,
        confidence: float = 0.8,
        complexity: str = "Medium",
        implementing_components: Optional[List[str]] = None,
        metrics: Optional[Dict[str, Any]] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> str:
        """
        Add a feature to a project or domain.
        
        Args:
            project_name: Name of the project
            feature_name: Name of the feature to add
            domain_name: Optional domain to which the feature belongs
            description: Optional description of the feature
            priority: Priority level (Low, Medium, High, Critical)
            status: Status of the feature (Planned, InDevelopment, Completed, Deprecated)
            requirements: Optional list of requirements with format:
                         [{"description": "Req description", "priority": "High", "source": "Customer"}]
            confidence: Confidence level for this feature (0.0-1.0)
            complexity: Complexity level (Low, Medium, High, Very High)
            implementing_components: Optional list of component names that implement this feature
            metrics: Optional metrics to track for this feature as key-value pairs
            metadata: Optional additional metadata as key-value pairs
            
        Returns:
            JSON string with result information
        """
        try:
            self._ensure_initialized()
            
            if not self.neo4j_driver:
                return dict_to_json({
                    "status": "error",
                    "message": "Neo4j driver not initialized"
                })
            
            # Create feature entity
            feature_query = """
            MERGE (f:Entity {name: $feature_name, entityType: "Feature"})
            ON CREATE SET f.created = datetime(),
                          f.description = $description,
                          f.priority = $priority,
                          f.status = $status,
                          f.confidence = $confidence,
                          f.complexity = $complexity,
                          f.lastUpdated = datetime()
            RETURN f
            """
            
            feature_params = {
                "feature_name": feature_name,
                "description": description or f"Feature: {feature_name}",
                "priority": priority,
                "status": status,
                "confidence": confidence,
                "complexity": complexity
            }
            
            self._safe_execute_query(
                feature_query,
                parameters_=feature_params,
                database_=self.neo4j_database
            )
            
            # Link to domain if provided
            if domain_name:
                domain_link_query = """
                MATCH (d:Entity {name: $domain_name, entityType: "Domain"})
                MATCH (f:Entity {name: $feature_name, entityType: "Feature"})
                MERGE (d)-[r:CONTAINS]->(f)
                ON CREATE SET r.since = datetime(),
                              r.confidence = $confidence
                RETURN d, f
                """
                
                domain_link_params = {
                    "domain_name": domain_name,
                    "feature_name": feature_name,
                    "confidence": confidence
                }
                
                domain_result = self._safe_execute_query(
                    domain_link_query,
                    parameters_=domain_link_params,
                    database_=self.neo4j_database
                )
                
                # Check if the domain exists
                if not domain_result or not domain_result[0]:
                    self.logger.warn(f"Domain '{domain_name}' not found when adding feature")
                    return dict_to_json({
                        "status": "error",
                        "message": f"Domain '{domain_name}' not found. Please create the domain first."
                    })
            
            # Link to project container
            container_link_query = """
            MATCH (pc:MemoryContainer {name: $project_name, type: "ProjectContainer"})
            MATCH (f:Entity {name: $feature_name, entityType: "Feature"})
            MERGE (pc)-[r:CONTAINS]->(f)
            ON CREATE SET r.since = datetime(),
                          r.confidence = $confidence
            RETURN pc, f
            """
            
            container_link_params = {
                "project_name": project_name,
                "feature_name": feature_name,
                "confidence": confidence
            }
            
            container_result = self._safe_execute_query(
                container_link_query,
                parameters_=container_link_params,
                database_=self.neo4j_database
            )
            
            # Check if the project container exists
            if not container_result or not container_result[0]:
                self.logger.warn(f"Project container '{project_name}' not found when adding feature")
                
                # Create project container if it doesn't exist
                container_creation = self.create_project_container(project_name)
                container_data = json.loads(container_creation)
                
                if container_data.get("status") != "success":
                    self.logger.error(f"Failed to create missing project container: {container_data.get('message')}")
                    return dict_to_json({
                        "status": "warning",
                        "message": f"Feature created but project container could not be created: {container_data.get('message')}",
                        "data": {
                            "feature_name": feature_name,
                            "project_name": project_name
                        }
                    })
                    
                # Try linking again
                self._safe_execute_query(
                    container_link_query,
                    parameters_=container_link_params,
                    database_=self.neo4j_database
                )
            
            # Create requirements if provided
            if requirements and isinstance(requirements, list):
                self._create_feature_requirements(feature_name, requirements)
            
            # Link implementing components if provided
            if implementing_components and isinstance(implementing_components, list):
                self._link_implementing_components(feature_name, implementing_components, confidence)
            
            # Add metrics if provided
            if metrics and isinstance(metrics, dict):
                self._add_metrics_to_feature(feature_name, metrics)
            
            # Add metadata if provided
            if metadata and isinstance(metadata, dict):
                self._add_metadata_to_feature(feature_name, metadata)
            
            self.logger.info(f"Added feature '{feature_name}' to project '{project_name}'")
            
            result_data = {
                "project_name": project_name,
                "feature_name": feature_name,
                "created": datetime.datetime.now().isoformat()
            }
            
            if domain_name:
                result_data["domain_name"] = domain_name
            
            return dict_to_json({
                "status": "success",
                "message": f"Feature '{feature_name}' added successfully",
                "data": result_data
            })
            
        except Exception as e:
            error_info = extract_error(e)
            self.logger.error(f"Error adding feature to project: {error_info}")
            return dict_to_json({
                "status": "error",
                "message": f"Failed to add feature to project: {error_info}"
            })
    
    def _create_feature_requirements(self, feature_name: str, requirements: List[Dict[str, Any]]) -> None:
        """
        Create requirements for a feature.
        
        Args:
            feature_name: Name of the feature
            requirements: List of requirements with format:
                         [{"description": "Req description", "priority": "High", "source": "Customer"}]
        """
        try:
            for i, req in enumerate(requirements):
                description = req.get("description")
                
                if not description:
                    self.logger.warn("Skipping requirement with missing description")
                    continue
                
                # Create a unique name for the requirement if not provided
                req_name = req.get("name", f"{feature_name}_Req_{i+1}")
                priority = req.get("priority", "Medium")
                source = req.get("source", "Unspecified")
                status = req.get("status", "Proposed")
                confidence = req.get("confidence", 0.8)
                
                # Create requirement entity
                req_query = """
                MERGE (r:Entity {name: $req_name, entityType: "Requirement"})
                ON CREATE SET r.created = datetime(),
                              r.description = $description,
                              r.priority = $priority,
                              r.source = $source,
                              r.status = $status,
                              r.confidence = $confidence,
                              r.lastUpdated = datetime()
                RETURN r
                """
                
                req_params = {
                    "req_name": req_name,
                    "description": description,
                    "priority": priority,
                    "source": source,
                    "status": status,
                    "confidence": confidence
                }
                
                self._safe_execute_query(
                    req_query,
                    parameters_=req_params,
                    database_=self.neo4j_database
                )
                
                # Link requirement to feature
                link_query = """
                MATCH (f:Entity {name: $feature_name, entityType: "Feature"})
                MATCH (r:Entity {name: $req_name, entityType: "Requirement"})
                MERGE (f)-[rel:HAS_REQUIREMENT]->(r)
                ON CREATE SET rel.created = datetime(),
                              rel.confidence = $confidence
                RETURN f, rel, r
                """
                
                link_params = {
                    "feature_name": feature_name,
                    "req_name": req_name,
                    "confidence": confidence
                }
                
                self._safe_execute_query(
                    link_query,
                    parameters_=link_params,
                    database_=self.neo4j_database
                )
                
                self.logger.debug(f"Created requirement '{req_name}' for feature '{feature_name}'")
                
        except Exception as e:
            self.logger.error(f"Error creating feature requirements: {str(e)}")
    
    def _link_implementing_components(self, feature_name: str, component_names: List[str], confidence: float) -> None:
        """
        Link components that implement a feature.
        
        Args:
            feature_name: Name of the feature
            component_names: List of component names that implement this feature
            confidence: Confidence level for the implementation relationships
        """
        try:
            for component_name in component_names:
                # Create implementation relationship
                impl_query = """
                MATCH (c:Entity {name: $component_name, entityType: "Component"})
                MATCH (f:Entity {name: $feature_name, entityType: "Feature"})
                MERGE (c)-[r:IMPLEMENTS]->(f)
                ON CREATE SET r.created = datetime(),
                              r.confidence = $confidence,
                              r.coverage = "Full"
                ON MATCH SET r.lastUpdated = datetime(),
                             r.confidence = $confidence
                RETURN c, r, f
                """
                
                impl_params = {
                    "component_name": component_name,
                    "feature_name": feature_name,
                    "confidence": confidence
                }
                
                result = self._safe_execute_query(
                    impl_query,
                    parameters_=impl_params,
                    database_=self.neo4j_database
                )
                
                if not result or not result[0]:
                    self.logger.warn(f"Could not link component '{component_name}' as implementing '{feature_name}' - component may not exist")
                else:
                    self.logger.debug(f"Linked component '{component_name}' as implementing feature '{feature_name}'")
                
        except Exception as e:
            self.logger.error(f"Error linking implementing components: {str(e)}")
    
    def _add_metrics_to_feature(self, feature_name: str, metrics: Dict[str, Any]) -> None:
        """
        Add metrics to a feature.
        
        Args:
            feature_name: Name of the feature
            metrics: Dictionary of metrics as key-value pairs
        """
        try:
            # Create metrics nodes for the feature
            for metric_name, metric_value in metrics.items():
                if metric_name and metric_value is not None:
                    # Create a unique name for the metric
                    unique_metric_name = f"{feature_name}_{metric_name}"
                    
                    # Determine metric type
                    metric_type = "Numeric"
                    if isinstance(metric_value, str):
                        metric_type = "Text"
                    elif isinstance(metric_value, bool):
                        metric_type = "Boolean"
                    
                    # Convert value to string for storage
                    string_value = str(metric_value)
                    
                    # Create metric entity
                    metric_query = """
                    MERGE (m:Metric {name: $unique_name})
                    ON CREATE SET m.created = datetime(),
                                  m.type = $metric_type,
                                  m.metricName = $metric_name,
                                  m.value = $metric_value,
                                  m.timestamp = datetime()
                    ON MATCH SET m.value = $metric_value,
                                 m.timestamp = datetime(),
                                 m.lastUpdated = datetime()
                    RETURN m
                    """
                    
                    metric_params = {
                        "unique_name": unique_metric_name,
                        "metric_type": metric_type,
                        "metric_name": metric_name,
                        "metric_value": string_value
                    }
                    
                    self._safe_execute_query(
                        metric_query,
                        parameters_=metric_params,
                        database_=self.neo4j_database
                    )
                    
                    # Link metric to feature
                    link_query = """
                    MATCH (f:Entity {name: $feature_name, entityType: "Feature"})
                    MATCH (m:Metric {name: $unique_name})
                    MERGE (f)-[r:HAS_METRIC]->(m)
                    ON CREATE SET r.created = datetime()
                    RETURN f, r, m
                    """
                    
                    link_params = {
                        "feature_name": feature_name,
                        "unique_name": unique_metric_name
                    }
                    
                    self._safe_execute_query(
                        link_query,
                        parameters_=link_params,
                        database_=self.neo4j_database
                    )
                    
                    self.logger.debug(f"Added metric '{metric_name}' to feature '{feature_name}'")
                
        except Exception as e:
            self.logger.error(f"Error adding metrics to feature: {str(e)}")
    
    def _add_metadata_to_feature(self, feature_name: str, metadata: Dict[str, Any]) -> None:
        """
        Add metadata to a feature as properties.
        
        Args:
            feature_name: Name of the feature
            metadata: Dictionary of metadata to add
        """
        try:
            # Filter out None values and convert to properties
            filtered_metadata = {k: v for k, v in metadata.items() if v is not None}
            
            if not filtered_metadata:
                return
                
            # Build dynamic property setting
            props = []
            params = {"feature_name": feature_name}
            
            for i, (key, value) in enumerate(filtered_metadata.items()):
                param_name = f"meta_{i}"
                props.append(f"f.{key} = ${param_name}")
                params[param_name] = value
            
            if not props:
                return
                
            # Update feature entity with metadata
            metadata_query = f"""
            MATCH (f:Entity {{name: $feature_name, entityType: "Feature"}})
            SET {', '.join(props)},
                f.lastUpdated = datetime()
            RETURN f
            """
            
            self._safe_execute_query(
                metadata_query,
                parameters_=params,
                database_=self.neo4j_database
            )
            
        except Exception as e:
            self.logger.error(f"Error adding metadata to feature: {str(e)}")

    def record_decision(
        self,
        project_name: str,
        decision_name: str,
        description: str,
        alternatives: Optional[List[str]] = None,
        rationale: Optional[str] = None,
        constraints: Optional[str] = None,
        affects: Optional[List[Dict[str, Any]]] = None,
        confidence: float = 0.8,
        impact: str = "Medium",
        decision_date: Optional[str] = None,
        review_date: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> str:
        """
        Record a design/architecture decision for a project.
        
        Args:
            project_name: Name of the project
            decision_name: Name of the decision to record
            description: Description of the decision
            alternatives: Optional list of alternative options considered
            rationale: Optional explanation of why this decision was made
            constraints: Optional constraints that influenced the decision
            affects: Optional list of entities affected by this decision with format:
                   [{"name": "EntityName", "impact": "High", "direction": "Positive"}]
            confidence: Confidence level for this decision (0.0-1.0)
            impact: Impact level of the decision (Low, Medium, High, Critical)
            decision_date: Optional date when the decision was made (ISO format)
            review_date: Optional date when the decision should be reviewed (ISO format)
            metadata: Optional additional metadata as key-value pairs
            
        Returns:
            JSON string with result information
        """
        try:
            self._ensure_initialized()
            
            if not self.neo4j_driver:
                return dict_to_json({
                    "status": "error",
                    "message": "Neo4j driver not initialized"
                })
            
            # Format alternatives as a string
            alternatives_str = None
            if alternatives and isinstance(alternatives, list):
                alternatives_str = json.dumps(alternatives)
                
            # Validate and format dates
            decision_date_param = None
            if decision_date:
                try:
                    # Validate date format
                    datetime.datetime.fromisoformat(decision_date)
                    decision_date_param = decision_date
                except ValueError:
                    self.logger.warn(f"Invalid decision date format: {decision_date}, using current date")
                    decision_date_param = datetime.datetime.now().isoformat()
            else:
                decision_date_param = datetime.datetime.now().isoformat()
                
            review_date_param = None
            if review_date:
                try:
                    # Validate date format
                    datetime.datetime.fromisoformat(review_date)
                    review_date_param = review_date
                except ValueError:
                    self.logger.warn(f"Invalid review date format: {review_date}, ignoring")
            
            # Create decision entity
            decision_query = """
            MERGE (d:Entity {name: $decision_name, entityType: "Decision"})
            ON CREATE SET d.created = datetime(),
                          d.description = $description,
                          d.alternatives = $alternatives,
                          d.confidence = $confidence,
                          d.impact = $impact,
                          d.decisionDate = datetime($decision_date),
                          d.status = "Active",
                          d.lastUpdated = datetime()
            """
            
            if review_date_param:
                decision_query += ", d.reviewDate = datetime($review_date)"
                
            decision_query += " RETURN d"
            
            decision_params = {
                "decision_name": decision_name,
                "description": description,
                "alternatives": alternatives_str,
                "confidence": confidence,
                "impact": impact,
                "decision_date": decision_date_param
            }
            
            if review_date_param:
                decision_params["review_date"] = review_date_param
            
            decision_result = self._safe_execute_query(
                decision_query,
                parameters_=decision_params,
                database_=self.neo4j_database
            )
            
            # Create rationale entity if provided
            if rationale:
                rationale_name = f"{decision_name}_Rationale"
                
                rationale_query = """
                MERGE (r:Entity {name: $rationale_name, entityType: "Rationale"})
                ON CREATE SET r.created = datetime(),
                              r.description = $rationale,
                              r.confidence = $confidence,
                              r.lastUpdated = datetime()
                """
                
                if constraints:
                    rationale_query += ", r.constraints = $constraints"
                    
                rationale_query += " RETURN r"
                
                rationale_params = {
                    "rationale_name": rationale_name,
                    "rationale": rationale,
                    "confidence": confidence
                }
                
                if constraints:
                    rationale_params["constraints"] = constraints
                
                rationale_result = self._safe_execute_query(
                    rationale_query,
                    parameters_=rationale_params,
                    database_=self.neo4j_database
                )
                
                # Link decision to rationale
                link_query = """
                MATCH (d:Entity {name: $decision_name, entityType: "Decision"})
                MATCH (r:Entity {name: $rationale_name, entityType: "Rationale"})
                MERGE (d)-[rel:HAS_RATIONALE]->(r)
                ON CREATE SET rel.created = datetime(),
                              rel.strength = "Strong",
                              rel.confidence = $confidence
                RETURN d, rel, r
                """
                
                link_params = {
                    "decision_name": decision_name,
                    "rationale_name": rationale_name,
                    "confidence": confidence
                }
                
                self._safe_execute_query(
                    link_query,
                    parameters_=link_params,
                    database_=self.neo4j_database
                )
            
            # Link to project container
            container_link_query = """
            MATCH (pc:MemoryContainer {name: $project_name, type: "ProjectContainer"})
            MATCH (d:Entity {name: $decision_name, entityType: "Decision"})
            MERGE (pc)-[r:CONTAINS]->(d)
            ON CREATE SET r.since = datetime(),
                          r.confidence = $confidence
            RETURN pc, d
            """
            
            container_link_params = {
                "project_name": project_name,
                "decision_name": decision_name,
                "confidence": confidence
            }
            
            container_result = self._safe_execute_query(
                container_link_query,
                parameters_=container_link_params,
                database_=self.neo4j_database
            )
            
            # Check if the project container exists
            if not container_result or not container_result[0]:
                self.logger.warn(f"Project container '{project_name}' not found when recording decision")
                
                # Create project container if it doesn't exist
                container_creation = self.create_project_container(project_name)
                container_data = json.loads(container_creation)
                
                if container_data.get("status") != "success":
                    self.logger.error(f"Failed to create missing project container: {container_data.get('message')}")
                    return dict_to_json({
                        "status": "warning",
                        "message": f"Decision recorded but project container could not be created: {container_data.get('message')}",
                        "data": {
                            "decision_name": decision_name,
                            "project_name": project_name
                        }
                    })
                    
                # Try linking again
                self._safe_execute_query(
                    container_link_query,
                    parameters_=container_link_params,
                    database_=self.neo4j_database
                )
                
            # Link to rationale container as well if created
            if rationale:
                rationale_name = f"{decision_name}_Rationale"
                
                rationale_container_query = """
                MATCH (pc:MemoryContainer {name: $project_name, type: "ProjectContainer"})
                MATCH (r:Entity {name: $rationale_name, entityType: "Rationale"})
                MERGE (pc)-[rel:CONTAINS]->(r)
                ON CREATE SET rel.since = datetime(),
                              rel.confidence = $confidence
                RETURN pc, r
                """
                
                rationale_container_params = {
                    "project_name": project_name,
                    "rationale_name": rationale_name,
                    "confidence": confidence
                }
                
                self._safe_execute_query(
                    rationale_container_query,
                    parameters_=rationale_container_params,
                    database_=self.neo4j_database
                )
            
            # Create affects relationships if provided
            if affects and isinstance(affects, list):
                self._create_affects_relationships(decision_name, affects, confidence)
            
            # Add metadata if provided
            if metadata and isinstance(metadata, dict):
                self._add_metadata_to_decision(decision_name, metadata)
            
            self.logger.info(f"Recorded decision '{decision_name}' for project '{project_name}'")
            
            result_data = {
                "project_name": project_name,
                "decision_name": decision_name,
                "created": datetime.datetime.now().isoformat()
            }
            
            if rationale:
                result_data["rationale_name"] = f"{decision_name}_Rationale"
            
            return dict_to_json({
                "status": "success",
                "message": f"Decision '{decision_name}' recorded successfully",
                "data": result_data
            })
            
        except Exception as e:
            error_info = extract_error(e)
            self.logger.error(f"Error recording decision: {error_info}")
            return dict_to_json({
                "status": "error",
                "message": f"Failed to record decision: {error_info}"
            })
    
    def _create_affects_relationships(self, decision_name: str, affects: List[Dict[str, Any]], confidence: float) -> None:
        """
        Create relationships between a decision and entities it affects.
        
        Args:
            decision_name: Name of the decision
            affects: List of entities affected by this decision with format:
                   [{"name": "EntityName", "impact": "High", "direction": "Positive"}]
            confidence: Confidence level for the relationship
        """
        try:
            for affected in affects:
                entity_name = affected.get("name")
                
                if not entity_name:
                    self.logger.warn("Skipping affected entity with missing name")
                    continue
                
                # Get affect properties
                impact = affected.get("impact", "Medium")
                direction = affected.get("direction", "Neutral")
                affect_confidence = affected.get("confidence", confidence)
                
                # Create affects relationship
                affects_query = """
                MATCH (d:Entity {name: $decision_name, entityType: "Decision"})
                MATCH (e:Entity {name: $entity_name})
                MERGE (d)-[r:AFFECTS]->(e)
                ON CREATE SET r.created = datetime(),
                              r.impact = $impact,
                              r.direction = $direction,
                              r.confidence = $affect_confidence
                ON MATCH SET r.lastUpdated = datetime(),
                             r.impact = $impact,
                             r.direction = $direction,
                             r.confidence = $affect_confidence
                RETURN d, r, e
                """
                
                affects_params = {
                    "decision_name": decision_name,
                    "entity_name": entity_name,
                    "impact": impact,
                    "direction": direction,
                    "affect_confidence": affect_confidence
                }
                
                result = self._safe_execute_query(
                    affects_query,
                    parameters_=affects_params,
                    database_=self.neo4j_database
                )
                
                if not result or not result[0]:
                    self.logger.warn(f"Could not create affects relationship to '{entity_name}' - entity may not exist")
                else:
                    self.logger.debug(f"Created affects relationship from decision '{decision_name}' to entity '{entity_name}'")
                
        except Exception as e:
            self.logger.error(f"Error creating affects relationships: {str(e)}")
    
    def _add_metadata_to_decision(self, decision_name: str, metadata: Dict[str, Any]) -> None:
        """
        Add metadata to a decision as properties.
        
        Args:
            decision_name: Name of the decision
            metadata: Dictionary of metadata to add
        """
        try:
            # Filter out None values and convert to properties
            filtered_metadata = {k: v for k, v in metadata.items() if v is not None}
            
            if not filtered_metadata:
                return
                
            # Build dynamic property setting
            props = []
            params = {"decision_name": decision_name}
            
            for i, (key, value) in enumerate(filtered_metadata.items()):
                param_name = f"meta_{i}"
                props.append(f"d.{key} = ${param_name}")
                params[param_name] = value
            
            if not props:
                return
                
            # Update decision entity with metadata
            metadata_query = f"""
            MATCH (d:Entity {{name: $decision_name, entityType: "Decision"}})
            SET {', '.join(props)},
                d.lastUpdated = datetime()
            RETURN d
            """
            
            self._safe_execute_query(
                metadata_query,
                parameters_=params,
                database_=self.neo4j_database
            )
            
        except Exception as e:
            self.logger.error(f"Error adding metadata to decision: {str(e)}")
            
    def create_project_relationship(
        self,
        from_entity: str,
        to_entity: str,
        relationship_type: str,
        properties: Optional[Dict[str, Any]] = None,
        confidence: float = 0.9,
        evidence: Optional[str] = None,
        project_name: Optional[str] = None
    ) -> str:
        """
        Create a relationship between two project entities.
        
        Args:
            from_entity: Name of the source entity
            to_entity: Name of the target entity
            relationship_type: Type of relationship to create (e.g., DEPENDS_ON, RELATES_TO)
            properties: Optional additional properties for the relationship
            confidence: Confidence level for this relationship (0.0-1.0)
            evidence: Optional evidence or justification for this relationship
            project_name: Optional project name for validation
            
        Returns:
            JSON string with result information
        """
        try:
            self._ensure_initialized()
            
            if not self.neo4j_driver:
                return dict_to_json({
                    "status": "error",
                    "message": "Neo4j driver not initialized"
                })
            
            # Validate relationship type - convert to uppercase and ensure no spaces
            valid_relationship_type = relationship_type.upper().replace(" ", "_")
            
            # Prepare properties for the relationship
            rel_properties = {
                "created": datetime.datetime.now().isoformat(),
                "confidence": confidence
            }
            
            # Add evidence if provided
            if evidence:
                rel_properties["evidence"] = evidence
                
            # Add additional properties if provided
            if properties and isinstance(properties, dict):
                for key, value in properties.items():
                    if key and value is not None and key not in rel_properties:
                        rel_properties[key] = value
            
            # Build parameters
            params = {
                "from_entity": from_entity,
                "to_entity": to_entity,
                "rel_properties": rel_properties
            }
            
            # Create relationship query
            relationship_query = """
            MATCH (from:Entity {name: $from_entity})
            MATCH (to:Entity {name: $to_entity})
            """
            
            # Add optional project validation
            if project_name:
                relationship_query += """
                MATCH (pc:MemoryContainer {name: $project_name, type: "ProjectContainer"})
                WHERE (pc)-[:CONTAINS]->(from) AND (pc)-[:CONTAINS]->(to)
                """
                params["project_name"] = project_name
            
            # Complete the query
            relationship_query += f"""
            MERGE (from)-[r:{valid_relationship_type}]->(to)
            ON CREATE SET r += $rel_properties
            ON MATCH SET r.lastUpdated = datetime(),
                         r.confidence = $rel_properties.confidence
            """
            
            # Add evidence update if provided
            if evidence:
                relationship_query += ", r.evidence = $rel_properties.evidence"
                
            # Add property updates for additional properties
            if properties and isinstance(properties, dict):
                for key in properties.keys():
                    if key and key not in ["created", "lastUpdated"]:
                        relationship_query += f", r.{key} = $rel_properties.{key}"
            
            relationship_query += " RETURN from, r, to"
            
            # Execute the query
            result = self._safe_execute_query(
                relationship_query,
                parameters_=params,
                database_=self.neo4j_database
            )
            
            # Check if the entities exist and relationship was created
            if not result or not result[0]:
                entity_check_query = """
                OPTIONAL MATCH (from:Entity {name: $from_entity})
                OPTIONAL MATCH (to:Entity {name: $to_entity})
                RETURN from IS NOT NULL as from_exists, to IS NOT NULL as to_exists
                """
                
                entity_check_params = {
                    "from_entity": from_entity,
                    "to_entity": to_entity
                }
                
                check_result = self._safe_execute_query(
                    entity_check_query,
                    parameters_=entity_check_params,
                    database_=self.neo4j_database
                )
                
                if check_result and check_result[0]:
                    record = check_result[0][0]
                    from_exists = record.get("from_exists", False)
                    to_exists = record.get("to_exists", False)
                    
                    if not from_exists and not to_exists:
                        return dict_to_json({
                            "status": "error",
                            "message": f"Both source entity '{from_entity}' and target entity '{to_entity}' do not exist"
                        })
                    elif not from_exists:
                        return dict_to_json({
                            "status": "error",
                            "message": f"Source entity '{from_entity}' does not exist"
                        })
                    elif not to_exists:
                        return dict_to_json({
                            "status": "error",
                            "message": f"Target entity '{to_entity}' does not exist"
                        })
                    elif project_name:
                        return dict_to_json({
                            "status": "error",
                            "message": f"One or both entities are not part of project '{project_name}'"
                        })
                
                return dict_to_json({
                    "status": "error",
                    "message": "Failed to create relationship for unknown reason"
                })
            
            self.logger.info(f"Created relationship {from_entity} -[{valid_relationship_type}]-> {to_entity}")
            
            result_data = {
                "from_entity": from_entity,
                "to_entity": to_entity,
                "relationship_type": valid_relationship_type,
                "confidence": confidence,
                "created": datetime.datetime.now().isoformat()
            }
            
            if project_name:
                result_data["project_name"] = project_name
            
            return dict_to_json({
                "status": "success",
                "message": f"Relationship from '{from_entity}' to '{to_entity}' created successfully",
                "data": result_data
            })
            
        except Exception as e:
            error_info = extract_error(e)
            self.logger.error(f"Error creating project relationship: {error_info}")
            return dict_to_json({
                "status": "error",
                "message": f"Failed to create relationship: {error_info}"
            })

    def get_project_structure(
        self,
        project_name: str,
        include_types: Optional[List[str]] = None,
        exclude_types: Optional[List[str]] = None,
        format: str = "hierarchical"
    ) -> str:
        """
        Retrieves the structure of a project, including domains, components, and other entities.
        
        Args:
            project_name: Name of the project
            include_types: Optional list of entity types to include (if None, include all)
            exclude_types: Optional list of entity types to exclude
            format: Output format - 'hierarchical' (tree) or 'flat' (list)
            
        Returns:
            JSON string with project structure information
        """
        try:
            self._ensure_initialized()
            
            if not self.neo4j_driver:
                return dict_to_json({
                    "status": "error",
                    "message": "Neo4j driver not initialized"
                })
                
            # Build entity type filter
            entity_filter = ""
            params = {"project_name": project_name}
            
            if include_types and isinstance(include_types, list):
                valid_types = [t for t in include_types if t and isinstance(t, str)]
                if valid_types:
                    entity_filter = "AND e.entityType IN $include_types"
                    params["include_types"] = valid_types
                    
            if exclude_types and isinstance(exclude_types, list):
                valid_excludes = [t for t in exclude_types if t and isinstance(t, str)]
                if valid_excludes:
                    if entity_filter:
                        entity_filter += " AND e.entityType NOT IN $exclude_types"
                    else:
                        entity_filter = "AND e.entityType NOT IN $exclude_types"
                    params["exclude_types"] = valid_excludes
            
            # Check if project exists
            project_check_query = """
            MATCH (p:MemoryContainer {name: $project_name, type: "ProjectContainer"})
            RETURN count(p) > 0 as exists
            """
            
            check_result = self._safe_execute_query(
                project_check_query,
                parameters_={"project_name": project_name},
                database_=self.neo4j_database
            )
            
            project_exists = False
            if check_result and isinstance(check_result, list) and check_result:
                try:
                    project_exists = bool(check_result[0].get("exists", False))
                except Exception:
                    # If we can't access the result properly, assume the project doesn't exist
                    pass
            
            if not project_exists:
                return dict_to_json({
                    "status": "error",
                    "message": f"Project '{project_name}' does not exist"
                })
                
            # Different queries based on format
            if format.lower() == "hierarchical":
                # Get project hierarchy with relationships between entities
                hierarchy_query = f"""
                MATCH (p:MemoryContainer {{name: $project_name, type: "ProjectContainer"}})-[:CONTAINS]->(e:Entity) {entity_filter}
                OPTIONAL MATCH (e)-[r]->(other:Entity)
                WHERE (p)-[:CONTAINS]->(other)
                RETURN e.name as name, e.entityType as type, e.description as description, 
                       e.status as status, e.confidence as confidence, e.dateCreated as dateCreated,
                       collect(DISTINCT {{
                           type: type(r), 
                           target: other.name, 
                           targetType: other.entityType
                       }}) as relationships
                """
                
                result = self._safe_execute_query(
                    hierarchy_query,
                    parameters_=params,
                    database_=self.neo4j_database
                )
                
                # Construct tree
                entities = {}
                relationships = []
                
                # First pass: create all entity nodes
                if result and isinstance(result, list) and result:
                    for record in result:
                        try:
                            # Access values by field name from Neo4j record
                            entity_name = record.get("name")
                            entity_type = record.get("type")
                            description = record.get("description", "")
                            status = record.get("status", "Unknown")
                            confidence = record.get("confidence", 0.0)
                            date_created = record.get("dateCreated", "")
                            record_relationships = record.get("relationships", [])
                            
                            if entity_name and entity_type:
                                entities[entity_name] = {
                                    "name": entity_name,
                                    "type": entity_type,
                                    "description": description or "",
                                    "status": status or "Unknown",
                                    "confidence": confidence or 0.0,
                                    "dateCreated": date_created or "",
                                    "children": []
                                }
                                
                                # Extract relationships
                                if record_relationships:
                                    for rel in record_relationships:
                                        if isinstance(rel, dict):
                                            rel_type = rel.get("type")
                                            target = rel.get("target")
                                            if rel_type and target:
                                                relationships.append({
                                                    "source": entity_name,
                                                    "type": rel_type,
                                                    "target": target
                                                })
                        except Exception as e:
                            self.logger.warn(f"Error processing record: {str(e)}")
                            continue
                
                # Second pass: build hierarchy
                # First collect all entities that are targets (children)
                children = set()
                for rel in relationships:
                    children.add(rel["target"])
                
                # Root entities are those that are not children of any other entity
                root_entities = [name for name in entities.keys() if name not in children]
                
                # Build tree starting from roots
                tree = []
                
                for root in root_entities:
                    if root in entities:
                        tree.append(self._build_entity_tree(root, entities, relationships))
                
                # Sort tree by entity type and name
                tree.sort(key=lambda x: (x.get("type", ""), x.get("name", "")))
                
                return dict_to_json({
                    "status": "success",
                    "project_name": project_name,
                    "structure": tree,
                    "entity_count": len(entities),
                    "format": "hierarchical"
                })
                
            else:  # flat format
                flat_query = f"""
                MATCH (p:MemoryContainer {{name: $project_name, type: "ProjectContainer"}})-[:CONTAINS]->(e:Entity) {entity_filter}
                RETURN e.name as name, e.entityType as type, e.description as description, 
                       e.status as status, e.confidence as confidence, e.dateCreated as dateCreated
                ORDER BY e.entityType, e.name
                """
                
                result = self._safe_execute_query(
                    flat_query,
                    parameters_=params,
                    database_=self.neo4j_database
                )
                
                entities = []
                if result and isinstance(result, list) and result:
                    for record in result:
                        try:
                            # Access values by field name from Neo4j record
                            entity_name = record.get("name")
                            entity_type = record.get("type")
                            description = record.get("description", "")
                            status = record.get("status", "Unknown")
                            confidence = record.get("confidence", 0.0)
                            date_created = record.get("dateCreated", "")
                            
                            if entity_name and entity_type:
                                entities.append({
                                    "name": entity_name,
                                    "type": entity_type,
                                    "description": description or "",
                                    "status": status or "Unknown",
                                    "confidence": confidence or 0.0,
                                    "dateCreated": date_created or ""
                                })
                        except Exception as e:
                            self.logger.warn(f"Error processing record: {str(e)}")
                            continue
                
                return dict_to_json({
                    "status": "success",
                    "project_name": project_name,
                    "entities": entities,
                    "entity_count": len(entities),
                    "format": "flat"
                })
                
        except Exception as e:
            error_info = extract_error(e)
            self.logger.error(f"Error getting project structure: {error_info}")
            return dict_to_json({
                "status": "error",
                "message": f"Failed to get project structure: {error_info}"
            })
    
    def _build_entity_tree(self, entity_name: str, entities: Dict[str, Dict], relationships: List[Dict]) -> Dict:
        """
        Helper method to recursively build entity tree from flat relationship data.
        
        Args:
            entity_name: Name of the current entity
            entities: Dictionary of all entities
            relationships: List of all relationships
            
        Returns:
            Dictionary representing the entity and its children
        """
        if entity_name not in entities:
            return {}
            
        entity = entities[entity_name].copy()
        
        # Find all direct children (entities that this entity has a relationship to)
        children = []
        for rel in relationships:
            if rel["source"] == entity_name:
                child_name = rel["target"]
                if child_name in entities and child_name != entity_name:  # Avoid self-references
                    child = self._build_entity_tree(child_name, entities, relationships)
                    if child:
                        # Add relationship type to child
                        child["relationshipType"] = rel["type"]
                        children.append(child)
        
        if children:
            # Sort children by type and name
            children.sort(key=lambda x: (x.get("type", ""), x.get("name", "")))
            entity["children"] = children
        
        return entity
        
    def track_entity_evolution(
        self,
        entity_name: str,
        changes: Dict[str, Any],
        reason: Optional[str] = None,
        confidence: float = 0.9,
        user: Optional[str] = None,
        change_type: str = "Update"
    ) -> str:
        """
        Track changes to an entity over time, creating a historical record.
        
        Args:
            entity_name: Name of the entity to track
            changes: Dictionary of changes (property name -> new value)
            reason: Optional reason for the change
            confidence: Confidence level for this change (0.0-1.0)
            user: Optional user who made the change
            change_type: Type of change (Update, Create, Delete, etc.)
            
        Returns:
            JSON string with result information
        """
        try:
            self._ensure_initialized()
            
            if not self.neo4j_driver:
                return dict_to_json({
                    "status": "error",
                    "message": "Neo4j driver not initialized"
                })
                
            # Check if entity exists
            entity_query = """
            MATCH (e:Entity {name: $entity_name})
            RETURN e.entityType as entityType
            """
            
            result = self._safe_execute_query(
                entity_query,
                entity_name=entity_name,
                database_=self.neo4j_database
            )
            
            entity_type = None
            if result and len(result) > 0:
                for record in result:
                    if "entityType" in record:
                        entity_type = record["entityType"]
                        break
            
            if not entity_type:
                return dict_to_json({
                    "status": "error",
                    "message": f"Entity '{entity_name}' not found"
                })
            
            # Create evolution node to track the change
            change_id = generate_id()
            timestamp = datetime.datetime.now().isoformat()
            
            # Filter out None values
            filtered_changes = {k: v for k, v in changes.items() if v is not None}
            
            # Convert changes to JSON string
            changes_json = json.dumps(filtered_changes)
            
            evolution_query = """
            MATCH (e:Entity {name: $entity_name})
            CREATE (c:Change {
                id: $change_id,
                timestamp: $timestamp,
                type: $change_type,
                changes: $changes,
                reason: $reason,
                confidence: $confidence,
                user: $user
            })
            CREATE (e)-[r:HAS_CHANGE]->(c)
            SET e.lastUpdated = $timestamp
            """
            
            for key, value in filtered_changes.items():
                # Update the entity properties with changes
                evolution_query += f", e.{key} = $change_{key}"
            
            evolution_query += " RETURN c.id as changeId"
            
            # Prepare parameters
            params = {
                "entity_name": entity_name,
                "change_id": change_id,
                "timestamp": timestamp,
                "change_type": change_type,
                "changes": changes_json,
                "reason": reason,
                "confidence": confidence,
                "user": user
            }
            
            # Add change values as parameters
            for key, value in filtered_changes.items():
                params[f"change_{key}"] = value
            
            result = self._safe_execute_query(
                evolution_query,
                parameters_=params,
                database_=self.neo4j_database
            )
            
            self.logger.info(f"Tracked evolution for entity '{entity_name}': {change_type}")
            
            return dict_to_json({
                "status": "success",
                "message": f"Evolution tracked for entity '{entity_name}'",
                "data": {
                    "entity_name": entity_name,
                    "entity_type": entity_type,
                    "change_id": change_id,
                    "timestamp": timestamp,
                    "change_type": change_type
                }
            })
            
        except Exception as e:
            error_info = extract_error(e)
            self.logger.error(f"Error tracking entity evolution: {error_info}")
            return dict_to_json({
                "status": "error",
                "message": f"Failed to track entity evolution: {error_info}"
            })

    def discover_project_structure(
        self,
        project_name: str,
        content: str,
        content_type: str = "text",
        confidence: float = 0.7,
        auto_create: bool = False
    ) -> str:
        """
        Analyze content to discover project structure like domains, components, and relationships.
        
        Args:
            project_name: Name of the project
            content: Text content to analyze (documentation, code summary, etc.)
            content_type: Type of content ('text', 'json', 'code')
            confidence: Confidence level for discovered items (0.0-1.0)
            auto_create: Whether to automatically create discovered entities
            
        Returns:
            JSON string with discovered structure information
        """
        try:
            self._ensure_initialized()
            
            if not self.neo4j_driver:
                return dict_to_json({
                    "status": "error",
                    "message": "Neo4j driver not initialized"
                })
                
            # Check if project exists
            project_check_query = """
            MATCH (p:MemoryContainer {name: $project_name, type: "ProjectContainer"})
            RETURN count(p) > 0 as exists
            """
            
            check_result = self._safe_execute_query(
                project_check_query,
                parameters_={"project_name": project_name},
                database_=self.neo4j_database
            )
            
            # Create project if it doesn't exist and auto_create is enabled
            if (not check_result or not check_result[0] or not check_result[0][0]) and auto_create:
                self.create_project_container(
                    project_name=project_name,
                    description=f"Auto-discovered project: {project_name}",
                    confidence=confidence
                )
                
            # Simple text-based extraction logic
            discovered = {
                "domains": [],
                "components": [],
                "features": [],
                "relationships": []
            }
            
            # Different parsing logic based on content_type
            if content_type.lower() == "json":
                try:
                    # Try to parse JSON structure directly
                    parsed = json.loads(content)
                    
                    # Extract domains
                    if "domains" in parsed and isinstance(parsed["domains"], list):
                        for domain in parsed["domains"]:
                            if isinstance(domain, dict) and "name" in domain:
                                domain_name = domain.get("name")
                                description = domain.get("description", "")
                                
                                discovered["domains"].append({
                                    "name": domain_name,
                                    "description": description
                                })
                    
                    # Extract components
                    if "components" in parsed and isinstance(parsed["components"], list):
                        for component in parsed["components"]:
                            if isinstance(component, dict) and "name" in component:
                                component_name = component.get("name")
                                description = component.get("description", "")
                                domain = component.get("domain", "")
                                
                                discovered["components"].append({
                                    "name": component_name,
                                    "description": description,
                                    "domain": domain
                                })
                    
                    # Extract features
                    if "features" in parsed and isinstance(parsed["features"], list):
                        for feature in parsed["features"]:
                            if isinstance(feature, dict) and "name" in feature:
                                feature_name = feature.get("name")
                                description = feature.get("description", "")
                                
                                discovered["features"].append({
                                    "name": feature_name,
                                    "description": description
                                })
                    
                    # Extract relationships
                    if "relationships" in parsed and isinstance(parsed["relationships"], list):
                        for rel in parsed["relationships"]:
                            if isinstance(rel, dict) and "from" in rel and "to" in rel and "type" in rel:
                                discovered["relationships"].append({
                                    "from": rel.get("from"),
                                    "to": rel.get("to"),
                                    "type": rel.get("type")
                                })
                                
                except json.JSONDecodeError:
                    self.logger.warn("Failed to parse content as JSON, falling back to text analysis")
                    content_type = "text"
            
            # Text-based analysis (simplistic)
            if content_type.lower() == "text":
                # Simple pattern matching
                domain_pattern = r"(?:Domain|Module):\s*([A-Za-z0-9_-]+)(?:\s*\(([^)]+)\))?"
                component_pattern = r"(?:Component|Service|Class):\s*([A-Za-z0-9_-]+)(?:\s*\(([^)]+)\))?"
                feature_pattern = r"(?:Feature|Functionality):\s*([A-Za-z0-9_-]+)(?:\s*\(([^)]+)\))?"
                relationship_pattern = r"([A-Za-z0-9_-]+)\s+(depends on|uses|calls|includes|contains)\s+([A-Za-z0-9_-]+)"
                
                import re
                
                # Find domains
                for match in re.finditer(domain_pattern, content):
                    name = match.group(1)
                    description = match.group(2) if match.group(2) else ""
                    
                    # Check if already discovered
                    if not any(d["name"] == name for d in discovered["domains"]):
                        discovered["domains"].append({
                            "name": name,
                            "description": description
                        })
                
                # Find components
                for match in re.finditer(component_pattern, content):
                    name = match.group(1)
                    description = match.group(2) if match.group(2) else ""
                    
                    # Check if already discovered
                    if not any(c["name"] == name for c in discovered["components"]):
                        discovered["components"].append({
                            "name": name,
                            "description": description
                        })
                
                # Find features
                for match in re.finditer(feature_pattern, content):
                    name = match.group(1)
                    description = match.group(2) if match.group(2) else ""
                    
                    # Check if already discovered
                    if not any(f["name"] == name for f in discovered["features"]):
                        discovered["features"].append({
                            "name": name,
                            "description": description
                        })
                
                # Find relationships
                for match in re.finditer(relationship_pattern, content):
                    from_entity = match.group(1)
                    rel_type = match.group(2).replace(" ", "_").upper()
                    to_entity = match.group(3)
                    
                    # Check if already discovered
                    if not any(r["from"] == from_entity and r["to"] == to_entity and r["type"] == rel_type 
                              for r in discovered["relationships"]):
                        discovered["relationships"].append({
                            "from": from_entity,
                            "to": to_entity,
                            "type": rel_type
                        })
                    
            # Auto-create discovered entities if enabled
            if auto_create:
                # Create domains
                for domain in discovered["domains"]:
                    self.add_domain(
                        project_name=project_name,
                        domain_name=domain["name"],
                        description=domain["description"],
                        confidence=confidence
                    )
                
                # Create components
                for component in discovered["components"]:
                    # Find potential domain for this component
                    domain = component.get("domain")
                    
                    # If no explicit domain, try to infer based on relationships
                    if not domain:
                        for rel in discovered["relationships"]:
                            if rel["to"] == component["name"] and rel["type"] == "CONTAINS":
                                for d in discovered["domains"]:
                                    if d["name"] == rel["from"]:
                                        domain = d["name"]
                                        break
                                if domain:
                                    break
                    
                    self.add_component(
                        project_name=project_name,
                        component_name=component["name"],
                        domain_name=domain,
                        responsibility=component.get("description", ""),
                        confidence=confidence
                    )
                
                # Create features
                for feature in discovered["features"]:
                    self.add_feature(
                        project_name=project_name,
                        feature_name=feature["name"],
                        description=feature["description"],
                        confidence=confidence
                    )
                
                # Create relationships
                for rel in discovered["relationships"]:
                    self.create_project_relationship(
                        from_entity=rel["from"],
                        to_entity=rel["to"],
                        relationship_type=rel["type"],
                        confidence=confidence,
                        project_name=project_name
                    )
            
            return dict_to_json({
                "status": "success",
                "message": "Project structure discovered",
                "data": {
                    "project_name": project_name,
                    "discovered": discovered,
                    "auto_created": auto_create,
                    "counts": {
                        "domains": len(discovered["domains"]),
                        "components": len(discovered["components"]),
                        "features": len(discovered["features"]),
                        "relationships": len(discovered["relationships"])
                    }
                }
            })
            
        except Exception as e:
            error_info = extract_error(e)
            self.logger.error(f"Error discovering project structure: {error_info}")
            return dict_to_json({
                "status": "error",
                "message": f"Failed to discover project structure: {error_info}"
            })

    def transfer_knowledge(
        self,
        from_project: str,
        to_project: str,
        entity_types: Optional[List[str]] = None,
        entity_names: Optional[List[str]] = None,
        include_relationships: bool = True,
        confidence: float = 0.8
    ) -> str:
        """
        Transfer knowledge entities and relationships from one project to another.
        
        Args:
            from_project: Source project name
            to_project: Target project name
            entity_types: Optional list of entity types to transfer (if None, transfer all)
            entity_names: Optional list of specific entity names to transfer
            include_relationships: Whether to also transfer relationships between entities
            confidence: Confidence level for the transferred entities (0.0-1.0)
            
        Returns:
            JSON string with transfer results
        """
        try:
            self._ensure_initialized()
            
            if not self.neo4j_driver:
                return dict_to_json({
                    "status": "error",
                    "message": "Neo4j driver not initialized"
                })
                
            # Verify both projects exist
            projects_query = """
            MATCH (p1:MemoryContainer {name: $from_project, type: "ProjectContainer"})
            MATCH (p2:MemoryContainer {name: $to_project, type: "ProjectContainer"})
            RETURN count(p1) > 0 as from_exists, count(p2) > 0 as to_exists
            """
            
            projects_result = self._safe_execute_query(
                projects_query,
                parameters_={"from_project": from_project, "to_project": to_project},
                database_=self.neo4j_database
            )
            
            from_exists = False
            to_exists = False
            
            if projects_result and len(projects_result) > 0:
                for record in projects_result:
                    from_exists = record[0] if len(record) > 0 else False
                    to_exists = record[1] if len(record) > 1 else False
                    break
            
            if not from_exists:
                return dict_to_json({
                    "status": "error",
                    "message": f"Source project '{from_project}' does not exist"
                })
                
            if not to_exists:
                return dict_to_json({
                    "status": "error",
                    "message": f"Target project '{to_project}' does not exist"
                })
            
            # Build entity filter
            entity_filter = ""
            query_params = {
                "from_project": from_project,
                "to_project": to_project,
                "confidence": confidence
            }
            
            if entity_types and isinstance(entity_types, list):
                valid_types = [t for t in entity_types if t and isinstance(t, str)]
                if valid_types:
                    entity_filter += "AND e.entityType IN $entity_types"
                    query_params["entity_types"] = valid_types
            
            if entity_names and isinstance(entity_names, list):
                valid_names = [n for n in entity_names if n and isinstance(n, str)]
                if valid_names:
                    if entity_filter:
                        entity_filter += " AND e.name IN $entity_names"
                    else:
                        entity_filter += "AND e.name IN $entity_names"
                    query_params["entity_names"] = valid_names
            
            # Get entities to transfer
            entities_query = f"""
            MATCH (from_pc:MemoryContainer {{name: $from_project, type: "ProjectContainer"}})-[:CONTAINS]->(e:Entity) {entity_filter}
            RETURN e.name as name, e.entityType as type, e.description as description,
                   e.status as status, e.dateCreated as dateCreated
            """
            
            entities_result = self._safe_execute_query(
                entities_query,
                parameters_=query_params,
                database_=self.neo4j_database
            )
            
            # Process and transfer entities
            transferred_entities = []
            entity_mapping = {}  # Map original names to new names if renamed
            
            if entities_result and len(entities_result) > 0:
                for record in entities_result:
                    entity_name = record[0] if len(record) > 0 else None
                    entity_type = record[1] if len(record) > 1 else None
                    description = record[2] if len(record) > 2 else ""
                    status = record[3] if len(record) > 3 else "Unknown"
                    
                    if entity_name and entity_type:
                        # Check if entity already exists in target project
                        check_query = """
                        MATCH (to_pc:MemoryContainer {name: $to_project, type: "ProjectContainer"})
                        MATCH (e:Entity {name: $entity_name})
                        RETURN exists((to_pc)-[:CONTAINS]->(e)) as exists
                        """
                        
                        check_result = self._safe_execute_query(
                            check_query,
                            parameters_={"to_project": to_project, "entity_name": entity_name},
                            database_=self.neo4j_database
                        )
                        
                        entity_exists = False
                        if check_result and len(check_result) > 0:
                            for check_record in check_result:
                                entity_exists = check_record[0] if len(check_record) > 0 else False
                                break
                        
                        # If entity exists, generate a new name
                        target_name = entity_name
                        if entity_exists:
                            target_name = f"{entity_name}_from_{from_project}"
                            entity_mapping[entity_name] = target_name
                        
                        # Create entity in target project
                        transfer_query = """
                        MATCH (to_pc:MemoryContainer {name: $to_project, type: "ProjectContainer"})
                        MERGE (e:Entity {name: $target_name})
                        ON CREATE SET e.entityType = $entity_type,
                                      e.description = $description,
                                      e.status = $status,
                                      e.confidence = $confidence,
                                      e.dateCreated = datetime(),
                                      e.transferredFrom = $from_project
                        ON MATCH SET e.description = $description,
                                     e.confidence = $confidence,
                                     e.lastUpdated = datetime(),
                                     e.transferredFrom = $from_project
                        MERGE (to_pc)-[:CONTAINS]->(e)
                        RETURN e.name as name
                        """
                        
                        transfer_result = self._safe_execute_query(
                            transfer_query,
                            parameters_={
                                "to_project": to_project,
                                "target_name": target_name,
                                "entity_type": entity_type,
                                "description": description,
                                "status": status,
                                "confidence": confidence,
                                "from_project": from_project
                            },
                            database_=self.neo4j_database
                        )
                        
                        transferred_entities.append({
                            "original_name": entity_name,
                            "target_name": target_name,
                            "type": entity_type,
                            "renamed": entity_name != target_name
                        })
            
            # Transfer relationships if needed
            transferred_relationships = []
            
            if include_relationships and transferred_entities:
                # Get names of all transferred entities
                original_names = [e["original_name"] for e in transferred_entities]
                
                # Find relationships between transferred entities
                rels_query = """
                MATCH (from_pc:MemoryContainer {name: $from_project, type: "ProjectContainer"})
                MATCH (from_pc)-[:CONTAINS]->(e1:Entity)
                MATCH (from_pc)-[:CONTAINS]->(e2:Entity)
                MATCH (e1)-[r]->(e2)
                WHERE e1.name IN $entity_names AND e2.name IN $entity_names
                RETURN e1.name as from_name, type(r) as rel_type, e2.name as to_name
                """
                
                rels_result = self._safe_execute_query(
                    rels_query,
                    parameters_={"from_project": from_project, "entity_names": original_names},
                    database_=self.neo4j_database
                )
                
                if rels_result and len(rels_result) > 0:
                    for record in rels_result:
                        from_name = record[0] if len(record) > 0 else None
                        rel_type = record[1] if len(record) > 1 else None
                        to_name = record[2] if len(record) > 2 else None
                        
                        if from_name and rel_type and to_name:
                            # Map to target names if renamed
                            target_from = entity_mapping.get(from_name, from_name)
                            target_to = entity_mapping.get(to_name, to_name)
                            
                            # Create relationship in target project
                            rel_transfer_result = self.create_project_relationship(
                                from_entity=target_from,
                                to_entity=target_to,
                                relationship_type=str(rel_type) if rel_type is not None else "RELATED_TO",
                                confidence=confidence,
                                project_name=to_project,
                                evidence=f"Transferred from {from_project}"
                            )
                            
                            transferred_relationships.append({
                                "from": target_from,
                                "to": target_to,
                                "type": rel_type
                            })
            
            return dict_to_json({
                "status": "success",
                "message": f"Successfully transferred knowledge from '{from_project}' to '{to_project}'",
                "data": {
                    "from_project": from_project,
                    "to_project": to_project,
                    "transferred_entities": transferred_entities,
                    "transferred_relationships": transferred_relationships,
                    "counts": {
                        "entities": len(transferred_entities),
                        "relationships": len(transferred_relationships)
                    }
                }
            })
            
        except Exception as e:
            error_info = extract_error(e)
            self.logger.error(f"Error transferring knowledge: {error_info}")
            return dict_to_json({
                "status": "error",
                "message": f"Failed to transfer knowledge: {error_info}"
            })
            
    def consolidate_project_entities(
        self,
        project_name: str,
        entity_names: List[str],
        primary_entity: str,
        reason: Optional[str] = None,
        preserve_originals: bool = False
    ) -> str:
        """
        Consolidate (merge) multiple related or duplicate entities in a project.
        
        Args:
            project_name: Name of the project
            entity_names: List of entity names to consolidate
            primary_entity: Name of the primary entity to keep/create
            reason: Optional reason for consolidation
            preserve_originals: Whether to preserve original entities or delete them
            
        Returns:
            JSON string with consolidation results
        """
        try:
            self._ensure_initialized()
            
            if not self.neo4j_driver:
                return dict_to_json({
                    "status": "error",
                    "message": "Neo4j driver not initialized"
                })
                
            # Check if project exists
            project_query = """
            MATCH (pc:MemoryContainer {name: $project_name, type: "ProjectContainer"})
            RETURN count(pc) > 0 as exists
            """
            
            project_result = self._safe_execute_query(
                project_query,
                parameters_={"project_name": project_name},
                database_=self.neo4j_database
            )
            
            project_exists = False
            if project_result and project_result[0]:
                project_exists = bool(project_result[0][0])
                
            if not project_exists:
                return dict_to_json({
                    "status": "error",
                    "message": f"Project '{project_name}' does not exist"
                })
                
            # Validate entity list
            if not entity_names or not isinstance(entity_names, list) or len(entity_names) < 2:
                return dict_to_json({
                    "status": "error",
                    "message": "At least two entity names must be provided for consolidation"
                })
                
            if primary_entity and primary_entity not in entity_names:
                entity_names.append(primary_entity)
                
            # Get details of entities to consolidate
            entities_query = """
            MATCH (pc:MemoryContainer {name: $project_name, type: "ProjectContainer"})
            MATCH (pc)-[:CONTAINS]->(e:Entity)
            WHERE e.name IN $entity_names
            RETURN e.name as name, e.entityType as type, e.description as description,
                   e.confidence as confidence
            """
            
            entities_result = self._safe_execute_query(
                entities_query,
                parameters_={"project_name": project_name, "entity_names": entity_names},
                database_=self.neo4j_database
            )
            
            # Process entity information
            found_entities = []
            entity_types = set()
            primary_type = None
            best_description = ""
            max_confidence = 0.0
            
            if entities_result and isinstance(entities_result, list) and len(entities_result) > 0:
                for record in entities_result:
                    # When accessing record data, ensure we're accessing the right values
                    # from the Neo4j records (which are tuples or have a get method)
                    try:
                        # Try to access by index first (tuple-like records)
                        if isinstance(record, (list, tuple)):
                            entity_name = record[0] if len(record) > 0 else None
                            entity_type = record[1] if len(record) > 1 else None
                            description = record[2] if len(record) > 2 else ""
                            confidence = record[3] if len(record) > 3 else 0.0
                        # Try to access by key (dict-like records)
                        elif hasattr(record, "get"):
                            entity_name = record.get("name")
                            entity_type = record.get("type")
                            description = record.get("description", "")
                            confidence = record.get("confidence", 0.0)
                        else:
                            # Skip records we can't process
                            continue
                        
                        if entity_name and entity_type:
                            found_entities.append(entity_name)
                            entity_types.add(entity_type)
                            
                            # If this is the primary entity, use its type
                            if entity_name == primary_entity:
                                primary_type = entity_type
                            
                            # Keep track of best description (non-empty)
                            if description and isinstance(description, str) and len(description) > len(best_description):
                                best_description = description
                                
                            # Track highest confidence - safely convert to float
                            try:
                                conf_value = float(confidence) if confidence is not None else 0.0
                                if conf_value > max_confidence:
                                    max_confidence = conf_value
                            except (ValueError, TypeError):
                                # If confidence isn't convertible to float, skip it
                                pass
                    except Exception as e:
                        self.logger.warn(f"Error processing entity record: {str(e)}")
                        continue
                        
            # Check if we found all entities
            missing_entities = set(entity_names) - set(found_entities)
            if missing_entities:
                return dict_to_json({
                    "status": "error",
                    "message": f"The following entities were not found in project '{project_name}': {', '.join(missing_entities)}"
                })
                
            # If no entities of the same type, warn about type mixing
            if len(entity_types) > 1:
                # If primary entity specified, use its type
                if not primary_type:
                    # Otherwise use most common type
                    type_counts = {}
                    # Ensure we're iterating over a valid sequence
                    if entities_result and isinstance(entities_result, list) and len(entities_result) > 0:
                        for e in entities_result:
                            try:
                                # Try tuple-like access
                                if isinstance(e, (list, tuple)) and len(e) > 1:
                                    e_type = e[1]
                                # Try dict-like access
                                elif hasattr(e, "get"):
                                    e_type = e.get("type")
                                else:
                                    continue
                                    
                                if e_type:
                                    type_counts[e_type] = type_counts.get(e_type, 0) + 1
                            except Exception:
                                continue
                        
                        # Only try to find max if we have types
                        if type_counts:
                            try:
                                primary_type = max(type_counts.items(), key=lambda x: x[1])[0]
                            except ValueError:
                                # Fallback if we can't determine max
                                if entity_types:
                                    primary_type = next(iter(entity_types))
                        elif entity_types:
                            # If counting failed but we have entity_types, just use the first one
                            primary_type = next(iter(entity_types))
            elif entity_types:
                # Only one type, use it
                primary_type = next(iter(entity_types))
                
            # Create consolidated entity
            consolidated_query = """
            MATCH (pc:MemoryContainer {name: $project_name, type: "ProjectContainer"})
            MERGE (ce:Entity {name: $primary_entity})
            ON CREATE SET ce.entityType = $entity_type,
                          ce.description = $description,
                          ce.confidence = $confidence,
                          ce.dateCreated = datetime(),
                          ce.isConsolidated = true,
                          ce.consolidationReason = $reason,
                          ce.consolidatedEntities = $entity_names
            ON MATCH SET ce.description = CASE WHEN ce.description = '' THEN $description ELSE ce.description END,
                         ce.confidence = $confidence,
                         ce.lastUpdated = datetime(),
                         ce.isConsolidated = true,
                         ce.consolidationReason = $reason,
                         ce.consolidatedEntities = $entity_names
            MERGE (pc)-[:CONTAINS]->(ce)
            RETURN ce.name as name
            """
            
            consolidation_params = {
                "project_name": project_name,
                "primary_entity": primary_entity,
                "entity_type": primary_type,
                "description": best_description,
                "confidence": max_confidence,
                "reason": reason or "Manual entity consolidation",
                "entity_names": entity_names
            }
            
            self._safe_execute_query(
                consolidated_query,
                parameters_=consolidation_params,
                database_=self.neo4j_database
            )
            
            # Copy relationships from all entities to the primary entity
            relationships_copied = 0
            for entity_name in entity_names:
                if entity_name == primary_entity:
                    continue
                    
                # Copy incoming relationships
                incoming_query = """
                MATCH (pc:MemoryContainer {name: $project_name, type: "ProjectContainer"})
                MATCH (pc)-[:CONTAINS]->(source:Entity)
                MATCH (pc)-[:CONTAINS]->(target:Entity {name: $entity_name})
                MATCH (source)-[r]->(target)
                WHERE source.name <> $primary_entity
                MATCH (pc)-[:CONTAINS]->(primary:Entity {name: $primary_entity})
                MERGE (source)-[r2:RELATIONSHIP {type: type(r)}]->(primary)
                ON CREATE SET r2 = properties(r),
                              r2.created = datetime(),
                              r2.consolidatedFrom = $entity_name
                RETURN count(r2) as count
                """
                
                incoming_result = self._safe_execute_query(
                    incoming_query,
                    parameters_={
                        "project_name": project_name,
                        "entity_name": entity_name,
                        "primary_entity": primary_entity
                    },
                    database_=self.neo4j_database
                )
                
                if incoming_result and incoming_result[0] and incoming_result[0][0]:
                    relationships_copied += int(incoming_result[0][0])
                
                # Copy outgoing relationships
                outgoing_query = """
                MATCH (pc:MemoryContainer {name: $project_name, type: "ProjectContainer"})
                MATCH (pc)-[:CONTAINS]->(source:Entity {name: $entity_name})
                MATCH (pc)-[:CONTAINS]->(target:Entity)
                MATCH (source)-[r]->(target)
                WHERE target.name <> $primary_entity
                MATCH (pc)-[:CONTAINS]->(primary:Entity {name: $primary_entity})
                MERGE (primary)-[r2:RELATIONSHIP {type: type(r)}]->(target)
                ON CREATE SET r2 = properties(r),
                              r2.created = datetime(),
                              r2.consolidatedFrom = $entity_name
                RETURN count(r2) as count
                """
                
                outgoing_result = self._safe_execute_query(
                    outgoing_query,
                    parameters_={
                        "project_name": project_name,
                        "entity_name": entity_name,
                        "primary_entity": primary_entity
                    },
                    database_=self.neo4j_database
                )
                
                if outgoing_result and outgoing_result[0] and outgoing_result[0][0]:
                    relationships_copied += int(outgoing_result[0][0])
            
            # Delete original entities if not preserving them
            deleted_entities = []
            if not preserve_originals:
                for entity_name in entity_names:
                    if entity_name == primary_entity:
                        continue
                        
                    delete_query = """
                    MATCH (pc:MemoryContainer {name: $project_name, type: "ProjectContainer"})
                    MATCH (pc)-[:CONTAINS]->(e:Entity {name: $entity_name})
                    WITH e, pc
                    OPTIONAL MATCH (e)-[r]-()
                    DELETE r, e
                    RETURN count(e) as deleted
                    """
                    
                    delete_result = self._safe_execute_query(
                        delete_query,
                        parameters_={"project_name": project_name, "entity_name": entity_name},
                        database_=self.neo4j_database
                    )
                    
                    if delete_result and delete_result[0] and delete_result[0][0] and int(delete_result[0][0]) > 0:
                        deleted_entities.append(entity_name)
            
            # Create a "consolidation event" to track the merge
            event_query = """
            MATCH (pc:MemoryContainer {name: $project_name, type: "ProjectContainer"})
            CREATE (e:Event {
                id: $event_id,
                type: "EntityConsolidation",
                timestamp: datetime(),
                description: $event_description,
                reason: $reason,
                source: $entity_names,
                target: $primary_entity
            })
            CREATE (pc)-[:CONTAINS]->(e)
            RETURN e.id as id
            """
            
            event_params = {
                "project_name": project_name,
                "event_id": generate_id(),
                "event_description": f"Consolidated {len(entity_names)} entities into '{primary_entity}'",
                "reason": reason or "Manual entity consolidation",
                "entity_names": entity_names,
                "primary_entity": primary_entity
            }
            
            self._safe_execute_query(
                event_query,
                parameters_=event_params,
                database_=self.neo4j_database
            )
            
            self.logger.info(f"Consolidated {len(entity_names)} entities into '{primary_entity}' in project '{project_name}'")
            
            return dict_to_json({
                "status": "success",
                "message": f"Successfully consolidated entities into '{primary_entity}'",
                "data": {
                    "project_name": project_name,
                    "primary_entity": primary_entity,
                    "entity_type": primary_type,
                    "consolidated_entities": entity_names,
                    "relationships_copied": relationships_copied,
                    "deleted_entities": deleted_entities,
                    "preserved": preserve_originals
                }
            })
            
        except Exception as e:
            error_info = extract_error(e)
            self.logger.error(f"Error consolidating entities: {error_info}")
            return dict_to_json({
                "status": "error",
                "message": f"Failed to consolidate entities: {error_info}"
            })