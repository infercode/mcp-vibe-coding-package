import os
import json
import time
import traceback
from typing import List, Dict, Any, Optional, Union, cast, TYPE_CHECKING
from neo4j import GraphDatabase

from src.types import Entity, KnowledgeGraph, Observation, Relation
from src.utils import dict_to_json, extract_error, generate_id
from src.embedding_manager import LiteLLMEmbeddingManager
from src.logger import Logger, get_logger


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
                
                # Use basic text matching in Cypher
                text_query = """
                MATCH (e:Entity)
                WHERE e.name CONTAINS $search_text OR e.entityType CONTAINS $search_text
                WITH e
                MATCH (e)<-[:DESCRIBES]-(o:Observation)
                WITH e, collect(o.content) as observations
                RETURN e.name as name, e.entityType as entityType, observations
                LIMIT $limit
                """
                
                text_results = self.neo4j_driver.execute_query(
                    text_query,
                    search_text=query,
                    limit=limit,
                    database_=self.neo4j_database
                )
                
                # Process text search results
                records = text_results[0] if text_results and len(text_results) > 0 else []
                for record in records:
                    entity_name = record["name"]
                    entity_type = record["entityType"]
                    observations = record["observations"]
                    
                    entity = {
                        "name": entity_name,
                        "entityType": entity_type or "Unknown",
                        "observations": observations,
                        "project": project_name
                    }
                    
                    formatted_result = {
                        "entity": entity,
                        "score": 1.0  # Default score for text search
                    }
                    formatted_results.append(formatted_result)
            
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