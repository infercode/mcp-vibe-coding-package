import os
import time
import json
import datetime
from typing import Any, Dict, List, Optional, Tuple, cast, Union
from typing_extensions import LiteralString
from neo4j import GraphDatabase, Session, Transaction
import traceback

from src.logger import Logger, get_logger
from src.embedding_manager import LiteLLMEmbeddingManager
# Import neo4j query validation utilities
from src.models.neo4j_queries import CypherQuery, CypherParameters
from src.utils import dict_to_json

class BaseManager:
    """Base manager for Neo4j graph database connections and core functionality."""
    
    def __init__(self, logger: Optional[Logger] = None):
        """
        Initialize the base manager.
        
        Args:
            logger: Optional logger instance
        """
        self.logger = logger or get_logger()
        self.initialized = False
        self.neo4j_driver = None
        
        # Default project name for memory operations if none is provided
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
            embedding_config["additional_params"] = {"device": "cpu"}
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

    def initialize(self) -> bool:
        """
        Initialize the base manager and connect to the graph database.
        
        Returns:
            True if initialization was successful, False otherwise
        """
        if self.initialized:
            self.logger.info("Already initialized, skipping")
            return True

        try:
            self.logger.debug(f"Initializing with Neo4j URI: {self.neo4j_uri}, User: {self.neo4j_user}, DB: {self.neo4j_database}")
            
            # Make sure password is properly handled as a string
            # This ensures special characters are treated correctly
            if self.neo4j_password is not None:
                # Convert to string first to handle any data type
                password = str(self.neo4j_password)
                
                # Some special characters need special handling
                # Particularly problematic are: !, $, &, *, (, ), |, ;, <, >, ', ", \, etc.
                # Neo4j driver will handle the escaping internally for most cases,
                # but we need to make sure the password is passed as is without shell interpretation
                
                # Log password length but not content for security
                self.logger.debug(f"Password length: {len(password)} characters")
                if any(char in password for char in "!$&*();|<>'\"\\"):
                    self.logger.debug("Password contains special characters that may require escaping")
            else:
                password = ""
                
            self.logger.debug(f"Password length: {len(password)} characters")
            
            # Initialize Neo4j driver with connection pooling parameters
            self.neo4j_driver = GraphDatabase.driver(
                self.neo4j_uri,
                auth=(self.neo4j_user, password),
                # Connection pooling parameters
                max_connection_lifetime=30 * 60,  # 30 minutes
                max_connection_pool_size=50,      # Max 50 connections in the pool
                connection_acquisition_timeout=60, # 60 seconds timeout for acquiring a connection
                keep_alive=True                   # Keep connections alive
            )
            
            self.logger.debug("Driver initialized, testing connection")
            
            # Test the Neo4j connection with retry
            self._test_neo4j_connection_with_retry(max_retries=3)
            
            self.logger.debug("Connection successful, checking embeddings config")
            
            # Setup vector index if embeddings are enabled
            if self.embedding_enabled:
                self.logger.debug(f"Setting up vector index with provider: {self.embedder_provider}")
                self._setup_vector_index()
                self.logger.info(f"Memory graph manager initialized successfully with {self.embedder_provider} embedder")
            else:
                self.logger.info("Memory graph manager initialized successfully without embeddings")
                
            self.initialized = True
            self.logger.debug("Initialization complete")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to initialize memory manager: {str(e)}")
            return False

    def _test_neo4j_connection_with_retry(self, max_retries=3, initial_delay=1.0):
        """Test Neo4j connection with exponential backoff retry mechanism."""
        if not self.neo4j_driver:
            self.logger.debug("Neo4j driver not initialized, skipping connection test")
            return
            
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
            self.logger.debug(f"Testing Neo4j connection to {self.neo4j_uri} with user {self.neo4j_user} and database {self.neo4j_database}")
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
            
        except Exception as e:
            error_message = str(e)
            self.logger.debug(f"Raw connection error: {error_message}")
            
            if "Authentication failed" in error_message:
                self.logger.error(f"Neo4j authentication failed. Please check your credentials: {error_message}")
            elif "Connection refused" in error_message:
                self.logger.error(f"Neo4j connection refused. Is the database running? {error_message}")
            elif "Failed to establish connection" in error_message:
                self.logger.error(f"Failed to establish Neo4j connection. Check network and server: {error_message}")
            elif "AuthenticationRateLimit" in error_message:
                self.logger.error(f"Neo4j authentication rate limit reached. Wait a few seconds and try again: {error_message}")
            else:
                self.logger.error(f"Neo4j connection test failed: {error_message}")
            
            raise

    def close(self) -> None:
        """Close the Neo4j driver connection."""
        if self.neo4j_driver:
            self.neo4j_driver.close()
            self.logger.info("Neo4j driver closed")

    def safe_execute_query(
        self, query: str, parameters: Optional[Dict[str, Any]] = None, database: Optional[str] = None
    ) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
        """
        Safely execute a Neo4j query with proper typing and parameter handling.
        
        Args:
            query: The Cypher query to execute
            parameters: Optional parameters for the query
            database: Optional database name to use
            
        Returns:
            Tuple containing (records, summary)
        """
        if not self.neo4j_driver:
            self.logger.warn("Neo4j driver not initialized, skipping query execution")
            return [], {}
            
        try:
            # Cast query to LiteralString for Neo4j driver type checking
            query_literal = cast(LiteralString, query)
            
            # Prepare parameters - handle float conversion if needed
            params = {}
            if parameters:
                for key, value in parameters.items():
                    # Neo4j Python driver has issues with some float parameters in 4.4+
                    # Convert specific float parameters to strings where necessary
                    if isinstance(value, float) and not isinstance(value, bool):
                        params[key] = str(value)
                    else:
                        params[key] = value
            
            # Execute query with or without database specified
            if database:
                result = self.neo4j_driver.execute_query(
                    query_literal,
                    parameters_=params,
                    database_=database
                )
            else:
                result = self.neo4j_driver.execute_query(
                    query_literal,
                    parameters_=params,
                    database_=self.neo4j_database
                )
            
            # Extract records and summary
            records = result[0] if result and len(result) > 0 else []
            summary = result[1] if result and len(result) > 1 else {}
            
            # Convert Neo4j Record objects to dictionaries
            records_dict = [dict(record) for record in records]
            
            # Convert ResultSummary to dictionary if needed
            summary_dict = {}
            if summary and not isinstance(summary, dict):
                # Extract common attributes from ResultSummary
                summary_dict = {
                    "query_type": getattr(summary, "query_type", None),
                    "database": getattr(summary, "database", None),
                    "server_info": str(getattr(summary, "server", ""))
                }
                
                # Handle counters separately
                if hasattr(summary, "counters"):
                    counter_dict = {}
                    for k in dir(summary.counters):
                        if not k.startswith('_') and not callable(getattr(summary.counters, k)):
                            counter_dict[k] = getattr(summary.counters, k)
                    summary_dict["counters"] = counter_dict
            else:
                summary_dict = summary
            
            return records_dict, summary_dict
        except Exception as e:
            self.logger.error(f"Error executing query: {str(e)}")
            self.logger.error(f"Query: {query}")
            self.logger.error(f"Parameters: {parameters}")
            raise

    def ensure_initialized(self) -> None:
        """Ensure the base manager is initialized."""
        if not self.initialized:
            self.initialize()

    def generate_embedding(self, text: str) -> Optional[List[float]]:
        """
        Generate an embedding for the given text.
        
        Args:
            text: The text to generate an embedding for
            
        Returns:
            A list of floats representing the embedding, or None if embedding fails
        """
        if not self.embedding_enabled:
            return None
            
        return self.embedding_manager.generate_embedding(text)
        
    # Neo4j Query Validation Methods
    
    def execute_read_query(self, query: str, parameters: Optional[Dict[str, Any]] = None, 
                         database: Optional[str] = None) -> List[Dict[str, Any]]:
        """
        Execute a read-only query with validation.
        
        Args:
            query: The Cypher query to execute
            parameters: Optional parameters for the query
            database: Optional database name to use
            
        Returns:
            List of record dictionaries
        """
        records, _ = self.safe_execute_query(query, parameters, database)
        return records
        
    def execute_write_query(self, query: str, parameters: Optional[Dict[str, Any]] = None,
                          database: Optional[str] = None) -> List[Dict[str, Any]]:
        """
        Execute a write query with validation.
        
        Args:
            query: The Cypher query to execute
            parameters: Optional parameters for the query
            database: Optional database name to use
            
        Returns:
            List of record dictionaries
        """
        records, _ = self.safe_execute_query(query, parameters, database)
        return records
    
    def safe_execute_read_query(self, query: str, parameters: Optional[Dict[str, Any]] = None,
                              database: Optional[str] = None) -> List[Dict[str, Any]]:
        """
        Validate and execute a read-only query.
        
        Args:
            query: The Cypher query to execute
            parameters: Optional parameters for the query
            database: Optional database name to use
            
        Returns:
            List of record dictionaries
            
        Raises:
            ValueError: If query is invalid or contains destructive operations
        """
        try:
            # Manually check for destructive operations - these patterns already use word boundaries
            destructive_patterns = [
                r"(?i)\b(CREATE|DELETE|REMOVE|DROP|SET)\b",
                r"(?i)\b(MERGE|DETACH DELETE)\b",
                r"(?i)\b(CREATE|DROP) (INDEX|CONSTRAINT)\b",
                r"(?i)\.drop\(.*\)"
            ]
            
            import re
            for pattern in destructive_patterns:
                if re.search(pattern, query):
                    raise ValueError(f"Destructive operation detected in read-only query: {pattern}")
                    
            # Sanitize parameters manually
            sanitized_params = parameters
            if parameters:
                # Convert dates and complex objects to strings
                sanitized_params = {}
                for key, value in parameters.items():
                    if isinstance(value, (datetime.datetime, datetime.date)):
                        sanitized_params[key] = str(value)
                    else:
                        sanitized_params[key] = value
                        
            # Create query model
            params_model = None
            if sanitized_params:
                params_model = CypherParameters(parameters=sanitized_params)
                
            validated_query = CypherQuery(
                query=query,
                parameters=params_model,
                read_only=True,
                database=database
            )
            
            # Execute the validated query
            records, _ = self._execute_validated_query(validated_query)
            return records
            
        except ValueError as e:
            self.logger.error(f"Query validation error: {e}")
            raise
        except Exception as e:
            self.logger.error(f"Error executing read query: {e}")
            raise
    
    def safe_execute_write_query(self, query: str, parameters: Optional[Dict[str, Any]] = None,
                               database: Optional[str] = None) -> List[Dict[str, Any]]:
        """
        Validate and execute a write query.
        
        Args:
            query: The Cypher query to execute
            parameters: Optional parameters for the query
            database: Optional database name to use
            
        Returns:
            List of record dictionaries
            
        Raises:
            ValueError: If query is invalid
        """
        try:
            # Sanitize parameters manually
            sanitized_params = parameters
            if parameters:
                # Convert dates and complex objects to strings
                sanitized_params = {}
                for key, value in parameters.items():
                    if isinstance(value, (datetime.datetime, datetime.date)):
                        sanitized_params[key] = str(value)
                    else:
                        sanitized_params[key] = value
                        
            # Create query model
            params_model = None
            if sanitized_params:
                params_model = CypherParameters(parameters=sanitized_params)
                
            validated_query = CypherQuery(
                query=query,
                parameters=params_model,
                read_only=False,
                database=database
            )
            
            # Execute the validated query
            records, _ = self._execute_validated_query(validated_query)
            return records
            
        except ValueError as e:
            self.logger.error(f"Query validation error: {e}")
            raise
        except Exception as e:
            self.logger.error(f"Error executing write query: {e}")
            raise
    
    def _execute_validated_query(self, query_model: CypherQuery) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
        """
        Execute a validated CypherQuery.
        
        Args:
            query_model: Validated CypherQuery model
            
        Returns:
            Tuple of (records, summary)
            
        Raises:
            Exception: If execution fails
        """
        # Extract parameters dictionary or empty dict if None
        parameters = {}
        if query_model.parameters:
            parameters = query_model.parameters.parameters
        
        # Execute query with the appropriate database
        db = query_model.database or self.neo4j_database
        return self.safe_execute_query(query_model.query, parameters, db)

    def _handle_error(self, e: Exception, operation: str) -> str:
        """Handle exceptions and return a formatted error response."""
        error_msg = f"Error during {operation}: {str(e)}"
        self.logger.error(error_msg)
        return dict_to_json({
            "status": "error",
            "message": error_msg
        }) 