#!/usr/bin/env python3
"""
Graph Configuration Settings

Pydantic models for graph database configuration settings.
"""

from typing import Optional, Dict, Any, List, cast
from enum import Enum
from datetime import datetime
from pydantic import BaseModel, Field, validator, root_validator
import os

class DatabaseType(str, Enum):
    """Database types supported by the graph memory system."""
    NEO4J = "neo4j"
    MEMORY = "memory"
    MOCK = "mock"


class Neo4jConfig(BaseModel):
    """Neo4j database configuration."""
    uri: str = Field(..., description="URI of the Neo4j database")
    username: str = Field(..., description="Username for authentication")
    password: str = Field(..., description="Password for authentication")
    database: str = Field("neo4j", description="Name of the database to connect to")
    
    @validator('uri')
    def valid_uri(cls, v: str) -> str:
        """Validate that the URI is correctly formatted."""
        if not v.startswith(('bolt://', 'neo4j://', 'neo4j+s://')):
            raise ValueError("Neo4j URI must start with bolt://, neo4j:// or neo4j+s://")
        return v


class MemoryConfig(BaseModel):
    """In-memory database configuration."""
    persistence_file: Optional[str] = Field(None, description="File path for persisting memory data")
    load_on_startup: bool = Field(True, description="Whether to load persisted data on startup")


class MockConfig(BaseModel):
    """Mock database configuration for testing."""
    response_file: Optional[str] = Field(None, description="File path for mock responses")
    delay: float = Field(0.1, description="Artificial delay in seconds to simulate database operations")


class EmbeddingConfig(BaseModel):
    """Embedding model configuration."""
    model_name: str = Field("text-embedding-ada-002", description="Name of the embedding model")
    provider: str = Field("openai", description="Provider of the embedding model")
    api_key: Optional[str] = Field(None, description="API key for the embedding provider")
    dimensions: int = Field(1536, description="Dimensions of the embedding vectors")
    
    @validator('api_key', pre=True)
    def get_api_key(cls, v: Optional[str], values: Dict[str, Any]) -> Optional[str]:
        """Get API key from environment if not provided."""
        if v is None and values.get('provider') == 'openai':
            v = os.environ.get('OPENAI_API_KEY')
        return v


class GraphConfig(BaseModel):
    """Main configuration for the graph database."""
    database_type: DatabaseType = Field(DatabaseType.MEMORY, description="Type of database to use")
    neo4j: Optional[Neo4jConfig] = Field(None, description="Neo4j configuration if using Neo4j")
    memory: Optional[MemoryConfig] = Field(None, description="Memory configuration if using in-memory database")
    mock: Optional[MockConfig] = Field(None, description="Mock configuration if using mock database")
    embedding: Optional[EmbeddingConfig] = Field(None, description="Embedding model configuration")
    default_client_id: str = Field("default", description="Default client ID for graph operations")
    log_level: str = Field("INFO", description="Logging level")
    
    @root_validator(pre=True)
    def validate_config(cls, values: Dict[str, Any]) -> Dict[str, Any]:
        """Validate that the appropriate configuration is provided for the selected database type."""
        db_type = values.get('database_type')
        if db_type == DatabaseType.NEO4J and not values.get('neo4j'):
            raise ValueError("Neo4j configuration is required when database_type is neo4j")
        if db_type == DatabaseType.MEMORY and not values.get('memory'):
            # Create default memory config if not provided
            values['memory'] = {
                "persistence_file": None,
                "load_on_startup": True
            }
        if db_type == DatabaseType.MOCK and not values.get('mock'):
            # Create default mock config if not provided
            values['mock'] = {
                "response_file": None,
                "delay": 0.1
            }
        return values
    
    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> 'GraphConfig':
        """Create a GraphConfig from a dictionary."""
        return cls(**config_dict)
    
    @classmethod
    def from_env(cls) -> 'GraphConfig':
        """Create a GraphConfig from environment variables."""
        db_type = os.environ.get('GRAPH_DB_TYPE', 'memory')
        
        config: Dict[str, Any] = {
            'database_type': db_type,
            'log_level': os.environ.get('LOG_LEVEL', 'INFO'),
            'default_client_id': os.environ.get('DEFAULT_CLIENT_ID', 'default')
        }
        
        # Add the database-specific configuration
        if db_type == 'neo4j':
            config['neo4j'] = {
                'uri': os.environ.get('NEO4J_URI', 'bolt://localhost:7687'),
                'username': os.environ.get('NEO4J_USERNAME', 'neo4j'),
                'password': os.environ.get('NEO4J_PASSWORD', ''),
                'database': os.environ.get('NEO4J_DATABASE', 'neo4j')
            }
        elif db_type == 'memory':
            config['memory'] = {
                'persistence_file': os.environ.get('MEMORY_PERSISTENCE_FILE'),
                'load_on_startup': os.environ.get('MEMORY_LOAD_ON_STARTUP', 'true').lower() == 'true'
            }
        elif db_type == 'mock':
            config['mock'] = {
                'response_file': os.environ.get('MOCK_RESPONSE_FILE'),
                'delay': float(os.environ.get('MOCK_DELAY', '0.1'))
            }
        
        # Add embedding configuration if needed
        if os.environ.get('USE_EMBEDDINGS', 'false').lower() == 'true':
            config['embedding'] = {
                'model_name': os.environ.get('EMBEDDING_MODEL', 'text-embedding-ada-002'),
                'provider': os.environ.get('EMBEDDING_PROVIDER', 'openai'),
                'api_key': os.environ.get('OPENAI_API_KEY'),
                'dimensions': int(os.environ.get('EMBEDDING_DIMENSIONS', '1536'))
            }
        
        return cls.from_dict(config)


class ClientConfig(BaseModel):
    """Configuration for a specific client."""
    client_id: str = Field(..., description="Unique identifier for the client")
    graph_config: GraphConfig = Field(..., description="Graph configuration for this client")
    metadata: Optional[Dict[str, Any]] = Field(None, description="Additional metadata for the client")
    
    # TODO: Update to model_config when we can ensure compatibility
    # @deprecated in Pydantic v2, will be removed in v3
    class Config:
        arbitrary_types_allowed = True


class ClientRegistry(BaseModel):
    """Registry of all client configurations."""
    clients: Dict[str, ClientConfig] = Field(default_factory=dict, description="Dictionary of client configurations")
    default_config: GraphConfig = Field(..., description="Default configuration for new clients")
    
    def get_client_config(self, client_id: Optional[str] = None) -> ClientConfig:
        """Get the configuration for a specific client or create one if it doesn't exist."""
        if not client_id:
            client_id = self.default_config.default_client_id
            
        if client_id not in self.clients:
            # Create a new client config
            self.clients[client_id] = ClientConfig(
                client_id=client_id,
                graph_config=self.default_config,
                metadata={"created_at": str(datetime.now())}
            )
            
        return self.clients[client_id]
    
    def update_client_config(self, client_id: str, config_update: Dict[str, Any]) -> ClientConfig:
        """Update the configuration for a specific client."""
        client_config = self.get_client_config(client_id)
        
        # Update the graph configuration
        if 'graph_config' in config_update:
            graph_config_dict = client_config.graph_config.dict()
            graph_config_dict.update(config_update['graph_config'])
            client_config.graph_config = GraphConfig.from_dict(graph_config_dict)
            
        # Update metadata
        if 'metadata' in config_update:
            if client_config.metadata is None:
                client_config.metadata = {}
            client_config.metadata.update(config_update['metadata'])
            
        self.clients[client_id] = client_config
        return client_config 