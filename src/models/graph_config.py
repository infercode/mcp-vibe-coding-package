#!/usr/bin/env python3
"""
Graph Configuration Settings

Pydantic models for graph database configuration settings.
"""

from typing import Optional, Dict, Any, List, Annotated, Literal, Union, cast
from enum import Enum
from datetime import datetime
from pydantic import (
    BaseModel, 
    Field, 
    field_validator, 
    model_validator, 
    ConfigDict,
    computed_field,
    BeforeValidator,
    AfterValidator,
    PlainSerializer,
    AliasChoices,
    model_serializer
)
import os
from functools import lru_cache


class DatabaseType(str, Enum):
    """Database types supported by the graph memory system."""
    NEO4J = "neo4j"
    MEMORY = "memory"
    MOCK = "mock"


class Neo4jConfig(BaseModel):
    """Neo4j database configuration."""
    uri: Annotated[
        str, 
        Field(
            description="URI of the Neo4j database",
            examples=["bolt://localhost:7687", "neo4j://neo4j.example.com:7687"]
        )
    ]
    username: Annotated[str, Field(description="Username for authentication")]
    password: Annotated[str, Field(description="Password for authentication")]
    database: Annotated[str, Field(default="neo4j", description="Name of the database to connect to")]
    
    model_config = ConfigDict(
        validate_assignment=True,
        frozen=False,
        json_schema_extra={
            "examples": [
                {
                    "uri": "bolt://localhost:7687",
                    "username": "neo4j",
                    "password": "password",
                    "database": "neo4j"
                }
            ]
        }
    )
    
    @field_validator('uri')
    @classmethod
    def valid_uri(cls, v: str) -> str:
        """Validate that the URI is correctly formatted."""
        if not v.startswith(('bolt://', 'neo4j://', 'neo4j+s://')):
            raise ValueError("Neo4j URI must start with bolt://, neo4j:// or neo4j+s://")
        return v
    
    @computed_field
    def connection_string(self) -> str:
        """Generate a complete connection string."""
        return f"{self.uri}/{self.database}"


class MemoryConfig(BaseModel):
    """In-memory database configuration."""
    persistence_file: Optional[str] = Field(None, description="File path for persisting memory data")
    load_on_startup: bool = Field(True, description="Whether to load persisted data on startup")
    
    model_config = ConfigDict(
        validate_assignment=True,
        json_schema_extra={
            "examples": [
                {
                    "persistence_file": "/tmp/graph_memory.json",
                    "load_on_startup": True
                }
            ]
        }
    )
    
    @computed_field
    def is_persistent(self) -> bool:
        """Check if this configuration uses persistence."""
        return self.persistence_file is not None


class MockConfig(BaseModel):
    """Mock database configuration for testing."""
    response_file: Optional[str] = Field(None, description="File path for mock responses")
    delay: Annotated[
        float, 
        Field(default=0.1, ge=0.0, le=10.0, description="Artificial delay in seconds to simulate database operations")
    ]
    
    model_config = ConfigDict(
        validate_assignment=True,
        json_schema_extra={
            "examples": [
                {
                    "response_file": "tests/mock_responses.json",
                    "delay": 0.1
                }
            ]
        }
    )


class EmbeddingConfig(BaseModel):
    """Embedding model configuration."""
    model_name: Annotated[
        str, 
        Field(
            default="text-embedding-ada-002", 
            description="Name of the embedding model"
        )
    ]
    provider: Annotated[
        str, 
        Field(
            default="openai", 
            description="Provider of the embedding model"
        )
    ]
    api_key: Optional[str] = Field(None, description="API key for the embedding provider")
    dimensions: Annotated[
        int, 
        Field(
            default=1536, 
            ge=1, 
            description="Dimensions of the embedding vectors"
        )
    ]
    
    model_config = ConfigDict(
        validate_assignment=True,
        json_schema_extra={
            "examples": [
                {
                    "model_name": "text-embedding-ada-002",
                    "provider": "openai",
                    "dimensions": 1536
                }
            ]
        }
    )
    
    @field_validator('provider')
    @classmethod
    def validate_provider(cls, v: str) -> str:
        """Validate the provider is supported."""
        allowed_providers = ["openai", "azure", "huggingface", "cohere", "vertex", "ollama", "mistral"]
        if v.lower() not in allowed_providers:
            raise ValueError(f"Provider '{v}' not supported. Use one of: {', '.join(allowed_providers)}")
        return v.lower()
    
    @field_validator('api_key', mode='before')
    @classmethod
    def get_api_key(cls, v: Optional[str], info) -> Optional[str]:
        """Get API key from environment if not provided."""
        if v is None:
            provider = info.data.get('provider', '').lower()
            if provider == 'openai':
                v = os.environ.get('OPENAI_API_KEY')
            elif provider == 'azure':
                v = os.environ.get('AZURE_OPENAI_API_KEY')
            elif provider == 'cohere':
                v = os.environ.get('COHERE_API_KEY')
        return v
    
    @computed_field
    def is_configured(self) -> bool:
        """Check if the embedding configuration is valid for use."""
        # A basic check - in a real system, this could be more sophisticated
        return self.provider in ["openai", "azure", "cohere"] and self.api_key is not None


class GraphConfig(BaseModel):
    """Main configuration for the graph database."""
    database_type: Annotated[
        DatabaseType, 
        Field(
            default=DatabaseType.MEMORY, 
            description="Type of database to use"
        )
    ]
    neo4j: Optional[Neo4jConfig] = Field(None, description="Neo4j configuration if using Neo4j")
    memory: Optional[MemoryConfig] = Field(None, description="Memory configuration if using in-memory database")
    mock: Optional[MockConfig] = Field(None, description="Mock configuration if using mock database")
    embedding: Optional[EmbeddingConfig] = Field(None, description="Embedding model configuration")
    default_client_id: Annotated[
        str, 
        Field(
            default="default", 
            min_length=1, 
            max_length=64,
            description="Default client ID for graph operations"
        )
    ]
    log_level: Annotated[
        str, 
        Field(
            default="INFO",
            pattern="^(DEBUG|INFO|WARNING|ERROR|CRITICAL)$",
            description="Logging level"
        )
    ]
    
    model_config = ConfigDict(
        validate_assignment=True,
        json_schema_extra={
            "examples": [
                {
                    "database_type": "neo4j",
                    "neo4j": {
                        "uri": "bolt://localhost:7687",
                        "username": "neo4j",
                        "password": "password"
                    },
                    "default_client_id": "app1",
                    "log_level": "INFO"
                }
            ]
        }
    )
    
    @model_validator(mode='before')
    @classmethod
    def validate_config(cls, data: Dict[str, Any]) -> Dict[str, Any]:
        """Validate that the appropriate configuration is provided for the selected database type."""
        if isinstance(data, dict):  # Ensure we're working with a dict
            db_type = data.get('database_type')
            
            # Convert string to enum if needed
            if isinstance(db_type, str):
                try:
                    db_type = DatabaseType(db_type.lower())
                    data['database_type'] = db_type
                except ValueError:
                    raise ValueError(f"Invalid database_type: {db_type}")
            
            # Check and create default configurations based on database type
            if db_type == DatabaseType.NEO4J and not data.get('neo4j'):
                raise ValueError("Neo4j configuration is required when database_type is neo4j")
            
            if db_type == DatabaseType.MEMORY and not data.get('memory'):
                # Create default memory config if not provided
                data['memory'] = {
                    "persistence_file": None,
                    "load_on_startup": True
                }
                
            if db_type == DatabaseType.MOCK and not data.get('mock'):
                # Create default mock config if not provided
                data['mock'] = {
                    "response_file": None,
                    "delay": 0.1
                }
                
        return data
    
    @computed_field
    def database_description(self) -> str:
        """Generate a human-readable description of the database configuration."""
        if self.database_type == DatabaseType.NEO4J and self.neo4j:
            return f"Neo4j database at {self.neo4j.uri}"
        elif self.database_type == DatabaseType.MEMORY and self.memory:
            persistence = f" (persisted to {self.memory.persistence_file})" if self.memory.persistence_file else " (non-persistent)"
            return f"In-memory database{persistence}"
        elif self.database_type == DatabaseType.MOCK and self.mock:
            return f"Mock database with {self.mock.delay}s delay"
        return f"Unknown database type: {self.database_type}"
    
    @computed_field
    def has_embeddings(self) -> bool:
        """Check if this configuration has embedding support."""
        return self.embedding is not None and (
            self.embedding.provider in ["openai", "azure", "cohere"] and 
            self.embedding.api_key is not None
        )
    
    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> 'GraphConfig':
        """Create a GraphConfig from a dictionary."""
        try:
            return cls.model_validate(config_dict)
        except Exception as e:
            raise ValueError(f"Invalid configuration: {str(e)}")
    
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
        if db_type.lower() == 'neo4j':
            config['neo4j'] = {
                'uri': os.environ.get('NEO4J_URI', 'bolt://localhost:7687'),
                'username': os.environ.get('NEO4J_USERNAME', 'neo4j'),
                'password': os.environ.get('NEO4J_PASSWORD', ''),
                'database': os.environ.get('NEO4J_DATABASE', 'neo4j')
            }
        elif db_type.lower() == 'memory':
            config['memory'] = {
                'persistence_file': os.environ.get('MEMORY_PERSISTENCE_FILE'),
                'load_on_startup': os.environ.get('MEMORY_LOAD_ON_STARTUP', 'true').lower() == 'true'
            }
        elif db_type.lower() == 'mock':
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
    
    # Add a serialization method with customization
    @model_serializer
    def serialize_model(self) -> Dict[str, Any]:
        """Custom serialization that can include or exclude computed fields."""
        data = self.model_dump(exclude_none=True)
        # Add computed fields
        data["database_description"] = self.database_description
        data["has_embeddings"] = self.has_embeddings
        return data


class ClientConfig(BaseModel):
    """Configuration for a specific client."""
    client_id: Annotated[
        str, 
        Field(
            min_length=1,
            max_length=64,
            description="Unique identifier for the client"
        )
    ]
    graph_config: GraphConfig = Field(..., description="Graph configuration for this client")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Additional metadata for the client")
    
    model_config = ConfigDict(
        arbitrary_types_allowed=True,
        validate_assignment=True,
        json_schema_extra={
            "examples": [
                {
                    "client_id": "web_client",
                    "metadata": {"created_at": "2023-01-01T00:00:00Z"}
                }
            ]
        }
    )
    
    @computed_field
    def created_timestamp(self) -> Optional[datetime]:
        """Extract created timestamp from metadata if available."""
        created_str = self.metadata.get("created_at")
        if isinstance(created_str, str):
            try:
                return datetime.fromisoformat(created_str.replace('Z', '+00:00'))
            except (ValueError, TypeError):
                pass
        return None


class ClientRegistry(BaseModel):
    """Registry of all client configurations."""
    clients: Dict[str, ClientConfig] = Field(default_factory=dict, description="Dictionary of client configurations")
    default_config: GraphConfig = Field(..., description="Default configuration for new clients")
    
    model_config = ConfigDict(
        validate_assignment=True,
    )
    
    @computed_field
    def client_count(self) -> int:
        """Get the number of registered clients."""
        return len(self.clients)
    
    def get_client_config(self, client_id: Optional[str] = None) -> ClientConfig:
        """Get the configuration for a specific client or create one if it doesn't exist."""
        if not client_id:
            client_id = self.default_config.default_client_id
            
        if client_id not in self.clients:
            # Create a new client config
            self.clients[client_id] = ClientConfig(
                client_id=client_id,
                graph_config=self.default_config,
                metadata={"created_at": datetime.now().isoformat()}
            )
            
        return self.clients[client_id]
    
    def update_client_config(self, client_id: str, config_update: Dict[str, Any]) -> ClientConfig:
        """Update the configuration for a specific client."""
        client_config = self.get_client_config(client_id)
        
        # Update the graph configuration
        if 'graph_config' in config_update:
            graph_config_dict = client_config.graph_config.model_dump()
            graph_config_dict.update(config_update['graph_config'])
            client_config.graph_config = GraphConfig.from_dict(graph_config_dict)
            
        # Update metadata
        if 'metadata' in config_update:
            if client_config.metadata is None:
                client_config.metadata = {}
            client_config.metadata.update(config_update['metadata'])
            
        self.clients[client_id] = client_config
        return client_config


@lru_cache()
def get_default_config() -> GraphConfig:
    """Get the default configuration with caching for performance."""
    return GraphConfig.from_env() 