"""
Settings Model for MCP Graph Memory

This module provides Pydantic models for application configuration,
leveraging Pydantic's settings management capabilities.
"""

import os
from typing import Dict, List, Optional, Any, Union
from pydantic import BaseModel, Field, field_validator, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Neo4jSettings(BaseModel):
    """Neo4j database connection settings."""
    uri: str = Field("bolt://localhost:7687", description="Neo4j connection URI")
    username: str = Field("neo4j", description="Neo4j username")
    password: str = Field("password", description="Neo4j password")
    database: str = Field("neo4j", description="Neo4j database name")
    max_connection_lifetime: int = Field(3600, description="Maximum lifetime of a connection in seconds")
    max_connection_pool_size: int = Field(50, description="Maximum number of connections in the pool")
    connection_timeout: int = Field(30, description="Connection timeout in seconds")


class VectorSettings(BaseModel):
    """Vector embedding settings."""
    enabled: bool = Field(False, description="Whether vector embeddings are enabled")
    provider: str = Field("openai", description="Embedding provider (openai, azure, huggingface, etc.)")
    model: str = Field("text-embedding-ada-002", description="Embedding model name")
    dimensions: int = Field(1536, description="Embedding dimensions")
    api_key: Optional[str] = Field(None, description="API key for the embedding provider")
    api_base: Optional[str] = Field(None, description="API base URL for the embedding provider")
    index_name: str = Field("graph_embeddings", description="Neo4j index name for vector search")
    
    @field_validator('provider')
    @classmethod
    def validate_provider(cls, v: str) -> str:
        """Validate that the provider is one of the allowed types."""
        allowed_providers = [
            'openai', 'azure', 'huggingface', 'vertex', 'gemini', 
            'mistral', 'ollama', 'lmstudio', 'none'
        ]
        if v.lower() not in allowed_providers:
            raise ValueError(f"Provider must be one of {allowed_providers}")
        return v.lower()


class LoggingSettings(BaseModel):
    """Logging configuration."""
    level: str = Field("INFO", description="Logging level")
    format: str = Field(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        description="Log message format"
    )
    file: Optional[str] = Field(None, description="Log file path (None for console logging)")


def create_neo4j_settings() -> Neo4jSettings:
    """Factory function to create Neo4j settings with defaults."""
    return Neo4jSettings(
        uri="bolt://localhost:7687",
        username="neo4j",
        password="password",
        database="neo4j",
        max_connection_lifetime=3600,
        max_connection_pool_size=50,
        connection_timeout=30
    )


def create_vector_settings() -> VectorSettings:
    """Factory function to create vector settings with defaults."""
    return VectorSettings(
        enabled=False,
        provider="openai",
        model="text-embedding-ada-002",
        dimensions=1536,
        api_key=None,
        api_base=None,
        index_name="graph_embeddings"
    )


def create_logging_settings() -> LoggingSettings:
    """Factory function to create logging settings with defaults."""
    return LoggingSettings(
        level="INFO",
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        file=None
    )


class GraphMemorySettings(BaseSettings):
    """Main settings model for MCP Graph Memory."""
    neo4j: Neo4jSettings = Field(default_factory=create_neo4j_settings, description="Neo4j database settings")
    vector: VectorSettings = Field(default_factory=create_vector_settings, description="Vector embedding settings")
    logging: LoggingSettings = Field(default_factory=create_logging_settings, description="Logging settings")
    default_project_name: str = Field("default", description="Default project name")
    mode: str = Field("sse", description="Mode (sse or stdio)")
    client_timeout: int = Field(3600, description="Client session timeout in seconds")
    cleanup_interval: int = Field(300, description="Session cleanup interval in seconds")
    config_file_path: Optional[str] = Field(None, description="Path to the config file")
    
    # Nested model settings
    model_config = SettingsConfigDict(
        # Allow environment variables to override config
        env_nested_delimiter="__",
        # Extra config options
        validate_assignment=True,
        extra="ignore",
        env_file=".env",
        env_file_encoding="utf-8"
    )
    
    @model_validator(mode='after')
    def check_environment_overrides(cls, values):
        """Apply environment variable overrides for specific fields."""
        # Neo4j settings from env vars
        if os.environ.get("NEO4J_URI"):
            values.neo4j.uri = os.environ.get("NEO4J_URI")
        if os.environ.get("NEO4J_USERNAME"):
            values.neo4j.username = os.environ.get("NEO4J_USERNAME")
        if os.environ.get("NEO4J_PASSWORD"):
            values.neo4j.password = os.environ.get("NEO4J_PASSWORD")
        if os.environ.get("NEO4J_DATABASE"):
            values.neo4j.database = os.environ.get("NEO4J_DATABASE")
            
        # Vector settings from env vars
        if os.environ.get("EMBEDDER_PROVIDER"):
            provider = os.environ.get("EMBEDDER_PROVIDER", "").lower()
            values.vector.enabled = provider != "none"
            if provider != "none":
                values.vector.provider = provider
                
        if os.environ.get("EMBEDDER_API_KEY"):
            values.vector.api_key = os.environ.get("EMBEDDER_API_KEY")
        if os.environ.get("EMBEDDER_MODEL"):
            values.vector.model = os.environ.get("EMBEDDER_MODEL")
            
        # Mode settings
        if any(os.environ.get(var, "").lower() in ("1", "true", "yes") 
               for var in ["MCP_STDIO_MODE", "STDIO_MODE"]):
            values.mode = "stdio"
            
        return values
    
    def load_from_file(self, file_path: Optional[str] = None) -> "GraphMemorySettings":
        """Load settings from a JSON file."""
        import json
        path = file_path or self.config_file_path
        if not path:
            return self
            
        try:
            with open(path, "r") as f:
                config_data = json.load(f)
            
            # Apply configuration from file
            for key, value in config_data.items():
                if hasattr(self, key):
                    setattr(self, key, value)
                    
            return self
        except Exception as e:
            print(f"Error loading config from {path}: {str(e)}")
            return self
            

def get_settings() -> GraphMemorySettings:
    """Get the application settings, with environment and file overrides applied."""
    # Create with all default values
    settings = GraphMemorySettings(
        neo4j=create_neo4j_settings(),
        vector=create_vector_settings(),
        logging=create_logging_settings(),
        default_project_name="default",
        mode="sse",
        client_timeout=3600,
        cleanup_interval=300,
        config_file_path=None
    )
    
    # Try to load from the default config file if it exists
    default_config_path = os.environ.get("CONFIG_FILE_PATH", "mcp_unified_config_default.json")
    if os.path.exists(default_config_path):
        settings.config_file_path = default_config_path
        settings = settings.load_from_file()
        
    return settings 