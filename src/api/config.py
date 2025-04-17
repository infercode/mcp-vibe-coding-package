from pydantic_settings import BaseSettings
from typing import Optional
from functools import lru_cache

class Settings(BaseSettings):
    """Application settings."""
    # API Settings
    api_title: str = "Memory System API"
    api_description: str = "API endpoints for interacting with the graph-based memory system"
    api_version: str = "1.0.0"
    
    # Server Settings
    host: str = "0.0.0.0"
    port: int = 8000
    debug: bool = False
    use_sse: bool = True
    log_level: str = "INFO"
    
    # Client Settings
    client_id: str = "api-client"  # Default client ID for API operations
    use_project_as_client: bool = True  # Use project ID as client ID for isolation
    
    # Neo4j Settings
    neo4j_uri: str = "bolt://localhost:7687"
    neo4j_user: str = "neo4j"
    neo4j_password: str = "password"
    neo4j_database: str = "neo4j"
    
    # Embedding Settings
    embedding_provider: str = "openai"
    embedding_model: Optional[str] = None
    embedding_api_key: Optional[str] = None
    embedding_dimensions: Optional[int] = None
    embedder_provider: Optional[str] = None
    openai_api_key: Optional[str] = None
    
    class Config:
        env_file = ".env"
        case_sensitive = False

@lru_cache()
def get_settings() -> Settings:
    """Get cached settings instance."""
    return Settings() 