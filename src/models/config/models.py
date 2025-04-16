#!/usr/bin/env python3
"""
Configuration Pydantic Models

This module provides Pydantic models for configuration-related operations
such as managing unified configuration, Neo4j settings, and embedding settings.
"""

from typing import Dict, List, Optional, Any, Union, Literal, Annotated
from datetime import datetime
from pydantic import (
    BaseModel, 
    Field, 
    field_validator, 
    model_validator, 
    ConfigDict,
    computed_field
)
import json
import re

# Configuration Models

class Neo4jConfig(BaseModel):
    """Model for Neo4j database configuration."""
    uri: Annotated[str, Field(description="Neo4j database URI")]
    username: Annotated[str, Field(description="Neo4j database username")]
    password: Annotated[str, Field(description="Neo4j database password")]
    database: Annotated[str, Field(default="neo4j", description="Neo4j database name")]
    
    model_config = ConfigDict(
        validate_assignment=True,
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
    def validate_uri_format(cls, v: str) -> str:
        """Validate that the URI has the correct format."""
        if not v.startswith(('bolt://', 'neo4j://', 'neo4j+s://')):
            raise ValueError("Neo4j URI must start with 'bolt://', 'neo4j://', or 'neo4j+s://'")
        return v
    
    @computed_field
    def connection_string(self) -> str:
        """Generate a complete connection string."""
        return f"{self.uri}/{self.database}"


class EmbeddingConfig(BaseModel):
    """Model for embedding service configuration."""
    enabled: Annotated[bool, Field(default=False, description="Whether embeddings are enabled")]
    provider: Annotated[str, Field(default="openai", description="Embedding service provider (openai, cohere, etc.)")]
    model: Annotated[str, Field(default="text-embedding-ada-002", description="Embedding model to use")]
    api_key: Annotated[Optional[str], Field(default=None, description="API key for the embedding service")]
    api_base: Annotated[Optional[str], Field(default=None, description="Base URL for the embedding service API")]
    dimensions: Annotated[int, Field(default=1536, description="Dimensions of the embedding vectors")]
    additional_params: Annotated[Optional[Dict[str, Any]], Field(default=None, description="Additional parameters for the embedding service")]
    
    model_config = ConfigDict(
        validate_assignment=True,
        json_schema_extra={
            "examples": [
                {
                    "enabled": True,
                    "provider": "openai",
                    "model": "text-embedding-ada-002",
                    "dimensions": 1536
                }
            ]
        }
    )
    
    @model_validator(mode='after')
    @classmethod
    def validate_provider_config(cls, model):
        """Validate that the necessary fields are provided for the selected provider."""
        if not model.enabled:
            return model
            
        if model.provider == "openai":
            if not model.api_key:
                raise ValueError("API key is required for OpenAI embeddings")
                
        elif model.provider == "cohere":
            if not model.api_key:
                raise ValueError("API key is required for Cohere embeddings")
                
        return model
    
    @property
    def is_configured(self) -> bool:
        """Check if the embedding configuration is valid for use."""
        if not self.enabled:
            return False
        return (self.provider in ["openai", "azure", "cohere"] and 
                self.api_key is not None and 
                self.dimensions > 0)


class UnifiedConfig(BaseModel):
    """Model for unified configuration."""
    project_name: Annotated[str, Field(default="default", description="Name of the project")]
    client_id: Annotated[str, Field(description="Client ID for this configuration")]
    timestamp: Annotated[str, Field(default_factory=lambda: datetime.now().isoformat(), description="Timestamp of the configuration")]
    neo4j: Annotated[Neo4jConfig, Field(description="Neo4j database configuration")]
    embedding: Annotated[EmbeddingConfig, Field(description="Embedding service configuration")]
    additional_settings: Annotated[Optional[Dict[str, Any]], Field(default=None, description="Additional configuration settings")]
    
    model_config = ConfigDict(
        validate_assignment=True,
        json_schema_extra={
            "examples": [
                {
                    "project_name": "my_project",
                    "client_id": "client_123",
                    "neo4j": {
                        "uri": "bolt://localhost:7687",
                        "username": "neo4j",
                        "password": "password"
                    },
                    "embedding": {
                        "enabled": True,
                        "provider": "openai",
                        "model": "text-embedding-ada-002"
                    }
                }
            ]
        }
    )
    
    @property
    def has_embeddings(self) -> bool:
        """Check if this configuration has embedding support."""
        if not self.embedding:
            return False
        return self.embedding.is_configured


class ConfigUpdate(BaseModel):
    """Model for configuration updates."""
    project_name: Annotated[Optional[str], Field(default=None, description="Name of the project")]
    client_id: Annotated[Optional[str], Field(default=None, description="Client ID for this configuration")]
    neo4j: Annotated[Optional[Dict[str, Any]], Field(default=None, description="Neo4j database configuration updates")]
    embedding: Annotated[Optional[Dict[str, Any]], Field(default=None, description="Embedding service configuration updates")]
    additional_settings: Annotated[Optional[Dict[str, Any]], Field(default=None, description="Additional configuration settings updates")]
    
    model_config = ConfigDict(
        validate_assignment=True,
        json_schema_extra={
            "examples": [
                {
                    "project_name": "updated_project",
                    "embedding": {
                        "enabled": True,
                        "model": "text-embedding-3-small"
                    }
                }
            ]
        }
    )
    
    @computed_field
    def has_updates(self) -> bool:
        """Check if the update has any non-None fields."""
        return any([
            self.project_name is not None,
            self.client_id is not None,
            self.neo4j is not None,
            self.embedding is not None,
            self.additional_settings is not None
        ])


# Status Models

class Neo4jStatus(BaseModel):
    """Model for Neo4j connection status."""
    connected: Annotated[bool, Field(description="Whether the connection to Neo4j is established")]
    version: Annotated[Optional[str], Field(default=None, description="Neo4j server version")]
    address: Annotated[Optional[str], Field(default=None, description="Neo4j server address")]
    message: Annotated[str, Field(description="Status message")]
    timestamp: Annotated[str, Field(default_factory=lambda: datetime.now().isoformat(), description="Timestamp of the status check")]
    
    model_config = ConfigDict(
        validate_assignment=True
    )


class EmbeddingStatus(BaseModel):
    """Model for embedding API status."""
    available: Annotated[bool, Field(description="Whether the embedding API is available")]
    provider: Annotated[str, Field(description="Embedding provider name")]
    model: Annotated[str, Field(description="Embedding model name")]
    dimensions: Annotated[int, Field(description="Number of dimensions in the embeddings")]
    message: Annotated[str, Field(description="Status message")]
    timestamp: Annotated[str, Field(default_factory=lambda: datetime.now().isoformat(), description="Timestamp of the status check")]
    
    model_config = ConfigDict(
        validate_assignment=True
    )


class MemoryStatus(BaseModel):
    """Model for memory system status."""
    operational: Annotated[bool, Field(description="Whether the memory system is operational")]
    neo4j: Annotated[Neo4jStatus, Field(description="Neo4j connection status")]
    embedding: Annotated[EmbeddingStatus, Field(description="Embedding API status")]
    message: Annotated[str, Field(description="Status message")]
    timestamp: Annotated[str, Field(default_factory=lambda: datetime.now().isoformat(), description="Timestamp of the status check")]
    
    model_config = ConfigDict(
        validate_assignment=True
    )
    
    @computed_field
    def components_status(self) -> Dict[str, bool]:
        """Get the status of all components."""
        return {
            "neo4j": self.neo4j.connected,
            "embedding": self.embedding.available,
            "overall": self.operational
        }


# Response Models

class ErrorDetails(BaseModel):
    """Model for error details."""
    code: Annotated[str, Field(description="Error code")]
    timestamp: Annotated[str, Field(default_factory=lambda: datetime.now().isoformat(), description="Timestamp of the error")]
    details: Annotated[Optional[Dict[str, Any]], Field(default=None, description="Additional details about the error")]
    
    model_config = ConfigDict(
        validate_assignment=True
    )


class ErrorResponse(BaseModel):
    """Model for error responses."""
    status: Annotated[str, Field(default="error", description="Response status")]
    message: Annotated[str, Field(description="Error message")]
    error: Annotated[ErrorDetails, Field(description="Error details")]
    
    model_config = ConfigDict(
        validate_assignment=True
    )


class ConfigResponse(BaseModel):
    """Model for configuration responses."""
    status: Annotated[str, Field(default="success", description="Response status")]
    message: Annotated[str, Field(description="Response message")]
    data: Annotated[Any, Field(default=None, description="Response data")]
    timestamp: Annotated[str, Field(default_factory=lambda: datetime.now().isoformat(), description="Timestamp of the response")]
    
    model_config = ConfigDict(
        validate_assignment=True
    )


# Helper functions

def create_error_response(message: str, code: str, details: Optional[Dict[str, Any]] = None, status: str = "error") -> ErrorResponse:
    """Create a standardized error response."""
    error_details = ErrorDetails(
        code=code,
        timestamp=datetime.now().isoformat(),
        details=details
    )
    return ErrorResponse(status=status, message=message, error=error_details)


def create_success_response(message: str, data: Any = None, status: str = "success") -> ConfigResponse:
    """Create a standardized success response."""
    return ConfigResponse(
        status=status, 
        message=message, 
        data=data,
        timestamp=datetime.now().isoformat()
    )


def model_to_json(model: Union[ErrorResponse, ConfigResponse, BaseModel]) -> str:
    """Convert a Pydantic model to a JSON string."""
    return model.model_dump_json()


def model_to_dict(model: BaseModel) -> Dict[str, Any]:
    """Convert a Pydantic model to a dictionary."""
    return model.model_dump() 