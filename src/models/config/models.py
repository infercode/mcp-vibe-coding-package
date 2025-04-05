#!/usr/bin/env python3
"""
Configuration Pydantic Models

This module provides Pydantic models for configuration-related operations
such as managing unified configuration, Neo4j settings, and embedding settings.
"""

from typing import Dict, List, Optional, Any, Union, Literal
from datetime import datetime
from pydantic import BaseModel, Field, field_validator, model_validator, root_validator, ValidationError
import json
import re

# Configuration Models

class Neo4jConfig(BaseModel):
    """Model for Neo4j database configuration."""
    uri: str = Field(description="Neo4j database URI")
    username: str = Field(description="Neo4j database username")
    password: str = Field(description="Neo4j database password")
    database: str = Field(default="neo4j", description="Neo4j database name")
    
    @field_validator('uri')
    def validate_uri_format(cls, v):
        """Validate that the URI has the correct format."""
        if not v.startswith(('bolt://', 'neo4j://', 'neo4j+s://')):
            raise ValueError("Neo4j URI must start with 'bolt://', 'neo4j://', or 'neo4j+s://'")
        return v

class EmbeddingConfig(BaseModel):
    """Model for embedding service configuration."""
    enabled: bool = Field(default=False, description="Whether embeddings are enabled")
    provider: str = Field(default="openai", description="Embedding service provider (openai, cohere, etc.)")
    model: str = Field(default="text-embedding-ada-002", description="Embedding model to use")
    api_key: Optional[str] = Field(default=None, description="API key for the embedding service")
    api_base: Optional[str] = Field(default=None, description="Base URL for the embedding service API")
    dimensions: int = Field(default=1536, description="Dimensions of the embedding vectors")
    additional_params: Optional[Dict[str, Any]] = Field(default=None, description="Additional parameters for the embedding service")
    
    @model_validator(mode='after')
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

class UnifiedConfig(BaseModel):
    """Model for unified configuration."""
    project_name: str = Field(default="default", description="Name of the project")
    client_id: str = Field(description="Client ID for this configuration")
    timestamp: str = Field(default_factory=lambda: datetime.now().isoformat(), description="Timestamp of the configuration")
    neo4j: Neo4jConfig = Field(description="Neo4j database configuration")
    embedding: EmbeddingConfig = Field(description="Embedding service configuration")
    additional_settings: Optional[Dict[str, Any]] = Field(default=None, description="Additional configuration settings")

class ConfigUpdate(BaseModel):
    """Model for configuration updates."""
    project_name: Optional[str] = Field(default=None, description="Name of the project")
    client_id: Optional[str] = Field(default=None, description="Client ID for this configuration")
    neo4j: Optional[Dict[str, Any]] = Field(default=None, description="Neo4j database configuration updates")
    embedding: Optional[Dict[str, Any]] = Field(default=None, description="Embedding service configuration updates")
    additional_settings: Optional[Dict[str, Any]] = Field(default=None, description="Additional configuration settings updates")

# Status Models

class Neo4jStatus(BaseModel):
    """Model for Neo4j connection status."""
    connected: bool = Field(description="Whether the connection to Neo4j is established")
    version: Optional[str] = Field(default=None, description="Neo4j server version")
    address: Optional[str] = Field(default=None, description="Neo4j server address")
    message: str = Field(description="Status message")
    timestamp: str = Field(default_factory=lambda: datetime.now().isoformat(), description="Timestamp of the status check")

class EmbeddingStatus(BaseModel):
    """Model for embedding API status."""
    available: bool = Field(description="Whether the embedding API is available")
    provider: str = Field(description="Embedding provider name")
    model: str = Field(description="Embedding model name")
    dimensions: int = Field(description="Number of dimensions in the embeddings")
    message: str = Field(description="Status message")
    timestamp: str = Field(default_factory=lambda: datetime.now().isoformat(), description="Timestamp of the status check")

class MemoryStatus(BaseModel):
    """Model for memory system status."""
    operational: bool = Field(description="Whether the memory system is operational")
    neo4j: Neo4jStatus = Field(description="Neo4j connection status")
    embedding: EmbeddingStatus = Field(description="Embedding API status")
    message: str = Field(description="Status message")
    timestamp: str = Field(default_factory=lambda: datetime.now().isoformat(), description="Timestamp of the status check")

# Response Models

class ErrorDetails(BaseModel):
    """Model for error details."""
    code: str = Field(description="Error code")
    timestamp: str = Field(default_factory=lambda: datetime.now().isoformat(), description="Timestamp of the error")
    details: Optional[Dict[str, Any]] = Field(default=None, description="Additional details about the error")

class ErrorResponse(BaseModel):
    """Model for error responses."""
    status: str = Field(default="error", description="Response status")
    message: str = Field(description="Error message")
    error: ErrorDetails = Field(description="Error details")

class ConfigResponse(BaseModel):
    """Model for configuration responses."""
    status: str = Field(default="success", description="Response status")
    message: str = Field(description="Response message")
    data: Any = Field(default=None, description="Response data")
    timestamp: str = Field(default_factory=lambda: datetime.now().isoformat(), description="Timestamp of the response")

# Helper functions

def create_error_response(message: str, code: str, details: Optional[Dict[str, Any]] = None, status: str = "error") -> ErrorResponse:
    """Create a standardized error response."""
    error_details = ErrorDetails(code=code, details=details)
    return ErrorResponse(status=status, message=message, error=error_details)

def create_success_response(message: str, data: Any = None, status: str = "success") -> ConfigResponse:
    """Create a standardized success response."""
    return ConfigResponse(status=status, message=message, data=data)

def model_to_json(model: Union[ErrorResponse, ConfigResponse, BaseModel]) -> str:
    """Convert a Pydantic model to a JSON string."""
    return model.model_dump_json()

def model_to_dict(model: BaseModel) -> Dict[str, Any]:
    """Convert a Pydantic model to a dictionary."""
    return model.model_dump() 