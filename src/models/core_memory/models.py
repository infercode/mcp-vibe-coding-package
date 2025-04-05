#!/usr/bin/env python3
"""
Core Memory Pydantic Models

This module provides Pydantic models for core memory operations such as
entity creation, relationship management, and observations.
"""

from typing import Dict, List, Optional, Any, Union
from datetime import datetime
from pydantic import BaseModel, Field, field_validator, model_validator
import json

# Core Entity Models

class ObservationCreate(BaseModel):
    """Model for creating a new observation."""
    content: str = Field(..., description="The content of the observation")
    metadata: Optional[Dict[str, Any]] = Field(None, description="Optional metadata for the observation")

class EntityCreate(BaseModel):
    """Model for creating a single entity in the graph database."""
    name: str = Field(description="The unique name/identifier of the entity")
    entity_type: str = Field(description="The type/category of the entity")
    observations: Optional[List[str]] = Field(default=[], description="List of observations about the entity")
    metadata: Optional[Dict[str, Any]] = Field(default_factory=dict, description="Additional metadata for the entity")

class EntitiesCreate(BaseModel):
    """Model for creating multiple entities in a single request."""
    entities: List[EntityCreate] = Field(description="List of entities to create")
    
    @field_validator('entities')
    @classmethod
    def validate_entities(cls, v: List[EntityCreate]) -> List[EntityCreate]:
        """Validate that there is at least one entity."""
        if not v:
            raise ValueError("At least one entity must be provided")
        return v

# Core Relationship Models

class RelationshipCreate(BaseModel):
    """Model for creating a relationship between two entities."""
    from_entity: str = Field(description="The source entity name/identifier")
    to_entity: str = Field(description="The target entity name/identifier")
    relationship_type: str = Field(description="The type of relationship")
    weight: Optional[float] = Field(default=1.0, description="The strength/weight of the relationship")
    metadata: Optional[Dict[str, Any]] = Field(default_factory=dict, description="Additional metadata for the relationship")

class RelationshipsCreate(BaseModel):
    """Model for creating multiple relationships in a single request."""
    relationships: List[RelationshipCreate] = Field(description="List of relationships to create")
    
    @field_validator('relationships')
    @classmethod
    def validate_relationships(cls, v: List[RelationshipCreate]) -> List[RelationshipCreate]:
        """Validate that there is at least one relationship."""
        if not v:
            raise ValueError("At least one relationship must be provided")
        return v

# Core Observation Models

class EntityObservation(BaseModel):
    """Model for adding observations to an entity."""
    entity_name: str = Field(description="The entity name/identifier to add observations to")
    contents: List[str] = Field(description="List of observation content strings")
    metadata: Optional[Dict[str, Any]] = Field(default_factory=dict, description="Additional metadata for the observations")

class ObservationsCreate(BaseModel):
    """Model for adding multiple observations in a single request."""
    observations: List[EntityObservation] = Field(description="List of observations to add")
    
    @field_validator('observations')
    @classmethod
    def validate_observations(cls, v: List[EntityObservation]) -> List[EntityObservation]:
        """Validate that there is at least one observation."""
        if not v:
            raise ValueError("At least one observation must be provided")
        return v

# Search Models

class SearchQuery(BaseModel):
    """Model for searching nodes in the graph database."""
    query: str = Field(description="The search query string")
    limit: int = Field(default=10, ge=1, le=100, description="Maximum number of results to return")
    project_name: Optional[str] = Field(default=None, description="Optional project name to search in")
    
    @field_validator('query')
    def query_not_empty(cls, v):
        if not v or v.strip() == '':
            raise ValueError("Search query cannot be empty")
        return v

# Delete Models

class EntityDelete(BaseModel):
    """Model for deleting an entity from the graph database."""
    entity_name: str = Field(description="The name/identifier of the entity to delete")

class RelationshipDelete(BaseModel):
    """Model for deleting a relationship from the graph database."""
    from_entity: str = Field(description="The source entity name/identifier")
    to_entity: str = Field(description="The target entity name/identifier")
    relationship_type: str = Field(description="The type of relationship to delete")

class ObservationDelete(BaseModel):
    """Model for deleting an observation from an entity."""
    entity_name: str = Field(description="The entity name/identifier to delete observation from")
    content: str = Field(description="The content of the observation to delete")

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

class SuccessResponse(BaseModel):
    """Model for success responses."""
    status: str = Field(default="success", description="Response status")
    message: str = Field(description="Success message")
    data: Any = Field(default=None, description="Response data")
    timestamp: str = Field(default_factory=lambda: datetime.now().isoformat(), description="Timestamp of the response")

# Helper functions

def create_error_response(message: str, code: str, details: Optional[Dict[str, Any]] = None, status: str = "error") -> ErrorResponse:
    """Create a standardized error response."""
    error_details = ErrorDetails(code=code, details=details)
    return ErrorResponse(status=status, message=message, error=error_details)

def create_success_response(message: str, data: Any = None, status: str = "success") -> SuccessResponse:
    """Create a standardized success response."""
    return SuccessResponse(status=status, message=message, data=data)

def model_to_json(model: Union[ErrorResponse, SuccessResponse, BaseModel]) -> str:
    """Convert a Pydantic model to a JSON string."""
    return model.model_dump_json()

def model_to_dict(model: BaseModel) -> Dict[str, Any]:
    """Convert a Pydantic model to a dictionary."""
    return model.model_dump() 