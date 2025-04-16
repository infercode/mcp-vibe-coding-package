#!/usr/bin/env python3
"""
Core Memory Pydantic Models

This module provides Pydantic models for core memory operations such as
entity creation, relationship management, and observations.
"""

from typing import Dict, List, Optional, Any, Union, Annotated
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

# Core Entity Models

class ObservationCreate(BaseModel):
    """Model for creating a new observation."""
    content: Annotated[str, Field(description="The content of the observation")]
    metadata: Annotated[Optional[Dict[str, Any]], Field(default=None, description="Optional metadata for the observation")]
    
    model_config = ConfigDict(
        validate_assignment=True,
        json_schema_extra={
            "examples": [
                {
                    "content": "This entity represents a key component",
                    "metadata": {"timestamp": "2023-01-01T00:00:00Z", "source": "user_input"}
                }
            ]
        }
    )


class EntityCreate(BaseModel):
    """Model for creating a single entity in the graph database."""
    name: Annotated[str, Field(description="The unique name/identifier of the entity")]
    entity_type: Annotated[str, Field(description="The type/category of the entity")]
    observations: Annotated[Optional[List[str]], Field(default=[], description="List of observations about the entity")]
    metadata: Annotated[Optional[Dict[str, Any]], Field(default_factory=dict, description="Additional metadata for the entity")]
    
    model_config = ConfigDict(
        validate_assignment=True,
        json_schema_extra={
            "examples": [
                {
                    "name": "AuthService",
                    "entity_type": "Component",
                    "observations": ["Handles user authentication", "Manages JWT tokens"],
                    "metadata": {"created_at": "2023-01-01T00:00:00Z"}
                }
            ]
        }
    )


class EntitiesCreate(BaseModel):
    """Model for creating multiple entities in a single request."""
    entities: Annotated[List[EntityCreate], Field(description="List of entities to create")]
    
    model_config = ConfigDict(
        validate_assignment=True
    )
    
    @field_validator('entities')
    @classmethod
    def validate_entities(cls, v: List[EntityCreate]) -> List[EntityCreate]:
        """Validate that there is at least one entity."""
        if not v:
            raise ValueError("At least one entity must be provided")
        return v
    
    @computed_field
    def count(self) -> int:
        """Get the number of entities to create."""
        return len(self.entities)


# Core Relationship Models

class RelationshipCreate(BaseModel):
    """Model for creating a relationship between two entities."""
    from_entity: Annotated[str, Field(description="The source entity name/identifier")]
    to_entity: Annotated[str, Field(description="The target entity name/identifier")]
    relationship_type: Annotated[str, Field(description="The type of relationship")]
    weight: Annotated[Optional[float], Field(default=1.0, description="The strength/weight of the relationship")]
    metadata: Annotated[Optional[Dict[str, Any]], Field(default_factory=dict, description="Additional metadata for the relationship")]
    
    model_config = ConfigDict(
        validate_assignment=True,
        json_schema_extra={
            "examples": [
                {
                    "from_entity": "AuthService",
                    "to_entity": "UserDatabase",
                    "relationship_type": "DEPENDS_ON",
                    "weight": 1.0,
                    "metadata": {"critical": True}
                }
            ]
        }
    )
    
    @computed_field
    def relationship_description(self) -> str:
        """Generate a human-readable description of the relationship."""
        return f"{self.from_entity} --[{self.relationship_type}]--> {self.to_entity}"


class RelationshipsCreate(BaseModel):
    """Model for creating multiple relationships in a single request."""
    relationships: Annotated[List[RelationshipCreate], Field(description="List of relationships to create")]
    
    model_config = ConfigDict(
        validate_assignment=True
    )
    
    @field_validator('relationships')
    @classmethod
    def validate_relationships(cls, v: List[RelationshipCreate]) -> List[RelationshipCreate]:
        """Validate that there is at least one relationship."""
        if not v:
            raise ValueError("At least one relationship must be provided")
        return v
    
    @computed_field
    def count(self) -> int:
        """Get the number of relationships to create."""
        return len(self.relationships)


# Core Observation Models

class EntityObservation(BaseModel):
    """Model for adding observations to an entity."""
    entity_name: Annotated[str, Field(description="The entity name/identifier to add observations to")]
    contents: Annotated[List[str], Field(description="List of observation content strings")]
    metadata: Annotated[Optional[Dict[str, Any]], Field(default_factory=dict, description="Additional metadata for the observations")]
    
    model_config = ConfigDict(
        validate_assignment=True,
        json_schema_extra={
            "examples": [
                {
                    "entity_name": "AuthService",
                    "contents": ["Updated to handle OAuth", "Added rate limiting"],
                    "metadata": {"updated_at": "2023-01-01T00:00:00Z"}
                }
            ]
        }
    )


class ObservationsCreate(BaseModel):
    """Model for adding multiple observations in a single request."""
    observations: Annotated[List[EntityObservation], Field(description="List of observations to add")]
    
    model_config = ConfigDict(
        validate_assignment=True
    )
    
    @field_validator('observations')
    @classmethod
    def validate_observations(cls, v: List[EntityObservation]) -> List[EntityObservation]:
        """Validate that there is at least one observation."""
        if not v:
            raise ValueError("At least one observation must be provided")
        return v
    
    @computed_field
    def total_observations(self) -> int:
        """Get the total number of observations across all entities."""
        return sum(len(obs.contents) for obs in self.observations)


# Search Models

class SearchQuery(BaseModel):
    """Model for searching nodes in the graph database."""
    query: Annotated[str, Field(description="The search query string")]
    limit: Annotated[int, Field(default=10, ge=1, le=100, description="Maximum number of results to return")]
    project_name: Annotated[Optional[str], Field(default=None, description="Optional project name to search in")]
    
    model_config = ConfigDict(
        validate_assignment=True,
        json_schema_extra={
            "examples": [
                {
                    "query": "authentication service",
                    "limit": 10,
                    "project_name": "my_project"
                }
            ]
        }
    )
    
    @field_validator('query')
    @classmethod
    def query_not_empty(cls, v: str) -> str:
        if not v or v.strip() == '':
            raise ValueError("Search query cannot be empty")
        return v
    
    @computed_field
    def has_project_filter(self) -> bool:
        """Check if this query has a project filter."""
        return self.project_name is not None


# Delete Models

class EntityDelete(BaseModel):
    """Model for deleting an entity from the graph database."""
    entity_name: Annotated[str, Field(description="The name/identifier of the entity to delete")]
    
    model_config = ConfigDict(
        validate_assignment=True
    )


class RelationshipDelete(BaseModel):
    """Model for deleting a relationship from the graph database."""
    from_entity: Annotated[str, Field(description="The source entity name/identifier")]
    to_entity: Annotated[str, Field(description="The target entity name/identifier")]
    relationship_type: Annotated[str, Field(description="The type of relationship to delete")]
    
    model_config = ConfigDict(
        validate_assignment=True
    )


class ObservationDelete(BaseModel):
    """Model for deleting an observation from an entity."""
    entity_name: Annotated[str, Field(description="The entity name/identifier to delete observation from")]
    content: Annotated[str, Field(description="The content of the observation to delete")]
    
    model_config = ConfigDict(
        validate_assignment=True
    )


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


class SuccessResponse(BaseModel):
    """Model for success responses."""
    status: Annotated[str, Field(default="success", description="Response status")]
    message: Annotated[str, Field(description="Success message")]
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


def create_success_response(message: str, data: Any = None, status: str = "success") -> SuccessResponse:
    """Create a standardized success response."""
    return SuccessResponse(
        status=status, 
        message=message, 
        data=data,
        timestamp=datetime.now().isoformat()
    )


def model_to_json(model: Union[ErrorResponse, SuccessResponse, BaseModel]) -> str:
    """Convert a Pydantic model to a JSON string."""
    return model.model_dump_json()


def model_to_dict(model: BaseModel) -> Dict[str, Any]:
    """Convert a Pydantic model to a dictionary."""
    return model.model_dump() 