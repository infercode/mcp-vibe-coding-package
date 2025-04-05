"""
Project Memory Models

This module contains Pydantic models for the project memory system, providing
schema validation, serialization, and documentation.
"""

from typing import Dict, List, Optional, Any, Union, Callable
from datetime import datetime
from pydantic import BaseModel, Field, field_validator, model_validator


class Metadata(BaseModel):
    """Metadata for various entities in the project memory system."""
    client_id: Optional[str] = None
    created_at: datetime = Field(default_factory=datetime.now)
    updated_at: Optional[datetime] = None
    source: Optional[str] = None
    importance: Optional[str] = None
    confidence: Optional[float] = Field(None, ge=0.0, le=1.0)
    additional_info: Optional[Dict[str, Any]] = None


def create_metadata() -> Metadata:
    """Factory function to create a new Metadata instance."""
    return Metadata(
        client_id=None,
        created_at=datetime.now(),
        updated_at=None,
        source=None,
        importance=None,
        confidence=None,
        additional_info=None
    )


class ProjectContainer(BaseModel):
    """Project container model for creating and updating projects."""
    name: str = Field(..., description="The name of the project container")
    description: Optional[str] = Field(None, description="Description of the project")
    tags: Optional[List[str]] = Field(default_factory=list, description="List of tags for categorizing the project")
    metadata: Optional[Metadata] = Field(default_factory=create_metadata, description="Additional metadata for the project")


class ProjectContainerCreate(ProjectContainer):
    """Model for creating a new project container."""
    pass


class ProjectContainerUpdate(BaseModel):
    """Model for updating an existing project container."""
    id: str = Field(..., description="The ID of the project container to update")
    name: Optional[str] = Field(None, description="New name for the project")
    description: Optional[str] = Field(None, description="New description for the project")
    tags: Optional[List[str]] = Field(None, description="Updated list of tags")
    metadata: Optional[Metadata] = Field(None, description="Updated metadata for the project")
    
    @model_validator(mode='after')
    def check_at_least_one_field(cls, values):
        """Ensure at least one field besides id is provided for the update."""
        update_fields = [field for field in ['name', 'description', 'tags', 'metadata'] 
                         if getattr(values, field) is not None]
        if not update_fields:
            raise ValueError("At least one field must be provided for update")
        return values


class ComponentCreate(BaseModel):
    """Model for creating a new component within a project."""
    project_id: str = Field(..., description="The ID of the project this component belongs to")
    name: str = Field(..., description="The name of the component")
    description: Optional[str] = Field(None, description="Description of the component")
    type: str = Field(..., description="The type of component (e.g., SERVICE, LIBRARY, UI)")
    tags: Optional[List[str]] = Field(default_factory=list, description="List of tags for the component")
    metadata: Optional[Metadata] = Field(default_factory=create_metadata, description="Additional metadata")


class ComponentUpdate(BaseModel):
    """Model for updating an existing component."""
    id: str = Field(..., description="The ID of the component to update")
    name: Optional[str] = Field(None, description="New name for the component")
    description: Optional[str] = Field(None, description="New description for the component")
    type: Optional[str] = Field(None, description="Updated component type")
    tags: Optional[List[str]] = Field(None, description="Updated list of tags")
    metadata: Optional[Metadata] = Field(None, description="Updated metadata")
    
    @model_validator(mode='after')
    def check_at_least_one_field(cls, values):
        """Ensure at least one field besides id is provided for the update."""
        update_fields = [field for field in ['name', 'description', 'type', 'tags', 'metadata'] 
                         if getattr(values, field) is not None]
        if not update_fields:
            raise ValueError("At least one field must be provided for update")
        return values


class DomainEntityCreate(BaseModel):
    """Model for creating a domain entity within a project."""
    project_id: str = Field(..., description="The ID of the project this entity belongs to")
    name: str = Field(..., description="The name of the domain entity")
    type: str = Field(..., description="The type of entity (e.g., DECISION, FEATURE, REQUIREMENT)")
    description: Optional[str] = Field(None, description="Description of the entity")
    content: Optional[str] = Field(None, description="Detailed content or explanation")
    tags: Optional[List[str]] = Field(default_factory=list, description="List of tags")
    metadata: Optional[Metadata] = Field(default_factory=create_metadata, description="Additional metadata")
    
    @field_validator('type')
    @classmethod
    def validate_entity_type(cls, v: str) -> str:
        """Validate that the entity type is one of the allowed types."""
        allowed_types = ['DECISION', 'FEATURE', 'REQUIREMENT', 'SPECIFICATION', 'CONSTRAINT', 'RISK']
        if v.upper() not in allowed_types:
            raise ValueError(f"Entity type must be one of {allowed_types}")
        return v.upper()


class RelationshipCreate(BaseModel):
    """Model for creating a relationship between entities."""
    source_id: str = Field(..., description="The ID of the source entity")
    target_id: str = Field(..., description="The ID of the target entity")
    relationship_type: str = Field(..., description="The type of relationship")
    metadata: Optional[Metadata] = Field(default_factory=create_metadata, description="Additional metadata")
    
    @field_validator('relationship_type')
    @classmethod
    def validate_relationship_type(cls, v: str) -> str:
        """Validate that the relationship type is one of the allowed types."""
        allowed_types = [
            'CONTAINS', 'IMPLEMENTS', 'DEPENDS_ON', 'RELATED_TO', 'SUPERSEDES',
            'ALTERNATIVE_TO', 'INFLUENCES', 'LEADS_TO', 'DERIVES_FROM', 'CONSTRAINS'
        ]
        if v.upper() not in allowed_types:
            raise ValueError(f"Relationship type must be one of {allowed_types}")
        return v.upper()


class SearchQuery(BaseModel):
    """Model for search queries."""
    query: str = Field(..., description="The search query text")
    entity_types: Optional[List[str]] = Field(None, description="Filter by entity types")
    relationship_types: Optional[List[str]] = Field(None, description="Filter by relationship types")
    tags: Optional[List[str]] = Field(None, description="Filter by tags")
    limit: int = Field(10, description="Maximum number of results to return", ge=1, le=100)
    confidence_threshold: Optional[float] = Field(None, description="Minimum confidence threshold", ge=0.0, le=1.0)


class BaseResponse(BaseModel):
    """Base response model with status and timestamp."""
    status: str = Field(..., description="Status of the operation (success or error)")
    timestamp: datetime = Field(default_factory=datetime.now, description="Timestamp of the response")


class SuccessResponse(BaseResponse):
    """Success response model."""
    status: str = "success"
    message: Optional[str] = Field(None, description="Success message")


class ErrorDetail(BaseModel):
    """Error detail model for standardized error responses."""
    code: str = Field(..., description="Error code")
    message: str = Field(..., description="Error message")
    details: Optional[Dict[str, Any]] = Field(None, description="Additional error details")


class ErrorResponse(BaseResponse):
    """Error response model."""
    status: str = "error"
    error: ErrorDetail = Field(..., description="Error details")


class ProjectContainerResponse(SuccessResponse):
    """Response model for project container operations."""
    project_id: Optional[str] = Field(None, description="ID of the created or updated project")
    project: Optional[Dict[str, Any]] = Field(None, description="Project container data")


class ComponentResponse(SuccessResponse):
    """Response model for component operations."""
    component_id: Optional[str] = Field(None, description="ID of the created or updated component")
    component: Optional[Dict[str, Any]] = Field(None, description="Component data")


class SearchResponse(SuccessResponse):
    """Response model for search operations."""
    results: List[Dict[str, Any]] = Field(default_factory=list, description="Search results")
    total_count: int = Field(0, description="Total number of results found")
    query: str = Field(..., description="The search query that was executed") 