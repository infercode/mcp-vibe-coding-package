"""
Lesson Memory Models

This module contains Pydantic models for the lesson memory system, providing
schema validation, serialization, and documentation.
"""

from typing import Dict, List, Optional, Any, Union
from datetime import datetime
from enum import Enum
from pydantic import BaseModel, Field, field_validator, model_validator


class LessonObservationType(str, Enum):
    """Enumeration of standard lesson observation types."""
    WHAT_WAS_LEARNED = "WhatWasLearned"  # Factual knowledge gained
    WHY_IT_MATTERS = "WhyItMatters"      # Importance and consequences
    HOW_TO_APPLY = "HowToApply"          # Application guidance
    ROOT_CAUSE = "RootCause"             # Underlying causes
    EVIDENCE = "Evidence"                # Examples and supporting data


class LessonRelationType(str, Enum):
    """Enumeration of lesson relationship types."""
    BUILDS_ON = "BUILDS_ON"              # Extends or enhances previous knowledge
    SUPERSEDES = "SUPERSEDES"            # Replaces outdated information
    CONTRADICTS = "CONTRADICTS"          # Provides opposing viewpoints
    ORIGINATED_FROM = "ORIGINATED_FROM"  # Tracks source of a lesson
    SOLVED_WITH = "SOLVED_WITH"          # Connects to solutions
    RELATED_TO = "RELATED_TO"            # General relationship
    DEPENDS_ON = "DEPENDS_ON"            # Dependency relationship
    IMPLIES = "IMPLIES"                  # Logical implication
    SIMILAR_TO = "SIMILAR_TO"            # Similarity relationship


class Metadata(BaseModel):
    """Metadata for lesson memory entities."""
    client_id: Optional[str] = None
    created_at: datetime = Field(default_factory=datetime.now)
    updated_at: Optional[datetime] = None
    confidence: Optional[float] = Field(None, ge=0.0, le=1.0, description="Confidence score (0.0-1.0)")
    importance: Optional[str] = None
    tags: Optional[List[str]] = Field(default_factory=list)
    source: Optional[str] = None
    additional_info: Optional[Dict[str, Any]] = None


def create_metadata() -> Metadata:
    """Factory function to create a new Metadata instance."""
    return Metadata(
        client_id=None,
        created_at=datetime.now(),
        updated_at=None,
        confidence=None,
        importance=None,
        tags=[],
        source=None,
        additional_info=None
    )


class LessonObservation(BaseModel):
    """Model for a lesson observation."""
    id: Optional[str] = None
    entity_id: str = Field(..., description="ID of the entity this observation belongs to")
    content: str = Field(..., description="Content of the observation")
    type: LessonObservationType = Field(..., description="Type of observation")
    created_at: datetime = Field(default_factory=datetime.now)
    updated_at: Optional[datetime] = None
    metadata: Optional[Metadata] = Field(default_factory=create_metadata)


class LessonObservationCreate(BaseModel):
    """Model for creating a new lesson observation."""
    entity_name: str = Field(..., description="Name of the entity to add observation to")
    content: str = Field(..., description="Content of the observation")
    observation_type: str = Field(..., description="Type of observation")
    container_name: Optional[str] = Field(None, description="Optional container to verify entity membership")
    
    @field_validator('observation_type')
    @classmethod
    def validate_observation_type(cls, v: str) -> str:
        """Validate that the observation type is one of the standard types."""
        try:
            return LessonObservationType(v).value
        except ValueError:
            # Allow custom types but log a warning
            return v


class LessonObservationUpdate(BaseModel):
    """Model for updating an existing lesson observation."""
    entity_name: str = Field(..., description="Name of the entity")
    observation_id: str = Field(..., description="ID of the observation to update")
    content: str = Field(..., description="New content for the observation")
    observation_type: Optional[str] = Field(None, description="Optional new type for the observation")
    
    @field_validator('observation_type')
    @classmethod
    def validate_observation_type(cls, v: Optional[str]) -> Optional[str]:
        """Validate that the observation type is one of the standard types."""
        if v is None:
            return None
        try:
            return LessonObservationType(v).value
        except ValueError:
            # Allow custom types but log a warning
            return v


class StructuredLessonObservations(BaseModel):
    """Model for creating structured lesson observations."""
    entity_name: str = Field(..., description="Name of the entity")
    what_was_learned: Optional[str] = Field(None, description="Factual knowledge gained")
    why_it_matters: Optional[str] = Field(None, description="Importance and consequences")
    how_to_apply: Optional[str] = Field(None, description="Application guidance")
    root_cause: Optional[str] = Field(None, description="Underlying causes")
    evidence: Optional[str] = Field(None, description="Examples and supporting data")
    container_name: Optional[str] = Field(None, description="Optional container to verify entity membership")
    
    @model_validator(mode='after')
    def check_at_least_one_observation(cls, values):
        """Ensure at least one observation field is provided."""
        observation_fields = [
            'what_was_learned', 'why_it_matters', 'how_to_apply', 
            'root_cause', 'evidence'
        ]
        
        if not any(getattr(values, field) for field in observation_fields):
            raise ValueError("At least one observation must be provided")
            
        return values


class LessonEntity(BaseModel):
    """Model for a lesson entity."""
    id: Optional[str] = None
    name: str = Field(..., description="Name of the entity")
    entity_type: str = Field(..., description="Type of the entity")
    observations: List[str] = Field(default_factory=list, description="List of observation IDs")
    containers: List[str] = Field(default_factory=list, description="List of container names this entity belongs to")
    domain: str = Field("lesson", description="Domain of the entity")
    metadata: Optional[Metadata] = Field(default_factory=create_metadata)


class LessonEntityCreate(BaseModel):
    """Model for creating a new lesson entity."""
    container_name: str = Field(..., description="Name of the container to add the entity to")
    entity_name: str = Field(..., description="Name of the entity")
    entity_type: str = Field(..., description="Type of the entity")
    observations: Optional[List[str]] = Field(default_factory=list, description="Optional list of observations")
    metadata: Optional[Dict[str, Any]] = Field(None, description="Optional metadata for the entity")


class LessonEntityUpdate(BaseModel):
    """Model for updating an existing lesson entity."""
    entity_name: str = Field(..., description="Name of the entity to update")
    updates: Dict[str, Any] = Field(..., description="Dictionary of fields to update")
    container_name: Optional[str] = Field(None, description="Optional container to verify entity membership")


class LessonContainer(BaseModel):
    """Model for a lesson container."""
    id: Optional[str] = None
    name: str = Field(..., description="Name of the container")
    description: Optional[str] = Field(None, description="Description of the container")
    entities: List[str] = Field(default_factory=list, description="List of entity IDs in this container")
    tags: Optional[List[str]] = Field(default_factory=list, description="Tags for the container")
    metadata: Optional[Metadata] = Field(default_factory=create_metadata)


class LessonContainerCreate(BaseModel):
    """Model for creating a new lesson container."""
    name: str = Field(..., description="Name of the container")
    description: Optional[str] = Field(None, description="Description of the container")
    tags: Optional[List[str]] = Field(default_factory=list, description="Tags for the container")
    metadata: Optional[Dict[str, Any]] = Field(None, description="Optional metadata for the container")


class LessonContainerUpdate(BaseModel):
    """Model for updating an existing lesson container."""
    container_name: str = Field(..., description="Name of the container to update")
    updates: Dict[str, Any] = Field(..., description="Dictionary of fields to update")


class LessonRelationship(BaseModel):
    """Model for a relationship between lesson entities."""
    id: Optional[str] = None
    source_id: str = Field(..., description="ID of the source entity")
    target_id: str = Field(..., description="ID of the target entity")
    relationship_type: LessonRelationType = Field(..., description="Type of relationship")
    properties: Optional[Dict[str, Any]] = Field(default_factory=dict, description="Additional properties")
    created_at: datetime = Field(default_factory=datetime.now)
    metadata: Optional[Metadata] = Field(default_factory=create_metadata)


class LessonRelationshipCreate(BaseModel):
    """Model for creating a relationship between lesson entities."""
    source_name: str = Field(..., description="Name of the source entity")
    target_name: str = Field(..., description="Name of the target entity")
    relationship_type: str = Field(..., description="Type of relationship")
    properties: Optional[Dict[str, Any]] = Field(default_factory=dict, description="Additional properties")
    
    @field_validator('relationship_type')
    @classmethod
    def validate_relationship_type(cls, v: str) -> str:
        """Validate that the relationship type is one of the standard types."""
        try:
            return LessonRelationType(v).value
        except ValueError:
            # Allow custom types but log a warning
            return v


class SearchQuery(BaseModel):
    """Model for lesson search queries."""
    container_name: Optional[str] = Field(None, description="Optional container to search within")
    search_term: Optional[str] = Field(None, description="Text to search for")
    entity_type: Optional[str] = Field(None, description="Entity type to filter by")
    tags: Optional[List[str]] = Field(None, description="Tags to filter by")
    limit: int = Field(50, description="Maximum number of results", ge=1, le=1000)
    semantic: bool = Field(False, description="Whether to use semantic search")


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


class EntityResponse(SuccessResponse):
    """Response model for entity operations."""
    entity: Optional[Dict[str, Any]] = Field(None, description="Entity data")


class ObservationResponse(SuccessResponse):
    """Response model for observation operations."""
    entity: Optional[str] = Field(None, description="Entity name")
    observations: Optional[List[Dict[str, Any]]] = Field(None, description="List of observations")
    observations_by_type: Optional[Dict[str, List[Dict[str, Any]]]] = Field(None, description="Observations grouped by type")


class ContainerResponse(SuccessResponse):
    """Response model for container operations."""
    container: Optional[Dict[str, Any]] = Field(None, description="Container data")


class RelationshipResponse(SuccessResponse):
    """Response model for relationship operations."""
    relationship: Optional[Dict[str, Any]] = Field(None, description="Relationship data")


class SearchResponse(SuccessResponse):
    """Response model for search operations."""
    results: List[Dict[str, Any]] = Field(default_factory=list, description="Search results")
    total_count: int = Field(0, description="Total number of results found")
    query: Optional[Dict[str, Any]] = Field(None, description="The search query that was executed") 