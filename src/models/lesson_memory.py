"""
Lesson Memory Models

This module contains Pydantic models for the lesson memory system, providing
schema validation, serialization, and documentation.
"""

from typing import Dict, List, Optional, Any, Union, Annotated
from datetime import datetime
from enum import Enum
from pydantic import (
    BaseModel, 
    Field, 
    field_validator, 
    model_validator, 
    ConfigDict,
    computed_field,
    model_serializer
)


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
    created_at: Annotated[datetime, Field(default_factory=datetime.now)]
    updated_at: Optional[datetime] = None
    confidence: Annotated[Optional[float], Field(None, ge=0.0, le=1.0, description="Confidence score (0.0-1.0)")]
    importance: Optional[str] = None
    tags: Annotated[Optional[List[str]], Field(default_factory=list)]
    source: Optional[str] = None
    additional_info: Optional[Dict[str, Any]] = None
    
    model_config = ConfigDict(
        validate_assignment=True,
        json_schema_extra={
            "examples": [
                {
                    "client_id": "app1",
                    "created_at": "2023-01-01T00:00:00Z",
                    "confidence": 0.85,
                    "tags": ["important", "validated"]
                }
            ]
        }
    )
    
    @computed_field
    def age(self) -> Optional[float]:
        """Calculate the age of this metadata in seconds."""
        if not self.created_at:
            return None
        return (datetime.now() - self.created_at).total_seconds()
    
    @model_serializer
    def serialize_model(self) -> Dict[str, Any]:
        """Custom serialization for metadata."""
        data = self.model_dump(exclude_none=True)
        if self.created_at:
            data["age"] = self.age
        return data


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
    entity_id: Annotated[str, Field(description="ID of the entity this observation belongs to")]
    content: Annotated[str, Field(description="Content of the observation")]
    type: Annotated[LessonObservationType, Field(description="Type of observation")]
    created_at: Annotated[datetime, Field(default_factory=datetime.now)]
    updated_at: Optional[datetime] = None
    metadata: Optional[Metadata] = Field(default_factory=create_metadata)
    
    model_config = ConfigDict(
        validate_assignment=True,
        json_schema_extra={
            "examples": [
                {
                    "entity_id": "lesson-123",
                    "content": "Python list comprehensions are more efficient than map() for simple transformations",
                    "type": "WhatWasLearned",
                }
            ]
        }
    )
    
    @computed_field
    def content_preview(self) -> str:
        """Get a preview of the observation content."""
        if not self.content:
            return ""
        max_length = 50
        if len(self.content) <= max_length:
            return self.content
        return self.content[:max_length] + "..."
    
    @computed_field
    def age_days(self) -> Optional[float]:
        """Calculate the age of this observation in days."""
        if not self.created_at:
            return None
        return (datetime.now() - self.created_at).total_seconds() / 86400
    
    @model_serializer
    def serialize_model(self) -> Dict[str, Any]:
        """Custom serialization with content preview."""
        data = self.model_dump(exclude_none=True)
        data["content_preview"] = self.content_preview
        if self.created_at:
            data["age_days"] = self.age_days
        return data


class LessonObservationCreate(BaseModel):
    """Model for creating a new lesson observation."""
    entity_name: Annotated[str, Field(description="Name of the entity to add observation to")]
    content: Annotated[str, Field(description="Content of the observation")]
    observation_type: Annotated[str, Field(description="Type of observation")]
    container_name: Optional[str] = Field(None, description="Optional container to verify entity membership")
    
    model_config = ConfigDict(
        validate_assignment=True,
        json_schema_extra={
            "examples": [
                {
                    "entity_name": "PythonPerformance",
                    "content": "List comprehensions are typically faster than equivalent for loops",
                    "observation_type": "WhatWasLearned",
                }
            ]
        }
    )
    
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
    entity_name: Annotated[str, Field(description="Name of the entity")]
    observation_id: Annotated[str, Field(description="ID of the observation to update")]
    content: Annotated[str, Field(description="New content for the observation")]
    observation_type: Optional[str] = Field(None, description="Optional new type for the observation")
    
    model_config = ConfigDict(
        validate_assignment=True
    )
    
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
    entity_name: Annotated[str, Field(description="Name of the entity")]
    what_was_learned: Optional[str] = Field(None, description="Factual knowledge gained")
    why_it_matters: Optional[str] = Field(None, description="Importance and consequences")
    how_to_apply: Optional[str] = Field(None, description="Application guidance")
    root_cause: Optional[str] = Field(None, description="Underlying causes")
    evidence: Optional[str] = Field(None, description="Examples and supporting data")
    container_name: Optional[str] = Field(None, description="Optional container to verify entity membership")
    
    model_config = ConfigDict(
        validate_assignment=True,
        json_schema_extra={
            "examples": [
                {
                    "entity_name": "ErrorHandling",
                    "what_was_learned": "Always use specific exception types rather than catching generic Exception",
                    "why_it_matters": "Helps prevent silently catching critical errors",
                    "how_to_apply": "Replace 'except Exception:' with specific exception types"
                }
            ]
        }
    )
    
    @model_validator(mode='after')
    @classmethod
    def check_at_least_one_observation(cls, values):
        """Ensure at least one observation field is provided."""
        observation_fields = [
            'what_was_learned', 'why_it_matters', 'how_to_apply', 
            'root_cause', 'evidence'
        ]
        
        if not any(getattr(values, field) for field in observation_fields):
            raise ValueError("At least one observation must be provided")
            
        return values
    
    @computed_field
    def observation_count(self) -> int:
        """Count how many observation fields are provided."""
        observation_fields = [
            'what_was_learned', 'why_it_matters', 'how_to_apply', 
            'root_cause', 'evidence'
        ]
        return sum(1 for field in observation_fields if getattr(self, field) is not None)
    
    @model_serializer
    def serialize_model(self) -> Dict[str, Any]:
        """Custom serialization with observation count."""
        data = self.model_dump(exclude_none=True)
        data["observation_count"] = self.observation_count
        return data


class LessonEntity(BaseModel):
    """Model for a lesson entity."""
    id: Optional[str] = None
    name: Annotated[str, Field(description="Name of the entity")]
    entity_type: Annotated[str, Field(description="Type of the entity")]
    observations: Annotated[List[str], Field(default_factory=list, description="List of observation IDs")]
    containers: Annotated[List[str], Field(default_factory=list, description="List of container names this entity belongs to")]
    domain: Annotated[str, Field(default="lesson", description="Domain of the entity")]
    metadata: Optional[Metadata] = Field(default_factory=create_metadata)
    
    model_config = ConfigDict(
        validate_assignment=True,
        json_schema_extra={
            "examples": [
                {
                    "name": "PythonPerformance",
                    "entity_type": "Concept",
                    "observations": ["obs-123", "obs-456"],
                    "containers": ["PythonLessons"]
                }
            ]
        }
    )
    
    @computed_field
    def observation_count(self) -> int:
        """Get the number of observations for this entity."""
        return len(self.observations)
    
    @computed_field
    def full_identifier(self) -> str:
        """Get a unique identifier for this entity including domain and type."""
        return f"{self.domain}:{self.entity_type}:{self.name}"
    
    @model_serializer
    def serialize_model(self) -> Dict[str, Any]:
        """Custom serialization including computed fields."""
        data = self.model_dump(exclude_none=True)
        data["observation_count"] = self.observation_count
        data["full_identifier"] = self.full_identifier
        return data


class LessonEntityCreate(BaseModel):
    """Model for creating a new lesson entity."""
    container_name: Annotated[str, Field(description="Name of the container to add the entity to")]
    entity_name: Annotated[str, Field(description="Name of the entity")]
    entity_type: Annotated[str, Field(description="Type of the entity")]
    observations: Optional[List[str]] = Field(default_factory=list, description="Optional list of observations")
    metadata: Optional[Dict[str, Any]] = Field(None, description="Optional metadata for the entity")
    
    model_config = ConfigDict(
        validate_assignment=True
    )


class LessonEntityUpdate(BaseModel):
    """Model for updating an existing lesson entity."""
    entity_name: Annotated[str, Field(description="Name of the entity to update")]
    updates: Annotated[Dict[str, Any], Field(description="Dictionary of fields to update")]
    container_name: Optional[str] = Field(None, description="Optional container to verify entity membership")
    
    model_config = ConfigDict(
        validate_assignment=True
    )


class LessonContainer(BaseModel):
    """Model for a lesson container."""
    id: Optional[str] = None
    name: Annotated[str, Field(description="Name of the container")]
    description: Optional[str] = Field(None, description="Description of the container")
    entities: Annotated[List[str], Field(default_factory=list, description="List of entity IDs in this container")]
    tags: Optional[List[str]] = Field(default_factory=list, description="Tags for the container")
    metadata: Optional[Metadata] = Field(default_factory=create_metadata)
    
    model_config = ConfigDict(
        validate_assignment=True,
        json_schema_extra={
            "examples": [
                {
                    "name": "PythonLessons",
                    "description": "Lessons learned about Python programming",
                    "tags": ["python", "programming", "backend"]
                }
            ]
        }
    )
    
    @computed_field
    def entity_count(self) -> int:
        """Get the number of entities in this container."""
        return len(self.entities)
    
    @model_serializer
    def serialize_model(self) -> Dict[str, Any]:
        """Custom serialization including entity count."""
        data = self.model_dump(exclude_none=True)
        data["entity_count"] = self.entity_count
        return data


class LessonContainerCreate(BaseModel):
    """Model for creating a new lesson container."""
    description: Optional[str] = Field(None, description="Description of the container")
    tags: Optional[List[str]] = Field(default_factory=list, description="Tags for the container")
    metadata: Optional[Dict[str, Any]] = Field(None, description="Optional metadata for the container")
    
    model_config = ConfigDict(
        validate_assignment=True
    )


class LessonContainerUpdate(BaseModel):
    """Model for updating an existing lesson container."""
    container_name: Annotated[str, Field(description="Name of the container to update")]
    updates: Annotated[Dict[str, Any], Field(description="Dictionary of fields to update")]
    
    model_config = ConfigDict(
        validate_assignment=True
    )


class LessonRelationship(BaseModel):
    """Model for a relationship between lesson entities."""
    id: Optional[str] = None
    source_id: Annotated[str, Field(description="ID of the source entity")]
    target_id: Annotated[str, Field(description="ID of the target entity")]
    relationship_type: Annotated[LessonRelationType, Field(description="Type of relationship")]
    properties: Optional[Dict[str, Any]] = Field(default_factory=dict, description="Additional properties")
    created_at: Annotated[datetime, Field(default_factory=datetime.now)]
    metadata: Optional[Metadata] = Field(default_factory=create_metadata)
    
    model_config = ConfigDict(
        validate_assignment=True,
        json_schema_extra={
            "examples": [
                {
                    "source_id": "lesson-123",
                    "target_id": "lesson-456",
                    "relationship_type": "BUILDS_ON",
                    "properties": {"strength": 0.8}
                }
            ]
        }
    )
    
    @computed_field
    def relationship_label(self) -> str:
        """Generate a human-readable label for this relationship."""
        return f"{self.source_id} --[{self.relationship_type}]--> {self.target_id}"
    
    @model_serializer
    def serialize_model(self) -> Dict[str, Any]:
        """Custom serialization with relationship label."""
        data = self.model_dump(exclude_none=True)
        data["relationship_label"] = self.relationship_label
        return data


class LessonRelationshipCreate(BaseModel):
    """Model for creating a relationship between lesson entities."""
    source_name: Annotated[str, Field(description="Name of the source entity")]
    target_name: Annotated[str, Field(description="Name of the target entity")]
    relationship_type: Annotated[str, Field(description="Type of relationship")]
    properties: Optional[Dict[str, Any]] = Field(default_factory=dict, description="Additional properties")
    
    model_config = ConfigDict(
        validate_assignment=True,
        json_schema_extra={
            "examples": [
                {
                    "source_name": "ErrorHandling",
                    "target_name": "PythonPerformance",
                    "relationship_type": "RELATED_TO"
                }
            ]
        }
    )
    
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
    limit: Annotated[int, Field(default=50, description="Maximum number of results", ge=1, le=1000)]
    semantic: Annotated[bool, Field(default=False, description="Whether to use semantic search")]
    
    model_config = ConfigDict(
        validate_assignment=True,
        json_schema_extra={
            "examples": [
                {
                    "container_name": "PythonLessons",
                    "search_term": "error handling",
                    "limit": 20,
                    "semantic": True
                }
            ]
        }
    )
    
    @computed_field
    def has_filters(self) -> bool:
        """Check if any filters are applied to this search."""
        return any([
            self.container_name,
            self.entity_type,
            self.tags
        ])
    
    @model_serializer
    def serialize_model(self) -> Dict[str, Any]:
        """Custom serialization with filter status."""
        data = self.model_dump(exclude_none=True)
        data["has_filters"] = self.has_filters
        return data


class BaseResponse(BaseModel):
    """Base response model with status and timestamp."""
    status: Annotated[str, Field(description="Status of the operation (success or error)")]
    timestamp: Annotated[datetime, Field(default_factory=datetime.now, description="Timestamp of the response")]
    
    model_config = ConfigDict(
        validate_assignment=True
    )


class SuccessResponse(BaseResponse):
    """Success response model."""
    status: str = "success"
    message: Optional[str] = Field(None, description="Success message")
    
    model_config = ConfigDict(
        validate_assignment=True
    )


class ErrorDetail(BaseModel):
    """Error detail model for standardized error responses."""
    code: Annotated[str, Field(description="Error code")]
    message: Annotated[str, Field(description="Error message")]
    details: Optional[Dict[str, Any]] = Field(None, description="Additional error details")
    
    model_config = ConfigDict(
        validate_assignment=True
    )


class ErrorResponse(BaseResponse):
    """Error response model."""
    status: str = "error"
    error: Annotated[ErrorDetail, Field(description="Error details")]
    
    model_config = ConfigDict(
        validate_assignment=True
    )


class EntityResponse(SuccessResponse):
    """Response model for entity operations."""
    entity: Optional[Dict[str, Any]] = Field(None, description="Entity data")
    
    model_config = ConfigDict(
        validate_assignment=True
    )


class ObservationResponse(SuccessResponse):
    """Response model for observation operations."""
    entity: Optional[str] = Field(None, description="Entity name")
    observations: Optional[List[Dict[str, Any]]] = Field(None, description="List of observations")
    observations_by_type: Optional[Dict[str, List[Dict[str, Any]]]] = Field(None, description="Observations grouped by type")
    
    model_config = ConfigDict(
        validate_assignment=True
    )


class ContainerResponse(SuccessResponse):
    """Response model for container operations."""
    container: Optional[Dict[str, Any]] = Field(None, description="Container data")
    
    model_config = ConfigDict(
        validate_assignment=True
    )


class RelationshipResponse(SuccessResponse):
    """Response model for relationship operations."""
    relationship: Optional[Dict[str, Any]] = Field(None, description="Relationship data")
    
    model_config = ConfigDict(
        validate_assignment=True
    )


class SearchResponse(SuccessResponse):
    """Response model for search operations."""
    results: Annotated[List[Dict[str, Any]], Field(default_factory=list, description="Search results")]
    total_count: Annotated[int, Field(default=0, description="Total number of results found")]
    query: Optional[Dict[str, Any]] = Field(None, description="The search query that was executed")
    
    model_config = ConfigDict(
        validate_assignment=True
    )
    
    @computed_field
    def has_results(self) -> bool:
        """Check if the search returned any results."""
        return self.total_count > 0
    
    @model_serializer
    def serialize_model(self) -> Dict[str, Any]:
        """Custom serialization that includes has_results."""
        data = self.model_dump(exclude_none=True)
        data["has_results"] = self.has_results
        return data 