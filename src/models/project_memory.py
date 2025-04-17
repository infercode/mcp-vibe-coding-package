"""
Project Memory Models

This module contains Pydantic models for the project memory system, providing
schema validation, serialization, and documentation.
"""

from typing import Dict, List, Optional, Any, Union, Callable, Annotated
from datetime import datetime
from pydantic import (
    BaseModel, 
    Field, 
    field_validator, 
    model_validator, 
    ConfigDict,
    computed_field,
    model_serializer
)


class Metadata(BaseModel):
    """Metadata for various entities in the project memory system."""
    client_id: Optional[str] = None
    created_at: Annotated[datetime, Field(default_factory=datetime.now)]
    updated_at: Optional[datetime] = None
    source: Optional[str] = None
    importance: Optional[str] = None
    confidence: Annotated[Optional[float], Field(None, ge=0.0, le=1.0)]
    additional_info: Optional[Dict[str, Any]] = None
    
    model_config = ConfigDict(
        validate_assignment=True,
        json_schema_extra={
            "examples": [
                {
                    "client_id": "app1",
                    "created_at": "2023-01-01T00:00:00Z",
                    "importance": "high",
                    "confidence": 0.85
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
        source=None,
        importance=None,
        confidence=None,
        additional_info=None
    )


class ProjectContainer(BaseModel):
    """Project container model for creating and updating projects."""
    name: Annotated[str, Field(description="The name of the project container")]
    description: Optional[str] = Field(None, description="Description of the project")
    tags: Annotated[Optional[List[str]], Field(default_factory=list, description="List of tags for categorizing the project")]
    metadata: Optional[Metadata] = Field(default_factory=create_metadata, description="Additional metadata for the project")
    
    model_config = ConfigDict(
        validate_assignment=True,
        json_schema_extra={
            "examples": [
                {
                    "name": "Customer Portal Redesign",
                    "description": "Project to redesign the customer portal UI and UX",
                    "tags": ["frontend", "ui", "ux"]
                }
            ]
        }
    )
    
    @computed_field
    def has_tags(self) -> bool:
        """Check if this project has any tags."""
        return bool(self.tags)
    
    @model_serializer
    def serialize_model(self) -> Dict[str, Any]:
        """Custom serialization for project container."""
        data = self.model_dump(exclude_none=True)
        data["has_tags"] = self.has_tags
        return data


class ProjectContainerCreate(ProjectContainer):
    """Model for creating a new project container."""
    
    model_config = ConfigDict(
        validate_assignment=True
    )


class ProjectContainerUpdate(BaseModel):
    """Model for updating an existing project container."""
    id: Annotated[str, Field(description="The ID of the project container to update")]
    name: Optional[str] = Field(None, description="New name for the project")
    description: Optional[str] = Field(None, description="New description for the project")
    tags: Optional[List[str]] = Field(None, description="Updated list of tags")
    metadata: Optional[Metadata] = Field(None, description="Updated metadata for the project")
    
    model_config = ConfigDict(
        validate_assignment=True
    )
    
    @model_validator(mode='after')
    @classmethod
    def check_at_least_one_field(cls, values):
        """Ensure at least one field besides id is provided for the update."""
        update_fields = [field for field in ['name', 'description', 'tags', 'metadata'] 
                         if getattr(values, field) is not None]
        if not update_fields:
            raise ValueError("At least one field must be provided for update")
        return values
    
    @computed_field
    def update_count(self) -> int:
        """Count how many fields are being updated."""
        update_fields = [field for field in ['name', 'description', 'tags', 'metadata'] 
                         if getattr(self, field) is not None]
        return len(update_fields)


class ComponentCreate(BaseModel):
    """Model for creating a new component within a project."""
    project_id: Annotated[str, Field(description="The ID of the project this component belongs to")]
    name: Annotated[str, Field(description="The name of the component")]
    description: Optional[str] = Field(None, description="Description of the component")
    type: Annotated[str, Field(description="The type of component (e.g., SERVICE, LIBRARY, UI)")]
    tags: Annotated[Optional[List[str]], Field(default_factory=list, description="List of tags for the component")]
    metadata: Optional[Metadata] = Field(default_factory=create_metadata, description="Additional metadata")
    
    model_config = ConfigDict(
        validate_assignment=True,
        json_schema_extra={
            "examples": [
                {
                    "project_id": "proj-123",
                    "name": "Authentication Service",
                    "type": "SERVICE",
                    "description": "Handles user authentication and token management"
                }
            ]
        }
    )
    
    @computed_field
    def component_identifier(self) -> str:
        """Generate a unique identifier for this component."""
        return f"{self.project_id}:{self.type}:{self.name}"


class ComponentUpdate(BaseModel):
    """Model for updating an existing component."""
    id: Annotated[str, Field(description="The ID of the component to update")]
    name: Optional[str] = Field(None, description="New name for the component")
    description: Optional[str] = Field(None, description="New description for the component")
    type: Optional[str] = Field(None, description="Updated component type")
    tags: Optional[List[str]] = Field(None, description="Updated list of tags")
    metadata: Optional[Metadata] = Field(None, description="Updated metadata")
    
    model_config = ConfigDict(
        validate_assignment=True
    )
    
    @model_validator(mode='after')
    @classmethod
    def check_at_least_one_field(cls, values):
        """Ensure at least one field besides id is provided for the update."""
        update_fields = [field for field in ['name', 'description', 'type', 'tags', 'metadata'] 
                         if getattr(values, field) is not None]
        if not update_fields:
            raise ValueError("At least one field must be provided for update")
        return values
    
    @computed_field
    def update_count(self) -> int:
        """Count how many fields are being updated."""
        update_fields = [field for field in ['name', 'description', 'type', 'tags', 'metadata'] 
                         if getattr(self, field) is not None]
        return len(update_fields)


class DomainEntityCreate(BaseModel):
    """Model for creating a domain entity within a project."""
    project_id: Annotated[str, Field(description="The ID of the project this entity belongs to")]
    name: Annotated[str, Field(description="The name of the domain entity")]
    type: Annotated[str, Field(description="The type of entity (e.g., DECISION, FEATURE, REQUIREMENT)")]
    description: Optional[str] = Field(None, description="Description of the entity")
    content: Optional[str] = Field(None, description="Detailed content or explanation")
    tags: Annotated[Optional[List[str]], Field(default_factory=list, description="List of tags")]
    metadata: Optional[Metadata] = Field(default_factory=create_metadata, description="Additional metadata")
    
    model_config = ConfigDict(
        validate_assignment=True,
        json_schema_extra={
            "examples": [
                {
                    "project_id": "proj-123",
                    "name": "Use JWT for Authentication",
                    "type": "DECISION",
                    "description": "Decision to use JWT tokens for authentication"
                }
            ]
        }
    )
    
    @field_validator('type')
    @classmethod
    def validate_entity_type(cls, v: str) -> str:
        """Validate that the entity type is one of the allowed types."""
        allowed_types = ['DECISION', 'FEATURE', 'REQUIREMENT', 'SPECIFICATION', 'CONSTRAINT', 'RISK']
        if v.upper() not in allowed_types:
            raise ValueError(f"Entity type must be one of {allowed_types}")
        return v.upper()
    
    @computed_field
    def entity_identifier(self) -> str:
        """Generate a unique identifier for this entity."""
        return f"{self.project_id}:{self.type}:{self.name}"
    
    @model_serializer
    def serialize_model(self) -> Dict[str, Any]:
        """Custom serialization for domain entity."""
        data = self.model_dump(exclude_none=True)
        data["entity_identifier"] = self.entity_identifier
        return data


class RelationshipCreate(BaseModel):
    """Model for creating a relationship between entities."""
    source_id: Annotated[str, Field(description="The ID of the source entity")]
    target_id: Annotated[str, Field(description="The ID of the target entity")]
    relationship_type: Annotated[str, Field(description="The type of relationship")]
    metadata: Optional[Metadata] = Field(default_factory=create_metadata, description="Additional metadata")
    
    model_config = ConfigDict(
        validate_assignment=True,
        json_schema_extra={
            "examples": [
                {
                    "source_id": "entity-123",
                    "target_id": "entity-456",
                    "relationship_type": "DEPENDS_ON"
                }
            ]
        }
    )
    
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
    
    @computed_field
    def relationship_label(self) -> str:
        """Generate a human-readable label for this relationship."""
        return f"{self.source_id} --[{self.relationship_type}]--> {self.target_id}"
    
    @model_serializer
    def serialize_model(self) -> Dict[str, Any]:
        """Custom serialization for relationship."""
        data = self.model_dump(exclude_none=True)
        data["relationship_label"] = self.relationship_label
        return data


class SearchQuery(BaseModel):
    """Model for search queries."""
    query: Annotated[str, Field(description="The search query text")]
    entity_types: Optional[List[str]] = Field(None, description="Filter by entity types")
    relationship_types: Optional[List[str]] = Field(None, description="Filter by relationship types")
    tags: Optional[List[str]] = Field(None, description="Filter by tags")
    limit: Annotated[int, Field(default=10, description="Maximum number of results to return", ge=1, le=100)]
    confidence_threshold: Annotated[Optional[float], Field(None, description="Minimum confidence threshold", ge=0.0, le=1.0)]
    
    model_config = ConfigDict(
        validate_assignment=True,
        json_schema_extra={
            "examples": [
                {
                    "query": "authentication",
                    "entity_types": ["DECISION", "FEATURE"],
                    "limit": 20
                }
            ]
        }
    )
    
    @computed_field
    def has_filters(self) -> bool:
        """Check if this query has any filters applied."""
        return any([
            self.entity_types,
            self.relationship_types,
            self.tags,
            self.confidence_threshold is not None
        ])


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


class ProjectContainerResponse(SuccessResponse):
    """Response model for project container operations."""
    project_id: Optional[str] = Field(None, description="ID of the created or updated project")
    project: Optional[Dict[str, Any]] = Field(None, description="Project container data")
    
    model_config = ConfigDict(
        validate_assignment=True
    )


class ComponentResponse(SuccessResponse):
    """Response model for component operations."""
    component_id: Optional[str] = Field(None, description="ID of the created or updated component")
    component: Optional[Dict[str, Any]] = Field(None, description="Component data")
    
    model_config = ConfigDict(
        validate_assignment=True
    )


class SearchResponse(SuccessResponse):
    """Response model for search operations."""
    results: Annotated[List[Dict[str, Any]], Field(default_factory=list, description="Search results")]
    total_count: Annotated[int, Field(default=0, description="Total number of results found")]
    query: Annotated[str, Field(description="The search query that was executed")]
    
    model_config = ConfigDict(
        validate_assignment=True
    )
    
    @computed_field
    def has_results(self) -> bool:
        """Check if the search returned any results."""
        return self.total_count > 0
    
    @model_serializer
    def serialize_model(self) -> Dict[str, Any]:
        """Custom serialization with has_results field."""
        data = self.model_dump(exclude_none=True)
        data["has_results"] = self.has_results
        return data


# Version and Tag Models

class VersionCreate(BaseModel):
    """Model for creating a new version of a component."""
    component_name: Annotated[str, Field(description="Name of the component")]
    domain_name: Annotated[str, Field(description="Name of the domain")]
    container_name: Annotated[str, Field(description="Name of the project container")]
    version_number: Annotated[str, Field(description="Version number (e.g., '1.0.0')")]
    commit_hash: Optional[str] = Field(None, description="Commit hash from version control")
    content: Optional[str] = Field(None, description="Content of the component at this version")
    changes: Optional[str] = Field(None, description="Description of changes from previous version")
    metadata: Optional[Dict[str, Any]] = Field(None, description="Additional metadata")
    
    model_config = ConfigDict(
        validate_assignment=True,
        json_schema_extra={
            "examples": [
                {
                    "component_name": "Authentication Service",
                    "domain_name": "Backend",
                    "container_name": "E-commerce Platform",
                    "version_number": "1.0.0",
                    "commit_hash": "a1b2c3d4",
                    "changes": "Initial version"
                }
            ]
        }
    )

    @field_validator('version_number')
    @classmethod
    def validate_version_number(cls, v: str) -> str:
        """Validate version number format."""
        if not v or not isinstance(v, str):
            raise ValueError("Version number must be a non-empty string")
        return v

class VersionGetRequest(BaseModel):
    """Model for retrieving a version of a component."""
    component_name: Annotated[str, Field(description="Name of the component")]
    domain_name: Annotated[str, Field(description="Name of the domain")]
    container_name: Annotated[str, Field(description="Name of the project container")]
    version_number: Optional[str] = None
    
    model_config = ConfigDict(
        validate_assignment=True
    )

class VersionResponse(SuccessResponse):
    """Response model for version operations."""
    version_id: Optional[str] = Field(None, description="ID of the created or retrieved version")
    version: Optional[Dict[str, Any]] = Field(None, description="Version data")
    component_name: Optional[str] = Field(None, description="Name of the component")
    
    model_config = ConfigDict(
        validate_assignment=True
    )

class VersionListResponse(SuccessResponse):
    """Response model for listing versions."""
    component_name: str = Field(..., description="Name of the component")
    domain_name: str = Field(..., description="Name of the domain")
    container_name: str = Field(..., description="Name of the project container")
    version_count: int = Field(0, description="Number of versions")
    versions: List[Dict[str, Any]] = Field(default_factory=list, description="List of versions")
    
    model_config = ConfigDict(
        validate_assignment=True
    )

class TagCreate(BaseModel):
    """Model for creating a tag on a version."""
    component_name: Annotated[str, Field(description="Name of the component")]
    domain_name: Annotated[str, Field(description="Name of the domain")]
    container_name: Annotated[str, Field(description="Name of the project container")]
    version_number: Annotated[str, Field(description="Version number to tag")]
    tag_name: Annotated[str, Field(description="Name of the tag")]
    tag_description: Optional[str] = Field(None, description="Description of the tag")
    
    model_config = ConfigDict(
        validate_assignment=True
    )

class TagResponse(SuccessResponse):
    """Response model for tag operations."""
    tag_id: Optional[str] = Field(None, description="ID of the created tag")
    tag: Optional[Dict[str, Any]] = Field(None, description="Tag data")
    component_name: Optional[str] = Field(None, description="Name of the component")
    version_number: Optional[str] = Field(None, description="Version number")
    
    model_config = ConfigDict(
        validate_assignment=True
    )

# Additional models for version control operations

class CommitData(BaseModel):
    """Model for a commit from version control."""
    hash: str = Field(..., description="Commit hash")
    version: str = Field(..., description="Version number associated with the commit")
    date: Optional[str] = Field(None, description="Commit date")
    author: Optional[str] = Field(None, description="Commit author")
    message: Optional[str] = Field(None, description="Commit message")
    content: Optional[str] = Field(None, description="Content at this commit")
    
    model_config = ConfigDict(
        validate_assignment=True
    )

class SyncRequest(BaseModel):
    """Model for a version control sync request."""
    component_name: str = Field(..., description="Name of the component")
    domain_name: str = Field(..., description="Name of the domain")
    container_name: str = Field(..., description="Name of the project container")
    commits: List[CommitData] = Field(..., description="List of commits to sync")
    
    model_config = ConfigDict(
        validate_assignment=True
    )

class VersionCompareRequest(BaseModel):
    """Model for comparing two versions of a component."""
    component_name: Annotated[str, Field(description="Name of the component")]
    domain_name: Annotated[str, Field(description="Name of the domain")]
    container_name: Annotated[str, Field(description="Name of the project container")]
    version1: Annotated[str, Field(description="First version number to compare")]
    version2: Annotated[str, Field(description="Second version number to compare")]
    
    model_config = ConfigDict(
        validate_assignment=True
    )
    
    @model_validator(mode='after')
    @classmethod
    def validate_versions(cls, values):
        """Ensure both version numbers are provided and valid."""
        if not values.version1 or not values.version2:
            raise ValueError("Both version numbers must be provided")
        return values 