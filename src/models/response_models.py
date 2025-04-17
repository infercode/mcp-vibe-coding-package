"""
Standard Response Models

This module provides standardized Pydantic models for API responses
that are used across different components of the MCP system.
"""

from typing import Dict, List, Optional, Any, Union, Callable, Annotated
from datetime import datetime
from pydantic import BaseModel, Field, ConfigDict, computed_field, model_serializer

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
    data: Optional[Dict[str, Any]] = Field(None, description="Response data")
    
    model_config = ConfigDict(
        validate_assignment=True,
        json_schema_extra={
            "examples": [
                {
                    "status": "success",
                    "timestamp": "2023-01-01T12:00:00Z",
                    "message": "Operation completed successfully"
                }
            ]
        }
    )


class ErrorDetail(BaseModel):
    """Error detail model for standardized error responses."""
    code: str = Field(..., description="Error code")
    message: str = Field(..., description="Error message")
    details: Optional[Dict[str, Any]] = Field(None, description="Additional error details")
    
    model_config = ConfigDict(
        validate_assignment=True,
        json_schema_extra={
            "examples": [
                {
                    "code": "validation_error",
                    "message": "Invalid input data",
                    "details": {"field": "name", "error": "Field cannot be empty"}
                }
            ]
        }
    )


class ErrorResponse(BaseModel):
    """Error response model."""
    status: str = "error"
    timestamp: datetime = Field(default_factory=datetime.now)
    error: ErrorDetail = Field(..., description="Error details")
    
    model_config = ConfigDict(
        validate_assignment=True,
        json_schema_extra={
            "examples": [
                {
                    "status": "error",
                    "timestamp": "2023-01-01T12:00:00Z",
                    "error": {
                        "code": "validation_error",
                        "message": "Invalid input data",
                        "details": {"field": "name", "error": "Field cannot be empty"}
                    }
                }
            ]
        }
    )


def create_error_response(
    message: str, 
    code: str = "internal_error", 
    details: Optional[Dict[str, Any]] = None
) -> ErrorResponse:
    """
    Create a standardized error response using Pydantic models.
    
    Args:
        message: Error message
        code: Error code
        details: Optional additional error details
        
    Returns:
        ErrorResponse model instance
    """
    error_detail = ErrorDetail(
        code=code,
        message=message,
        details=details
    )
    
    return ErrorResponse(
        status="error",
        timestamp=datetime.now(),
        error=error_detail
    )


def create_success_response(
    message: Optional[str] = None,
    data: Optional[Dict[str, Any]] = None
) -> SuccessResponse:
    """
    Create a standardized success response using Pydantic models.
    
    Args:
        message: Optional success message
        data: Optional additional data
        
    Returns:
        SuccessResponse model instance
    """
    return SuccessResponse(
        status="success",
        timestamp=datetime.now(),
        message=message,
        data=data
    )


# Context Models

class LessonContextModel(BaseModel):
    """Pydantic model for lesson memory context."""
    container_name: Optional[str] = Field(None, description="Container name to use for operations")
    project_name: Optional[str] = Field(None, description="Project name for context")
    created_at: datetime = Field(default_factory=datetime.now, description="Context creation timestamp")
    operations_available: List[str] = Field(
        default_factory=lambda: [
            "create_container", "get_container", "list_containers", "container_exists", 
            "create", "observe", "relate", "search", "track", "update", "consolidate", "evolve"
        ],
        description="List of available operations in this context"
    )
    
    model_config = ConfigDict(
        validate_assignment=True,
        json_schema_extra={
            "examples": [
                {
                    "container_name": "Lessons",
                    "project_name": "Project1",
                    "created_at": "2023-01-01T12:00:00Z"
                }
            ]
        }
    )
    
    @computed_field
    def usage(self) -> str:
        """Usage information for this context."""
        return "Use this context for batch lesson memory operations with shared container and project settings"

    @model_serializer
    def serialize_model(self) -> Dict[str, Any]:
        """Custom serialization for lesson context."""
        data = self.model_dump(exclude_none=True)
        data["usage"] = self.usage
        return data


class ProjectContextModel(BaseModel):
    """Pydantic model for project memory context."""
    project_name: Optional[str] = Field(None, description="Project name to use for operations")
    created_at: datetime = Field(default_factory=datetime.now, description="Context creation timestamp")
    operations_available: List[str] = Field(
        default_factory=lambda: [
            "create_component", "create_domain_entity", "relate", "search", 
            "get_structure", "add_observation", "update", "delete_entity", "delete_relationship"
        ],
        description="List of available operations in this context"
    )
    
    model_config = ConfigDict(
        validate_assignment=True,
        json_schema_extra={
            "examples": [
                {
                    "project_name": "E-commerce Platform",
                    "created_at": "2023-01-01T12:00:00Z"
                }
            ]
        }
    )
    
    @computed_field
    def usage(self) -> str:
        """Usage information for this context."""
        return "Use this context for batch project memory operations with shared project settings"

    @model_serializer
    def serialize_model(self) -> Dict[str, Any]:
        """Custom serialization for project context."""
        data = self.model_dump(exclude_none=True)
        data["usage"] = self.usage
        return data


class LessonContextResponse(SuccessResponse):
    """Response model for lesson context operations."""
    context: LessonContextModel = Field(..., description="Lesson context information")
    
    model_config = ConfigDict(
        validate_assignment=True,
        json_schema_extra={
            "examples": [
                {
                    "status": "success",
                    "timestamp": "2023-01-01T12:00:00Z",
                    "message": "Lesson memory context created for project 'ProjectName' and container 'ContainerName'",
                    "context": {
                        "project_name": "ProjectName",
                        "container_name": "ContainerName",
                        "created_at": "2023-01-01T12:00:00Z"
                    }
                }
            ]
        }
    )


class ProjectContextResponse(SuccessResponse):
    """Response model for project context operations."""
    context: ProjectContextModel = Field(..., description="Project context information")
    
    model_config = ConfigDict(
        validate_assignment=True,
        json_schema_extra={
            "examples": [
                {
                    "status": "success",
                    "timestamp": "2023-01-01T12:00:00Z",
                    "message": "Project memory context created for project 'ProjectName'",
                    "context": {
                        "project_name": "ProjectName",
                        "created_at": "2023-01-01T12:00:00Z"
                    }
                }
            ]
        }
    )


# Search Models

class SearchQueryBase(BaseModel):
    """Base model for search queries."""
    query: Optional[str] = Field(None, description="Search query text")
    limit: int = Field(default=10, description="Maximum number of results to return", ge=1, le=100)
    semantic: bool = Field(default=True, description="Whether to use semantic search")
    confidence_threshold: Optional[float] = Field(None, description="Minimum confidence threshold", ge=0.0, le=1.0)
    
    model_config = ConfigDict(
        validate_assignment=True,
        json_schema_extra={
            "examples": [
                {
                    "query": "database optimization techniques",
                    "limit": 5,
                    "semantic": True,
                    "confidence_threshold": 0.6
                }
            ]
        }
    )
    
    @computed_field
    def has_query(self) -> bool:
        """Check if query parameter is provided."""
        return self.query is not None and self.query.strip() != ""


class LessonSearchQuery(SearchQueryBase):
    """Model for lesson memory search queries."""
    container_name: Optional[str] = Field(None, description="Container name to search in")
    entity_types: Optional[List[str]] = Field(None, description="Filter by entity types")
    tags: Optional[List[str]] = Field(None, description="Filter by tags")
    
    model_config = ConfigDict(
        validate_assignment=True,
        json_schema_extra={
            "examples": [
                {
                    "query": "error handling best practices",
                    "container_name": "Lessons",
                    "entity_types": ["BestPractice", "Pattern"],
                    "tags": ["error-handling", "robust-code"]
                }
            ]
        }
    )
    
    @computed_field
    def has_filters(self) -> bool:
        """Check if any filters are applied."""
        return any([
            self.container_name is not None,
            self.entity_types is not None and len(self.entity_types) > 0,
            self.tags is not None and len(self.tags) > 0
        ])


class ProjectSearchQuery(SearchQueryBase):
    """Model for project memory search queries."""
    project_id: Optional[str] = Field(None, description="Project ID to search in")
    entity_types: Optional[List[str]] = Field(None, description="Filter by entity types")
    relationship_types: Optional[List[str]] = Field(None, description="Filter by relationship types")
    tags: Optional[List[str]] = Field(None, description="Filter by tags")
    
    model_config = ConfigDict(
        validate_assignment=True,
        json_schema_extra={
            "examples": [
                {
                    "query": "authentication service",
                    "project_id": "e-commerce-platform",
                    "entity_types": ["SERVICE", "COMPONENT"]
                }
            ]
        }
    )
    
    @computed_field
    def has_filters(self) -> bool:
        """Check if any filters are applied."""
        return any([
            self.project_id is not None,
            self.entity_types is not None and len(self.entity_types) > 0,
            self.relationship_types is not None and len(self.relationship_types) > 0,
            self.tags is not None and len(self.tags) > 0
        ])


class SearchResultItem(BaseModel):
    """Model for a single search result item."""
    id: str = Field(..., description="ID of the entity")
    name: str = Field(..., description="Name of the entity")
    entity_type: str = Field(..., description="Type of the entity")
    score: Optional[float] = Field(None, description="Relevance score")
    confidence: Optional[float] = Field(None, description="Confidence score")
    snippet: Optional[str] = Field(None, description="Snippet of matching content")
    metadata: Optional[Dict[str, Any]] = Field(None, description="Additional metadata")
    
    model_config = ConfigDict(
        validate_assignment=True,
        json_schema_extra={
            "examples": [
                {
                    "id": "entity-123",
                    "name": "DatabaseOptimization",
                    "entity_type": "BestPractice",
                    "score": 0.92,
                    "confidence": 0.85,
                    "snippet": "Indexes should be created for frequently queried columns..."
                }
            ]
        }
    )


class SearchResponse(SuccessResponse):
    """Response model for search operations."""
    results: List[SearchResultItem] = Field(default_factory=list, description="Search results")
    total_count: int = Field(default=0, description="Total number of results found")
    query: Dict[str, Any] = Field(..., description="The search query parameters used")
    is_semantic: bool = Field(default=False, description="Whether semantic search was used")
    
    model_config = ConfigDict(
        validate_assignment=True,
        json_schema_extra={
            "examples": [
                {
                    "status": "success",
                    "timestamp": "2023-01-01T12:00:00Z",
                    "message": "Search completed successfully",
                    "results": [
                        {
                            "id": "entity-123",
                            "name": "DatabaseOptimization",
                            "entity_type": "BestPractice",
                            "score": 0.92
                        }
                    ],
                    "total_count": 1,
                    "query": {
                        "text": "database optimization",
                        "limit": 10,
                        "semantic": True
                    },
                    "is_semantic": True
                }
            ]
        }
    )
    
    @computed_field
    def has_results(self) -> bool:
        """Check if any results were found."""
        return len(self.results) > 0
    
    @model_serializer
    def serialize_model(self) -> Dict[str, Any]:
        """Custom serialization for search response."""
        data = self.model_dump(exclude_none=True)
        data["has_results"] = self.has_results
        return data


class VectorSearchOptions(BaseModel):
    """Options for vector/semantic search."""
    embedding_dimensions: int = Field(default=1536, description="Dimensions of the embedding vectors")
    distance_metric: str = Field(default="cosine", description="Distance metric for similarity calculation")
    score_threshold: float = Field(default=0.7, description="Minimum similarity score threshold", ge=0.0, le=1.0)
    
    model_config = ConfigDict(
        validate_assignment=True,
        json_schema_extra={
            "examples": [
                {
                    "embedding_dimensions": 1536,
                    "distance_metric": "cosine", 
                    "score_threshold": 0.75
                }
            ]
        }
    ) 