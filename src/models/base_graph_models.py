#!/usr/bin/env python3
"""
Base Graph Models

This module provides common base models for entity/relationship systems that can be
used across different domains (project memory, lesson memory, etc.).
"""

from typing import Dict, List, Optional, Any, Set, ClassVar, Union, Type, Annotated
from datetime import datetime
from enum import Enum
from pydantic import (
    BaseModel, 
    Field, 
    field_validator, 
    model_validator, 
    create_model,
    ConfigDict,
    computed_field,
    model_serializer
)

# Common Metadata model that can be extended by different domains
class BaseMetadata(BaseModel):
    """Base metadata model for all graph entities and relationships."""
    client_id: Optional[str] = Field(None, description="Client identifier")
    created_at: Annotated[datetime, Field(default_factory=datetime.now, description="Creation timestamp")]
    updated_at: Optional[datetime] = Field(None, description="Last update timestamp")
    source: Optional[str] = Field(None, description="Source of the data")
    confidence: Annotated[Optional[float], Field(None, ge=0.0, le=1.0, description="Confidence score (0.0-1.0)")]
    additional_info: Optional[Dict[str, Any]] = Field(None, description="Additional metadata")
    
    model_config = ConfigDict(
        validate_assignment=True,
        json_schema_extra={
            "examples": [
                {
                    "client_id": "app1",
                    "created_at": "2023-01-01T00:00:00Z",
                    "confidence": 0.85,
                    "source": "system"
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
        # Add computed fields
        if self.created_at:
            data["age"] = self.age
        return data


# Base Entity model
class BaseEntity(BaseModel):
    """Base model for all entity types across different domains."""
    id: Optional[str] = Field(None, description="Unique identifier")
    name: Annotated[str, Field(description="Name of the entity")]
    entity_type: Annotated[str, Field(description="Type of the entity")]
    domain: Annotated[str, Field(description="Domain this entity belongs to (e.g., 'project', 'lesson')")]
    metadata: Optional[BaseMetadata] = Field(None, description="Entity metadata")
    
    # Class variable to store allowed entity types - can be overridden by subclasses
    allowed_entity_types: ClassVar[Set[str]] = set()
    
    model_config = ConfigDict(
        validate_assignment=True,
        json_schema_extra={
            "examples": [
                {
                    "id": "entity-123",
                    "name": "ExampleEntity",
                    "entity_type": "DOCUMENT",
                    "domain": "knowledge"
                }
            ]
        }
    )

    @model_validator(mode='after')
    @classmethod
    def validate_entity_type(cls, values):
        """Validate entity type if allowed_entity_types is defined in the subclass."""
        entity = values
        if entity.allowed_entity_types and entity.entity_type not in entity.allowed_entity_types:
            raise ValueError(f"Entity type '{entity.entity_type}' not allowed. Must be one of: {entity.allowed_entity_types}")
        return entity
    
    @computed_field
    def qualified_name(self) -> str:
        """Generate a qualified name combining domain, type and name."""
        return f"{self.domain}:{self.entity_type}:{self.name}"
    
    @model_serializer
    def serialize_model(self) -> Dict[str, Any]:
        """Custom serialization that includes qualified_name."""
        data = self.model_dump(exclude_none=True)
        # Add computed fields
        data["qualified_name"] = self.qualified_name
        return data


# Base Relationship model
class BaseRelationship(BaseModel):
    """Base model for all relationships between entities."""
    id: Optional[str] = Field(None, description="Unique identifier")
    source_id: Annotated[str, Field(description="ID of the source entity")]
    target_id: Annotated[str, Field(description="ID of the target entity")]
    relationship_type: Annotated[str, Field(description="Type of relationship")]
    domain: Annotated[str, Field(description="Domain this relationship belongs to")]
    properties: Optional[Dict[str, Any]] = Field(None, description="Additional properties")
    metadata: Optional[BaseMetadata] = Field(None, description="Relationship metadata")
    
    # Class variable to store allowed relationship types - can be overridden by subclasses
    allowed_relationship_types: ClassVar[Set[str]] = set()
    
    model_config = ConfigDict(
        validate_assignment=True,
        json_schema_extra={
            "examples": [
                {
                    "id": "rel-abc",
                    "source_id": "entity-123",
                    "target_id": "entity-456",
                    "relationship_type": "CONNECTS_TO",
                    "domain": "knowledge"
                }
            ]
        }
    )

    @model_validator(mode='after')
    @classmethod
    def validate_relationship_type(cls, values):
        """Validate relationship type if allowed_relationship_types is defined in the subclass."""
        relationship = values
        if relationship.allowed_relationship_types and relationship.relationship_type not in relationship.allowed_relationship_types:
            raise ValueError(
                f"Relationship type '{relationship.relationship_type}' not allowed. "
                f"Must be one of: {relationship.allowed_relationship_types}"
            )
        return relationship
    
    @computed_field
    def relationship_signature(self) -> str:
        """Generate a signature describing this relationship."""
        return f"{self.source_id} --[{self.relationship_type}]--> {self.target_id}"
    
    @model_serializer
    def serialize_model(self) -> Dict[str, Any]:
        """Custom serialization with relationship signature."""
        data = self.model_dump(exclude_none=True)
        # Add computed fields
        data["relationship_signature"] = self.relationship_signature
        return data


# Base Container model
class BaseContainer(BaseModel):
    """Base model for containers that group entities."""
    id: Optional[str] = Field(None, description="Unique identifier")
    name: Annotated[str, Field(description="Name of the container")]
    description: Optional[str] = Field(None, description="Description of the container")
    domain: Annotated[str, Field(description="Domain this container belongs to")]
    entities: Annotated[List[str], Field(default_factory=list, description="IDs of entities in this container")]
    tags: Optional[List[str]] = Field(default_factory=list, description="Tags for the container")
    metadata: Optional[BaseMetadata] = Field(None, description="Container metadata")
    
    model_config = ConfigDict(
        validate_assignment=True,
        json_schema_extra={
            "examples": [
                {
                    "id": "container-001",
                    "name": "Main Knowledge Base",
                    "domain": "knowledge",
                    "entities": ["entity-123", "entity-456"],
                    "tags": ["knowledge", "main"]
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
        # Add computed fields
        data["entity_count"] = self.entity_count
        return data


# Base Observation model
class BaseObservation(BaseModel):
    """Base model for observations attached to entities."""
    id: Optional[str] = Field(None, description="Unique identifier")
    entity_id: Annotated[str, Field(description="ID of the entity this observation belongs to")]
    content: Annotated[str, Field(description="Content of the observation")]
    observation_type: Annotated[str, Field(description="Type of observation")]
    created_at: Annotated[datetime, Field(default_factory=datetime.now, description="Creation timestamp")]
    updated_at: Optional[datetime] = Field(None, description="Last update timestamp")
    metadata: Optional[BaseMetadata] = Field(None, description="Observation metadata")
    
    # Class variable to store allowed observation types - can be overridden by subclasses
    allowed_observation_types: ClassVar[Set[str]] = set()
    
    model_config = ConfigDict(
        validate_assignment=True,
        json_schema_extra={
            "examples": [
                {
                    "id": "obs-123",
                    "entity_id": "entity-456",
                    "content": "This entity demonstrates key concepts in the domain",
                    "observation_type": "INSIGHT"
                }
            ]
        }
    )

    @model_validator(mode='after')
    @classmethod
    def validate_observation_type(cls, values):
        """Validate observation type if allowed_observation_types is defined in the subclass."""
        observation = values
        if observation.allowed_observation_types and observation.observation_type not in observation.allowed_observation_types:
            raise ValueError(
                f"Observation type '{observation.observation_type}' not allowed. "
                f"Must be one of: {observation.allowed_observation_types}"
            )
        return observation
    
    @computed_field
    def content_preview(self) -> str:
        """Get a preview of the observation content."""
        if not self.content:
            return ""
        max_length = 50
        if len(self.content) <= max_length:
            return self.content
        return self.content[:max_length] + "..."
    
    @model_serializer
    def serialize_model(self) -> Dict[str, Any]:
        """Custom serialization including content preview."""
        data = self.model_dump(exclude_none=True)
        # Add computed fields
        data["content_preview"] = self.content_preview
        return data


# Base models for entity and relationship creation
class BaseEntityCreate(BaseModel):
    """Base model for creating new entities."""
    name: Annotated[str, Field(description="Name of the entity")]
    entity_type: Annotated[str, Field(description="Type of the entity")]
    container_id: Optional[str] = Field(None, description="ID of the container this entity belongs to")
    description: Optional[str] = Field(None, description="Description of the entity")
    metadata: Optional[Dict[str, Any]] = Field(None, description="Entity metadata")
    
    # Class variable to store allowed entity types - can be overridden by subclasses
    allowed_entity_types: ClassVar[Set[str]] = set()
    
    model_config = ConfigDict(
        validate_assignment=True,
        json_schema_extra={
            "examples": [
                {
                    "name": "New Entity",
                    "entity_type": "DOCUMENT",
                    "container_id": "container-001",
                    "description": "A new entity to be created"
                }
            ]
        }
    )

    @model_validator(mode='after')
    @classmethod
    def validate_entity_type(cls, values):
        """Validate entity type if allowed_entity_types is defined in the subclass."""
        entity = values
        if entity.allowed_entity_types and entity.entity_type not in entity.allowed_entity_types:
            raise ValueError(f"Entity type '{entity.entity_type}' not allowed. Must be one of: {entity.allowed_entity_types}")
        return entity


class BaseRelationshipCreate(BaseModel):
    """Base model for creating new relationships."""
    source_id: Annotated[str, Field(description="ID of the source entity")]
    target_id: Annotated[str, Field(description="ID of the target entity")]
    relationship_type: Annotated[str, Field(description="Type of relationship")]
    properties: Optional[Dict[str, Any]] = Field(None, description="Additional properties")
    metadata: Optional[Dict[str, Any]] = Field(None, description="Relationship metadata")
    
    # Class variable to store allowed relationship types - can be overridden by subclasses
    allowed_relationship_types: ClassVar[Set[str]] = set()
    
    model_config = ConfigDict(
        validate_assignment=True,
        json_schema_extra={
            "examples": [
                {
                    "source_id": "entity-123",
                    "target_id": "entity-456",
                    "relationship_type": "CONNECTS_TO",
                    "properties": {"weight": 0.8, "label": "strong connection"}
                }
            ]
        }
    )

    @model_validator(mode='after')
    @classmethod
    def validate_relationship_type(cls, values):
        """Validate relationship type if allowed_relationship_types is defined in the subclass."""
        relationship = values
        if relationship.allowed_relationship_types and relationship.relationship_type not in relationship.allowed_relationship_types:
            raise ValueError(
                f"Relationship type '{relationship.relationship_type}' not allowed. "
                f"Must be one of: {relationship.allowed_relationship_types}"
            )
        return relationship


# Base update models
class BaseEntityUpdate(BaseModel):
    """Base model for updating entities."""
    entity_id: Annotated[str, Field(description="ID of the entity to update")]
    name: Optional[str] = Field(None, description="Updated name")
    description: Optional[str] = Field(None, description="Updated description")
    entity_type: Optional[str] = Field(None, description="Updated entity type")
    tags: Optional[List[str]] = Field(None, description="Updated tags")
    metadata: Optional[Dict[str, Any]] = Field(None, description="Updated metadata")
    
    # Class variable to store allowed entity types - can be overridden by subclasses
    allowed_entity_types: ClassVar[Set[str]] = set()
    
    model_config = ConfigDict(
        validate_assignment=True,
        json_schema_extra={
            "examples": [
                {
                    "entity_id": "entity-123",
                    "name": "Updated Entity Name",
                    "tags": ["updated", "modified"]
                }
            ]
        }
    )

    @model_validator(mode='after')
    @classmethod
    def validate_entity_type(cls, values):
        """Validate entity type if allowed_entity_types is defined and entity_type is provided."""
        entity = values
        if entity.entity_type and entity.allowed_entity_types and entity.entity_type not in entity.allowed_entity_types:
            raise ValueError(f"Entity type '{entity.entity_type}' not allowed. Must be one of: {entity.allowed_entity_types}")
        return entity


# Base Search Query
class BaseSearchQuery(BaseModel):
    """Base model for search queries."""
    query: Optional[str] = Field(None, description="Search query text")
    entity_types: Optional[List[str]] = Field(None, description="Filter by entity types")
    relationship_types: Optional[List[str]] = Field(None, description="Filter by relationship types")
    domain: Optional[str] = Field(None, description="Filter by domain")
    container_id: Optional[str] = Field(None, description="Filter by container")
    tags: Optional[List[str]] = Field(None, description="Filter by tags")
    limit: Annotated[int, Field(default=10, description="Maximum number of results", ge=1, le=100)]
    confidence_threshold: Annotated[Optional[float], Field(None, description="Minimum confidence threshold", ge=0.0, le=1.0)]
    semantic: Annotated[bool, Field(default=False, description="Whether to use semantic search")]
    
    model_config = ConfigDict(
        validate_assignment=True,
        json_schema_extra={
            "examples": [
                {
                    "query": "knowledge graph",
                    "entity_types": ["DOCUMENT", "CONCEPT"],
                    "domain": "knowledge",
                    "limit": 20,
                    "semantic": True
                }
            ]
        }
    )
    
    @computed_field
    def has_filters(self) -> bool:
        """Check if this query has any active filters."""
        return any([
            self.entity_types, 
            self.relationship_types,
            self.domain,
            self.container_id,
            self.tags,
            self.confidence_threshold is not None
        ])
    
    @model_serializer
    def serialize_model(self) -> Dict[str, Any]:
        """Custom serialization with filter information."""
        data = self.model_dump(exclude_none=True)
        # Add computed fields
        data["has_filters"] = self.has_filters
        return data


# Helper functions to create domain-specific models
def create_domain_entity_model(
    name: str,
    allowed_types: List[str],
    domain: str,
    extra_fields: Optional[Dict[str, Any]] = None
):
    """
    Create a domain-specific entity model.
    
    Args:
        name: Name of the model class
        allowed_types: List of allowed entity types
        domain: Domain this entity belongs to
        extra_fields: Additional fields specific to this domain
        
    Returns:
        A new BaseEntity subclass with domain-specific configurations
    """
    fields = {}
    if extra_fields:
        fields.update(extra_fields)
    
    # Create the new model class
    model_class = create_model(
        name,
        __base__=BaseEntity,
        **fields
    )
    
    # Set class variables and defaults for Pydantic v2
    model_class.allowed_entity_types = set(allowed_types)
    
    # Update domain default in model_fields (v2 way) instead of __fields__ (v1 way)
    if hasattr(model_class, 'model_fields') and 'domain' in model_class.model_fields:
        model_class.model_fields['domain'].default = domain
    
    # Define a model config with proper configuration
    model_class.model_config = ConfigDict(
        validate_assignment=True,
        extra='ignore'
    )
    
    return model_class


def create_domain_relationship_model(
    name: str,
    allowed_types: List[str],
    domain: str,
    extra_fields: Optional[Dict[str, Any]] = None
):
    """
    Create a domain-specific relationship model.
    
    Args:
        name: Name of the model class
        allowed_types: List of allowed relationship types
        domain: Domain this relationship belongs to
        extra_fields: Additional fields specific to this domain
        
    Returns:
        A new BaseRelationship subclass with domain-specific configurations
    """
    fields = {}
    if extra_fields:
        fields.update(extra_fields)
    
    # Create the new model class
    model_class = create_model(
        name,
        __base__=BaseRelationship,
        **fields
    )
    
    # Set class variables and defaults for Pydantic v2
    model_class.allowed_relationship_types = set(allowed_types)
    
    # Update domain default in model_fields (v2 way) instead of __fields__ (v1 way)
    if hasattr(model_class, 'model_fields') and 'domain' in model_class.model_fields:
        model_class.model_fields['domain'].default = domain
    
    # Define a model config with proper configuration
    model_class.model_config = ConfigDict(
        validate_assignment=True,
        extra='ignore'
    )
    
    return model_class 