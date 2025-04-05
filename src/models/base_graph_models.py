#!/usr/bin/env python3
"""
Base Graph Models

This module provides common base models for entity/relationship systems that can be
used across different domains (project memory, lesson memory, etc.).
"""

from typing import Dict, List, Optional, Any, Set, ClassVar, Union, Type
from datetime import datetime
from enum import Enum
from pydantic import BaseModel, Field, field_validator, model_validator, create_model

# Common Metadata model that can be extended by different domains
class BaseMetadata(BaseModel):
    """Base metadata model for all graph entities and relationships."""
    client_id: Optional[str] = Field(None, description="Client identifier")
    created_at: datetime = Field(default_factory=datetime.now, description="Creation timestamp")
    updated_at: Optional[datetime] = Field(None, description="Last update timestamp")
    source: Optional[str] = Field(None, description="Source of the data")
    confidence: Optional[float] = Field(None, ge=0.0, le=1.0, description="Confidence score (0.0-1.0)")
    additional_info: Optional[Dict[str, Any]] = Field(None, description="Additional metadata")


# Base Entity model
class BaseEntity(BaseModel):
    """Base model for all entity types across different domains."""
    id: Optional[str] = Field(None, description="Unique identifier")
    name: str = Field(..., description="Name of the entity")
    entity_type: str = Field(..., description="Type of the entity")
    domain: str = Field(..., description="Domain this entity belongs to (e.g., 'project', 'lesson')")
    metadata: Optional[BaseMetadata] = Field(None, description="Entity metadata")
    
    # Class variable to store allowed entity types - can be overridden by subclasses
    allowed_entity_types: ClassVar[Set[str]] = set()

    @model_validator(mode='after')
    def validate_entity_type(cls, values):
        """Validate entity type if allowed_entity_types is defined in the subclass."""
        entity = values
        if entity.allowed_entity_types and entity.entity_type not in entity.allowed_entity_types:
            raise ValueError(f"Entity type '{entity.entity_type}' not allowed. Must be one of: {entity.allowed_entity_types}")
        return entity


# Base Relationship model
class BaseRelationship(BaseModel):
    """Base model for all relationships between entities."""
    id: Optional[str] = Field(None, description="Unique identifier")
    source_id: str = Field(..., description="ID of the source entity")
    target_id: str = Field(..., description="ID of the target entity")
    relationship_type: str = Field(..., description="Type of relationship")
    domain: str = Field(..., description="Domain this relationship belongs to")
    properties: Optional[Dict[str, Any]] = Field(None, description="Additional properties")
    metadata: Optional[BaseMetadata] = Field(None, description="Relationship metadata")
    
    # Class variable to store allowed relationship types - can be overridden by subclasses
    allowed_relationship_types: ClassVar[Set[str]] = set()

    @model_validator(mode='after')
    def validate_relationship_type(cls, values):
        """Validate relationship type if allowed_relationship_types is defined in the subclass."""
        relationship = values
        if relationship.allowed_relationship_types and relationship.relationship_type not in relationship.allowed_relationship_types:
            raise ValueError(
                f"Relationship type '{relationship.relationship_type}' not allowed. "
                f"Must be one of: {relationship.allowed_relationship_types}"
            )
        return relationship


# Base Container model
class BaseContainer(BaseModel):
    """Base model for containers that group entities."""
    id: Optional[str] = Field(None, description="Unique identifier")
    name: str = Field(..., description="Name of the container")
    description: Optional[str] = Field(None, description="Description of the container")
    domain: str = Field(..., description="Domain this container belongs to")
    entities: List[str] = Field(default_factory=list, description="IDs of entities in this container")
    tags: Optional[List[str]] = Field(default_factory=list, description="Tags for the container")
    metadata: Optional[BaseMetadata] = Field(None, description="Container metadata")


# Base Observation model
class BaseObservation(BaseModel):
    """Base model for observations attached to entities."""
    id: Optional[str] = Field(None, description="Unique identifier")
    entity_id: str = Field(..., description="ID of the entity this observation belongs to")
    content: str = Field(..., description="Content of the observation")
    observation_type: str = Field(..., description="Type of observation")
    created_at: datetime = Field(default_factory=datetime.now, description="Creation timestamp")
    updated_at: Optional[datetime] = Field(None, description="Last update timestamp")
    metadata: Optional[BaseMetadata] = Field(None, description="Observation metadata")
    
    # Class variable to store allowed observation types - can be overridden by subclasses
    allowed_observation_types: ClassVar[Set[str]] = set()

    @model_validator(mode='after')
    def validate_observation_type(cls, values):
        """Validate observation type if allowed_observation_types is defined in the subclass."""
        observation = values
        if observation.allowed_observation_types and observation.observation_type not in observation.allowed_observation_types:
            raise ValueError(
                f"Observation type '{observation.observation_type}' not allowed. "
                f"Must be one of: {observation.allowed_observation_types}"
            )
        return observation


# Base models for entity and relationship creation
class BaseEntityCreate(BaseModel):
    """Base model for creating new entities."""
    name: str = Field(..., description="Name of the entity")
    entity_type: str = Field(..., description="Type of the entity")
    container_id: Optional[str] = Field(None, description="ID of the container this entity belongs to")
    description: Optional[str] = Field(None, description="Description of the entity")
    metadata: Optional[Dict[str, Any]] = Field(None, description="Entity metadata")
    
    # Class variable to store allowed entity types - can be overridden by subclasses
    allowed_entity_types: ClassVar[Set[str]] = set()

    @model_validator(mode='after')
    def validate_entity_type(cls, values):
        """Validate entity type if allowed_entity_types is defined in the subclass."""
        entity = values
        if entity.allowed_entity_types and entity.entity_type not in entity.allowed_entity_types:
            raise ValueError(f"Entity type '{entity.entity_type}' not allowed. Must be one of: {entity.allowed_entity_types}")
        return entity


class BaseRelationshipCreate(BaseModel):
    """Base model for creating new relationships."""
    source_id: str = Field(..., description="ID of the source entity")
    target_id: str = Field(..., description="ID of the target entity")
    relationship_type: str = Field(..., description="Type of relationship")
    properties: Optional[Dict[str, Any]] = Field(None, description="Additional properties")
    metadata: Optional[Dict[str, Any]] = Field(None, description="Relationship metadata")
    
    # Class variable to store allowed relationship types - can be overridden by subclasses
    allowed_relationship_types: ClassVar[Set[str]] = set()

    @model_validator(mode='after')
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
    entity_id: str = Field(..., description="ID of the entity to update")
    name: Optional[str] = Field(None, description="Updated name")
    description: Optional[str] = Field(None, description="Updated description")
    entity_type: Optional[str] = Field(None, description="Updated entity type")
    tags: Optional[List[str]] = Field(None, description="Updated tags")
    metadata: Optional[Dict[str, Any]] = Field(None, description="Updated metadata")
    
    # Class variable to store allowed entity types - can be overridden by subclasses
    allowed_entity_types: ClassVar[Set[str]] = set()

    @model_validator(mode='after')
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
    limit: int = Field(10, description="Maximum number of results", ge=1, le=100)
    confidence_threshold: Optional[float] = Field(None, description="Minimum confidence threshold", ge=0.0, le=1.0)
    semantic: bool = Field(False, description="Whether to use semantic search")


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
    
    # Set class variables
    model_class.allowed_entity_types = set(allowed_types)
    model_class.__annotations__['domain'] = str
    model_class.__fields__['domain'].default = domain
    
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
    
    # Set class variables
    model_class.allowed_relationship_types = set(allowed_types)
    model_class.__annotations__['domain'] = str
    model_class.__fields__['domain'].default = domain
    
    return model_class 