#!/usr/bin/env python3
"""
Base Graph Models Example

This script demonstrates how to use the base graph models to create
domain-specific models and validate entity and relationship types.
"""

import sys
import os
from datetime import datetime
from typing import Dict, List, Optional, Any, Set, cast

# Add parent directory to path to allow imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from src.models.base_graph_models import (
    BaseEntity, BaseRelationship, BaseContainer, BaseObservation,
    BaseMetadata, BaseEntityCreate, BaseRelationshipCreate,
    BaseEntityUpdate, BaseSearchQuery,
    create_domain_entity_model, create_domain_relationship_model
)


def create_custom_domain_models():
    """Create custom domain models for a specific use case."""
    print("\n=== Example: Creating Custom Domain Models ===")
    
    # Define allowed types for a "knowledge" domain
    knowledge_entity_types = ["CONCEPT", "TOPIC", "FACT", "DEFINITION", "THEORY"]
    knowledge_relationship_types = ["RELATED_TO", "PREREQUISITE_FOR", "PART_OF", "APPLIES_TO", "CONTRADICTS"]
    
    # Create domain-specific entity model with custom fields
    # Note: This uses create_model which requires different param formatting
    # Extra fields need to be (type, default_value) tuples
    KnowledgeEntity = create_domain_entity_model(
        name="KnowledgeEntity",
        allowed_types=knowledge_entity_types,
        domain="knowledge",
        extra_fields={
            "difficulty_level": (Optional[str], None),
            "keywords": (List[str], [])
        }
    )
    
    # Create domain-specific relationship model
    KnowledgeRelationship = create_domain_relationship_model(
        name="KnowledgeRelationship",
        allowed_types=knowledge_relationship_types,
        domain="knowledge",
        extra_fields={
            "strength": (float, 0.5),
            "bidirectional": (bool, False)
        }
    )
    
    print(f"✅ Created custom domain entity model: KnowledgeEntity")
    print(f"   Allowed types: {KnowledgeEntity.allowed_entity_types}")
    print(f"   Domain: {KnowledgeEntity.model_fields['domain'].default}")
    
    print(f"✅ Created custom domain relationship model: KnowledgeRelationship")
    print(f"   Allowed types: {KnowledgeRelationship.allowed_relationship_types}")
    print(f"   Domain: {KnowledgeRelationship.model_fields['domain'].default}")
    
    # Create an instance of KnowledgeEntity
    try:
        # Create a completely filled metadata object to avoid linter warnings
        metadata = BaseMetadata(
            client_id="example-client",
            created_at=datetime.now(),
            updated_at=None,
            source=None,
            confidence=0.9,
            additional_info=None
        )
        
        # Note: When creating KnowledgeEntity, we don't need to pass domain explicitly
        # because it's set by the model factory, but we do need to pass all other fields
        # The domain field is auto-set when using create_domain_entity_model
        entity_data = {
            "id": "concept-1",
            "name": "Neural Networks",
            "entity_type": "CONCEPT",
            "metadata": metadata,
            "domain": "knowledge"
        }
        
        # Add dynamic fields
        # This won't please the linter but will work at runtime
        entity_data["difficulty_level"] = "ADVANCED"
        entity_data["keywords"] = ["AI", "Machine Learning", "Deep Learning"]
        
        entity = KnowledgeEntity(**entity_data)
        
        print(f"✅ Created entity: {entity.name} (type: {entity.entity_type})")
        # Using getattr to access dynamic fields to avoid linter warnings
        print(f"   Difficulty: {getattr(entity, 'difficulty_level', 'Unknown')}")
        print(f"   Keywords: {getattr(entity, 'keywords', [])}")
    except Exception as e:
        print(f"❌ Error creating entity: {str(e)}")
    
    # Test validation with invalid type
    try:
        # Similar approach for testing invalid entity
        invalid_data = {
            "id": "invalid-1",
            "name": "Invalid Entity",
            "entity_type": "INVALID_TYPE",  # This should fail validation
            "metadata": None,
            "domain": "knowledge"
        }
        
        # Add dynamic fields
        invalid_data["difficulty_level"] = None
        invalid_data["keywords"] = []
        
        invalid_entity = KnowledgeEntity(**invalid_data)
        print(f"✅ Created invalid entity (shouldn't happen): {invalid_entity.name}")
    except Exception as e:
        print(f"❌ Expected error for invalid entity type: {str(e)}")
    
    return KnowledgeEntity, KnowledgeRelationship


def use_base_entity_models():
    """Demonstrate how to use the base entity models directly."""
    print("\n=== Example: Using Base Entity Models ===")
    
    # Create a custom entity class with allowed types
    class ProjectEntity(BaseEntity):
        allowed_entity_types = {"TASK", "MILESTONE", "RESOURCE", "DOCUMENT"}
        
        class Config:
            validate_assignment = True
    
    # Create a valid entity
    try:
        # Create the metadata object properly with all fields
        meta = BaseMetadata(
            client_id="example-client",
            created_at=datetime.now(),
            updated_at=None,
            source=None,
            confidence=0.95,
            additional_info={"tags": ["documentation", "high-priority"]}
        )
        
        task = ProjectEntity(
            id="task-1",
            name="Complete documentation",
            entity_type="TASK",
            domain="project",
            metadata=meta
        )
        print(f"✅ Created project entity: {task.name} (type: {task.entity_type})")
        # Safely access metadata attributes
        if task.metadata:
            print(f"   Metadata: confidence={task.metadata.confidence}, client={task.metadata.client_id}")
            if task.metadata.additional_info:
                print(f"   Tags: {task.metadata.additional_info.get('tags', [])}")
    except Exception as e:
        print(f"❌ Error creating project entity: {str(e)}")
    
    # Create an invalid entity (wrong type)
    try:
        invalid_task = ProjectEntity(
            id="invalid-1",
            name="Invalid Task",
            entity_type="BUG",  # This should fail validation
            domain="project",
            metadata=None  # Add metadata parameter
        )
        print(f"✅ Created invalid entity (shouldn't happen): {invalid_task.name}")
    except Exception as e:
        print(f"❌ Expected error for invalid entity type: {str(e)}")


def use_relationship_models():
    """Demonstrate how to use the base relationship models."""
    print("\n=== Example: Using Relationship Models ===")
    
    # Create a custom relationship class with allowed types
    class TaskRelationship(BaseRelationship):
        allowed_relationship_types = {"DEPENDS_ON", "BLOCKS", "RELATED_TO", "PART_OF"}
        
        class Config:
            validate_assignment = True
    
    # Create a valid relationship
    try:
        # Create a complete metadata object
        rel_metadata = BaseMetadata(
            client_id="example-client",
            created_at=datetime.now(),
            updated_at=None,
            source=None,
            confidence=1.0,
            additional_info=None
        )
        
        relationship = TaskRelationship(
            id="rel-1",
            source_id="task-1",
            target_id="task-2",
            relationship_type="DEPENDS_ON",
            domain="project",
            properties={"critical_path": True, "delay_impact": "high"},
            metadata=rel_metadata
        )
        print(f"✅ Created relationship: {relationship.relationship_type}")
        print(f"   From {relationship.source_id} to {relationship.target_id}")
        print(f"   Properties: {relationship.properties}")
    except Exception as e:
        print(f"❌ Error creating relationship: {str(e)}")
    
    # Create an invalid relationship (wrong type)
    try:
        invalid_rel = TaskRelationship(
            id=None,  # Add required params
            source_id="task-1",
            target_id="task-3",
            relationship_type="INVALID_TYPE",  # This should fail validation
            domain="project",
            properties=None,  # Add required params
            metadata=None     # Add required params
        )
        print(f"✅ Created invalid relationship (shouldn't happen): {invalid_rel.relationship_type}")
    except Exception as e:
        print(f"❌ Expected error for invalid relationship type: {str(e)}")


def use_search_query():
    """Demonstrate how to use the search query model."""
    print("\n=== Example: Using Search Query ===")
    
    try:
        query = BaseSearchQuery(
            query="important task",
            entity_types=["TASK", "MILESTONE"],
            relationship_types=None,  # Add required param
            domain="project",
            container_id=None,        # Add required param
            tags=["high-priority"],
            limit=25,
            confidence_threshold=0.8,
            semantic=True
        )
        print(f"✅ Created search query: '{query.query}'")
        print(f"   Filtering by: {query.entity_types}")
        print(f"   Domain: {query.domain}, Tags: {query.tags}")
        print(f"   Limit: {query.limit}, Confidence threshold: {query.confidence_threshold}")
    except Exception as e:
        print(f"❌ Error creating search query: {str(e)}")
    
    # Test validation with invalid limit
    try:
        invalid_query = BaseSearchQuery(
            query="test",
            entity_types=None,          # Add required params
            relationship_types=None,
            domain=None,
            container_id=None,
            tags=None,
            confidence_threshold=None,
            semantic=False,
            limit=200  # This should fail validation (max is 100)
        )
        print(f"✅ Created invalid query (shouldn't happen): limit={invalid_query.limit}")
    except Exception as e:
        print(f"❌ Expected error for invalid limit: {str(e)}")


def main():
    """Run all examples."""
    print("=== Base Graph Models Examples ===")
    
    # Demonstrate creating custom domain models
    KnowledgeEntity, KnowledgeRelationship = create_custom_domain_models()
    
    # Demonstrate using base entity models
    use_base_entity_models()
    
    # Demonstrate using relationship models
    use_relationship_models()
    
    # Demonstrate using search query
    use_search_query()
    
    print("\n=== Examples Complete ===")


if __name__ == "__main__":
    main() 