#!/usr/bin/env python3
"""
Example Usage of Pydantic Models in MCP Graph Memory

This script demonstrates how to use Pydantic models for data validation
and serialization in the MCP Graph Memory system.
"""

import json
import sys
import os
from datetime import datetime
from typing import Dict, Any, List, Optional

# Add parent directory to path to allow imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from src.models.lesson_memory import (
    LessonContainerCreate, LessonEntityCreate, LessonObservationCreate,
    StructuredLessonObservations, LessonRelationshipCreate, SearchQuery
)
from src.models.graph_config import GraphConfig, DatabaseType, Neo4jConfig, MemoryConfig, MockConfig
from src.models.lesson_responses import (
    create_entity_response, create_observation_response, 
    create_container_response, create_lesson_error_response,
    model_to_json
)
from src.models.responses import model_to_dict

def example_container():
    """Example of creating a lesson container with validation."""
    print("\n=== Example: Creating a Lesson Container ===")
    
    # Valid container
    try:
        container = LessonContainerCreate(
            name="Geography Lessons",
            description="Container for geography-related lessons",
            tags=["geography", "education", "lessons"],
            metadata={"created_by": "example_script", "priority": "high"}
        )
        print(f"✅ Valid container created: {container.name}")
        print(f"   Tags: {container.tags}")
        print(f"   JSON: {json.dumps(model_to_dict(container), indent=2)}")
    except Exception as e:
        print(f"❌ Error creating container: {str(e)}")
    
    # Invalid container (missing required field - intentional error)
    try:
        # Name is a required field - this should fail
        data = {
            "description": "This will fail validation",
            "tags": ["invalid"]
        }
        # Using the dict unpacking to trigger validation error
        container = LessonContainerCreate(**data)
        print(f"✅ Container created (shouldn't happen): {container}")
    except Exception as e:
        print(f"❌ Expected error: {str(e)}")

def example_entity():
    """Example of creating a lesson entity with validation."""
    print("\n=== Example: Creating a Lesson Entity ===")
    
    # Valid entity
    try:
        # Note: The observations field should be a list of strings according to the model,
        # but we're treating it as a list of dictionaries for this example
        entity = LessonEntityCreate(
            container_name="Geography Lessons",
            entity_name="Mount Everest",
            entity_type="Mountain",
            observations=[],  # Empty list for this example
            metadata={"elevation_m": 8848, "first_summit": "1953"}
        )
        print(f"✅ Valid entity created: {entity.entity_name}")
        print(f"   Type: {entity.entity_type}")
        # Safe check for observations length
        obs_count = len(entity.observations) if entity.observations is not None else 0
        print(f"   Observations: {obs_count}")
    except Exception as e:
        print(f"❌ Error creating entity: {str(e)}")
    
    # Invalid entity (missing required field - intentional error)
    try:
        data = {
            "container_name": "Geography Lessons",
            "entity_type": "Mountain",
            # Missing entity_name which is required
            "observations": None,  # Explicitly set to None
            "metadata": None  # Explicitly set to None
        }
        # Using dict unpacking to trigger validation error
        entity = LessonEntityCreate(**data)
        print(f"✅ Entity created (shouldn't happen): {entity}")
    except Exception as e:
        print(f"❌ Expected error: {str(e)}")

def example_structured_observations():
    """Example of creating structured observations."""
    print("\n=== Example: Creating Structured Observations ===")
    
    try:
        observations = StructuredLessonObservations(
            entity_name="Climate Change",
            what_was_learned="Global temperatures have risen by about 1°C since pre-industrial times",
            why_it_matters="Rising temperatures lead to more extreme weather events and sea level rise",
            how_to_apply="Reduce carbon emissions and implement sustainable practices",
            root_cause="Greenhouse gas emissions from human activities",
            evidence="Scientific research, temperature records, glacial melting patterns",
            container_name="Environmental Studies"
        )
        print(f"✅ Structured observations created for: {observations.entity_name}")
        print(f"   Types included: {', '.join([k for k, v in observations.model_dump().items() if v and k not in ['entity_name', 'container_name']])}")
    except Exception as e:
        print(f"❌ Error creating structured observations: {str(e)}")

def example_search_query():
    """Example of creating a search query with validation."""
    print("\n=== Example: Creating a Search Query ===")
    
    try:
        query = SearchQuery(
            container_name="Geography Lessons",
            search_term="mountain",
            entity_type="Mountain",
            tags=None,  # Explicitly set as None
            limit=5,
            semantic=True
        )
        print(f"✅ Search query created for term: '{query.search_term}'")
        print(f"   Container: {query.container_name}")
        print(f"   Limit: {query.limit}, Semantic: {query.semantic}")
    except Exception as e:
        print(f"❌ Error creating search query: {str(e)}")

def example_graph_config():
    """Example of creating a graph configuration."""
    print("\n=== Example: Creating a Graph Configuration ===")
    
    # Neo4j configuration
    try:
        neo4j_config = GraphConfig(
            database_type=DatabaseType.NEO4J,
            neo4j=Neo4jConfig(
                uri="bolt://localhost:7687",
                username="neo4j",
                password="password",
                database="neo4j"  # Adding the database parameter
            ),
            memory=None,
            mock=None,
            embedding=None,
            default_client_id="default",
            log_level="DEBUG"
        )
        print(f"✅ Neo4j config created: {neo4j_config.database_type}")
        # Check if neo4j is defined before accessing attributes
        uri = neo4j_config.neo4j.uri if neo4j_config.neo4j else "Not configured"
        print(f"   URI: {uri}")
        print(f"   Log level: {neo4j_config.log_level}")
    except Exception as e:
        print(f"❌ Error creating Neo4j config: {str(e)}")
    
    # In-memory configuration (with defaults)
    try:
        memory_config = GraphConfig(
            database_type=DatabaseType.MEMORY,
            neo4j=None,
            memory=MemoryConfig(persistence_file=None, load_on_startup=True),
            mock=None,
            embedding=None,
            default_client_id="example_client",
            log_level="INFO"
        )
        print(f"✅ Memory config created: {memory_config.database_type}")
        print(f"   Default client: {memory_config.default_client_id}")
        print(f"   Memory config: {memory_config.memory}")
    except Exception as e:
        print(f"❌ Error creating memory config: {str(e)}")
    
    # Invalid configuration (missing required configuration for Neo4j - intentional error)
    try:
        # This should fail because Neo4j configuration is required when database_type is NEO4J
        invalid_config = GraphConfig(
            database_type=DatabaseType.NEO4J,
            neo4j=None,  # This will trigger validation error
            memory=None,
            mock=None,
            embedding=None,
            default_client_id="default",
            log_level="INFO"
        )
        print(f"✅ Invalid config created (shouldn't happen): {invalid_config}")
    except Exception as e:
        print(f"❌ Expected error: {str(e)}")

def example_response_models():
    """Example of using response models."""
    print("\n=== Example: Using Response Models ===")
    
    # Entity response
    entity_data = {
        "id": "123",
        "name": "Mount Everest",
        "type": "Mountain",
        "created_at": str(datetime.now()),
        "observations": [
            {"id": "456", "content": "Highest mountain on Earth", "type": "fact"}
        ]
    }
    
    response = create_entity_response(
        entity_data=entity_data,
        message="Entity retrieved successfully"
    )
    
    print(f"✅ Created entity response with status: {response.status}")
    print(f"   Message: {response.message}")
    print(f"   JSON output: {model_to_json(response)[:100]}...")
    
    # Error response
    error_response = create_lesson_error_response(
        message="Entity not found",
        code="entity_not_found"
    )
    
    print(f"✅ Created error response with status: {error_response.status}")
    print(f"   Error code: {error_response.error.code}")
    print(f"   Message: {error_response.error.message}")

def main():
    """Run all examples."""
    print("=== MCP Graph Memory Pydantic Examples ===")
    
    example_container()
    example_entity()
    example_structured_observations()
    example_search_query()
    example_graph_config()
    example_response_models()
    
    print("\n=== Examples Complete ===")

if __name__ == "__main__":
    main() 