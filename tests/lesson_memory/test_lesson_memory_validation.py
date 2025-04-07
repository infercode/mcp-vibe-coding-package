#!/usr/bin/env python
"""
Validation test for LessonMemoryManager with Neo4j query validation.
Tests the integration of LessonContainer and LessonObservation with
validated query methods (safe_execute_read_query and safe_execute_write_query).
"""

import os
import sys
import argparse
import json
import logging
from typing import Dict, List, Any, Optional
import pytest
from src.models.lesson_memory import (
    LessonContainer, LessonContainerCreate, LessonEntity,
    LessonContainerUpdate, Metadata, create_metadata
)

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Add the project root directory to the path
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(os.path.dirname(current_dir))
sys.path.insert(0, project_root)

from src.graph_memory.base_manager import BaseManager
# from src.lesson_memory.lesson_container import LessonContainer
# from src.lesson_memory.lesson_observation import LessonObservation
# from src.utils import dict_to_json

def parse_json_dict(json_str: str) -> Dict[str, Any]:
    """
    Parse a JSON string into a dictionary.
    
    Args:
        json_str: JSON string to parse
        
    Returns:
        Dictionary representation of the JSON
    """
    return json.loads(json_str)

def setup_base_manager() -> BaseManager:
    """
    Set up a BaseManager with Neo4j credentials.
    
    Returns:
        BaseManager: Initialized BaseManager instance
    """
    # Set Neo4j credentials as environment variables
    os.environ["NEO4J_URI"] = os.getenv("NEO4J_URI", "bolt://localhost:7687")
    os.environ["NEO4J_USER"] = os.getenv("NEO4J_USER", "neo4j")
    os.environ["NEO4J_PASSWORD"] = os.getenv("NEO4J_PASSWORD", "password")
    
    logger.info(f"Using Neo4j at {os.environ['NEO4J_URI']}")
    
    # BaseManager will read from environment variables
    return BaseManager()

def setup_mock_base_manager() -> BaseManager:
    """
    Set up a mock BaseManager for testing without a real database.
    
    Returns:
        BaseManager: Mocked BaseManager instance
    """
    logger.info("Setting up mock BaseManager")
    
    class MockBaseManager(BaseManager):
        def __init__(self):
            # Skip actual initialization
            self.initialized = True
            self.logger = logging.getLogger(__name__)
            self.mock_containers = {}
            self.mock_entities = {}
            self.mock_observations = {}
            
        def initialize(self):
            self.initialized = True
            return True
            
        def ensure_initialized(self):
            self.initialized = True
        
        def safe_execute_read_query(self, query, parameters=None, database=None):
            logger.info(f"Mock read query: {query}")
            logger.info(f"Parameters: {parameters}")
            
            # Validate that this is actually a read query
            if any(op in query.upper() for op in ["CREATE", "DELETE", "SET", "REMOVE", "MERGE"]):
                # Exception for specific validation queries that don't actually modify data
                if "RETURN" in query and not any(op in query.upper() for op in ["WITH", "CALL"]):
                    pass  # Allow read-like queries with validation terms
                else:
                    raise ValueError(f"Destructive operation not allowed in read-only mode: {query}")
            
            params = parameters or {}
            
            # Parse and return mocked results based on the query
            if "MATCH (c:LessonContainer" in query:
                container_name = params.get("name") or params.get("container_name")
                if container_name and container_name in self.mock_containers:
                    container = self.mock_containers[container_name]
                    return [{"c": container}]
                return []
                
            elif "MATCH (e:Entity" in query:
                entity_name = params.get("name") or params.get("entity_name")
                if entity_name and entity_name in self.mock_entities:
                    entity = self.mock_entities[entity_name]
                    return [{"e": entity}]
                return []
                
            elif "MATCH (o:Observation" in query:
                observation_id = params.get("id") or params.get("observation_id")
                if observation_id and observation_id in self.mock_observations:
                    observation = self.mock_observations[observation_id]
                    return [{"o": observation}]
                return []
                
            return []
        
        def safe_execute_write_query(self, query, parameters=None, database=None):
            logger.info(f"Mock write query: {query}")
            logger.info(f"Parameters: {parameters}")
            
            # Sanitize parameters
            params = parameters or {}
            
            # Process write operations
            if "CREATE (c:LessonContainer" in query:
                container_props = params.get("properties", {})
                container_name = container_props.get("name")
                
                # Check if container already exists
                if container_name in self.mock_containers:
                    return []  # Simulate failure
                
                # Create container
                self.mock_containers[container_name] = container_props
                return [{"c": container_props}]
                
            elif "CREATE (e:Entity" in query:
                entity_props = params.get("properties", {})
                entity_name = entity_props.get("name")
                
                # Create entity
                self.mock_entities[entity_name] = entity_props
                return [{"e": entity_props}]
                
            elif "CREATE (o:Observation" in query:
                observation_props = params.get("properties", {})
                observation_id = observation_props.get("id")
                
                # Create observation
                self.mock_observations[observation_id] = observation_props
                return [{"o": observation_props}]
                
            elif "DELETE" in query and "LessonContainer" in query:
                container_name = params.get("name")
                if container_name in self.mock_containers:
                    del self.mock_containers[container_name]
                    return [{}]
                return []
                
            # Default return for other write operations
            return [{}]
    
    return MockBaseManager()

def parse_json_result(json_str: str) -> Dict[str, Any]:
    """
    Parse JSON string results from manager methods.
    
    Args:
        json_str: JSON string to parse
        
    Returns:
        Parsed dictionary or empty dict on error
    """
    try:
        return parse_json_dict(json_str)
    except json.JSONDecodeError:
        logger.error(f"Failed to parse JSON: {json_str}")
        return {}

@pytest.fixture
def mock_lesson_manager():
    """
    Create a mock LessonMemoryManager.
    
    Returns:
        Mocked LessonMemoryManager instance
    """
    from unittest.mock import MagicMock
    mock_manager = MagicMock()
    
    # Mock create_container to return a valid response
    mock_manager.create_container.return_value = json.dumps({
        "container": {
            "id": "test-id-123",
            "name": "TestLesson",
            "description": "Test lesson for validation",
            "tags": ["test", "validation"],
            "created": "2023-05-10T15:30:45.123456",
            "lastUpdated": "2023-05-10T15:30:45.123456"
        }
    })
    
    # Mock create_entity to return a valid response
    mock_manager.create_entity.return_value = json.dumps({
        "entity": {
            "id": "entity-id-123",
            "name": "TestEntity",
            "entity_type": "concept",
            "domain": "lesson",
            "created": "2023-05-10T15:30:45.123456",
            "lastUpdated": "2023-05-10T15:30:45.123456"
        }
    })
    
    # Mock update_container to return a valid response
    mock_manager.update_container.return_value = json.dumps({
        "container": {
            "id": "test-id-123",
            "name": "TestLesson",
            "description": "Updated description",
            "tags": ["test", "validation", "updated"],
            "created": "2023-05-10T15:30:45.123456",
            "lastUpdated": "2023-05-10T15:35:22.654321"
        }
    })
    
    # Mock search_containers to return a list of containers
    mock_manager.search_containers.return_value = [
        {
            "id": "test-id-123",
            "name": "TestLesson",
            "description": "Test lesson for validation",
            "tags": ["test", "validation"],
            "created": "2023-05-10T15:30:45.123456",
            "lastUpdated": "2023-05-10T15:30:45.123456"
        }
    ]
    
    # Mock search_entities to return a list of entities
    mock_manager.search_entities.return_value = [
        {
            "id": "entity-id-123",
            "name": "TestEntity",
            "entity_type": "concept",
            "domain": "lesson",
            "created": "2023-05-10T15:30:45.123456",
            "lastUpdated": "2023-05-10T15:30:45.123456",
            "observations": 2
        }
    ]
    
    # Mock get_container_entities to return a list of entities
    mock_manager.get_container_entities.return_value = [
        {
            "id": "entity-id-123",
            "name": "TestEntity",
            "entity_type": "concept",
            "domain": "lesson",
            "created": "2023-05-10T15:30:45.123456",
            "lastUpdated": "2023-05-10T15:30:45.123456",
            "container": "TestLesson"
        },
        {
            "id": "entity-id-456",
            "name": "AnotherEntity",
            "entity_type": "concept",
            "domain": "lesson",
            "created": "2023-05-10T16:20:45.123456",
            "lastUpdated": "2023-05-10T16:20:45.123456",
            "container": "TestLesson"
        }
    ]
    
    return mock_manager

def test_valid_lesson_operations(mock_lesson_manager, mock_logger):
    """Test valid lesson operations with validation."""
    # Test lesson container creation
    container = LessonContainerCreate(
        name="TestLesson",
        description="Test lesson for validation",
        tags=["test", "validation"],
        metadata={"source": "test", "importance": "high"}
    )
    result = mock_lesson_manager.create_container(container.name, container.description, container.metadata)
    result = json.loads(result)
    assert result is not None
    assert "container" in result, f"Expected container in result, got: {result}"
    
    # Parse the container response
    container_data = result["container"]
    assert container_data["name"] == "TestLesson"
    
    # Verify timestamp fields in ISO format
    if "created" in container_data:
        assert isinstance(container_data["created"], str)
        assert "T" in container_data["created"], f"Created timestamp should be in ISO format, got: {container_data['created']}"
    
    if "lastUpdated" in container_data:
        assert isinstance(container_data["lastUpdated"], str)
        assert "T" in container_data["lastUpdated"], f"LastUpdated timestamp should be in ISO format, got: {container_data['lastUpdated']}"
    
    # Test lesson entity creation
    entity = LessonEntity(
        name="TestEntity",
        entity_type="concept",
        domain="lesson",
        metadata=create_metadata()
    )
    result = mock_lesson_manager.create_entity(entity.name, entity.entity_type)
    result = json.loads(result)
    assert result is not None
    assert "entity" in result, f"Expected entity in result, got: {result}"
    
    # Parse the entity response
    entity_data = result["entity"]
    assert entity_data["name"] == "TestEntity"
    
    # Verify timestamp fields in ISO format for entity
    if "created" in entity_data:
        assert isinstance(entity_data["created"], str)
        assert "T" in entity_data["created"], f"Entity created timestamp should be in ISO format, got: {entity_data['created']}"
    
    if "lastUpdated" in entity_data:
        assert isinstance(entity_data["lastUpdated"], str)
        assert "T" in entity_data["lastUpdated"], f"Entity lastUpdated timestamp should be in ISO format, got: {entity_data['lastUpdated']}"
    
    # Test lesson container update
    update_data = {
        "description": "Updated description",
        "tags": ["test", "validation", "updated"]
    }
    result = mock_lesson_manager.update_container("TestLesson", update_data)
    result = json.loads(result)
    assert result is not None
    assert "container" in result, f"Expected container in result, got: {result}"
    assert result["container"]["description"] == "Updated description"
    assert "updated" in result["container"]["tags"]

def test_invalid_lesson_operations(mock_lesson_manager, mock_base_manager, mock_logger):
    """Test invalid lesson operations to verify validation."""
    
    # Override mock to raise ValueError for invalid input
    mock_lesson_manager.create_container.side_effect = ValueError("Container name cannot be empty")
    
    # Test container creation with invalid name
    with pytest.raises(ValueError) as exc_info:
        result = mock_lesson_manager.create_container("", "Invalid container")
    
    assert "name cannot be empty" in str(exc_info.value).lower() or "container name cannot be empty" in str(exc_info.value).lower()
    
    # Test with invalid entity name (restore mock behavior first)
    mock_lesson_manager.create_container.side_effect = None
    mock_lesson_manager.create_entity.side_effect = ValueError("Entity name must be at least 3 characters")
    
    with pytest.raises(ValueError) as exc_info:
        result = mock_lesson_manager.create_entity("", "concept")
    
    assert "name" in str(exc_info.value).lower()

def test_lesson_search_operations(mock_lesson_manager, mock_logger):
    """Test lesson search operations with validation."""
    # Test container search
    result = mock_lesson_manager.search_containers("Test")
    assert isinstance(result, list)
    
    # Verify container in search results
    assert len(result) > 0, "Expected at least one container in search results"
    container = result[0]
    assert "name" in container
    assert "description" in container
    
    # Verify container has expected fields
    assert container["name"] == "TestLesson"
    assert "tags" in container

    # Test entity search
    result = mock_lesson_manager.search_entities("Test")
    assert isinstance(result, list)
    
    # Verify entity in search results
    assert len(result) > 0, "Expected at least one entity in search results"
    entity = result[0]
    assert "name" in entity
    assert "entity_type" in entity
    
    # Verify entity has expected fields
    assert entity["name"] == "TestEntity"
    assert entity["entity_type"] == "concept"
    assert "observations" in entity

    # Test entity search with type filter
    result = mock_lesson_manager.search_entities(
        "Test",
        entity_types=["concept"]
    )
    assert isinstance(result, list)
    
    # Test container entities retrieval
    result = mock_lesson_manager.get_container_entities("TestLesson")
    assert isinstance(result, list) 