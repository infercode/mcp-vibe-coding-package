"""
Mock integration tests for the Graph Memory system using Neo4j mocks instead of a real database.
"""

import pytest
from unittest.mock import MagicMock, patch
import json

from src.graph_memory import GraphMemoryManager
from src.graph_memory.base_manager import BaseManager


@pytest.fixture
def mock_logger():
    """Create a mock logger for testing."""
    return MagicMock()


@pytest.fixture
def mock_base_manager(mock_logger):
    """Create a mock base manager for testing."""
    mock_manager = MagicMock(spec=BaseManager)
    # Set up logger with required methods
    mock_logger.info = MagicMock()
    mock_logger.debug = MagicMock()
    mock_logger.error = MagicMock()
    mock_logger.warning = MagicMock()
    mock_manager.logger = mock_logger
    mock_manager.ensure_initialized.return_value = None
    
    # Add required properties for GraphMemoryManager
    mock_manager.default_project_name = "default-test"
    mock_manager.neo4j_uri = "bolt://localhost:7687"
    mock_manager.neo4j_user = "neo4j"
    mock_manager.neo4j_password = "password"
    mock_manager.neo4j_database = "neo4j"
    
    return mock_manager


@pytest.fixture
def graph_memory_manager(mock_base_manager):
    """Create a GraphMemoryManager with mock base manager."""
    # Mock the BaseManager constructor to avoid creating a new instance
    with patch('src.graph_memory.BaseManager', return_value=mock_base_manager):
        manager = GraphMemoryManager(logger=mock_base_manager.logger)
        # Inject our mock base manager directly
        manager.base_manager = mock_base_manager
        return manager


def test_create_relationship(graph_memory_manager, mock_base_manager, mock_logger):
    """Test creating a relationship with mock responses."""
    # Create a simpler test that doesn't depend on internal implementations
    
    # Setup a straightforward mock for the relation_manager
    mock_relation_manager = MagicMock()
    mock_relation_manager.create_relationship.return_value = json.dumps({
        "status": "success",
        "relationship": {
            "from": "TestEntity1",
            "to": "TestEntity2",
            "type": "TEST_RELATION"
        }
    })
    
    # Replace the real relation_manager with our mock
    graph_memory_manager.relation_manager = mock_relation_manager
    
    # Create relationship data
    relation = {
        "from": "TestEntity1",
        "to": "TestEntity2",
        "relationType": "TEST_RELATION"
    }
    
    # Call the method
    graph_memory_manager.create_relationship(relation)
    
    # Verify the method was called
    graph_memory_manager.relation_manager.create_relationship.assert_called_once()


def test_get_entities(graph_memory_manager, mock_base_manager, mock_logger):
    """Test retrieving entities with mock responses."""
    # Set up mock for get_entities
    mock_record = MagicMock()
    mock_entity = MagicMock()
    mock_entity.items.return_value = [
        ("name", "TestEntity1"),
        ("type", "TEST"),
        ("created", "2023-01-01T12:00:00"),
        ("lastUpdated", "2023-01-01T12:00:00")
    ]
    mock_record.get.return_value = mock_entity
    mock_base_manager.safe_execute_read_query.return_value = [mock_record]
    
    # Call method
    result = graph_memory_manager.get_entities()
    
    # Verify result
    assert result is not None
    assert isinstance(result, list)
    assert len(result) > 0
    assert mock_base_manager.safe_execute_read_query.called


def test_add_observation(graph_memory_manager, mock_base_manager, mock_logger):
    """Test adding an observation with mock responses."""
    # Mock the observation_manager directly
    graph_memory_manager.observation_manager = MagicMock()
    graph_memory_manager.observation_manager.add_observations.return_value = json.dumps({
        "added": [{
            "entity": "TestEntity1",
            "content": "Test observation content",
            "id": "obs123",
            "created": "2023-01-01T12:00:00"
        }]
    })
    
    # Create observation data
    observation = {
        "entity": "TestEntity1",
        "content": "Test observation content"
    }
    
    # Call method
    result = graph_memory_manager.add_observation(observation)
    
    # Verify result
    assert result is not None
    result_obj = json.loads(result)
    assert "added" in result_obj
    assert isinstance(result_obj["added"], list)
    assert len(result_obj["added"]) > 0
    assert graph_memory_manager.observation_manager.add_observations.called 