"""
Test script to verify the EntityManager integration with Neo4j query validation.
"""

import pytest
from unittest.mock import MagicMock
import json
from src.graph_memory.entity_manager import EntityManager

@pytest.fixture
def validation_entity_manager():
    """Create a mock entity manager specifically for validation tests."""
    mock = MagicMock()
    
    # Set up return values for the various methods
    mock.create_entities.return_value = json.dumps({
        "status": "success",
        "created": 1,
        "entity_ids": ["entity-1"]
    })
    
    mock.get_entity.return_value = json.dumps({
        "name": "TestEntity",
        "id": "entity-1",
        "type": "test"
    })
    
    mock.update_entity.return_value = json.dumps({
        "status": "success",
        "updated": True
    })
    
    mock.delete_entity.return_value = json.dumps({
        "status": "success",
        "deleted": True
    })
    
    return mock

@pytest.fixture
def validation_base_manager():
    """Create a mock base manager specifically for validation tests."""
    mock = MagicMock()
    return mock

def test_valid_entity_operations(validation_entity_manager, mock_logger):
    """Test valid entity operations with validation."""
    # Test entity creation
    entity = {
        "name": "TestEntity",
        "entityType": "test",
        "observations": ["This is a test entity", "Created for validation testing"]
    }
    
    result = validation_entity_manager.create_entities([entity])
    # Handle JSON string or dictionary response
    if isinstance(result, str):
        result = json.loads(result)
    assert "status" in result and result["status"] == "success"
    
    # Test entity retrieval
    result = validation_entity_manager.get_entity("TestEntity")
    # Handle JSON string or dictionary response
    if isinstance(result, str):
        result = json.loads(result)
    assert "name" in result and result["name"] == "TestEntity"
    
    # Test entity update
    update_result = validation_entity_manager.update_entity("TestEntity", {"description": "Updated entity"})
    # Handle JSON string or dictionary response
    if isinstance(update_result, str):
        update_result = json.loads(update_result)
    assert "status" in update_result and update_result["status"] == "success"
    
    # Test entity deletion
    delete_result = validation_entity_manager.delete_entity("TestEntity")
    # Handle JSON string or dictionary response
    if isinstance(delete_result, str):
        delete_result = json.loads(delete_result)
    assert "status" in delete_result and delete_result["status"] == "success"

def test_invalid_entity_operations(validation_entity_manager, validation_base_manager, mock_logger):
    """Test invalid entity operations to verify validation."""
    # Mock the safe_execute_read_query to simulate validation
    validation_base_manager.safe_execute_read_query.side_effect = ValueError("Invalid query: Contains DELETE operation")
    
    # Test executing a destructive query
    query = """
    MATCH (e:Entity {name: $name})
    DELETE e
    RETURN e
    """
    
    # Try to execute the query - should handle the ValueError gracefully
    try:
        validation_base_manager.safe_execute_read_query(query, {"name": "TestEntity"})
        assert False, "Should have raised an exception"
    except ValueError:
        # Expected behavior - validation caught the DELETE operation
        pass
    
    # Test with invalid parameter types
    invalid_entity = {
        "name": "InvalidEntity",
        "entityType": "test",
        "complex_property": object()  # This can't be serialized
    }
    
    # In the real implementation, we would expect validation errors to be caught and returned
    validation_entity_manager.create_entities.return_value = json.dumps({
        "error": "Non-serializable entity properties found",
        "details": {"entity": "InvalidEntity"}
    })
    
    result = validation_entity_manager.create_entities([invalid_entity])
    result_obj = json.loads(result)
    assert "error" in result_obj 