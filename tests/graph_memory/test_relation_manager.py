import pytest
from unittest.mock import patch, MagicMock

from src.graph_memory.relation_manager import RelationManager


def test_init(mock_base_manager):
    """Test RelationManager initialization."""
    relation_manager = RelationManager(mock_base_manager)
    assert relation_manager.base_manager == mock_base_manager


def test_create_relationship(mock_base_manager):
    """Test creating a relationship between two entities."""
    # Test data
    source_entity = "source_entity"
    target_entity = "target_entity"
    relation_type = "TEST_RELATION"
    properties = {"weight": 0.9}
    
    # Setup mock response
    mock_response = {
        "source": {"id": 123, "name": source_entity},
        "target": {"id": 456, "name": target_entity},
        "relationship": {"type": relation_type, "properties": properties}
    }
    mock_base_manager.execute_query.return_value = [mock_response]
    
    # Create relation manager and test method
    relation_manager = RelationManager(mock_base_manager)
    result = relation_manager.create_relationship(
        source_entity=source_entity,
        target_entity=target_entity,
        relation_type=relation_type,
        properties=properties
    )
    
    # Assertions
    assert result is not None
    assert result.get("type") == relation_type
    assert result.get("source_name") == source_entity
    assert result.get("target_name") == target_entity
    assert result.get("properties", {}).get("weight") == 0.9
    
    # Verify query execution
    mock_base_manager.execute_query.assert_called_once()
    # Verify the query contains appropriate MATCH statements and CREATE/MERGE for relationship
    call_args = mock_base_manager.execute_query.call_args[0]
    assert "MATCH" in call_args[0]
    assert any(keyword in call_args[0] for keyword in ["CREATE", "MERGE"])
    assert source_entity in str(call_args)
    assert target_entity in str(call_args)
    assert relation_type in str(call_args)


def test_create_relationship_entities_not_found(mock_base_manager):
    """Test creating a relationship when entities don't exist."""
    # Setup mock response (empty result)
    mock_base_manager.execute_query.return_value = []
    
    # Create relation manager and test method
    relation_manager = RelationManager(mock_base_manager)
    result = relation_manager.create_relationship(
        source_entity="non_existent_source",
        target_entity="non_existent_target",
        relation_type="TEST_RELATION"
    )
    
    # Assertions
    assert result is None
    
    # Verify query execution
    mock_base_manager.execute_query.assert_called_once()


def test_create_relationships_batch(mock_base_manager, sample_relations):
    """Test batch creation of multiple relationships."""
    # Setup mock responses
    mock_responses = [
        {
            "source": {"id": 123, "name": rel["from"]},
            "target": {"id": 456, "name": rel["to"]},
            "relationship": {"type": rel["relationType"], "properties": {}}
        }
        for rel in sample_relations
    ]
    mock_base_manager.execute_query.return_value = mock_responses
    
    # Create relation manager and test method
    relation_manager = RelationManager(mock_base_manager)
    results = relation_manager.create_relationships(sample_relations)
    
    # Assertions
    assert len(results) == len(sample_relations)
    for i, result in enumerate(results):
        assert result.get("type") == sample_relations[i]["relationType"]
        assert result.get("source_name") == sample_relations[i]["from"]
        assert result.get("target_name") == sample_relations[i]["to"]
    
    # Verify query execution
    mock_base_manager.execute_query.assert_called_once()
    # Verify the query includes batch operation syntax (e.g., UNWIND)
    call_args = mock_base_manager.execute_query.call_args[0]
    assert "UNWIND" in call_args[0] or all(rel["from"] in str(call_args) for rel in sample_relations)


def test_get_relationships_by_source(mock_base_manager, sample_relation):
    """Test retrieving relationships by source entity."""
    # Setup mock response
    source_entity = sample_relation["from"]
    mock_response = {
        "source": {"id": 123, "name": source_entity},
        "target": {"id": 456, "name": sample_relation["to"]},
        "relationship": {"type": sample_relation["relationType"], "properties": sample_relation.get("properties", {})}
    }
    mock_base_manager.execute_query.return_value = [mock_response]
    
    # Create relation manager and test method
    relation_manager = RelationManager(mock_base_manager)
    results = relation_manager.get_relationships(entity_name=source_entity)
    
    # Assertions
    assert len(results) == 1
    assert results[0].get("type") == sample_relation["relationType"]
    assert results[0].get("source_name") == source_entity
    assert results[0].get("target_name") == sample_relation["to"]
    
    # Verify query execution
    mock_base_manager.execute_query.assert_called_once()
    # Verify the query contains proper MATCH statement for the source entity
    call_args = mock_base_manager.execute_query.call_args[0]
    assert "MATCH" in call_args[0]
    assert source_entity in str(call_args)


def test_get_relationships_by_type(mock_base_manager, sample_relation):
    """Test retrieving relationships by relation type."""
    # Setup mock response
    source_entity = sample_relation["from"]
    relation_type = sample_relation["relationType"]
    mock_response = {
        "source": {"id": 123, "name": source_entity},
        "target": {"id": 456, "name": sample_relation["to"]},
        "relationship": {"type": relation_type, "properties": sample_relation.get("properties", {})}
    }
    mock_base_manager.execute_query.return_value = [mock_response]
    
    # Create relation manager and test method
    relation_manager = RelationManager(mock_base_manager)
    results = relation_manager.get_relationships(entity_name=source_entity, relation_type=relation_type)
    
    # Assertions
    assert len(results) == 1
    assert results[0].get("type") == relation_type
    
    # Verify query execution
    mock_base_manager.execute_query.assert_called_once()
    # Verify the query filters by relation type
    call_args = mock_base_manager.execute_query.call_args[0]
    assert relation_type in str(call_args)


def test_get_relationships_not_found(mock_base_manager):
    """Test retrieving relationships when none exist."""
    # Setup mock response (empty result)
    mock_base_manager.execute_query.return_value = []
    
    # Create relation manager and test method
    relation_manager = RelationManager(mock_base_manager)
    results = relation_manager.get_relationships(entity_name="non_existent_entity")
    
    # Assertions
    assert len(results) == 0
    
    # Verify query execution
    mock_base_manager.execute_query.assert_called_once()


def test_delete_relationship(mock_base_manager, sample_relation):
    """Test deleting a relationship."""
    # Setup mock response
    mock_base_manager.execute_query.return_value = [{"success": True, "count": 1}]
    
    # Create relation manager and test method
    relation_manager = RelationManager(mock_base_manager)
    result = relation_manager.delete_relationship(
        from_entity=sample_relation["from"],
        to_entity=sample_relation["to"],
        relation_type=sample_relation["relationType"]
    )
    
    # Assertions
    assert result is True
    
    # Verify query execution
    mock_base_manager.execute_query.assert_called_once()
    # Verify the query contains DELETE statement
    call_args = mock_base_manager.execute_query.call_args[0]
    assert "DELETE" in call_args[0]
    assert sample_relation["from"] in str(call_args)
    assert sample_relation["to"] in str(call_args)
    assert sample_relation["relationType"] in str(call_args)


def test_delete_relationship_not_found(mock_base_manager):
    """Test deleting a non-existent relationship."""
    # Setup mock response (no deletion)
    mock_base_manager.execute_query.return_value = [{"success": True, "count": 0}]
    
    # Create relation manager and test method
    relation_manager = RelationManager(mock_base_manager)
    result = relation_manager.delete_relationship(
        from_entity="non_existent_source",
        to_entity="non_existent_target",
        relation_type="NON_EXISTENT_RELATION"
    )
    
    # Assertions
    assert result is False
    
    # Verify query execution
    mock_base_manager.execute_query.assert_called_once()


def test_delete_all_relationships(mock_base_manager):
    """Test deleting all relationships for an entity."""
    # Setup mock response
    mock_base_manager.execute_query.return_value = [{"success": True, "count": 5}]
    
    # Create relation manager and test method
    relation_manager = RelationManager(mock_base_manager)
    result = relation_manager.delete_all_relationships(entity_name="test_entity")
    
    # Assertions
    assert result == 5
    
    # Verify query execution
    mock_base_manager.execute_query.assert_called_once()
    # Verify the query contains DELETE statement
    call_args = mock_base_manager.execute_query.call_args[0]
    assert "DELETE" in call_args[0]
    assert "test_entity" in str(call_args)


def test_update_relationship_properties(mock_base_manager, sample_relation):
    """Test updating properties of a relationship."""
    # Setup updated properties
    updated_properties = {"weight": 0.8, "new_property": "test"}
    
    # Setup mock response
    mock_response = {
        "source": {"id": 123, "name": sample_relation["from"]},
        "target": {"id": 456, "name": sample_relation["to"]},
        "relationship": {"type": sample_relation["relationType"], "properties": updated_properties}
    }
    mock_base_manager.execute_query.return_value = [mock_response]
    
    # Create relation manager and test method
    relation_manager = RelationManager(mock_base_manager)
    result = relation_manager.update_relationship_properties(
        from_entity=sample_relation["from"],
        to_entity=sample_relation["to"],
        relation_type=sample_relation["relationType"],
        properties=updated_properties
    )
    
    # Assertions
    assert result is not None
    assert result.get("properties", {}).get("weight") == 0.8
    assert result.get("properties", {}).get("new_property") == "test"
    
    # Verify query execution
    mock_base_manager.execute_query.assert_called_once()
    # Verify the query contains SET statement for property updates
    call_args = mock_base_manager.execute_query.call_args[0]
    assert "SET" in call_args[0]


def test_error_handling(mock_base_manager, sample_relation):
    """Test error handling during relationship operations."""
    # Setup mock to raise exception
    mock_base_manager.execute_query.side_effect = Exception("Database error")
    
    # Create relation manager and test methods
    relation_manager = RelationManager(mock_base_manager)
    
    # Test create_relationship error handling
    result = relation_manager.create_relationship(
        source_entity=sample_relation["from"],
        target_entity=sample_relation["to"],
        relation_type=sample_relation["relationType"]
    )
    assert result is None
    
    # Test get_relationships error handling
    results = relation_manager.get_relationships(entity_name=sample_relation["from"])
    assert len(results) == 0
    
    # Test delete_relationship error handling
    result = relation_manager.delete_relationship(
        from_entity=sample_relation["from"],
        to_entity=sample_relation["to"],
        relation_type=sample_relation["relationType"]
    )
    assert result is False 