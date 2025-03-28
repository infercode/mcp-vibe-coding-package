import pytest
from unittest.mock import patch, MagicMock

from src.graph_memory.entity_manager import EntityManager


def test_init(mock_base_manager):
    """Test EntityManager initialization."""
    entity_manager = EntityManager(mock_base_manager)
    assert entity_manager.base_manager == mock_base_manager


def test_create_entity(mock_base_manager, sample_entity):
    """Test creation of a single entity."""
    # Setup mock response
    mock_base_manager.execute_query.return_value = [{"e": {"id": 123, "properties": sample_entity}}]
    
    # Create entity manager and test method
    entity_manager = EntityManager(mock_base_manager)
    result = entity_manager.create_entity(sample_entity)
    
    # Assertions
    assert result is not None
    assert result.get("id") == 123
    assert result.get("name") == sample_entity["name"]
    assert result.get("entityType") == sample_entity["entityType"]
    
    # Verify query execution
    mock_base_manager.execute_query.assert_called_once()
    # Verify that the query contains appropriate CREATE/MERGE statement
    call_args = mock_base_manager.execute_query.call_args[0]
    assert any(keyword in call_args[0] for keyword in ["CREATE", "MERGE"])


def test_create_entities_batch(mock_base_manager, sample_entities):
    """Test batch creation of multiple entities."""
    # Setup mock response
    mock_responses = [
        {"e": {"id": 123, "properties": sample_entities[0]}},
        {"e": {"id": 124, "properties": sample_entities[1]}}
    ]
    mock_base_manager.execute_query.return_value = mock_responses
    
    # Create entity manager and test method
    entity_manager = EntityManager(mock_base_manager)
    results = entity_manager.create_entities(sample_entities)
    
    # Assertions
    assert len(results) == 2
    assert results[0].get("id") == 123
    assert results[1].get("id") == 124
    assert results[0].get("name") == sample_entities[0]["name"]
    assert results[1].get("name") == sample_entities[1]["name"]
    
    # Verify query execution
    mock_base_manager.execute_query.assert_called_once()
    # Verify that the query handles multiple entities (UNWIND or multiple statements)
    call_args = mock_base_manager.execute_query.call_args[0]
    assert "UNWIND" in call_args[0] or sample_entities[0]["name"] in call_args[0]


def test_get_entity_by_name(mock_base_manager, sample_entity):
    """Test retrieving an entity by name."""
    # Setup mock response
    mock_base_manager.execute_query.return_value = [{"e": {"id": 123, "properties": sample_entity}}]
    
    # Create entity manager and test method
    entity_manager = EntityManager(mock_base_manager)
    result = entity_manager.get_entity(sample_entity["name"])
    
    # Assertions
    assert result is not None
    assert result.get("name") == sample_entity["name"]
    assert result.get("entityType") == sample_entity["entityType"]
    
    # Verify query execution
    mock_base_manager.execute_query.assert_called_once()
    # Verify the query contains MATCH statement
    call_args = mock_base_manager.execute_query.call_args[0]
    assert "MATCH" in call_args[0]
    assert sample_entity["name"] in str(call_args)


def test_get_entity_by_name_not_found(mock_base_manager):
    """Test retrieving a non-existent entity."""
    # Setup mock response (empty result)
    mock_base_manager.execute_query.return_value = []
    
    # Create entity manager and test method
    entity_manager = EntityManager(mock_base_manager)
    result = entity_manager.get_entity("non_existent_entity")
    
    # Assertions
    assert result is None
    
    # Verify query execution
    mock_base_manager.execute_query.assert_called_once()


def test_update_entity(mock_base_manager, sample_entity):
    """Test updating an entity's properties."""
    # Setup updated entity data
    updated_entity = dict(sample_entity)
    updated_entity["description"] = "Updated description"
    
    # Setup mock response
    mock_base_manager.execute_query.return_value = [{"e": {"id": 123, "properties": updated_entity}}]
    
    # Create entity manager and test method
    entity_manager = EntityManager(mock_base_manager)
    update_data = {"description": "Updated description"}
    result = entity_manager.update_entity(sample_entity["name"], update_data)
    
    # Assertions
    assert result is not None
    assert result.get("description") == "Updated description"
    
    # Verify query execution
    mock_base_manager.execute_query.assert_called_once()
    # Verify the query contains SET for updates
    call_args = mock_base_manager.execute_query.call_args[0]
    assert "SET" in call_args[0]


def test_delete_entity(mock_base_manager):
    """Test deleting an entity."""
    # Setup mock response
    mock_base_manager.execute_query.return_value = [{"success": True, "count": 1}]
    
    # Create entity manager and test method
    entity_manager = EntityManager(mock_base_manager)
    result = entity_manager.delete_entity("test_entity")
    
    # Assertions
    assert result is True
    
    # Verify query execution
    mock_base_manager.execute_query.assert_called_once()
    # Verify that the query contains DELETE
    call_args = mock_base_manager.execute_query.call_args[0]
    assert "DELETE" in call_args[0]


def test_get_entities_by_type(mock_base_manager, sample_entities):
    """Test retrieving entities by type."""
    # All sample entities have the same type
    entity_type = sample_entities[0]["entityType"]
    
    # Setup mock response
    mock_base_manager.execute_query.return_value = [
        {"e": {"id": 123, "properties": sample_entities[0]}},
        {"e": {"id": 124, "properties": sample_entities[1]}}
    ]
    
    # Create entity manager and test method
    entity_manager = EntityManager(mock_base_manager)
    results = entity_manager.get_entities_by_type(entity_type)
    
    # Assertions
    assert len(results) == 2
    assert results[0].get("entityType") == entity_type
    assert results[1].get("entityType") == entity_type
    
    # Verify query execution
    mock_base_manager.execute_query.assert_called_once()
    # Verify the query filters by entity type
    call_args = mock_base_manager.execute_query.call_args[0]
    assert entity_type in str(call_args)


def test_add_tags_to_entity(mock_base_manager, sample_entity):
    """Test adding tags to an entity."""
    # Setup updated entity with new tags
    updated_entity = dict(sample_entity)
    updated_entity["tags"] = sample_entity.get("tags", []) + ["new_tag"]
    
    # Setup mock response
    mock_base_manager.execute_query.return_value = [{"e": {"id": 123, "properties": updated_entity}}]
    
    # Create entity manager and test method
    entity_manager = EntityManager(mock_base_manager)
    result = entity_manager.add_tags_to_entity(sample_entity["name"], ["new_tag"])
    
    # Assertions
    assert result is not None
    assert "new_tag" in result.get("tags", [])
    
    # Verify query execution
    mock_base_manager.execute_query.assert_called_once()


@pytest.mark.parametrize("search_term,entity_type,expected_count", [
    ("test", "TestType", 1),
    ("non_existent", "TestType", 0),
    (None, "TestType", 2),
    ("test", None, 1),
])
def test_search_entities(mock_base_manager, sample_entities, search_term, entity_type, expected_count):
    """Test searching for entities with various filters."""
    # Setup mock response based on expected count
    if expected_count > 0:
        mock_response = [{"e": {"id": 123, "properties": sample_entities[0]}}]
        if expected_count > 1:
            mock_response.append({"e": {"id": 124, "properties": sample_entities[1]}})
        mock_base_manager.execute_query.return_value = mock_response
    else:
        mock_base_manager.execute_query.return_value = []
    
    # Create entity manager and test method
    entity_manager = EntityManager(mock_base_manager)
    results = entity_manager.search_entities(entity_type=entity_type, search_term=search_term)
    
    # Assertions
    assert len(results) == expected_count
    
    # Verify query execution
    mock_base_manager.execute_query.assert_called_once()
    
    # Verify appropriate WHERE clauses in the query
    if search_term or entity_type:
        call_args = mock_base_manager.execute_query.call_args[0]
        assert "WHERE" in call_args[0]
        
        if entity_type:
            assert entity_type in str(call_args)
        
        if search_term:
            assert "CONTAINS" in call_args[0] or "=~" in call_args[0]


def test_create_entity_error_handling(mock_base_manager, sample_entity):
    """Test error handling during entity creation."""
    # Setup mock to raise exception
    mock_base_manager.execute_query.side_effect = Exception("Database error")
    
    # Create entity manager and test method
    entity_manager = EntityManager(mock_base_manager)
    
    # Call should not raise an exception but return None
    result = entity_manager.create_entity(sample_entity)
    assert result is None


def test_entity_with_valid_schema(mock_base_manager):
    """Test validation of entity schema."""
    # Valid entity schema
    valid_entity = {
        "name": "valid_entity",
        "entityType": "ValidType"
    }
    
    # Setup mock response
    mock_base_manager.execute_query.return_value = [{"e": {"id": 123, "properties": valid_entity}}]
    
    # Create entity manager and test method
    entity_manager = EntityManager(mock_base_manager)
    result = entity_manager.create_entity(valid_entity)
    
    # Assertions
    assert result is not None
    mock_base_manager.execute_query.assert_called_once()


@pytest.mark.parametrize("invalid_entity", [
    # Entity missing name
    {"entityType": "InvalidType"},
    # Entity missing type
    {"name": "invalid_entity"},
    # Entity with wrong data types
    {"name": 123, "entityType": "InvalidType"},
    # Completely empty entity
    {}
])
def test_entity_with_invalid_schema(mock_base_manager, invalid_entity):
    """Test validation fails for invalid entity schemas."""
    # Create entity manager
    entity_manager = EntityManager(mock_base_manager)
    
    # Invalid schemas should be rejected
    result = entity_manager.create_entity(invalid_entity)
    
    # Should return None or raise ValidationError depending on implementation
    assert result is None
    
    # Verify no query execution attempted for invalid schema
    mock_base_manager.execute_query.assert_not_called() 