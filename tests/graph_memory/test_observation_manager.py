import pytest
from unittest.mock import MagicMock, patch, call
import json

from src.graph_memory.observation_manager import ObservationManager


def test_init(mock_base_manager):
    """Test initialization of ObservationManager"""
    manager = ObservationManager(mock_base_manager)
    
    assert manager.base_manager == mock_base_manager
    assert manager.logger == mock_base_manager.logger


def test_add_observation(mock_base_manager):
    """Test adding an observation to an entity"""
    # Setup
    with patch.object(ObservationManager, '_add_observation_to_entity') as mock_add:
        # Mock return value as a tuple (observation_id, created_timestamp)
        mock_add.return_value = ("obs-123", "2023-01-01T12:00:00")
        mock_base_manager.ensure_initialized = MagicMock()
        
        manager = ObservationManager(mock_base_manager)
        entity_name = "test_entity"
        
        # Execute
        observation_data = {
            "entity": entity_name,
            "content": "This is a test observation",
            "type": "test_observation"
        }
        
        result = json.loads(manager.add_observations([observation_data]))
        
        # Verify
        assert "added" in result
        assert len(result["added"]) == 1
        assert result["added"][0]["entity"] == entity_name
        assert result["added"][0]["content"] == "This is a test observation"
        
        # Verify the method was called with correct parameters (5 parameters)
        # entity_name, content, observation_type, metadata, confidence
        mock_add.assert_called_once_with(
            entity_name, 
            "This is a test observation", 
            "test_observation",  # type
            None,                # metadata
            None                 # confidence
        )


def test_add_observation_entity_not_found(mock_base_manager):
    """Test adding an observation to a non-existent entity"""
    # Setup
    with patch.object(ObservationManager, '_add_observation_to_entity') as mock_add:
        
        mock_add.side_effect = Exception("Entity not found")
        mock_base_manager.ensure_initialized = MagicMock()
        
        manager = ObservationManager(mock_base_manager)
        entity_name = "nonexistent_entity"
        
        # Execute
        observation_data = {
            "entity": entity_name,
            "content": "This is a test observation"
        }
        
        result = json.loads(manager.add_observations([observation_data]))
        
        # Verify
        assert "added" in result
        assert len(result["added"]) == 0
        assert "errors" in result
        assert len(result["errors"]) == 1
        assert result["errors"][0]["entity"] == entity_name
        assert "Entity not found" in result["errors"][0]["error"]
        
        # Verify the method was called with correct parameters (all 5 parameters)
        mock_add.assert_called_once_with(
            entity_name, 
            "This is a test observation", 
            None,  # type
            None,  # metadata
            None   # confidence
        )


def test_add_observations_batch(mock_base_manager):
    """Test adding multiple observations in a batch"""
    # Setup
    with patch.object(ObservationManager, '_add_observation_to_entity') as mock_add:
        
        # Return tuples of (id, timestamp) for each call
        mock_add.side_effect = [
            ("obs-1", "2023-01-01T12:00:00"), 
            ("obs-2", "2023-01-01T12:01:00")
        ]
        mock_base_manager.ensure_initialized = MagicMock()
        
        manager = ObservationManager(mock_base_manager)
        entity_name = "test_entity"
        
        # Execute
        observations = [
            {
                "entity": entity_name,
                "content": "First observation",
                "type": "test"
            },
            {
                "entity": entity_name,
                "content": "Second observation",
                "type": "test"
            }
        ]
        
        result = json.loads(manager.add_observations(observations))
        
        # Verify
        assert "added" in result
        assert len(result["added"]) == 2
        assert result["added"][0]["entity"] == entity_name
        assert result["added"][0]["content"] == "First observation"
        assert result["added"][1]["entity"] == entity_name
        assert result["added"][1]["content"] == "Second observation"
        
        # Verify the method was called with correct parameters
        expected_calls = [
            call(entity_name, "First observation", "test", None, None),
            call(entity_name, "Second observation", "test", None, None)
        ]
        mock_add.assert_has_calls(expected_calls)


def test_add_observations_batch_mixed_entities(mock_base_manager):
    """Test adding observations to multiple different entities in a batch"""
    # Setup
    with patch.object(ObservationManager, '_add_observation_to_entity') as mock_add:
        
        # Return tuples of (id, timestamp) for each call
        mock_add.side_effect = [
            ("obs-1", "2023-01-01T12:00:00"), 
            ("obs-2", "2023-01-01T12:01:00")
        ]
        mock_base_manager.ensure_initialized = MagicMock()
        
        manager = ObservationManager(mock_base_manager)
        
        # Execute
        observations = [
            {
                "entity": "entity-1",
                "content": "Observation for entity 1",
                "type": "test"
            },
            {
                "entity": "entity-2",
                "content": "Observation for entity 2",
                "type": "test"
            }
        ]
        
        result = json.loads(manager.add_observations(observations))
        
        # Verify
        assert "added" in result
        assert len(result["added"]) == 2
        assert result["added"][0]["entity"] == "entity-1"
        assert result["added"][0]["content"] == "Observation for entity 1"
        assert result["added"][1]["entity"] == "entity-2"
        assert result["added"][1]["content"] == "Observation for entity 2"
        
        # Verify the method was called with correct parameters
        expected_calls = [
            call("entity-1", "Observation for entity 1", "test", None, None),
            call("entity-2", "Observation for entity 2", "test", None, None)
        ]
        mock_add.assert_has_calls(expected_calls)


def test_get_observations(mock_base_manager):
    """Test retrieving observations for an entity"""
    # Setup
    entity_name = "test_entity"
    
    # Create mock records for different queries
    # Mock for checking if entity exists
    mock_entity_record = MagicMock()
    mock_entity_record.get = MagicMock(return_value={"name": entity_name})
    
    # Mock for observations
    class MockObservationRecord:
        def get(self, key):
            data = {
                "id": "obs-1" if key == "id" else "obs-2",
                "content": "Test observation",
                "type": "test_type",
                "created": "2023-01-01",
                "lastUpdated": "2023-01-02"
            }
            return data.get(key)
    
    mock_obs_records = [MockObservationRecord(), MockObservationRecord()]
    
    # Setup mock responses
    mock_base_manager.safe_execute_read_query = MagicMock(side_effect=[
        [mock_entity_record],  # For entity check
        mock_obs_records       # For observations query
    ])
    mock_base_manager.ensure_initialized = MagicMock()
    
    # Create manager and test method
    manager = ObservationManager(mock_base_manager)
    result = json.loads(manager.get_entity_observations(entity_name))
    
    # Verify
    assert "observations" in result
    assert len(result["observations"]) == 2
    assert result["observations"][0]["entity"] == entity_name
    assert result["observations"][0]["content"] == "Test observation"
    assert result["observations"][0]["type"] == "test_type"
    
    # Verify query execution
    assert mock_base_manager.safe_execute_read_query.call_count == 2


def test_get_observations_entity_not_found(mock_base_manager):
    """Test retrieving observations for a non-existent entity"""
    # Setup
    entity_name = "nonexistent_entity"
    
    # Setup mock response for entity check (not found)
    mock_base_manager.safe_execute_read_query = MagicMock(return_value=[])
    mock_base_manager.ensure_initialized = MagicMock()
    
    # Create manager and test method
    manager = ObservationManager(mock_base_manager)
    result = json.loads(manager.get_entity_observations(entity_name))
    
    # Verify
    assert "error" in result
    assert entity_name in result["error"]
    assert "not found" in result["error"].lower()
    
    # Verify query execution
    mock_base_manager.safe_execute_read_query.assert_called_once()


def test_get_observations_with_filter(mock_base_manager):
    """Test retrieving observations with type filter"""
    # Setup
    entity_name = "test_entity"
    observation_type = "test_type"
    
    # Create mock records for different queries
    # Mock for checking if entity exists
    mock_entity_record = MagicMock()
    mock_entity_record.get = MagicMock(return_value={"name": entity_name})
    
    # Mock for observations
    class MockObservationRecord:
        def get(self, key):
            data = {
                "id": "obs-1",
                "content": "Test observation",
                "type": observation_type,
                "created": "2023-01-01",
                "lastUpdated": "2023-01-02"
            }
            return data.get(key)
    
    mock_obs_record = MockObservationRecord()
    
    # Setup mock responses
    mock_base_manager.safe_execute_read_query = MagicMock(side_effect=[
        [mock_entity_record],  # For entity check
        [mock_obs_record]      # For observations query
    ])
    mock_base_manager.ensure_initialized = MagicMock()
    
    # Create manager and test method
    manager = ObservationManager(mock_base_manager)
    result = json.loads(manager.get_entity_observations(entity_name, observation_type))
    
    # Verify
    assert "observations" in result
    assert len(result["observations"]) == 1
    assert result["observations"][0]["entity"] == entity_name
    assert result["observations"][0]["type"] == observation_type
    
    # Verify query execution with correct parameters
    assert mock_base_manager.safe_execute_read_query.call_count == 2


def test_delete_observation(mock_base_manager):
    """Test deleting an observation"""
    # Setup
    entity_name = "test_entity"
    observation_id = "obs-123"
    
    # Mock records for entity check and deletion
    mock_deletion_record = MagicMock()
    mock_deletion_record.get = MagicMock(return_value=1)  # 1 relationship deleted
    
    # Setup mock responses
    mock_base_manager.safe_execute_write_query = MagicMock(return_value=[mock_deletion_record])
    mock_base_manager.ensure_initialized = MagicMock()
    
    # Create manager and test method
    manager = ObservationManager(mock_base_manager)
    result = json.loads(manager.delete_observation(entity_name, observation_id=observation_id))
    
    # Verify
    assert "status" in result
    assert result["status"] == "success"
    assert "message" in result
    assert entity_name in result["message"]
    
    # Verify query execution
    mock_base_manager.safe_execute_write_query.assert_called_once()


def test_delete_observation_not_found(mock_base_manager):
    """Test deleting a non-existent observation"""
    # Setup
    entity_name = "test_entity"
    observation_id = "nonexistent-obs"
    
    # Create mock record for deletion (nothing deleted)
    mock_record = MagicMock()
    mock_record.get = MagicMock(return_value=0)  # 0 observations deleted
    
    # Setup mock responses
    mock_base_manager.safe_execute_write_query = MagicMock(return_value=[mock_record])
    mock_base_manager.ensure_initialized = MagicMock()
    
    # Create manager and test method
    manager = ObservationManager(mock_base_manager)
    result = json.loads(manager.delete_observation(entity_name, observation_id=observation_id))
    
    # Verify
    assert "status" in result
    assert result["status"] == "success"
    assert "message" in result
    assert "No matching observations" in result["message"]
    
    # Verify query executions
    mock_base_manager.safe_execute_write_query.assert_called_once()


def test_update_observation(mock_base_manager):
    """Test updating an observation"""
    # Setup
    entity_name = "test_entity"
    observation_id = "obs-1"
    updated_content = "Updated content"
    updated_type = "updated_type"
    
    # Mock records for entity check and update
    class MockObservationRecord:
        def get(self, key):
            if key == "o":
                mock_obs = MagicMock()
                mock_obs.items.return_value = {
                    "id": observation_id,
                    "content": updated_content,
                    "type": updated_type,
                    "created": "2023-01-01",
                    "lastUpdated": "2023-02-01"
                }.items()
                return mock_obs
            return None
    
    mock_observation_record = MockObservationRecord()
    
    # Setup mock responses
    mock_base_manager.safe_execute_read_query = MagicMock(return_value=[mock_observation_record])
    mock_base_manager.safe_execute_write_query = MagicMock(return_value=[mock_observation_record])
    mock_base_manager.ensure_initialized = MagicMock()
    
    # Create manager and test method
    manager = ObservationManager(mock_base_manager)
    result = json.loads(manager.update_observation(entity_name, observation_id, updated_content, updated_type))
    
    # Verify
    assert "observation" in result
    assert result["observation"]["id"] == observation_id
    assert result["observation"]["content"] == updated_content
    assert result["observation"]["type"] == updated_type
    assert result["observation"]["entity"] == entity_name
    
    # Verify query executions
    mock_base_manager.safe_execute_read_query.assert_called_once()
    mock_base_manager.safe_execute_write_query.assert_called_once()


def test_update_observation_not_found(mock_base_manager):
    """Test updating a non-existent observation"""
    # Setup
    entity_name = "test_entity"
    observation_id = "nonexistent-obs"
    updated_content = "Updated content"
    
    # Setup mock response for observation check (not found)
    mock_base_manager.safe_execute_read_query = MagicMock(return_value=[])
    mock_base_manager.ensure_initialized = MagicMock()
    
    # Create manager and test method
    manager = ObservationManager(mock_base_manager)
    result = json.loads(manager.update_observation(entity_name, observation_id, updated_content))
    
    # Verify
    assert "error" in result
    assert observation_id in result["error"]
    assert "not found" in result["error"].lower()
    
    # Verify query execution
    mock_base_manager.safe_execute_read_query.assert_called_once()


def test_error_handling(mock_base_manager):
    """Test error handling during observation operations."""
    # Setup base manager to raise exception
    mock_base_manager.ensure_initialized = MagicMock()
    mock_base_manager.safe_execute_read_query = MagicMock(side_effect=Exception("Database error"))
    mock_base_manager.safe_execute_write_query = MagicMock(side_effect=Exception("Database error"))
    
    # Create observation manager and test methods
    manager = ObservationManager(mock_base_manager)
    entity_name = "test_entity"
    observation_id = "obs-1"
    
    # Test get_observations error handling
    get_result = json.loads(manager.get_entity_observations(entity_name))
    assert "error" in get_result
    assert "Database error" in get_result["error"]
    
    # Test update_observation error handling
    update_result = json.loads(manager.update_observation(entity_name, observation_id, "Updated content"))
    assert "error" in update_result
    assert "Database error" in update_result["error"]
    
    # Test delete_observation error handling
    delete_result = json.loads(manager.delete_observation(entity_name, observation_id=observation_id))
    assert "error" in delete_result
    assert "Database error" in delete_result["error"]
    
    # Test add_observations error handling with global exception
    with patch.object(ObservationManager, '_add_observation_to_entity') as mock_add:
        mock_add.side_effect = Exception("Internal method error")
        add_result = json.loads(manager.add_observations([{
            "entity": entity_name,
            "content": "Test observation"
        }]))
        assert "errors" in add_result
        assert "Internal method error" in str(add_result["errors"][0]["error"])


def test_add_single_observation(mock_base_manager):
    """Test adding a single observation."""
    # Test data
    entity_name = "test_entity"
    observation_content = "Test observation content"
    observation_type = "COMMENT"
    
    # Create observation data
    observation_data = {
        "entity": entity_name,
        "content": observation_content,
        "type": observation_type
    }
    
    # Setup
    with patch.object(ObservationManager, '_add_observation_to_entity') as mock_add:
        
        # Return tuple of (id, timestamp)
        mock_add.return_value = ("obs-123", "2023-01-01T12:00:00")
        mock_base_manager.ensure_initialized = MagicMock()
        
        manager = ObservationManager(mock_base_manager)
        
        # Execute
        result = json.loads(manager.add_observations([observation_data]))
        
        # Verify
        assert "added" in result
        assert len(result["added"]) == 1
        assert result["added"][0]["entity"] == entity_name
        assert result["added"][0]["content"] == observation_content
        assert result["added"][0]["type"] == observation_type
        
        # Verify the method was called with correct parameters
        mock_add.assert_called_once_with(
            entity_name, 
            observation_content, 
            observation_type,
            None,  # metadata
            None   # confidence
        ) 