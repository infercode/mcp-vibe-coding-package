import pytest
from unittest.mock import MagicMock, patch, call
import json

from src.graph_memory import ObservationManager


def test_init(mock_base_manager, mock_entity_manager, mock_logger):
    """Test initialization of ObservationManager"""
    manager = ObservationManager(mock_base_manager, mock_entity_manager, mock_logger)
    
    assert manager.base_manager == mock_base_manager
    assert manager.entity_manager == mock_entity_manager
    assert manager.logger == mock_logger


def test_add_observation(mock_base_manager, mock_entity_manager, mock_logger, sample_entity):
    """Test adding an observation to an entity"""
    # Setup
    manager = ObservationManager(mock_base_manager, mock_entity_manager, mock_logger)
    entity_id = "entity-123"
    
    # Mock entity retrieval
    mock_entity_manager.get_entity.return_value = json.dumps(sample_entity)
    
    # Mock query execution
    mock_base_manager.execute_query.return_value = {"results": [{"observation_id": "obs-123"}]}
    
    # Execute
    observation_data = {
        "entity_id": entity_id,
        "content": "This is a test observation",
        "metadata": {"source": "test", "confidence": 0.95}
    }
    
    result = manager.add_observation(observation_data)
    
    # Verify
    result_obj = json.loads(result)
    assert result_obj["status"] == "success"
    assert "observation_id" in result_obj
    
    # Verify entity_manager was called for lookup
    mock_entity_manager.get_entity.assert_called_once_with(entity_id)
    
    # Verify base_manager.execute_query was called with correct parameters
    mock_base_manager.execute_query.assert_called_once()
    # Extract the query from the call
    query = mock_base_manager.execute_query.call_args[0][0]
    assert "CREATE" in query
    assert "Observation" in query
    assert "content:" in query
    assert "HAS_OBSERVATION" in query


def test_add_observation_entity_not_found(mock_base_manager, mock_entity_manager, mock_logger):
    """Test adding an observation to a non-existent entity"""
    # Setup
    manager = ObservationManager(mock_base_manager, mock_entity_manager, mock_logger)
    entity_id = "nonexistent-entity"
    
    # Mock entity retrieval - not found
    mock_entity_manager.get_entity.return_value = None
    
    # Execute
    observation_data = {
        "entity_id": entity_id,
        "content": "This is a test observation"
    }
    
    result = manager.add_observation(observation_data)
    
    # Verify
    result_obj = json.loads(result)
    assert result_obj["status"] == "error"
    assert "not found" in result_obj["message"].lower()
    
    # Verify entity_manager was called for lookup
    mock_entity_manager.get_entity.assert_called_once_with(entity_id)
    
    # Verify base_manager.execute_query was NOT called
    mock_base_manager.execute_query.assert_not_called()


def test_add_observations_batch(mock_base_manager, mock_entity_manager, mock_logger, sample_entity):
    """Test adding multiple observations in a batch"""
    # Setup
    manager = ObservationManager(mock_base_manager, mock_entity_manager, mock_logger)
    entity_id = "entity-123"
    
    # Mock entity retrieval
    mock_entity_manager.get_entity.return_value = json.dumps(sample_entity)
    
    # Mock query execution for batch addition
    mock_base_manager.execute_query.return_value = {"results": [
        {"observation_id": "obs-1"}, 
        {"observation_id": "obs-2"}
    ]}
    
    # Execute
    observations = [
        {
            "entity_id": entity_id,
            "content": "First observation",
            "metadata": {"order": 1}
        },
        {
            "entity_id": entity_id,
            "content": "Second observation",
            "metadata": {"order": 2}
        }
    ]
    
    result = manager.add_observations_batch(observations)
    
    # Verify
    result_obj = json.loads(result)
    assert result_obj["status"] == "success"
    assert "observation_ids" in result_obj
    assert len(result_obj["observation_ids"]) == 2
    
    # Verify entity_manager was called for lookup
    mock_entity_manager.get_entity.assert_called_once_with(entity_id)
    
    # Verify base_manager.execute_query was called with correct parameters
    mock_base_manager.execute_query.assert_called_once()
    # Extract the query from the call
    query = mock_base_manager.execute_query.call_args[0][0]
    assert "CREATE" in query
    assert "Observation" in query
    assert "UNWIND" in query  # Batch processing
    assert "HAS_OBSERVATION" in query


def test_add_observations_batch_mixed_entities(mock_base_manager, mock_entity_manager, mock_logger, sample_entities):
    """Test adding observations to multiple different entities in a batch"""
    # Setup
    manager = ObservationManager(mock_base_manager, mock_entity_manager, mock_logger)
    
    # Mock entity retrieval for different entities
    def get_entity_side_effect(entity_id):
        if entity_id == "entity-1":
            return json.dumps(sample_entities[0])
        elif entity_id == "entity-2":
            return json.dumps(sample_entities[1])
        return None
    
    mock_entity_manager.get_entity.side_effect = get_entity_side_effect
    
    # Mock query execution for batch addition
    mock_base_manager.execute_query.return_value = {"results": [
        {"observation_id": "obs-1"}, 
        {"observation_id": "obs-2"}
    ]}
    
    # Execute
    observations = [
        {
            "entity_id": "entity-1",
            "content": "Observation for entity 1",
            "metadata": {"source": "test1"}
        },
        {
            "entity_id": "entity-2",
            "content": "Observation for entity 2",
            "metadata": {"source": "test2"}
        }
    ]
    
    result = manager.add_observations_batch(observations)
    
    # Verify
    result_obj = json.loads(result)
    assert result_obj["status"] == "success"
    assert "observation_ids" in result_obj
    assert len(result_obj["observation_ids"]) == 2
    
    # Verify entity_manager was called for both lookups
    mock_entity_manager.get_entity.assert_has_calls([
        call("entity-1"),
        call("entity-2")
    ])
    
    # Verify base_manager.execute_query was called
    mock_base_manager.execute_query.assert_called_once()


def test_get_observations(mock_base_manager, mock_entity_manager, mock_logger, sample_entity):
    """Test retrieving observations for an entity"""
    # Setup
    manager = ObservationManager(mock_base_manager, mock_entity_manager, mock_logger)
    entity_id = "entity-123"
    
    # Mock entity retrieval
    mock_entity_manager.get_entity.return_value = json.dumps(sample_entity)
    
    # Mock observations retrieval
    mock_observations = [
        {
            "observation": {
                "id": "obs-1",
                "content": "First observation",
                "created_at": "2023-01-01T12:00:00",
                "metadata": {"source": "test1"}
            }
        },
        {
            "observation": {
                "id": "obs-2",
                "content": "Second observation",
                "created_at": "2023-01-02T12:00:00",
                "metadata": {"source": "test2"}
            }
        }
    ]
    
    mock_base_manager.execute_query.return_value = {"results": mock_observations}
    
    # Execute
    result = manager.get_observations(entity_id)
    
    # Verify
    result_list = json.loads(result)
    assert len(result_list) == 2
    assert result_list[0]["id"] == "obs-1"
    assert result_list[1]["id"] == "obs-2"
    
    # Verify entity_manager was called for lookup
    mock_entity_manager.get_entity.assert_called_once_with(entity_id)
    
    # Verify base_manager.execute_query was called with correct parameters
    mock_base_manager.execute_query.assert_called_once()
    # Extract the query from the call
    query = mock_base_manager.execute_query.call_args[0][0]
    assert "MATCH" in query
    assert "HAS_OBSERVATION" in query
    assert entity_id in query


def test_get_observations_entity_not_found(mock_base_manager, mock_entity_manager, mock_logger):
    """Test retrieving observations for a non-existent entity"""
    # Setup
    manager = ObservationManager(mock_base_manager, mock_entity_manager, mock_logger)
    entity_id = "nonexistent-entity"
    
    # Mock entity retrieval - not found
    mock_entity_manager.get_entity.return_value = None
    
    # Execute
    result = manager.get_observations(entity_id)
    
    # Verify
    result_obj = json.loads(result)
    assert result_obj["status"] == "error"
    assert "not found" in result_obj["message"].lower()
    
    # Verify entity_manager was called for lookup
    mock_entity_manager.get_entity.assert_called_once_with(entity_id)
    
    # Verify base_manager.execute_query was NOT called
    mock_base_manager.execute_query.assert_not_called()


def test_get_observations_with_filter(mock_base_manager, mock_entity_manager, mock_logger, sample_entity):
    """Test retrieving observations with specific filters"""
    # Setup
    manager = ObservationManager(mock_base_manager, mock_entity_manager, mock_logger)
    entity_id = "entity-123"
    
    # Mock entity retrieval
    mock_entity_manager.get_entity.return_value = json.dumps(sample_entity)
    
    # Mock filtered observations retrieval
    mock_observations = [
        {
            "observation": {
                "id": "obs-1",
                "content": "Filtered observation",
                "created_at": "2023-01-01T12:00:00",
                "metadata": {"source": "test1", "importance": "high"}
            }
        }
    ]
    
    mock_base_manager.execute_query.return_value = {"results": mock_observations}
    
    # Execute with filter
    result = manager.get_observations(
        entity_id,
        filters={"metadata.importance": "high"}
    )
    
    # Verify
    result_list = json.loads(result)
    assert len(result_list) == 1
    assert result_list[0]["id"] == "obs-1"
    assert result_list[0]["metadata"]["importance"] == "high"
    
    # Verify entity_manager was called for lookup
    mock_entity_manager.get_entity.assert_called_once_with(entity_id)
    
    # Verify base_manager.execute_query was called with correct parameters
    mock_base_manager.execute_query.assert_called_once()
    # Extract the query from the call
    query = mock_base_manager.execute_query.call_args[0][0]
    assert "MATCH" in query
    assert "HAS_OBSERVATION" in query
    assert "importance:" in query or "importance =" in query
    assert "high" in query


def test_delete_observation(mock_base_manager, mock_entity_manager, mock_logger):
    """Test deleting a specific observation"""
    # Setup
    manager = ObservationManager(mock_base_manager, mock_entity_manager, mock_logger)
    entity_id = "entity-123"
    observation_id = "obs-123"
    
    # Mock deletion query
    mock_base_manager.execute_query.return_value = {"results": [{"deleted": True}]}
    
    # Execute
    result = manager.delete_observation(entity_id, observation_id)
    
    # Verify
    result_obj = json.loads(result)
    assert result_obj["status"] == "success"
    
    # Verify base_manager.execute_query was called with correct parameters
    mock_base_manager.execute_query.assert_called_once()
    # Extract the query from the call
    query = mock_base_manager.execute_query.call_args[0][0]
    assert "MATCH" in query
    assert "DETACH DELETE" in query
    assert entity_id in query
    assert observation_id in query


def test_delete_observation_not_found(mock_base_manager, mock_entity_manager, mock_logger):
    """Test deleting a non-existent observation"""
    # Setup
    manager = ObservationManager(mock_base_manager, mock_entity_manager, mock_logger)
    entity_id = "entity-123"
    observation_id = "nonexistent-obs"
    
    # Mock deletion query - not found
    mock_base_manager.execute_query.return_value = {"results": []}
    
    # Execute
    result = manager.delete_observation(entity_id, observation_id)
    
    # Verify
    result_obj = json.loads(result)
    assert result_obj["status"] == "error"
    assert "not found" in result_obj["message"].lower()
    
    # Verify base_manager.execute_query was called
    mock_base_manager.execute_query.assert_called_once()


def test_delete_all_observations(mock_base_manager, mock_entity_manager, mock_logger, sample_entity):
    """Test deleting all observations for an entity"""
    # Setup
    manager = ObservationManager(mock_base_manager, mock_entity_manager, mock_logger)
    entity_id = "entity-123"
    
    # Mock entity retrieval
    mock_entity_manager.get_entity.return_value = json.dumps(sample_entity)
    
    # Mock deletion query
    mock_base_manager.execute_query.return_value = {"results": [{"deleted_count": 5}]}
    
    # Execute
    result = manager.delete_all_observations(entity_id)
    
    # Verify
    result_obj = json.loads(result)
    assert result_obj["status"] == "success"
    assert result_obj["deleted_count"] == 5
    
    # Verify entity_manager was called for lookup
    mock_entity_manager.get_entity.assert_called_once_with(entity_id)
    
    # Verify base_manager.execute_query was called with correct parameters
    mock_base_manager.execute_query.assert_called_once()
    # Extract the query from the call
    query = mock_base_manager.execute_query.call_args[0][0]
    assert "MATCH" in query
    assert "DETACH DELETE" in query
    assert "HAS_OBSERVATION" in query
    assert entity_id in query


def test_delete_all_observations_entity_not_found(mock_base_manager, mock_entity_manager, mock_logger):
    """Test deleting all observations for a non-existent entity"""
    # Setup
    manager = ObservationManager(mock_base_manager, mock_entity_manager, mock_logger)
    entity_id = "nonexistent-entity"
    
    # Mock entity retrieval - not found
    mock_entity_manager.get_entity.return_value = None
    
    # Execute
    result = manager.delete_all_observations(entity_id)
    
    # Verify
    result_obj = json.loads(result)
    assert result_obj["status"] == "error"
    assert "not found" in result_obj["message"].lower()
    
    # Verify entity_manager was called for lookup
    mock_entity_manager.get_entity.assert_called_once_with(entity_id)
    
    # Verify base_manager.execute_query was NOT called
    mock_base_manager.execute_query.assert_not_called()


def test_update_observation(mock_base_manager, mock_entity_manager, mock_logger):
    """Test updating an observation"""
    # Setup
    manager = ObservationManager(mock_base_manager, mock_entity_manager, mock_logger)
    entity_id = "entity-123"
    observation_id = "obs-123"
    
    # Mock update query
    mock_base_manager.execute_query.return_value = {"results": [{"updated": True}]}
    
    # Execute
    update_data = {
        "content": "Updated observation content",
        "metadata": {
            "updated": True,
            "importance": "high"
        }
    }
    
    result = manager.update_observation(entity_id, observation_id, update_data)
    
    # Verify
    result_obj = json.loads(result)
    assert result_obj["status"] == "success"
    
    # Verify base_manager.execute_query was called with correct parameters
    mock_base_manager.execute_query.assert_called_once()
    # Extract the query from the call
    query = mock_base_manager.execute_query.call_args[0][0]
    assert "MATCH" in query
    assert "SET" in query
    assert entity_id in query
    assert observation_id in query
    assert "Updated observation content" in query


def test_update_observation_not_found(mock_base_manager, mock_entity_manager, mock_logger):
    """Test updating a non-existent observation"""
    # Setup
    manager = ObservationManager(mock_base_manager, mock_entity_manager, mock_logger)
    entity_id = "entity-123"
    observation_id = "nonexistent-obs"
    
    # Mock update query - not found
    mock_base_manager.execute_query.return_value = {"results": []}
    
    # Execute
    update_data = {
        "content": "Updated observation content"
    }
    
    result = manager.update_observation(entity_id, observation_id, update_data)
    
    # Verify
    result_obj = json.loads(result)
    assert result_obj["status"] == "error"
    assert "not found" in result_obj["message"].lower()
    
    # Verify base_manager.execute_query was called
    mock_base_manager.execute_query.assert_called_once()


def test_count_observations(mock_base_manager, mock_entity_manager, mock_logger, sample_entity):
    """Test counting observations for an entity"""
    # Setup
    manager = ObservationManager(mock_base_manager, mock_entity_manager, mock_logger)
    entity_id = "entity-123"
    
    # Mock entity retrieval
    mock_entity_manager.get_entity.return_value = json.dumps(sample_entity)
    
    # Mock count query
    mock_base_manager.execute_query.return_value = {"results": [{"count": 10}]}
    
    # Execute
    result = manager.count_observations(entity_id)
    
    # Verify
    result_obj = json.loads(result)
    assert result_obj["count"] == 10
    
    # Verify entity_manager was called for lookup
    mock_entity_manager.get_entity.assert_called_once_with(entity_id)
    
    # Verify base_manager.execute_query was called with correct parameters
    mock_base_manager.execute_query.assert_called_once()
    # Extract the query from the call
    query = mock_base_manager.execute_query.call_args[0][0]
    assert "MATCH" in query
    assert "COUNT" in query
    assert "HAS_OBSERVATION" in query
    assert entity_id in query


def test_get_recent_observations(mock_base_manager, mock_entity_manager, mock_logger, sample_entity):
    """Test retrieving recent observations for an entity"""
    # Setup
    manager = ObservationManager(mock_base_manager, mock_entity_manager, mock_logger)
    entity_id = "entity-123"
    
    # Mock entity retrieval
    mock_entity_manager.get_entity.return_value = json.dumps(sample_entity)
    
    # Mock recent observations retrieval
    mock_observations = [
        {
            "observation": {
                "id": "obs-1",
                "content": "Recent observation 1",
                "created_at": "2023-01-02T12:00:00",
                "metadata": {"source": "test1"}
            }
        },
        {
            "observation": {
                "id": "obs-2",
                "content": "Recent observation 2",
                "created_at": "2023-01-01T12:00:00",
                "metadata": {"source": "test2"}
            }
        }
    ]
    
    mock_base_manager.execute_query.return_value = {"results": mock_observations}
    
    # Execute
    result = manager.get_recent_observations(entity_id, limit=2)
    
    # Verify
    result_list = json.loads(result)
    assert len(result_list) == 2
    assert result_list[0]["id"] == "obs-1"
    assert result_list[1]["id"] == "obs-2"
    
    # Verify entity_manager was called for lookup
    mock_entity_manager.get_entity.assert_called_once_with(entity_id)
    
    # Verify base_manager.execute_query was called with correct parameters
    mock_base_manager.execute_query.assert_called_once()
    # Extract the query from the call
    query = mock_base_manager.execute_query.call_args[0][0]
    assert "MATCH" in query
    assert "HAS_OBSERVATION" in query
    assert "ORDER BY" in query
    assert "LIMIT 2" in query
    assert entity_id in query 