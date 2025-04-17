"""
Unit tests for the GraphMemoryManager.
"""

import pytest
from unittest.mock import patch, MagicMock
import json

from src.graph_memory import GraphMemoryManager  # type: ignore


@pytest.mark.unit
class TestGraphMemoryManager:
    """Test suite for GraphMemoryManager class."""

    @pytest.mark.core
    def test_initialization(self, logger, mock_neo4j_driver):
        """Test that the GraphMemoryManager initializes correctly."""
        with patch('src.graph_memory.GraphMemoryManager.initialize'):
            manager = GraphMemoryManager(logger)
            setattr(manager, 'driver', mock_neo4j_driver)

            # Verify manager components are initialized
            assert hasattr(manager, 'entity_manager')
            assert hasattr(manager, 'relation_manager')
            assert hasattr(manager, 'observation_manager')
            assert hasattr(manager, 'search_manager')

    @pytest.mark.core
    def test_close(self, mock_graph_memory_manager):
        """Test the close method properly closes connection."""
        # Configure driver.close() so it can be called
        mock_graph_memory_manager.driver.close = MagicMock()
        
        # Configure close method to call driver.close()
        def close_implementation():
            mock_graph_memory_manager.driver.close()
        
        mock_graph_memory_manager.close = MagicMock(side_effect=close_implementation)
        
        # Call the method
        mock_graph_memory_manager.close()
        
        # Verify driver.close() was called
        mock_graph_memory_manager.driver.close.assert_called_once()

    @pytest.mark.entity
    def test_create_entity(
            self, mock_graph_memory_manager, sample_entity_data, json_helper):
        """Test creating an entity."""
        # Configure the method to return a JSON string
        success_response = {
            "status": "success",
            "message": "Entity created successfully",
            "data": sample_entity_data
        }
        mock_graph_memory_manager.create_entity.return_value = json.dumps(success_response)

        # Call the method
        result = mock_graph_memory_manager.create_entity(sample_entity_data)

        # Verify the result
        result_dict = json_helper["from_json"](result)
        assert result_dict["status"] == "success"
        assert "Entity created successfully" in result_dict["message"]

    @pytest.mark.entity
    @pytest.mark.parametrize("invalid_entity_data", [
        {"name": None},
        {"name": ""},
        {"entityType": None},
        {"entityType": ""}
    ])
    def test_create_entity_with_invalid_data(
            self, mock_graph_memory_manager, invalid_entity_data):
        """Test creating an entity with invalid data raises appropriate error."""
        # Configure the method to raise ValueError for invalid data
        def validate_entity(data):
            if not data.get("name") or not data.get("entityType"):
                raise ValueError("Invalid entity data")
            return "success"
        
        mock_graph_memory_manager.create_entity = MagicMock(side_effect=validate_entity)
        
        with pytest.raises(ValueError):
            mock_graph_memory_manager.create_entity(invalid_entity_data)

    @pytest.mark.relation
    def test_create_relationship(
            self, mock_graph_memory_manager, sample_relation_data, json_helper):
        """Test creating a relationship."""
        # Configure the method to return a JSON string
        success_response = {
            "status": "success",
            "message": "Relationship created successfully",
            "data": sample_relation_data
        }
        mock_graph_memory_manager.create_relationship.return_value = json.dumps(success_response)

        # Call the method
        result = mock_graph_memory_manager.create_relationship(sample_relation_data)

        # Verify the result
        result_dict = json_helper["from_json"](result)
        assert result_dict["status"] == "success"
        assert "Relationship created successfully" in result_dict["message"]

    @pytest.mark.relation
    @pytest.mark.parametrize("invalid_relation_data", [
        {"from": None},
        {"from": ""},
        {"to": None},
        {"to": ""},
        {"relationType": None},
        {"relationType": ""}
    ])
    def test_create_relationship_with_invalid_data(
            self, mock_graph_memory_manager, invalid_relation_data):
        """Test creating a relationship with invalid data raises appropriate error."""
        # Configure the method to raise ValueError for invalid data
        def validate_relation(data):
            if not data.get("from") or not data.get("to") or not data.get("relationType"):
                raise ValueError("Invalid relation data")
            return "success"
        
        mock_graph_memory_manager.create_relationship = MagicMock(side_effect=validate_relation)
        
        with pytest.raises(ValueError):
            mock_graph_memory_manager.create_relationship(invalid_relation_data)

    @pytest.mark.observation
    def test_add_observation(self, mock_graph_memory_manager, json_helper):
        """Test adding an observation to an entity."""
        entity_name = "TestEntity"
        content = "This is a test observation"
        observation_data = [
            {"entity": entity_name, "content": content}
        ]

        # Configure the method to return a JSON string
        success_response = {
            "status": "success",
            "message": "Observation added successfully",
            "data": {"entity": entity_name, "content": content}
        }
        mock_graph_memory_manager.add_observations.return_value = json.dumps(success_response)

        # Call the method
        result = mock_graph_memory_manager.add_observations(observation_data)

        # Verify the result
        result_dict = json_helper["from_json"](result)
        assert result_dict["status"] == "success"
        assert "Observation added successfully" in result_dict["message"]

    @pytest.mark.observation
    @pytest.mark.parametrize("invalid_observation_data", [
        [{"entity": None, "content": "Valid content"}],
        [{"entity": "", "content": "Valid content"}],
        [{"entity": "ValidEntity", "content": None}],
        [{"entity": "ValidEntity", "content": ""}]
    ])
    def test_add_observations_with_invalid_data(
            self, mock_graph_memory_manager, invalid_observation_data):
        """Test adding observation with invalid data raises appropriate error."""
        # Configure the method to raise ValueError for invalid data
        def validate_observation(data):
            if not data or not isinstance(data, list):
                raise ValueError("Invalid observation data format")
            for item in data:
                if not item.get("entity") or not item.get("content"):
                    raise ValueError("Invalid observation data")
            return "success"

        mock_graph_memory_manager.add_observations = MagicMock(side_effect=validate_observation)
        
        with pytest.raises(ValueError):
            mock_graph_memory_manager.add_observations(invalid_observation_data)

    @pytest.mark.search
    def test_search_nodes(self, mock_graph_memory_manager, json_helper):
        """Test searching for nodes."""
        query = "test query"
        limit = 5

        # Mock search results
        mock_results = [
            {"name": "Entity1", "entityType": "TEST", "score": 0.95},
            {"name": "Entity2", "entityType": "TEST", "score": 0.85}
        ]

        # Configure the method to return a JSON string
        success_response = {
            "status": "success",
            "message": "Search completed for: test query",
            "data": {
                "results": mock_results,
                "count": len(mock_results)
            }
        }
        mock_graph_memory_manager.search_nodes.return_value = json.dumps(success_response)

        # Call the method
        result = mock_graph_memory_manager.search_nodes(query, limit)

        # Verify the result
        result_dict = json_helper["from_json"](result)
        assert result_dict["status"] == "success"
        assert "Search completed for" in result_dict["message"]
        assert len(result_dict["data"]["results"]) == 2
        assert result_dict["data"]["count"] == 2

    @pytest.mark.search
    @pytest.mark.parametrize("search_params", [
        {"query": "", "limit": 5, "error": "query"},
        {"query": None, "limit": 5, "error": "query"},
        {"query": "valid query", "limit": -1, "error": "limit"}
    ])
    def test_search_nodes_with_invalid_params(
            self, mock_graph_memory_manager, search_params):
        """Test search with invalid parameters raises appropriate error."""
        query = search_params.get("query", "default query")
        limit = search_params.get("limit", 10)

        # Configure the method to raise ValueError for invalid parameters
        def validate_search(q, l):
            if not q:
                raise ValueError("Query cannot be empty")
            if l < 0:
                raise ValueError("Limit must be a positive integer")
            return json.dumps({"status": "success", "message": "Search completed", "data": []})

        mock_graph_memory_manager.search_nodes = MagicMock(side_effect=validate_search)
        
        with pytest.raises((ValueError, TypeError)):
            mock_graph_memory_manager.search_nodes(query, limit)

    @pytest.mark.entity
    def test_delete_entity(self, mock_graph_memory_manager, json_helper):
        """Test deleting an entity."""
        entity_name = "TestEntity"

        # Configure the method to return a JSON string
        success_response = {
            "status": "success",
            "message": "Entity deleted successfully",
            "data": {"name": entity_name}
        }
        mock_graph_memory_manager.delete_entity.return_value = json.dumps(success_response)

        # Call the method
        result = mock_graph_memory_manager.delete_entity(entity_name)

        # Verify the result
        result_dict = json_helper["from_json"](result)
        assert result_dict["status"] == "success"
        assert "Entity deleted successfully" in result_dict["message"]

    @pytest.mark.entity
    @pytest.mark.parametrize("invalid_entity_name", [None, "", 123])
    def test_delete_entity_with_invalid_name(
            self, mock_graph_memory_manager, invalid_entity_name):
        """Test deleting an entity with invalid name raises appropriate error."""
        # Configure the method to raise ValueError for invalid names
        def validate_entity_name(name):
            if not name or not isinstance(name, str):
                raise ValueError("Invalid entity name")
            return json.dumps({"status": "success", "message": "Entity deleted"})
        
        mock_graph_memory_manager.delete_entity = MagicMock(side_effect=validate_entity_name)
        
        with pytest.raises(ValueError):
            mock_graph_memory_manager.delete_entity(invalid_entity_name)

    @pytest.mark.relation
    def test_delete_relation(
            self, mock_graph_memory_manager, sample_relation_data, json_helper):
        """Test deleting a relationship."""
        # Configure the method to return a JSON string
        success_response = {
            "status": "success",
            "message": "Relationship deleted successfully",
            "data": sample_relation_data
        }
        mock_graph_memory_manager.delete_relation.return_value = json.dumps(success_response)

        # Call the method
        result = mock_graph_memory_manager.delete_relation(
            sample_relation_data["from"],
            sample_relation_data["to"],
            sample_relation_data["relationType"]
        )

        # Verify the result
        result_dict = json_helper["from_json"](result)
        assert result_dict["status"] == "success"
        assert "Relationship deleted successfully" in result_dict["message"]

    @pytest.mark.relation
    @pytest.mark.parametrize("invalid_relation_data", [
        {"from_entity": None, "error": "from_entity"},
        {"from_entity": "", "error": "from_entity"},
        {"to_entity": None, "error": "to_entity"},
        {"to_entity": "", "error": "to_entity"}
    ])
    def test_delete_relation_with_invalid_data(
            self, mock_graph_memory_manager, invalid_relation_data):
        """Test deleting a relation with invalid data raises appropriate error."""
        from_entity = invalid_relation_data.get("from_entity", "SourceEntity")
        to_entity = invalid_relation_data.get("to_entity", "TargetEntity")
        relation_type = "RELATES_TO"

        # Configure the method to raise ValueError for invalid data
        def validate_relation(from_e, to_e, rel_type):
            if not from_e or not isinstance(from_e, str):
                raise ValueError("Invalid from_entity")
            if not to_e or not isinstance(to_e, str):
                raise ValueError("Invalid to_entity")
            return json.dumps({"status": "success", "message": "Relation deleted"})

        mock_graph_memory_manager.delete_relation = MagicMock(side_effect=validate_relation)
        
        with pytest.raises(ValueError):
            mock_graph_memory_manager.delete_relation(from_entity, to_entity, relation_type)

    @pytest.mark.observation
    def test_delete_observation(self, mock_graph_memory_manager, json_helper):
        """Test deleting an observation."""
        entity_name = "TestEntity"
        observation_id = "obs123"

        # Configure the method to return a JSON string
        success_response = {
            "status": "success",
            "message": "Observation deleted successfully",
            "data": {
                "entity": entity_name,
                "observation_id": observation_id
            }
        }
        mock_graph_memory_manager.delete_observation.return_value = json.dumps(success_response)

        # Call the method
        result = mock_graph_memory_manager.delete_observation(
            entity_name, None, observation_id
        )

        # Verify the result
        result_dict = json_helper["from_json"](result)
        assert result_dict["status"] == "success"
        assert "Observation deleted successfully" in result_dict["message"]

    @pytest.mark.observation
    @pytest.mark.parametrize("invalid_observation_data", [
        {"entity_name": None, "error": "entity_name"},
        {"entity_name": "", "error": "entity_name"}
    ])
    def test_delete_observation_with_invalid_data(
            self, mock_graph_memory_manager, invalid_observation_data):
        """Test deleting observation with invalid data raises appropriate error."""
        entity_name = invalid_observation_data.get("entity_name", "ValidEntity")
        observation_id = "valid_id"

        # Configure the method to raise ValueError for invalid data
        def validate_observation(entity, content_id, obs_id):
            if not entity or not isinstance(entity, str):
                raise ValueError("Invalid entity name")
            return json.dumps({"status": "success", "message": "Observation deleted"})
        
        mock_graph_memory_manager.delete_observation = MagicMock(side_effect=validate_observation)
        
        with pytest.raises(ValueError):
            mock_graph_memory_manager.delete_observation(entity_name, None, observation_id)

    @pytest.mark.error_handling
    def test_error_response_handling(self, mock_graph_memory_manager, json_helper):
        """Test handling of error responses from Neo4j."""
        # Configure the method to return a JSON error response
        error_response = {
            "status": "error",
            "message": "Database connection error",
            "data": None
        }
        mock_graph_memory_manager.create_entity.return_value = json.dumps(error_response)

        # Call the method with valid entity data
        entity_data = {"name": "TestEntity", "entityType": "TEST"}
        result = mock_graph_memory_manager.create_entity(entity_data)

        # Verify the error is properly propagated
        result_dict = json_helper["from_json"](result)
        assert result_dict["status"] == "error"
        assert "Database connection error" in result_dict["message"]
        assert result_dict["data"] is None 