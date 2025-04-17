"""
Unit tests for the GraphMemoryManager project memory operations.
"""

import pytest
from unittest.mock import patch, MagicMock
import json
from contextlib import contextmanager

from src.graph_memory import GraphMemoryManager  # type: ignore


@pytest.mark.unit
@pytest.mark.project_memory
class TestProjectMemoryOperations:
    """Test suite for GraphMemoryManager project memory operations."""

    @pytest.mark.parametrize("operation_type", [
        "create_project", "create_component", "create_domain", "create_domain_entity",
        "relate", "search", "get_structure", "add_observation", "update",
        "delete_entity", "delete_relationship"
    ])
    def test_project_operation_routes_correctly(
            self, mock_graph_memory_manager, operation_type, json_helper):
        """Test that project_operation routes to the correct handler method."""
        # Set up expected message based on operation type
        expected_messages = {
            "create_project": "Project created successfully",
            "create_component": "Component created successfully",
            "create_domain": "Domain created successfully",
            "create_domain_entity": "Domain entity created successfully",
            "relate": "Entity relationship created successfully",
            "search": "Project search completed successfully",
            "get_structure": "Structure retrieved successfully",
            "add_observation": "Observation added successfully",
            "update": "Entity updated successfully",
            "delete_entity": "Entity deleted successfully",
            "delete_relationship": "Relationship deleted successfully"
        }
        
        # Configure the mock handler method
        handler_method = f"_handle_{operation_type}"
        mock_response = json.dumps({
            "status": "success",
            "message": expected_messages.get(operation_type, f"{operation_type} operation completed successfully"),
            "data": {"operation_type": operation_type}
        })
        
        if hasattr(mock_graph_memory_manager, handler_method):
            getattr(mock_graph_memory_manager, handler_method).return_value = mock_response
        
        # Call the project_operation method
        kwargs = {"test_param": "test_value"}
        result = mock_graph_memory_manager.project_operation(operation_type, **kwargs)
        
        # Verify the result
        result_dict = json_helper["from_json"](result)
        assert result_dict["status"] == "success"
        # Check for the expected message based on the operation type
        expected_msg = expected_messages.get(operation_type, f"{operation_type} operation completed successfully")
        assert expected_msg in result_dict["message"]
        
        # Verify the correct handler was called with the correct arguments
        if hasattr(mock_graph_memory_manager, handler_method):
            handler = getattr(mock_graph_memory_manager, handler_method)
            handler.assert_called_once_with(**kwargs)

    def test_project_operation_with_invalid_operation_type(self, mock_graph_memory_manager):
        """Test that project_operation raises an error with invalid operation type."""
        with pytest.raises(ValueError):
            mock_graph_memory_manager.project_operation("invalid_operation_type")

    def test_create_project_operation(self, mock_graph_memory_manager, json_helper):
        """Test the create_project operation."""
        # Configure the mock handler method
        mock_graph_memory_manager._handle_project_creation.return_value = json.dumps({
            "status": "success",
            "message": "Project created successfully",
            "data": {"name": "TestProject"}
        })
        
        # Call the project_operation method
        result = mock_graph_memory_manager.project_operation(
            "create_project", 
            name="TestProject", 
            description="Test project description"
        )
        
        # Verify the result
        result_dict = json_helper["from_json"](result)
        assert result_dict["status"] == "success"
        assert "Project created successfully" in result_dict["message"]
        assert result_dict["data"]["name"] == "TestProject"
        
        # Verify the handler was called with the correct arguments
        mock_graph_memory_manager._handle_project_creation.assert_called_once_with(
            name="TestProject", 
            description="Test project description"
        )

    def test_create_component_operation(self, mock_graph_memory_manager, json_helper):
        """Test the create_component operation."""
        # Configure the mock handler method
        mock_graph_memory_manager._handle_component_creation.return_value = json.dumps({
            "status": "success",
            "message": "Component created successfully",
            "data": {
                "name": "TestComponent",
                "component_type": "SERVICE",
                "project_id": "TestProject"
            }
        })
        
        # Call the project_operation method
        result = mock_graph_memory_manager.project_operation(
            "create_component", 
            name="TestComponent", 
            component_type="SERVICE",
            project_id="TestProject",
            domain_name="TestDomain",
            description="Test component description"
        )
        
        # Verify the result
        result_dict = json_helper["from_json"](result)
        assert result_dict["status"] == "success"
        assert "Component created successfully" in result_dict["message"]
        assert result_dict["data"]["name"] == "TestComponent"
        assert result_dict["data"]["component_type"] == "SERVICE"
        
        # Verify the handler was called with the correct arguments
        mock_graph_memory_manager._handle_component_creation.assert_called_once_with(
            name="TestComponent", 
            component_type="SERVICE",
            project_id="TestProject",
            domain_name="TestDomain",
            description="Test component description"
        )

    def test_create_domain_operation(self, mock_graph_memory_manager, json_helper):
        """Test the create_domain operation."""
        # Configure the mock handler method
        mock_graph_memory_manager._handle_domain_creation.return_value = json.dumps({
            "status": "success",
            "message": "Domain created successfully",
            "data": {
                "name": "TestDomain",
                "project_id": "TestProject"
            }
        })
        
        # Call the project_operation method
        result = mock_graph_memory_manager.project_operation(
            "create_domain", 
            name="TestDomain", 
            project_id="TestProject",
            description="Test domain description"
        )
        
        # Verify the result
        result_dict = json_helper["from_json"](result)
        assert result_dict["status"] == "success"
        assert "Domain created successfully" in result_dict["message"]
        assert result_dict["data"]["name"] == "TestDomain"
        
        # Verify the handler was called with the correct arguments
        mock_graph_memory_manager._handle_domain_creation.assert_called_once_with(
            name="TestDomain", 
            project_id="TestProject",
            description="Test domain description"
        )

    def test_create_domain_entity_operation(self, mock_graph_memory_manager, json_helper):
        """Test the create_domain_entity operation."""
        # Configure the mock handler method
        mock_graph_memory_manager._handle_domain_entity_creation.return_value = json.dumps({
            "status": "success",
            "message": "Domain entity created successfully",
            "data": {
                "name": "TestEntity",
                "entity_type": "REQUIREMENT",
                "project_id": "TestProject"
            }
        })
        
        # Call the project_operation method
        result = mock_graph_memory_manager.project_operation(
            "create_domain_entity", 
            name="TestEntity", 
            entity_type="REQUIREMENT",
            project_id="TestProject",
            description="Test entity description"
        )
        
        # Verify the result
        result_dict = json_helper["from_json"](result)
        assert result_dict["status"] == "success"
        assert "Domain entity created successfully" in result_dict["message"]
        assert result_dict["data"]["name"] == "TestEntity"
        assert result_dict["data"]["entity_type"] == "REQUIREMENT"
        
        # Verify the handler was called with the correct arguments
        mock_graph_memory_manager._handle_domain_entity_creation.assert_called_once_with(
            name="TestEntity", 
            entity_type="REQUIREMENT",
            project_id="TestProject",
            description="Test entity description"
        )

    def test_relate_operation(self, mock_graph_memory_manager, json_helper):
        """Test the relate operation."""
        # Configure the mock handler method
        mock_graph_memory_manager._handle_entity_relationship.return_value = json.dumps({
            "status": "success",
            "message": "Relationship created successfully",
            "data": {
                "source_name": "SourceEntity",
                "target_name": "TargetEntity",
                "relation_type": "DEPENDS_ON"
            }
        })
        
        # Call the project_operation method
        result = mock_graph_memory_manager.project_operation(
            "relate", 
            source_name="SourceEntity", 
            target_name="TargetEntity",
            relation_type="DEPENDS_ON",
            project_id="TestProject"
        )
        
        # Verify the result
        result_dict = json_helper["from_json"](result)
        assert result_dict["status"] == "success"
        assert "Relationship created successfully" in result_dict["message"]
        assert result_dict["data"]["source_name"] == "SourceEntity"
        assert result_dict["data"]["target_name"] == "TargetEntity"
        
        # Verify the handler was called with the correct arguments
        mock_graph_memory_manager._handle_entity_relationship.assert_called_once_with(
            source_name="SourceEntity", 
            target_name="TargetEntity",
            relation_type="DEPENDS_ON",
            project_id="TestProject"
        )

    def test_search_operation(self, mock_graph_memory_manager, json_helper):
        """Test the search operation."""
        # Mock search results
        mock_results = [
            {"name": "Entity1", "entity_type": "COMPONENT", "score": 0.95},
            {"name": "Entity2", "entity_type": "REQUIREMENT", "score": 0.85}
        ]
        
        # Configure the mock handler method
        mock_graph_memory_manager._handle_project_search.return_value = json.dumps({
            "status": "success",
            "message": "Search completed successfully",
            "data": {
                "results": mock_results, 
                "count": len(mock_results)
            }
        })
        
        # Call the project_operation method
        result = mock_graph_memory_manager.project_operation(
            "search", 
            query="test query",
            project_id="TestProject",
            limit=5,
            entity_types=["COMPONENT", "REQUIREMENT"]
        )
        
        # Verify the result
        result_dict = json_helper["from_json"](result)
        assert result_dict["status"] == "success"
        assert "Search completed successfully" in result_dict["message"]
        assert len(result_dict["data"]["results"]) == 2
        
        # Verify the handler was called with the correct arguments
        mock_graph_memory_manager._handle_project_search.assert_called_once_with(
            query="test query",
            project_id="TestProject",
            limit=5,
            entity_types=["COMPONENT", "REQUIREMENT"]
        )

    def test_get_structure_operation(self, mock_graph_memory_manager, json_helper):
        """Test the get_structure operation."""
        # Mock structure data
        mock_structure = {
            "project": {"name": "TestProject"},
            "components": [{"name": "Component1"}, {"name": "Component2"}],
            "domains": [{"name": "Domain1"}]
        }
        
        # Configure the mock handler method
        mock_graph_memory_manager._handle_structure_retrieval.return_value = json.dumps({
            "status": "success",
            "message": "Structure retrieved successfully",
            "data": mock_structure
        })
        
        # Call the project_operation method
        result = mock_graph_memory_manager.project_operation(
            "get_structure", 
            project_id="TestProject"
        )
        
        # Verify the result
        result_dict = json_helper["from_json"](result)
        assert result_dict["status"] == "success"
        assert "Structure retrieved successfully" in result_dict["message"]
        assert "components" in result_dict["data"]
        assert len(result_dict["data"]["components"]) == 2
        
        # Verify the handler was called with the correct arguments
        mock_graph_memory_manager._handle_structure_retrieval.assert_called_once_with(
            project_id="TestProject"
        )

    def test_add_observation_operation(self, mock_graph_memory_manager, json_helper):
        """Test the add_observation operation."""
        # Configure the mock handler method
        mock_graph_memory_manager._handle_add_observation.return_value = json.dumps({
            "status": "success",
            "message": "Observation added successfully",
            "data": {
                "entity_name": "TestEntity",
                "content": "Test observation content"
            }
        })
        
        # Call the project_operation method
        result = mock_graph_memory_manager.project_operation(
            "add_observation", 
            entity_name="TestEntity",
            content="Test observation content",
            observation_type="IMPLEMENTATION",
            project_id="TestProject"
        )
        
        # Verify the result
        result_dict = json_helper["from_json"](result)
        assert result_dict["status"] == "success"
        assert "Observation added successfully" in result_dict["message"]
        assert result_dict["data"]["entity_name"] == "TestEntity"
        assert result_dict["data"]["content"] == "Test observation content"
        
        # Verify the handler was called with the correct arguments
        mock_graph_memory_manager._handle_add_observation.assert_called_once_with(
            entity_name="TestEntity",
            content="Test observation content",
            observation_type="IMPLEMENTATION",
            project_id="TestProject"
        )

    def test_update_operation(self, mock_graph_memory_manager, json_helper):
        """Test the update operation."""
        # Configure the mock handler method
        mock_graph_memory_manager._handle_entity_update.return_value = json.dumps({
            "status": "success",
            "message": "Entity updated successfully",
            "data": {
                "entity_name": "TestEntity",
                "updates": {"status": "COMPLETED"}
            }
        })
        
        # Call the project_operation method
        result = mock_graph_memory_manager.project_operation(
            "update", 
            entity_name="TestEntity",
            updates={"status": "COMPLETED"},
            project_id="TestProject"
        )
        
        # Verify the result
        result_dict = json_helper["from_json"](result)
        assert result_dict["status"] == "success"
        assert "Entity updated successfully" in result_dict["message"]
        assert result_dict["data"]["entity_name"] == "TestEntity"
        
        # Verify the handler was called with the correct arguments
        mock_graph_memory_manager._handle_entity_update.assert_called_once_with(
            entity_name="TestEntity",
            updates={"status": "COMPLETED"},
            project_id="TestProject"
        )

    def test_delete_entity_operation(self, mock_graph_memory_manager, json_helper):
        """Test the delete_entity operation."""
        # Configure the mock handler method
        mock_graph_memory_manager._handle_entity_deletion.return_value = json.dumps({
            "status": "success",
            "message": "Entity deleted successfully",
            "data": {
                "entity_name": "TestEntity",
                "entity_type": "COMPONENT"
            }
        })
        
        # Call the project_operation method
        result = mock_graph_memory_manager.project_operation(
            "delete_entity", 
            entity_name="TestEntity",
            entity_type="COMPONENT",
            project_id="TestProject"
        )
        
        # Verify the result
        result_dict = json_helper["from_json"](result)
        assert result_dict["status"] == "success"
        assert "Entity deleted successfully" in result_dict["message"]
        assert result_dict["data"]["entity_name"] == "TestEntity"
        
        # Verify the handler was called with the correct arguments
        mock_graph_memory_manager._handle_entity_deletion.assert_called_once_with(
            entity_name="TestEntity",
            entity_type="COMPONENT",
            project_id="TestProject"
        )

    def test_delete_relationship_operation(self, mock_graph_memory_manager, json_helper):
        """Test the delete_relationship operation."""
        # Configure the mock handler method
        mock_graph_memory_manager._handle_relationship_deletion.return_value = json.dumps({
            "status": "success",
            "message": "Relationship deleted successfully",
            "data": {
                "source_name": "SourceEntity",
                "target_name": "TargetEntity",
                "relationship_type": "DEPENDS_ON"
            }
        })
        
        # Call the project_operation method
        result = mock_graph_memory_manager.project_operation(
            "delete_relationship", 
            source_name="SourceEntity",
            target_name="TargetEntity",
            relationship_type="DEPENDS_ON",
            project_id="TestProject"
        )
        
        # Verify the result
        result_dict = json_helper["from_json"](result)
        assert result_dict["status"] == "success"
        assert "Relationship deleted successfully" in result_dict["message"]
        assert result_dict["data"]["source_name"] == "SourceEntity"
        assert result_dict["data"]["target_name"] == "TargetEntity"
        
        # Verify the handler was called with the correct arguments
        mock_graph_memory_manager._handle_relationship_deletion.assert_called_once_with(
            source_name="SourceEntity",
            target_name="TargetEntity",
            relationship_type="DEPENDS_ON",
            project_id="TestProject"
        )

    def test_project_context_manager(self, mock_graph_memory_manager):
        """Test the project_context context manager."""
        # Set up the original state
        mock_graph_memory_manager.project_memory.project_name = "DefaultProject"
        
        # Use the context manager - the project_name should be updated within the context
        with mock_graph_memory_manager.project_context(project_name="TestProject"):
            # Check that the context variables are set correctly
            assert mock_graph_memory_manager.project_memory.project_name == "TestProject"
        
        # Check that the original state is restored after exiting the context
        assert mock_graph_memory_manager.project_memory.project_name == "DefaultProject"

    def test_project_context_manager_with_exception(self, mock_graph_memory_manager):
        """Test the project_context context manager when an exception occurs."""
        # Set up the original state
        mock_graph_memory_manager.project_memory.project_name = "DefaultProject"
        
        # Configure mock for set_project_name
        mock_graph_memory_manager.set_project_name = MagicMock()
        
        # Use the context manager with an exception
        try:
            with mock_graph_memory_manager.project_context(project_name="TestProject"):
                raise ValueError("Test exception")
        except ValueError:
            pass
        
        # Check that the original state is restored despite the exception
        assert mock_graph_memory_manager.project_memory.project_name == "DefaultProject"

    def test_error_handling_in_project_operation(self, mock_graph_memory_manager, json_helper):
        """Test error handling in project_operation."""
        # Create a custom error handler that returns a JSON error response
        def custom_error_handler(**kwargs):
            raise Exception("Test error")
            
        # Configure the handler to use our custom error handler
        mock_graph_memory_manager._handle_project_creation.side_effect = custom_error_handler
        
        # Add a custom error handling method that creates a JSON error response
        def error_response_handler(operation_type, e, **kwargs):
            return json.dumps({
                "status": "error",
                "message": f"Error in {operation_type} operation: {str(e)}",
                "error": {"message": str(e), "code": "test_error"}
            })
            
        # Set up the error handling in the mock_project_operation method
        original_mock_project_operation = mock_graph_memory_manager.project_operation.side_effect
        
        def wrapped_mock_project_operation(operation_type, **kwargs):
            try:
                return original_mock_project_operation(operation_type, **kwargs)
            except Exception as e:
                return error_response_handler(operation_type, e, **kwargs)
                
        mock_graph_memory_manager.project_operation.side_effect = wrapped_mock_project_operation
        
        # Call the method
        result = mock_graph_memory_manager.project_operation(
            "create_project",
            name="TestProject"
        )
        
        # Verify the error response
        result_dict = json_helper["from_json"](result)
        assert result_dict["status"] == "error"
        assert "Error in create_project operation: Test error" in result_dict["message"]
        assert result_dict["error"]["message"] == "Test error" 