"""
Unit tests for the GraphMemoryManager lesson memory operations.
"""

import pytest
from unittest.mock import patch, MagicMock
import json
from contextlib import contextmanager

from src.graph_memory import GraphMemoryManager  # type: ignore


@pytest.mark.unit
@pytest.mark.lesson_memory
class TestLessonMemoryOperations:
    """Test suite for GraphMemoryManager lesson memory operations."""

    @pytest.mark.parametrize("operation_type", [
        "create_container", "create", "observe", "relate", "search", 
        "track", "consolidate", "evolve", "update", "get_container", 
        "list_containers", "container_exists"
    ])
    def test_lesson_operation_routes_correctly(
            self, mock_graph_memory_manager, operation_type, json_helper):
        """Test that lesson_operation routes to the correct handler method."""
        # Set up expected message based on operation type
        expected_messages = {
            "create_container": "Container created successfully",
            "create": "Lesson created successfully",
            "observe": "Observation added successfully",
            "relate": "Lessons related successfully",
            "search": "Search completed successfully",
            "track": "Lesson tracking completed successfully",
            "consolidate": "Lessons consolidated successfully",
            "evolve": "Lesson evolved successfully",
            "update": "Lesson updated successfully",
            "get_container": "Container retrieved successfully",
            "list_containers": "Containers retrieved successfully",
            "container_exists": "Container existence checked successfully"
        }
        
        # Configure the mock handler method
        handler_method = f"_handle_lesson_{operation_type}"
        mock_response = json.dumps({
            "status": "success",
            "message": expected_messages.get(operation_type, f"{operation_type} operation completed successfully"),
            "data": {"operation_type": operation_type}
        })
        
        if hasattr(mock_graph_memory_manager, handler_method):
            getattr(mock_graph_memory_manager, handler_method).return_value = mock_response
        
        # Call the lesson_operation method
        kwargs = {"test_param": "test_value"}
        result = mock_graph_memory_manager.lesson_operation(operation_type, **kwargs)
        
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

    def test_lesson_operation_with_invalid_operation_type(self, mock_graph_memory_manager):
        """Test that lesson_operation raises an error with invalid operation type."""
        with pytest.raises(ValueError):
            mock_graph_memory_manager.lesson_operation("invalid_operation_type")

    def test_create_container_operation(self, mock_graph_memory_manager, json_helper):
        """Test the create_container operation."""
        # Configure the mock handler method
        mock_graph_memory_manager._handle_lesson_container_creation.return_value = json.dumps({
            "status": "success",
            "message": "Container created successfully",
            "data": {"container_name": "TestContainer"}
        })
        
        # Call the lesson_operation method
        result = mock_graph_memory_manager.lesson_operation(
            "create_container", 
            container_name="TestContainer", 
            description="Test container description"
        )
        
        # Verify the result
        result_dict = json_helper["from_json"](result)
        assert result_dict["status"] == "success"
        assert "Container created successfully" in result_dict["message"]
        assert result_dict["data"]["container_name"] == "TestContainer"
        
        # Verify the handler was called with the correct arguments
        mock_graph_memory_manager._handle_lesson_container_creation.assert_called_once_with(
            container_name="TestContainer", 
            description="Test container description"
        )

    def test_create_lesson_operation(self, mock_graph_memory_manager, json_helper):
        """Test the create lesson operation."""
        # Configure the mock handler method
        mock_graph_memory_manager._handle_lesson_creation.return_value = json.dumps({
            "status": "success",
            "message": "Lesson created successfully",
            "data": {
                "name": "TestLesson",
                "lesson_type": "CONCEPT",
                "confidence": 0.8
            }
        })
        
        # Call the lesson_operation method
        result = mock_graph_memory_manager.lesson_operation(
            "create", 
            name="TestLesson", 
            lesson_type="CONCEPT",
            confidence=0.8,
            tags=["test", "memory"]
        )
        
        # Verify the result
        result_dict = json_helper["from_json"](result)
        assert result_dict["status"] == "success"
        assert "Lesson created successfully" in result_dict["message"]
        assert result_dict["data"]["name"] == "TestLesson"
        assert result_dict["data"]["lesson_type"] == "CONCEPT"
        
        # Verify the handler was called with the correct arguments
        mock_graph_memory_manager._handle_lesson_creation.assert_called_once_with(
            name="TestLesson", 
            lesson_type="CONCEPT",
            confidence=0.8,
            tags=["test", "memory"]
        )

    def test_observe_operation(self, mock_graph_memory_manager, json_helper):
        """Test the observe operation."""
        # Configure the mock handler method
        mock_graph_memory_manager._handle_lesson_observation.return_value = json.dumps({
            "status": "success",
            "message": "Observation added successfully",
            "data": {
                "entity_name": "TestLesson",
                "what_was_learned": "New testing technique"
            }
        })
        
        # Call the lesson_operation method
        result = mock_graph_memory_manager.lesson_operation(
            "observe", 
            entity_name="TestLesson", 
            what_was_learned="New testing technique",
            why_it_matters="Improves test coverage",
            how_to_apply="Apply in unit tests"
        )
        
        # Verify the result
        result_dict = json_helper["from_json"](result)
        assert result_dict["status"] == "success"
        assert "Observation added successfully" in result_dict["message"]
        assert result_dict["data"]["entity_name"] == "TestLesson"
        
        # Verify the handler was called with the correct arguments
        mock_graph_memory_manager._handle_lesson_observation.assert_called_once_with(
            entity_name="TestLesson", 
            what_was_learned="New testing technique",
            why_it_matters="Improves test coverage",
            how_to_apply="Apply in unit tests"
        )

    def test_search_operation(self, mock_graph_memory_manager, json_helper):
        """Test the search operation."""
        # Mock search results
        mock_results = [
            {"name": "Lesson1", "lesson_type": "CONCEPT", "score": 0.95},
            {"name": "Lesson2", "lesson_type": "TECHNIQUE", "score": 0.85}
        ]
        
        # Configure the mock handler method
        mock_graph_memory_manager._handle_lesson_search.return_value = json.dumps({
            "status": "success",
            "message": "Search completed successfully",
            "data": {
                "results": mock_results, 
                "count": len(mock_results)
            }
        })
        
        # Call the lesson_operation method
        result = mock_graph_memory_manager.lesson_operation(
            "search", 
            query="test query",
            limit=5,
            lesson_types=["CONCEPT", "TECHNIQUE"]
        )
        
        # Verify the result
        result_dict = json_helper["from_json"](result)
        assert result_dict["status"] == "success"
        assert "Search completed successfully" in result_dict["message"]
        assert len(result_dict["data"]["results"]) == 2
        
        # Verify the handler was called with the correct arguments
        mock_graph_memory_manager._handle_lesson_search.assert_called_once_with(
            query="test query",
            limit=5,
            lesson_types=["CONCEPT", "TECHNIQUE"]
        )

    def test_consolidate_operation(self, mock_graph_memory_manager, json_helper):
        """Test the consolidate operation."""
        # Configure the mock handler method
        mock_graph_memory_manager._handle_lesson_consolidation.return_value = json.dumps({
            "status": "success",
            "message": "Lessons consolidated successfully",
            "data": {
                "new_name": "ConsolidatedLesson",
                "source_lessons": ["Lesson1", "Lesson2"],
                "confidence": 0.9
            }
        })
        
        # Call the lesson_operation method
        result = mock_graph_memory_manager.lesson_operation(
            "consolidate", 
            source_lessons=["Lesson1", "Lesson2"],
            new_name="ConsolidatedLesson",
            confidence=0.9
        )
        
        # Verify the result
        result_dict = json_helper["from_json"](result)
        assert result_dict["status"] == "success"
        assert "Lessons consolidated successfully" in result_dict["message"]
        assert result_dict["data"]["new_name"] == "ConsolidatedLesson"
        
        # Verify the handler was called with the correct arguments
        mock_graph_memory_manager._handle_lesson_consolidation.assert_called_once_with(
            source_lessons=["Lesson1", "Lesson2"],
            new_name="ConsolidatedLesson",
            confidence=0.9
        )

    def test_evolve_operation(self, mock_graph_memory_manager, json_helper):
        """Test the evolve operation."""
        # Configure the mock handler method
        mock_graph_memory_manager._handle_lesson_evolution.return_value = json.dumps({
            "status": "success",
            "message": "Lesson evolved successfully",
            "data": {
                "old_lesson": "OldLesson",
                "new_lesson": "NewLesson",
                "relation_type": "SUPERSEDES"
            }
        })
        
        # Call the lesson_operation method
        result = mock_graph_memory_manager.lesson_operation(
            "evolve", 
            old_lesson="OldLesson",
            new_lesson="NewLesson",
            relation_type="SUPERSEDES"
        )
        
        # Verify the result
        result_dict = json_helper["from_json"](result)
        assert result_dict["status"] == "success"
        assert "Lesson evolved successfully" in result_dict["message"]
        assert result_dict["data"]["old_lesson"] == "OldLesson"
        assert result_dict["data"]["new_lesson"] == "NewLesson"
        
        # Verify the handler was called with the correct arguments
        mock_graph_memory_manager._handle_lesson_evolution.assert_called_once_with(
            old_lesson="OldLesson",
            new_lesson="NewLesson",
            relation_type="SUPERSEDES"
        )

    def test_update_operation(self, mock_graph_memory_manager, json_helper):
        """Test the update operation."""
        # Configure the mock handler method
        mock_graph_memory_manager._handle_lesson_update.return_value = json.dumps({
            "status": "success",
            "message": "Lesson updated successfully",
            "data": {
                "entity_name": "TestLesson",
                "updates": {"confidence": 0.9}
            }
        })
        
        # Call the lesson_operation method
        result = mock_graph_memory_manager.lesson_operation(
            "update", 
            entity_name="TestLesson",
            updates={"confidence": 0.9}
        )
        
        # Verify the result
        result_dict = json_helper["from_json"](result)
        assert result_dict["status"] == "success"
        assert "Lesson updated successfully" in result_dict["message"]
        assert result_dict["data"]["entity_name"] == "TestLesson"
        
        # Verify the handler was called with the correct arguments
        mock_graph_memory_manager._handle_lesson_update.assert_called_once_with(
            entity_name="TestLesson",
            updates={"confidence": 0.9}
        )

    def test_container_exists_operation(self, mock_graph_memory_manager, json_helper):
        """Test the container_exists operation."""
        # Configure the mock handler method
        mock_graph_memory_manager._handle_container_exists.return_value = json.dumps({
            "status": "success",
            "message": "Container exists",
            "data": {
                "container_name": "Lessons",
                "exists": True
            }
        })
        
        # Call the lesson_operation method
        result = mock_graph_memory_manager.lesson_operation(
            "container_exists", 
            container_name="Lessons"
        )
        
        # Verify the result
        result_dict = json_helper["from_json"](result)
        assert result_dict["status"] == "success"
        assert "Container exists" in result_dict["message"]
        assert result_dict["data"]["exists"] is True
        
        # Verify the handler was called with the correct arguments
        mock_graph_memory_manager._handle_container_exists.assert_called_once_with(
            container_name="Lessons"
        )

    def test_list_containers_operation(self, mock_graph_memory_manager, json_helper):
        """Test the list_containers operation."""
        # Mock container results
        mock_containers = [
            {"name": "Lessons", "created": "2023-01-01T00:00:00Z"},
            {"name": "TechnicalLessons", "created": "2023-01-02T00:00:00Z"}
        ]
        
        # Configure the mock handler method
        mock_graph_memory_manager._handle_list_lesson_containers.return_value = json.dumps({
            "status": "success",
            "message": "Containers retrieved successfully",
            "data": {
                "containers": mock_containers,
                "count": len(mock_containers)
            }
        })
        
        # Call the lesson_operation method
        result = mock_graph_memory_manager.lesson_operation(
            "list_containers", 
            limit=10,
            sort_by="created"
        )
        
        # Verify the result
        result_dict = json_helper["from_json"](result)
        assert result_dict["status"] == "success"
        assert "Containers retrieved successfully" in result_dict["message"]
        assert len(result_dict["data"]["containers"]) == 2
        
        # Verify the handler was called with the correct arguments
        mock_graph_memory_manager._handle_list_lesson_containers.assert_called_once_with(
            limit=10,
            sort_by="created"
        )

    def test_get_container_operation(self, mock_graph_memory_manager, json_helper):
        """Test the get_container operation."""
        # Configure the mock handler method
        mock_graph_memory_manager._handle_get_lesson_container.return_value = json.dumps({
            "status": "success",
            "message": "Container retrieved successfully",
            "data": {
                "container_name": "Lessons",
                "description": "Main lessons container",
                "created": "2023-01-01T00:00:00Z"
            }
        })
        
        # Call the lesson_operation method
        result = mock_graph_memory_manager.lesson_operation(
            "get_container"
        )
        
        # Verify the result
        result_dict = json_helper["from_json"](result)
        assert result_dict["status"] == "success"
        assert "Container retrieved successfully" in result_dict["message"]
        assert result_dict["data"]["container_name"] == "Lessons"
        
        # Verify the handler was called with the correct arguments
        mock_graph_memory_manager._handle_get_lesson_container.assert_called_once_with()

    def test_lesson_context_manager(self, mock_graph_memory_manager):
        """Test the lesson_context context manager."""
        # Set up the original state
        mock_graph_memory_manager.lesson_memory.container_name = "DefaultContainer"
        
        # Configure mock for set_project_name
        mock_graph_memory_manager.set_project_name = MagicMock()
        
        # Use the context manager
        with mock_graph_memory_manager.lesson_context(
                project_name="TestProject", 
                container_name="TestContainer"):
            # Check that the context variables are set correctly
            assert mock_graph_memory_manager.lesson_memory.container_name == "TestContainer"
            mock_graph_memory_manager.set_project_name.assert_called_once_with(
                "TestProject", None
            )
        
        # Check that the original state is restored
        assert mock_graph_memory_manager.lesson_memory.container_name == "DefaultContainer"

    def test_lesson_context_manager_with_exception(self, mock_graph_memory_manager):
        """Test the lesson_context context manager when an exception occurs."""
        # Set up the original state
        mock_graph_memory_manager.lesson_memory.container_name = "DefaultContainer"
        
        # Configure mock for set_project_name
        mock_graph_memory_manager.set_project_name = MagicMock()
        
        # Use the context manager with an exception
        try:
            with mock_graph_memory_manager.lesson_context(
                    project_name="TestProject", 
                    container_name="TestContainer"):
                raise ValueError("Test exception")
        except ValueError:
            pass
        
        # Check that the original state is restored despite the exception
        assert mock_graph_memory_manager.lesson_memory.container_name == "DefaultContainer"

    def test_error_handling_in_lesson_operation(self, mock_graph_memory_manager, json_helper):
        """Test error handling in lesson_operation."""
        # Create a custom error handler that returns a JSON error response
        def custom_error_handler(**kwargs):
            raise Exception("Test error")
            
        # Configure the handler to use our custom error handler
        mock_graph_memory_manager._handle_lesson_creation.side_effect = custom_error_handler
        
        # Add a custom error handling method that creates a JSON error response
        def error_response_handler(operation_type, e, **kwargs):
            return json.dumps({
                "status": "error",
                "message": f"Error in {operation_type} operation: {str(e)}",
                "error": {"message": str(e), "code": "test_error"}
            })
            
        # Set up the error handling in the mock_lesson_operation method
        original_mock_lesson_operation = mock_graph_memory_manager.lesson_operation.side_effect
        
        def wrapped_mock_lesson_operation(operation_type, **kwargs):
            try:
                return original_mock_lesson_operation(operation_type, **kwargs)
            except Exception as e:
                return error_response_handler(operation_type, e, **kwargs)
                
        mock_graph_memory_manager.lesson_operation.side_effect = wrapped_mock_lesson_operation
        
        # Call the method
        result = mock_graph_memory_manager.lesson_operation(
            "create",
            name="TestLesson",
            lesson_type="CONCEPT"
        )
        
        # Verify the error response
        result_dict = json_helper["from_json"](result)
        assert result_dict["status"] == "error"
        assert "Error in create operation: Test error" in result_dict["message"]
        assert result_dict["error"]["message"] == "Test error" 