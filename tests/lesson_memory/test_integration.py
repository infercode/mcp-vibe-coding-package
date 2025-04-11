"""
Integration tests for the Lesson Memory integration with GraphMemoryManager.

These tests verify that the layered access approach (Operation Categories and
Context Management) works correctly with the underlying LessonMemoryManager.
"""

import json
import pytest
from unittest.mock import MagicMock, patch

from src.graph_memory import GraphMemoryManager


@pytest.fixture
def graph_manager():
    """Create a graph memory manager with mocked lesson memory."""
    with patch.object(GraphMemoryManager, 'initialize', return_value=True):
        manager = GraphMemoryManager()
        
    # Mock the LessonMemoryManager
    manager.lesson_memory = MagicMock()
    
    # Set a dummy default project name
    manager.default_project_name = "TestProject"
    return manager


def test_lesson_operation_create(graph_manager):
    """Test that the lesson_operation method correctly dispatches create operations."""
    # Mock the expected return value
    mock_result = {"status": "success", "entity": {"name": "TestLesson"}}
    graph_manager.lesson_memory.create_lesson_entity.return_value = json.dumps(mock_result)
    
    # Call the operation dispatcher
    result = graph_manager.lesson_operation(
        "create",
        name="TestLesson",
        lesson_type="BestPractice",
        container_name="Lessons"
    )
    
    # Verify the correct method was called with right parameters
    graph_manager.lesson_memory.create_lesson_entity.assert_called_once_with(
        "Lessons", "TestLesson", "BestPractice", None, None
    )
    
    # Verify the result was returned correctly
    assert result == json.dumps(mock_result)


def test_lesson_operation_observe(graph_manager):
    """Test that the lesson_operation method correctly dispatches observe operations."""
    # Mock the expected return value
    mock_result = {"status": "success", "observations": [{"content": "Test content"}]}
    graph_manager.lesson_memory.create_structured_lesson_observations.return_value = json.dumps(mock_result)
    
    # Call the operation dispatcher
    result = graph_manager.lesson_operation(
        "observe",
        entity_name="TestLesson",
        what_was_learned="Test content",
        container_name="Lessons"
    )
    
    # Verify the correct method was called with right parameters
    graph_manager.lesson_memory.create_structured_lesson_observations.assert_called_once()
    call_kwargs = graph_manager.lesson_memory.create_structured_lesson_observations.call_args[1]
    assert call_kwargs["entity_name"] == "TestLesson"
    assert call_kwargs["what_was_learned"] == "Test content"
    assert call_kwargs["container_name"] == "Lessons"
    
    # Verify the result was returned correctly
    assert result == json.dumps(mock_result)


def test_lesson_operation_search(graph_manager):
    """Test that the lesson_operation method correctly dispatches search operations."""
    # Mock the expected return value
    mock_result = {"status": "success", "results": [{"name": "TestLesson"}]}
    graph_manager.lesson_memory.search_lesson_semantic.return_value = json.dumps(mock_result)
    
    # Call the operation dispatcher
    result = graph_manager.lesson_operation(
        "search",
        query="test search"
    )
    
    # Verify the correct method was called with right parameters
    graph_manager.lesson_memory.search_lesson_semantic.assert_called_once_with(
        query="test search", 
        limit=50, 
        container_name=None
    )
    
    # Verify the result was returned correctly
    assert result == json.dumps(mock_result)


def test_lesson_operation_error_handling(graph_manager):
    """Test that the lesson_operation method correctly handles errors."""
    # Set up the mock to raise an exception
    graph_manager.lesson_memory.create_lesson_entity.side_effect = ValueError("Test error")
    
    # Call the operation dispatcher
    result = graph_manager.lesson_operation(
        "create",
        name="TestLesson",
        lesson_type="BestPractice"
    )
    
    # Parse the result
    result_dict = json.loads(result)
    
    # Verify the error was handled correctly
    assert result_dict["status"] == "error"
    assert "Test error" in result_dict["error"]


def test_lesson_context(graph_manager):
    """Test that the lesson_context method correctly creates a context manager."""
    # Mock the expected return value for create method
    mock_result = {"status": "success", "entity": {"name": "ContextLesson"}}
    graph_manager.lesson_memory.create_lesson_entity.return_value = json.dumps(mock_result)
    
    # Use the context manager
    with graph_manager.lesson_context("TestProject", "TestContainer") as context:
        # Verify context object has the right attributes
        assert context.container_name == "TestContainer"
        assert context.lesson_memory == graph_manager.lesson_memory
        
        # Call a method on the context
        result = context.create("ContextLesson", "BestPractice")
        
        # Verify the underlying method was called correctly
        graph_manager.lesson_memory.create_lesson_entity.assert_called_once_with(
            "TestContainer", "ContextLesson", "BestPractice", None, None
        )
        
        # Verify the result was returned correctly
        assert result == json.dumps(mock_result)


def test_context_project_restoration(graph_manager):
    """Test that the lesson_context method correctly restores the project name."""
    original_project = "OriginalProject"
    graph_manager.default_project_name = original_project
    
    # Use the context manager with a different project
    with graph_manager.lesson_context("TempProject"):
        # Verify project was changed
        assert graph_manager.default_project_name == "TempProject"
    
    # Verify project was restored after context exit
    assert graph_manager.default_project_name == original_project


def test_lesson_context_observe(graph_manager):
    """Test observation creation through the context manager."""
    # Mock the expected return value
    mock_result = {"status": "success", "observations": [{"content": "Context observation"}]}
    graph_manager.lesson_memory.create_structured_lesson_observations.return_value = json.dumps(mock_result)
    
    # Use the context manager
    with graph_manager.lesson_context("TestProject", "TestContainer") as context:
        # Call the observe method
        result = context.observe(
            "ContextLesson",
            what_was_learned="Context observation",
            why_it_matters="Testing context"
        )
        
        # Verify the underlying method was called correctly
        graph_manager.lesson_memory.create_structured_lesson_observations.assert_called_once()
        call_kwargs = graph_manager.lesson_memory.create_structured_lesson_observations.call_args[1]
        assert call_kwargs["entity_name"] == "ContextLesson"
        assert call_kwargs["what_was_learned"] == "Context observation"
        assert call_kwargs["why_it_matters"] == "Testing context"
        assert call_kwargs["container_name"] == "TestContainer"
        
        # Verify the result was returned correctly
        assert result == json.dumps(mock_result)


def test_lesson_operation_invalid_type(graph_manager):
    """Test that the lesson_operation method correctly handles invalid operation types."""
    # Call the operation dispatcher with an invalid type
    result = graph_manager.lesson_operation("invalid_operation_type")
    
    # Parse the result
    result_dict = json.loads(result)
    
    # Verify the error was handled correctly
    assert result_dict["status"] == "error"
    assert "Unknown operation type" in result_dict["error"]
    assert result_dict["code"] == "invalid_operation" 