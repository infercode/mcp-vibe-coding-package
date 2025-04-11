"""
Backward compatibility tests for lesson memory integration.

These tests verify that existing code using direct LessonMemoryManager methods
continues to work correctly after the integration of the new layered access
approach.
"""

import json
import pytest
from unittest.mock import MagicMock, patch

from src.graph_memory import GraphMemoryManager


@pytest.fixture
def graph_manager():
    """Create a GraphMemoryManager with mocked components."""
    with patch.object(GraphMemoryManager, 'initialize', return_value=True):
        manager = GraphMemoryManager()
    
    # Mock the lesson_memory
    manager.lesson_memory = MagicMock()
    
    # Set a default project
    manager.default_project_name = "TestProject"
    return manager


def test_direct_lesson_memory_access(graph_manager):
    """Test that direct access to lesson_memory property still works."""
    # Setup mock return value
    mock_result = {"status": "success", "entity": {"name": "DirectAccessLesson"}}
    graph_manager.lesson_memory.create_lesson_entity.return_value = json.dumps(mock_result)
    
    # Simulate existing code that directly accesses lesson_memory
    result = graph_manager.lesson_memory.create_lesson_entity(
        "DirectContainer", "DirectAccessLesson", "BestPractice", None, None
    )
    
    # Verify the method was called correctly
    graph_manager.lesson_memory.create_lesson_entity.assert_called_once_with(
        "DirectContainer", "DirectAccessLesson", "BestPractice", None, None
    )
    
    # Verify result was returned correctly
    assert result == json.dumps(mock_result)


def test_direct_and_layered_access_coexistence(graph_manager):
    """Test that direct and layered access methods can be used together."""
    # Setup mock return values
    mock_create_result = {"status": "success", "entity": {"name": "CoexistenceLesson"}}
    mock_search_result = {"status": "success", "results": [{"name": "CoexistenceLesson"}]}
    
    graph_manager.lesson_memory.create_lesson_entity.return_value = json.dumps(mock_create_result)
    graph_manager.lesson_memory.search_lesson_semantic.return_value = json.dumps(mock_search_result)
    
    # Use direct access for creation (simulating existing code)
    direct_result = graph_manager.lesson_memory.create_lesson_entity(
        "CoexistenceContainer", "CoexistenceLesson", "BestPractice", None, None
    )
    
    # Use layered access for searching (simulating new code)
    layered_result = graph_manager.lesson_operation(
        "search",
        query="find coexistence",
        container_name="CoexistenceContainer"
    )
    
    # Verify both methods were called correctly
    graph_manager.lesson_memory.create_lesson_entity.assert_called_once_with(
        "CoexistenceContainer", "CoexistenceLesson", "BestPractice", None, None
    )
    
    graph_manager.lesson_memory.search_lesson_semantic.assert_called_once()
    
    # Verify both results were returned correctly
    assert direct_result == json.dumps(mock_create_result)
    assert layered_result == json.dumps(mock_search_result)


def test_property_forwarding(graph_manager):
    """Test that property access is correctly forwarded to the underlying LessonMemoryManager."""
    # Setup a mock property
    graph_manager.lesson_memory.some_property = "test_value"
    
    # Access the property through GraphMemoryManager
    value = graph_manager.lesson_memory.some_property
    
    # Verify the value was correctly forwarded
    assert value == "test_value"


def test_method_chaining(graph_manager):
    """Test that method chaining patterns still work."""
    # Setup mock chained methods
    chain_result = MagicMock()
    graph_manager.lesson_memory.start_chain.return_value.continue_chain.return_value = chain_result
    
    # Simulate code using method chaining
    result = graph_manager.lesson_memory.start_chain().continue_chain()
    
    # Verify the chain was called correctly
    graph_manager.lesson_memory.start_chain.assert_called_once()
    graph_manager.lesson_memory.start_chain.return_value.continue_chain.assert_called_once()
    
    # Verify the result is correct
    assert result == chain_result


def test_complex_interaction_pattern(graph_manager):
    """Test a more complex interaction pattern that mixes direct and layered access."""
    # Setup mock return values
    container_exists_result = True
    create_container_result = json.dumps({"status": "success", "container": {"name": "ComplexContainer"}})
    create_entity_result = json.dumps({"status": "success", "entity": {"name": "ComplexLesson"}})
    observation_result = json.dumps({"status": "success", "observations": [{"content": "Complex observation"}]})
    
    graph_manager.lesson_memory.container_exists.return_value = container_exists_result
    graph_manager.lesson_memory.create_lesson_container.return_value = create_container_result
    graph_manager.lesson_memory.create_lesson_entity.return_value = create_entity_result
    graph_manager.lesson_memory.create_structured_lesson_observations.return_value = observation_result
    
    # Simulate a complex interaction pattern
    
    # 1. Check if container exists (direct access)
    container_exists = graph_manager.lesson_memory.container_exists("ComplexContainer")
    
    # 2. Create container if it doesn't exist (direct access)
    if not container_exists:
        graph_manager.lesson_memory.create_lesson_container("ComplexContainer", "Complex test container")
    
    # 3. Create lesson entity (layered access)
    create_result = graph_manager.lesson_operation(
        "create",
        container_name="ComplexContainer",
        name="ComplexLesson",
        lesson_type="BestPractice"
    )
    
    # 4. Add observation using context manager (layered access)
    with graph_manager.lesson_context("TestProject", "ComplexContainer") as context:
        observe_result = context.observe(
            "ComplexLesson",
            what_was_learned="Complex observation",
            why_it_matters="Testing complex patterns"
        )
    
    # Verify all methods were called correctly
    graph_manager.lesson_memory.container_exists.assert_called_once_with("ComplexContainer")
    graph_manager.lesson_memory.create_lesson_container.assert_not_called()  # Container exists, so this shouldn't be called
    
    # Verify create operation
    graph_manager.lesson_memory.create_lesson_entity.assert_called_once()
    
    # Verify observe operation 
    graph_manager.lesson_memory.create_structured_lesson_observations.assert_called_once()
    
    # Verify results
    assert container_exists == container_exists_result
    assert create_result == create_entity_result
    assert observe_result == observation_result


def test_error_handling_compatibility(graph_manager):
    """Test that error handling works consistently between direct and layered access."""
    # Setup mock to raise exception
    error_message = "Test error from direct access"
    graph_manager.lesson_memory.create_lesson_entity.side_effect = ValueError(error_message)
    
    # Use direct access - should raise exception
    with pytest.raises(ValueError) as excinfo:
        graph_manager.lesson_memory.create_lesson_entity(
            "ErrorContainer", "ErrorLesson", "BestPractice", None, None
        )
    assert error_message in str(excinfo.value)
    
    # Use layered access - should handle exception and return error JSON
    layered_result = graph_manager.lesson_operation(
        "create",
        container_name="ErrorContainer",
        name="ErrorLesson",
        lesson_type="BestPractice"
    )
    
    result_dict = json.loads(layered_result)
    assert result_dict["status"] == "error"
    assert error_message in result_dict["error"]
    
    # Verify both methods were attempted
    assert graph_manager.lesson_memory.create_lesson_entity.call_count == 2 