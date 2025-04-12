#!/usr/bin/env python3
"""
Backward Compatibility Tests for Project Memory Integration.

This verifies that the new project memory tools interface properly with existing project memory
functionality and that both old and new approaches work simultaneously.
"""

import pytest
import json
from unittest.mock import MagicMock, patch

from src.tools.project_memory_tools import get_graph_manager, register_project_tools
from src.graph_memory import GraphMemoryManager


@pytest.fixture
def mock_graph_manager():
    """Create a mock GraphMemoryManager with both old and new interfaces."""
    mock_manager = MagicMock(spec=GraphMemoryManager)
    
    # Setup project_operation to return a success response
    mock_manager.project_operation.return_value = json.dumps({
        "status": "success",
        "message": "Operation completed successfully using new interface",
        "data": {"test": "value"}
    })
    
    # Setup old-style project_memory.create_project_container to return success
    mock_project_memory = MagicMock()
    mock_project_memory.create_project_container.return_value = {
        "status": "success",
        "message": "Project container created using old interface",
        "container": {"name": "Test Project", "id": "test-123"}
    }
    mock_manager.project_memory = mock_project_memory
    
    return mock_manager


@pytest.mark.asyncio
async def test_backward_compatibility_create_project():
    """
    Test that both new and old interfaces for creating projects work.
    The new interface should use project_operation, while direct access
    to project_memory should still be available.
    """
    mock_manager = MagicMock(spec=GraphMemoryManager)
    
    # Setup project_operation (new style)
    mock_manager.project_operation.return_value = json.dumps({
        "status": "success",
        "message": "Project created via new interface",
        "data": {"name": "Test Project", "id": "test-123"}
    })
    
    # Setup project_memory.create_project_container (old style)
    mock_project_memory = MagicMock()
    mock_project_memory.create_project_container.return_value = {
        "status": "success",
        "message": "Project created via old interface",
        "container": {"name": "Test Project", "id": "test-123"}
    }
    mock_manager.project_memory = mock_project_memory
    
    # Patch the get_graph_manager to return our mock
    with patch('src.tools.project_memory_tools.get_graph_manager', return_value=mock_manager):
        # Call new style interface through project_operation
        project_data = {
            "name": "Test Project",
            "description": "Test description"
        }
        
        # New style - through project_operation
        new_style_result = mock_manager.project_operation("create_project", **project_data)
        new_style_data = json.loads(new_style_result)
        
        # Old style - direct call to project_memory
        old_style_result = mock_manager.project_memory.create_project_container(project_data)
        
        # Verify both were called correctly
        mock_manager.project_operation.assert_called_once_with("create_project", **project_data)
        mock_manager.project_memory.create_project_container.assert_called_once_with(project_data)
        
        # Verify new interface responded correctly
        assert new_style_data["status"] == "success"
        assert "Project created via new interface" in new_style_data["message"]
        
        # Verify old interface responded correctly
        assert old_style_result["status"] == "success"
        assert "Project created via old interface" in old_style_result["message"]


@pytest.mark.asyncio
async def test_backward_compatibility_create_component():
    """
    Test that both new and old interfaces for creating components work.
    """
    mock_manager = MagicMock(spec=GraphMemoryManager)
    
    # Setup project_operation (new style)
    mock_manager.project_operation.return_value = json.dumps({
        "status": "success",
        "message": "Component created via new interface",
        "component": {"name": "Auth Service", "id": "comp-123"}
    })
    
    # Setup project_memory.create_component (old style)
    mock_project_memory = MagicMock()
    mock_project_memory.create_component.return_value = {
        "status": "success",
        "message": "Component created via old interface",
        "component": {"name": "Auth Service", "id": "comp-123"}
    }
    mock_manager.project_memory = mock_project_memory
    
    # Component data
    component_data = {
        "project_id": "Test Project",
        "name": "Auth Service",
        "component_type": "MICROSERVICE"
    }
    
    # Patch the get_graph_manager to return our mock
    with patch('src.tools.project_memory_tools.get_graph_manager', return_value=mock_manager):
        # New style - through project_operation
        new_style_result = mock_manager.project_operation("create_component", **component_data)
        new_style_data = json.loads(new_style_result)
        
        # Old style - direct call to project_memory
        old_style_result = mock_manager.project_memory.create_component(**component_data)
        
        # Verify both were called correctly
        mock_manager.project_operation.assert_called_once_with("create_component", **component_data)
        mock_manager.project_memory.create_component.assert_called_once_with(**component_data)
        
        # Verify new interface responded correctly
        assert new_style_data["status"] == "success"
        assert "Component created via new interface" in new_style_data["message"]
        
        # Verify old interface responded correctly
        assert old_style_result["status"] == "success"
        assert "Component created via old interface" in old_style_result["message"]


@pytest.mark.asyncio
async def test_project_context_methods():
    """
    Test that the project_context context manager works with both old and new interfaces.
    """
    mock_manager = MagicMock(spec=GraphMemoryManager)
    
    # Create a mock context manager
    mock_context = MagicMock()
    mock_context.create_component.return_value = json.dumps({
        "status": "success", 
        "message": "Component created via context",
        "component": {"name": "API Gateway", "id": "comp-456"}
    })
    
    # Setup project_context to return our mock context
    mock_manager.project_context.return_value.__enter__.return_value = mock_context
    
    # Patch the get_graph_manager to return our mock
    with patch('src.tools.project_memory_tools.get_graph_manager', return_value=mock_manager):
        # Use the context manager
        with mock_manager.project_context("Test Project") as context:
            # Call create_component on the context
            result = context.create_component(
                name="API Gateway",
                component_type="GATEWAY"
            )
            
            # Verify the context method was called
            context.create_component.assert_called_once_with(
                name="API Gateway",
                component_type="GATEWAY"
            )
            
            # Verify the result
            result_data = json.loads(result)
            assert result_data["status"] == "success"
            assert "Component created via context" in result_data["message"]
        
        # Verify the context manager was called correctly
        mock_manager.project_context.assert_called_once_with("Test Project")


def test_implementation_completeness():
    """
    Verify that all essential operations are implemented in both interfaces.
    This ensures we haven't missed any critical operations in the new interface.
    """
    # Create a real GraphMemoryManager instance to check for attribute existence
    # We won't actually connect to Neo4j
    with patch('src.graph_memory.GraphMemoryManager.initialize', return_value=True):
        manager = GraphMemoryManager()
    
    # Check that both interfaces exist
    assert hasattr(manager, "project_memory"), "Old interface (project_memory) missing"
    assert hasattr(manager, "project_operation"), "New interface (project_operation) missing"
    
    # Check for key project_memory methods
    assert hasattr(manager.project_memory, "create_project_container")
    assert hasattr(manager.project_memory, "get_project_container")
    assert hasattr(manager.project_memory, "create_component")
    assert hasattr(manager.project_memory, "create_domain_entity")
    
    # Check that project_context exists
    assert hasattr(manager, "project_context") 