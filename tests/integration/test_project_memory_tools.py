#!/usr/bin/env python3
"""
Integration tests for Project Memory Tools.

This verifies that the Project Memory Tools correctly interface with GraphMemoryManager.
"""

import pytest
import json
from unittest.mock import MagicMock, patch

from src.tools.project_memory_tools import get_graph_manager, register_project_tools
from src.graph_memory import GraphMemoryManager


class MockServer:
    """Mock server for testing tool registration."""
    
    def __init__(self):
        self.tools = {}
    
    def tool(self):
        """Decorator mock for tool registration."""
        def decorator(func):
            self.tools[func.__name__] = func
            return func
        return decorator


class MockClientManager:
    """Mock for client manager function."""
    
    def __init__(self, graph_manager):
        self.graph_manager = graph_manager
    
    def __call__(self, client_id=None):
        return self.graph_manager


@pytest.fixture
def mock_graph_manager():
    """Create a mock GraphMemoryManager."""
    mock_manager = MagicMock(spec=GraphMemoryManager)
    # Setup project_operation to return a success response
    mock_manager.project_operation.return_value = json.dumps({
        "status": "success",
        "message": "Operation completed successfully",
        "data": {"test": "value"}
    })
    return mock_manager


@pytest.fixture
def server():
    """Create a MockServer instance."""
    return MockServer()


@pytest.fixture
def tools(server, mock_graph_manager):
    """Register and return tools with mock dependencies."""
    # Create a mock client manager function
    get_client_manager = MockClientManager(mock_graph_manager)
    
    # Register tools with mock dependencies
    with patch('src.tools.project_memory_tools.get_graph_manager', return_value=mock_graph_manager):
        registered_tools = register_project_tools(server, get_client_manager)
    
    return registered_tools


@pytest.mark.asyncio
async def test_project_memory_tool_registration(tools):
    """Test that project memory tools are properly registered."""
    assert "project_memory_tool" in tools
    assert "project_memory_context" in tools
    assert callable(tools["project_memory_tool"])
    assert callable(tools["project_memory_context"])


@pytest.mark.asyncio
async def test_project_memory_tool_create_project(server, mock_graph_manager):
    """Test project_memory_tool with create_project operation."""
    # Set up mock client manager
    get_client_manager = MockClientManager(mock_graph_manager)
    
    # Register tools with mock dependencies
    with patch('src.tools.project_memory_tools.get_graph_manager', return_value=mock_graph_manager):
        tools = register_project_tools(server, get_client_manager)
    
    # Call project_memory_tool with create_project operation
    result = await server.tools["project_memory_tool"](
        operation_type="create_project",
        name="Test Project",
        description="A test project"
    )
    
    # Verify GraphMemoryManager.project_operation was called with correct parameters
    mock_graph_manager.project_operation.assert_called_once_with(
        "create_project",
        name="Test Project",
        description="A test project"
    )
    
    # Verify result is passed through from GraphMemoryManager
    result_data = json.loads(result)
    assert result_data["status"] == "success"
    assert "message" in result_data
    assert "data" in result_data


@pytest.mark.asyncio
async def test_project_memory_tool_create_component(server, mock_graph_manager):
    """Test project_memory_tool with create_component operation."""
    # Set up mock client manager
    get_client_manager = MockClientManager(mock_graph_manager)
    
    # Register tools with mock dependencies
    with patch('src.tools.project_memory_tools.get_graph_manager', return_value=mock_graph_manager):
        tools = register_project_tools(server, get_client_manager)
    
    # Call project_memory_tool with create_component operation
    result = await server.tools["project_memory_tool"](
        operation_type="create_component",
        project_id="Test Project",
        name="Authentication Service",
        component_type="MICROSERVICE"
    )
    
    # Verify GraphMemoryManager.project_operation was called with correct parameters
    mock_graph_manager.project_operation.assert_called_once_with(
        "create_component",
        project_id="Test Project",
        name="Authentication Service",
        component_type="MICROSERVICE"
    )
    
    # Verify result is passed through from GraphMemoryManager
    result_data = json.loads(result)
    assert result_data["status"] == "success"


@pytest.mark.asyncio
async def test_project_memory_tool_invalid_operation(server, mock_graph_manager):
    """Test project_memory_tool with invalid operation type."""
    # Set up mock client manager
    get_client_manager = MockClientManager(mock_graph_manager)
    
    # Register tools with mock dependencies
    with patch('src.tools.project_memory_tools.get_graph_manager', return_value=mock_graph_manager):
        tools = register_project_tools(server, get_client_manager)
    
    # Call project_memory_tool with invalid operation
    result = await server.tools["project_memory_tool"](
        operation_type="invalid_operation",
        name="Test Project"
    )
    
    # Verify GraphMemoryManager.project_operation was NOT called
    mock_graph_manager.project_operation.assert_not_called()
    
    # Verify error response
    result_data = json.loads(result)
    assert result_data["status"] == "error"
    assert "Invalid operation type" in result_data["error"]


@pytest.mark.asyncio
async def test_project_memory_tool_error_handling(server, mock_graph_manager):
    """Test error handling in project_memory_tool."""
    # Set up mock client manager
    get_client_manager = MockClientManager(mock_graph_manager)
    
    # Configure mock to raise an exception
    mock_graph_manager.project_operation.side_effect = Exception("Test error")
    
    # Register tools with mock dependencies
    with patch('src.tools.project_memory_tools.get_graph_manager', return_value=mock_graph_manager):
        tools = register_project_tools(server, get_client_manager)
    
    # Call project_memory_tool
    result = await server.tools["project_memory_tool"](
        operation_type="create_project",
        name="Test Project"
    )
    
    # Verify error response
    result_data = json.loads(result)
    assert result_data["status"] == "error"
    assert "Operation failed" in result_data["error"]
    assert "Test error" in result_data["error"]


@pytest.mark.asyncio
async def test_project_memory_context_creation(server, mock_graph_manager):
    """Test project_memory_context tool."""
    # Set up mock client manager
    get_client_manager = MockClientManager(mock_graph_manager)
    
    # Register tools with mock dependencies
    with patch('src.tools.project_memory_tools.get_graph_manager', return_value=mock_graph_manager):
        tools = register_project_tools(server, get_client_manager)
    
    # Call project_memory_context
    result = await server.tools["project_memory_context"](
        context_data={"project_name": "Test Project"}
    )
    
    # Verify result
    result_data = json.loads(result)
    assert result_data["status"] == "success"
    assert "context" in result_data
    assert result_data["context"]["project_name"] == "Test Project"
    assert "created_at" in result_data["context"]
    assert "operations_available" in result_data["context"]


@pytest.mark.asyncio
async def test_project_memory_context_missing_project_name(server, mock_graph_manager):
    """Test project_memory_context with missing project name."""
    # Set up mock client manager
    get_client_manager = MockClientManager(mock_graph_manager)
    
    # Register tools with mock dependencies
    with patch('src.tools.project_memory_tools.get_graph_manager', return_value=mock_graph_manager):
        tools = register_project_tools(server, get_client_manager)
    
    # Call project_memory_context without project_name
    result = await server.tools["project_memory_context"](
        context_data={}
    )
    
    # Verify error response
    result_data = json.loads(result)
    assert result_data["status"] == "error"
    assert "Project name is required" in result_data["error"] 