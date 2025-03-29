import pytest
from unittest.mock import MagicMock, patch, call
import json

from src.lesson_memory import LessonMemoryManager
from src.graph_memory.base_manager import BaseManager


@pytest.fixture
def mock_logger():
    """Create a mock logger."""
    return MagicMock()


@pytest.fixture
def mock_base_manager(mock_logger):
    """Create a mock base manager."""
    mock_manager = MagicMock(spec=BaseManager)
    mock_manager.logger = mock_logger
    mock_manager.ensure_initialized.return_value = None
    return mock_manager


@pytest.fixture
def mock_container():
    """Create a mock container component."""
    mock = MagicMock()
    # Set up return values for container methods
    mock.create_container.return_value = json.dumps({
        "container": {
            "id": "test-id-123",
            "name": "Test Lesson Container",
            "description": "This is a test container"
        }
    })
    mock.get_container.return_value = json.dumps({
        "container": {
            "id": "test-id-123",
            "name": "Test Lesson Container",
            "description": "This is a test container"
        }
    })
    mock.update_container.return_value = json.dumps({
        "container": {
            "id": "test-id-123",
            "name": "Test Lesson Container",
            "description": "Updated description"
        }
    })
    mock.delete_container.return_value = json.dumps({
        "status": "success",
        "message": "Container deleted successfully"
    })
    return mock


def test_init(mock_base_manager):
    """Test initialization of LessonMemoryManager."""
    manager = LessonMemoryManager(base_manager=mock_base_manager)
    assert manager.base_manager == mock_base_manager
    assert manager.logger == mock_base_manager.logger
    assert hasattr(manager, "container")
    assert hasattr(manager, "entity")
    assert hasattr(manager, "relation")
    assert hasattr(manager, "observation")
    assert hasattr(manager, "evolution")
    assert hasattr(manager, "consolidation")


def test_create_container(mock_base_manager, mock_logger):
    """Test creating a lesson container."""
    # Set up mock container.create_container to return a success response
    mock_container = MagicMock()
    mock_container.create_container.return_value = json.dumps({
        "container": {
            "id": "test-id-123",
            "name": "Test Lesson Container",
            "description": "This is a test container"
        }
    })
    
    # Create manager with mocked container
    manager = LessonMemoryManager(base_manager=mock_base_manager)
    manager.container = mock_container
    
    # Call create_container
    result = manager.create_container("Test Lesson Container", "This is a test container", {})
    
    # Verify container.create_container was called with correct args
    mock_container.create_container.assert_called_once_with(
        "Test Lesson Container", "This is a test container", {}
    )
    
    # Verify result
    result_dict = json.loads(result)
    assert "container" in result_dict
    assert result_dict["container"]["name"] == "Test Lesson Container"
    assert result_dict["container"]["description"] == "This is a test container"


def test_create_container_missing_name(mock_base_manager):
    """Test creating a lesson container with missing name."""
    # Create manager with mock container
    manager = LessonMemoryManager(base_manager=mock_base_manager)
    
    # Create a mock for the container component that would raise an exception
    mock_container = MagicMock()
    mock_container.create_container.side_effect = TypeError("name cannot be empty")
    manager.container = mock_container
    
    # Attempt to create container with empty string as name (which would fail)
    with pytest.raises(TypeError):
        manager.create_container("", "This is a test container", {})
    
    # Verify container.create_container was called with empty string
    mock_container.create_container.assert_called_once_with("", "This is a test container", {})


def test_get_container(mock_base_manager):
    """Test getting a lesson container by ID."""
    # Set up mock container.get_container to return a success response
    mock_container = MagicMock()
    mock_container.get_container.return_value = json.dumps({
        "container": {
            "id": "test-id-123",
            "name": "Test Lesson Container",
            "description": "This is a test container"
        }
    })
    
    # Create manager with mocked container
    manager = LessonMemoryManager(base_manager=mock_base_manager)
    manager.container = mock_container
    
    # Call get_container
    result = manager.get_container("Test Lesson Container")
    
    # Verify container.get_container was called with correct args
    mock_container.get_container.assert_called_once_with("Test Lesson Container")
    
    # Verify result
    result_dict = json.loads(result)
    assert "container" in result_dict
    assert result_dict["container"]["name"] == "Test Lesson Container"


def test_get_container_not_found(mock_base_manager):
    """Test getting a lesson container that doesn't exist."""
    # Set up mock container.get_container to return an error response
    mock_container = MagicMock()
    mock_container.get_container.return_value = json.dumps({
        "error": "Lesson container 'NonExistentContainer' not found"
    })
    
    # Create manager with mocked container
    manager = LessonMemoryManager(base_manager=mock_base_manager)
    manager.container = mock_container
    
    # Call get_container
    result = manager.get_container("NonExistentContainer")
    
    # Verify container.get_container was called with correct args
    mock_container.get_container.assert_called_once_with("NonExistentContainer")
    
    # Verify result
    result_dict = json.loads(result)
    assert "error" in result_dict
    assert "not found" in result_dict["error"]


def test_update_container(mock_base_manager):
    """Test updating a lesson container."""
    # Set up mock container.update_container to return a success response
    mock_container = MagicMock()
    mock_container.update_container.return_value = json.dumps({
        "container": {
            "id": "test-id-123",
            "name": "Test Lesson Container",
            "description": "Updated description"
        }
    })
    
    # Create manager with mocked container
    manager = LessonMemoryManager(base_manager=mock_base_manager)
    manager.container = mock_container
    
    # Call update_container
    updates = {"description": "Updated description"}
    result = manager.update_container("Test Lesson Container", updates)
    
    # Verify container.update_container was called with correct args
    mock_container.update_container.assert_called_once_with("Test Lesson Container", updates)
    
    # Verify result
    result_dict = json.loads(result)
    assert "container" in result_dict
    assert result_dict["container"]["description"] == "Updated description"


def test_delete_container(mock_base_manager):
    """Test deleting a lesson container."""
    # Set up mock container.delete_container to return a success response
    mock_container = MagicMock()
    mock_container.delete_container.return_value = json.dumps({
        "status": "success",
        "message": "Container deleted successfully"
    })
    
    # Create manager with mocked container
    manager = LessonMemoryManager(base_manager=mock_base_manager)
    manager.container = mock_container
    
    # Call delete_container with delete_entities flag
    delete_entities = True
    result = manager.delete_container("Test Lesson Container", delete_entities)
    
    # Verify container.delete_container was called with correct args
    mock_container.delete_container.assert_called_once_with(
        "Test Lesson Container", delete_entities
    )
    
    # Verify result
    result_dict = json.loads(result)
    assert "status" in result_dict
    assert result_dict["status"] == "success" 