import pytest
from unittest.mock import patch, MagicMock
import json

from src.graph_memory import GraphMemoryManager


@pytest.fixture
def mock_project_manager():
    """Create a mock project memory manager."""
    project_manager = MagicMock()
    
    # Mock specific return values
    project_manager.delete_project_container.return_value = json.dumps({
        "status": "success",
        "message": "Project deleted successfully"
    })
    
    project_manager.delete_project_domain.return_value = json.dumps({
        "status": "success",
        "message": "Domain deleted successfully"
    })
    
    project_manager.delete_project_component.return_value = json.dumps({
        "status": "success",
        "message": "Component deleted successfully"
    })
    
    project_manager.delete_project_dependency.return_value = json.dumps({
        "status": "success",
        "message": "Relationship deleted successfully"
    })
    
    return project_manager


@pytest.fixture
def mock_graph_memory_manager(mock_project_manager):
    """Create a mock graph memory manager with the project_operation method."""
    manager = MagicMock(spec=GraphMemoryManager)
    
    # Set the real project_operation method
    # This is the critical part - we want to test the actual method implementation
    manager.project_operation = GraphMemoryManager.project_operation.__get__(manager)
    manager.project_memory = mock_project_manager
    manager._ensure_initialized = MagicMock()
    manager._handle_project_creation = MagicMock(return_value=json.dumps({
        "status": "success",
        "message": "Project created successfully"
    }))
    manager._handle_component_creation = MagicMock(return_value=json.dumps({
        "status": "success",
        "message": "Component created successfully"
    }))
    manager._handle_domain_entity_creation = MagicMock(return_value=json.dumps({
        "status": "success",
        "message": "Domain entity created successfully"
    }))
    manager._handle_entity_relationship = MagicMock(return_value=json.dumps({
        "status": "success",
        "message": "Entity relationship created successfully"
    }))
    manager._handle_project_search = MagicMock(return_value=json.dumps({
        "status": "success",
        "results": [{"name": "Test Entity"}]
    }))
    manager._handle_structure_retrieval = MagicMock(return_value=json.dumps({
        "status": "success",
        "structure": {"projects": [], "domains": []}
    }))
    manager._handle_add_observation = MagicMock(return_value=json.dumps({
        "status": "success",
        "message": "Observation added successfully"
    }))
    manager._handle_entity_update = MagicMock(return_value=json.dumps({
        "status": "success",
        "message": "Entity updated successfully"
    }))
    manager._handle_entity_deletion = MagicMock(return_value=json.dumps({
        "status": "success",
        "message": "Entity deleted successfully"
    }))
    manager._handle_relationship_deletion = MagicMock(return_value=json.dumps({
        "status": "success",
        "message": "Relationship deleted successfully"
    }))
    
    return manager


def test_project_operation_create_project(mock_graph_memory_manager):
    """Test project_operation with create_project operation."""
    result = mock_graph_memory_manager.project_operation(
        operation_type="create_project",
        name="Test Project",
        description="Test description"
    )
    
    # Verify the correct handler was called
    mock_graph_memory_manager._handle_project_creation.assert_called_once_with(
        name="Test Project",
        description="Test description"
    )
    
    # Verify result
    assert json.loads(result)["status"] == "success"


def test_project_operation_invalid_operation(mock_graph_memory_manager):
    """Test project_operation with an invalid operation type."""
    result = mock_graph_memory_manager.project_operation(
        operation_type="invalid_operation"
    )
    
    # Verify error response
    assert json.loads(result)["status"] == "error"
    assert "Unknown operation type" in json.loads(result)["error"]


def test_project_operation_delete_entity(mock_graph_memory_manager):
    """Test project_operation with delete_entity operation."""
    # Test case 1: Delete a project
    result = mock_graph_memory_manager.project_operation(
        operation_type="delete_entity",
        entity_name="Test Project",
        entity_type="project",
        delete_contents=True
    )
    
    # Verify the correct handler was called
    mock_graph_memory_manager._handle_entity_deletion.assert_called_with(
        entity_name="Test Project",
        entity_type="project",
        delete_contents=True
    )
    
    # Verify result
    assert json.loads(result)["status"] == "success"
    
    # Test case 2: Delete a domain
    result = mock_graph_memory_manager.project_operation(
        operation_type="delete_entity",
        entity_name="Backend",
        entity_type="domain",
        container_name="Test Project",
        delete_contents=True
    )
    
    # Verify the correct handler was called
    mock_graph_memory_manager._handle_entity_deletion.assert_called_with(
        entity_name="Backend",
        entity_type="domain",
        container_name="Test Project",
        delete_contents=True
    )
    
    # Verify result
    assert json.loads(result)["status"] == "success"
    
    # Test case 3: Delete a component
    result = mock_graph_memory_manager.project_operation(
        operation_type="delete_entity",
        entity_name="Auth Service",
        entity_type="component",
        container_name="Test Project",
        domain_name="Backend"
    )
    
    # Verify the correct handler was called
    mock_graph_memory_manager._handle_entity_deletion.assert_called_with(
        entity_name="Auth Service",
        entity_type="component",
        container_name="Test Project",
        domain_name="Backend"
    )
    
    # Verify result
    assert json.loads(result)["status"] == "success"
    
    # Test case 4: Delete an observation
    result = mock_graph_memory_manager.project_operation(
        operation_type="delete_entity",
        entity_name="Auth Service",
        entity_type="observation",
        observation_content="Security issue found"
    )
    
    # Verify the correct handler was called
    mock_graph_memory_manager._handle_entity_deletion.assert_called_with(
        entity_name="Auth Service",
        entity_type="observation",
        observation_content="Security issue found"
    )
    
    # Verify result
    assert json.loads(result)["status"] == "success"


def test_project_operation_delete_relationship(mock_graph_memory_manager):
    """Test project_operation with delete_relationship operation."""
    # Test deleting a component dependency
    result = mock_graph_memory_manager.project_operation(
        operation_type="delete_relationship",
        source_name="Auth Service",
        target_name="User Database",
        relationship_type="DEPENDS_ON",
        container_name="Test Project",
        domain_name="Backend"
    )
    
    # Verify the correct handler was called
    mock_graph_memory_manager._handle_relationship_deletion.assert_called_with(
        source_name="Auth Service",
        target_name="User Database",
        relationship_type="DEPENDS_ON",
        container_name="Test Project",
        domain_name="Backend"
    )
    
    # Verify result
    assert json.loads(result)["status"] == "success"


def test_all_project_operations(mock_graph_memory_manager):
    """Test all project_operation types to ensure they're correctly routed."""
    operations = {
        "create_project": mock_graph_memory_manager._handle_project_creation,
        "create_component": mock_graph_memory_manager._handle_component_creation,
        "create_domain_entity": mock_graph_memory_manager._handle_domain_entity_creation,
        "relate_entities": mock_graph_memory_manager._handle_entity_relationship,
        "search": mock_graph_memory_manager._handle_project_search,
        "get_structure": mock_graph_memory_manager._handle_structure_retrieval,
        "add_observation": mock_graph_memory_manager._handle_add_observation,
        "update": mock_graph_memory_manager._handle_entity_update,
        "delete_entity": mock_graph_memory_manager._handle_entity_deletion,
        "delete_relationship": mock_graph_memory_manager._handle_relationship_deletion,
    }
    
    # Test each operation with minimal parameters
    for operation_type, handler in operations.items():
        # Reset mock call history
        handler.reset_mock()
        
        # Prepare basic parameters for each operation type
        params = {
            "create_project": {"name": "Test Project"},
            "create_component": {"name": "Test Component", "component_type": "SERVICE", "project_id": "Test Project"},
            "create_domain_entity": {"name": "Test Entity", "entity_type": "DECISION", "project_id": "Test Project"},
            "relate_entities": {"source_name": "A", "target_name": "B", "relation_type": "RELATES_TO", "project_id": "P"},
            "search": {"query": "test", "project_id": "Test Project"},
            "get_structure": {"project_id": "Test Project"},
            "add_observation": {"entity_name": "Test Entity", "content": "Test observation"},
            "update": {"entity_name": "Test Entity", "updates": {"description": "Updated"}},
            "delete_entity": {"entity_name": "Test Entity", "entity_type": "component"},
            "delete_relationship": {"source_name": "A", "target_name": "B", "relationship_type": "RELATES_TO"},
        }
        
        # Call project_operation with the appropriate parameters
        mock_graph_memory_manager.project_operation(
            operation_type=operation_type,
            **params[operation_type]
        )
        
        # Verify the correct handler was called exactly once
        assert handler.call_count == 1
        
        # Verify handler was called with the right parameters
        handler.assert_called_once_with(**params[operation_type]) 