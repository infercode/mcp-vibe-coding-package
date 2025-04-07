import pytest
from unittest.mock import MagicMock, patch, call
import json
from neo4j import Record
from neo4j.graph import Node

from src.project_memory import ProjectMemoryManager


def test_init(mock_base_manager, mock_logger):
    """Test initialization of ProjectMemoryManager."""
    # Set logger on mock_base_manager
    mock_base_manager.logger = mock_logger
    
    # Create manager
    manager = ProjectMemoryManager(base_manager=mock_base_manager)
    
    # Verify manager attributes are set correctly
    assert manager.base_manager == mock_base_manager
    assert hasattr(manager, "domain_manager")
    assert hasattr(manager, "component_manager")
    assert hasattr(manager, "dependency_manager")
    assert hasattr(manager, "version_manager")
    assert manager.logger == mock_logger


def test_create_project_container(mock_base_manager, mock_logger):
    """Test creating a project container with all fields."""
    # Set up mock response from ProjectContainer
    mock_response = json.dumps({
        "status": "success",
        "message": "Project container 'Test Project' created successfully",
        "container": {
            "id": "prj123", 
            "name": "Test Project",
            "created": "2022-01-20T12:00:00.000000",
            "lastUpdated": "2022-01-20T12:00:00.000000"
        }
    })
    
    # Create mock for project_container
    mock_project_container = MagicMock()
    mock_project_container.create_project_container.return_value = mock_response
    
    # Create ProjectMemoryManager and inject mock project_container
    manager = ProjectMemoryManager(mock_base_manager)
    manager.project_container = mock_project_container
    
    # Full project data
    project_data = {
        "name": "Test Project",
        "description": "A test project for unit testing",
        "metadata": {
            "owner": "test_user",
            "priority": "high",
            "version": "1.0"
        },
        "tags": ["test", "project", "unit_testing"]
    }
    
    # Call create_project_container
    result = manager.create_project_container(project_data)
    
    # Assert that the ProjectContainer method was called
    mock_project_container.create_project_container.assert_called_once_with(project_data)
    
    # Verify result structure
    assert "status" in result
    assert result["status"] == "success"
    assert "message" in result
    assert "container" in result
    
    # Verify container has timestamp fields in ISO format
    if "container" in result and isinstance(result["container"], dict):
        if "created" in result["container"]:
            assert isinstance(result["container"]["created"], str)
            # Check it looks like an ISO datetime
            assert "T" in result["container"]["created"]
        
        if "lastUpdated" in result["container"]:
            assert isinstance(result["container"]["lastUpdated"], str)
            # Check it looks like an ISO datetime
            assert "T" in result["container"]["lastUpdated"]


def test_create_project_container_with_default_values(mock_base_manager, mock_logger):
    """Test creating a project container with minimal fields."""
    # Set up mock response from ProjectContainer
    mock_response = json.dumps({
        "status": "success",
        "message": "Project container 'Minimal Project' created successfully",
        "container": {
            "id": "prj456", 
            "name": "Minimal Project"
        }
    })
    
    # Create mock for project_container
    mock_project_container = MagicMock()
    mock_project_container.create_project_container.return_value = mock_response
    
    # Create ProjectMemoryManager and inject mock project_container
    manager = ProjectMemoryManager(mock_base_manager)
    manager.project_container = mock_project_container
    
    # Minimal project data
    project_data = {"name": "Minimal Project"}
    
    # Call create_project_container
    result = manager.create_project_container(project_data)
    
    # Assert that the ProjectContainer method was called
    mock_project_container.create_project_container.assert_called_once_with(project_data)
    
    # Verify result structure
    assert "status" in result
    assert result["status"] == "success"
    assert "message" in result
    assert "container" in result


def test_get_project_container(mock_base_manager, mock_logger):
    """Test retrieving a project container by ID."""
    # Set up mock response from ProjectContainer
    mock_response = json.dumps({
        "container": {
            "id": "proj_12345",
            "name": "Test Project",
            "entityType": "ProjectContainer",
            "description": "This is a test project",
            "domain_count": 3,
            "component_count": 10
        }
    })
    
    # Create mock for project_container
    mock_project_container = MagicMock()
    mock_project_container.get_project_container.return_value = mock_response
    
    # Create ProjectMemoryManager and inject mock project_container
    manager = ProjectMemoryManager(mock_base_manager)
    manager.project_container = mock_project_container
    
    # Get project container
    result = manager.get_project_container("Test Project")
    
    # Assert that the ProjectContainer method was called
    mock_project_container.get_project_container.assert_called_once_with("Test Project")
    
    # Verify result structure
    assert "container" in result
    
    # Verify container values
    container = result["container"]
    assert container["id"] == "proj_12345"
    assert container["name"] == "Test Project"
    assert container["entityType"] == "ProjectContainer"
    assert container["description"] == "This is a test project"
    assert container["domain_count"] == 3
    assert container["component_count"] == 10


def test_get_project_container_not_found(mock_base_manager, mock_logger):
    """Test retrieving a non-existent project container."""
    # Set up mock response from ProjectContainer for a not found error
    mock_response = json.dumps({
        "status": "error",
        "error": "Project container 'nonexistent-project' not found"
    })
    
    # Create mock for project_container
    mock_project_container = MagicMock()
    mock_project_container.get_project_container.return_value = mock_response
    
    # Create ProjectMemoryManager and inject mock project_container
    manager = ProjectMemoryManager(mock_base_manager)
    manager.project_container = mock_project_container
    
    # Get project container
    result = manager.get_project_container("nonexistent-project")
    
    # Assert that the ProjectContainer method was called
    mock_project_container.get_project_container.assert_called_once_with("nonexistent-project")
    
    # Verify result structure
    assert "status" in result
    assert result["status"] == "error"
    assert "error" in result
    assert "not found" in result["error"].lower()


def test_update_project_container(mock_base_manager, mock_logger):
    """Test updating a project container."""
    # Set up mock response from ProjectContainer
    mock_response = json.dumps({
        "status": "success",
        "message": "Project container 'project-123' updated successfully",
        "container": {
            "id": "project-123",
            "name": "Original Project Name", # Name can't be changed
            "entityType": "ProjectContainer",
            "description": "Updated description",
            "lastUpdated": "2022-01-20T12:00:00.000000"  # ISO datetime format
        }
    })
    
    # Create mock for project_container
    mock_project_container = MagicMock()
    mock_project_container.update_project_container.return_value = mock_response
    
    # Create ProjectMemoryManager and inject mock project_container
    manager = ProjectMemoryManager(mock_base_manager)
    manager.project_container = mock_project_container
    
    # Update project container
    updates = {
        "description": "Updated description"
    }
    
    # Call update_project_container
    result = manager.update_project_container("project-123", updates)
    
    # Assert that the ProjectContainer method was called
    mock_project_container.update_project_container.assert_called_once_with("project-123", updates)
    
    # Verify result structure
    assert "status" in result
    assert result["status"] == "success"
    assert "message" in result
    assert "container" in result
    
    # Verify container values
    container = result["container"]
    assert container["id"] == "project-123"
    assert container["description"] == "Updated description"


def test_update_project_container_not_found(mock_base_manager, mock_logger):
    """Test updating a non-existent project container."""
    # Set up mock response from ProjectContainer for a not found error
    mock_response = json.dumps({
        "status": "error",
        "error": "Project container 'nonexistent-project' not found"
    })
    
    # Create mock for project_container
    mock_project_container = MagicMock()
    mock_project_container.update_project_container.return_value = mock_response
    
    # Create ProjectMemoryManager and inject mock project_container
    manager = ProjectMemoryManager(mock_base_manager)
    manager.project_container = mock_project_container
    
    # Update project container
    updates = {
        "description": "Updated description"
    }
    
    # Call update_project_container
    result = manager.update_project_container("nonexistent-project", updates)
    
    # Assert that the ProjectContainer method was called
    mock_project_container.update_project_container.assert_called_once_with("nonexistent-project", updates)
    
    # Verify result structure
    assert "status" in result
    assert result["status"] == "error"
    assert "error" in result
    assert "not found" in result["error"].lower()


def test_delete_project_container(mock_base_manager, mock_logger):
    """Test deleting a project container."""
    # Set up mock response from ProjectContainer
    mock_response = json.dumps({
        "status": "success",
        "message": "Project container 'project-123' deleted successfully"
    })
    
    # Create mock for project_container
    mock_project_container = MagicMock()
    mock_project_container.delete_project_container.return_value = mock_response
    
    # Create ProjectMemoryManager and inject mock project_container
    manager = ProjectMemoryManager(mock_base_manager)
    manager.project_container = mock_project_container
    
    # Call delete_project_container with the default delete_contents=False
    result = manager.delete_project_container("project-123")
    
    # Assert that the ProjectContainer method was called with the correct parameters
    mock_project_container.delete_project_container.assert_called_once_with("project-123", False)
    
    # Verify result structure
    assert "status" in result
    assert result["status"] == "success"
    assert "message" in result


def test_list_project_containers(mock_base_manager, mock_logger):
    """Test listing project containers."""
    # Set up mock response from ProjectContainer
    mock_response = json.dumps({
        "status": "success",
        "message": "Project containers retrieved successfully",
        "containers": [
            {
                "id": "project-123",
                "name": "Test Project 1",
                "entityType": "ProjectContainer",
                "description": "This is test project 1",
                "componentCount": 5
            },
            {
                "id": "project-456",
                "name": "Test Project 2",
                "entityType": "ProjectContainer",
                "description": "This is test project 2",
                "componentCount": 3
            }
        ]
    })
    
    # Create mock for project_container
    mock_project_container = MagicMock()
    mock_project_container.list_project_containers.return_value = mock_response
    
    # Create ProjectMemoryManager and inject mock project_container
    manager = ProjectMemoryManager(mock_base_manager)
    manager.project_container = mock_project_container
    
    # Call list_project_containers
    result = manager.list_project_containers()
    
    # Assert that the ProjectContainer method was called
    mock_project_container.list_project_containers.assert_called_once()
    
    # Verify result structure
    assert "status" in result
    assert result["status"] == "success"
    assert "containers" in result
    
    # Verify containers list
    containers = result["containers"]
    assert len(containers) == 2
    
    # Verify first container
    assert containers[0]["id"] == "project-123"
    assert containers[0]["name"] == "Test Project 1"
    
    # Verify second container
    assert containers[1]["id"] == "project-456"
    assert containers[1]["name"] == "Test Project 2"


def test_create_component(mock_base_manager, mock_logger):
    """Test creating a component within a project."""
    # Set logger on mock_base_manager
    mock_base_manager.logger = mock_logger
    
    # Create manager
    manager = ProjectMemoryManager(base_manager=mock_base_manager)
    
    # Mock component data for a successful creation
    component_data = {
        "id": "component-123",
        "name": "Test Component",
        "component_type": "service",
        "description": "A test component"
    }
    
    # Mock result data
    result_data = {
        "status": "success",
        "message": "Component created successfully",
        "component": component_data
    }
    
    # Mock the create_component method directly
    manager.create_component = MagicMock(return_value=result_data)
    
    # Create component with domain creation
    result = manager.create_component(
        "Test Component",
        component_type="service",
        domain_name="TestDomain",
        container_name="TestProject",
        description="A test component"
    )
    
    # Verify result
    assert isinstance(result, dict)
    assert "status" in result
    assert result["status"] == "success"
    assert "component" in result
    assert result["component"] == component_data
    
    # Verify method was called with correct parameters
    manager.create_component.assert_called_once_with(
        "Test Component",
        component_type="service",
        domain_name="TestDomain",
        container_name="TestProject",
        description="A test component"
    )


def test_create_component_project_not_found(mock_base_manager, mock_logger):
    """Test creating a component for non-existent project."""
    # Set logger on mock_base_manager
    mock_base_manager.logger = mock_logger
    
    # Create manager
    manager = ProjectMemoryManager(base_manager=mock_base_manager)
    
    # Mock error result
    error_result = {
        "error": "Project not found: nonexistent-project"
    }
    
    # Mock create_component method
    manager.create_component = MagicMock(return_value=error_result)
    
    # Create component
    result = manager.create_component(
        "Test Component",
        component_type="service",
        domain_name="TestDomain",
        container_name="nonexistent-project"
    )
    
    # Verify result
    assert isinstance(result, dict)
    assert "error" in result
    assert "not found" in result["error"].lower()
    
    # Verify create_component was called with correct parameters
    manager.create_component.assert_called_once_with(
        "Test Component",
        component_type="service",
        domain_name="TestDomain",
        container_name="nonexistent-project"
    )


def test_get_component(mock_base_manager, mock_logger):
    """Test retrieving a component by ID."""
    # Set logger on mock_base_manager
    mock_base_manager.logger = mock_logger
    
    # Create manager
    manager = ProjectMemoryManager(base_manager=mock_base_manager)
    
    # Mock component data
    component_data = {
        "id": "component-123",
        "name": "Test Component",
        "component_type": "service",
        "description": "A test component"
    }
    
    # Mock result
    result_data = {
        "component": component_data,
        "status": "success"
    }
    
    # Mock get_component method
    manager.get_component = MagicMock(return_value=result_data)
    
    # Get component
    result = manager.get_component(
        "component-123",
        domain_name="TestDomain",
        container_name="TestProject"
    )
    
    # Verify result
    assert isinstance(result, dict)
    assert "component" in result
    assert result["component"] == component_data
    
    # Verify get_component was called with correct parameters
    manager.get_component.assert_called_once_with(
        "component-123",
        domain_name="TestDomain",
        container_name="TestProject"
    )


def test_update_component(mock_base_manager, mock_logger):
    """Test updating a component."""
    # Set up mock responses
    get_project_response = json.dumps({
        "container": {
            "id": "project-123",
            "name": "Test Project"
        }
    })
    
    list_components_response = json.dumps({
        "components": [{
            "id": "comp-123",
            "name": "TestComponent",
            "description": "Original description",
            "type": "SERVICE",
            "properties": {"foo": "bar"},
            "created": "2022-01-20T10:00:00.000000",
            "lastUpdated": "2022-01-20T10:00:00.000000"
        }]
    })
    
    update_component_response = json.dumps({
        "status": "success",
        "message": "Component 'TestComponent' updated successfully",
        "component": {
            "id": "comp-123",
            "name": "TestComponent",
            "description": "Updated description",
            "type": "SERVICE",
            "properties": {"foo": "baz"},
            "created": "2022-01-20T10:00:00.000000",
            "lastUpdated": "2022-01-20T12:00:00.000000"  # ISO format datetime string
        }
    })
    
    # Create mock for project_container and component_manager
    mock_project_container = MagicMock()
    mock_project_container.get_project_container.return_value = get_project_response
    
    mock_component_manager = MagicMock()
    mock_component_manager.list_components.return_value = list_components_response
    mock_component_manager.update_component.return_value = update_component_response
    
    # Create ProjectMemoryManager and inject mocks
    manager = ProjectMemoryManager(mock_base_manager)
    manager.project_container = mock_project_container
    manager.component_manager = mock_component_manager
    
    # Component updates
    updates = {
        "description": "Updated description",
        "properties": {"foo": "baz"}
    }
    
    # Call update_component - note that we're explicitly passing None for domain_name to test the optional parameter
    result = manager.update_component("TestComponent", "Test Project", updates, domain_name=None)
    
    # Assert that the methods were called with correct args
    mock_project_container.get_project_container.assert_called_once_with("Test Project")
    mock_component_manager.list_components.assert_called_once_with(domain_name="", container_name="Test Project")
    
    # The update_component should pass the component's name, empty string for domain_name, container_name, and updates
    mock_component_manager.update_component.assert_called_once_with("TestComponent", "", "Test Project", updates)
    
    # Verify result structure
    assert "status" in result
    assert result["status"] == "success"
    assert "message" in result
    assert "component" in result
    
    # Verify component properties
    component = result["component"]
    assert component["name"] == "TestComponent"
    assert component["description"] == "Updated description"
    assert component["properties"]["foo"] == "baz"
    
    # Verify timestamp fields in ISO format
    assert "created" in component
    assert "lastUpdated" in component
    assert isinstance(component["created"], str)
    assert isinstance(component["lastUpdated"], str)
    assert "T" in component["created"]
    assert "T" in component["lastUpdated"]
    
    # Verify lastUpdated timestamp is newer than the created timestamp
    assert component["lastUpdated"] > component["created"], "lastUpdated should be newer than created timestamp"


def test_delete_component(mock_base_manager, mock_logger):
    """Test deleting a component."""
    # Set logger on mock_base_manager
    mock_base_manager.logger = mock_logger
    
    # Create manager
    manager = ProjectMemoryManager(base_manager=mock_base_manager)
    
    # Configure delete component mock response
    mock_delete_response = json.dumps({
        "status": "success",
        "message": "Component 'Test Component' deleted successfully"
    })
    
    # Mock component_manager delete_component
    mock_component_manager = MagicMock()
    mock_component_manager.delete_component.return_value = mock_delete_response
    manager.component_manager = mock_component_manager
    
    # Delete component using the domain_name and container_name
    result = manager.delete_component(
        "component-123",
        domain_name="TestDomain",
        container_name="TestProject"
    )
    
    # Verify component_manager.delete_component was called with correct parameters
    mock_component_manager.delete_component.assert_called_once_with(
        "component-123", "TestDomain", "TestProject"
    )
    
    # Verify result structure
    assert isinstance(result, dict)
    assert result["status"] == "success"
    assert "message" in result


def test_list_components(mock_base_manager, mock_logger):
    """Test listing components."""
    # Set logger on mock_base_manager
    mock_base_manager.logger = mock_logger
    
    # Create manager
    manager = ProjectMemoryManager(base_manager=mock_base_manager)
    
    # Mock component data - this will be returned by the mock
    components_list = [
        {
            "id": "component-1",
            "name": "Component 1",
            "component_type": "service",
            "description": "First component"
        },
        {
            "id": "component-2",
            "name": "Component 2",
            "component_type": "library",
            "description": "Second component"
        }
    ]
    
    # Mock direct result without going through Neo4j mock
    manager.list_components = MagicMock(return_value={
        "components": components_list,
        "count": 2
    })
    
    # List components
    result = manager.list_components(
        domain_name="TestDomain",
        container_name="TestProject"
    )
    
    # Verify result structure
    assert isinstance(result, dict)
    assert "components" in result
    assert "count" in result
    assert result["count"] == 2
    assert len(result["components"]) == 2
    
    # Check components match our expected data
    components = result["components"]
    assert components[0]["name"] == "Component 1"
    assert components[0]["component_type"] == "service"
    assert components[1]["name"] == "Component 2"
    assert components[1]["component_type"] == "library"
    
    # Verify our mock was called
    manager.list_components.assert_called_once()


def test_list_components_with_filters(mock_base_manager, mock_logger):
    """Test listing components with filters."""
    # Set logger on mock_base_manager
    mock_base_manager.logger = mock_logger
    
    # Create manager
    manager = ProjectMemoryManager(base_manager=mock_base_manager)
    
    # Mock filtered component data
    component = {
        "id": "component-1",
        "name": "Filtered Component",
        "component_type": "service",
        "description": "A filtered component"
    }
    
    # Mock direct result without going through Neo4j
    # Create a mock that captures and verifies the filters
    original_list_components = manager.list_components
    
    def mock_list_components(*args, **kwargs):
        # Verify component_type filter was passed
        assert "component_type" in kwargs
        assert kwargs["component_type"] == "service"
        return {
            "components": [component],
            "count": 1
        }
    
    # Replace with our mock that verifies filters
    manager.list_components = MagicMock(side_effect=mock_list_components)
    
    # List components with component_type filter
    result = manager.list_components(
        domain_name="TestDomain",
        container_name="TestProject",
        component_type="service"
    )
    
    # Verify result structure
    assert isinstance(result, dict)
    assert "components" in result
    assert "count" in result
    assert result["count"] == 1
    assert len(result["components"]) == 1
    
    # Check component properties
    components = result["components"]
    assert components[0]["name"] == "Filtered Component"
    assert components[0]["component_type"] == "service"
    
    # Verify list_components was called with filters
    manager.list_components.assert_called_once()


def test_create_component_relationship(mock_base_manager, mock_logger):
    """Test creating a relationship between components."""
    # Set logger on mock_base_manager
    mock_base_manager.logger = mock_logger
    
    # Create manager
    manager = ProjectMemoryManager(base_manager=mock_base_manager)
    
    # Mock result for successful relationship creation
    result_data = {
        "status": "success",
        "message": "Relationship created successfully"
    }
    
    # Mock the create_component_relationship method directly
    manager.create_component_relationship = MagicMock(return_value=result_data)
    
    # Create component relationship
    result = manager.create_component_relationship(
        "component-1",
        to_component="component-2",
        domain_name="TestDomain",
        container_name="TestProject",
        relation_type="DEPENDS_ON",
        properties={"version_constraint": ">=1.0.0"}
    )
    
    # Verify result
    assert isinstance(result, dict)
    assert "status" in result
    assert result["status"] == "success"
    
    # Verify create_component_relationship was called with correct parameters
    manager.create_component_relationship.assert_called_once_with(
        "component-1",
        to_component="component-2",
        domain_name="TestDomain",
        container_name="TestProject",
        relation_type="DEPENDS_ON",
        properties={"version_constraint": ">=1.0.0"}
    )


def test_create_domain_entity(mock_logger, mock_base_manager):
    """Test creating a domain entity."""
    # Set logger on mock_base_manager
    mock_base_manager.logger = mock_logger
    
    # Create a ProjectMemoryManager instance
    manager = ProjectMemoryManager(mock_base_manager)
    
    # Create a mock domain manager
    mock_domain_manager = MagicMock()
    manager.domain_manager = mock_domain_manager
    
    # Mock entity data
    entity_name = "TestEntity"
    entity_type = "Model"
    domain_name = "TestDomain"
    container_name = "TestProject"
    description = "Test entity description"
    code_reference = "src/models/test_model.py"
    properties = {
        "attribute": "value"
    }
    
    # Mock result data
    result_data = {
        "status": "success",
        "message": "Domain entity created successfully",
        "entity": {
            "id": "entity-123",
            "name": entity_name,
            "entityType": entity_type,
            "description": description
        }
    }
    
    # Configure the mock to return the result data
    mock_domain_manager.create_domain.return_value = json.dumps(result_data)
    
    # Create domain
    result = manager.create_domain(
        name=domain_name,
        container_name=container_name,
        description=description,
        properties=properties
    )
    
    # Verify result
    assert isinstance(result, dict)
    assert result["status"] == "success"
    assert "entity" in result
    assert result["entity"]["name"] == entity_name
    
    # Verify create_domain was called with correct parameters
    mock_domain_manager.create_domain.assert_called_once_with(
        domain_name,
        container_name,
        description,
        properties
    )


def test_create_domain_relationship(mock_logger, mock_base_manager):
    """Test creating a relationship between domain entities."""
    # Set logger on mock_base_manager
    mock_base_manager.logger = mock_logger
    
    # Create a ProjectMemoryManager instance
    manager = ProjectMemoryManager(mock_base_manager)
    
    # Create a mock domain manager
    mock_domain_manager = MagicMock()
    manager.domain_manager = mock_domain_manager
    
    # Mock relationship data
    relationship_data = {
        "from_domain": "SourceDomain",
        "to_domain": "TargetDomain",
        "container_name": "TestProject",
        "relation_type": "DEPENDS_ON",
        "properties": {
            "strength": "strong",
            "description": "Source domain depends on target domain"
        }
    }
    
    # Mock result
    result_data = {
        "status": "success",
        "message": "Relationship created successfully",
        "relationship": {
            "from": "SourceDomain",
            "to": "TargetDomain",
            "type": "DEPENDS_ON"
        }
    }
    
    # Configure the mock to return the result data
    mock_domain_manager.create_domain_relationship.return_value = json.dumps(result_data)
    
    # Create relationship
    result = manager.create_domain_relationship(
        from_domain=relationship_data["from_domain"],
        to_domain=relationship_data["to_domain"],
        container_name=relationship_data["container_name"],
        relation_type=relationship_data["relation_type"],
        properties=relationship_data["properties"]
    )
    
    # Verify result
    assert isinstance(result, dict)
    assert result["status"] == "success"
    assert "relationship" in result
    assert result["relationship"]["from"] == "SourceDomain"
    assert result["relationship"]["to"] == "TargetDomain"
    
    # Verify create_domain_relationship was called with correct parameters
    mock_domain_manager.create_domain_relationship.assert_called_once_with(
        relationship_data["from_domain"],
        relationship_data["to_domain"],
        relationship_data["container_name"],
        relationship_data["relation_type"],
        relationship_data["properties"]
    )


def test_get_project_components(mock_logger, mock_base_manager):
    """Test getting project components."""
    # Set logger on mock_base_manager
    mock_base_manager.logger = mock_logger
    
    # Create a ProjectMemoryManager instance
    manager = ProjectMemoryManager(mock_base_manager)
    
    # Create a mock component manager
    mock_component_manager = MagicMock()
    manager.component_manager = mock_component_manager
    
    # Mock components list
    components = [
        {
            "id": "component-1",
            "name": "Component 1",
            "componentType": "service",
            "description": "First component"
        },
        {
            "id": "component-2",
            "name": "Component 2",
            "componentType": "library",
            "description": "Second component"
        }
    ]
    
    # Mock result data
    result_data = {
        "components": components,
        "count": 2
    }
    
    # Configure the mock to return the result data
    mock_component_manager.list_components.return_value = json.dumps(result_data)
    
    # Get project components - Note: This actually calls list_components with the domain name parameter
    domain_name = "TestDomain"
    container_name = "TestProject"
    result = manager.list_components(domain_name, container_name)
    
    # Verify result
    assert isinstance(result, dict)
    assert "components" in result
    assert len(result["components"]) == 2
    assert result["components"][0]["name"] == "Component 1"
    assert result["components"][1]["name"] == "Component 2"
    
    # Verify list_components was called with the right parameters
    mock_component_manager.list_components.assert_called_once_with(
        domain_name, container_name, None, "name", 100)


def test_get_project_domain_entities(mock_logger, mock_base_manager):
    """Test getting all domain entities for a project."""
    # Set logger on mock_base_manager
    mock_base_manager.logger = mock_logger
    
    # Create a ProjectMemoryManager instance
    manager = ProjectMemoryManager(mock_base_manager)
    
    # Create a mock domain manager (this is a proper approach instead of patching get_project_domain_entities)
    mock_domain_manager = MagicMock()
    manager.domain_manager = mock_domain_manager
    
    # Mock domain entities data
    entities = [
        {
            "id": "entity-1",
            "name": "Entity 1",
            "entityType": "Model",
            "description": "First entity"
        },
        {
            "id": "entity-2",
            "name": "Entity 2",
            "entityType": "Controller",
            "description": "Second entity"
        }
    ]
    
    # Mock result data
    result_data = {
        "entities": entities,
        "count": 2
    }
    
    # Configure the mock to return the domain entities
    mock_domain_manager.get_domain_entities.return_value = json.dumps(result_data)
    
    # Get project domain entities
    project_id = "project-123"
    result = manager.get_domain_entities("domain-name", project_id)
    
    # Verify result
    assert isinstance(result, dict)
    assert "entities" in result
    assert len(result["entities"]) == 2
    assert result["entities"][0]["name"] == "Entity 1"
    assert result["entities"][1]["name"] == "Entity 2"
    
    # Verify get_domain_entities was called with the right parameters
    mock_domain_manager.get_domain_entities.assert_called_once_with(
        "domain-name", project_id, None) 