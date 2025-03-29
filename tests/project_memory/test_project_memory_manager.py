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
    """Test creating a project container."""
    # Set logger on mock_base_manager
    mock_base_manager.logger = mock_logger
    
    # Create manager
    manager = ProjectMemoryManager(base_manager=mock_base_manager)
    
    # Mock entity data
    entity_data = {
        "id": "proj_12345",
        "name": "Test Project",
        "entityType": "ProjectContainer",
        "description": "This is a test project"
    }
    
    # Create mock entity
    mock_entity = MagicMock()
    mock_entity.items.return_value = entity_data.items()
    
    # Mock record
    mock_record = MagicMock()
    mock_record.get.return_value = mock_entity
    
    # Mock safe_execute_query
    mock_base_manager.safe_execute_query.return_value = ([mock_record], None)
    
    # Create project container
    result = manager.create_project_container(
        name="Test Project",
        description="This is a test project"
    )
    
    # Verify result is a dictionary
    assert isinstance(result, dict)
    
    # Verify content of the result dictionary
    assert "status" in result
    assert result["status"] == "success"
    assert "message" in result
    assert "container" in result
    assert result["container"] == entity_data
    
    # Verify safe_execute_query was called
    mock_base_manager.safe_execute_query.assert_called_once()


def test_create_project_container_with_default_values(mock_base_manager, mock_logger):
    """Test creating a project container with default values."""
    # Set logger on mock_base_manager
    mock_base_manager.logger = mock_logger
    
    # Create manager
    manager = ProjectMemoryManager(base_manager=mock_base_manager)
    
    # Mock entity data
    entity_data = {
        "id": "proj_12345",
        "name": "Minimal Project",
        "entityType": "ProjectContainer"
    }
    
    # Create mock entity
    mock_entity = MagicMock()
    mock_entity.items.return_value = entity_data.items()
    
    # Mock record
    mock_record = MagicMock()
    mock_record.get.return_value = mock_entity
    
    # Mock safe_execute_query
    mock_base_manager.safe_execute_query.return_value = ([mock_record], None)
    
    # Create project container with minimal parameters
    result = manager.create_project_container(name="Minimal Project")
    
    # Verify result
    assert isinstance(result, dict)
    assert result["status"] == "success"
    assert "message" in result
    assert "container" in result
    assert result["container"] == entity_data
    
    # Verify safe_execute_query was called
    mock_base_manager.safe_execute_query.assert_called_once()
    
    # Check parameters
    call_args = mock_base_manager.safe_execute_query.call_args
    assert "name" in call_args[0][1]
    assert call_args[0][1]["name"] == "Minimal Project"


def test_get_project_container(mock_base_manager, mock_logger):
    """Test retrieving a project container by ID."""
    # Set logger on mock_base_manager
    mock_base_manager.logger = mock_logger
    
    # Create manager
    manager = ProjectMemoryManager(base_manager=mock_base_manager)
    
    # Mock entity data
    entity_data = {
        "id": "proj_12345",
        "name": "Test Project",
        "entityType": "ProjectContainer",
        "description": "This is a test project"
    }
    
    # Create expected result with counts added
    expected_container = entity_data.copy()
    expected_container["domain_count"] = 3
    expected_container["component_count"] = 10
    
    # Create mock entity
    mock_entity = MagicMock()
    mock_entity.items.return_value = entity_data.items()
    
    # Mock record
    mock_record = MagicMock()
    mock_record.get.return_value = mock_entity
    
    # Create mock domain and component count records
    domain_record = MagicMock()
    domain_record.__getitem__.return_value = 3
    
    component_record = MagicMock()
    component_record.__getitem__.return_value = 10
    
    # Set up the mock to return different values for each call
    mock_base_manager.safe_execute_query.side_effect = [
        ([mock_record], None),        # First call - project query
        ([domain_record], None),      # Second call - domain count
        ([component_record], None)    # Third call - component count
    ]
    
    # Get project container
    result = manager.get_project_container("Test Project")
    
    # Verify result
    assert "container" in result
    assert result["container"] == expected_container
    
    # Verify safe_execute_query was called 3 times
    assert mock_base_manager.safe_execute_query.call_count == 3


def test_get_project_container_not_found(mock_base_manager, mock_logger):
    """Test retrieving a non-existent project container."""
    # Set logger on mock_base_manager
    mock_base_manager.logger = mock_logger
    
    # Create manager
    manager = ProjectMemoryManager(base_manager=mock_base_manager)
    
    # Mock safe_execute_query to return empty results (no project found)
    mock_base_manager.safe_execute_query.return_value = ([], None)
    
    # Get project container
    result = manager.get_project_container("nonexistent-project")
    
    # Verify result
    assert "error" in result
    assert "not found" in result["error"].lower()
    
    # Verify safe_execute_query was called with the right project name
    mock_base_manager.safe_execute_query.assert_called_once()
    call_args = mock_base_manager.safe_execute_query.call_args
    assert call_args[0][1]["name"] == "nonexistent-project"


def test_update_project_container(mock_component_managers, mock_logger):
    """Test updating a project container."""
    # Create manager
    manager = ProjectMemoryManager(**mock_component_managers, logger=mock_logger)
    
    # Mock entity retrieval and update
    mock_component_managers["entity_manager"].get_entity.return_value = json.dumps({
        "id": "project-123",
        "name": "Original Project Name",
        "type": "ProjectContainer",
        "metadata": {
            "description": "Original description",
            "tags": ["original", "tags"]
        }
    })
    
    mock_component_managers["entity_manager"].update_entity.return_value = json.dumps({
        "status": "success",
        "id": "project-123"
    })
    
    # Update project container
    project_data = {
        "id": "project-123",
        "name": "Updated Project Name",
        "description": "Updated description",
        "tags": ["updated", "tags"]
    }
    
    result = manager.update_project_container(project_data)
    
    # Verify result
    result_obj = json.loads(result)
    assert result_obj["status"] == "success"
    
    # Verify entity_manager calls
    mock_component_managers["entity_manager"].get_entity.assert_called_once_with("project-123")
    mock_component_managers["entity_manager"].update_entity.assert_called_once()
    
    # Check update data
    update_data = mock_component_managers["entity_manager"].update_entity.call_args[0][0]
    assert update_data["name"] == "Updated Project Name"
    assert update_data["metadata"]["description"] == "Updated description"
    assert update_data["metadata"]["tags"] == ["updated", "tags"]


def test_update_project_container_not_found(mock_component_managers, mock_logger):
    """Test updating a non-existent project container."""
    # Create manager
    manager = ProjectMemoryManager(**mock_component_managers, logger=mock_logger)
    
    # Mock entity retrieval - not found
    mock_component_managers["entity_manager"].get_entity.return_value = None
    
    # Update project container
    project_data = {
        "id": "nonexistent-project",
        "name": "Updated Project Name"
    }
    
    result = manager.update_project_container(project_data)
    
    # Verify result
    result_obj = json.loads(result)
    assert result_obj["status"] == "error"
    assert "not found" in result_obj["message"].lower()
    
    # Verify entity_manager calls
    mock_component_managers["entity_manager"].get_entity.assert_called_once_with("nonexistent-project")
    mock_component_managers["entity_manager"].update_entity.assert_not_called()


def test_delete_project_container(mock_component_managers, mock_logger):
    """Test deleting a project container."""
    # Create manager
    manager = ProjectMemoryManager(**mock_component_managers, logger=mock_logger)
    
    # Mock entity deletion
    mock_component_managers["entity_manager"].delete_entity.return_value = json.dumps({
        "status": "success"
    })
    
    # Delete project container
    result = manager.delete_project_container("project-123")
    
    # Verify result
    result_obj = json.loads(result)
    assert result_obj["status"] == "success"
    
    # Verify entity_manager was called correctly
    mock_component_managers["entity_manager"].delete_entity.assert_called_once_with("project-123")


def test_list_project_containers(mock_component_managers, mock_logger):
    """Test listing project containers."""
    # Create manager
    manager = ProjectMemoryManager(**mock_component_managers, logger=mock_logger)
    
    # Mock entity search
    mock_projects = [
        {
            "id": "project-1",
            "name": "Project 1",
            "type": "ProjectContainer",
            "metadata": {
                "description": "First project",
                "tags": ["tag1", "tag2"]
            }
        },
        {
            "id": "project-2",
            "name": "Project 2",
            "type": "ProjectContainer",
            "metadata": {
                "description": "Second project",
                "tags": ["tag3", "tag4"]
            }
        }
    ]
    mock_component_managers["entity_manager"].get_entities_by_type.return_value = json.dumps(mock_projects)
    
    # List project containers
    result = manager.list_project_containers()
    
    # Verify result
    result_list = json.loads(result)
    assert len(result_list) == 2
    assert result_list[0]["name"] == "Project 1"
    assert result_list[1]["name"] == "Project 2"
    
    # Verify entity_manager was called correctly
    mock_component_managers["entity_manager"].get_entities_by_type.assert_called_once_with("ProjectContainer")


def test_create_component(mock_component_managers, mock_logger):
    """Test creating a component within a project."""
    # Create manager
    manager = ProjectMemoryManager(**mock_component_managers, logger=mock_logger)
    
    # Mock project retrieval
    mock_component_managers["entity_manager"].get_entity.return_value = json.dumps({
        "id": "project-123",
        "name": "Test Project",
        "type": "ProjectContainer"
    })
    
    # Mock entity and relation creation
    mock_component_managers["entity_manager"].create_entity.return_value = json.dumps({
        "id": "component-123",
        "name": "Test Component"
    })
    
    mock_component_managers["relation_manager"].create_relationship.return_value = json.dumps({
        "status": "success"
    })
    
    # Create component
    component_data = {
        "project_id": "project-123",
        "name": "Test Component",
        "component_type": "service",
        "description": "A test component",
        "properties": {
            "language": "python",
            "version": "1.0"
        },
        "tags": ["test", "component"]
    }
    
    result = manager.create_component(component_data)
    
    # Verify result
    result_obj = json.loads(result)
    assert result_obj["status"] == "success"
    assert "component_id" in result_obj
    
    # Verify entity_manager and relation_manager calls
    mock_component_managers["entity_manager"].get_entity.assert_called_once_with("project-123")
    mock_component_managers["entity_manager"].create_entity.assert_called_once()
    mock_component_managers["relation_manager"].create_relationship.assert_called_once()
    
    # Check entity data
    entity_data = mock_component_managers["entity_manager"].create_entity.call_args[0][0]
    assert entity_data["name"] == "Test Component"
    assert entity_data["type"] == "Component"
    assert entity_data["metadata"]["component_type"] == "service"
    assert entity_data["metadata"]["properties"]["language"] == "python"


def test_create_component_project_not_found(mock_component_managers, mock_logger):
    """Test creating a component for non-existent project."""
    # Create manager
    manager = ProjectMemoryManager(**mock_component_managers, logger=mock_logger)
    
    # Mock project retrieval - not found
    mock_component_managers["entity_manager"].get_entity.return_value = None
    
    # Create component
    component_data = {
        "project_id": "nonexistent-project",
        "name": "Test Component",
        "component_type": "service"
    }
    
    result = manager.create_component(component_data)
    
    # Verify result
    result_obj = json.loads(result)
    assert result_obj["status"] == "error"
    assert "project not found" in result_obj["message"].lower()
    
    # Verify entity_manager calls
    mock_component_managers["entity_manager"].get_entity.assert_called_once_with("nonexistent-project")
    mock_component_managers["entity_manager"].create_entity.assert_not_called()
    mock_component_managers["relation_manager"].create_relationship.assert_not_called()


def test_get_component(mock_component_managers, mock_logger):
    """Test retrieving a component by ID."""
    # Create manager
    manager = ProjectMemoryManager(**mock_component_managers, logger=mock_logger)
    
    # Mock component retrieval
    mock_component_managers["entity_manager"].get_entity.return_value = json.dumps({
        "id": "component-123",
        "name": "Test Component",
        "type": "Component",
        "metadata": {
            "component_type": "service",
            "description": "A test component",
            "properties": {
                "language": "python"
            }
        }
    })
    
    # Get component
    result = manager.get_component("component-123")
    
    # Verify result
    result_obj = json.loads(result)
    assert result_obj["id"] == "component-123"
    assert result_obj["name"] == "Test Component"
    assert result_obj["metadata"]["component_type"] == "service"
    
    # Verify entity_manager was called correctly
    mock_component_managers["entity_manager"].get_entity.assert_called_once_with("component-123")


def test_update_component(mock_component_managers, mock_logger):
    """Test updating a component."""
    # Create manager
    manager = ProjectMemoryManager(**mock_component_managers, logger=mock_logger)
    
    # Mock component retrieval and update
    mock_component_managers["entity_manager"].get_entity.return_value = json.dumps({
        "id": "component-123",
        "name": "Original Component Name",
        "type": "Component",
        "metadata": {
            "component_type": "library",
            "description": "Original description",
            "properties": {
                "language": "javascript"
            }
        }
    })
    
    mock_component_managers["entity_manager"].update_entity.return_value = json.dumps({
        "status": "success",
        "id": "component-123"
    })
    
    # Update component
    component_data = {
        "id": "component-123",
        "name": "Updated Component Name",
        "component_type": "service",
        "description": "Updated description",
        "properties": {
            "language": "typescript",
            "version": "2.0"
        }
    }
    
    result = manager.update_component(component_data)
    
    # Verify result
    result_obj = json.loads(result)
    assert result_obj["status"] == "success"
    
    # Verify entity_manager calls
    mock_component_managers["entity_manager"].get_entity.assert_called_once_with("component-123")
    mock_component_managers["entity_manager"].update_entity.assert_called_once()
    
    # Check update data
    update_data = mock_component_managers["entity_manager"].update_entity.call_args[0][0]
    assert update_data["name"] == "Updated Component Name"
    assert update_data["metadata"]["component_type"] == "service"
    assert update_data["metadata"]["description"] == "Updated description"
    assert update_data["metadata"]["properties"]["language"] == "typescript"
    assert update_data["metadata"]["properties"]["version"] == "2.0"


def test_delete_component(mock_component_managers, mock_logger):
    """Test deleting a component."""
    # Create manager
    manager = ProjectMemoryManager(**mock_component_managers, logger=mock_logger)
    
    # Mock entity deletion
    mock_component_managers["entity_manager"].delete_entity.return_value = json.dumps({
        "status": "success"
    })
    
    # Delete component
    result = manager.delete_component("component-123")
    
    # Verify result
    result_obj = json.loads(result)
    assert result_obj["status"] == "success"
    
    # Verify entity_manager was called correctly
    mock_component_managers["entity_manager"].delete_entity.assert_called_once_with("component-123")


def test_list_components(mock_component_managers, mock_logger):
    """Test listing components."""
    # Create manager
    manager = ProjectMemoryManager(**mock_component_managers, logger=mock_logger)
    
    # Mock entity search
    mock_components = [
        {
            "id": "component-1",
            "name": "Component 1",
            "type": "Component",
            "metadata": {
                "component_type": "service",
                "description": "First component"
            }
        },
        {
            "id": "component-2",
            "name": "Component 2",
            "type": "Component",
            "metadata": {
                "component_type": "library",
                "description": "Second component"
            }
        }
    ]
    mock_component_managers["entity_manager"].get_entities_by_type.return_value = json.dumps(mock_components)
    
    # List components (no filters)
    result = manager.list_components()
    
    # Verify result
    result_list = json.loads(result)
    assert len(result_list) == 2
    assert result_list[0]["name"] == "Component 1"
    assert result_list[1]["name"] == "Component 2"
    
    # Verify entity_manager was called correctly
    mock_component_managers["entity_manager"].get_entities_by_type.assert_called_once_with("Component")


def test_list_components_with_filters(mock_component_managers, mock_logger):
    """Test listing components with filters."""
    # Create manager
    manager = ProjectMemoryManager(**mock_component_managers, logger=mock_logger)
    
    # Mock entity search
    mock_filtered_components = [
        {
            "id": "component-1",
            "name": "Filtered Component",
            "type": "Component",
            "metadata": {
                "component_type": "service",
                "project_id": "project-123"
            }
        }
    ]
    mock_component_managers["entity_manager"].get_entities_by_type.return_value = json.dumps(mock_filtered_components)
    
    # List components with filters
    result = manager.list_components(
        project_id="project-123",
        component_type="service"
    )
    
    # Verify result
    result_list = json.loads(result)
    assert len(result_list) == 1
    assert result_list[0]["name"] == "Filtered Component"
    assert result_list[0]["metadata"]["component_type"] == "service"
    
    # Verify entity_manager was called correctly
    mock_component_managers["entity_manager"].get_entities_by_type.assert_called_once_with("Component")


def test_create_component_relationship(mock_component_managers, mock_logger):
    """Test creating a relationship between components."""
    # Create manager
    manager = ProjectMemoryManager(**mock_component_managers, logger=mock_logger)
    
    # Mock source and target component retrieval
    mock_component_managers["entity_manager"].get_entity.side_effect = [
        json.dumps({
            "id": "component-1",
            "name": "Source Component",
            "type": "Component"
        }),
        json.dumps({
            "id": "component-2",
            "name": "Target Component",
            "type": "Component"
        })
    ]
    
    # Mock relation creation
    mock_component_managers["relation_manager"].create_relationship.return_value = json.dumps({
        "status": "success"
    })
    
    # Create component relationship
    relationship_data = {
        "source_id": "component-1",
        "target_id": "component-2",
        "relationship_type": "DEPENDS_ON",
        "properties": {
            "version_constraint": ">=1.0.0"
        }
    }
    
    result = manager.create_component_relationship(relationship_data)
    
    # Verify result
    result_obj = json.loads(result)
    assert result_obj["status"] == "success"
    
    # Verify entity_manager and relation_manager calls
    assert mock_component_managers["entity_manager"].get_entity.call_count == 2
    mock_component_managers["relation_manager"].create_relationship.assert_called_once()
    
    # Check relationship data
    rel_data = mock_component_managers["relation_manager"].create_relationship.call_args[0][0]
    assert rel_data["from_entity"] == "component-1"
    assert rel_data["to_entity"] == "component-2"
    assert rel_data["relation_type"] == "DEPENDS_ON"
    assert rel_data["properties"]["version_constraint"] == ">=1.0.0"


def test_create_domain_entity(mock_component_managers, mock_logger):
    """Test creating a domain entity."""
    # Create manager
    manager = ProjectMemoryManager(**mock_component_managers, logger=mock_logger)
    
    # Mock project and component retrieval
    mock_component_managers["entity_manager"].get_entity.side_effect = [
        json.dumps({
            "id": "project-123",
            "name": "Test Project",
            "type": "ProjectContainer"
        }),
        json.dumps({
            "id": "component-123",
            "name": "Test Component",
            "type": "Component"
        })
    ]
    
    # Mock entity and relation creation
    mock_component_managers["entity_manager"].create_entity.return_value = json.dumps({
        "id": "domain-entity-123",
        "name": "Test Domain Entity"
    })
    
    mock_component_managers["relation_manager"].create_relationship.return_value = json.dumps({
        "status": "success"
    })
    
    # Create domain entity
    entity_data = {
        "project_id": "project-123",
        "component_id": "component-123",
        "name": "Test Domain Entity",
        "entity_type": "model",
        "description": "A test domain entity",
        "code_reference": "src/models/TestModel.ts",
        "properties": {
            "attributes": ["id", "name", "description"],
            "relationships": ["belongs_to_component"]
        }
    }
    
    result = manager.create_domain_entity(entity_data)
    
    # Verify result
    result_obj = json.loads(result)
    assert result_obj["status"] == "success"
    assert "entity_id" in result_obj
    
    # Verify entity_manager and relation_manager calls
    assert mock_component_managers["entity_manager"].get_entity.call_count == 2
    mock_component_managers["entity_manager"].create_entity.assert_called_once()
    assert mock_component_managers["relation_manager"].create_relationship.call_count == 2  # Project and Component relations
    
    # Check entity data
    entity_data = mock_component_managers["entity_manager"].create_entity.call_args[0][0]
    assert entity_data["name"] == "Test Domain Entity"
    assert entity_data["type"] == "DomainEntity"
    assert entity_data["metadata"]["entity_type"] == "model"
    assert entity_data["metadata"]["code_reference"] == "src/models/TestModel.ts"
    assert "id" in entity_data["metadata"]["properties"]["attributes"]


def test_create_domain_relationship(mock_component_managers, mock_logger):
    """Test creating a relationship between domain entities."""
    # Create manager
    manager = ProjectMemoryManager(**mock_component_managers, logger=mock_logger)
    
    # Mock source and target domain entity retrieval
    mock_component_managers["entity_manager"].get_entity.side_effect = [
        json.dumps({
            "id": "domain-entity-1",
            "name": "Source Domain Entity",
            "type": "DomainEntity"
        }),
        json.dumps({
            "id": "domain-entity-2",
            "name": "Target Domain Entity",
            "type": "DomainEntity"
        })
    ]
    
    # Mock relation creation
    mock_component_managers["relation_manager"].create_relationship.return_value = json.dumps({
        "status": "success"
    })
    
    # Create domain relationship
    relationship_data = {
        "source_id": "domain-entity-1",
        "target_id": "domain-entity-2",
        "relationship_type": "HAS_MANY",
        "properties": {
            "foreign_key": "source_id",
            "cascade_delete": True
        }
    }
    
    result = manager.create_domain_relationship(relationship_data)
    
    # Verify result
    result_obj = json.loads(result)
    assert result_obj["status"] == "success"
    
    # Verify entity_manager and relation_manager calls
    assert mock_component_managers["entity_manager"].get_entity.call_count == 2
    mock_component_managers["relation_manager"].create_relationship.assert_called_once()
    
    # Check relationship data
    rel_data = mock_component_managers["relation_manager"].create_relationship.call_args[0][0]
    assert rel_data["from_entity"] == "domain-entity-1"
    assert rel_data["to_entity"] == "domain-entity-2"
    assert rel_data["relation_type"] == "HAS_MANY"
    assert rel_data["properties"]["foreign_key"] == "source_id"


def test_get_project_components(mock_component_managers, mock_logger):
    """Test getting all components for a project."""
    # Create manager
    manager = ProjectMemoryManager(**mock_component_managers, logger=mock_logger)
    
    # Mock project retrieval
    mock_component_managers["entity_manager"].get_entity.return_value = json.dumps({
        "id": "project-123",
        "name": "Test Project",
        "type": "ProjectContainer"
    })
    
    # Mock relation retrieval
    mock_relations = [
        {
            "r": {"type": "HAS_COMPONENT", "properties": {}},
            "component": {
                "id": "component-1", 
                "properties": {
                    "name": "Component 1",
                    "type": "Component",
                    "metadata": {
                        "component_type": "service"
                    }
                }
            }
        },
        {
            "r": {"type": "HAS_COMPONENT", "properties": {}},
            "component": {
                "id": "component-2", 
                "properties": {
                    "name": "Component 2",
                    "type": "Component",
                    "metadata": {
                        "component_type": "library"
                    }
                }
            }
        }
    ]
    mock_component_managers["relation_manager"].get_relationships.return_value = json.dumps(mock_relations)
    
    # Get project components
    result = manager.get_project_components("project-123")
    
    # Verify result
    result_list = json.loads(result)
    assert len(result_list) == 2
    assert result_list[0]["name"] == "Component 1"
    assert result_list[1]["name"] == "Component 2"
    
    # Verify entity_manager and relation_manager calls
    mock_component_managers["entity_manager"].get_entity.assert_called_once_with("project-123")
    mock_component_managers["relation_manager"].get_relationships.assert_called_once()


def test_get_project_domain_entities(mock_component_managers, mock_logger):
    """Test getting all domain entities for a project."""
    # Create manager
    manager = ProjectMemoryManager(**mock_component_managers, logger=mock_logger)
    
    # Mock project retrieval
    mock_component_managers["entity_manager"].get_entity.return_value = json.dumps({
        "id": "project-123",
        "name": "Test Project",
        "type": "ProjectContainer"
    })
    
    # Mock relation retrieval
    mock_relations = [
        {
            "r": {"type": "HAS_DOMAIN_ENTITY", "properties": {}},
            "domain_entity": {
                "id": "domain-entity-1", 
                "properties": {
                    "name": "Domain Entity 1",
                    "type": "DomainEntity",
                    "metadata": {
                        "entity_type": "model"
                    }
                }
            }
        },
        {
            "r": {"type": "HAS_DOMAIN_ENTITY", "properties": {}},
            "domain_entity": {
                "id": "domain-entity-2", 
                "properties": {
                    "name": "Domain Entity 2",
                    "type": "DomainEntity",
                    "metadata": {
                        "entity_type": "controller"
                    }
                }
            }
        }
    ]
    mock_component_managers["relation_manager"].get_relationships.return_value = json.dumps(mock_relations)
    
    # Get project domain entities
    result = manager.get_project_domain_entities("project-123")
    
    # Verify result
    result_list = json.loads(result)
    assert len(result_list) == 2
    assert result_list[0]["name"] == "Domain Entity 1"
    assert result_list[1]["name"] == "Domain Entity 2"
    
    # Verify entity_manager and relation_manager calls
    mock_component_managers["entity_manager"].get_entity.assert_called_once_with("project-123")
    mock_component_managers["relation_manager"].get_relationships.assert_called_once() 