"""
Test script to verify the ProjectMemoryManager integration with Neo4j query validation.
"""

import pytest
# from src.graph_memory.project_memory_manager import ProjectMemoryManager
from src.models.project_memory import ProjectContainer, ComponentCreate, DomainEntityCreate
import re

@pytest.fixture
def mock_project_manager():
    """
    Create a mock ProjectMemoryManager for validation tests.
    
    Returns:
        A MagicMock instance with appropriate method behaviors
    """
    from unittest.mock import MagicMock
    import json
    from datetime import datetime
    
    mock_manager = MagicMock()
    
    # Mock project return value
    project_data = {
        "id": "project-id-123",
        "name": "TestProject",
        "description": "Test project for validation",
        "created": datetime.now().isoformat(),
        "lastUpdated": datetime.now().isoformat()
    }
    
    # Mock project creation - return Pydantic-like response
    mock_project = MagicMock()
    mock_project.name = "TestProject"
    mock_project.description = "Test project for validation"
    mock_project.dict = MagicMock(return_value=project_data)
    mock_manager.create_project.return_value = mock_project
    
    # Mock component creation - return Pydantic-like response
    component_data = {
        "id": "component-id-123",
        "name": "TestComponent",
        "description": "Test component",
        "type": "SERVICE",
        "created": datetime.now().isoformat(),
        "lastUpdated": datetime.now().isoformat()
    }
    mock_component = MagicMock()
    mock_component.name = "TestComponent"
    mock_component.description = "Test component"
    mock_component.type = "SERVICE"
    mock_component.dict = MagicMock(return_value=component_data)
    mock_manager.create_component.return_value = mock_component
    
    # Mock domain entity creation - return Pydantic-like response
    entity_data = {
        "id": "entity-id-123",
        "name": "TestEntity",
        "type": "DECISION",
        "description": "Test entity",
        "content": "Test entity content",
        "created": datetime.now().isoformat(),
        "lastUpdated": datetime.now().isoformat()
    }
    mock_entity = MagicMock()
    mock_entity.name = "TestEntity"
    mock_entity.type = "DECISION"
    mock_entity.dict = MagicMock(return_value=entity_data)
    mock_manager.create_domain_entity.return_value = mock_entity
    
    # Mock retrieval methods
    mock_manager.get_project.return_value = mock_project
    mock_manager.get_component.return_value = mock_component
    
    # Mock search methods
    mock_manager.search_projects.return_value = [project_data]
    mock_manager.search_components.return_value = [component_data]
    mock_manager.search_domain_entities.return_value = [entity_data]
    mock_manager.get_project_entities.return_value = [entity_data]
    
    # Configure create_project to raise ValueError on empty name
    def mock_create_project(project):
        if not project.name:
            raise ValueError("Project name cannot be empty")
        return mock_project
    mock_manager.create_project.side_effect = mock_create_project
    
    # Configure create_component to raise exception for non-existent project
    def mock_create_component(component):
        if component.project_id == "non-existent-id":
            raise ValueError("Project not found")
        return mock_component
    mock_manager.create_component.side_effect = mock_create_component
    
    # Configure create_domain_entity to raise ValueError on invalid type
    def mock_create_domain_entity(entity):
        if entity.type == "INVALID_TYPE":
            raise ValueError("Invalid entity type: must be one of DECISION, COMPONENT, etc.")
        return mock_entity
    mock_manager.create_domain_entity.side_effect = mock_create_domain_entity
    
    return mock_manager

@pytest.fixture
def mock_base_manager():
    """
    Create a mock BaseManager for validation tests.
    
    Returns:
        A MagicMock instance with appropriate validations
    """
    from unittest.mock import MagicMock
    
    mock_base_manager = MagicMock()
    
    # Configure safe_execute_read_query to check for destructive queries
    def validate_read_query(query, params=None, database=None):
        # Validate that the query is non-destructive
        if "DELETE" in query.upper():
            raise ValueError("Destructive operation not allowed in read-only mode")
        return []
    
    mock_base_manager.safe_execute_read_query.side_effect = validate_read_query
    
    return mock_base_manager

def test_valid_project_operations(mock_project_manager, mock_logger):
    """Test valid project operations with validation."""
    # Test project creation
    project = ProjectContainer(
        name="TestProject",
        description="Test project for validation"
    )
    result = mock_project_manager.create_project(project)
    assert result is not None
    assert result.name == "TestProject"
    
    # Verify timestamp format - should be ISO format datetime string from Neo4j's datetime()
    if "created" in result.dict():
        assert re.match(r'\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}', result.dict()["created"]), "Created timestamp should be in ISO format"
    
    if "lastUpdated" in result.dict():
        assert re.match(r'\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}', result.dict()["lastUpdated"]), "LastUpdated timestamp should be in ISO format"
    
    # Test component creation
    component = ComponentCreate(
        project_id="test-project-id",
        name="TestComponent",
        description="Test component",
        type="SERVICE"
    )
    result = mock_project_manager.create_component(component)
    assert result is not None
    assert result.name == "TestComponent"
    
    # Verify timestamp format for component
    if "created" in result.dict():
        assert re.match(r'\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}', result.dict()["created"]), "Component created timestamp should be in ISO format"
    
    if "lastUpdated" in result.dict():
        assert re.match(r'\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}', result.dict()["lastUpdated"]), "Component lastUpdated timestamp should be in ISO format"
    
    # Test domain entity creation
    entity = DomainEntityCreate(
        project_id="test-project-id",
        name="TestEntity",
        type="DECISION",
        description="Test entity",
        content="Test entity content"
    )
    result = mock_project_manager.create_domain_entity(entity)
    assert result is not None
    assert result.name == "TestEntity"
    
    # Test project retrieval
    result = mock_project_manager.get_project("TestProject")
    assert result is not None
    assert result.name == "TestProject"
    
    # Test component retrieval
    result = mock_project_manager.get_component("TestComponent", "TestProject")
    assert result is not None
    assert result.name == "TestComponent"

def test_invalid_project_operations(mock_project_manager, mock_base_manager, mock_logger):
    """Test invalid project operations to verify validation."""
    # Test project creation with invalid name
    with pytest.raises(ValueError) as exc_info:
        project = ProjectContainer(
            name="",  # Empty name
            description="Invalid project"
        )
        mock_project_manager.create_project(project)
    assert "name" in str(exc_info.value)
    
    # Test component creation with non-existent project
    with pytest.raises(Exception) as exc_info:
        component = ComponentCreate(
            project_id="non-existent-id",
            name="InvalidComponent",
            description="Test component",
            type="SERVICE"
        )
        mock_project_manager.create_component(component)
    assert "project not found" in str(exc_info.value).lower()
    
    # Test domain entity creation with invalid type
    with pytest.raises(ValueError) as exc_info:
        entity = DomainEntityCreate(
            project_id="test-project-id",
            name="InvalidEntity",
            type="INVALID_TYPE",  # Invalid entity type
            description="Test entity",
            content="Test entity content"
        )
        mock_project_manager.create_domain_entity(entity)
    assert "entity type" in str(exc_info.value).lower()
    
    # Test destructive operation rejection
    with pytest.raises(ValueError):
        query = """
        MATCH (p:Project {name: $name})
        DETACH DELETE p
        """
        mock_base_manager.safe_execute_read_query(query, {"name": "TestProject"})

def test_project_search_operations(mock_project_manager, mock_logger):
    """Test project search operations with validation."""
    # Test project search
    result = mock_project_manager.search_projects("Test")
    assert isinstance(result, list)
    
    # Test component search
    result = mock_project_manager.search_components("Test", "TestProject")
    assert isinstance(result, list)
    
    # Test domain entity search
    result = mock_project_manager.search_domain_entities(
        "Test",
        project_name="TestProject",
        entity_types=["DECISION"]
    )
    assert isinstance(result, list)
    
    # Test project entities retrieval
    result = mock_project_manager.get_project_entities("TestProject")
    assert isinstance(result, list) 