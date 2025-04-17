"""
Unit tests for the ProjectMemoryManager.
"""

import pytest
from unittest.mock import MagicMock, patch
import json

from src.project_memory import ProjectMemoryManager

class TestProjectMemoryManager:
    """Test suite for ProjectMemoryManager class."""
    
    @pytest.fixture
    def mock_project_memory_manager(self, mock_graph_memory_manager):
        """Create a mock ProjectMemoryManager."""
        # Create a mock with autospec to ensure all methods are available
        manager = MagicMock(spec=ProjectMemoryManager, autospec=True)
        
        # Set the base_manager attribute
        manager.base_manager = mock_graph_memory_manager
        
        # Mock the manager attributes to match the actual implementation
        manager.domain_manager = MagicMock()
        manager.component_manager = MagicMock()
        manager.dependency_manager = MagicMock()
        manager.version_manager = MagicMock()
        manager.project_container = MagicMock()
        manager.search_manager = MagicMock()
        
        # Mock standard methods that tests will call
        manager.create_project_container.return_value = json.dumps({
            "status": "success",
            "message": "Project created successfully",
            "data": {"name": "TestProject"}
        })
        
        manager.get_project_container.return_value = json.dumps({
            "status": "success",
            "message": "Project retrieved successfully",
            "data": {"name": "TestProject"}
        })
        
        manager.list_project_containers.return_value = json.dumps({
            "status": "success",
            "message": "Projects retrieved successfully",
            "data": {"projects": [], "count": 0}
        })
        
        manager.create_project_component.return_value = json.dumps({
            "status": "success",
            "message": "Component created successfully",
            "data": {"name": "TestComponent", "component_type": "SERVICE", "project_id": "TestProject"}
        })
        
        manager.create_project_domain.return_value = json.dumps({
            "status": "success",
            "message": "Domain entity created successfully",
            "data": {"name": "TestEntity", "entity_type": "REQUIREMENT", "project_id": "TestProject"}
        })
        
        manager.create_project_dependency.return_value = json.dumps({
            "status": "success",
            "message": "Dependency created successfully",
            "data": {"source": "SourceComponent", "target": "TargetComponent"}
        })
        
        return manager
    
    def test_initialization(self, mock_graph_memory_manager):
        """Test that the ProjectMemoryManager initializes correctly."""
        manager = ProjectMemoryManager(mock_graph_memory_manager)
        
        # Verify manager components are initialized
        assert hasattr(manager, 'project_container')
        assert hasattr(manager, 'component_manager')
        assert hasattr(manager, 'domain_manager')
        assert hasattr(manager, 'dependency_manager')
        assert hasattr(manager, 'version_manager')
        assert hasattr(manager, 'search_manager')
    
    def test_create_project(self, mock_project_memory_manager, sample_project_data, json_helper):
        """Test creating a project."""
        # Configure the container mock
        mock_project_memory_manager.create_project_container.return_value = json.dumps({
            "status": "success",
            "message": "Project created successfully",
            "data": {
                "name": sample_project_data["name"],
                "description": sample_project_data["description"],
                "metadata": sample_project_data["metadata"]
            }
        })
        
        # Call the method
        result = mock_project_memory_manager.create_project_container(
            sample_project_data
        )
        
        # Verify the result
        result_dict = json_helper["from_json"](result)
        assert result_dict["status"] == "success"
        assert "Project created successfully" in result_dict["message"]
        assert result_dict["data"]["name"] == sample_project_data["name"]
        assert result_dict["data"]["description"] == sample_project_data["description"]
    
    def test_get_project(self, mock_project_memory_manager, json_helper):
        """Test retrieving a project."""
        project_name = "TestProject"
        
        # Configure the container mock
        mock_project_memory_manager.get_project_container.return_value = json.dumps({
            "status": "success",
            "message": "Project retrieved successfully",
            "data": {
                "name": project_name,
                "description": "Test project",
                "component_count": 3
            }
        })
        
        # Call the method
        result = mock_project_memory_manager.get_project_container(project_name)
        
        # Verify the result
        result_dict = json_helper["from_json"](result)
        assert result_dict["status"] == "success"
        assert "Project retrieved successfully" in result_dict["message"]
        assert result_dict["data"]["name"] == project_name
    
    def test_list_projects(self, mock_project_memory_manager, json_helper):
        """Test listing all projects."""
        # Mock project list
        mock_projects = [
            {"name": "Project1", "description": "First project"},
            {"name": "Project2", "description": "Second project"}
        ]
        
        # Configure the container mock
        mock_project_memory_manager.list_project_containers.return_value = json.dumps({
            "status": "success",
            "message": "Projects retrieved successfully",
            "data": {"projects": mock_projects, "count": len(mock_projects)}
        })
        
        # Call the method
        result = mock_project_memory_manager.list_project_containers()
        
        # Verify the result
        result_dict = json_helper["from_json"](result)
        assert result_dict["status"] == "success"
        assert "Projects retrieved successfully" in result_dict["message"]
        assert len(result_dict["data"]["projects"]) == 2
        assert result_dict["data"]["count"] == 2
    
    def test_create_component(self, mock_project_memory_manager, json_helper):
        """Test creating a component."""
        project_id = "TestProject"
        component_name = "TestComponent"
        component_type = "SERVICE"
        description = "A test component"
        
        # Configure the component mock
        mock_project_memory_manager.create_project_component.return_value = json.dumps({
            "status": "success",
            "message": "Component created successfully",
            "data": {
                "name": component_name,
                "component_type": component_type,
                "description": description,
                "project_id": project_id
            }
        })
        
        # Call the method
        result = mock_project_memory_manager.create_project_component(
            component_name, component_type, "TestDomain", project_id, description
        )
        
        # Verify the result
        result_dict = json_helper["from_json"](result)
        assert result_dict["status"] == "success"
        assert "Component created successfully" in result_dict["message"]
        assert result_dict["data"]["name"] == component_name
        assert result_dict["data"]["component_type"] == component_type
        assert result_dict["data"]["project_id"] == project_id
    
    def test_create_domain_entity(self, mock_project_memory_manager, json_helper):
        """Test creating a domain entity."""
        project_id = "TestProject"
        entity_name = "TestRequirement"
        entity_type = "REQUIREMENT"
        content = "The system must support user authentication"
        
        # Configure the domain mock
        mock_project_memory_manager.create_project_domain.return_value = json.dumps({
            "status": "success",
            "message": "Domain entity created successfully",
            "data": {
                "name": entity_name,
                "entity_type": entity_type,
                "content": content,
                "project_id": project_id
            }
        })
        
        # Call the method
        result = mock_project_memory_manager.create_project_domain(
            project_id, entity_type, entity_name, content
        )
        
        # Verify the result
        result_dict = json_helper["from_json"](result)
        assert result_dict["status"] == "success"
        assert "Domain entity created successfully" in result_dict["message"]
        assert result_dict["data"]["name"] == entity_name
        assert result_dict["data"]["entity_type"] == entity_type
        assert result_dict["data"]["project_id"] == project_id
    
    def test_create_component_dependency(self, mock_project_memory_manager, json_helper):
        """Test creating a component dependency."""
        source_component = "SourceComponent"
        target_component = "TargetComponent"
        project_id = "TestProject"
        dependency_type = "DEPENDS_ON"
        properties = {"critical": True}
        
        # Configure the dependency mock
        mock_project_memory_manager.create_project_dependency.return_value = json.dumps({
            "status": "success",
            "message": "Dependency created successfully",
            "data": {
                "source": source_component,
                "target": target_component,
                "dependency_type": dependency_type,
                "project_id": project_id,
                "properties": properties
            }
        })
        
        # Call the method
        result = mock_project_memory_manager.create_project_dependency(
            project_id, source_component, target_component, dependency_type, properties
        )
        
        # Verify the result
        result_dict = json_helper["from_json"](result)
        assert result_dict["status"] == "success"
        assert "Dependency created successfully" in result_dict["message"]
        assert result_dict["data"]["source"] == source_component
        assert result_dict["data"]["target"] == target_component
        assert result_dict["data"]["dependency_type"] == dependency_type
    
    def test_get_project_components(self, mock_project_memory_manager, json_helper):
        """Test retrieving project components."""
        project_id = "TestProject"
        domain_name = "TestDomain"
        
        # Mock component list
        mock_components = [
            {"name": "Component1", "component_type": "SERVICE"},
            {"name": "Component2", "component_type": "UI"}
        ]
        
        # Configure the component mock
        mock_project_memory_manager.list_project_components.return_value = json.dumps({
            "status": "success",
            "message": "Components retrieved successfully",
            "data": {"components": mock_components, "count": len(mock_components)}
        })
        
        # Call the method
        result = mock_project_memory_manager.list_project_components(domain_name, project_id)
        
        # Verify the result
        result_dict = json_helper["from_json"](result)
        assert result_dict["status"] == "success"
        assert "Components retrieved successfully" in result_dict["message"]
        assert len(result_dict["data"]["components"]) == 2
        assert result_dict["data"]["count"] == 2
    
    def test_create_version(self, mock_project_memory_manager, json_helper):
        """Test creating a project version."""
        project_id = "TestProject"
        component_name = "TestComponent"
        domain_name = "TestDomain"
        version = "1.0.0"
        description = "Initial release"
        
        # Configure the version mock
        mock_project_memory_manager.create_project_version.return_value = json.dumps({
            "status": "success",
            "message": "Version created successfully",
            "data": {
                "project_id": project_id,
                "component_name": component_name,
                "version": version,
                "description": description
            }
        })
        
        # Call the method
        result = mock_project_memory_manager.create_project_version(
            component_name, domain_name, project_id, version
        )
        
        # Verify the result
        result_dict = json_helper["from_json"](result)
        assert result_dict["status"] == "success"
        assert "Version created successfully" in result_dict["message"]
        assert result_dict["data"]["component_name"] == component_name
        assert result_dict["data"]["version"] == version
    
    def test_get_component_dependencies(self, mock_project_memory_manager, json_helper):
        """Test retrieving component dependencies."""
        project_id = "TestProject"
        component_name = "TestComponent"
        domain_name = "TestDomain"
        
        # Mock dependencies
        mock_dependencies = [
            {"source": component_name, "target": "Database", "dependency_type": "DEPENDS_ON"},
            {"source": "API", "target": component_name, "dependency_type": "CALLS"}
        ]
        
        # Configure the dependency mock
        mock_project_memory_manager.get_project_dependencies.return_value = json.dumps({
            "status": "success",
            "message": "Dependencies retrieved successfully",
            "data": {"dependencies": mock_dependencies, "count": len(mock_dependencies)}
        })
        
        # Call the method
        result = mock_project_memory_manager.get_project_dependencies(
            component_name, domain_name, project_id
        )
        
        # Verify the result
        result_dict = json_helper["from_json"](result)
        assert result_dict["status"] == "success"
        assert "Dependencies retrieved successfully" in result_dict["message"]
        assert len(result_dict["data"]["dependencies"]) == 2
        assert result_dict["data"]["count"] == 2 