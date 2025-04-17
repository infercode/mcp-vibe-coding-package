"""
Common test fixtures and utilities for testing the Neo4j Graph Memory System.
"""

import os
import asyncio
import pytest
from unittest.mock import MagicMock, patch
import json
from typing import Dict, List, Any, Optional
from dotenv import load_dotenv
from contextlib import contextmanager

# Load environment variables
load_dotenv()

# Import necessary components
from src.graph_memory import GraphMemoryManager
from src.logger import get_logger, LogLevel
from src.session_manager import SessionManager

# Initialize logger for tests
@pytest.fixture
def logger():
    """Get a configured logger instance."""
    logger = get_logger()
    logger.set_level(LogLevel.DEBUG)
    return logger

# Mock Neo4j session and driver
@pytest.fixture
def mock_neo4j_driver():
    """Create a mock Neo4j driver."""
    driver = MagicMock()
    session = MagicMock()
    transaction = MagicMock()
    result = MagicMock()
    
    # Configure mocks
    driver.session.return_value = session
    session.begin_transaction.return_value = transaction
    transaction.run.return_value = result
    
    # Mock the result configuration
    result.data.return_value = []
    
    with patch('neo4j.GraphDatabase.driver', return_value=driver):
        yield driver

# Mock GraphMemoryManager
@pytest.fixture
def mock_graph_memory_manager(logger, mock_neo4j_driver):
    """Create a mock GraphMemoryManager with stubbed Neo4j interactions."""
    # Create a mock GraphMemoryManager
    mock_manager = MagicMock(spec=GraphMemoryManager)
    
    # Set up required attributes and methods for core functionality
    mock_manager.initialize.return_value = None
    mock_manager.driver = mock_neo4j_driver
    mock_manager.logger = logger
    
    # Make close method call driver.close()
    def close_method():
        mock_neo4j_driver.close()
        return "Connection closed successfully"
    mock_manager.close.side_effect = close_method
    
    # Mock entity_manager and methods
    mock_manager.entity_manager = MagicMock()
    mock_manager.entity_manager.validate_entity_data.side_effect = lambda data: None if all(k in data for k in ['name', 'entity_type']) else ValueError("Invalid entity data")
    mock_manager.entity_manager.create_entity.return_value = json.dumps({"status": "success", "message": "Entity created successfully", "data": {}})
    mock_manager.entity_manager.delete_entity.return_value = json.dumps({"status": "success", "message": "Entity deleted successfully", "data": {}})
    
    # Mock relation_manager and methods
    mock_manager.relation_manager = MagicMock()
    mock_manager.relation_manager.validate_relation_data.side_effect = lambda data: None if all(k in data for k in ['source_id', 'target_id', 'relation_type']) else ValueError("Invalid relation data")
    mock_manager.relation_manager.create_relationship.return_value = json.dumps({"status": "success", "message": "Relationship created successfully", "data": {}})
    mock_manager.relation_manager.delete_relationship.return_value = json.dumps({"status": "success", "message": "Relationship deleted successfully", "data": {}})
    
    # Mock observation_manager and methods
    mock_manager.observation_manager = MagicMock()
    mock_manager.observation_manager.validate_observation_data.side_effect = lambda data: None if all(k in data for k in ['entity_id', 'content']) else ValueError("Invalid observation data")
    mock_manager.observation_manager.add_observation.return_value = json.dumps({"status": "success", "message": "Observation added successfully", "data": {}})
    mock_manager.observation_manager.delete_observation.return_value = json.dumps({"status": "success", "message": "Observation deleted successfully", "data": {}})
    
    # Mock search_manager and methods
    mock_manager.search_manager = MagicMock()
    mock_manager.search_manager.validate_search_params.side_effect = lambda params: None if isinstance(params, dict) and 'query' in params else ValueError("Invalid search parameters")
    mock_manager.search_manager.search_nodes.return_value = json.dumps({"status": "success", "message": "Search completed successfully", "data": {"results": []}})

    # Set up for lesson_memory operations
    mock_manager.lesson_memory = MagicMock()
    mock_manager.lesson_memory.container_name = "DefaultContainer"
    
    # Create handler methods with appropriate return values for lesson memory operations
    lesson_operation_handlers = {
        "_handle_lesson_container_creation": "Container created successfully",
        "_handle_lesson_creation": "Lesson created successfully",
        "_handle_lesson_observation": "Observation added successfully",
        "_handle_lesson_relationship": "Lessons related successfully",
        "_handle_lesson_search": "Search completed successfully",
        "_handle_lesson_tracking": "Lesson tracking completed successfully",
        "_handle_lesson_consolidation": "Lessons consolidated successfully",
        "_handle_lesson_evolution": "Lesson evolved successfully",
        "_handle_lesson_update": "Lesson updated successfully",
        "_handle_get_lesson_container": "Container retrieved successfully",
        "_handle_list_lesson_containers": "Containers retrieved successfully",
        "_handle_container_exists": "Container existence checked successfully"
    }
    
    # Set up the handler methods with appropriate return values
    for handler, message in lesson_operation_handlers.items():
        handler_mock = MagicMock()
        handler_mock.return_value = json.dumps({
            "status": "success",
            "message": message,
            "data": {"operation": handler.replace("_handle_lesson_", "")}
        })
        setattr(mock_manager, handler, handler_mock)
    
    # Map operation types to handler methods for lesson_operation
    lesson_operation_map = {
        "create_container": "_handle_lesson_container_creation",
        "create": "_handle_lesson_creation",
        "observe": "_handle_lesson_observation",
        "relate": "_handle_lesson_relationship",
        "search": "_handle_lesson_search",
        "track": "_handle_lesson_tracking",
        "consolidate": "_handle_lesson_consolidation", 
        "evolve": "_handle_lesson_evolution",
        "update": "_handle_lesson_update",
        "get_container": "_handle_get_lesson_container",
        "list_containers": "_handle_list_lesson_containers",
        "container_exists": "_handle_container_exists"
    }
    
    # Mock the lesson_operation method
    def mock_lesson_operation(operation_type, **kwargs):
        if operation_type not in lesson_operation_map:
            raise ValueError(f"Invalid operation type: {operation_type}")
            
        handler_method = lesson_operation_map.get(operation_type)
        if handler_method and hasattr(mock_manager, handler_method):
            return getattr(mock_manager, handler_method)(**kwargs)
        return json.dumps({"status": "error", "message": "Handler not found", "data": {}})
        
    mock_manager.lesson_operation.side_effect = mock_lesson_operation
    
    # Set up for project_memory operations  
    mock_manager.project_memory = MagicMock()
    mock_manager.project_memory.project_name = "DefaultProject"
    
    # Create handler methods with appropriate return values for project memory operations
    project_operation_handlers = {
        "_handle_project_creation": "Project created successfully",
        "_handle_component_creation": "Component created successfully",
        "_handle_domain_creation": "Domain created successfully",
        "_handle_domain_entity_creation": "Domain entity created successfully",
        "_handle_entity_relationship": "Entity relationship created successfully",
        "_handle_project_search": "Project search completed successfully",
        "_handle_structure_retrieval": "Structure retrieved successfully",
        "_handle_add_observation": "Observation added successfully",
        "_handle_entity_update": "Entity updated successfully",
        "_handle_entity_deletion": "Entity deleted successfully",
        "_handle_relationship_deletion": "Relationship deleted successfully"
    }
    
    # Set up the handler methods with appropriate return values
    for handler, message in project_operation_handlers.items():
        handler_mock = MagicMock()
        handler_mock.return_value = json.dumps({
            "status": "success",
            "message": message,
            "data": {"operation": handler.replace("_handle_", "")}
        })
        setattr(mock_manager, handler, handler_mock)
    
    # Map operation types to handler methods for project_operation
    project_operation_map = {
        "create_project": "_handle_project_creation",
        "create_component": "_handle_component_creation",
        "create_domain": "_handle_domain_creation",
        "create_domain_entity": "_handle_domain_entity_creation",
        "relate": "_handle_entity_relationship",
        "search": "_handle_project_search",
        "get_structure": "_handle_structure_retrieval",
        "add_observation": "_handle_add_observation",
        "update": "_handle_entity_update",
        "delete_entity": "_handle_entity_deletion",
        "delete_relationship": "_handle_relationship_deletion"
    }
    
    # Mock the project_operation method
    def mock_project_operation(operation_type, **kwargs):
        if operation_type not in project_operation_map:
            raise ValueError(f"Invalid operation type: {operation_type}")
            
        handler_method = project_operation_map.get(operation_type)
        if handler_method and hasattr(mock_manager, handler_method):
            return getattr(mock_manager, handler_method)(**kwargs)
        return json.dumps({"status": "error", "message": "Handler not found", "data": {}})
        
    mock_manager.project_operation.side_effect = mock_project_operation
    
    # Mock contextmanager methods
    @contextmanager
    def mock_lesson_context(project_name=None, container_name=None):
        original_container = mock_manager.lesson_memory.container_name
        try:
            if project_name:
                mock_manager.set_project_name(project_name, None)
            if container_name:
                mock_manager.lesson_memory.container_name = container_name
            yield MagicMock()
        finally:
            mock_manager.lesson_memory.container_name = original_container
    
    mock_manager.lesson_context = mock_lesson_context
    
    @contextmanager
    def mock_project_context(project_name=None):
        original_project = mock_manager.project_memory.project_name
        try:
            if project_name:
                mock_manager.project_memory.project_name = project_name
            yield MagicMock()
        finally:
            mock_manager.project_memory.project_name = original_project
    
    mock_manager.project_context = mock_project_context
    
    # Add error handling for test_error_handling_in_lesson_operation
    def mock_standardize_response(**kwargs):
        return kwargs.get("result_json", "{}")
    
    mock_manager._standardize_response = MagicMock(side_effect=mock_standardize_response)
    
    return mock_manager

# Real GraphMemoryManager (use only when Neo4j is available)
@pytest.fixture
def graph_memory_manager(logger):
    """
    Create a real GraphMemoryManager connected to Neo4j.
    
    Note: This requires a running Neo4j instance with the correct credentials.
    Skip tests using this fixture if Neo4j is not available.
    """
    try:
        manager = GraphMemoryManager(logger)
        yield manager
        manager.close()
    except Exception as e:
        pytest.skip(f"Neo4j connection failed: {e}")

# SessionManager fixture
@pytest.fixture
def session_manager():
    """Create a SessionManager instance."""
    manager = SessionManager(inactive_timeout=60, cleanup_interval=30)
    yield manager
    # Clean up
    asyncio.run(manager.stop_cleanup_task())

# Environment configuration
@pytest.fixture
def env_config():
    """Provide environment configuration values."""
    return {
        "neo4j_uri": os.getenv("NEO4J_URI", "bolt://localhost:7687"),
        "neo4j_user": os.getenv("NEO4J_USER", "neo4j"),
        "neo4j_password": os.getenv("NEO4J_PASSWORD", "password"),
        "neo4j_database": os.getenv("NEO4J_DATABASE", "neo4j"),
        "embedder_provider": os.getenv("EMBEDDER_PROVIDER", "none"),
    }

# Test data fixtures
@pytest.fixture
def sample_entity_data():
    """Sample entity data for testing."""
    return {
        "name": "TestEntity",
        "entityType": "TEST",
        "observations": ["This is a test entity"]
    }

@pytest.fixture
def sample_relation_data():
    """Sample relation data for testing."""
    return {
        "from": "SourceEntity",
        "to": "TargetEntity",
        "relationType": "TEST_RELATION"
    }

@pytest.fixture
def sample_lesson_data():
    """Sample lesson data for testing."""
    return {
        "name": "TestLesson",
        "entity_type": "LESSON",
        "observations": ["This is a test lesson"],
        "metadata": {
            "confidence": 0.8,
            "tags": ["test", "example"]
        }
    }

@pytest.fixture
def sample_project_data():
    """Sample project data for testing."""
    return {
        "name": "TestProject",
        "description": "A test project",
        "metadata": {
            "status": "active",
            "priority": "high"
        }
    }

# Helper function to create and read JSON data
@pytest.fixture
def json_helper():
    """Helper for working with JSON data in tests."""
    def to_json(data):
        return json.dumps(data)
        
    def from_json(json_str):
        return json.loads(json_str)
        
    return {
        "to_json": to_json,
        "from_json": from_json
    }

# Mock server object
@pytest.fixture
def mock_server():
    """Create a mock server object for tool registration."""
    server = MagicMock()
    server.tool = MagicMock(return_value=lambda func: func)
    return server 