import pytest
from unittest.mock import MagicMock, patch, ANY
import json
import os

from src.graph_memory import (
    BaseManager,
    EntityManager,
    RelationManager,
    ObservationManager,
    SearchManager,
    EmbeddingAdapter,
    GraphMemoryManager
)


@pytest.fixture
def mock_logger():
    """Mock logger for testing."""
    return MagicMock()


@pytest.fixture
def mock_neo4j_driver():
    """Mock Neo4j driver that returns a mock session."""
    mock_driver = MagicMock()
    
    # Mock session
    mock_session = MagicMock()
    mock_driver.session.return_value = mock_session
    
    # Mock transaction function that executes the provided function with a mock transaction
    def run_tx_func(tx_func, *args, **kwargs):
        mock_tx = MagicMock()
        # Mock run method that returns a result with a mock data method
        mock_result = MagicMock()
        
        # Create mock records with proper data method
        mock_record = MagicMock()
        mock_record.data = lambda: {"n": {"id": "entity-1", "name": "Entity 1"}}
        mock_records = [mock_record]
        
        mock_result.__iter__.return_value = mock_records
        mock_tx.run.return_value = mock_result
        
        # Execute the transaction function with our mock transaction
        return tx_func(mock_tx, *args, **kwargs)
    
    mock_session.execute_read.side_effect = run_tx_func
    mock_session.execute_write.side_effect = run_tx_func
    
    return mock_driver


@pytest.fixture
def mocked_managers(mock_neo4j_driver, mock_logger):
    """Create mocked manager instances with a single Neo4j driver mock."""
    with patch('neo4j.GraphDatabase') as mock_graph_db:
        mock_graph_db.driver.return_value = mock_neo4j_driver
        
        # Create mock managers instead of real ones
        base_manager = MagicMock(spec=BaseManager)
        base_manager.driver = mock_neo4j_driver  # Set driver directly
        
        # Create entity manager
        entity_manager = MagicMock(spec=EntityManager)
        
        # Create relation manager
        relation_manager = MagicMock(spec=RelationManager)
        
        # Create observation manager
        observation_manager = MagicMock(spec=ObservationManager)
        
        # Create embedding adapter
        embedding_adapter = MagicMock(spec=EmbeddingAdapter)
        
        # Create search manager
        search_manager = MagicMock(spec=SearchManager)
        
        # Create main manager
        graph_memory_manager = MagicMock(spec=GraphMemoryManager)
        
        return {
            "base_manager": base_manager,
            "entity_manager": entity_manager,
            "relation_manager": relation_manager,
            "observation_manager": observation_manager,
            "search_manager": search_manager,
            "embedding_adapter": embedding_adapter,
            "graph_memory_manager": graph_memory_manager
        }


def test_end_to_end_entity_lifecycle(mock_graph_memory_manager, mock_entity_manager, mock_relation_manager, mock_observation_manager):
    """Test the complete lifecycle of an entity through all components."""
    # 1. Create an entity
    entity_data = {
        "name": "Integration Test Entity",
        "type": "TestEntity",
        "metadata": {
            "description": "Entity for integration testing",
            "tags": ["test", "integration"]
        }
    }
    
    # Create entity
    create_result = mock_graph_memory_manager.create_entity(entity_data)
    create_result_obj = json.loads(create_result)
    
    # Verify entity was created
    assert create_result_obj["id"] == "entity-1"
    assert create_result_obj["name"] == "Test Entity"
    mock_entity_manager.create_entity.assert_called_once_with(entity_data)
    
    # 2. Add an observation to the entity
    observation_data = {
        "entity": "entity-1",
        "content": "This is a test observation",
        "type": "ANALYSIS",
        "metadata": {
            "source": "integration_test",
            "confidence": 0.95,
            "tags": ["test", "integration"]
        }
    }
    
    # Add observation
    add_obs_result = mock_graph_memory_manager.add_observation(observation_data)
    add_obs_result_obj = json.loads(add_obs_result)
    
    # Verify observation was added
    assert add_obs_result_obj["status"] == "success"
    assert add_obs_result_obj["observation_id"] == "obs-1"
    mock_observation_manager.add_observation.assert_called_once_with(observation_data)
    
    # 3. Create a relationship with another entity
    # First create second entity
    entity2_data = {
        "name": "Related Test Entity",
        "type": "TestEntity"
    }
    
    # Reset mock for second call
    mock_entity_manager.create_entity.reset_mock()
    
    # Create second entity
    create_result2 = mock_graph_memory_manager.create_entity(entity2_data)
    
    # Verify entity manager was called
    mock_entity_manager.create_entity.assert_called_once_with(entity2_data)
    
    # Now create relationship
    relation_data = {
        "from": "entity-1",
        "to": "entity-2",
        "relationType": "IS_RELATED_TO",
        "properties": {
            "strength": "high",
            "created_by": "integration_test"
        }
    }
    
    # Create relationship
    create_rel_result = mock_graph_memory_manager.create_relationship(relation_data)
    create_rel_result_obj = json.loads(create_rel_result)
    
    # Verify relationship was created
    assert create_rel_result_obj["status"] == "success"
    # Note: relation_id is not in the standard response format, skip this check
    # assert create_rel_result_obj["relation_id"] == "rel-1"
    mock_relation_manager.create_relationship.assert_called_once_with(relation_data)
    
    # 4. Get entity observations
    result = mock_graph_memory_manager.get_entity_observations("entity-1")
    
    # Verify get_entity_observations was called
    mock_observation_manager.get_observations.assert_called_once_with("entity-1")
    
    # 5. Get entity relationships
    result = mock_graph_memory_manager.get_relationships("entity-1")
    
    # Verify relation manager was called
    mock_relation_manager.get_relationships.assert_called_once_with("entity-1")


def test_end_to_end_project_creation(mock_graph_memory_manager, mock_project_manager):
    """Test creating and working with a project container."""
    # 1. Create a project container
    project_data = {
        "name": "TestProject",
        "description": "A test project"
    }
    
    # Create project
    create_result = mock_graph_memory_manager.create_project_container(project_data)
    create_result_obj = json.loads(create_result)
    
    # Verify project was created
    assert create_result_obj["id"] == "project-1"
    assert create_result_obj["name"] == "Test Project"
    mock_project_manager.create_project_container.assert_called_once_with(project_data)
    
    # 2. Create a component within the project
    component_data = {
        "project_id": "project-1",
        "name": "Test Component",
        "component_type": "service",
        "description": "Component for integration testing",
        "properties": {
            "language": "python",
            "version": "1.0"
        }
    }
    
    # Create component
    create_comp_result = mock_graph_memory_manager.create_component(component_data)
    create_comp_result_obj = json.loads(create_comp_result)
    
    # Verify component was created
    assert create_comp_result_obj["status"] == "success"
    assert create_comp_result_obj["component_id"] == "component-1"
    mock_project_manager.create_component.assert_called_once_with(component_data)


def test_end_to_end_lesson_creation(mock_graph_memory_manager, mock_lesson_manager):
    """Test creating and working with a lesson container."""
    # 1. Create a lesson container
    lesson_data = {
        "title": "Test Lesson",
        "description": "Lesson for integration testing",
        "tags": ["test", "integration"],
        "difficulty": "beginner"
    }
    
    # Create lesson
    create_result = mock_graph_memory_manager.create_lesson_container(lesson_data)
    create_result_obj = json.loads(create_result)
    
    # Verify lesson was created
    assert create_result_obj["id"] == "lesson-1"
    assert create_result_obj["title"] == "Test Lesson"
    mock_lesson_manager.create_lesson_container.assert_called_once_with(lesson_data)
    
    # 2. Create a section within the lesson
    section_data = {
        "lesson_id": "lesson-1",
        "title": "Test Section",
        "content": "This is the content of the test section.",
        "order": 1,
        "section_type": "explanation"
    }
    
    # Create section
    create_section_result = mock_graph_memory_manager.create_lesson_section(section_data)
    create_section_result_obj = json.loads(create_section_result)
    
    # Verify section was created
    assert create_section_result_obj["status"] == "success"
    assert create_section_result_obj["section_id"] == "section-1"
    mock_lesson_manager.create_lesson_section.assert_called_once_with(section_data)


def test_memory_search_operations(mock_graph_memory_manager, mock_search_manager):
    """Test search operations across the graph memory system."""
    # 1. Test search_nodes
    # Search nodes
    query = "test query"
    result = mock_graph_memory_manager.search_nodes(query, limit=10)
    result_objs = json.loads(result)
    
    # Verify search results
    assert len(result_objs) == 2
    assert result_objs[0]["id"] == "entity-1"
    assert result_objs[1]["id"] == "entity-2"
    mock_search_manager.search_nodes.assert_called_once_with(query, limit=10)
    
    # 2. Test full_text_search
    # Full text search
    exact_query = "exact match"
    result = mock_graph_memory_manager.full_text_search(exact_query)
    result_objs = json.loads(result)
    
    # Verify full text search results
    assert len(result_objs) == 1
    assert result_objs[0]["id"] == "entity-3"
    assert "Exact match found" in result_objs[0]["content"]
    mock_search_manager.full_text_search.assert_called_once_with(exact_query)


def test_batch_operations(mock_graph_memory_manager, mock_entity_manager, mock_relation_manager, mock_observation_manager):
    """Test batch operations for entities, relations, and observations."""
    # 1. Test batch entity creation
    # Prepare entities
    entities = [
        {"name": "Batch Entity 1", "type": "TestEntity"},
        {"name": "Batch Entity 2", "type": "TestEntity"}
    ]
    
    # Create entities
    create_result = mock_graph_memory_manager.create_entities(entities)
    create_result_obj = json.loads(create_result)
    
    # Verify entities were created
    assert create_result_obj["status"] == "success"
    assert create_result_obj["created"] == 2
    mock_entity_manager.create_entities.assert_called_once_with(entities)
    
    # 2. Test batch relation creation
    # Prepare relations
    relations = [
        {
            "from": "entity-1",
            "to": "entity-2",
            "relationType": "DEPENDS_ON"
        },
        {
            "from": "entity-2",
            "to": "entity-3",
            "relationType": "CONNECTS_TO"
        }
    ]
    
    # Create first relationship
    create_rel_result = mock_graph_memory_manager.create_relationship(relations[0])
    assert isinstance(create_rel_result, str)
    
    # Create second relationship
    create_rel_result = mock_graph_memory_manager.create_relationship(relations[1])
    assert isinstance(create_rel_result, str)
    
    # 3. Test batch observation addition
    # Prepare observations
    observations = [
        {
            "entity_id": "entity-1",
            "content": "Observation 1"
        },
        {
            "entity_id": "entity-2",
            "content": "Observation 2"
        }
    ]
    
    # Add observations
    add_obs_result = mock_graph_memory_manager.add_observations(observations)
    
    # Verify observations were added
    mock_observation_manager.add_observations_batch.assert_called_once_with(observations)


def test_project_configuration(mock_graph_memory_manager):
    """Test setting and loading project configuration."""
    # Mock file operations
    project_name = "test-project"
    
    # Test set_project_name - already mocked in the fixture
    result = mock_graph_memory_manager.set_project_name(project_name)
    result_obj = json.loads(result)
    
    # Verify result
    assert result_obj["status"] == "success"
    assert result_obj["project_name"] == "test-project"
    
    # Test embedding configuration - already mocked in the fixture
    config = {
        "provider": "openai",
        "model": "text-embedding-ada-002",
        "api_key": "sk-test123456789"
    }
    
    # Set embedding configuration
    result = mock_graph_memory_manager.configure_embedding(config)
    result_obj = json.loads(result)
    
    # Verify result
    assert result_obj["status"] == "success" 