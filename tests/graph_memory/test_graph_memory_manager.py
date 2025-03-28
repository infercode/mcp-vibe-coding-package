import pytest
from unittest.mock import patch, MagicMock, call
import json
import os

from src.graph_memory import GraphMemoryManager
from src.lesson_memory import LessonMemoryManager
from src.project_memory import ProjectMemoryManager


def test_initialization(mock_logger):
    """Test that GraphMemoryManager initializes correctly."""
    # Create GraphMemoryManager with all components mocked
    with patch('src.graph_memory.base_manager.BaseManager') as mock_base_manager_cls, \
         patch('src.graph_memory.entity_manager.EntityManager') as mock_entity_manager_cls, \
         patch('src.graph_memory.relation_manager.RelationManager') as mock_relation_manager_cls, \
         patch('src.graph_memory.observation_manager.ObservationManager') as mock_observation_manager_cls, \
         patch('src.graph_memory.search_manager.SearchManager') as mock_search_manager_cls, \
         patch('src.graph_memory.embedding_adapter.EmbeddingAdapter') as mock_embedding_adapter_cls, \
         patch('src.lesson_memory.LessonMemoryManager') as mock_lesson_memory_cls, \
         patch('src.project_memory.ProjectMemoryManager') as mock_project_memory_cls:
        
        # Create manager
        manager = GraphMemoryManager(
            logger=mock_logger,
            embedding_api_key="test_key",
            embedding_model="test_model",
            neo4j_uri="bolt://localhost:7687",
            neo4j_username="neo4j",
            neo4j_password="password"
        )
        
        # Verify all components are created
        assert isinstance(manager.base_manager, MagicMock)
        assert isinstance(manager.entity_manager, MagicMock)
        assert isinstance(manager.relation_manager, MagicMock)
        assert isinstance(manager.observation_manager, MagicMock)
        assert isinstance(manager.search_manager, MagicMock)
        assert isinstance(manager.embedding_adapter, MagicMock)
        assert isinstance(manager.lesson_memory, MagicMock)
        assert isinstance(manager.project_memory, MagicMock)
        
        # Verify environment variables
        assert manager.embedding_api_key == "test_key"
        assert manager.embedding_model == "test_model"
        assert manager.neo4j_uri == "bolt://localhost:7687"
        assert manager.neo4j_user == "neo4j"
        assert manager.neo4j_password == "password"
        
        # Verify base properties
        assert manager.default_project_name == "default"
        assert manager.embedding_enabled is True


def test_initialize_method(mock_graph_memory_manager):
    """Test the initialize method."""
    # Configure mocks
    mock_base_manager = mock_graph_memory_manager._mock_managers['base_manager']
    mock_embedding_adapter = mock_graph_memory_manager._mock_managers['embedding_adapter']
    
    mock_embedding_adapter.init_embedding_manager.return_value = True
    mock_base_manager.initialize.return_value = True
    mock_base_manager.neo4j_uri = "bolt://localhost:7687"
    mock_base_manager.neo4j_user = "neo4j"
    mock_base_manager.neo4j_password = "password"
    mock_base_manager.neo4j_database = "neo4j"
    mock_base_manager.embedder_provider = "openai"
    mock_base_manager.neo4j_driver = MagicMock()
    
    # Call the method
    result = mock_graph_memory_manager.initialize()
    
    # Verify initialization flow
    mock_embedding_adapter.init_embedding_manager.assert_called_once_with(
        api_key="test_key",
        model_name="test_model"
    )
    mock_base_manager.initialize.assert_called_once()
    
    # Verify result
    assert result is True


def test_close_method(mock_graph_memory_manager):
    """Test the close method."""
    # Configure mocks
    mock_base_manager = mock_graph_memory_manager._mock_managers['base_manager']
    
    # Call the method
    mock_graph_memory_manager.close()
    
    # Verify close was called
    mock_base_manager.close.assert_called_once()


def test_entity_operations(mock_graph_memory_manager, sample_entity, sample_entities):
    """Test entity operations delegation."""
    # Configure mocks
    mock_entity_manager = mock_graph_memory_manager._mock_managers['entity_manager']
    
    # Test create_entities
    mock_entity_manager.create_entities.return_value = '{"status": "success"}'
    result = mock_graph_memory_manager.create_entities(sample_entities)
    mock_entity_manager.create_entities.assert_called_once_with(sample_entities)
    assert result == '{"status": "success"}'
    
    # Test get_entity
    mock_entity_manager.get_entity.return_value = '{"name": "test_entity"}'
    result = mock_graph_memory_manager.get_entity("test_entity")
    mock_entity_manager.get_entity.assert_called_once_with("test_entity")
    assert result == '{"name": "test_entity"}'
    
    # Test delete_entity
    mock_entity_manager.delete_entity.return_value = '{"status": "success"}'
    result = mock_graph_memory_manager.delete_entity("test_entity")
    mock_entity_manager.delete_entity.assert_called_once_with("test_entity")
    assert result == '{"status": "success"}'


def test_relation_operations(mock_graph_memory_manager, sample_relation, sample_relations):
    """Test relation operations delegation."""
    # Configure mocks
    mock_relation_manager = mock_graph_memory_manager._mock_managers['relation_manager']
    
    # Test create_relations
    mock_relation_manager.create_relationships.return_value = '{"status": "success"}'
    result = mock_graph_memory_manager.create_relations(sample_relations)
    mock_relation_manager.create_relationships.assert_called_once_with(sample_relations)
    assert result == '{"status": "success"}'
    
    # Test get_relations
    relation_type = sample_relation["relationType"]
    mock_relation_manager.get_relationships.return_value = '[{"type": "CONNECTS_TO"}]'
    result = mock_graph_memory_manager.get_relations("test_entity", relation_type)
    mock_relation_manager.get_relationships.assert_called_once_with(
        entity_name="test_entity", relation_type=relation_type
    )
    assert result == '[{"type": "CONNECTS_TO"}]'
    
    # Test delete_relation
    from_entity = sample_relation["from"]
    to_entity = sample_relation["to"]
    mock_relation_manager.delete_relationship.return_value = '{"status": "success"}'
    result = mock_graph_memory_manager.delete_relation(from_entity, to_entity, relation_type)
    mock_relation_manager.delete_relationship.assert_called_once_with(
        from_entity=from_entity, to_entity=to_entity, relation_type=relation_type
    )
    assert result == '{"status": "success"}'


def test_observation_operations(mock_graph_memory_manager, sample_observation, sample_observations):
    """Test observation operations delegation."""
    # Configure mocks
    mock_observation_manager = mock_graph_memory_manager._mock_managers['observation_manager']
    
    # Test add_observations
    mock_observation_manager.add_observations.return_value = '{"status": "success"}'
    result = mock_graph_memory_manager.add_observations(sample_observations)
    mock_observation_manager.add_observations.assert_called_once_with(sample_observations)
    assert result == '{"status": "success"}'
    
    # Test get_entity_observations
    mock_observation_manager.get_entity_observations.return_value = '[{"content": "Test"}]'
    result = mock_graph_memory_manager.get_entity_observations("test_entity", "OBSERVATION")
    mock_observation_manager.get_entity_observations.assert_called_once_with(
        entity_name="test_entity", observation_type="OBSERVATION"
    )
    assert result == '[{"content": "Test"}]'
    
    # Test delete_observation
    mock_observation_manager.delete_observation.return_value = '{"status": "success"}'
    result = mock_graph_memory_manager.delete_observation("test_entity", "Test content")
    mock_observation_manager.delete_observation.assert_called_once_with(
        entity_name="test_entity", observation_content="Test content", observation_id=None
    )
    assert result == '{"status": "success"}'


def test_search_operations(mock_graph_memory_manager):
    """Test search operations delegation."""
    # Configure mocks
    mock_search_manager = mock_graph_memory_manager._mock_managers['search_manager']
    
    # Test search_entities
    mock_search_manager.search_entities.return_value = '[{"name": "test_entity"}]'
    result = mock_graph_memory_manager.search_entities("test", 10, ["Entity"], True)
    mock_search_manager.search_entities.assert_called_once_with(
        search_term="test", limit=10, entity_types=["Entity"], semantic=True
    )
    assert result == '[{"name": "test_entity"}]'
    
    # Test search_nodes
    mock_search_manager.search_nodes.return_value = '[{"name": "test_entity"}]'
    result = mock_graph_memory_manager.search_nodes("test", 10, "project1")
    mock_search_manager.search_nodes.assert_called_once_with(
        query="test", limit=10, project_name="project1"
    )
    assert result == '[{"name": "test_entity"}]'


def test_lesson_memory_operations(mock_graph_memory_manager):
    """Test lesson memory operations delegation."""
    # Configure mocks
    mock_lesson_memory = mock_graph_memory_manager._mock_managers['lesson_memory']
    
    # Test create_lesson
    mock_lesson_memory.create_lesson.return_value = '{"id": "lesson1"}'
    result = mock_graph_memory_manager.create_lesson("Test Lesson", "Test problem")
    mock_lesson_memory.create_lesson.assert_called_once()
    assert result == '{"id": "lesson1"}'
    
    # Test get_lessons
    mock_lesson_memory.get_lessons.return_value = '[{"name": "Test Lesson"}]'
    result = mock_graph_memory_manager.get_lessons()
    mock_lesson_memory.get_lessons.assert_called_once()
    assert result == '[{"name": "Test Lesson"}]'


def test_project_memory_operations(mock_graph_memory_manager):
    """Test project memory operations delegation."""
    # Configure mocks
    mock_project_memory = mock_graph_memory_manager._mock_managers['project_memory']
    
    # Test create_project_container
    mock_project_memory.create_project_container.return_value = '{"id": "project1"}'
    result = mock_graph_memory_manager.create_project_container("Test Project", "Test description")
    mock_project_memory.create_project_container.assert_called_once_with(
        "Test Project", "Test description", None
    )
    assert result == '{"id": "project1"}'
    
    # Test get_project_container
    mock_project_memory.get_project_container.return_value = '{"name": "Test Project"}'
    result = mock_graph_memory_manager.get_project_container("project1")
    mock_project_memory.get_project_container.assert_called_once_with("project1")
    assert result == '{"name": "Test Project"}'


def test_config_operations(mock_graph_memory_manager):
    """Test configuration operations."""
    # Configure mocks
    mock_embedding_adapter = mock_graph_memory_manager._mock_managers['embedding_adapter']
    
    # Test apply_client_config
    config = {
        "provider": "openai",
        "model": "text-embedding-3-small",
        "api_key": "new_key",
        "project_name": "test_project"
    }
    mock_embedding_adapter.configure_embedding.return_value = {"status": "success"}
    result = mock_graph_memory_manager.apply_client_config(config)
    mock_embedding_adapter.configure_embedding.assert_called_once()
    assert result["status"] == "success"


def test_get_current_config(mock_graph_memory_manager):
    """Test get_current_config method."""
    # Setup initial property values
    mock_graph_memory_manager.neo4j_uri = "bolt://localhost:7687"
    mock_graph_memory_manager.neo4j_user = "neo4j"
    mock_graph_memory_manager.embedder_provider = "openai"
    mock_graph_memory_manager.default_project_name = "default"
    
    # Call the method
    result = mock_graph_memory_manager.get_current_config()
    
    # Verify config contains expected keys
    assert "neo4j_uri" in result
    assert "neo4j_user" in result
    assert "embedder_provider" in result
    assert "project_name" in result


def test_check_connection(mock_graph_memory_manager):
    """Test check_connection method."""
    # Setup mock driver
    mock_driver = MagicMock()
    mock_graph_memory_manager.neo4j_driver = mock_driver
    mock_driver.safe_execute_query.return_value = ([{"message": "Connection test"}], None)
    
    # Call the method
    result = mock_graph_memory_manager.check_connection()
    
    # Verify result
    assert result is True
    mock_driver.safe_execute_query.assert_called_once()


def test_set_project_name(mock_graph_memory_manager):
    """Test set_project_name method."""
    # Configure mocks
    mock_base_manager = mock_graph_memory_manager.base_manager
    
    # Call the method
    result = mock_graph_memory_manager.set_project_name("new_project")
    
    # Verify result
    assert mock_graph_memory_manager.default_project_name == "new_project"
    assert result is True


def test_get_all_memories(mock_graph_memory_manager):
    """Test get_all_memories method."""
    # Setup mocks
    mock_base_manager = mock_graph_memory_manager.base_manager
    # Mock base_manager.safe_execute_query
    mock_base_manager.safe_execute_query.return_value = ([{"node": {"properties": {"name": "entity1"}}}], None)
    
    # Call the method
    result = mock_graph_memory_manager.get_all_memories()
    
    # Verify base_manager.safe_execute_query was called
    mock_base_manager.safe_execute_query.assert_called_once()
    
    # verify result format
    assert isinstance(json.loads(result), dict)


def test_error_handling(mock_logger):
    """Test error handling in the facade class."""
    # Create manager with failing components
    with patch('src.graph_memory.base_manager.BaseManager') as mock_base_manager_cls, \
         patch('src.graph_memory.entity_manager.EntityManager'), \
         patch('src.graph_memory.relation_manager.RelationManager'), \
         patch('src.graph_memory.observation_manager.ObservationManager'), \
         patch('src.graph_memory.search_manager.SearchManager'), \
         patch('src.graph_memory.embedding_adapter.EmbeddingAdapter') as mock_embedding_adapter_cls, \
         patch('src.lesson_memory.LessonMemoryManager'), \
         patch('src.project_memory.ProjectMemoryManager'):
        
        # Setup mocks to fail
        mock_base_manager = MagicMock()
        mock_base_manager_cls.return_value = mock_base_manager
        mock_base_manager.initialize.return_value = False
        
        mock_embedding_adapter = MagicMock()
        mock_embedding_adapter_cls.return_value = mock_embedding_adapter
        mock_embedding_adapter.init_embedding_manager.return_value = False
        
        # Create manager
        manager = GraphMemoryManager(logger=mock_logger)
        
        # Test initialization failure
        result = manager.initialize()
        assert result is False
        mock_logger.error.assert_called()


@pytest.mark.parametrize("method_name,args,kwargs,expected", [
    ("create_entities", [[{"name": "test"}]], {}, '{"status": "success"}'),
    ("get_entity", ["test_entity"], {}, '{"name": "test_entity"}'),
    ("delete_entity", ["test_entity"], {}, '{"status": "success"}'),
    ("create_relations", [[{"from": "a", "to": "b", "relationType": "R"}]], {}, '{"status": "success"}'),
    ("get_relations", ["test_entity"], {}, '[{"type": "RELATION"}]'),
    ("add_observations", [[{"entity": "e", "content": "c"}]], {}, '{"status": "success"}'),
    ("get_entity_observations", ["test_entity"], {}, '[{"content": "Test"}]'),
])
def test_method_delegation(mock_graph_memory_manager, method_name, args, kwargs, expected):
    """Test method delegation for various methods."""
    # Get the appropriate component manager
    if method_name in ["create_entities", "get_entity", "delete_entity"]:
        mock_component_manager = mock_graph_memory_manager.entity_manager
    elif method_name in ["create_relations", "get_relations"]:
        mock_component_manager = mock_graph_memory_manager.relation_manager
    elif method_name in ["add_observations", "get_entity_observations"]:
        mock_component_manager = mock_graph_memory_manager.observation_manager
    else:
        mock_component_manager = None
    
    # Set the return value for the method
    if mock_component_manager:
        method = getattr(mock_component_manager, method_name.replace("create_relations", "create_relationships")
                                                            .replace("get_relations", "get_relationships"))
        method.return_value = expected
    
    # Call the method on the facade
    result = getattr(mock_graph_memory_manager, method_name)(*args, **kwargs)
    
    # Verify the result
    assert result == expected
    
    # Verify the component method was called with correct arguments
    if mock_component_manager:
        method = getattr(mock_component_manager, method_name.replace("create_relations", "create_relationships")
                                                            .replace("get_relations", "get_relationships"))
        method.assert_called_once()


def test_init(mock_logger):
    """Test initialization of GraphMemoryManager."""
    # Create GraphMemoryManager with all components mocked
    with patch('src.graph_memory.base_manager.BaseManager') as mock_base_manager_cls, \
         patch('src.graph_memory.entity_manager.EntityManager') as mock_entity_manager_cls, \
         patch('src.graph_memory.relation_manager.RelationManager') as mock_relation_manager_cls, \
         patch('src.graph_memory.observation_manager.ObservationManager') as mock_observation_manager_cls, \
         patch('src.graph_memory.search_manager.SearchManager') as mock_search_manager_cls, \
         patch('src.graph_memory.embedding_adapter.EmbeddingAdapter') as mock_embedding_adapter_cls, \
         patch('src.lesson_memory.LessonMemoryManager') as mock_lesson_memory_cls, \
         patch('src.project_memory.ProjectMemoryManager') as mock_project_memory_cls:
        
        # Create manager
        manager = GraphMemoryManager(
            logger=mock_logger,
            embedding_api_key="test_key",
            embedding_model="test_model",
            neo4j_uri="bolt://localhost:7687",
            neo4j_username="neo4j",
            neo4j_password="password"
        )
        
        # Verify all components are created
        assert isinstance(manager.base_manager, MagicMock)
        assert isinstance(manager.entity_manager, MagicMock)
        assert isinstance(manager.relation_manager, MagicMock)
        assert isinstance(manager.observation_manager, MagicMock)
        assert isinstance(manager.search_manager, MagicMock)
        assert isinstance(manager.embedding_adapter, MagicMock)
        assert isinstance(manager.lesson_memory, MagicMock)
        assert isinstance(manager.project_memory, MagicMock)
        
        # Verify environment variables
        assert manager.embedding_api_key == "test_key"
        assert manager.embedding_model == "test_model"
        assert manager.neo4j_uri == "bolt://localhost:7687"
        assert manager.neo4j_user == "neo4j"
        assert manager.neo4j_password == "password"
        
        # Verify base properties
        assert manager.default_project_name == "default"
        assert manager.embedding_enabled is True


def test_initialize_connection(mock_graph_memory_manager):
    """Test initialize connection method."""
    # Configure mocks
    mock_base_manager = mock_graph_memory_manager.base_manager
    mock_embedding_adapter = mock_graph_memory_manager.embedding_adapter
    
    # Mock initialize connection success
    mock_base_manager.initialize_connection.return_value = True
    
    # Call initialize
    result = mock_graph_memory_manager.initialize()
    
    # Verify result
    assert result is True
    
    # Verify base_manager.initialize_connection was called
    mock_base_manager.initialize_connection.assert_called_once()


def test_initialize_connection_error(mock_graph_memory_manager, mock_logger):
    """Test initialize connection handles errors properly."""
    # Configure mocks
    mock_base_manager = mock_graph_memory_manager.base_manager
    
    # Mock initialize connection error
    mock_base_manager.initialize_connection.side_effect = Exception("Connection error")
    
    # Call initialize
    result = mock_graph_memory_manager.initialize()
    
    # Verify result
    assert result is False
    
    # Verify logger.error was called
    mock_logger.error.assert_called()


def test_create_entity(mock_graph_memory_manager, sample_entity):
    """Test entity creation delegation."""
    # Configure mocks
    mock_entity_manager = mock_graph_memory_manager.entity_manager
    
    # Mock entity manager create_entity
    expected_result = json.dumps({"id": "entity-123", "name": "Test Entity"})
    mock_entity_manager.create_entity.return_value = expected_result
    
    # Call create_entity
    result = mock_graph_memory_manager.create_entity(sample_entity)
    
    # Verify result
    assert result == expected_result
    
    # Verify entity_manager.create_entity was called with correct args
    mock_entity_manager.create_entity.assert_called_once_with(sample_entity)


def test_create_entities(mock_graph_memory_manager, sample_entities):
    """Test batch entity creation delegation."""
    # Configure mocks
    mock_entity_manager = mock_graph_memory_manager.entity_manager
    
    # Mock entity manager create_entities
    expected_result = json.dumps({"status": "success", "created": 2})
    mock_entity_manager.create_entities.return_value = expected_result
    
    # Call create_entities
    result = mock_graph_memory_manager.create_entities(sample_entities)
    
    # Verify result
    assert result == expected_result
    
    # Verify entity_manager.create_entities was called with correct args
    mock_entity_manager.create_entities.assert_called_once_with(sample_entities)


def test_get_entity(mock_graph_memory_manager):
    """Test entity retrieval delegation."""
    # Configure mocks
    mock_entity_manager = mock_graph_memory_manager.entity_manager
    
    # Mock entity manager get_entity
    entity_id = "entity-123"
    expected_result = json.dumps({"id": entity_id, "name": "Test Entity"})
    mock_entity_manager.get_entity.return_value = expected_result
    
    # Call get_entity
    result = mock_graph_memory_manager.get_entity(entity_id)
    
    # Verify result
    assert result == expected_result
    
    # Verify entity_manager.get_entity was called with correct args
    mock_entity_manager.get_entity.assert_called_once_with(entity_id)


def test_create_relationship(mock_graph_memory_manager, sample_relation):
    """Test relationship creation delegation."""
    # Configure mocks
    mock_relation_manager = mock_graph_memory_manager.relation_manager
    
    # Mock relation manager create_relationship
    expected_result = json.dumps({"status": "success", "relation_id": "rel-123"})
    mock_relation_manager.create_relationship.return_value = expected_result
    
    # Call create_relationship
    result = mock_graph_memory_manager.create_relationship(sample_relation)
    
    # Verify result
    assert result == expected_result
    
    # Verify relation_manager.create_relationship was called with correct args
    mock_relation_manager.create_relationship.assert_called_once_with(sample_relation)


def test_create_relationships(mock_graph_memory_manager, sample_relations):
    """Test batch relationship creation delegation."""
    # Configure mocks
    mock_relation_manager = mock_graph_memory_manager.relation_manager
    
    # Mock relation manager create_relationships
    expected_result = json.dumps({"status": "success", "created": 2})
    mock_relation_manager.create_relationships.return_value = expected_result
    
    # Call create_relationships
    result = mock_graph_memory_manager.create_relationships(sample_relations)
    
    # Verify result
    assert result == expected_result
    
    # Verify relation_manager.create_relationships was called with correct args
    mock_relation_manager.create_relationships.assert_called_once_with(sample_relations)


def test_get_relationships(mock_graph_memory_manager):
    """Test relationship retrieval delegation."""
    # Configure mocks
    mock_relation_manager = mock_graph_memory_manager.relation_manager
    
    # Mock relation manager get_relationships
    entity_id = "entity-123"
    expected_result = json.dumps([{"id": "rel-1"}, {"id": "rel-2"}])
    mock_relation_manager.get_relationships.return_value = expected_result
    
    # Call get_relationships
    result = mock_graph_memory_manager.get_relationships(entity_id)
    
    # Verify result
    assert result == expected_result
    
    # Verify relation_manager.get_relationships was called with correct args
    mock_relation_manager.get_relationships.assert_called_once_with(entity_id)


def test_add_observation(mock_graph_memory_manager, sample_observation):
    """Test observation addition delegation."""
    # Configure mocks
    mock_observation_manager = mock_graph_memory_manager.observation_manager
    
    # Mock observation manager add_observation
    expected_result = json.dumps({"status": "success", "observation_id": "obs-123"})
    mock_observation_manager.add_observation.return_value = expected_result
    
    # Call add_observation
    result = mock_graph_memory_manager.add_observation(sample_observation)
    
    # Verify result
    assert result == expected_result
    
    # Verify observation_manager.add_observation was called with correct args
    mock_observation_manager.add_observation.assert_called_once_with(sample_observation)


def test_add_observations(mock_graph_memory_manager, sample_observations):
    """Test batch observation addition delegation."""
    # Configure mocks
    mock_observation_manager = mock_graph_memory_manager.observation_manager
    
    # Mock observation manager add_observations_batch
    expected_result = json.dumps({"status": "success", "added": 2})
    mock_observation_manager.add_observations_batch.return_value = expected_result
    
    # Call add_observations
    result = mock_graph_memory_manager.add_observations(sample_observations)
    
    # Verify result
    assert result == expected_result
    
    # Verify observation_manager.add_observations_batch was called with correct args
    mock_observation_manager.add_observations_batch.assert_called_once_with(sample_observations)


def test_get_observations(mock_graph_memory_manager):
    """Test observation retrieval delegation."""
    # Configure mocks
    mock_observation_manager = mock_graph_memory_manager.observation_manager
    
    # Mock observation manager get_observations
    entity_id = "entity-123"
    expected_result = json.dumps([{"id": "obs-1"}, {"id": "obs-2"}])
    mock_observation_manager.get_observations.return_value = expected_result
    
    # Call get_observations
    result = mock_graph_memory_manager.get_observations(entity_id)
    
    # Verify result
    assert result == expected_result
    
    # Verify observation_manager.get_observations was called with correct args
    mock_observation_manager.get_observations.assert_called_once_with(entity_id)


def test_search_nodes(mock_graph_memory_manager):
    """Test search nodes delegation."""
    # Configure mocks
    mock_search_manager = mock_graph_memory_manager._mock_managers['search_manager']
    
    # Mock search manager search_nodes
    query = "test query"
    limit = 10
    expected_result = json.dumps([{"id": "entity-1"}, {"id": "entity-2"}])
    mock_search_manager.search_nodes.return_value = expected_result
    
    # Call search_nodes
    result = mock_graph_memory_manager.search_nodes(query, limit=limit)
    
    # Verify result
    assert result == expected_result
    
    # Verify search_manager.search_nodes was called with correct args
    mock_search_manager.search_nodes.assert_called_once_with(query, limit=limit)


def test_full_text_search(mock_graph_memory_manager):
    """Test full text search delegation."""
    # Configure mocks
    mock_search_manager = mock_graph_memory_manager._mock_managers['search_manager']
    
    # Mock search manager full_text_search
    query = "exact phrase"
    expected_result = json.dumps([{"id": "entity-1"}])
    mock_search_manager.full_text_search.return_value = expected_result
    
    # Call full_text_search
    result = mock_graph_memory_manager.full_text_search(query)
    
    # Verify result
    assert result == expected_result
    
    # Verify search_manager.full_text_search was called with correct args
    mock_search_manager.full_text_search.assert_called_once_with(query)


def test_close_connection(mock_graph_memory_manager):
    """Test close connection delegation."""
    # Call close_connection
    mock_graph_memory_manager.close_connection()
    
    # Verify base_manager.close_connection was called
    mock_graph_memory_manager.base_manager.close_connection.assert_called_once()


def test_create_lesson_container(mock_graph_memory_manager):
    """Test lesson container creation delegation."""
    # Create mock lesson manager
    mock_lesson_manager = MagicMock(spec=LessonMemoryManager)
    mock_graph_memory_manager.lesson_manager = mock_lesson_manager
    
    # Mock lesson manager create_lesson_container
    lesson_data = {"title": "Test Lesson", "description": "Test description"}
    expected_result = json.dumps({"id": "lesson-123", "title": "Test Lesson"})
    mock_lesson_manager.create_lesson_container.return_value = expected_result
    
    # Call create_lesson_container
    result = mock_graph_memory_manager.create_lesson_container(lesson_data)
    
    # Verify result
    assert result == expected_result
    
    # Verify lesson_manager.create_lesson_container was called with correct args
    mock_lesson_manager.create_lesson_container.assert_called_once_with(lesson_data)


def test_create_project_container(mock_graph_memory_manager):
    """Test project container creation delegation."""
    # Create mock project manager
    mock_project_manager = MagicMock(spec=ProjectMemoryManager)
    mock_graph_memory_manager.project_manager = mock_project_manager
    
    # Mock project manager create_project_container
    project_data = {"name": "Test Project", "description": "Test description"}
    expected_result = json.dumps({"id": "project-123", "name": "Test Project"})
    mock_project_manager.create_project_container.return_value = expected_result
    
    # Call create_project_container
    result = mock_graph_memory_manager.create_project_container(project_data)
    
    # Verify result
    assert result == expected_result
    
    # Verify project_manager.create_project_container was called with correct args
    mock_project_manager.create_project_container.assert_called_once_with(project_data)


def test_set_project_name_with_file_operations(mock_graph_memory_manager):
    """Test set project name method with file operations."""
    # Mock project name creation
    project_name = "test-project"
    
    # Mock os.makedirs to do nothing
    with patch("os.makedirs", return_value=None):
        # Mock open to not actually open a file
        with patch("builtins.open", MagicMock()):
            # Call set_project_name
            result = mock_graph_memory_manager.set_project_name(project_name)
    
    # Verify result
    assert result == json.dumps({"status": "success", "project_name": project_name})
    
    # Verify project name was set
    assert mock_graph_memory_manager.project_name == project_name


def test_delete_all_memories(mock_graph_memory_manager):
    """Test delete_all_memories method."""
    # Setup mocks
    mock_base_manager = mock_graph_memory_manager.base_manager
    # Mock base_manager.safe_execute_query
    mock_base_manager.safe_execute_query.return_value = ({"results": [{"deleted": True}]}, None)
    
    # Call the method
    result = mock_graph_memory_manager.delete_all_memories()
    
    # Verify base_manager.safe_execute_query was called with a DELETE query
    mock_base_manager.safe_execute_query.assert_called_once()
    query, params = mock_base_manager.safe_execute_query.call_args[0]
    assert "DELETE" in query.upper()
    
    # Verify result indicates success
    assert json.loads(result)["status"] == "success"


def test_load_project_name(mock_graph_memory_manager):
    """Test load project name method."""
    # Mock file operations
    project_name = "test-project"
    
    # Mock os.path.exists to return True
    with patch("os.path.exists", return_value=True):
        # Mock open to return a file that contains the project name
        mock_file = MagicMock()
        mock_file.__enter__.return_value.read.return_value = project_name
        with patch("builtins.open", return_value=mock_file):
            # Call _load_project_name (it's private but we test it anyway)
            mock_graph_memory_manager._load_project_name()
    
    # Verify project_name was loaded
    assert mock_graph_memory_manager.project_name == project_name


def test_configure_embedding(mock_graph_memory_manager):
    """Test configure embedding method."""
    # Mock file operations
    config = {
        "provider": "openai",
        "model": "text-embedding-ada-002",
        "api_key": "sk-test"
    }
    
    # Mock os.makedirs to do nothing
    with patch("os.makedirs", return_value=None):
        # Mock open to not actually open a file
        with patch("builtins.open", MagicMock()):
            # Call configure_embedding
            result = mock_graph_memory_manager.configure_embedding(config)
    
    # Verify result
    assert "status" in json.loads(result)
    assert json.loads(result)["status"] == "success"


def test_debug_dump_neo4j(mock_graph_memory_manager):
    """Test debug dump neo4j method."""
    # Setup mocks
    mock_base_manager = mock_graph_memory_manager.base_manager
    
    # Mock base_manager.safe_execute_query
    mock_nodes = [{"node": {"properties": {"name": "entity1"}}}]
    mock_rels = [{"rel": {"properties": {"type": "relation1"}}}]
    mock_base_manager.safe_execute_query.side_effect = [
        (mock_nodes, None),
        (mock_rels, None)
    ]
    
    # Call debug_dump_neo4j
    result = mock_graph_memory_manager.debug_dump_neo4j()
    
    # Verify result
    result_obj = json.loads(result)
    assert "nodes" in result_obj
    assert "relationships" in result_obj
    
    # Verify base_manager.safe_execute_query was called twice
    assert mock_base_manager.safe_execute_query.call_count == 2 