import pytest
from unittest.mock import MagicMock, patch
import json

from src.graph_memory.base_manager import BaseManager
from src.graph_memory import GraphMemoryManager
from src.graph_memory.entity_manager import EntityManager
from src.graph_memory.relation_manager import RelationManager
from src.graph_memory.observation_manager import ObservationManager
from src.graph_memory.search_manager import SearchManager
from src.graph_memory.embedding_adapter import EmbeddingAdapter
from src.lesson_memory import LessonMemoryManager
from src.project_memory import ProjectMemoryManager


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
def mock_base_manager(mock_neo4j_driver, mock_logger):
    """Mock base manager with a Neo4j driver."""
    with patch('neo4j.GraphDatabase') as mock_graph_db:
        mock_graph_db.driver.return_value = mock_neo4j_driver
        
        base_manager = MagicMock()
        base_manager.driver = mock_neo4j_driver
        base_manager.execute_read.side_effect = lambda query, **params: json.dumps([{"id": "entity-1"}])
        base_manager.execute_write.side_effect = lambda query, **params: json.dumps({"id": "entity-1"})
        
        return base_manager


@pytest.fixture
def mock_entity_manager(mock_base_manager, mock_logger):
    """Mock entity manager."""
    entity_manager = MagicMock()
    entity_manager.create_entity.return_value = json.dumps({"id": "entity-1", "name": "Test Entity"})
    entity_manager.get_entity.return_value = json.dumps({"id": "entity-1", "name": "Test Entity", "type": "TestEntity"})
    entity_manager.create_entities.return_value = json.dumps({"status": "success", "created": 2, "entity_ids": ["entity-1", "entity-2"]})
    
    return entity_manager


@pytest.fixture
def mock_relation_manager(mock_base_manager, mock_entity_manager, mock_logger):
    """Mock relation manager."""
    relation_manager = MagicMock()
    relation_manager.create_relationship.return_value = json.dumps({"status": "success", "relation_id": "rel-1"})
    relation_manager.get_relationships.return_value = json.dumps([
        {"relation": {"id": "rel-1", "type": "RELATED_TO"}, "target": {"id": "entity-2", "name": "Related Entity"}}
    ])
    relation_manager.create_relationships.return_value = json.dumps({"status": "success", "created": 2, "relation_ids": ["rel-1", "rel-2"]})
    
    return relation_manager


@pytest.fixture
def mock_observation_manager(mock_base_manager, mock_entity_manager, mock_logger):
    """Mock observation manager."""
    observation_manager = MagicMock()
    observation_manager.add_observation.return_value = json.dumps({"status": "success", "observation_id": "obs-1"})
    observation_manager.get_observations.return_value = json.dumps([
        {"id": "obs-1", "content": "Test observation", "created_at": "2023-01-01T00:00:00Z"}
    ])
    observation_manager.add_observations_batch.return_value = json.dumps({"status": "success", "added": 2, "observation_ids": ["obs-1", "obs-2"]})
    
    return observation_manager


@pytest.fixture
def mock_embedding_adapter(mock_logger):
    """Mock embedding adapter."""
    embedding_adapter = MagicMock()
    embedding_adapter.is_available.return_value = True
    embedding_adapter.get_embedding.return_value = [0.1] * 1536  # OpenAI embedding dimension
    embedding_adapter.embedding_dimension = 1536
    
    return embedding_adapter


@pytest.fixture
def mock_search_manager(mock_base_manager, mock_embedding_adapter, mock_logger):
    """Mock search manager."""
    search_manager = MagicMock()
    search_manager.search_nodes.return_value = json.dumps([
        {"id": "entity-1", "name": "Search Result 1", "score": 0.95},
        {"id": "entity-2", "name": "Search Result 2", "score": 0.85}
    ])
    search_manager.full_text_search.return_value = json.dumps([
        {"id": "entity-3", "name": "Text Result", "content": "Exact match found"}
    ])
    
    return search_manager


@pytest.fixture
def mock_lesson_manager(mock_logger):
    """Mock lesson manager."""
    lesson_manager = MagicMock()
    lesson_manager.create_lesson_container.return_value = json.dumps({"id": "lesson-1", "title": "Test Lesson"})
    lesson_manager.get_lesson_container.return_value = json.dumps({"id": "lesson-1", "title": "Test Lesson"})
    lesson_manager.create_lesson_section.return_value = json.dumps({"status": "success", "section_id": "section-1"})
    
    return lesson_manager


@pytest.fixture
def mock_project_manager(mock_logger):
    """Mock project manager."""
    project_manager = MagicMock()
    project_manager.create_project_container.return_value = json.dumps({"id": "project-1", "name": "Test Project"})
    project_manager.get_project_container.return_value = json.dumps({"id": "project-1", "name": "Test Project"})
    project_manager.create_component.return_value = json.dumps({"status": "success", "component_id": "component-1"})
    
    return project_manager


@pytest.fixture
def mock_graph_memory_manager(
    mock_base_manager, 
    mock_entity_manager, 
    mock_relation_manager, 
    mock_observation_manager,
    mock_search_manager,
    mock_embedding_adapter,
    mock_lesson_manager,
    mock_project_manager,
    mock_logger
):
    """Mock graph memory manager with all component managers mocked."""
    graph_memory_manager = MagicMock()
    
    # Set component managers
    graph_memory_manager.base_manager = mock_base_manager
    graph_memory_manager.entity_manager = mock_entity_manager
    graph_memory_manager.relation_manager = mock_relation_manager
    graph_memory_manager.observation_manager = mock_observation_manager
    graph_memory_manager.search_manager = mock_search_manager
    graph_memory_manager.embedding_adapter = mock_embedding_adapter
    graph_memory_manager.lesson_manager = mock_lesson_manager
    graph_memory_manager.project_manager = mock_project_manager
    
    # Forward calls to respective managers
    graph_memory_manager.create_entity.side_effect = mock_entity_manager.create_entity
    graph_memory_manager.get_entity.side_effect = mock_entity_manager.get_entity
    graph_memory_manager.create_entities.side_effect = mock_entity_manager.create_entities
    
    graph_memory_manager.create_relationship.side_effect = mock_relation_manager.create_relationship
    graph_memory_manager.get_relationships.side_effect = mock_relation_manager.get_relationships
    graph_memory_manager.create_relationships.side_effect = mock_relation_manager.create_relationships
    
    graph_memory_manager.add_observation.side_effect = mock_observation_manager.add_observation
    graph_memory_manager.get_observations.side_effect = mock_observation_manager.get_observations
    graph_memory_manager.add_observations.side_effect = mock_observation_manager.add_observations_batch
    
    graph_memory_manager.search_nodes.side_effect = mock_search_manager.search_nodes
    graph_memory_manager.full_text_search.side_effect = mock_search_manager.full_text_search
    
    graph_memory_manager.create_lesson_container.side_effect = mock_lesson_manager.create_lesson_container
    graph_memory_manager.create_lesson_section.side_effect = mock_lesson_manager.create_lesson_section
    
    graph_memory_manager.create_project_container.side_effect = mock_project_manager.create_project_container
    graph_memory_manager.create_component.side_effect = mock_project_manager.create_component
    
    # Mock configuration functions
    graph_memory_manager.set_project_name.return_value = json.dumps({"status": "success", "project_name": "test-project"})
    graph_memory_manager.configure_embedding.return_value = json.dumps({"status": "success"})
    
    return graph_memory_manager


# Test data fixtures
@pytest.fixture
def sample_entity():
    """Sample entity for testing."""
    return {
        "name": "test_entity",
        "entityType": "TestType",
        "description": "A test entity for pytest",
        "properties": {
            "key1": "value1",
            "key2": "value2"
        },
        "tags": ["test", "pytest", "entity"]
    }


@pytest.fixture
def sample_entities():
    """Sample batch of entities for testing."""
    return [
        {
            "name": "test_entity_1",
            "entityType": "TestType",
            "description": "First test entity"
        },
        {
            "name": "test_entity_2",
            "entityType": "TestType",
            "description": "Second test entity"
        }
    ]


@pytest.fixture
def sample_relation():
    """Sample relation for testing."""
    return {
        "from": "source_entity",
        "to": "target_entity",
        "relationType": "TEST_RELATION",
        "properties": {"weight": 0.9}
    }


@pytest.fixture
def sample_relations():
    """Sample batch of relations for testing."""
    return [
        {
            "from": "source_entity_1",
            "to": "target_entity_1",
            "relationType": "TEST_RELATION"
        },
        {
            "from": "source_entity_2",
            "to": "target_entity_2",
            "relationType": "TEST_RELATION"
        }
    ]


@pytest.fixture
def sample_observation():
    """Sample observation for testing."""
    return {
        "entity": "test_entity",
        "content": "This is a test observation."
    }


@pytest.fixture
def sample_observations():
    """Sample batch of observations for testing."""
    return [
        {
            "entity": "test_entity_1",
            "content": "First test observation"
        },
        {
            "entity": "test_entity_2",
            "content": "Second test observation"
        }
    ]


@pytest.fixture
def performance_test_entities():
    """Generate a larger set of entities for performance testing."""
    return [
        {
            "name": f"benchmark_entity_{i}",
            "entityType": "BenchmarkType",
            "description": f"Entity for benchmarking operations {i}",
            "tags": ["benchmark", "test", f"tag_{i}"]
        }
        for i in range(100)  # Create 100 sample entities
    ]


@pytest.fixture
def performance_test_relations():
    """Generate a larger set of relations for performance testing."""
    return [
        {
            "from": f"benchmark_entity_{i}",
            "to": f"benchmark_entity_{i+1}",
            "relationType": "RELATED_TO"
        }
        for i in range(99)  # Create 99 sample relations
    ]


@pytest.fixture
def performance_test_observations():
    """Generate a larger set of observations for performance testing."""
    return [
        {
            "entity": f"benchmark_entity_{i}",
            "content": f"Observation {j} for entity {i}"
        }
        for i in range(10) for j in range(10)  # Create 100 sample observations
    ] 