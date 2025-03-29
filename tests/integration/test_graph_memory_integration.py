import pytest
from unittest.mock import MagicMock, patch
import json
import os

from src.graph_memory import GraphMemoryManager


@pytest.fixture
def mock_neo4j_connection():
    """Mock Neo4j connection for integration tests."""
    with patch('src.graph_memory.base_manager.GraphDatabase') as mock_driver_class:
        # Configure the mock driver to return suitable results
        driver_instance = MagicMock()
        mock_driver_class.driver.return_value = driver_instance
        
        # Mock successful connection
        driver_instance.verify_connectivity.return_value = True
        
        # Setup session and transaction mocks
        mock_session = MagicMock()
        driver_instance.session.return_value = mock_session
        
        mock_transaction = MagicMock()
        mock_session.__enter__.return_value = mock_session
        mock_session.begin_transaction.return_value = mock_transaction
        mock_transaction.__enter__.return_value = mock_transaction
        
        # Setup default behavior for execute_query
        def execute_query_side_effect(query, parameters=None, database=None):
            # Return empty records and success summary by default
            return [], {"result_available_after": 0, "result_consumed_after": 0}
        
        driver_instance.execute_query.side_effect = execute_query_side_effect
        
        yield driver_instance


@pytest.fixture
def manager(mock_neo4j_connection, mock_logger):
    """Create a GraphMemoryManager with mocked components for integration testing."""
    # Mock LiteLLM Embeddings
    with patch('src.embedding_manager.LiteLLMEmbeddingManager') as mock_embeddings_class:
        # Configure mock embedding manager
        mock_embedding_manager = MagicMock()
        mock_embeddings_class.return_value = mock_embedding_manager
        
        # Set up standard embedding response methods
        mock_embedding_manager.get_embedding.return_value = {"embedding": [0.1, 0.2, 0.3, 0.4], "status": "success"}
        mock_embedding_manager.get_embeddings_batch.return_value = {"embeddings": [[0.1, 0.2, 0.3, 0.4]], "status": "success"}
        mock_embedding_manager.similarity_score.return_value = {"score": 0.85, "status": "success"}
        
        # Create manager with test configuration
        manager = GraphMemoryManager(
            logger=mock_logger,
            neo4j_uri="bolt://localhost:7687",
            neo4j_username="neo4j",
            neo4j_password="password"
        )
        
        # Configure mock_neo4j_connection to return appropriate data for queries
        def execute_query_with_data(query, parameters=None, database=None, database_=None, **kwargs):
            # Prepare records based on query
            records = []
            
            # Entity creation
            if "CREATE (e:Entity" in query:
                records = [{"e": {"id": "entity-123", "name": parameters.get("name", "test_entity") if parameters else "test_entity"}}]
            
            # Entity retrieval
            elif "MATCH (e:Entity" in query and "RETURN e" in query:
                # If we're searching for a specific entity by name
                if parameters and "name" in parameters:
                    records = [{"e": {"id": "entity-123", "name": parameters["name"]}}]
                # Otherwise return all entities
                else:
                    records = [
                        {"e": {"id": "entity-1", "name": "Entity 1"}},
                        {"e": {"id": "entity-2", "name": "Entity 2"}}
                    ]
            
            # Relation creation
            elif "MATCH (from:Entity" in query and "MATCH (to:Entity" in query and "CREATE (from)-[r:" in query:
                rel_type = "RELATED_TO"
                if parameters and "relationType" in parameters:
                    rel_type = parameters["relationType"]
                records = [{"r": {"id": "relation-123", "type": rel_type}}]
            
            # Relation retrieval
            elif "MATCH (e:Entity" in query and "MATCH (e)-[r:" in query:
                records = [
                    {"r": {"id": "relation-1", "type": "CONNECTS_TO"}, 
                     "target": {"id": "entity-2", "name": "Target Entity"}}
                ]
            
            # Observation creation
            elif "MATCH (e:Entity" in query and "CREATE (e)-[r:HAS_OBSERVATION]->(o:Observation" in query:
                content = "Test observation"
                if parameters and "content" in parameters:
                    content = parameters["content"]
                records = [{"o": {"id": "obs-123", "content": content}}]
            
            # Observation retrieval
            elif "MATCH (e:Entity" in query and "MATCH (e)-[:HAS_OBSERVATION]->(o:Observation" in query:
                records = [
                    {"o": {"id": "obs-1", "content": "Observation 1", "type": "OBSERVATION"}},
                    {"o": {"id": "obs-2", "content": "Observation 2", "type": "OBSERVATION"}}
                ]
            
            # Semantic search
            elif "vector:" in query.lower() or "embedding" in query.lower():
                records = [
                    {"entity": {"id": "entity-1", "name": "Semantic Result"}, "score": 0.95}
                ]
            
            # Text search
            elif "where" in query.lower() and ("contains" in query.lower() or "=" in query):
                records = [
                    {"entity": {"id": "entity-2", "name": "Text Search Result"}}
                ]
            
            # Return records and a success summary
            summary = {"result_available_after": 0, "result_consumed_after": 0}
            return records, summary
        
        mock_neo4j_connection.execute_query.side_effect = execute_query_with_data
        
        # Patch initialize to prevent actual connection attempt
        with patch.object(GraphMemoryManager, 'initialize'):
            manager.base_manager.embedding_enabled = True
            manager.initialize()
            yield manager


def test_entity_creation_and_retrieval(manager):
    """Test creating an entity and retrieving it."""
    # Create entity
    entity_data = {
        "name": "TestEntity", 
        "type": "Person",
        "description": "A test entity"
    }
    result = manager.create_entities([entity_data])
    
    # Verify result is a string or dict
    if isinstance(result, str):
        try:
            result_obj = json.loads(result)
            assert isinstance(result_obj, dict)
        except json.JSONDecodeError:
            assert False, f"Invalid JSON response: {result}"
    else:
        # Result is already a dict
        assert isinstance(result, dict)
    
    # Retrieve entity - the exact response structure may vary
    entity_response = manager.get_entity("TestEntity")
    
    # Verify we got a response
    assert entity_response is not None
    
    # If string, try to parse as JSON
    if isinstance(entity_response, str):
        try:
            entity_obj = json.loads(entity_response)
            # In some implementations it might return the entity directly or under an 'entity' key
            if 'entity' in entity_obj:
                entity = entity_obj['entity']
            else:
                entity = entity_obj
        except json.JSONDecodeError:
            assert False, f"Invalid JSON response for entity: {entity_response}"
    else:
        # Already a dict
        entity = entity_response.get('entity', entity_response)
    
    # At this point, entity should be a dict with entity data
    assert isinstance(entity, dict)


def test_entity_relation_integration(manager):
    """Test creating entities and linking them with a relation."""
    # Create source entity
    source_entity = {
        "name": "SourceEntity",
        "type": "Component"
    }
    manager.create_entities([source_entity])
    
    # Create target entity
    target_entity = {
        "name": "TargetEntity",
        "type": "Component"
    }
    manager.create_entities([target_entity])
    
    # Create relation between entities
    relation = {
        "from": "SourceEntity",
        "to": "TargetEntity",
        "relationType": "DEPENDS_ON",
        "properties": {
            "strength": "strong"
        }
    }
    result = manager.create_relations([relation])
    
    # Verify we got a result
    assert result is not None
    
    # Retrieve relations
    relations_response = manager.get_relations("SourceEntity", "DEPENDS_ON")
    
    # Verify we got a response
    assert relations_response is not None
    
    # If string, try to parse as JSON
    if isinstance(relations_response, str):
        try:
            relations_obj = json.loads(relations_response)
            # Some implementations might have relations under a specific key
            relations_list = relations_obj if isinstance(relations_obj, list) else relations_obj.get('relations', [])
        except json.JSONDecodeError:
            assert False, f"Invalid JSON response for relations: {relations_response}"
    else:
        # Already a list or dict
        relations_list = relations_response if isinstance(relations_response, list) else relations_response.get('relations', [])
    
    # Verify we have at least one relation
    assert len(relations_list) > 0


def test_entity_observation_integration(manager):
    """Test creating an entity and adding observations to it."""
    # Create entity
    entity_data = {
        "name": "ObservationTarget",
        "type": "Document"
    }
    manager.create_entities([entity_data])
    
    # Add observations to entity
    observations = [
        {
            "entity": "ObservationTarget",
            "content": "This is observation 1",
            "type": "COMMENT"
        },
        {
            "entity": "ObservationTarget",
            "content": "This is observation 2",
            "type": "ANALYSIS"
        }
    ]
    result = manager.add_observations(observations)
    
    # Verify we got a result
    assert result is not None
    
    # Retrieve observations
    observations_response = manager.get_entity_observations("ObservationTarget")
    
    # Verify we got a response
    assert observations_response is not None
    
    # If string, try to parse as JSON
    if isinstance(observations_response, str):
        try:
            observations_obj = json.loads(observations_response)
            # Some implementations might have observations under a specific key
            observations_list = observations_obj if isinstance(observations_obj, list) else observations_obj.get('observations', [])
        except json.JSONDecodeError:
            assert False, f"Invalid JSON response for observations: {observations_response}"
    else:
        # Already a list or dict
        observations_list = observations_response if isinstance(observations_response, list) else observations_response.get('observations', [])
    
    # Verify we have at least one observation
    assert len(observations_list) > 0


def test_semantic_search_integration(manager):
    """Test semantic search functionality."""
    # Create entities with descriptions
    entities = [
        {
            "name": "Entity1",
            "type": "Concept",
            "description": "A concept about artificial intelligence"
        },
        {
            "name": "Entity2",
            "type": "Concept",
            "description": "A concept about machine learning"
        }
    ]
    manager.create_entities(entities)
    
    # Perform semantic search
    search_results = manager.search_entities(
        search_term="neural networks and deep learning",
        semantic=True
    )
    results_list = json.loads(search_results)
    
    # Verify search results
    assert len(results_list) > 0
    assert results_list[0]["name"] == "Semantic Result"  # From the mock data


def test_text_search_integration(manager):
    """Test text-based search functionality."""
    # Create entities
    entities = [
        {
            "name": "TextEntity1",
            "type": "Document",
            "content": "This document contains information about programming"
        },
        {
            "name": "TextEntity2",
            "type": "Document",
            "content": "This document contains information about databases"
        }
    ]
    manager.create_entities(entities)
    
    # Perform text search
    search_results = manager.search_entities(
        search_term="programming",
        semantic=False
    )
    results_list = json.loads(search_results)
    
    # Verify search results
    assert len(results_list) > 0
    assert results_list[0]["name"] == "Text Search Result"  # From the mock data


def test_end_to_end_entity_lifecycle(manager):
    """Test complete lifecycle of an entity from creation to deletion."""
    # 1. Create entity
    entity_data = {
        "name": "LifecycleEntity",
        "type": "TestEntity",
        "description": "Entity for testing the complete lifecycle"
    }
    creation_result = manager.create_entities([entity_data])
    assert "success" in creation_result
    
    # 2. Add observations
    observation = {
        "entity": "LifecycleEntity",
        "content": "This is a lifecycle test observation",
        "type": "TEST"
    }
    obs_result = manager.add_observations([observation])
    assert "success" in obs_result
    
    # 3. Create relation
    # First create another entity
    manager.create_entities([{"name": "RelatedEntity", "type": "TestEntity"}])
    
    # Then create relation
    relation = {
        "from": "LifecycleEntity",
        "to": "RelatedEntity",
        "relationType": "TEST_RELATION"
    }
    rel_result = manager.create_relations([relation])
    assert "success" in rel_result
    
    # 4. Retrieve and verify entity
    entity = json.loads(manager.get_entity("LifecycleEntity"))
    assert entity["name"] == "LifecycleEntity"
    
    # 5. Retrieve and verify observations
    observations = json.loads(manager.get_entity_observations("LifecycleEntity"))
    assert len(observations) > 0
    
    # 6. Retrieve and verify relations
    relations = json.loads(manager.get_relations("LifecycleEntity", "TEST_RELATION"))
    assert len(relations) > 0
    
    # 7. Delete relation
    del_relation = manager.delete_relation("LifecycleEntity", "RelatedEntity", "TEST_RELATION")
    assert "success" in del_relation
    
    # 8. Delete observations
    del_obs = json.loads(manager.delete_observation("LifecycleEntity", "This is a lifecycle test observation"))
    assert del_obs["status"] == "success"
    
    # 9. Delete entity
    del_entity = manager.delete_entity("LifecycleEntity")
    assert "success" in del_entity


def test_project_and_lesson_memory_integration(manager):
    """Test integration between core memory and project/lesson memory."""
    # Create a project
    project_result = manager.create_project_container("TestProject", "A test project")
    assert "id" in json.loads(project_result)
    
    # Set as current project
    manager.set_project_name("TestProject")
    
    # Create entities with project association
    entity_data = {
        "name": "ProjectEntity",
        "type": "Component",
        "project": "TestProject"
    }
    manager.create_entities([entity_data])
    
    # Create a lesson
    lesson_result = manager.create_lesson("TestLesson", "A test problem")
    assert "id" in json.loads(lesson_result)
    
    # Retrieve project
    project = manager.get_project_container("TestProject")
    assert "TestProject" in project
    
    # Search entities in project
    results = manager.search_entities("Component", project_name="TestProject")
    results_list = json.loads(results)
    assert len(results_list) > 0


def test_node_search_integration(manager):
    """Test general node search across entity types."""
    # Create various entities
    entities = [
        {"name": "Person1", "type": "Person"},
        {"name": "Organization1", "type": "Organization"},
        {"name": "Document1", "type": "Document"}
    ]
    manager.create_entities(entities)
    
    # Search across all node types
    results = manager.search_nodes("test query")
    results_list = json.loads(results)
    
    # Verify results - should match what our mock setup returns
    assert len(results_list) > 0


def test_embedding_configuration(manager):
    """Test embedding configuration application."""
    # Modify embedding config
    config = {
        "provider": "openai",
        "model": "text-embedding-3-large",
        "api_key": "new_test_key",
        "project_name": "ConfigTest"
    }
    
    # Apply config 
    result = manager.apply_client_config(config)
    
    # Verify successful configuration
    assert result["status"] == "success"
    
    # Get current config
    current_config = manager.get_current_config()
    
    # Verify config was applied
    assert current_config["project_name"] == "ConfigTest"


def test_error_handling_integration(manager, mock_neo4j_connection):
    """Test error handling in integration scenarios."""
    # Make Neo4j connection fail temporarily
    original_side_effect = mock_neo4j_connection.execute_query.side_effect
    mock_neo4j_connection.execute_query.side_effect = Exception("Database connection error")
    
    # Attempt to create entity
    result = manager.create_entities([{"name": "ErrorEntity", "type": "Test"}])
    result_obj = json.loads(result)
    
    # Verify error was handled properly
    assert result_obj["status"] == "error"
    assert "Database connection error" in result_obj["message"]
    
    # Restore original behavior
    mock_neo4j_connection.execute_query.side_effect = original_side_effect


def test_memory_persistence_integration(manager, mock_neo4j_connection):
    """Test memory persistence by adding entities and retrieving them."""
    # First clear any existing mocked data
    mock_neo4j_connection.execute_query.reset_mock()
    
    # Define a custom response for entity list query
    entities_response = [
        {"e": {"id": "entity-1", "properties": {"name": "PersistenceTest1"}}},
        {"e": {"id": "entity-2", "properties": {"name": "PersistenceTest2"}}}
    ]
    
    # Configure mock to return our specific entities
    def side_effect_with_persistence(query, parameters=None):
        if "MATCH (e:Entity" in query and "RETURN e" in query and not parameters:
            return entities_response
        return []
        
    mock_neo4j_connection.execute_query.side_effect = side_effect_with_persistence
    
    # Get all memories to test persistence
    memories = manager.get_all_memories()
    memories_list = json.loads(memories)
    
    # Verify our predefined entities are included
    assert len(memories_list) == 2
    assert memories_list[0]["properties"]["name"] == "PersistenceTest1"
    assert memories_list[1]["properties"]["name"] == "PersistenceTest2"


def test_combined_search_integration(manager):
    """Test combined semantic and text search."""
    # Create test entities
    entities = [
        {"name": "SemanticEntity", "type": "Concept", "description": "Semantic description"},
        {"name": "TextEntity", "type": "Document", "content": "Text searchable content"}
    ]
    manager.create_entities(entities)
    
    # Set up mock to return different results for each search type
    def combined_search_side_effect(query, parameters=None):
        if "vector:" in query.lower() or "embedding" in query.lower():
            return [
                {"entity": {"id": "entity-1", "properties": {"name": "Semantic Result"}}, "score": 0.95}
            ]
        elif "where" in query.lower() and ("contains" in query.lower() or "=" in query):
            return [
                {"entity": {"id": "entity-2", "properties": {"name": "Text Search Result"}}}
            ]
        return []
    
    # Apply our combined search mock
    manager.base_manager.execute_query.side_effect = combined_search_side_effect
    
    # Perform combined search
    search_results = manager.search_entities(
        search_term="test query",
        semantic=True,
        combined=True
    )
    results_list = json.loads(search_results)
    
    # Verify both types of results are included
    assert len(results_list) == 2
    assert "Semantic Result" in [r["name"] for r in results_list]
    assert "Text Search Result" in [r["name"] for r in results_list] 