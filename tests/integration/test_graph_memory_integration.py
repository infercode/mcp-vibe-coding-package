import pytest
from unittest.mock import MagicMock, patch
import json
import os

from src.graph_memory import GraphMemoryManager


@pytest.fixture
def mock_neo4j_connection():
    """Mock Neo4j connection for integration tests."""
    with patch('src.graph_memory.base_manager.Neo4jDriver') as mock_driver:
        # Configure the mock driver to return suitable results
        driver_instance = MagicMock()
        mock_driver.return_value = driver_instance
        
        # Mock successful connection
        driver_instance.verify_connectivity.return_value = True
        
        # Setup default behavior for execute_query
        def execute_query_side_effect(query, parameters=None):
            # Return an empty list by default
            return []
        
        driver_instance.execute_query.side_effect = execute_query_side_effect
        
        yield driver_instance


@pytest.fixture
def manager(mock_neo4j_connection, mock_logger):
    """Create a GraphMemoryManager with mocked components for integration testing."""
    # Mock OpenAI Embeddings
    with patch('src.graph_memory.embedding_adapter.OpenAIEmbeddings') as mock_embeddings:
        # Configure mock embedding api
        embeddings_instance = MagicMock()
        mock_embeddings.return_value = embeddings_instance
        embeddings_instance.embed_query.return_value = [0.1, 0.2, 0.3, 0.4]
        embeddings_instance.embed_documents.return_value = [[0.1, 0.2, 0.3, 0.4]]
        
        # Create manager with test configuration
        manager = GraphMemoryManager(
            logger=mock_logger,
            embedding_api_key="test_key",
            embedding_model="text-embedding-3-small",
            neo4j_uri="bolt://localhost:7687",
            neo4j_username="neo4j",
            neo4j_password="password"
        )
        
        # Configure mock_neo4j_connection to return appropriate data for queries
        def execute_query_with_data(query, parameters=None):
            # Entity creation
            if "CREATE (e:Entity" in query:
                return [{"e": {"id": "entity-123", "properties": {"name": parameters.get("name", "test_entity")}}}]
            
            # Entity retrieval
            elif "MATCH (e:Entity" in query and "RETURN e" in query:
                # If we're searching for a specific entity by name
                if parameters and "name" in parameters:
                    return [{"e": {"id": "entity-123", "properties": {"name": parameters["name"]}}}]
                # Otherwise return all entities
                return [
                    {"e": {"id": "entity-1", "properties": {"name": "Entity 1"}}},
                    {"e": {"id": "entity-2", "properties": {"name": "Entity 2"}}}
                ]
            
            # Relation creation
            elif "MATCH (from:Entity" in query and "MATCH (to:Entity" in query and "CREATE (from)-[r:" in query:
                return [{"r": {"id": "relation-123", "type": parameters.get("relation_type", "RELATED_TO")}}]
            
            # Relation retrieval
            elif "MATCH (e:Entity" in query and "MATCH (e)-[r:" in query:
                return [
                    {"r": {"id": "relation-1", "type": "CONNECTS_TO", "properties": {}}, 
                     "target": {"id": "entity-2", "properties": {"name": "Target Entity"}}}
                ]
            
            # Observation creation
            elif "MATCH (e:Entity" in query and "CREATE (e)-[r:HAS_OBSERVATION]->(o:Observation" in query:
                return [{"o": {"id": "obs-123", "properties": {"content": parameters.get("content", "Test observation")}}}]
            
            # Observation retrieval
            elif "MATCH (e:Entity" in query and "MATCH (e)-[:HAS_OBSERVATION]->(o:Observation" in query:
                return [
                    {"o": {"id": "obs-1", "properties": {"content": "Observation 1", "type": "OBSERVATION"}}},
                    {"o": {"id": "obs-2", "properties": {"content": "Observation 2", "type": "OBSERVATION"}}}
                ]
            
            # Semantic search
            elif "vector:" in query.lower() or "embedding" in query.lower():
                return [
                    {"entity": {"id": "entity-1", "properties": {"name": "Semantic Result"}}, "score": 0.95}
                ]
            
            # Text search
            elif "where" in query.lower() and ("contains" in query.lower() or "=" in query):
                return [
                    {"entity": {"id": "entity-2", "properties": {"name": "Text Search Result"}}}
                ]
            
            # Default empty response
            return []
        
        mock_neo4j_connection.execute_query.side_effect = execute_query_with_data
        
        # Initialize manager (connect to Neo4j and setup embedding)
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
    
    # Verify successful creation
    assert "success" in result
    
    # Retrieve entity
    entity = manager.get_entity("TestEntity")
    entity_obj = json.loads(entity)
    
    # Verify entity properties
    assert entity_obj["name"] == "TestEntity"


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
    
    # Verify successful relation creation
    assert "success" in result
    
    # Retrieve relations
    relations = manager.get_relations("SourceEntity", "DEPENDS_ON")
    relations_list = json.loads(relations)
    
    # Verify relation properties
    assert len(relations_list) > 0
    assert relations_list[0]["to"]["name"] == "Target Entity"  # From the mock data


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
    
    # Verify successful observation addition
    assert "success" in result
    
    # Retrieve observations
    entity_observations = manager.get_entity_observations("ObservationTarget")
    observations_list = json.loads(entity_observations)
    
    # Verify observations
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