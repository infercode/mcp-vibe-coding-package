import pytest
import os
import json
import uuid
from datetime import datetime

from src.graph_memory import GraphMemoryManager


"""
=============================================================
Integration Tests for GraphMemoryManager with Neo4j Database
=============================================================

These tests interact with a real Neo4j database to verify that the GraphMemoryManager 
correctly handles memory operations in a real environment.

## Running the tests

To run these integration tests, you need:

1. A running Neo4j instance (local or remote)
2. Environment variables configured:
   - NEO4J_URI (default: bolt://localhost:7687)
   - NEO4J_USER (default: neo4j)
   - NEO4J_PASSWORD (default: password)
   - EMBEDDER_PROVIDER=none (to disable embedding requirements)

Run the tests with:
```
PYTHONPATH=. NEO4J_PASSWORD="your-password" EMBEDDER_PROVIDER=none poetry run pytest -xvs tests/graph_memory/test_graph_memory_integration.py
```

Note that the tests create a unique prefix for all test entities to avoid conflicts,
and all test data is cleaned up after tests complete.
"""


# Ensure embeddings are disabled for the test (doesn't require OPENAI_API_KEY)
os.environ["EMBEDDER_PROVIDER"] = "none"

# Configure Neo4j connection
os.environ["NEO4J_URI"] = "bolt://localhost:7687"
os.environ["NEO4J_USER"] = "neo4j"

# Password should be set by the caller with: 
# NEO4J_PASSWORD="your-password" when running the tests

# Generate unique test identifiers to avoid conflicts if tests run in parallel
TEST_PREFIX = f"test_{uuid.uuid4().hex[:8]}"


@pytest.fixture(scope="module")
def graph_manager():
    """
    Create a real GraphMemoryManager instance that connects to Neo4j.
    
    This fixture is module-scoped to reuse the connection across tests.
    """
    # Create a manager instance using environment variables
    print("\nCreating GraphMemoryManager instance...")
    manager = GraphMemoryManager()
    
    # Get Neo4j credentials for logging
    uri = os.environ.get("NEO4J_URI", "bolt://localhost:7687")
    user = os.environ.get("NEO4J_USER", "neo4j")
    embedder = os.environ.get("EMBEDDER_PROVIDER", "none")
    print(f"Using URI: {uri}")
    print(f"Using username: {user}")
    print(f"Using embedder: {embedder}")
    print(f"Password is{'nt' if 'NEO4J_PASSWORD' not in os.environ else ''} set in environment")
    
    # Initialize connection to Neo4j
    print("Initializing Neo4j connection...")
    try:
        manager.base_manager.embedder_provider = "none"  # Override to disable embeddings
        manager.base_manager.embedding_enabled = False
        
        # Attempt to initialize the manager
        success = manager.initialize()
        print(f"Initialization result: {success}")
        
        if not success:
            print("Failed to initialize the Neo4j connection:")
            print("- Check Neo4j is running at the configured URI")
            print("- Verify credentials are correct")
            print("- Ensure EMBEDDER_PROVIDER=none is set")
            
            # Try a direct Neo4j connection test
            try:
                from neo4j import GraphDatabase
                print("\nTesting direct Neo4j connection...")
                driver = GraphDatabase.driver(
                    uri, 
                    auth=(user, os.environ.get("NEO4J_PASSWORD", ""))
                )
                records = list(driver.execute_query(
                    "RETURN 1 as test", 
                    database_="neo4j"
                ).records)
                print(f"Direct connection test: {records}")
                driver.close()
            except Exception as e:
                print(f"Direct connection test failed: {str(e)}")
                
            pytest.skip("Failed to initialize connection to Neo4j. Skipping integration tests.")
            
    except Exception as e:
        print(f"Exception during initialization: {str(e)}")
        pytest.skip(f"Exception during Neo4j initialization: {str(e)}")
    
    # Create a unique project name for this test run
    project_name = f"{TEST_PREFIX}_project"
    print(f"Setting project name: {project_name}")
    manager.set_project_name(project_name)
    
    # Provide the manager to tests
    yield manager
    
    # Clean up all test data created by these tests
    try:
        # Delete all memories for this test project
        print("\nCleaning up test data...")
        manager.delete_all_memories()
        manager.close()
        print("Cleanup completed successfully.")
    except Exception as e:
        print(f"Warning: Cleanup after tests failed: {str(e)}")


def test_create_and_get_entity(graph_manager):
    """Test creating an entity and retrieving it."""
    # Create a unique entity name
    entity_name = f"{TEST_PREFIX}_entity_1"
    
    # Create test entity
    entity = {
        "name": entity_name,
        "entityType": "TestEntity",
        "description": "A test entity for integration testing",
        "properties": {
            "test_property": "test value",
            "created_at": datetime.now().isoformat()
        }
    }
    
    # Create the entity
    create_result = json.loads(graph_manager.create_entity(entity))
    
    # Verify creation was successful
    assert "id" in create_result, f"Entity creation failed: {create_result}"
    
    # Get the entity
    get_result = json.loads(graph_manager.get_entity(entity_name))
    
    # Verify entity data
    assert get_result["name"] == entity_name
    assert get_result["entityType"] == "TestEntity"
    assert "properties" in get_result
    assert get_result["properties"]["test_property"] == "test value"


def test_create_multiple_entities(graph_manager):
    """Test creating multiple entities in a batch."""
    # Create unique entity names
    base_name = f"{TEST_PREFIX}_batch"
    entities = [
        {
            "name": f"{base_name}_entity_{i}",
            "entityType": "BatchEntity",
            "description": f"Batch entity {i}",
            "properties": {"index": i}
        }
        for i in range(3)
    ]
    
    # Create entities in batch
    create_result = json.loads(graph_manager.create_entities(entities))
    
    # Verify creation was successful
    assert "status" in create_result
    assert create_result["status"] == "success"
    assert "created" in create_result
    assert create_result["created"] == 3
    
    # Verify we can retrieve each entity
    for i, entity in enumerate(entities):
        get_result = json.loads(graph_manager.get_entity(entity["name"]))
        assert get_result["name"] == entity["name"]
        assert get_result["properties"]["index"] == i


def test_create_and_query_relationship(graph_manager):
    """Test creating a relationship between entities and querying it."""
    # Create two entities
    entity1_name = f"{TEST_PREFIX}_source"
    entity2_name = f"{TEST_PREFIX}_target"
    
    entity1 = {
        "name": entity1_name,
        "entityType": "SourceEntity",
        "description": "Source entity for relationship test"
    }
    
    entity2 = {
        "name": entity2_name,
        "entityType": "TargetEntity",
        "description": "Target entity for relationship test"
    }
    
    # Create both entities
    graph_manager.create_entity(entity1)
    graph_manager.create_entity(entity2)
    
    # Create relationship
    relation = {
        "from": entity1_name,
        "to": entity2_name,
        "relationType": "CONNECTED_TO",
        "properties": {
            "strength": "strong",
            "created_at": datetime.now().isoformat()
        }
    }
    
    create_result = json.loads(graph_manager.create_relationship(relation))
    
    # Verify relationship creation
    assert "status" in create_result
    assert create_result["status"] == "success"
    
    # Query relationships from entity1
    relations_result = json.loads(graph_manager.get_relationships(entity1_name))
    
    # Verify relationship data
    assert len(relations_result) > 0
    found = False
    for rel in relations_result:
        if rel["target"]["name"] == entity2_name and rel["type"] == "CONNECTED_TO":
            found = True
            assert "properties" in rel
            assert rel["properties"]["strength"] == "strong"
    
    assert found, "Relationship was not found in query results"


def test_add_and_query_observations(graph_manager):
    """Test adding observations to an entity and querying them."""
    # Create an entity
    entity_name = f"{TEST_PREFIX}_obs_entity"
    
    entity = {
        "name": entity_name,
        "entityType": "ObservationEntity",
        "description": "Entity for observation tests"
    }
    
    graph_manager.create_entity(entity)
    
    # Add observations
    observations = [
        {
            "entity": entity_name,
            "content": "First observation about the entity",
            "metadata": {
                "source": "integration_test",
                "confidence": 0.9
            }
        },
        {
            "entity": entity_name,
            "content": "Second observation with different details",
            "metadata": {
                "source": "integration_test",
                "confidence": 0.8
            }
        }
    ]
    
    # Add observations in batch
    add_result = json.loads(graph_manager.add_observations(observations))
    
    # Verify observations were added
    assert add_result["status"] == "success"
    assert add_result["added"] == 2
    
    # Query observations
    obs_result = json.loads(graph_manager.get_observations(entity_name))
    
    # Verify observation data
    assert len(obs_result) >= 2
    
    # Check content of observations
    contents = [obs["content"] for obs in obs_result]
    assert "First observation about the entity" in contents
    assert "Second observation with different details" in contents


def test_search_nodes(graph_manager):
    """Test semantic search functionality."""
    # Create unique entities for search test
    base_name = f"{TEST_PREFIX}_search"
    
    # Create entities with varied descriptions for searching
    entities = [
        {
            "name": f"{base_name}_machine_learning",
            "entityType": "SearchEntity",
            "description": "Entity about machine learning algorithms and neural networks"
        },
        {
            "name": f"{base_name}_database",
            "entityType": "SearchEntity",
            "description": "Entity about database systems and SQL queries"
        },
        {
            "name": f"{base_name}_web_dev",
            "entityType": "SearchEntity",
            "description": "Entity about web development with JavaScript and React"
        }
    ]
    
    # Create all entities
    for entity in entities:
        graph_manager.create_entity(entity)
    
    # Perform search
    search_query = "neural networks and deep learning"
    search_results = json.loads(graph_manager.search_nodes(search_query, limit=5))
    
    # Check if search returned results
    assert len(search_results) > 0
    
    # The ML entity should be among top results
    ml_entity_found = False
    for result in search_results:
        if base_name in result["name"] and "machine_learning" in result["name"]:
            ml_entity_found = True
            break
    
    assert ml_entity_found, "Search didn't return the expected machine learning entity"


def test_create_project_container(graph_manager):
    """Test creating and retrieving a project container."""
    # Create unique project name
    project_name = f"{TEST_PREFIX}_project_container"
    
    # Project container data
    project_data = {
        "name": project_name,
        "description": "Test project container for integration tests",
        "metadata": {
            "owner": "integration_tests",
            "priority": "high"
        },
        "tags": ["test", "integration", "neo4j"]
    }
    
    # Create project container
    create_result = json.loads(graph_manager.create_project_container(project_data))
    
    # Verify creation was successful
    assert "status" in create_result
    assert create_result["status"] == "success"
    
    # Get project container
    get_result = json.loads(graph_manager.get_project_container(project_name))
    
    # Verify project data
    assert "container" in get_result
    container = get_result["container"]
    assert container["name"] == project_name
    assert container["description"] == "Test project container for integration tests"


def test_create_and_get_lesson_container(graph_manager):
    """Test creating a lesson container and retrieving it."""
    # Create unique lesson name
    lesson_title = f"{TEST_PREFIX}_lesson"
    
    # Lesson container data
    lesson_data = {
        "title": lesson_title,
        "description": "Test lesson container for integration tests",
        "difficulty": "intermediate",
        "metadata": {
            "author": "integration_tests",
            "category": "testing"
        },
        "tags": ["test", "integration", "knowledge"]
    }
    
    # Create lesson container
    create_result = json.loads(graph_manager.create_lesson_container(lesson_data))
    
    # Verify creation was successful
    assert "status" in create_result, f"Lesson creation failed: {create_result}"
    assert create_result["status"] == "success"
    assert "container" in create_result
    
    # Get the container ID for later use
    lesson_id = create_result["container"]["id"]
    
    # Get lesson container by ID
    get_result = json.loads(graph_manager.get_lesson_container(lesson_id))
    
    # Verify lesson data
    assert "container" in get_result
    container = get_result["container"]
    assert container["title"] == lesson_title
    assert container["description"] == "Test lesson container for integration tests"
    assert container["difficulty"] == "intermediate"


def test_create_lesson_section(graph_manager):
    """Test creating a section within a lesson container."""
    # Create a lesson container first
    lesson_title = f"{TEST_PREFIX}_sectioned_lesson"
    
    lesson_data = {
        "title": lesson_title,
        "description": "A lesson with sections"
    }
    
    # Create the lesson
    create_result = json.loads(graph_manager.create_lesson_container(lesson_data))
    lesson_id = create_result["container"]["id"]
    
    # Section data
    section_data = {
        "lesson_id": lesson_id,
        "title": "Introduction",
        "content": "This is the introduction section of the test lesson.",
        "order": 1,
        "section_type": "introduction"
    }
    
    # Create the section
    section_result = json.loads(graph_manager.create_lesson_section(section_data))
    
    # Verify section creation
    assert "status" in section_result
    assert section_result["status"] == "success"
    assert "section" in section_result
    
    # Create another section
    section_data2 = {
        "lesson_id": lesson_id,
        "title": "Main Content",
        "content": "This is the main content section of the test lesson.",
        "order": 2,
        "section_type": "content"
    }
    
    section_result2 = json.loads(graph_manager.create_lesson_section(section_data2))
    assert section_result2["status"] == "success"
    
    # Get the lesson again to verify sections are attached
    get_result = json.loads(graph_manager.get_lesson_container(lesson_id))
    
    # Check if sections are included in the response
    # Note: The exact response format depends on how your implementation returns sections
    assert "container" in get_result
    # Ideally we would verify sections here, but implementation details may vary


def test_lesson_relationship(graph_manager):
    """Test creating relationships between lesson containers."""
    # Create two lesson containers
    lesson1_title = f"{TEST_PREFIX}_prerequisite_lesson"
    lesson2_title = f"{TEST_PREFIX}_advanced_lesson"
    
    # Create first lesson
    lesson1_data = {
        "title": lesson1_title,
        "description": "A prerequisite lesson",
        "difficulty": "beginner"
    }
    lesson1_result = json.loads(graph_manager.create_lesson_container(lesson1_data))
    lesson1_id = lesson1_result["container"]["id"]
    
    # Create second lesson
    lesson2_data = {
        "title": lesson2_title,
        "description": "An advanced lesson that requires prerequisite knowledge",
        "difficulty": "advanced"
    }
    lesson2_result = json.loads(graph_manager.create_lesson_container(lesson2_data))
    lesson2_id = lesson2_result["container"]["id"]
    
    # Create relationship between lessons
    relationship_data = {
        "source_id": lesson1_id,
        "target_id": lesson2_id,
        "relationship_type": "PREREQUISITE_FOR",
        "metadata": {
            "strength": "required",
            "notes": "Complete the prerequisite before attempting this lesson"
        }
    }
    
    # Create the relationship
    rel_result = json.loads(graph_manager.create_lesson_relationship(relationship_data))
    
    # Verify relationship creation
    assert "status" in rel_result
    assert rel_result["status"] == "success"
    
    # Ideally, we would verify the relationship by querying it,
    # but the exact implementation of relationship retrieval may vary


def test_full_workflow(graph_manager):
    """Test a full workflow with multiple operations."""
    # Create unique project name for this test
    project_name = f"{TEST_PREFIX}_full_workflow"
    
    # 1. Create a project container
    project_data = {
        "name": project_name,
        "description": "Project for full workflow testing",
        "tags": ["test", "workflow"]
    }
    graph_manager.create_project_container(project_data)
    
    # 2. Create multiple entities
    entities = [
        {
            "name": f"{project_name}_entity_{i}",
            "entityType": "WorkflowEntity",
            "description": f"Workflow test entity {i}"
        }
        for i in range(3)
    ]
    graph_manager.create_entities(entities)
    
    # 3. Create relationships between entities
    for i in range(2):
        relation = {
            "from": f"{project_name}_entity_{i}",
            "to": f"{project_name}_entity_{i+1}",
            "relationType": "RELATED_TO"
        }
        graph_manager.create_relationship(relation)
    
    # 4. Add observations to entities
    for i in range(3):
        observation = {
            "entity": f"{project_name}_entity_{i}",
            "content": f"Observation about entity {i} in the workflow",
        }
        graph_manager.add_observation(observation)
    
    # 5. Query project components
    project_result = json.loads(graph_manager.get_project_container(project_name))
    assert project_result["container"]["name"] == project_name
    
    # 6. Test full text search
    search_result = json.loads(graph_manager.full_text_search("workflow test entity"))
    assert len(search_result) > 0
    
    # Verify we can access one of our entities by name
    entity_name = f"{project_name}_entity_1"
    entity_result = json.loads(graph_manager.get_entity(entity_name))
    assert entity_result["name"] == entity_name
    
    # Verify relationships
    rel_result = json.loads(graph_manager.get_relationships(f"{project_name}_entity_0"))
    assert len(rel_result) > 0
    
    # Verify we can get observations
    obs_result = json.loads(graph_manager.get_observations(f"{project_name}_entity_0"))
    assert len(obs_result) > 0


def test_complex_knowledge_graph(graph_manager):
    """Test creating a more complex knowledge graph with interconnected entities."""
    # Base name for all entities in this test
    base_name = f"{TEST_PREFIX}_complex"
    
    # Create a project container
    project_name = f"{base_name}_project"
    project_data = {"name": project_name, "description": "Complex knowledge graph test project"}
    graph_manager.create_project_container(project_data)
    
    # Create different types of entities
    entity_types = ["Person", "Organization", "Technology", "Project", "Concept"]
    entities_by_type = {}
    
    for entity_type in entity_types:
        entities = []
        # Create 3 entities of each type
        for i in range(3):
            entity_name = f"{base_name}_{entity_type.lower()}_{i}"
            entity = {
                "name": entity_name,
                "entityType": entity_type,
                "description": f"A {entity_type.lower()} entity for complex graph testing"
            }
            graph_manager.create_entity(entity)
            entities.append(entity_name)
        entities_by_type[entity_type] = entities
    
    # Create relationships between entities
    relationship_types = {
        ("Person", "Organization"): "WORKS_FOR",
        ("Person", "Technology"): "KNOWS",
        ("Organization", "Technology"): "USES",
        ("Person", "Project"): "CONTRIBUTES_TO",
        ("Technology", "Project"): "USED_IN",
        ("Concept", "Technology"): "IMPLEMENTED_BY",
        ("Person", "Concept"): "UNDERSTANDS"
    }
    
    # Create relationships
    created_relationships = []
    for (source_type, target_type), relation_type in relationship_types.items():
        source_entities = entities_by_type[source_type]
        target_entities = entities_by_type[target_type]
        
        # Create at least one relationship for each pair of types
        for i in range(min(len(source_entities), len(target_entities))):
            relation = {
                "from": source_entities[i],
                "to": target_entities[i],
                "relationType": relation_type,
                "properties": {"test_index": i}
            }
            result = graph_manager.create_relationship(relation)
            created_relationships.append((source_entities[i], target_entities[i], relation_type))
    
    # Add observations to some entities
    for entity_type, entities in entities_by_type.items():
        for i, entity_name in enumerate(entities):
            observation = {
                "entity": entity_name,
                "content": f"This {entity_type.lower()} entity has index {i}",
                "metadata": {"test_id": f"{base_name}_{i}"}
            }
            graph_manager.add_observation(observation)
    
    # Verify the graph is searchable
    search_results = json.loads(graph_manager.search_nodes("technology", limit=10))
    assert len(search_results) > 0
    
    # Verify we can look up relationships
    for source, target, rel_type in created_relationships[:3]:  # Check first 3 relationships
        rel_results = json.loads(graph_manager.get_relationships(source))
        
        # Find the specific relationship we created
        rel_found = False
        for rel in rel_results:
            if rel["target"]["name"] == target and rel["type"] == rel_type:
                rel_found = True
                break
        
        assert rel_found, f"Relationship {source} -[{rel_type}]-> {target} not found" 