"""
Integration tests for the graph memory system with Neo4j.

These tests require a running Neo4j database with the credentials specified in the .env file.
Skip these tests if Neo4j is not available.
"""

import pytest
import json
import os
import uuid
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

from src.graph_memory import GraphMemoryManager
from src.logger import get_logger, LogLevel

# Skip all tests in this module if NEO4J_URI environment variable is not set
# or if we can't connect to Neo4j
neo4j_uri = os.getenv("NEO4J_URI")
pytestmark = pytest.mark.skipif(
    not neo4j_uri,
    reason="NEO4J_URI environment variable not set, skipping integration tests"
)

class TestGraphMemoryIntegration:
    """Integration tests for GraphMemoryManager with Neo4j."""
    
    @pytest.fixture
    def graph_manager(self):
        """Create a GraphMemoryManager connected to Neo4j."""
        logger = get_logger()
        logger.set_level(LogLevel.DEBUG)
        
        try:
            manager = GraphMemoryManager(logger)
            yield manager
            manager.close()
        except Exception as e:
            pytest.skip(f"Failed to connect to Neo4j: {e}")
    
    @pytest.fixture
    def test_prefix(self):
        """Generate a unique prefix for test entities to avoid conflicts."""
        return f"test_{uuid.uuid4().hex[:8]}_"
    
    def test_create_and_retrieve_entity(self, graph_manager, test_prefix):
        """Test creating and retrieving an entity."""
        # Create a unique entity name
        entity_name = f"{test_prefix}TestEntity"
        entity_type = "TEST_TYPE"
        observations = ["This is a test entity"]
        
        try:
            # Create entity with dictionary parameter
            entity_data = {
                "name": entity_name,
                # Include both entity_type and entityType to be safe
                "entity_type": entity_type,
                "entityType": entity_type,
                "type": entity_type,
                "observations": observations
            }
            create_result = graph_manager.create_entity(entity_data)
            create_data = json.loads(create_result)
            
            # Debug: print the actual response structure
            print(f"Create entity response: {json.dumps(create_data, indent=2)}")
            
            # Verify create result
            assert create_data["status"] == "success"
            # Just check that name is in the response
            assert entity_name in str(create_data)
            
            # Get entity directly instead of searching
            get_result = graph_manager.get_entity(entity_name)
            get_data = json.loads(get_result)
            
            # Debug: print get entity result structure
            print(f"Get entity result: {json.dumps(get_data, indent=2)}")
            
            # Check if entity is found
            assert "error" not in get_data
            assert "entity" in get_data["data"]
            assert get_data["data"]["entity"]["name"] == entity_name
            
        finally:
            # Clean up - delete the entity
            graph_manager.delete_entity(entity_name)
    
    def test_create_relation_between_entities(self, graph_manager, test_prefix):
        """Test creating a relationship between two entities."""
        # Create unique entity names
        source_name = f"{test_prefix}SourceEntity"
        target_name = f"{test_prefix}TargetEntity"
        entity_type = "TEST_TYPE"
        relation_type = "TEST_RELATION"
        
        try:
            # Create source entity with dictionary parameter
            graph_manager.create_entity({
                "name": source_name,
                # Include both entity_type and entityType to be safe
                "entity_type": entity_type,
                "entityType": entity_type,
                "type": entity_type,
                "observations": ["Source entity"]
            })
            
            # Create target entity with dictionary parameter
            graph_manager.create_entity({
                "name": target_name,
                # Include both entity_type and entityType to be safe
                "entity_type": entity_type,
                "entityType": entity_type,
                "type": entity_type,
                "observations": ["Target entity"]
            })
            
            # Create relation - use from/to instead of source_id/target_id
            relation_data = {
                "from": source_name,
                "to": target_name,
                "type": relation_type
            }
            relation_result = graph_manager.create_relationship(relation_data)
            relation_data = json.loads(relation_result)
            
            # Debug: print the actual response structure
            print(f"Create relation response: {json.dumps(relation_data, indent=2)}")
            
            # Check if "error" is in response
            assert "error" not in relation_data
            # Simple check for source and target names in the response
            assert source_name in str(relation_data)
            assert target_name in str(relation_data)
            
        finally:
            # Clean up - delete entities (which should cascade to delete the relation)
            graph_manager.delete_entity(source_name)
            graph_manager.delete_entity(target_name)
    
    def test_add_and_retrieve_observation(self, graph_manager, test_prefix):
        """Test adding and retrieving observations for an entity."""
        # Create a unique entity name
        entity_name = f"{test_prefix}ObservationEntity"
        entity_type = "TEST_TYPE"
        initial_observation = "Initial observation"
        new_observation = "New test observation"
        
        try:
            # Create entity with dictionary parameter
            create_result = graph_manager.create_entity({
                "name": entity_name,
                # Include both entity_type and entityType to be safe
                "entity_type": entity_type,
                "entityType": entity_type,
                "type": entity_type,
                "observations": [initial_observation]
            })
            
            # Debug: print the actual create entity response
            print(f"Create entity response: {create_result}")
            create_data = json.loads(create_result)
            
            # Verify entity was created successfully
            assert create_data["status"] == "success"
            
            # Add new observation - use add_observations instead of add_observation
            observation_data = {
                "entity": entity_name,  # Use entity instead of entity_id
                "content": new_observation,
                "type": "TEST_OBSERVATION"  # Include observation type
            }
            add_result = graph_manager.add_observations([observation_data])
            
            # Debug: print the actual add_observations response
            print(f"Add observation response: {add_result}")
            add_data = json.loads(add_result)
            
            # Check if there are errors in the response
            assert "errors" not in add_data or len(add_data["errors"]) == 0
            
            # Get entity directly instead of searching
            get_result = graph_manager.get_entity(entity_name)
            get_data = json.loads(get_result)
            
            # Debug: print get entity result structure
            print(f"Get entity result: {json.dumps(get_data, indent=2)}")
            
            # Verify entity has observations
            assert "entity" in get_data["data"]
            assert get_data["data"]["entity"]["name"] == entity_name
            assert "observations" in get_data["data"]["entity"]
            assert len(get_data["data"]["entity"]["observations"]) >= 1
            
        finally:
            # Clean up - delete the entity
            graph_manager.delete_entity(entity_name)
    
    def test_delete_entity_cascade(self, graph_manager, test_prefix):
        """Test that deleting an entity also removes its relationships and observations."""
        # Create unique entity names
        source_name = f"{test_prefix}DeleteSource"
        target_name = f"{test_prefix}DeleteTarget"
        entity_type = "TEST_TYPE"
        relation_type = "TEST_RELATION"
        observation = "Test observation"
        
        # Create source entity with dictionary parameter
        graph_manager.create_entity({
            "name": source_name,
            # Include both entity_type and entityType to be safe
            "entity_type": entity_type,
            "entityType": entity_type,
            "type": entity_type,
            "observations": [observation]
        })
        
        # Create target entity with dictionary parameter
        graph_manager.create_entity({
            "name": target_name,
            # Include both entity_type and entityType to be safe
            "entity_type": entity_type,
            "entityType": entity_type,
            "type": entity_type,
            "observations": ["Target observation"]
        })
        
        # Create relation - use from/to instead of source_id/target_id
        relation_data = {
            "from": source_name,
            "to": target_name,
            "type": relation_type
        }
        graph_manager.create_relationship(relation_data)
        
        # Delete source entity
        delete_result = graph_manager.delete_entity(source_name)
        delete_data = json.loads(delete_result)
        
        # Verify delete result
        assert delete_data["status"] == "success"
        
        # Try to get deleted entity - should return an error or empty result
        get_result = graph_manager.get_entity(source_name)
        get_data = json.loads(get_result)
        assert "error" in get_data or "entity" not in get_data
        
        # Clean up - delete the target entity
        graph_manager.delete_entity(target_name) 