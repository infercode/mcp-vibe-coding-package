import unittest
from unittest.mock import patch, MagicMock

from src.graph_memory.base_manager import BaseManager
from src.graph_memory.entity_manager import EntityManager
from src.graph_memory.relation_manager import RelationManager
from src.graph_memory.observation_manager import ObservationManager
from src.graph_memory.search_manager import SearchManager
from src.graph_memory.embedding_adapter import EmbeddingAdapter

class TestGraphMemoryIntegration(unittest.TestCase):
    """Integration tests for the graph memory system components."""
    
    def setUp(self):
        """Set up test environment before each test."""
        # Create a mock logger
        self.mock_logger = MagicMock()
        
        # Set up real components with mocked Neo4j connection
        with patch('neo4j.GraphDatabase.driver') as self.mock_driver:
            # Set up mock driver instance
            self.mock_driver_instance = MagicMock()
            self.mock_driver.return_value = self.mock_driver_instance
            
            # Create real components
            self.base_manager = BaseManager(self.mock_logger)
            self.base_manager.initialize()
            
            # Mock the execute_query method to avoid actual database calls
            self.base_manager.execute_query = MagicMock()
            
            # Create dependent components
            self.entity_manager = EntityManager(self.base_manager)
            self.relation_manager = RelationManager(self.base_manager)
            self.observation_manager = ObservationManager(self.base_manager)
            
            # Mock embedding adapter
            self.mock_embedding_adapter = MagicMock(spec=EmbeddingAdapter)
            self.search_manager = SearchManager(self.base_manager, self.mock_embedding_adapter)
        
        # Sample test entity data
        self.test_entity = {
            "name": "integration_test_entity",
            "entityType": "TestEntity",
            "description": "An entity for integration testing",
            "tags": ["test", "integration"]
        }
        
        # Sample test relation data
        self.test_relation = {
            "source_entity": "integration_test_entity",
            "target_entity": "related_entity",
            "relation_type": "TEST_RELATION",
            "properties": {"strength": 0.9}
        }
        
        # Sample test observation data
        self.test_observation = {
            "entity_name": "integration_test_entity",
            "observation_type": "TEST_OBSERVATION",
            "content": "This is a test observation.",
            "metadata": {"confidence": 0.8}
        }
    
    def tearDown(self):
        """Clean up after each test."""
        self.base_manager.close()
    
    def test_entity_creation_and_retrieval(self):
        """Test entity creation and retrieval flow."""
        # Mock responses for entity creation
        self.base_manager.execute_query.return_value = [{
            "e": {"id": 123, "properties": self.test_entity}
        }]
        
        # Create an entity
        created_entity = self.entity_manager.create_entity(self.test_entity)
        
        # Verify entity creation query was called
        self.base_manager.execute_query.assert_called_once()
        # Reset mock for next operation
        self.base_manager.execute_query.reset_mock()
        
        # Mock response for entity retrieval
        self.base_manager.execute_query.return_value = [{
            "e": {"id": 123, "properties": self.test_entity}
        }]
        
        # Retrieve the entity
        retrieved_entity = self.entity_manager.get_entity("integration_test_entity")
        
        # Assertions
        self.assertEqual(retrieved_entity.get("name"), "integration_test_entity")
        self.assertEqual(retrieved_entity.get("entityType"), "TestEntity")
    
    def test_entity_relation_integration(self):
        """Test integration between entity and relation managers."""
        # Mock entity existence checks
        self.base_manager.execute_query.return_value = [
            {"e": {"id": 123, "properties": {"name": "integration_test_entity"}}},
            {"e": {"id": 456, "properties": {"name": "related_entity"}}}
        ]
        
        # Mock relationship creation
        relation_result = {
            "source": {"id": 123, "name": "integration_test_entity"},
            "target": {"id": 456, "name": "related_entity"},
            "relationship": {"type": "TEST_RELATION", "properties": {"strength": 0.9}}
        }
        self.base_manager.execute_query.return_value = [relation_result]
        
        # Create relationship
        created_relation = self.relation_manager.create_relationship(
            source_entity="integration_test_entity",
            target_entity="related_entity",
            relation_type="TEST_RELATION",
            properties={"strength": 0.9}
        )
        
        # Verify relationship creation
        self.assertIsNotNone(created_relation)
        self.assertEqual(created_relation.get("type"), "TEST_RELATION")
        self.assertEqual(created_relation.get("source_name"), "integration_test_entity")
        self.assertEqual(created_relation.get("target_name"), "related_entity")
        
        # Reset mock for next operation
        self.base_manager.execute_query.reset_mock()
        
        # Mock getting relationships
        self.base_manager.execute_query.return_value = [relation_result]
        
        # Get relationships
        relationships = self.relation_manager.get_relationships(
            entity_name="integration_test_entity"
        )
        
        # Assertions
        self.assertEqual(len(relationships), 1)
        self.assertEqual(relationships[0].get("type"), "TEST_RELATION")
    
    def test_entity_observation_integration(self):
        """Test integration between entity and observation managers."""
        # Mock entity existence check
        self.base_manager.execute_query.return_value = [
            {"e": {"id": 123, "properties": {"name": "integration_test_entity"}}}
        ]
        
        # Mock observation creation
        observation_result = {
            "entity": {"id": 123, "name": "integration_test_entity"},
            "observation": {
                "id": 789,
                "type": "TEST_OBSERVATION",
                "content": "This is a test observation.",
                "metadata": {"confidence": 0.8},
                "timestamp": "2025-03-29T10:00:00"
            }
        }
        self.base_manager.execute_query.return_value = [observation_result]
        
        # Add observation
        added_observation = self.observation_manager.add_observation(
            entity_name="integration_test_entity",
            observation_type="TEST_OBSERVATION",
            content="This is a test observation.",
            metadata={"confidence": 0.8}
        )
        
        # Verify observation creation
        self.assertIsNotNone(added_observation)
        self.assertEqual(added_observation.get("type"), "TEST_OBSERVATION")
        self.assertEqual(added_observation.get("content"), "This is a test observation.")
        
        # Reset mock for next operation
        self.base_manager.execute_query.reset_mock()
        
        # Mock getting observations
        self.base_manager.execute_query.return_value = [observation_result]
        
        # Get observations
        observations = self.observation_manager.get_entity_observations(
            entity_name="integration_test_entity"
        )
        
        # Assertions
        self.assertEqual(len(observations), 1)
        self.assertEqual(observations[0].get("type"), "TEST_OBSERVATION")
    
    def test_end_to_end_entity_lifecycle(self):
        """Test end-to-end entity lifecycle with all components."""
        # Setup mock responses for different operations
        entity_create_result = {"e": {"id": 123, "properties": self.test_entity}}
        relation_create_result = {
            "source": {"id": 123, "name": "integration_test_entity"},
            "target": {"id": 456, "name": "related_entity"},
            "relationship": {"type": "TEST_RELATION", "properties": {"strength": 0.9}}
        }
        observation_create_result = {
            "entity": {"id": 123, "name": "integration_test_entity"},
            "observation": {
                "id": 789,
                "type": "TEST_OBSERVATION",
                "content": "This is a test observation.",
                "metadata": {"confidence": 0.8},
                "timestamp": "2025-03-29T10:00:00"
            }
        }
        search_result = {"e": {"id": 123, "properties": self.test_entity}}
        
        # Mock embedding generation
        self.mock_embedding_adapter.generate_embedding.return_value = [0.1, 0.2, 0.3]
        
        # 1. Create entity
        self.base_manager.execute_query.return_value = [entity_create_result]
        entity = self.entity_manager.create_entity(self.test_entity)
        self.assertIsNotNone(entity)
        self.base_manager.execute_query.reset_mock()
        
        # 2. Add a relationship
        self.base_manager.execute_query.return_value = [relation_create_result]
        relation = self.relation_manager.create_relationship(
            source_entity="integration_test_entity",
            target_entity="related_entity",
            relation_type="TEST_RELATION",
            properties={"strength": 0.9}
        )
        self.assertIsNotNone(relation)
        self.base_manager.execute_query.reset_mock()
        
        # 3. Add an observation
        self.base_manager.execute_query.return_value = [observation_create_result]
        observation = self.observation_manager.add_observation(
            entity_name="integration_test_entity",
            observation_type="TEST_OBSERVATION",
            content="This is a test observation.",
            metadata={"confidence": 0.8}
        )
        self.assertIsNotNone(observation)
        self.base_manager.execute_query.reset_mock()
        
        # 4. Search for the entity
        self.base_manager.execute_query.return_value = [
            {"entity": {"id": 123, "properties": self.test_entity}, "score": 0.95}
        ]
        results = self.search_manager.semantic_search("integration test")
        self.assertEqual(len(results), 1)
        self.assertEqual(results[0].get("name"), "integration_test_entity")
        self.base_manager.execute_query.reset_mock()
        
        # 5. Delete the entity and verify cascading effects
        self.base_manager.execute_query.return_value = [{"success": True, "count": 1}]
        deleted = self.entity_manager.delete_entity("integration_test_entity")
        self.assertTrue(deleted)
    
    def test_cross_component_search(self):
        """Test search functionality that spans multiple components."""
        # Mock embedding generation
        self.mock_embedding_adapter.generate_embedding.return_value = [0.1, 0.2, 0.3]
        
        # Mock combined search results
        combined_search_results = [
            {
                "entity": {"id": 123, "properties": self.test_entity},
                "text_score": 0.85,
                "semantic_score": 0.9,
                "combined_score": 0.875
            }
        ]
        self.base_manager.execute_query.return_value = combined_search_results
        
        # Perform combined search
        results = self.search_manager.combined_search(
            text_query="integration test",
            semantic_query="entity for testing",
            entity_type="TestEntity",
            text_weight=0.4,
            semantic_weight=0.6
        )
        
        # Assertions
        self.assertEqual(len(results), 1)
        self.assertEqual(results[0].get("name"), "integration_test_entity")
        self.assertEqual(results[0].get("combined_score"), 0.875)
        
        # Verify embedding generation was called
        self.mock_embedding_adapter.generate_embedding.assert_called_once_with("entity for testing")
        
        # Verify query execution
        self.base_manager.execute_query.assert_called_once()

if __name__ == '__main__':
    unittest.main() 