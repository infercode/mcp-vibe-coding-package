import unittest
from unittest.mock import patch, MagicMock

from src.graph_memory.relation_manager import RelationManager
from src.graph_memory.base_manager import BaseManager

class TestRelationManager(unittest.TestCase):
    """Test suite for RelationManager class responsible for managing relationships between entities."""
    
    def setUp(self):
        """Set up test environment before each test."""
        # Create a mock base manager
        self.mock_base_manager = MagicMock(spec=BaseManager)
        
        # Create relation manager with mock base manager
        self.relation_manager = RelationManager(self.mock_base_manager)
        
        # Sample relationship data for testing
        self.sample_relation = {
            "source_entity": "source_test",
            "target_entity": "target_test",
            "relation_type": "TEST_RELATION",
            "properties": {
                "strength": 0.8,
                "description": "A test relationship"
            }
        }
    
    def test_create_relationship(self):
        """Test creation of a relationship between entities."""
        # Setup mock response from execute_query
        mock_result = {
            "source": {"name": "source_test", "id": 123},
            "target": {"name": "target_test", "id": 456},
            "relationship": {"type": "TEST_RELATION", "properties": {"strength": 0.8}}
        }
        self.mock_base_manager.execute_query.return_value = [mock_result]
        
        # Call method under test
        result = self.relation_manager.create_relationship(
            source_entity="source_test",
            target_entity="target_test",
            relation_type="TEST_RELATION",
            properties={"strength": 0.8, "description": "A test relationship"}
        )
        
        # Assertions
        self.assertIsNotNone(result)
        self.assertEqual(result.get("source_id"), 123)
        self.assertEqual(result.get("target_id"), 456)
        self.assertEqual(result.get("type"), "TEST_RELATION")
        
        # Verify query execution
        self.mock_base_manager.execute_query.assert_called_once()
        # Verify that the query contains CREATE or MERGE
        call_args = self.mock_base_manager.execute_query.call_args[0]
        self.assertTrue("CREATE" in call_args[0] or "MERGE" in call_args[0])
    
    def test_create_relationship_entities_not_found(self):
        """Test handling of relationship creation when entities don't exist."""
        # Setup mock response (empty result indicating entities not found)
        self.mock_base_manager.execute_query.return_value = []
        
        # Call method under test
        result = self.relation_manager.create_relationship(
            source_entity="nonexistent_source",
            target_entity="nonexistent_target",
            relation_type="TEST_RELATION"
        )
        
        # Assertions
        self.assertIsNone(result)
        
        # Verify query execution
        self.mock_base_manager.execute_query.assert_called_once()
    
    def test_get_relationships(self):
        """Test retrieving relationships for an entity."""
        # Setup mock response
        mock_results = [
            {
                "source": {"name": "source_test", "id": 123},
                "target": {"name": "target_test", "id": 456},
                "relationship": {"type": "TEST_RELATION", "properties": {"strength": 0.8}}
            },
            {
                "source": {"name": "source_test", "id": 123},
                "target": {"name": "target_test_2", "id": 789},
                "relationship": {"type": "ANOTHER_RELATION", "properties": {"strength": 0.5}}
            }
        ]
        self.mock_base_manager.execute_query.return_value = mock_results
        
        # Call method under test
        results = self.relation_manager.get_relationships(entity_name="source_test")
        
        # Assertions
        self.assertEqual(len(results), 2)
        self.assertEqual(results[0].get("source_name"), "source_test")
        self.assertEqual(results[0].get("target_name"), "target_test")
        self.assertEqual(results[0].get("type"), "TEST_RELATION")
        
        # Verify query execution
        self.mock_base_manager.execute_query.assert_called_once()
        # Verify that the query contains MATCH
        call_args = self.mock_base_manager.execute_query.call_args[0]
        self.assertTrue("MATCH" in call_args[0])
    
    def test_get_relationships_by_type(self):
        """Test retrieving relationships by type."""
        # Setup mock response
        mock_results = [
            {
                "source": {"name": "source_test", "id": 123},
                "target": {"name": "target_test", "id": 456},
                "relationship": {"type": "TEST_RELATION", "properties": {"strength": 0.8}}
            }
        ]
        self.mock_base_manager.execute_query.return_value = mock_results
        
        # Call method under test
        results = self.relation_manager.get_relationships_by_type(
            relation_type="TEST_RELATION"
        )
        
        # Assertions
        self.assertEqual(len(results), 1)
        self.assertEqual(results[0].get("type"), "TEST_RELATION")
        
        # Verify query execution
        self.mock_base_manager.execute_query.assert_called_once()
        # Verify that the query contains the relation type filter
        call_args = self.mock_base_manager.execute_query.call_args[0]
        self.assertTrue("TEST_RELATION" in call_args[0])
    
    def test_update_relationship(self):
        """Test updating a relationship's properties."""
        # Setup mock response
        updated_props = {"strength": 0.9, "description": "Updated description"}
        mock_result = {
            "source": {"name": "source_test", "id": 123},
            "target": {"name": "target_test", "id": 456},
            "relationship": {"type": "TEST_RELATION", "properties": updated_props}
        }
        self.mock_base_manager.execute_query.return_value = [mock_result]
        
        # Call method under test
        result = self.relation_manager.update_relationship(
            source_entity="source_test",
            target_entity="target_test",
            relation_type="TEST_RELATION",
            properties=updated_props
        )
        
        # Assertions
        self.assertIsNotNone(result)
        self.assertEqual(result.get("properties").get("strength"), 0.9)
        self.assertEqual(result.get("properties").get("description"), "Updated description")
        
        # Verify query execution
        self.mock_base_manager.execute_query.assert_called_once()
        # Verify that the query contains SET for updates
        call_args = self.mock_base_manager.execute_query.call_args[0]
        self.assertTrue("SET" in call_args[0])
    
    def test_delete_relationship(self):
        """Test deleting a relationship."""
        # Setup mock response
        self.mock_base_manager.execute_query.return_value = [{"deleted": True, "count": 1}]
        
        # Call method under test
        result = self.relation_manager.delete_relationship(
            source_entity="source_test",
            target_entity="target_test",
            relation_type="TEST_RELATION"
        )
        
        # Assertions
        self.assertTrue(result)
        
        # Verify query execution
        self.mock_base_manager.execute_query.assert_called_once()
        # Verify that the query contains DELETE
        call_args = self.mock_base_manager.execute_query.call_args[0]
        self.assertTrue("DELETE" in call_args[0])
    
    def test_delete_relationships_by_type(self):
        """Test deleting all relationships of a specific type."""
        # Setup mock response
        self.mock_base_manager.execute_query.return_value = [{"deleted": True, "count": 5}]
        
        # Call method under test
        result, count = self.relation_manager.delete_relationships_by_type("TEST_RELATION")
        
        # Assertions
        self.assertTrue(result)
        self.assertEqual(count, 5)
        
        # Verify query execution
        self.mock_base_manager.execute_query.assert_called_once()
        # Verify that the query contains DELETE
        call_args = self.mock_base_manager.execute_query.call_args[0]
        self.assertTrue("DELETE" in call_args[0])
        self.assertTrue("TEST_RELATION" in call_args[0])
    
    def test_get_relationship_types(self):
        """Test retrieving all relationship types in the graph."""
        # Setup mock response
        mock_results = [
            {"relationship_type": "TEST_RELATION", "count": 10},
            {"relationship_type": "ANOTHER_RELATION", "count": 5}
        ]
        self.mock_base_manager.execute_query.return_value = mock_results
        
        # Call method under test
        results = self.relation_manager.get_relationship_types()
        
        # Assertions
        self.assertEqual(len(results), 2)
        self.assertEqual(results[0].get("relationship_type"), "TEST_RELATION")
        self.assertEqual(results[0].get("count"), 10)
        
        # Verify query execution
        self.mock_base_manager.execute_query.assert_called_once()
        # Verify that the query gets relationship types
        call_args = self.mock_base_manager.execute_query.call_args[0]
        self.assertTrue("type(r)" in call_args[0] or "TYPE(r)" in call_args[0])
    
    def test_find_path_between_entities(self):
        """Test finding a path between two entities."""
        # Setup mock response with a path
        mock_path = [
            {
                "nodes": [
                    {"name": "entity1", "id": 1},
                    {"name": "entity2", "id": 2},
                    {"name": "entity3", "id": 3}
                ],
                "relationships": [
                    {"type": "CONNECTS_TO", "properties": {}},
                    {"type": "CONNECTS_TO", "properties": {}}
                ]
            }
        ]
        self.mock_base_manager.execute_query.return_value = mock_path
        
        # Call method under test
        path = self.relation_manager.find_path_between_entities("entity1", "entity3", max_depth=3)
        
        # Assertions
        self.assertIsNotNone(path)
        self.assertEqual(len(path.get("nodes")), 3)
        self.assertEqual(len(path.get("relationships")), 2)
        self.assertEqual(path.get("nodes")[0].get("name"), "entity1")
        self.assertEqual(path.get("nodes")[2].get("name"), "entity3")
        
        # Verify query execution
        self.mock_base_manager.execute_query.assert_called_once()
        # Verify that the query uses path finding
        call_args = self.mock_base_manager.execute_query.call_args[0]
        self.assertTrue("shortestPath" in call_args[0] or "allShortestPaths" in call_args[0])

if __name__ == '__main__':
    unittest.main() 