import unittest
from unittest.mock import patch, MagicMock

from src.graph_memory.entity_manager import EntityManager
from src.graph_memory.base_manager import BaseManager

class TestEntityManager(unittest.TestCase):
    """Test suite for EntityManager class responsible for CRUD operations on entities."""
    
    def setUp(self):
        """Set up test environment before each test."""
        # Create a mock base manager
        self.mock_base_manager = MagicMock(spec=BaseManager)
        
        # Create entity manager with mock base manager
        self.entity_manager = EntityManager(self.mock_base_manager)
        
        # Sample entities for testing
        self.sample_entity = {
            "name": "test_entity",
            "entityType": "TestType",
            "description": "A test entity",
            "properties": {
                "key1": "value1",
                "key2": "value2"
            },
            "tags": ["test", "example"]
        }
        
        self.sample_entities = [
            self.sample_entity,
            {
                "name": "test_entity_2",
                "entityType": "TestType",
                "description": "Another test entity"
            }
        ]
    
    def test_create_entity(self):
        """Test creation of a single entity."""
        # Setup mock response from execute_query
        self.mock_base_manager.execute_query.return_value = [{"e": {"id": 123, "properties": self.sample_entity}}]
        
        # Call method under test
        result = self.entity_manager.create_entity(self.sample_entity)
        
        # Assertions
        self.assertIsNotNone(result)
        self.assertEqual(result.get("id"), 123)
        
        # Verify query execution
        self.mock_base_manager.execute_query.assert_called_once()
        # Verify that the query contains MERGE or CREATE
        call_args = self.mock_base_manager.execute_query.call_args[0]
        self.assertTrue("CREATE" in call_args[0] or "MERGE" in call_args[0])
        
    def test_create_entities_batch(self):
        """Test batch creation of multiple entities."""
        # Setup mock response
        mock_responses = [
            {"e": {"id": 123, "properties": self.sample_entities[0]}},
            {"e": {"id": 124, "properties": self.sample_entities[1]}}
        ]
        self.mock_base_manager.execute_query.return_value = mock_responses
        
        # Call method under test
        results = self.entity_manager.create_entities(self.sample_entities)
        
        # Assertions
        self.assertEqual(len(results), 2)
        self.assertEqual(results[0].get("id"), 123)
        self.assertEqual(results[1].get("id"), 124)
        
        # Verify query execution
        self.mock_base_manager.execute_query.assert_called_once()
        
    def test_get_entity_by_name(self):
        """Test retrieving an entity by name."""
        # Setup mock response
        self.mock_base_manager.execute_query.return_value = [{"e": {"id": 123, "properties": self.sample_entity}}]
        
        # Call method under test
        result = self.entity_manager.get_entity("test_entity")
        
        # Assertions
        self.assertIsNotNone(result)
        self.assertEqual(result.get("name"), "test_entity")
        
        # Verify query execution
        self.mock_base_manager.execute_query.assert_called_once()
        # Verify that the query contains MATCH
        call_args = self.mock_base_manager.execute_query.call_args[0]
        self.assertTrue("MATCH" in call_args[0])
        
    def test_get_entity_by_name_not_found(self):
        """Test retrieving a non-existent entity."""
        # Setup mock response (empty result)
        self.mock_base_manager.execute_query.return_value = []
        
        # Call method under test
        result = self.entity_manager.get_entity("non_existent_entity")
        
        # Assertions
        self.assertIsNone(result)
        
        # Verify query execution
        self.mock_base_manager.execute_query.assert_called_once()
        
    def test_update_entity(self):
        """Test updating an entity's properties."""
        # Setup mock response
        updated_entity = dict(self.sample_entity)
        updated_entity["description"] = "Updated description"
        self.mock_base_manager.execute_query.return_value = [{"e": {"id": 123, "properties": updated_entity}}]
        
        # Call method under test
        update_data = {"description": "Updated description"}
        result = self.entity_manager.update_entity("test_entity", update_data)
        
        # Assertions
        self.assertIsNotNone(result)
        self.assertEqual(result.get("description"), "Updated description")
        
        # Verify query execution
        self.mock_base_manager.execute_query.assert_called_once()
        # Verify that the query contains SET for updates
        call_args = self.mock_base_manager.execute_query.call_args[0]
        self.assertTrue("SET" in call_args[0])
        
    def test_delete_entity(self):
        """Test deleting an entity."""
        # Setup mock response
        self.mock_base_manager.execute_query.return_value = [{"success": True, "count": 1}]
        
        # Call method under test
        result = self.entity_manager.delete_entity("test_entity")
        
        # Assertions
        self.assertTrue(result)
        
        # Verify query execution
        self.mock_base_manager.execute_query.assert_called_once()
        # Verify that the query contains DELETE
        call_args = self.mock_base_manager.execute_query.call_args[0]
        self.assertTrue("DELETE" in call_args[0])
        
    def test_get_entities_by_type(self):
        """Test retrieving entities by type."""
        # Setup mock response
        self.mock_base_manager.execute_query.return_value = [
            {"e": {"id": 123, "properties": self.sample_entities[0]}},
            {"e": {"id": 124, "properties": self.sample_entities[1]}}
        ]
        
        # Call method under test
        results = self.entity_manager.get_entities_by_type("TestType")
        
        # Assertions
        self.assertEqual(len(results), 2)
        self.assertEqual(results[0].get("entityType"), "TestType")
        self.assertEqual(results[1].get("entityType"), "TestType")
        
        # Verify query execution
        self.mock_base_manager.execute_query.assert_called_once()
        
    def test_add_tags_to_entity(self):
        """Test adding tags to an entity."""
        # Setup mock response
        updated_entity = dict(self.sample_entity)
        updated_entity["tags"] = ["test", "example", "new_tag"]
        self.mock_base_manager.execute_query.return_value = [{"e": {"id": 123, "properties": updated_entity}}]
        
        # Call method under test
        result = self.entity_manager.add_tags_to_entity("test_entity", ["new_tag"])
        
        # Assertions
        self.assertIsNotNone(result)
        self.assertIn("new_tag", result.get("tags"))
        
        # Verify query execution
        self.mock_base_manager.execute_query.assert_called_once()
        
    def test_search_entities(self):
        """Test searching for entities with filters."""
        # Setup mock response
        self.mock_base_manager.execute_query.return_value = [
            {"e": {"id": 123, "properties": self.sample_entity}}
        ]
        
        # Call method under test
        results = self.entity_manager.search_entities(entity_type="TestType", search_term="test")
        
        # Assertions
        self.assertEqual(len(results), 1)
        self.assertEqual(results[0].get("name"), "test_entity")
        
        # Verify query execution
        self.mock_base_manager.execute_query.assert_called_once()
        # Verify that the query contains appropriate WHERE clauses
        call_args = self.mock_base_manager.execute_query.call_args[0]
        self.assertTrue("WHERE" in call_args[0])
        
if __name__ == '__main__':
    unittest.main() 