import unittest
from unittest.mock import patch, MagicMock

from src.graph_memory.observation_manager import ObservationManager
from src.graph_memory.base_manager import BaseManager

class TestObservationManager(unittest.TestCase):
    """Test suite for ObservationManager class responsible for entity observations."""
    
    def setUp(self):
        """Set up test environment before each test."""
        # Create a mock base manager
        self.mock_base_manager = MagicMock(spec=BaseManager)
        
        # Create observation manager with mock base manager
        self.observation_manager = ObservationManager(self.mock_base_manager)
        
        # Sample observation data for testing
        self.sample_observation = {
            "entity_name": "test_entity",
            "observation_type": "DESCRIPTION",
            "content": "This is a test observation description.",
            "metadata": {
                "confidence": 0.9,
                "source": "unit_test"
            }
        }
    
    def test_add_observation(self):
        """Test adding an observation to an entity."""
        # Setup mock response
        mock_result = {
            "entity": {"name": "test_entity", "id": 123},
            "observation": {
                "id": 456,
                "type": "DESCRIPTION",
                "content": "This is a test observation description.",
                "metadata": {"confidence": 0.9, "source": "unit_test"},
                "timestamp": "2025-03-28T12:00:00"
            }
        }
        self.mock_base_manager.execute_query.return_value = [mock_result]
        
        # Call method under test
        result = self.observation_manager.add_observation(
            entity_name="test_entity",
            observation_type="DESCRIPTION",
            content="This is a test observation description.",
            metadata={"confidence": 0.9, "source": "unit_test"}
        )
        
        # Assertions
        self.assertIsNotNone(result)
        self.assertEqual(result.get("entity_id"), 123)
        self.assertEqual(result.get("observation_id"), 456)
        self.assertEqual(result.get("type"), "DESCRIPTION")
        self.assertEqual(result.get("content"), "This is a test observation description.")
        
        # Verify query execution
        self.mock_base_manager.execute_query.assert_called_once()
        # Verify that the query contains CREATE
        call_args = self.mock_base_manager.execute_query.call_args[0]
        self.assertTrue("CREATE" in call_args[0])
    
    def test_add_observation_entity_not_found(self):
        """Test handling of observation addition when entity doesn't exist."""
        # Setup mock response (empty result indicating entity not found)
        self.mock_base_manager.execute_query.return_value = []
        
        # Call method under test
        result = self.observation_manager.add_observation(
            entity_name="nonexistent_entity",
            observation_type="DESCRIPTION",
            content="This observation won't be added"
        )
        
        # Assertions
        self.assertIsNone(result)
        
        # Verify query execution
        self.mock_base_manager.execute_query.assert_called_once()
    
    def test_get_entity_observations(self):
        """Test retrieving all observations for an entity."""
        # Setup mock response
        mock_results = [
            {
                "entity": {"name": "test_entity", "id": 123},
                "observation": {
                    "id": 456,
                    "type": "DESCRIPTION",
                    "content": "This is a test observation description.",
                    "metadata": {"confidence": 0.9, "source": "unit_test"},
                    "timestamp": "2025-03-28T12:00:00"
                }
            },
            {
                "entity": {"name": "test_entity", "id": 123},
                "observation": {
                    "id": 789,
                    "type": "FEATURE",
                    "content": "This is a feature observation.",
                    "metadata": {"confidence": 0.8, "source": "unit_test"},
                    "timestamp": "2025-03-28T12:30:00"
                }
            }
        ]
        self.mock_base_manager.execute_query.return_value = mock_results
        
        # Call method under test
        results = self.observation_manager.get_entity_observations(entity_name="test_entity")
        
        # Assertions
        self.assertEqual(len(results), 2)
        self.assertEqual(results[0].get("entity_name"), "test_entity")
        self.assertEqual(results[0].get("type"), "DESCRIPTION")
        self.assertEqual(results[1].get("type"), "FEATURE")
        
        # Verify query execution
        self.mock_base_manager.execute_query.assert_called_once()
        # Verify that the query contains MATCH
        call_args = self.mock_base_manager.execute_query.call_args[0]
        self.assertTrue("MATCH" in call_args[0])
    
    def test_get_observation_by_type(self):
        """Test retrieving entity observations by type."""
        # Setup mock response
        mock_results = [
            {
                "entity": {"name": "test_entity", "id": 123},
                "observation": {
                    "id": 456,
                    "type": "DESCRIPTION",
                    "content": "This is a test observation description.",
                    "metadata": {"confidence": 0.9, "source": "unit_test"},
                    "timestamp": "2025-03-28T12:00:00"
                }
            }
        ]
        self.mock_base_manager.execute_query.return_value = mock_results
        
        # Call method under test
        results = self.observation_manager.get_observations_by_type(
            entity_name="test_entity", 
            observation_type="DESCRIPTION"
        )
        
        # Assertions
        self.assertEqual(len(results), 1)
        self.assertEqual(results[0].get("type"), "DESCRIPTION")
        
        # Verify query execution
        self.mock_base_manager.execute_query.assert_called_once()
        # Verify that the query contains the observation type filter
        call_args = self.mock_base_manager.execute_query.call_args[0]
        self.assertTrue("DESCRIPTION" in call_args[0])
    
    def test_update_observation(self):
        """Test updating an observation."""
        # Setup mock response
        updated_observation = {
            "entity": {"name": "test_entity", "id": 123},
            "observation": {
                "id": 456,
                "type": "DESCRIPTION",
                "content": "This is an updated description.",
                "metadata": {"confidence": 0.95, "source": "unit_test", "updated": True},
                "timestamp": "2025-03-28T13:00:00"
            }
        }
        self.mock_base_manager.execute_query.return_value = [updated_observation]
        
        # Call method under test
        update_data = {
            "content": "This is an updated description.",
            "metadata": {"confidence": 0.95, "source": "unit_test", "updated": True}
        }
        result = self.observation_manager.update_observation(
            entity_name="test_entity",
            observation_id=456,
            update_data=update_data
        )
        
        # Assertions
        self.assertIsNotNone(result)
        self.assertEqual(result.get("content"), "This is an updated description.")
        self.assertEqual(result.get("metadata").get("confidence"), 0.95)
        self.assertTrue(result.get("metadata").get("updated"))
        
        # Verify query execution
        self.mock_base_manager.execute_query.assert_called_once()
        # Verify that the query contains SET for updates
        call_args = self.mock_base_manager.execute_query.call_args[0]
        self.assertTrue("SET" in call_args[0])
    
    def test_delete_observation(self):
        """Test deleting an observation."""
        # Setup mock response
        self.mock_base_manager.execute_query.return_value = [{"deleted": True, "count": 1}]
        
        # Call method under test
        result = self.observation_manager.delete_observation(
            entity_name="test_entity",
            observation_id=456
        )
        
        # Assertions
        self.assertTrue(result)
        
        # Verify query execution
        self.mock_base_manager.execute_query.assert_called_once()
        # Verify that the query contains DELETE
        call_args = self.mock_base_manager.execute_query.call_args[0]
        self.assertTrue("DELETE" in call_args[0])
    
    def test_delete_entity_observations(self):
        """Test deleting all observations for an entity."""
        # Setup mock response
        self.mock_base_manager.execute_query.return_value = [{"deleted": True, "count": 5}]
        
        # Call method under test
        result, count = self.observation_manager.delete_entity_observations("test_entity")
        
        # Assertions
        self.assertTrue(result)
        self.assertEqual(count, 5)
        
        # Verify query execution
        self.mock_base_manager.execute_query.assert_called_once()
        # Verify that the query contains DELETE
        call_args = self.mock_base_manager.execute_query.call_args[0]
        self.assertTrue("DELETE" in call_args[0])
    
    def test_get_observation_types(self):
        """Test retrieving all observation types used in the graph."""
        # Setup mock response
        mock_results = [
            {"observation_type": "DESCRIPTION", "count": 10},
            {"observation_type": "FEATURE", "count": 5},
            {"observation_type": "BEHAVIOR", "count": 3}
        ]
        self.mock_base_manager.execute_query.return_value = mock_results
        
        # Call method under test
        results = self.observation_manager.get_observation_types()
        
        # Assertions
        self.assertEqual(len(results), 3)
        self.assertEqual(results[0].get("observation_type"), "DESCRIPTION")
        self.assertEqual(results[0].get("count"), 10)
        
        # Verify query execution
        self.mock_base_manager.execute_query.assert_called_once()
        # Verify that the query is counting observation types
        call_args = self.mock_base_manager.execute_query.call_args[0]
        self.assertTrue("observation_type" in call_args[0].lower() or "observationType" in call_args[0])
    
    def test_get_latest_observation(self):
        """Test retrieving the latest observation of a specific type for an entity."""
        # Setup mock response
        mock_result = {
            "entity": {"name": "test_entity", "id": 123},
            "observation": {
                "id": 789,
                "type": "DESCRIPTION",
                "content": "This is the latest description.",
                "metadata": {"confidence": 0.95},
                "timestamp": "2025-03-28T14:00:00"
            }
        }
        self.mock_base_manager.execute_query.return_value = [mock_result]
        
        # Call method under test
        result = self.observation_manager.get_latest_observation(
            entity_name="test_entity", 
            observation_type="DESCRIPTION"
        )
        
        # Assertions
        self.assertIsNotNone(result)
        self.assertEqual(result.get("content"), "This is the latest description.")
        self.assertEqual(result.get("timestamp"), "2025-03-28T14:00:00")
        
        # Verify query execution
        self.mock_base_manager.execute_query.assert_called_once()
        # Verify that the query orders by timestamp
        call_args = self.mock_base_manager.execute_query.call_args[0]
        self.assertTrue("ORDER BY" in call_args[0] and "timestamp" in call_args[0].lower())

if __name__ == '__main__':
    unittest.main() 