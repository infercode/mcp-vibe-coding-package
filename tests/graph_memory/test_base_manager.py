import unittest
import os
from unittest.mock import patch, MagicMock

from src.graph_memory.base_manager import BaseManager
from src.logger import get_logger

class TestBaseManager(unittest.TestCase):
    """Test suite for the BaseManager class responsible for Neo4j connections."""
    
    def setUp(self):
        """Set up test environment before each test."""
        self.logger = get_logger("test_base_manager")
        # Create the base manager with a test logger
        self.base_manager = BaseManager(self.logger)
        
        # Mock environment variables for testing
        self.env_patcher = patch.dict('os.environ', {
            'NEO4J_URI': 'bolt://localhost:7687',
            'NEO4J_USERNAME': 'neo4j',
            'NEO4J_PASSWORD': 'password',
            'NEO4J_DATABASE': 'neo4j'
        })
        self.env_patcher.start()
        
    def tearDown(self):
        """Clean up after each test."""
        # Stop the environment variable patch
        self.env_patcher.stop()
        
        # Ensure the connection is closed
        if hasattr(self.base_manager, 'driver') and self.base_manager.driver:
            self.base_manager.close()
    
    @patch('neo4j.GraphDatabase.driver')
    def test_initialize_connection(self, mock_driver):
        """Test that the connection is initialized correctly."""
        # Setup mock driver
        mock_driver_instance = MagicMock()
        mock_driver.return_value = mock_driver_instance
        
        # Test initialization
        success = self.base_manager.initialize()
        
        # Assertions
        self.assertTrue(success)
        mock_driver.assert_called_once_with(
            'bolt://localhost:7687',
            auth=('neo4j', 'password')
        )
        self.assertEqual(self.base_manager.database, 'neo4j')
        
    @patch('neo4j.GraphDatabase.driver')
    def test_initialize_connection_error(self, mock_driver):
        """Test that connection errors are handled correctly."""
        # Setup mock driver to raise an exception
        mock_driver.side_effect = Exception("Connection failed")
        
        # Test initialization with error
        success = self.base_manager.initialize()
        
        # Assertions
        self.assertFalse(success)
        self.assertIsNone(self.base_manager.driver)
        
    @patch('neo4j.GraphDatabase.driver')
    def test_safe_execute_query(self, mock_driver):
        """Test that queries are executed safely."""
        # Setup mock session and transaction
        mock_driver_instance = MagicMock()
        mock_session = MagicMock()
        mock_transaction = MagicMock()
        mock_driver.return_value = mock_driver_instance
        mock_driver_instance.session.return_value = mock_session
        mock_session.__enter__.return_value = mock_session
        mock_session.begin_transaction.return_value = mock_transaction
        mock_transaction.__enter__.return_value = mock_transaction
        
        # Setup mock result
        mock_result = MagicMock()
        mock_result.data.return_value = [{"result": "value"}]
        mock_transaction.run.return_value = mock_result
        
        # Initialize the connection
        self.base_manager.initialize()
        
        # Execute a test query
        query = "MATCH (n) RETURN n LIMIT 1"
        params = {"param": "value"}
        result = self.base_manager.execute_query(query, params)
        
        # Assertions
        self.assertEqual(result, [{"result": "value"}])
        mock_session.begin_transaction.assert_called_once()
        mock_transaction.run.assert_called_once_with(query, params)
        
    @patch('neo4j.GraphDatabase.driver')
    def test_safe_execute_query_error(self, mock_driver):
        """Test that query execution errors are handled correctly."""
        # Setup mock driver and session
        mock_driver_instance = MagicMock()
        mock_session = MagicMock()
        mock_driver.return_value = mock_driver_instance
        mock_driver_instance.session.return_value = mock_session
        mock_session.__enter__.return_value = mock_session
        
        # Set up mock transaction to raise an exception
        mock_session.begin_transaction.side_effect = Exception("Query execution failed")
        
        # Initialize the connection
        self.base_manager.initialize()
        
        # Execute a test query that will fail
        query = "INVALID QUERY"
        result = self.base_manager.execute_query(query)
        
        # Assertions
        self.assertIsNone(result)
        
    @patch('neo4j.GraphDatabase.driver')
    def test_vector_index_setup(self, mock_driver):
        """Test that vector index is set up correctly."""
        # Setup mock session and transaction
        mock_driver_instance = MagicMock()
        mock_session = MagicMock()
        mock_transaction = MagicMock()
        mock_driver.return_value = mock_driver_instance
        mock_driver_instance.session.return_value = mock_session
        mock_session.__enter__.return_value = mock_session
        mock_session.begin_transaction.return_value = mock_transaction
        mock_transaction.__enter__.return_value = mock_transaction
        
        # Initialize the connection
        self.base_manager.initialize()
        
        # Setup the vector index
        result = self.base_manager.setup_vector_index()
        
        # Assertions
        self.assertTrue(result)
        # Verify that the appropriate index creation queries were executed
        calls = mock_transaction.run.call_args_list
        self.assertGreaterEqual(len(calls), 1)
        
    @patch('neo4j.GraphDatabase.driver')
    def test_close_connection(self, mock_driver):
        """Test that the connection is closed correctly."""
        # Setup mock driver
        mock_driver_instance = MagicMock()
        mock_driver.return_value = mock_driver_instance
        
        # Initialize and then close the connection
        self.base_manager.initialize()
        self.base_manager.close()
        
        # Assertions
        mock_driver_instance.close.assert_called_once()
        self.assertIsNone(self.base_manager.driver)
        
    @patch('neo4j.GraphDatabase.driver')
    def test_connection_retry_logic(self, mock_driver):
        """Test that connection retry logic works correctly."""
        # Setup mock driver to fail on first attempt but succeed on second
        mock_driver_instance = MagicMock()
        mock_driver.side_effect = [Exception("First attempt failed"), mock_driver_instance]
        
        # Set retry count to 2
        self.base_manager.max_retries = 2
        
        # Initialize the connection
        success = self.base_manager.initialize()
        
        # Assertions
        self.assertTrue(success)
        self.assertEqual(mock_driver.call_count, 2)

if __name__ == '__main__':
    unittest.main() 