"""
Tests for Neo4j query validation utilities and integration with BaseManager.
"""

import unittest
from unittest.mock import patch, MagicMock, call
import sys
import os

# Add the project root to the Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.graph_memory.base_manager import BaseManager
from src.models.neo4j_queries import CypherQuery, CypherParameters


class TestNeo4jQueryValidation(unittest.TestCase):
    """Tests for Neo4j query validation utilities."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.logger_mock = MagicMock()
        self.base_manager = BaseManager(logger=self.logger_mock)
        
        # Create a patch for the safe_execute_query method
        self.safe_execute_query_patch = patch.object(
            self.base_manager, 'safe_execute_query',
            return_value=([{"result": "test"}], {"summary": "test"})
        )
        self.mock_safe_execute_query = self.safe_execute_query_patch.start()
        
    def tearDown(self):
        """Tear down test fixtures."""
        self.safe_execute_query_patch.stop()
        
    def test_execute_read_query(self):
        """Test that execute_read_query calls safe_execute_query with the right parameters."""
        query = "MATCH (n) RETURN n"
        params = {"name": "test"}
        
        result = self.base_manager.execute_read_query(query, params)
        
        self.mock_safe_execute_query.assert_called_once_with(query, params, None)
        self.assertEqual(result, [{"result": "test"}])
        
    def test_execute_write_query(self):
        """Test that execute_write_query calls safe_execute_query with the right parameters."""
        query = "CREATE (n:Test) RETURN n"
        params = {"name": "test"}
        
        result = self.base_manager.execute_write_query(query, params)
        
        self.mock_safe_execute_query.assert_called_once_with(query, params, None)
        self.assertEqual(result, [{"result": "test"}])
        
    def test_safe_execute_read_query(self):
        """Test that safe_execute_read_query validates and executes the query properly."""
        query = "MATCH (n) RETURN n"
        params = {"name": "test"}
        
        # Patch the _execute_validated_query method directly
        with patch.object(self.base_manager, '_execute_validated_query',
                         return_value=([{"result": "test"}], {"summary": "test"})) as mock_execute:
            result = self.base_manager.safe_execute_read_query(query, params)
            
            # Verify the query was executed with validation
            mock_execute.assert_called_once()
            self.assertEqual(result, [{"result": "test"}])
            
    def test_safe_execute_write_query(self):
        """Test that safe_execute_write_query validates and executes the query properly."""
        query = "CREATE (n:Test) RETURN n"
        params = {"name": "test"}
        
        # Patch the _execute_validated_query method directly
        with patch.object(self.base_manager, '_execute_validated_query',
                         return_value=([{"result": "test"}], {"summary": "test"})) as mock_execute:
            result = self.base_manager.safe_execute_write_query(query, params)
            
            # Verify the query was executed with validation
            mock_execute.assert_called_once()
            self.assertEqual(result, [{"result": "test"}])
            
    def test_execute_validated_query(self):
        """Test that _execute_validated_query extracts parameters and calls safe_execute_query."""
        query = "MATCH (n) RETURN n"
        params = {"name": "test"}
        query_model = MagicMock(spec=CypherQuery)
        query_model.query = query
        query_model.parameters = MagicMock(spec=CypherParameters)
        query_model.parameters.parameters = params
        query_model.database = "testdb"
        
        result, summary = self.base_manager._execute_validated_query(query_model)
        
        self.mock_safe_execute_query.assert_called_once_with(query, params, "testdb")
        self.assertEqual(result, [{"result": "test"}])
        self.assertEqual(summary, {"summary": "test"})
        
    def test_read_query_with_destructive_operation(self):
        """Test that safe_execute_read_query raises an error for destructive operations."""
        query = "DELETE (n) RETURN n"
        params = {"name": "test"}
        
        # We expect a ValueError for a destructive operation in a read query
        with self.assertRaises(ValueError):
            self.base_manager.safe_execute_read_query(query, params)
            
        self.logger_mock.error.assert_called()


if __name__ == "__main__":
    unittest.main() 