"""
Tests for Neo4j query validation utilities and integration with BaseManager.
"""

import pytest
from unittest.mock import patch, MagicMock, call
import sys
import os

# Add the project root to the Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.graph_memory.base_manager import BaseManager
from src.models.neo4j_queries import CypherQuery, CypherParameters


class TestNeo4jQueryValidation:
    """Tests for Neo4j query validation utilities."""
    
    def setup_method(self):
        """Set up test fixtures before each test."""
        self.logger_mock = MagicMock()
        self.base_manager = BaseManager(logger=self.logger_mock)
        
        # Create a patch for the safe_execute_query method
        self.safe_execute_query_patcher = patch.object(
            self.base_manager, 'safe_execute_query',
            return_value=([{"result": "test"}], {"summary": "test"})
        )
        self.mock_safe_execute_query = self.safe_execute_query_patcher.start()
        
    def teardown_method(self):
        """Tear down test fixtures after each test."""
        self.safe_execute_query_patcher.stop()
        
    def test_execute_read_query(self):
        """Test that execute_read_query calls safe_execute_query with the right parameters."""
        query = "MATCH (n) RETURN n"
        params = {"name": "test"}
        
        result = self.base_manager.execute_read_query(query, params)
        
        self.mock_safe_execute_query.assert_called_once_with(query, params, None)
        assert result == [{"result": "test"}]
        
    def test_execute_write_query(self):
        """Test that execute_write_query calls safe_execute_query with the right parameters."""
        query = "CREATE (n:Test) RETURN n"
        params = {"name": "test"}
        
        result = self.base_manager.execute_write_query(query, params)
        
        self.mock_safe_execute_query.assert_called_once_with(query, params, None)
        assert result == [{"result": "test"}]
        
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
            assert result == [{"result": "test"}]
            
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
            assert result == [{"result": "test"}]
            
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
        assert result == [{"result": "test"}]
        assert summary == {"summary": "test"}
        
    def test_read_query_with_destructive_operation(self):
        """Test that safe_execute_read_query raises an error for destructive operations."""
        query = "DELETE (n) RETURN n"
        params = {"name": "test"}
        
        # We expect a ValueError for a destructive operation in a read query
        with pytest.raises(ValueError):
            self.base_manager.safe_execute_read_query(query, params)
            
        # Verify that an error was logged
        assert self.logger_mock.error.called
    
    def test_read_query_with_past_tense_destructive_words(self):
        """Test that the validator allows past tense forms like CREATED and property names like created_at."""
        # Test with relationship type containing 'CREATED'
        query1 = "MATCH (u:User)-[r:CREATED]->(p:Post) RETURN u, p"
        params1 = {}
        
        # Mock the _execute_validated_query method
        with patch.object(self.base_manager, '_execute_validated_query',
                         return_value=([{"result": "test"}], {"summary": "test"})) as mock_execute:
            result1 = self.base_manager.safe_execute_read_query(query1, params1)
            
            # Verify the query was executed with validation
            mock_execute.assert_called_once()
            assert result1 == [{"result": "test"}]
        
        # Test with property name containing 'created_at'
        query2 = "MATCH (u:User) WHERE u.created_at > $start_date RETURN u"
        params2 = {"start_date": "2023-01-01"}
        
        # Mock the _execute_validated_query method again
        with patch.object(self.base_manager, '_execute_validated_query',
                         return_value=([{"result": "test"}], {"summary": "test"})) as mock_execute:
            result2 = self.base_manager.safe_execute_read_query(query2, params2)
            
            # Verify the query was executed with validation
            mock_execute.assert_called_once()
            assert result2 == [{"result": "test"}]
            
        # Test with mixed content that should be valid
        query3 = """
        MATCH (creator:User)-[r:CREATED]->(p:Post)
        WHERE p.created_at > $start_date
        AND p.deleted_at IS NULL
        RETURN creator.name, p.title, p.created_at
        """
        params3 = {"start_date": "2023-01-01"}
        
        # Mock the _execute_validated_query method once more
        with patch.object(self.base_manager, '_execute_validated_query',
                         return_value=([{"result": "test"}], {"summary": "test"})) as mock_execute:
            result3 = self.base_manager.safe_execute_read_query(query3, params3, database=None)
            
            # Verify the query was executed with validation
            mock_execute.assert_called_once()
            assert result3 == [{"result": "test"}] 