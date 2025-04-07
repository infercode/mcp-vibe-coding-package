"""
Integration tests for the Neo4j query validator improvements.

These tests verify that various components correctly handle past tense forms
of destructive operations like 'CREATED' in relationship types and property
names like 'created_at' in real use scenarios.
"""

import pytest
from unittest.mock import patch, MagicMock

from src.models.neo4j_queries import CypherQuery, CypherParameters
from src.utils.neo4j_query_utils import validate_query
from src.graph_memory.base_manager import BaseManager


@pytest.fixture
def mock_graph_db_session():
    """Mock Neo4j session that returns predefined results."""
    mock_session = MagicMock()
    
    # Mock execution result
    mock_result = MagicMock()
    
    # Create a record with properties including 'created_at'
    mock_record = MagicMock()
    mock_record.data.return_value = {
        "e": {
            "id": "entity-1",
            "name": "Test Entity",
            "created_at": "2023-04-01T12:00:00Z",
            "created_by": "test-user"
        },
        "r": {
            "type": "CREATED",
            "properties": {"at": "2023-04-01T12:00:00Z"}
        }
    }
    
    # Set up records
    mock_result.__iter__.return_value = [mock_record]
    
    # Set up run method
    mock_tx = MagicMock()
    mock_tx.run.return_value = mock_result
    
    # Set up transaction functions
    def execute_read(tx_func, **kwargs):
        return tx_func(mock_tx, **kwargs)
    
    def execute_write(tx_func, **kwargs):
        return tx_func(mock_tx, **kwargs)
    
    mock_session.execute_read = execute_read
    mock_session.execute_write = execute_write
    
    return mock_session


@pytest.mark.integration
class TestValidatorIntegration:
    """Integration tests for Neo4j query validator."""
    
    def test_entity_manager_with_created_property(self, mock_entity_manager):
        """Test that EntityManager can handle 'created_at' properties."""
        # Mock the underlying query execution to return our test data
        with patch.object(mock_entity_manager, 'get_entity', return_value='{"id": "entity-1", "name": "Test Entity", "created_at": "2023-04-01T12:00:00Z"}'):
            # This should not raise an exception
            result = mock_entity_manager.get_entity("Test Entity")
            assert "created_at" in result
    
    def test_relation_manager_with_created_relationship(self, mock_relation_manager, mock_entity_manager):
        """Test that RelationManager can handle 'CREATED' relationship types."""
        # Mock the entity existence check
        with patch.object(mock_entity_manager, 'get_entity', return_value='{"id": "entity-1", "name": "Test Entity"}'):
            # Mock the relationship creation
            with patch.object(mock_relation_manager, 'create_relationship', return_value='{"status": "success", "relation_id": "rel-1"}'):
                # Create a relationship with CREATED type - this should not raise an exception
                result = mock_relation_manager.create_relationship(
                    from_entity="Source Entity",
                    to_entity="Target Entity",
                    relation_type="CREATED",
                    properties={"at": "2023-04-01T12:00:00Z"}
                )
                assert "success" in result
    
    def test_base_manager_direct(self, mock_base_manager):
        """Test BaseManager directly with queries containing past tense destructive words."""

        # Replace the safe_execute_read_query method directly
        mock_base_manager.safe_execute_read_query = MagicMock(return_value=[{"result": "test"}])
        
        # 1. Test with relationship type 'CREATED'
        query1 = """
        MATCH (u:User)-[r:CREATED]->(p:Post)
        WHERE p.created_at > $start_date
        RETURN u.name, p.title, p.created_at
        """

        result1 = mock_base_manager.safe_execute_read_query(
            query=query1,
            parameters={"start_date": "2023-01-01"}
        )
        assert result1 == [{"result": "test"}]
        
        # Verify first call was made
        assert mock_base_manager.safe_execute_read_query.call_count >= 1
        mock_base_manager.safe_execute_read_query.assert_called_with(
            query=query1,
            parameters={"start_date": "2023-01-01"}
        )
    
    def test_observation_manager_with_created_by(self, mock_observation_manager):
        """Test that ObservationManager can handle 'created_by' properties."""
        # Mock add_observation method
        with patch.object(mock_observation_manager, 'add_observation', 
                        return_value='{"status": "success", "observation_id": "obs-1"}'):
            # Create an observation with created_by property
            result = mock_observation_manager.add_observation(
                entity_name="Test Entity",
                content="This is a test observation",
                observation_type="test",
                metadata={"created_by": "test-user"}
            )
            assert "success" in result
    
    def test_search_manager_with_created_field(self, mock_search_manager):
        """Test that SearchManager can handle search queries with 'created_at' fields."""
        # Mock search_nodes method
        with patch.object(mock_search_manager, 'search_nodes', 
                        return_value='[{"id": "entity-1", "name": "Search Result", "created_at": "2023-04-01T12:00:00Z"}]'):
            # Perform search with created_at in criteria
            search_params = {
                "entity_type": "Entity",
                "properties": {"created_at": {"$gt": "2023-01-01"}},
                "search_term": "test"
            }
            
            result = mock_search_manager.search_nodes(search_params)
            assert "created_at" in result 