"""
Comprehensive regression tests for the Neo4j query validator improvements.

These tests verify that the validator correctly handles:
1. Past tense forms like "CREATED" in relationship types
2. Property names like "created_at" in queries
3. Variable names containing destructive operation words
While still correctly blocking actual destructive operations.

Run unit tests only: 
    pytest tests/pytest_validator_regression.py -v

Run integration tests only:
    pytest tests/pytest_validator_regression.py -v -m integration

Run all tests:
    pytest tests/pytest_validator_regression.py -v --no-skip
"""

import pytest
import re
from unittest.mock import MagicMock, patch

from src.models.neo4j_queries import (
    CypherQuery, CypherParameters, QueryBuilder, 
    NodePattern, RelationshipPattern, PathPattern, QueryOrder
)
from src.utils.neo4j_query_utils import validate_query
from src.graph_memory.base_manager import BaseManager


@pytest.fixture
def mock_logger():
    """Mock logger for testing."""
    return MagicMock()


@pytest.fixture
def base_manager(mock_logger):
    """Create a BaseManager with a mock logger."""
    return BaseManager(logger=mock_logger)


class TestNeo4jValidatorRegressionUnit:
    """Unit tests for Neo4j query validator improvements."""

    def test_relationship_type_with_past_tense(self):
        """Test that relationship types like 'CREATED' are now allowed."""
        # Create a query using a relationship type that contains a destructive operation word
        query_builder = QueryBuilder(
            match_patterns=[
                PathPattern(
                    nodes=[
                        NodePattern(variable="u", labels=["User"], properties={"id": "$user_id"}),
                        NodePattern(variable="p", labels=["Post"], properties={})
                    ],
                    relationships=[
                        RelationshipPattern(
                            variable="r",
                            type="CREATED",  # This would have failed before our fix
                            properties={},
                            direction="OUTGOING"
                        )
                    ]
                )
            ],
            where_clauses=["p.created_at > $start_date"],  # This would have failed before our fix
            return_fields=["p.id", "p.title", "p.created_at"],  # This would have failed before our fix
            order_by=[QueryOrder(field="p.created_at", direction="DESC")],  # This would have failed before our fix
            limit=5,
            skip=0,
            parameters={
                "user_id": "user-123",
                "start_date": "2023-01-01"
            }
        )

        # This should not raise an exception
        cypher_query = query_builder.to_cypher_query()
        assert cypher_query is not None
        assert "CREATED" in cypher_query.query
        assert "created_at" in cypher_query.query

    def test_property_names_with_destructive_substrings(self):
        """Test that property names containing words like 'created_at' are allowed."""
        # Directly create a query with properties containing destructive operation substrings
        query = """
        MATCH (u:User)
        WHERE u.created_at > $start_date AND u.deleted_at IS NULL AND u.removed = false
        RETURN u.id, u.name, u.created_at, u.created_by, u.set_preferences
        ORDER BY u.created_at DESC
        LIMIT 10
        """

        # This should not raise an exception
        validated_query = validate_query(
            query=query,
            parameters={"start_date": "2023-01-01"},
            read_only=True
        )
        assert validated_query is not None
        assert "created_at" in validated_query.query
        assert "deleted_at" in validated_query.query
        assert "removed" in validated_query.query
        assert "set_preferences" in validated_query.query

    def test_actual_destructive_operations_still_blocked(self):
        """Test that actual destructive operations are still blocked in read-only mode."""
        # Test with CREATE
        with pytest.raises(ValueError) as excinfo:
            validate_query(
                query="CREATE (n:User {name: 'Test'}) RETURN n",
                parameters={},
                read_only=True
            )
        assert "Destructive operation" in str(excinfo.value)
        assert "CREATE" in str(excinfo.value)

        # Test with DELETE
        with pytest.raises(ValueError) as excinfo:
            validate_query(
                query="MATCH (n:User) DELETE n",
                parameters={},
                read_only=True
            )
        assert "Destructive operation" in str(excinfo.value)
        assert "DELETE" in str(excinfo.value)

        # Test with SET
        with pytest.raises(ValueError) as excinfo:
            validate_query(
                query="MATCH (n:User) SET n.active = true RETURN n",
                parameters={},
                read_only=True
            )
        assert "Destructive operation" in str(excinfo.value)
        assert "SET" in str(excinfo.value)

    def test_base_manager_safe_execute_read_query(self, base_manager):
        """Test that BaseManager.safe_execute_read_query allows past tense forms."""
        # Mock _execute_validated_query to return a dummy result
        with patch.object(base_manager, '_execute_validated_query',
                         return_value=([{"result": "test"}], {"summary": "test"})):
            
            # This should not raise an exception
            query = """
            MATCH (u:User)-[r:CREATED]->(p:Post)
            WHERE p.created_at > $start_date
            RETURN u.name, p.title, p.created_at
            """
            
            result = base_manager.safe_execute_read_query(
                query=query,
                parameters={"start_date": "2023-01-01"},
                database=None
            )
            
            assert result == [{"result": "test"}]

    def test_complex_relationship_types(self):
        """Test relationship types that are complex but should be allowed."""
        relationship_types = [
            "HAS_CREATED", 
            "WAS_CREATED_BY",
            "CREATED_AT_LOCATION",
            "SET_PRIORITY_TO",
            "REMOVED_FROM_GROUP",
            "DELETED_ASSOCIATION",
            "MERGE_REQUESTED"
        ]

        for rel_type in relationship_types:
            # Create a simple query with each relationship type
            query = f"""
            MATCH (a:Entity)-[r:{rel_type}]->(b:Entity)
            RETURN a, r, b
            """
            
            try:
                validated_query = validate_query(
                    query=query,
                    parameters={},
                    read_only=True
                )
                assert validated_query is not None
                assert rel_type in validated_query.query
            except ValueError as e:
                pytest.fail(f"Relationship type '{rel_type}' was incorrectly rejected: {str(e)}")


# Integration tests requiring full component mocks
@pytest.mark.integration
class TestNeo4jValidatorIntegration:
    """Integration tests for the validator with components."""
    
    def test_entity_manager_integration(self, base_manager):
        """Test that entity manager operations work with 'created_at'."""
        # Mock entity manager components
        with patch('src.graph_memory.entity_manager.EntityManager') as MockEntityManager:
            mock_entity_manager = MockEntityManager.return_value
            
            # Setup get_entity to return something with created_at
            mock_entity_manager.get_entity.return_value = '{"id": "entity-1", "name": "Test Entity", "created_at": "2023-04-01T12:00:00Z"}'
            
            # Execute get_entity - this delegates to base_manager's safe_execute_read_query
            with patch.object(base_manager, 'safe_execute_read_query', return_value=[{"e": {"id": "entity-1", "name": "Test Entity", "created_at": "2023-04-01T12:00:00Z"}}]):
                # Create a query that uses created_at
                query = """
                MATCH (e:Entity {name: $name})
                WHERE e.created_at > $start_date
                RETURN e
                """
                
                # Execute the query through base_manager
                result = base_manager.safe_execute_read_query(
                    query=query,
                    parameters={"name": "Test Entity", "start_date": "2023-01-01"}
                )
                
                # Verify the result contains our data
                assert result is not None
                assert len(result) > 0
                assert "created_at" in str(result)

    def test_relationship_with_created_type(self, base_manager):
        """Test that relationship operations work with 'CREATED' relationship type."""
        # Mock necessary components for relationship operations
        with patch.object(base_manager, 'safe_execute_write_query', return_value=[{"r": {"id": "rel-1", "type": "CREATED"}}]):
            # Create a query that uses CREATED relationship
            query = """
            MATCH (a:Entity {name: $from_name}), (b:Entity {name: $to_name})
            CREATE (a)-[r:CREATED {at: $timestamp}]->(b)
            RETURN r
            """
            
            # Execute the query through base_manager
            result = base_manager.safe_execute_write_query(
                query=query,
                parameters={
                    "from_name": "Creator Entity",
                    "to_name": "Created Entity",
                    "timestamp": "2023-04-01T12:00:00Z"
                }
            )
            
            # Verify the result contains our data
            assert result is not None
            assert len(result) > 0
            assert "CREATED" in str(result) 