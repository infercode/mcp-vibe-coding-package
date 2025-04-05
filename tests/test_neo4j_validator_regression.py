"""
Regression tests for the Neo4j query validator improvements.

These tests specifically verify that the validator correctly handles:
1. Past tense forms like "CREATED" in relationship types
2. Property names like "created_at" in queries
3. Variable names containing destructive operation words
While still correctly blocking actual destructive operations.
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


class TestNeo4jValidatorRegression:
    """Regression tests for Neo4j query validator improvements."""

    def setup_method(self):
        """Set up test fixtures."""
        self.logger = MagicMock()
        self.base_manager = BaseManager(logger=self.logger)

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

    def test_variable_names_with_destructive_substrings(self):
        """Test that variable names containing destructive operation words are allowed."""
        query = """
        MATCH (creator:User)-[created:AUTHORED]->(p:Post)
        WHERE creator.active = true
        WITH creator, collect(p) as created_posts
        MATCH (creator)-[:FOLLOWS]->(followed:User)
        RETURN creator.name, size(created_posts) as post_count, collect(followed.name) as following
        """

        # This should not raise an exception
        validated_query = validate_query(
            query=query,
            parameters={},
            read_only=True
        )
        assert validated_query is not None
        assert "creator" in validated_query.query
        assert "created:" in validated_query.query
        assert "created_posts" in validated_query.query

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

    def test_base_manager_safe_execute_read_query(self):
        """Test that BaseManager.safe_execute_read_query allows past tense forms."""
        # Mock _execute_validated_query to return a dummy result
        with patch.object(self.base_manager, '_execute_validated_query',
                         return_value=([{"result": "test"}], {"summary": "test"})):
            
            # This should not raise an exception
            query = """
            MATCH (u:User)-[r:CREATED]->(p:Post)
            WHERE p.created_at > $start_date
            RETURN u.name, p.title, p.created_at
            """
            
            result = self.base_manager.safe_execute_read_query(
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

    def test_mixed_legitimate_operations_and_properties(self):
        """Test queries with both legitimate operations and similar property names."""
        # This query should fail because it has an actual CREATE operation
        with pytest.raises(ValueError):
            query = """
            MATCH (u:User {created_at: $date})
            CREATE (p:Post {title: $title, content: $content})
            CREATE (u)-[r:CREATED]->(p)
            RETURN p
            """
            validate_query(query=query, parameters={"date": "2023-01-01", "title": "Test", "content": "Content"}, read_only=True)
            
        # But this similar read-only query should pass
        query = """
        MATCH (u:User {created_at: $date})
        MATCH (p:Post {title: $title, content: $content})
        MATCH (u)-[r:CREATED]->(p)
        RETURN p
        """
        validated_query = validate_query(
            query=query, 
            parameters={"date": "2023-01-01", "title": "Test", "content": "Content"}, 
            read_only=True
        )
        assert validated_query is not None

    def test_regex_edge_cases(self):
        """Test edge cases for the regex word boundary detection."""
        # Words that contain but don't exactly match destructive operations
        safe_words = [
            "CREATIVE", 
            "PROCREATE",
            "RECREATION",
            "UNDELETED",
            "PREDELETE",
            "DELETED_BY_USER",
            "UNSET",
            "PRESET",
            "SETTINGS",
            "EMERGED",
            "SUBMERGED"
        ]
        
        for word in safe_words:
            query = f"""
            MATCH (n:Node)
            WHERE n.property = '{word}' OR n.name = '{word.lower()}'
            RETURN n
            """
            
            try:
                validated_query = validate_query(query=query, parameters={}, read_only=True)
                assert validated_query is not None
                assert word in validated_query.query or word.lower() in validated_query.query
            except ValueError as e:
                pytest.fail(f"Safe word '{word}' was incorrectly rejected: {str(e)}")

    def test_cypher_query_model_direct(self):
        """Test the CypherQuery model's validate_query method directly."""
        # This should not raise an exception
        query = CypherQuery(
            query="MATCH (u:User)-[r:CREATED]->(p:Post) WHERE p.created_at > $date RETURN u, p",
            parameters=CypherParameters(parameters={"date": "2023-01-01"}),
            read_only=True,
            database=None
        )
        
        assert query is not None
        assert "CREATED" in query.query
        assert "created_at" in query.query

    def test_boundary_conditions(self):
        """Test boundary conditions where the destructive word is at the boundary."""
        queries = [
            # MERGE at the end of a relationship type
            "MATCH (a)-[r:CONNECTION_MERGE]->(b) RETURN a, b",
            
            # CREATE at the start of a property name
            "MATCH (n) WHERE n.create_date > $date RETURN n",
            
            # SET in the middle of a field name
            "RETURN user.preset_config AS config"
        ]
        
        for query_text in queries:
            try:
                validated_query = validate_query(query=query_text, parameters={}, read_only=True)
                assert validated_query is not None
            except ValueError as e:
                pytest.fail(f"Query '{query_text}' was incorrectly rejected: {str(e)}") 