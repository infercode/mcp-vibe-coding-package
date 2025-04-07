#!/usr/bin/env python3
"""
Test script to verify the fixes to the Neo4j query validation system,
particularly for relationship types like 'CREATED' and property names
like 'created_at'.
"""

import os
import sys
import logging
from typing import Dict, Any, List, Optional
import pytest

# Add the project root to the Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from src.models.neo4j_queries import (
    QueryBuilder, NodePattern, RelationshipPattern, PathPattern, 
    QueryOrder, CypherQuery, CypherParameters
)
from src.utils.neo4j_query_utils import validate_query
from src.logger import get_logger

# Initialize logger
logger = get_logger()
# Use Python's built-in logging method instead of trying to access the logger's setLevel
logging.getLogger().setLevel(logging.INFO)

def test_relationship_type_with_destructive_word():
    """Test that relationship types containing words like 'CREATED' are now allowed."""
    logger.info("Testing relationship type with 'CREATED'...")
    
    # Create node patterns
    user = NodePattern(variable="u", labels=["User"], properties={"id": "$user_id"})
    post = NodePattern(variable="p", labels=["Post"], properties={})
    
    # Create relationship pattern with CREATED type - should now be allowed
    created_rel = RelationshipPattern(
        variable="r",
        type="CREATED",  # Using the word that was previously triggering validation errors
        properties={},
        direction="OUTGOING"
    )
    
    # Create a path pattern
    path = PathPattern(
        nodes=[user, post],
        relationships=[created_rel]
    )
    
    # Build the query
    try:
        builder = QueryBuilder(
            match_patterns=[path],
            where_clauses=["p.created_at > $start_date"],  # Using created_at property
            return_fields=["p.id", "p.title", "p.created_at"],  # Using created_at in return fields
            order_by=[QueryOrder(field="p.created_at", direction="DESC")],
            limit=5,
            skip=0,  # Adding skip parameter
            parameters={
                "user_id": "user-123",
                "start_date": "2023-01-01"
            }
        )
        
        # Convert to CypherQuery - this would have failed before
        query = builder.to_cypher_query()
        logger.info("SUCCESS: Query with 'CREATED' relationship type was validated correctly")
        logger.info(f"Generated Query: {query.query}")
        return True
    except ValueError as e:
        logger.error(f"FAILED: Query validation still rejects 'CREATED': {str(e)}")
        return False
    except Exception as e:
        logger.error(f"ERROR: Unexpected exception: {str(e)}")
        return False

def test_property_with_destructive_word(mock_logger):
    """Test that property names containing words like 'created_at' are now allowed."""
    logger.info("Testing property name with 'created_at'...")
    
    # Create a direct Cypher query with created_at in properties
    query = """
    MATCH (u:User)-[r:AUTHORED]->(p:Post)
    WHERE p.created_at > $start_date
    RETURN u.name, p.title, p.created_at, p.created_by
    ORDER BY p.created_at DESC
    LIMIT 10
    """
    
    # Validate the query - this should pass now
    validated_query = validate_query(
        query=query,
        parameters={"start_date": "2023-01-01"},
        read_only=True
    )
    assert validated_query is not None
    assert "created_at" in validated_query.query

def test_actual_destructive_operations(mock_logger):
    """Test that actual destructive operations are still caught properly."""
    logger.info("Testing actual destructive operations...")
    
    # Test CREATE operation
    create_query = """
    CREATE (u:User {name: 'Test User'})
    RETURN u
    """
    
    with pytest.raises(ValueError) as exc_info:
        validate_query(query=create_query, read_only=True)
    assert "CREATE" in str(exc_info.value)
    
    # Test DELETE operation
    delete_query = """
    MATCH (u:User {name: 'Test User'})
    DELETE u
    """
    
    with pytest.raises(ValueError) as exc_info:
        validate_query(query=delete_query, read_only=True)
    assert "DELETE" in str(exc_info.value)
    
    # Test SET operation
    set_query = """
    MATCH (u:User {name: 'Test User'})
    SET u.active = true
    RETURN u
    """
    
    with pytest.raises(ValueError) as exc_info:
        validate_query(query=set_query, read_only=True)
    assert "SET" in str(exc_info.value)

def test_relationship_types_with_destructive_words(mock_logger):
    """Test that relationship types containing words like 'CREATED' are allowed."""
    # Test relationship type containing 'CREATED'
    query = """
    MATCH (u:User)-[r:CREATED]->(p:Post)
    RETURN u, p
    """
    
    validated_query = validate_query(
        query=query,
        parameters={},
        read_only=True
    )
    assert validated_query is not None
    assert "CREATED" in validated_query.query
    
    # Test relationship type with past tense
    query = """
    MATCH (u:User)-[r:WAS_CREATED_BY]->(p:Post)
    RETURN u, p
    """
    
    validated_query = validate_query(
        query=query,
        parameters={},
        read_only=True
    )
    assert validated_query is not None
    assert "WAS_CREATED_BY" in validated_query.query

def test_property_names_with_destructive_words(mock_logger):
    """Test that property names containing destructive words are allowed."""
    # Test property names with various destructive words
    query = """
    MATCH (n:Node)
    WHERE n.created_at > $start_date
    AND n.deleted_at IS NULL
    AND n.set_by = 'admin'
    AND n.removed_flag = false
    AND n.merged_at < $end_date
    RETURN n
    """
    
    validated_query = validate_query(
        query=query,
        parameters={
            "start_date": "2023-01-01",
            "end_date": "2023-12-31"
        },
        read_only=True
    )
    assert validated_query is not None
    assert all(word in validated_query.query for word in [
        "created_at", "deleted_at", "set_by", "removed_flag", "merged_at"
    ]) 