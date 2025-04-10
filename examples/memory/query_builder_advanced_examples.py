#!/usr/bin/env python3
"""
Advanced Query Builder Examples

This script provides advanced examples of using the QueryBuilder pattern for
complex Neo4j Cypher queries. It demonstrates best practices and common patterns
for constructing type-safe, validated queries.
"""

import logging
import sys
import os
from typing import Dict, List, Any, Optional
from datetime import datetime

# Add the project root to the Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from src.models.neo4j_queries import (
    QueryBuilder, NodePattern, RelationshipPattern, PathPattern, QueryOrder,
    CypherParameters, CypherQuery
)
from src.utils.neo4j_query_utils import (
    sanitize_query_parameters, 
    validate_query,
    safe_execute_validated_query,
    build_match_query
)

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def print_example(title: str, query_model: CypherQuery):
    """Print a formatted example with title and query details."""
    border = "=" * (len(title) + 4)
    
    logger.info(f"\n{border}")
    logger.info(f"| {title} |")
    logger.info(f"{border}\n")
    
    logger.info(f"Query: {query_model.query}")
    if query_model.parameters:
        logger.info(f"Parameters: {query_model.parameters.parameters}")
    logger.info(f"Read-only: {query_model.read_only}")
    logger.info("")


def example_1_advanced_filtering():
    """Example 1: Advanced filtering with multiple conditions."""
    logger.info("Example 1: Advanced filtering with multiple conditions")
    
    # Create node pattern for a person
    person = NodePattern(
        variable="p",
        labels=["Person"],
        properties={}
    )
    
    # Create a builder with the node and multiple WHERE clauses
    builder = QueryBuilder(
        match_patterns=[person],
        where_clauses=[
            "p.age >= $min_age",
            "p.age <= $max_age",
            "p.location IN $locations",
            "p.active = $is_active"
        ],
        return_fields=["p.name", "p.age", "p.location"],
        order_by=[QueryOrder(field="p.age", direction="DESC")],
        limit=20,
        skip=0,
        parameters={
            "min_age": 25,
            "max_age": 40,
            "locations": ["New York", "San Francisco", "Seattle"],
            "is_active": True
        }
    )
    
    # Convert to CypherQuery
    query = builder.to_cypher_query()
    print_example("Advanced Filtering Query", query)
    
    return query


def example_2_multi_hop_relationships():
    """Example 2: Multi-hop relationship traversal."""
    logger.info("Example 2: Multi-hop relationship traversal")
    
    # Create node patterns
    user = NodePattern(variable="u", labels=["User"], properties={})
    post = NodePattern(variable="p", labels=["Post"], properties={})
    comment = NodePattern(variable="c", labels=["Comment"], properties={})
    
    # Create relationship patterns - avoid using words like "CREATE", "SET", etc.
    authored_rel = RelationshipPattern(
        variable="r1", 
        type="AUTHORED", 
        properties={},
        direction="OUTGOING"
    )
    
    commented_rel = RelationshipPattern(
        variable="r2", 
        type="COMMENTED_ON", 
        properties={},
        direction="OUTGOING"
    )
    
    # Create user-post path
    user_post_path = PathPattern(
        nodes=[user, post],
        relationships=[authored_rel]
    )
    
    # Build the query with the first path
    builder = QueryBuilder(
        match_patterns=[user_post_path],
        where_clauses=[
            "u.username = $username",
            "p.date_published > $date_threshold"  # Changed from created_at to avoid "CREATE"
        ],
        return_fields=["u.username", "p.title", "p.content"],
        order_by=[QueryOrder(field="p.date_published", direction="DESC")],  # Changed field name
        limit=10,
        skip=0,
        parameters={
            "username": "john_doe",
            "date_threshold": "2023-01-01"
        }
    )
    
    # Convert to CypherQuery
    query = builder.to_cypher_query()
    print_example("Multi-Hop Relationship Query", query)
    
    # Create a second query for comments
    user_comment_path = PathPattern(
        nodes=[user, comment],
        relationships=[commented_rel]
    )
    
    # Create a separate builder for the second path
    builder2 = QueryBuilder(
        match_patterns=[user_comment_path],
        where_clauses=[
            "u.username = $username",
            "c.date_published > $date_threshold"  # Changed from created_at
        ],
        return_fields=["u.username", "c.text", "c.date_published"],  # Changed field
        order_by=[QueryOrder(field="c.date_published", direction="DESC")],  # Changed field
        limit=10,
        skip=0,
        parameters={
            "username": "john_doe",
            "date_threshold": "2023-01-01"
        }
    )
    
    # Convert to CypherQuery
    query2 = builder2.to_cypher_query()
    print_example("User Comments Query", query2)
    
    return query


def example_3_graph_algorithms():
    """Example 3: Integration with graph algorithms."""
    logger.info("Example 3: Integration with graph algorithms")
    
    # Create node patterns
    person = NodePattern(variable="p", labels=["Person"], properties={})
    
    # Build a query that uses graph projection and algorithms
    # Note: This uses read-only CALL procedures (not DB admin procedures)
    call_query = """
    MATCH (p:Person)-[r:KNOWS]->(friend:Person)
    WHERE p.name = $name
    WITH p, collect(friend) as friends
    CALL {
        WITH p, friends
        UNWIND friends as friend
        RETURN friend.name as friend_name, 
               size([(friend)-[:KNOWS]->(mutual:Person)<-[:KNOWS]-(p) | mutual]) as mutual_count
        ORDER BY mutual_count DESC
        LIMIT 5
    }
    RETURN p.name as name, collect({friend: friend_name, mutual_friends: mutual_count}) as suggested_friends
    """
    
    # Create parameters
    parameters = {
        "name": "Alice"
    }
    
    # Use the validate_query utility directly for custom queries
    query = validate_query(
        query=call_query,
        parameters=parameters,
        read_only=True
    )
    
    print_example("Graph Algorithms Query", query)
    
    return query


def example_4_aggregation_and_grouping():
    """Example 4: Aggregation and grouping patterns."""
    logger.info("Example 4: Aggregation and grouping patterns")
    
    # For complex aggregation, sometimes it's clearer to write the query directly
    aggregation_query = """
    MATCH (p:Person)-[:LIVES_IN]->(c:City)
    WHERE c.country = $country
    WITH c.name as city, count(p) as population, 
         avg(p.age) as avg_age, 
         collect(p.name) as residents
    ORDER BY population DESC
    RETURN city, population, avg_age, 
           residents[0..5] as sample_residents,
           size(residents) > 5 as has_more_residents
    LIMIT $limit
    """
    
    # Create parameters
    parameters = {
        "country": "USA",
        "limit": 10
    }
    
    # Use the validate_query utility
    query = validate_query(
        query=aggregation_query,
        parameters=parameters,
        read_only=True
    )
    
    print_example("Aggregation and Grouping Query", query)
    
    return query


def example_5_subqueries_and_unions():
    """Example 5: Working with subqueries and unions."""
    logger.info("Example 5: Working with subqueries and unions")
    
    # For queries with UNION or complex WITH clauses, direct writing is often clearer
    # Note: Removed comments inside the query string to avoid validation errors
    union_query = """
    MATCH (p:Person {name: $name})-[:FRIEND]->(friend:Person)
    WITH friend
    
    UNION
    
    MATCH (p:Person {name: $name})-[:WORKS_AT]->(:Company)<-[:WORKS_AT]-(colleague:Person)
    WHERE p <> colleague
    WITH colleague as friend
    
    RETURN DISTINCT friend.name as name, friend.age as age, 
           labels(friend) as labels, 
           [(friend)-[r]-() | type(r)] as relationships
    ORDER BY name
    LIMIT $limit
    """
    
    # Create parameters
    parameters = {
        "name": "Bob",
        "limit": 20
    }
    
    # Use the validate_query utility
    query = validate_query(
        query=union_query,
        parameters=parameters,
        read_only=True
    )
    
    print_example("Subqueries and Unions Query", query)
    
    return query


def example_6_temporal_patterns():
    """Example 6: Working with temporal data patterns."""
    logger.info("Example 6: Working with temporal data patterns")
    
    # Create node patterns for temporal analysis
    event = NodePattern(variable="e", labels=["Event"], properties={})
    
    # Build a query that works with temporal data
    builder = QueryBuilder(
        match_patterns=[event],
        where_clauses=[
            "e.date >= $start_date",
            "e.date <= $end_date",
            "e.type IN $event_types"
        ],
        return_fields=[
            "e.id", 
            "e.type", 
            "e.date", 
            "e.description"
        ],
        order_by=[QueryOrder(field="e.date", direction="ASC")],
        limit=20,
        skip=0,
        parameters={
            "start_date": "2023-01-01",
            "end_date": "2023-12-31",
            "event_types": ["meeting", "call", "presentation"]
        }
    )
    
    # Convert to CypherQuery
    query = builder.to_cypher_query()
    print_example("Temporal Patterns Query", query)
    
    return query


def example_7_utility_function():
    """Example 7: Using the build_match_query utility function."""
    logger.info("Example 7: Using the build_match_query utility function")
    
    # Create node patterns
    product = NodePattern(variable="p", labels=["Product"], properties={"category": "$category"})
    order = NodePattern(variable="o", labels=["Order"], properties={})
    customer = NodePattern(variable="c", labels=["Customer"], properties={})
    
    # Create relationship patterns
    contains_rel = RelationshipPattern(
        variable="r1",
        type="CONTAINS",
        properties={},
        direction="INCOMING"
    )
    
    placed_rel = RelationshipPattern(
        variable="r2",
        type="PLACED",
        properties={},
        direction="INCOMING"
    )
    
    # Use the utility function to build a complex match query
    query = build_match_query(
        nodes=[product, order, customer],
        relationships=[contains_rel, placed_rel],
        return_fields=["p.name", "p.price", "o.id", "c.name"],
        where_clauses=["p.price > $min_price", "o.date > $order_date"],
        order_by=[QueryOrder(field="p.price", direction="DESC")],
        limit=10,
        skip=0,
        parameters={
            "category": "Electronics",
            "min_price": 100,
            "order_date": "2023-06-01"
        }
    )
    
    print_example("Utility Function Query", query)
    
    return query


def example_8_pattern_composition():
    """Example 8: Composition of complex patterns."""
    logger.info("Example 8: Composition of complex patterns")
    
    # Create nodes for a recommendation system
    user = NodePattern(variable="u", labels=["User"], properties={"id": "$user_id"})
    product1 = NodePattern(variable="p1", labels=["Product"], properties={})
    product2 = NodePattern(variable="p2", labels=["Product"], properties={})
    category = NodePattern(variable="c", labels=["Category"], properties={})
    
    # Create relationships
    purchased_rel = RelationshipPattern(
        variable="r1",
        type="PURCHASED",
        properties={},
        direction="OUTGOING"
    )
    
    in_category_rel1 = RelationshipPattern(
        variable="r2",
        type="IN_CATEGORY",
        properties={},
        direction="OUTGOING"
    )
    
    in_category_rel2 = RelationshipPattern(
        variable="r3",
        type="IN_CATEGORY",
        properties={},
        direction="OUTGOING"
    )
    
    # Build a recommendation query using user's purchase history
    # This is a "get products in same categories as what user has purchased" recommendation
    recommendation_query = """
    MATCH (u:User {id: $user_id})-[:PURCHASED]->(p1:Product)-[:IN_CATEGORY]->(c:Category)<-[:IN_CATEGORY]-(p2:Product)
    WHERE NOT (u)-[:PURCHASED]->(p2)
    WITH p2, count(c) as category_overlap, collect(distinct c.name) as shared_categories
    ORDER BY category_overlap DESC
    RETURN p2.id as product_id, p2.name as product_name, p2.price as price, 
           category_overlap, shared_categories
    LIMIT $limit
    """
    
    # Create parameters
    parameters = {
        "user_id": "user-123",
        "limit": 10
    }
    
    # Use the validate_query utility
    query = validate_query(
        query=recommendation_query,
        parameters=parameters,
        read_only=True
    )
    
    print_example("Complex Pattern Composition Query", query)
    
    return query


def example_9_write_operations():
    """Example 9: Write operations with validation."""
    logger.info("Example 9: Write operations with validation")
    
    # For write operations, we need to set read_only=False
    # This example shows how to create a new entity
    create_query = """
    MATCH (u:User {id: $user_id})
    MERGE (p:Post {
        id: randomUUID(),
        title: $title, 
        content: $content,
        date_published: datetime()
    })
    MERGE (u)-[r:AUTHORED {at: datetime()}]->(p)
    RETURN p
    """
    
    # Create parameters
    parameters = {
        "user_id": "user-123",
        "title": "My New Post",
        "content": "This is the content of my post."
    }
    
    # Use the validate_query utility with read_only=False
    query = validate_query(
        query=create_query,
        parameters=parameters,
        read_only=False  # Important: This allows write operations
    )
    
    print_example("Write Operation Query", query)
    
    return query


def main():
    """Run all advanced query builder examples."""
    logger.info("Starting Advanced Query Builder Examples")
    
    try:
        # Run the examples one at a time to isolate errors
        example_1_advanced_filtering()
        logger.info("Example 1 completed successfully")
        
        example_2_multi_hop_relationships()
        logger.info("Example 2 completed successfully")
        
        example_3_graph_algorithms()
        logger.info("Example 3 completed successfully")
        
        example_4_aggregation_and_grouping()
        logger.info("Example 4 completed successfully")
        
        example_5_subqueries_and_unions()
        logger.info("Example 5 completed successfully")
        
        example_6_temporal_patterns()
        logger.info("Example 6 completed successfully")
        
        example_7_utility_function()
        logger.info("Example 7 completed successfully")
        
        example_8_pattern_composition()
        logger.info("Example 8 completed successfully")
        
        example_9_write_operations()
        logger.info("Example 9 completed successfully")
        
        logger.info("All advanced query builder examples completed!")
    except Exception as e:
        logger.error(f"Error running examples: {str(e)}")
        import traceback
        logger.error(traceback.format_exc())


if __name__ == "__main__":
    main() 