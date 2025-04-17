#!/usr/bin/env python3
"""
Neo4j Query Validation Example

This script demonstrates how to use the Neo4j query validation utilities
to create and execute type-safe and validated Neo4j queries.
"""

import sys
import os
import json
from datetime import datetime
from typing import Dict, Any, List, Optional

# Add the project root to the Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

# Ensure the utils directory exists
os.makedirs(os.path.join(os.path.dirname(__file__), "../utils"), exist_ok=True)

from src.models.neo4j_queries import (
    CypherParameters, CypherQuery, QueryBuilder, NodePattern,
    RelationshipPattern, PathPattern, QueryOrder
)
# Comment out the import until the utils module is properly integrated
# from src.utils.neo4j_query_utils import (
#     sanitize_query_parameters, validate_query, 
#     create_node_query, create_match_node_query, create_relationship_query
# )
from src.graph_memory.base_manager import BaseManager
from src.logger import get_logger

# Initialize logger
logger = get_logger()

# Placeholder functions until the utils module is properly integrated
def sanitize_query_parameters(parameters: Dict[str, Any]) -> Dict[str, Any]:
    """Placeholder for sanitize_query_parameters."""
    # Just return the parameters for now
    return parameters

def validate_query(query: str, parameters: Optional[Dict[str, Any]] = None, 
                  read_only: bool = True, database: Optional[str] = None) -> CypherQuery:
    """Placeholder for validate_query."""
    params_model = None
    if parameters:
        params_model = CypherParameters(parameters=parameters)
    
    return CypherQuery(
        query=query,
        parameters=params_model,
        read_only=read_only,
        database=database
    )

def create_node_query(node_label: str, properties: Dict[str, Any], variable: str = "n") -> CypherQuery:
    """Placeholder for create_node_query."""
    query = f"CREATE (n:{node_label} $props) RETURN n"
    return validate_query(query=query, parameters={"props": properties}, read_only=False)

def create_match_node_query(node_label: str, properties: Dict[str, Any],
                          return_fields: List[str], variable: str = "n") -> CypherQuery:
    """Placeholder for create_match_node_query."""
    query = f"MATCH (n:{node_label} $props) RETURN n"
    return validate_query(query=query, parameters={"props": properties}, read_only=True)

def create_relationship_query(from_label: str, from_props: Dict[str, Any],
                           to_label: str, to_props: Dict[str, Any],
                           rel_type: str, rel_props: Optional[Dict[str, Any]] = None) -> CypherQuery:
    """Placeholder for create_relationship_query."""
    query = f"MATCH (a:{from_label}), (b:{to_label}) CREATE (a)-[r:{rel_type}]->(b) RETURN a, r, b"
    return validate_query(query=query, parameters={}, read_only=False)

def print_json(data: Any, title: Optional[str] = None) -> None:
    """Print data as formatted JSON."""
    if title:
        print(f"\n=== {title} ===")
    print(json.dumps(data, indent=2, default=str))
    print()

def demonstrate_parameter_validation():
    """Demonstrate parameter validation using Pydantic models."""
    print("\n--- Parameter Validation Examples ---\n")
    
    # Example 1: Valid parameters
    valid_params = {
        "name": "John",
        "age": 30,
        "active": True,
        "tags": ["user", "admin"],
        "created_at": datetime.now()
    }
    
    try:
        sanitized = sanitize_query_parameters(valid_params)
        print_json(sanitized, "Sanitized Valid Parameters")
    except ValueError as e:
        print(f"❌ Error: {e}")
    
    # Example 2: Invalid parameters (nested object which is not supported)
    invalid_params = {
        "name": "Jane",
        "preferences": {"complex": object()}  # Contains an unsupported type
    }
    
    try:
        sanitize_query_parameters(invalid_params)
        print("✅ Parameters validated (should not happen)")
    except ValueError as e:
        print(f"❌ Expected error with invalid parameters: {e}")
    
    # Example 3: Testing with float values (should be converted to strings)
    float_params = {
        "id": 1,
        "score": 95.5,
        "ratio": 0.75
    }
    
    try:
        sanitized = sanitize_query_parameters(float_params)
        print_json(sanitized, "Parameters with Floats Converted")
    except ValueError as e:
        print(f"❌ Error: {e}")

def demonstrate_query_validation():
    """Demonstrate query validation using Pydantic models."""
    print("\n--- Query Validation Examples ---\n")
    
    # Example 1: Valid read-only query
    valid_query = "MATCH (p:Person {name: $name}) RETURN p"
    valid_params = {"name": "John"}
    
    try:
        query_model = validate_query(valid_query, valid_params, read_only=True)
        print_json(query_model.model_dump(), "Valid Read-Only Query")
    except ValueError as e:
        print(f"❌ Error: {e}")
    
    # Example 2: Invalid read-only query (contains CREATE)
    invalid_query = "MATCH (p:Person {name: $name}) CREATE (n:Note {text: 'Hello'}) RETURN p, n"
    
    try:
        validate_query(invalid_query, valid_params, read_only=True)
        print("✅ Query validated (should not happen)")
    except ValueError as e:
        print(f"❌ Expected error with invalid read-only query: {e}")
    
    # Example 3: Valid write query
    write_query = "CREATE (p:Person {name: $name, age: $age}) RETURN p"
    write_params = {"name": "Alice", "age": 25}
    
    try:
        query_model = validate_query(write_query, write_params, read_only=False)
        print_json(query_model.model_dump(), "Valid Write Query")
    except ValueError as e:
        print(f"❌ Error: {e}")

def demonstrate_query_builder():
    """Demonstrate query builder with validation."""
    print("\n--- Query Builder Examples ---\n")
    
    # Example 1: Building a simple match query
    builder = QueryBuilder(
        match_patterns=[],
        where_clauses=[],
        return_fields=[],
        order_by=None,
        limit=None,
        skip=None,
        parameters={}
    )
    
    # Add a node pattern to match
    person_node = NodePattern(
        variable="p",
        labels=["Person"],
        properties={"name": "$name"}
    )
    builder.match_patterns.append(person_node)
    
    # Add where clause
    builder.where_clauses.append("p.age > $min_age")
    
    # Add return fields
    builder.return_fields = ["p.name", "p.age", "p.email"]
    
    # Add order by
    builder.order_by = [QueryOrder(field="p.name", direction="ASC")]
    
    # Add limit
    builder.limit = 10
    
    # Add parameters
    builder.parameters = {"name": "J%", "min_age": 21}
    
    # Convert to CypherQuery
    query = builder.to_cypher_query()
    
    print_json(query.model_dump(), "Query Builder Result")
    
    # Example 2: Building a more complex query with path
    complex_builder = QueryBuilder(
        match_patterns=[],
        where_clauses=[],
        return_fields=[],
        order_by=None,
        limit=None,
        skip=None,
        parameters={}
    )
    
    # Create node patterns
    person_node = NodePattern(
        variable="p",
        labels=["Person"],
        properties={"name": "$name"}
    )
    
    company_node = NodePattern(
        variable="c",
        labels=["Company"],
        properties={}
    )
    
    # Create relationship pattern
    works_at_rel = RelationshipPattern(
        variable="r",
        type="WORKS_AT",
        properties={},
        direction="OUTGOING"
    )
    
    # Create path pattern with these nodes and relationship
    path = PathPattern(
        nodes=[person_node, company_node],
        relationships=[works_at_rel]
    )
    
    # Add path to match patterns
    complex_builder.match_patterns.append(path)
    
    # Add where clause
    complex_builder.where_clauses.append("c.founded < $year")
    
    # Add return fields
    complex_builder.return_fields = ["p.name", "c.name", "r.since"]
    
    # Add parameters
    complex_builder.parameters = {"name": "Alice", "year": 2010}
    
    # Convert to CypherQuery
    complex_query = complex_builder.to_cypher_query()
    
    print_json(complex_query.model_dump(), "Complex Query Builder Result")

def demonstrate_utility_functions():
    """Demonstrate utility functions for common query patterns."""
    print("\n--- Utility Function Examples ---\n")
    
    # Example 1: Create node query
    properties = {
        "name": "John Smith",
        "age": 30,
        "created_at": datetime.now()
    }
    
    node_query = create_node_query("Person", properties)
    print_json(node_query.model_dump(), "Create Node Query")
    
    # Example 2: Match node query
    match_properties = {"name": "John Smith"}
    return_fields = ["name", "age", "email"]
    
    match_query = create_match_node_query("Person", match_properties, return_fields)
    print_json(match_query.model_dump(), "Match Node Query")
    
    # Example 3: Create relationship query
    from_props = {"name": "John Smith"}
    to_props = {"name": "ACME Corp"}
    rel_props = {"since": "2020", "position": "Developer"}
    
    rel_query = create_relationship_query(
        "Person", from_props, 
        "Company", to_props, 
        "WORKS_AT", rel_props
    )
    print_json(rel_query.model_dump(), "Create Relationship Query")

def demonstrate_security_features():
    """Demonstrate security features of the query validation."""
    print("\n--- Security Features Examples ---\n")
    
    # Example 1: Detecting destructive operations in read-only mode
    insecure_queries = [
        "MATCH (n) DELETE n",
        "MATCH (n) SET n.sensitive = 'exposed'",
        "MATCH (n) REMOVE n.required",
        "DROP CONSTRAINT unique_name_constraint",
        "CALL db.index.fulltext.drop('node_fulltext')"
    ]
    
    for i, query in enumerate(insecure_queries, 1):
        try:
            validate_query(query, read_only=True)
            print(f"❌ Security breach! Query {i} was not detected as destructive")
        except ValueError as e:
            print(f"✅ Security protection {i}: {e}")
    
    # Example 2: Parameter injection protection
    try:
        bad_params = {"name": "John'; DROP DATABASE neo4j;--"}
        sanitize_query_parameters(bad_params)
        
        # This would pass validation because it's just a string - the real protection is
        # that parameters are passed separately from the query to Neo4j's driver
        print("\n✅ Parameters like SQL injection attempts are passed as literal strings and can't execute")
    except ValueError as e:
        print(f"\n❌ Error with parameters: {e}")

def main():
    """Run all demonstration functions."""
    print("=== Neo4j Query Validation Example ===")
    
    # Run the demonstrations
    demonstrate_parameter_validation()
    demonstrate_query_validation()
    demonstrate_query_builder()
    demonstrate_utility_functions()
    demonstrate_security_features()
    
    print("\n=== End of Demo ===")

if __name__ == "__main__":
    main() 