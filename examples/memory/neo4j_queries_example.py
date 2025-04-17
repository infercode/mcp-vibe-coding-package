#!/usr/bin/env python3
"""
Neo4j Queries Example

This script demonstrates how to use the Neo4j query models for building
and validating Cypher queries in a type-safe manner.
"""

import logging
from typing import Dict, Any, List
import datetime
import sys
import os

# Add the project root to the Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from src.models.neo4j_queries import (
    # Parameter models
    CypherString, CypherNumber, CypherBoolean, CypherList, CypherDict,
    NodeProperties, RelationshipProperties,
    
    # Query models
    CypherParameters, CypherQuery,
    
    # Result models
    NodeResult, RelationshipResult, PathResult, QueryResult,
    
    # Query builder models
    NodePattern, RelationshipPattern, PathPattern, QueryBuilder, QueryOrder
)

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def demonstrate_parameter_models():
    """Demonstrate how to use parameter models for type safety."""
    logger.info("Demonstrating parameter models")
    
    # String parameter
    string_param = CypherString(value="John Doe")
    logger.info(f"String parameter: {string_param.to_cypher_param()}")
    
    # Try invalid string with null byte (should raise error)
    try:
        CypherString(value="Bad\0String")
    except ValueError as e:
        logger.info(f"Validation caught invalid string: {e}")
    
    # Number parameter
    int_param = CypherNumber(value=42)
    float_param = CypherNumber(value=3.14)
    logger.info(f"Int parameter: {int_param.to_cypher_param()}")
    logger.info(f"Float parameter: {float_param.to_cypher_param()} (converted to string)")
    
    # Boolean parameter
    bool_param = CypherBoolean(value=True)
    logger.info(f"Boolean parameter: {bool_param.to_cypher_param()}")
    
    # List parameter
    list_param = CypherList(values=["apple", "banana", 3, True])
    logger.info(f"List parameter: {list_param.to_cypher_param()}")
    
    # Try invalid list (should raise error)
    try:
        CypherList(values=["apple", {"key": "value"}])  # Dict not allowed in list
    except ValueError as e:
        logger.info(f"Validation caught invalid list: {e}")
    
    # Dict parameter
    dict_param = CypherDict(values={"name": "John", "age": 30, "active": True})
    logger.info(f"Dict parameter: {dict_param.to_cypher_param()}")
    
    # NodeProperties
    node_props = NodeProperties(
        name="Entity1",
        type="Person",
        created_at=datetime.datetime.now(),
        updated_at=datetime.datetime.now(),
        tags=["user", "customer"],
        additional_properties={"score": 95, "status": "active"}
    )
    logger.info(f"Node properties: {node_props.model_dump()}")
    
    # RelationshipProperties
    rel_props = RelationshipProperties(
        type="KNOWS",
        weight=0.75,
        created_at=datetime.datetime.now(),
        confidence=0.9,
        additional_properties={"since": "2020", "context": "work"}
    )
    logger.info(f"Relationship properties: {rel_props.model_dump()}")
    
    return {
        "string_param": string_param,
        "int_param": int_param,
        "float_param": float_param,
        "bool_param": bool_param,
        "list_param": list_param,
        "dict_param": dict_param,
        "node_props": node_props,
        "rel_props": rel_props
    }


def demonstrate_query_models():
    """Demonstrate how to use Cypher query models."""
    logger.info("Demonstrating query models")
    
    # Create parameters 
    params = CypherParameters(parameters={
        "name": "John",
        "age": 30,
        "tags": ["user", "admin"],
        "active": True,
        "settings": {"theme": "dark", "notifications": True}
    })
    logger.info(f"Query parameters: {params.to_neo4j_parameters()}")
    
    # Create a read-only query
    read_query = CypherQuery(
        query="MATCH (p:Person) WHERE p.name = $name RETURN p",
        parameters=params,
        database=None,
        read_only=True
    )
    logger.info(f"Read query: {read_query.to_executable()}")
    
    # Try invalid read-only query (should raise error)
    try:
        CypherQuery(
            query="CREATE (p:Person {name: $name}) RETURN p",
            parameters=params,
            database=None,
            read_only=True
        )
    except ValueError as e:
        logger.info(f"Validation caught invalid read-only query: {e}")
    
    # Create a write query
    write_query = CypherQuery(
        query="CREATE (p:Person {name: $name, age: $age}) RETURN p",
        parameters=params,
        database=None,
        read_only=False
    )
    logger.info(f"Write query: {write_query.to_executable()}")
    
    return {
        "params": params,
        "read_query": read_query,
        "write_query": write_query
    }


def simulate_neo4j_result():
    """Simulate a Neo4j result for demonstration purposes."""
    # Create a mock Node
    class MockNode:
        def __init__(self, id, labels, properties):
            self.id = id
            self.labels = labels
            self._properties = properties
            
        def items(self):
            return self._properties.items()
    
    # Create a mock Relationship
    class MockRelationship:
        def __init__(self, id, type, properties, start_node, end_node):
            self.id = id
            self.type = type
            self._properties = properties
            self.start_node = start_node
            self.end_node = end_node
            
        def items(self):
            return self._properties.items()
    
    # Create a mock Path
    class MockPath:
        def __init__(self, nodes, relationships):
            self.nodes = nodes
            self.relationships = relationships
    
    # Create some mock nodes
    person_node = MockNode(
        id=1, 
        labels=["Person"], 
        properties={"name": "John", "age": 30}
    )
    
    company_node = MockNode(
        id=2, 
        labels=["Company"], 
        properties={"name": "ACME Inc.", "founded": 2005}
    )
    
    # Create a mock relationship
    works_at_rel = MockRelationship(
        id=101, 
        type="WORKS_AT", 
        properties={"since": 2018, "role": "Developer"},
        start_node=person_node,
        end_node=company_node
    )
    
    # Create a mock path
    path = MockPath(
        nodes=[person_node, company_node],
        relationships=[works_at_rel]
    )
    
    # Convert to result models
    node_result = NodeResult.from_neo4j_node(person_node)
    logger.info(f"Node result: {node_result.model_dump()}")
    
    rel_result = RelationshipResult.from_neo4j_relationship(works_at_rel)
    logger.info(f"Relationship result: {rel_result.model_dump()}")
    
    path_result = PathResult.from_neo4j_path(path)
    logger.info(f"Path result: {path_result.model_dump()}")
    
    # Also create a relationship from dictionary to test dictionary handling
    dict_rel = {
        "id": "202",
        "type": "MANAGES",
        "properties": {"department": "Engineering", "since": 2020},
        "start_node_id": "1",
        "end_node_id": "2"
    }
    dict_rel_result = RelationshipResult.from_neo4j_relationship(dict_rel)
    logger.info(f"Dictionary relationship result: {dict_rel_result.model_dump()}")
    
    # Create a mock query result
    records = [
        {"person": person_node, "company": company_node, "works_at": works_at_rel},
        {"path": path}
    ]
    
    summary = {
        "query_type": "READ",
        "counters": {"nodes_created": 0, "relationships_created": 0},
        "query_time": 15  # ms
    }
    
    query_result = QueryResult.from_neo4j_result(records, summary)
    logger.info(f"Query result: {query_result.model_dump()}")
    
    return {
        "node_result": node_result,
        "rel_result": rel_result,
        "path_result": path_result,
        "query_result": query_result
    }


def demonstrate_query_builder():
    """Demonstrate how to use the query builder for safer query construction."""
    logger.info("Demonstrating query builder")
    
    # Create node patterns
    person_node = NodePattern(
        variable="p",
        labels=["Person"],
        properties={"name": "John"}
    )
    logger.info(f"Person node pattern: {person_node.to_cypher()}")
    
    company_node = NodePattern(
        variable="c",
        labels=["Company"],
        properties={}
    )
    logger.info(f"Company node pattern: {company_node.to_cypher()}")
    
    # Create relationship pattern
    works_at_rel = RelationshipPattern(
        variable="r",
        type="WORKS_AT",
        properties={},
        direction="OUTGOING"
    )
    logger.info(f"Works at relationship pattern: {works_at_rel.to_cypher()}")
    
    # Create a path pattern
    works_at_path = PathPattern(
        nodes=[person_node, company_node],
        relationships=[works_at_rel]
    )
    logger.info(f"Works at path pattern: {works_at_path.to_cypher()}")
    
    # Create query builder
    query_builder = QueryBuilder(
        match_patterns=[works_at_path],
        where_clauses=["p.age > $min_age"],
        return_fields=["p", "r", "c"],
        order_by=[QueryOrder(field="p.name", direction="ASC")],
        limit=10,
        skip=0,
        parameters={"min_age": 25}
    )
    
    # Convert to CypherQuery
    query = query_builder.to_cypher_query()
    logger.info(f"Built query: {query.query}")
    logger.info(f"With parameters: {query.parameters.parameters if query.parameters else {}}")
    
    return {
        "person_node": person_node,
        "company_node": company_node,
        "works_at_rel": works_at_rel,
        "works_at_path": works_at_path,
        "query_builder": query_builder,
        "query": query
    }


def main():
    """Run the Neo4j query models examples."""
    logger.info("Starting Neo4j query models examples")
    
    # Demonstrate parameter models
    param_results = demonstrate_parameter_models()
    logger.info("Parameter models demonstration complete")
    
    # Demonstrate query models
    query_results = demonstrate_query_models()
    logger.info("Query models demonstration complete")
    
    # Simulate Neo4j results
    result_models = simulate_neo4j_result()
    logger.info("Result models demonstration complete")
    
    # Demonstrate query builder
    builder_results = demonstrate_query_builder()
    logger.info("Query builder demonstration complete")
    
    logger.info("All Neo4j query models examples complete")


if __name__ == "__main__":
    main() 