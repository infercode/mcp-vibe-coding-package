#!/usr/bin/env python3
"""
Neo4j Query Utilities

This module provides utility functions for working with Neo4j queries,
including validation, sanitization, and safe execution.
"""

from typing import Dict, Any, List, Optional, Union, Set
import re
from datetime import datetime, date

from src.models.neo4j_queries import (
    CypherParameters, CypherQuery, QueryBuilder, NodePattern,
    RelationshipPattern, PathPattern, QueryOrder
)
from src.logger import get_logger

# Initialize logger
logger = get_logger()

# Regular expressions for detecting destructive operations
DESTRUCTIVE_OPERATIONS = [
    r"(?i)\b(CREATE|DELETE|REMOVE|DROP|SET)\b",
    r"(?i)\b(MERGE|DETACH DELETE)\b",
    r"(?i)\b(CREATE|DROP) (INDEX|CONSTRAINT)\b",
    r"(?i)\.drop\(.*\)"
]

def is_json_serializable(obj: Any) -> bool:
    """
    Check if an object is JSON serializable.
    
    Args:
        obj: Object to check
        
    Returns:
        bool: True if serializable, False otherwise
    """
    if obj is None:
        return True
    if isinstance(obj, (str, int, float, bool)):
        return True
    if isinstance(obj, (datetime, date)):
        return True
    if isinstance(obj, dict):
        return all(is_json_serializable(v) for v in obj.values())
    if isinstance(obj, (list, tuple, set)):
        return all(is_json_serializable(v) for v in obj)
    return False

def sanitize_query_parameters(parameters: Dict[str, Any]) -> Dict[str, Any]:
    """
    Sanitize and validate Neo4j query parameters.
    
    Args:
        parameters: Dictionary of parameters to sanitize
        
    Returns:
        Dict[str, Any]: Sanitized parameters
        
    Raises:
        ValueError: If parameters contain non-serializable values
    """
    if not parameters:
        return {}
    
    # Check if all parameters are serializable
    for key, value in parameters.items():
        if not is_json_serializable(value):
            raise ValueError(f"Parameter '{key}' contains a non-serializable value: {type(value)}")
    
    # Create and validate with Pydantic
    try:
        params_model = CypherParameters(parameters=parameters)
        return params_model.parameters
    except Exception as e:
        logger.error(f"Error validating query parameters: {e}")
        raise ValueError(f"Invalid query parameters: {e}")

def validate_query(query: str, parameters: Optional[Dict[str, Any]] = None, 
                  read_only: bool = True, database: Optional[str] = None) -> CypherQuery:
    """
    Validate a Cypher query and its parameters.
    
    Args:
        query: Cypher query string
        parameters: Dictionary of query parameters (optional)
        read_only: If True, the query must not contain destructive operations
        database: Optional database name
        
    Returns:
        CypherQuery: Validated query model
        
    Raises:
        ValueError: If query is invalid or contains destructive operations in read-only mode
    """
    # Check for destructive operations in read-only mode
    if read_only:
        for pattern in DESTRUCTIVE_OPERATIONS:
            if re.search(pattern, query):
                raise ValueError(f"Destructive operation detected in read-only query: {pattern}")
    
    # Sanitize parameters if provided
    params_model = None
    if parameters:
        try:
            sanitized_params = sanitize_query_parameters(parameters)
            params_model = CypherParameters(parameters=sanitized_params)
        except ValueError as e:
            logger.error(f"Parameter validation error: {e}")
            raise
    
    # Create and validate CypherQuery
    try:
        query_model = CypherQuery(
            query=query,
            parameters=params_model,
            read_only=read_only,
            database=database
        )
        return query_model
    except Exception as e:
        logger.error(f"Query validation error: {e}")
        raise ValueError(f"Invalid Cypher query: {e}")

def safe_execute_validated_query(manager, query_model: CypherQuery):
    """
    Safely execute a validated CypherQuery using a manager.
    
    Args:
        manager: A manager with execute_query method
        query_model: Validated CypherQuery model
        
    Returns:
        The result of the query execution
        
    Raises:
        Exception: If execution fails
    """
    try:
        # Extract parameters dictionary or empty dict if None
        parameters = {}
        if query_model.parameters:
            parameters = query_model.parameters.parameters
        
        # Execute query with the appropriate method based on read_only flag
        if query_model.read_only:
            return manager.execute_read_query(query_model.query, parameters)
        else:
            return manager.execute_write_query(query_model.query, parameters)
    except Exception as e:
        logger.error(f"Query execution error: {e}")
        raise

def safe_execute_query(manager, query: str, parameters: Optional[Dict[str, Any]] = None, 
                     read_only: bool = True, database: Optional[str] = None):
    """
    Validate and safely execute a Cypher query.
    
    Args:
        manager: A manager with execute_query method
        query: Cypher query string
        parameters: Dictionary of query parameters (optional)
        read_only: If True, the query must not contain destructive operations
        database: Optional database name
        
    Returns:
        The result of the query execution
        
    Raises:
        ValueError: If query validation fails
        Exception: If execution fails
    """
    # Validate query
    query_model = validate_query(query, parameters, read_only, database)
    
    # Execute validated query
    return safe_execute_validated_query(manager, query_model)

def create_node_query(node_label: str, properties: Dict[str, Any], 
                    variable: str = "n") -> CypherQuery:
    """
    Create a Cypher query for creating a node.
    
    Args:
        node_label: Label for the node
        properties: Properties for the node
        variable: Variable name in the query
        
    Returns:
        CypherQuery: Validated query for creating a node
    """
    query = f"CREATE ({variable}:{node_label} $props) RETURN {variable}"
    sanitized_props = sanitize_query_parameters(properties)
    return validate_query(query=query, parameters={"props": sanitized_props}, read_only=False)

def create_match_node_query(node_label: str, properties: Dict[str, Any],
                          return_fields: List[str], variable: str = "n") -> CypherQuery:
    """
    Create a Cypher query for matching nodes.
    
    Args:
        node_label: Label for the node
        properties: Properties to match
        return_fields: Fields to return
        variable: Variable name in the query
        
    Returns:
        CypherQuery: Validated query for matching nodes
    """
    # Create the return clause
    if not return_fields:
        return_clause = variable
    else:
        return_fields_with_var = [f"{variable}.{field}" for field in return_fields]
        return_clause = ", ".join(return_fields_with_var)
    
    query = f"MATCH ({variable}:{node_label} $props) RETURN {return_clause}"
    sanitized_props = sanitize_query_parameters(properties)
    return validate_query(query=query, parameters={"props": sanitized_props}, read_only=True)

def create_relationship_query(from_label: str, from_props: Dict[str, Any],
                           to_label: str, to_props: Dict[str, Any],
                           rel_type: str, rel_props: Optional[Dict[str, Any]] = None) -> CypherQuery:
    """
    Create a Cypher query for creating a relationship between nodes.
    
    Args:
        from_label: Label for the source node
        from_props: Properties to match the source node
        to_label: Label for the target node
        to_props: Properties to match the target node
        rel_type: Type of relationship
        rel_props: Properties for the relationship (optional)
        
    Returns:
        CypherQuery: Validated query for creating a relationship
    """
    # Match the nodes
    query = f"MATCH (a:{from_label} $from_props), (b:{to_label} $to_props)"
    
    # Create the relationship
    if rel_props:
        query += f" CREATE (a)-[r:{rel_type} $rel_props]->(b) RETURN a, r, b"
        parameters = {
            "from_props": from_props,
            "to_props": to_props,
            "rel_props": rel_props
        }
    else:
        query += f" CREATE (a)-[r:{rel_type}]->(b) RETURN a, r, b"
        parameters = {
            "from_props": from_props,
            "to_props": to_props
        }
    
    sanitized_params = sanitize_query_parameters(parameters)
    return validate_query(query=query, parameters=sanitized_params, read_only=False)

def build_match_query(nodes: List[NodePattern], relationships: List[RelationshipPattern],
                    return_fields: List[str], where_clauses: Optional[List[str]] = None,
                    order_by: Optional[List[QueryOrder]] = None, 
                    limit: Optional[int] = None, skip: Optional[int] = None,
                    parameters: Optional[Dict[str, Any]] = None) -> CypherQuery:
    """
    Build and validate a MATCH query using the QueryBuilder.
    
    Args:
        nodes: List of NodePattern objects
        relationships: List of RelationshipPattern objects
        return_fields: Fields to return
        where_clauses: WHERE clauses (optional)
        order_by: ORDER BY clauses (optional)
        limit: LIMIT clause (optional)
        skip: SKIP clause (optional)
        parameters: Query parameters (optional)
        
    Returns:
        CypherQuery: Validated MATCH query
    """
    # Create path patterns from nodes and relationships
    paths = []
    
    if len(nodes) == len(relationships) + 1:
        # We can create a single path
        path = PathPattern(nodes=nodes, relationships=relationships)
        paths.append(path)
    else:
        # Create individual node patterns
        paths.extend(nodes)
    
    # Build the query
    builder = QueryBuilder(
        match_patterns=paths,
        where_clauses=where_clauses or [],
        return_fields=return_fields,
        order_by=order_by,
        limit=limit,
        skip=skip,
        parameters=parameters or {}
    )
    
    # Convert to CypherQuery and validate
    return builder.to_cypher_query()

def dump_neo4j_nodes(driver, database="neo4j"):
    """
    Debug utility to dump all nodes from Neo4j.
    
    Args:
        driver: Neo4j driver instance
        database: Database name to use
        
    Returns:
        A dictionary with all nodes and relationships
    """
    if not driver:
        return {"error": "Driver not initialized"}
    
    try:
        # Query to get all nodes
        node_query = """
        MATCH (n)
        RETURN n
        LIMIT 100
        """
        
        node_results = driver.execute_query(
            node_query,
            database_=database
        )
        
        # Process node results
        nodes = []
        if node_results and node_results[0]:
            for record in node_results[0]:
                if record and "n" in record:
                    node = record["n"]
                    node_data = {
                        "labels": list(node.labels),
                        "properties": dict(node.items())
                    }
                    # Clean up embedding data to reduce output size
                    if "embedding" in node_data["properties"]:
                        embedding = node_data["properties"]["embedding"]
                        if embedding:
                            node_data["properties"]["embedding"] = f"<embedding with {len(embedding)} dimensions>"
                    
                    nodes.append(node_data)
        
        # Query to get all relationships
        rel_query = """
        MATCH ()-[r]->()
        RETURN r
        LIMIT 100
        """
        
        rel_results = driver.execute_query(
            rel_query,
            database_=database
        )
        
        # Process relationship results
        relationships = []
        if rel_results and rel_results[0]:
            for record in rel_results[0]:
                if record and "r" in record:
                    rel = record["r"]
                    rel_data = {
                        "type": rel.type,
                        "properties": dict(rel.items()),
                        "start_node_id": rel.start_node.element_id,
                        "end_node_id": rel.end_node.element_id
                    }
                    relationships.append(rel_data)
        
        # Get database statistics
        stats_query = """
        MATCH (n)
        RETURN 
          count(n) as node_count,
          sum(size(keys(n))) as properties_count,
          count(DISTINCT labels(n)) as label_count
        """
        
        stats_results = driver.execute_query(
            stats_query,
            database_=database
        )
        
        statistics = {}
        if stats_results and stats_results[0] and stats_results[0][0]:
            record = stats_results[0][0]
            statistics = {
                "node_count": record["node_count"],
                "properties_count": record["properties_count"],
                "label_count": record["label_count"],
                "relationship_count": len(relationships)
            }
        
        return {
            "statistics": statistics,
            "nodes": nodes,
            "relationships": relationships
        }
    except Exception as e:
        from src.utils.common_utils import extract_error
        return {"error": extract_error(e)} 