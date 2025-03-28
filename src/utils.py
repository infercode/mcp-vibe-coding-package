import json
import random
import string
import uuid
import os
from typing import Any, Dict, Optional


def extract_error(error: Exception) -> str:
    """
    Extract the error message from an exception.
    
    Args:
        error: The exception to extract information from
        
    Returns:
        A string containing the error message
    """
    return f"{type(error).__name__}: {str(error)}"


def generate_id(prefix: str = "") -> str:
    """
    Generate a unique ID with an optional prefix.
    
    Args:
        prefix: Optional prefix for the ID (e.g., "les" for lesson, "obs" for observation)
    
    Returns:
        A unique ID string with optional prefix
    """
    unique_id = str(uuid.uuid4())
    if prefix:
        return f"{prefix}_{unique_id}"
    return unique_id


def dict_to_json(data: Dict[str, Any]) -> str:
    """
    Convert a dictionary to a JSON string.
    
    Args:
        data: The dictionary to convert
        
    Returns:
        A JSON string representation of the dictionary
    """
    return json.dumps(data, default=str)


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
        return {"error": extract_error(e)} 