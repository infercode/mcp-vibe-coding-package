#!/usr/bin/env python3
"""
Query Builder Utilities

This module provides utility functions for building common query patterns
using the QueryBuilder model. These utilities simplify creating complex
Cypher queries while maintaining type safety and validation.
"""

from typing import Dict, List, Any, Optional, Union, Tuple, Literal
from src.models.neo4j_queries import (
    QueryBuilder, NodePattern, RelationshipPattern, PathPattern, 
    QueryOrder, CypherQuery
)


def build_entity_search_query(
    entity_label: str,
    search_property: str,
    search_value: str,
    exact_match: bool = False,
    additional_properties: Optional[Dict[str, Any]] = None,
    additional_filters: Optional[List[str]] = None,
    return_fields: Optional[List[str]] = None,
    order_by_property: Optional[str] = None,
    order_direction: Literal["ASC", "DESC"] = "ASC",
    limit: int = 10,
    skip: int = 0
) -> CypherQuery:
    """
    Build a query to search entities based on property values.
    
    Args:
        entity_label: The label of the entity to search (e.g., 'Person')
        search_property: The property to search on (e.g., 'name')
        search_value: The value to search for
        exact_match: Whether to match exactly or use CONTAINS
        additional_properties: Additional properties to filter on
        additional_filters: Additional WHERE clause filters
        return_fields: Fields to return (defaults to all)
        order_by_property: Property to order by
        order_direction: Direction to order by ('ASC' or 'DESC')
        limit: Maximum number of results to return
        skip: Number of results to skip
        
    Returns:
        A validated CypherQuery object
    """
    # Create node pattern
    properties = additional_properties or {}
    node = NodePattern(
        variable="e",
        labels=[entity_label],
        properties={}
    )
    
    # Create builder
    builder = QueryBuilder(
        match_patterns=[node],
        where_clauses=[],
        return_fields=return_fields or ["e"],
        order_by=None,
        limit=limit,
        skip=skip,
        parameters={}
    )
    
    # Set up search condition
    if exact_match:
        builder.where_clauses.append(f"e.{search_property} = $search_value")
    else:
        builder.where_clauses.append(f"e.{search_property} CONTAINS $search_value")
    
    builder.parameters["search_value"] = search_value
    
    # Add additional property filters
    if additional_properties:
        for prop_name, prop_value in additional_properties.items():
            builder.where_clauses.append(f"e.{prop_name} = ${prop_name}")
            builder.parameters[prop_name] = prop_value
    
    # Add additional filters
    if additional_filters:
        builder.where_clauses.extend(additional_filters)
    
    # Add order by
    if order_by_property:
        builder.order_by = [QueryOrder(field=f"e.{order_by_property}", direction=order_direction)]
    
    return builder.to_cypher_query()


def build_relationship_query(
    from_label: str,
    from_id_property: str,
    from_id_value: str,
    relationship_type: str,
    to_label: Optional[str] = None,
    return_paths: bool = False,
    relationship_direction: Literal["OUTGOING", "INCOMING", "BOTH"] = "OUTGOING",
    additional_filters: Optional[List[str]] = None,
    limit: int = 50,
    skip: int = 0
) -> CypherQuery:
    """
    Build a query to find relationships between entities.
    
    Args:
        from_label: Label of the starting entity
        from_id_property: Property to identify the starting entity
        from_id_value: Value of the identifying property
        relationship_type: Type of relationship to find
        to_label: Label of the target entity (optional)
        return_paths: Whether to return full paths or just entities
        relationship_direction: Direction of relationship (OUTGOING, INCOMING, BOTH)
        additional_filters: Additional WHERE clause filters
        limit: Maximum number of results to return
        skip: Number of results to skip
        
    Returns:
        A validated CypherQuery object
    """
    # Create node patterns
    from_node = NodePattern(
        variable="from",
        labels=[from_label],
        properties={from_id_property: f"${from_id_property}"}
    )
    
    to_labels = [to_label] if to_label else []
    to_node = NodePattern(
        variable="to",
        labels=to_labels,
        properties={}
    )
    
    # Create relationship pattern
    rel = RelationshipPattern(
        variable="r",
        type=relationship_type,
        properties={},
        direction=relationship_direction
    )
    
    # Create path pattern
    path = PathPattern(
        nodes=[from_node, to_node],
        relationships=[rel]
    )
    
    # Create builder
    builder = QueryBuilder(
        match_patterns=[path],
        where_clauses=additional_filters or [],
        return_fields=[],
        order_by=None,
        limit=limit,
        skip=skip,
        parameters={from_id_property: from_id_value}
    )
    
    # Set return fields based on what to return
    if return_paths:
        builder.return_fields = ["from", "r", "to"]
    else:
        builder.return_fields = ["to"]
    
    return builder.to_cypher_query()


def build_property_aggregation_query(
    entity_label: str,
    group_by_property: str,
    aggregate_property: str,
    aggregation_type: str = "count",
    filters: Optional[List[str]] = None,
    having_clause: Optional[str] = None,
    limit: int = 20,
    skip: int = 0
) -> CypherQuery:
    """
    Build a query to aggregate properties across entities.
    
    Args:
        entity_label: Label of the entity to aggregate
        group_by_property: Property to group by
        aggregate_property: Property to aggregate
        aggregation_type: Type of aggregation (count, sum, avg, min, max)
        filters: WHERE clause filters
        having_clause: HAVING clause for filtering aggregated results
        limit: Maximum number of results to return
        skip: Number of results to skip
        
    Returns:
        A validated CypherQuery object
    """
    # Validate aggregation type
    valid_aggregations = ["count", "sum", "avg", "min", "max"]
    if aggregation_type not in valid_aggregations:
        raise ValueError(f"Invalid aggregation type. Must be one of: {valid_aggregations}")
    
    # Create aggregation query directly
    aggregation = f"{aggregation_type}(e.{aggregate_property})" if aggregation_type != "count" else "count(*)"
    
    query = f"""
    MATCH (e:{entity_label})
    """
    
    # Add filters
    if filters:
        query += f"WHERE {' AND '.join(filters)}\n"
    
    # Add grouping and aggregation
    query += f"""
    WITH e.{group_by_property} as group_key, {aggregation} as agg_value
    """
    
    # Add having clause
    if having_clause:
        query += f"HAVING {having_clause}\n"
    
    # Add return and ordering
    query += f"""
    RETURN group_key, agg_value
    ORDER BY agg_value DESC
    LIMIT {limit}
    SKIP {skip}
    """
    
    # For this type of query, we use the validate_query function directly
    from src.utils.neo4j_query_utils import validate_query
    return validate_query(query=query, read_only=True)


def build_path_finding_query(
    start_label: str,
    start_property: str,
    start_value: str,
    end_label: str,
    end_property: str,
    end_value: str,
    min_depth: int = 1,
    max_depth: int = 3,
    relationship_types: Optional[List[str]] = None,
    limit: int = 5
) -> CypherQuery:
    """
    Build a query to find paths between two entities.
    
    Args:
        start_label: Label of the starting entity
        start_property: Property to identify the starting entity
        start_value: Value of the starting entity property
        end_label: Label of the ending entity
        end_property: Property to identify the ending entity
        end_value: Value of the ending entity property
        min_depth: Minimum path length
        max_depth: Maximum path length
        relationship_types: Types of relationships to traverse
        limit: Maximum number of paths to return
        
    Returns:
        A validated CypherQuery object
    """
    # Create relationship type string
    rel_types = ""
    if relationship_types:
        rel_types = ":" + "|:".join(relationship_types)
    
    # Create path finding query
    query = f"""
    MATCH (start:{start_label} {{{start_property}: ${start_property}}})
    MATCH (end:{end_label} {{{end_property}: ${end_property}}})
    MATCH path = shortestPath((start)-[{rel_types}*{min_depth}..{max_depth}]-(end))
    RETURN path
    LIMIT {limit}
    """
    
    # For this type of query, we use the validate_query function directly
    from src.utils.neo4j_query_utils import validate_query
    return validate_query(
        query=query, 
        parameters={
            start_property: start_value,
            end_property: end_value
        },
        read_only=True
    )


def build_recommendation_query(
    entity_label: str,
    entity_id_property: str,
    entity_id_value: str,
    relation_type: str,
    recommendation_depth: int = 2,
    similarity_threshold: float = 0.1,
    max_recommendations: int = 10
) -> CypherQuery:
    """
    Build a recommendation query based on graph relationships.
    
    Args:
        entity_label: Label of the entity to get recommendations for
        entity_id_property: Property to identify the entity
        entity_id_value: Value of the identifying property
        relation_type: Type of relationship to base recommendations on
        recommendation_depth: Depth of recommendation traversal
        similarity_threshold: Minimum similarity score threshold
        max_recommendations: Maximum number of recommendations to return
        
    Returns:
        A validated CypherQuery object
    """
    # Collaborative filtering style recommendation query
    query = f"""
    MATCH (entity:{entity_label} {{{entity_id_property}: ${entity_id_property}}})
    MATCH (entity)-[:{relation_type}]->(common)<-[:{relation_type}]-(similar:{entity_label})
    WHERE entity <> similar 
    AND NOT (entity)-[:{relation_type}]->(similar)
    WITH entity, similar, count(common) as commonCount
    WITH entity, similar, commonCount,
         apoc.similarity.jaccard(
           [(entity)-[:{relation_type}]->(item) | id(item)], 
           [(similar)-[:{relation_type}]->(item) | id(item)]
         ) as similarity
    WHERE similarity >= {similarity_threshold}
    RETURN similar.{entity_id_property} as id, 
           similar.name as name,
           commonCount,
           similarity
    ORDER BY similarity DESC, commonCount DESC
    LIMIT {max_recommendations}
    """
    
    # For this type of query, we use the validate_query function directly
    from src.utils.neo4j_query_utils import validate_query
    return validate_query(
        query=query, 
        parameters={entity_id_property: entity_id_value},
        read_only=True
    )


def build_graph_statistics_query(
    entity_labels: Optional[List[str]] = None,
    relationship_types: Optional[List[str]] = None
) -> CypherQuery:
    """
    Build a query to get statistics about the graph.
    
    Args:
        entity_labels: Labels of entities to include in statistics
        relationship_types: Types of relationships to include in statistics
        
    Returns:
        A validated CypherQuery object
    """
    # Create filters for specific entity labels and relationship types
    node_filter = ""
    if entity_labels:
        labels_str = "|".join(f":{label}" for label in entity_labels)
        node_filter = f"WHERE any(label in labels(n) WHERE label in {entity_labels})"
    
    rel_filter = ""
    if relationship_types:
        rel_filter = f"WHERE type(r) in {relationship_types}"
    
    # Create statistics query
    query = f"""
    MATCH (n)
    {node_filter}
    CALL {{
        WITH n
        RETURN count(n) as nodeCount,
               labels(n) as labels
    }}
    
    WITH labels, nodeCount
    UNWIND labels as label
    WITH label, sum(nodeCount) as count
    
    RETURN 'Entity' as type, label as name, count
    
    UNION
    
    MATCH ()-[r]->()
    {rel_filter}
    RETURN 'Relationship' as type, type(r) as name, count(r) as count
    
    ORDER BY type, count DESC
    """
    
    # For this type of query, we use the validate_query function directly
    from src.utils.neo4j_query_utils import validate_query
    return validate_query(query=query, read_only=True) 