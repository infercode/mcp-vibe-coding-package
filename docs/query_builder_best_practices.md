# Neo4j Query Builder Best Practices

This document outlines best practices and patterns for using the QueryBuilder pattern for constructing Neo4j Cypher queries. The QueryBuilder provides a type-safe, validated approach to building complex queries.

## Table of Contents

1. [Basic Concepts](#basic-concepts)
2. [Simple Queries](#simple-queries)
3. [Path-Based Queries](#path-based-queries)
4. [Complex Filtering](#complex-filtering)
5. [Working with Results](#working-with-results)
6. [Common Patterns](#common-patterns)
7. [Utility Functions](#utility-functions)
8. [Performance Considerations](#performance-considerations)
9. [Security Considerations](#security-considerations)
10. [Troubleshooting](#troubleshooting)

## Basic Concepts

The QueryBuilder pattern provides a structured approach to building Cypher queries with proper validation:

```python
from src.models.neo4j_queries import QueryBuilder, NodePattern
from src.utils.neo4j_query_utils import safe_execute_validated_query

# Create node pattern
person = NodePattern(
    variable="p",
    labels=["Person"],
    properties={}
)

# Create builder
builder = QueryBuilder(
    match_patterns=[person],
    where_clauses=["p.age > $min_age"],
    return_fields=["p.name", "p.age"],
    order_by=None,
    limit=10,
    skip=0,
    parameters={"min_age": 25}
)

# Convert to CypherQuery
query = builder.to_cypher_query()

# Execute query
results = safe_execute_validated_query(manager, query)
```

### Key Components

- **NodePattern**: Represents a node in a Cypher query with variable name, labels, and properties
- **RelationshipPattern**: Represents a relationship with variable, type, properties, and direction
- **PathPattern**: Represents a path consisting of nodes and relationships
- **QueryBuilder**: Combines patterns, clauses, and parameters to build a complete query
- **CypherQuery**: The final validated query ready for execution

## Simple Queries

For simple entity retrieval, use a single NodePattern:

```python
# Get person by name
person = NodePattern(
    variable="p",
    labels=["Person"],
    properties={"name": "$name"}
)

builder = QueryBuilder(
    match_patterns=[person],
    where_clauses=[],
    return_fields=["p"],
    order_by=None,
    limit=1,
    skip=0,
    parameters={"name": "John Doe"}
)
```

## Path-Based Queries

For queries involving relationships between entities, use PathPattern:

```python
# Create node patterns
user = NodePattern(variable="u", labels=["User"], properties={})
post = NodePattern(variable="p", labels=["Post"], properties={})

# Create relationship pattern
created = RelationshipPattern(
    variable="r", 
    type="CREATED", 
    properties={},
    direction="OUTGOING"
)

# Create path pattern
path = PathPattern(
    nodes=[user, post],
    relationships=[created]
)

# Build query
builder = QueryBuilder(
    match_patterns=[path],
    where_clauses=["u.username = $username"],
    return_fields=["p.title", "p.content"],
    order_by=None,
    limit=10,
    skip=0,
    parameters={"username": "johndoe"}
)
```

### Important: Working with Multiple Paths

When working with multiple paths, ensure they form a connected graph:

```python
# CORRECT: Connecting paths through shared variables
user = NodePattern(variable="u", labels=["User"], properties={})
post = NodePattern(variable="p", labels=["Post"], properties={})
comment = NodePattern(variable="c", labels=["Comment"], properties={})

# First path: User created Post
path1 = PathPattern(
    nodes=[user, post],
    relationships=[RelationshipPattern(
        variable="r1", type="CREATED", properties={}, direction="OUTGOING"
    )]
)

# Second path: User commented on Post (connects to the same user and post variables)
path2 = PathPattern(
    nodes=[user, post],
    relationships=[RelationshipPattern(
        variable="r2", type="COMMENTED", properties={}, direction="OUTGOING"
    )]
)

builder = QueryBuilder(
    match_patterns=[path1, path2],
    # Other parameters...
)
```

If paths are not related, use separate QueryBuilder instances:

```python
# CORRECT: Separate unrelated paths into different queries
builder1 = QueryBuilder(match_patterns=[path1], ...)
query1 = builder1.to_cypher_query()

builder2 = QueryBuilder(match_patterns=[path2], ...)
query2 = builder2.to_cypher_query()
```

## Complex Filtering

For complex filtering, combine multiple WHERE clauses:

```python
builder = QueryBuilder(
    match_patterns=[person],
    where_clauses=[
        "p.age >= $min_age",
        "p.age <= $max_age",
        "p.location IN $locations",
        "p.active = $is_active"
    ],
    return_fields=["p.name", "p.age", "p.location"],
    parameters={
        "min_age": 25,
        "max_age": 40,
        "locations": ["New York", "San Francisco"],
        "is_active": True
    }
)
```

For date-based filtering:

```python
builder = QueryBuilder(
    match_patterns=[event],
    where_clauses=[
        "e.date >= $start_date",
        "e.date <= $end_date",
        "e.type IN $event_types"
    ],
    parameters={
        "start_date": "2023-01-01",
        "end_date": "2023-12-31",
        "event_types": ["meeting", "call"]
    }
)
```

## Working with Results

Process the results of a query using the QueryResult model:

```python
from src.models.neo4j_queries import QueryResult

query = builder.to_cypher_query()
records, summary = manager.execute_validated_query(query)

# Convert to QueryResult
result = QueryResult.from_neo4j_result(records, summary)

# Access records
for record in result.records:
    print(record)
```

## Common Patterns

### Entity Creation

For entity creation, use direct Cypher queries with validation:

```python
query = """
CREATE (p:Person {name: $name, age: $age, created_at: datetime()})
RETURN p
"""

validated_query = validate_query(
    query=query,
    parameters={"name": "John", "age": 30},
    read_only=False
)

result = safe_execute_validated_query(manager, validated_query)
```

### Entity Update

For updating entities:

```python
query = """
MATCH (p:Person {id: $id})
SET p.name = $name, p.age = $age, p.updated_at = datetime()
RETURN p
"""

validated_query = validate_query(
    query=query,
    parameters={"id": "123", "name": "John Smith", "age": 31},
    read_only=False
)
```

### Relationship Creation

For creating relationships:

```python
query = """
MATCH (a:Person {id: $person_id})
MATCH (b:Company {id: $company_id})
CREATE (a)-[r:WORKS_AT {since: $since}]->(b)
RETURN r
"""

validated_query = validate_query(
    query=query,
    parameters={"person_id": "123", "company_id": "456", "since": "2020"},
    read_only=False
)
```

## Utility Functions

The `query_builder_utils.py` module provides helper functions for common query patterns:

```python
from src.utils.query_builder_utils import build_entity_search_query

# Search for entities
query = build_entity_search_query(
    entity_label="Person",
    search_property="name",
    search_value="John",
    exact_match=False,
    limit=10
)

# Execute the query
results = safe_execute_validated_query(manager, query)
```

Available utility functions:

1. `build_entity_search_query`: Search entities by property values
2. `build_relationship_query`: Find relationships between entities
3. `build_property_aggregation_query`: Aggregate properties across entities
4. `build_path_finding_query`: Find paths between entities
5. `build_recommendation_query`: Generate recommendations
6. `build_graph_statistics_query`: Get statistics about the graph

## Performance Considerations

1. **Limit Results**: Always set appropriate limits to avoid large result sets
2. **Use Indexes**: Ensure that properties used in WHERE clauses are indexed
3. **Optimize Patterns**: Keep path patterns as simple as possible
4. **Parameterize Queries**: Always use parameters instead of string concatenation
5. **Batch Operations**: For bulk operations, use UNWIND for better performance

## Security Considerations

1. **Read-Only Queries**: Use `read_only=True` for queries that should not modify the database
2. **Parameter Validation**: Always sanitize parameters with `sanitize_query_parameters`
3. **Restricted Operations**: Avoid allowing user-provided query strings

## Troubleshooting

### Common Issues

1. **Invalid Node Patterns**: Ensure node variable names are valid identifiers
2. **Path Validation Errors**: Make sure the number of relationships is one less than the number of nodes
3. **Query Validation Errors**: Check for disallowed operations in read-only mode
4. **Parameter Type Errors**: Ensure parameter types match expected types
5. **Missing Parameters**: Verify all parameters referenced in the query are provided

### Error Handling

Always handle validation errors properly:

```python
try:
    query = builder.to_cypher_query()
    results = safe_execute_validated_query(manager, query)
except ValueError as e:
    logger.error(f"Query validation error: {e}")
    # Handle validation error
except Exception as e:
    logger.error(f"Query execution error: {e}")
    # Handle other errors
```

## Example Use Cases

### Social Network Queries

```python
# Find friends of friends
user = NodePattern(variable="u", labels=["User"], properties={"id": "$user_id"})
friend = NodePattern(variable="f", labels=["User"], properties={})
friend_of_friend = NodePattern(variable="fof", labels=["User"], properties={})

# User -> Friend relationship
friend_rel = RelationshipPattern(
    variable="r1", type="FRIEND", properties={}, direction="OUTGOING"
)

# Friend -> Friend of Friend relationship
fof_rel = RelationshipPattern(
    variable="r2", type="FRIEND", properties={}, direction="OUTGOING"
)

# Create path: User -> Friend -> Friend of Friend
path1 = PathPattern(nodes=[user, friend], relationships=[friend_rel])
path2 = PathPattern(nodes=[friend, friend_of_friend], relationships=[fof_rel])

builder = QueryBuilder(
    match_patterns=[path1, path2],
    where_clauses=["u.id <> fof.id", "NOT (u)-[:FRIEND]->(fof)"],
    return_fields=["fof.id", "fof.name", "count(f) as mutual_friends"],
    parameters={"user_id": "user-123"}
)
```

### E-commerce Recommendations

```python
# Product recommendations based on purchase history
query = """
MATCH (u:User {id: $user_id})-[:PURCHASED]->(p1:Product)-[:CATEGORY]->(c:Category)<-[:CATEGORY]-(p2:Product)
WHERE NOT (u)-[:PURCHASED]->(p2)
WITH p2, count(c) as category_overlap
ORDER BY category_overlap DESC
RETURN p2.id, p2.name, p2.price, category_overlap
LIMIT 10
"""

validated_query = validate_query(
    query=query,
    parameters={"user_id": "user-123"},
    read_only=True
)
``` 