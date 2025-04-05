# Neo4j QueryBuilder Usage Guide

This guide provides practical examples and recommendations for using the Neo4j QueryBuilder pattern effectively in your application.

## Introduction

The QueryBuilder pattern offers a structured, type-safe approach to building Cypher queries for Neo4j. By using this pattern, you can:

- Ensure queries are properly validated
- Prevent injection attacks
- Maintain type safety for parameters
- Enforce read-only restrictions where appropriate

## Basic Usage

### Simple Entity Queries

```python
from src.models.neo4j_queries import QueryBuilder, NodePattern, QueryOrder
from src.utils.neo4j_query_utils import safe_execute_validated_query

# Create a node pattern
person = NodePattern(
    variable="p",
    labels=["Person"],
    properties={}
)

# Build a query
builder = QueryBuilder(
    match_patterns=[person],
    where_clauses=["p.age > $min_age"],
    return_fields=["p.name", "p.age", "p.email"],
    order_by=[QueryOrder(field="p.name", direction="ASC")],
    limit=10,
    skip=0,
    parameters={"min_age": 25}
)

# Convert to a validated query
query = builder.to_cypher_query()

# Execute the query
results = safe_execute_validated_query(manager, query)
```

### Working with Relationships

For queries involving relationships between nodes:

```python
from src.models.neo4j_queries import RelationshipPattern, PathPattern

# Create node patterns
user = NodePattern(variable="u", labels=["User"], properties={"id": "$user_id"})
post = NodePattern(variable="p", labels=["Post"], properties={})

# Create relationship pattern
authored_rel = RelationshipPattern(
    variable="r",
    type="AUTHORED",  # Note: Avoid strings like "CREATE" that can trigger validation errors
    properties={},
    direction="OUTGOING"
)

# Create a path pattern
path = PathPattern(
    nodes=[user, post],
    relationships=[authored_rel]
)

# Build the query
builder = QueryBuilder(
    match_patterns=[path],
    where_clauses=["p.date_published > $start_date"],  # Use date_published instead of created_at to avoid "CREATE"
    return_fields=["p.id", "p.title", "p.content"],
    order_by=[QueryOrder(field="p.date_published", direction="DESC")],
    limit=5,
    skip=0,
    parameters={
        "user_id": "user-123",
        "start_date": "2023-01-01"
    }
)
```

## Utility Functions

The `query_builder_utils.py` module provides helper functions for common query patterns:

### Entity Search

```python
from src.utils.query_builder_utils import build_entity_search_query

query = build_entity_search_query(
    entity_label="Person",
    search_property="name",
    search_value="John",
    exact_match=False,
    additional_properties={"active": True},
    additional_filters=["e.created_at > $created_after"],
    return_fields=["e.id", "e.name", "e.email"],
    order_by_property="name",
    order_direction="ASC",
    limit=10
)

# Add any additional parameters
query.parameters.parameters["created_after"] = "2022-01-01"
```

### Relationship Queries

```python
from src.utils.query_builder_utils import build_relationship_query

query = build_relationship_query(
    from_label="User",
    from_id_property="id",
    from_id_value="user-123",
    relationship_type="FOLLOWS",
    to_label="User",
    return_paths=True,
    relationship_direction="OUTGOING",
    additional_filters=["r.since > $since_date"],
    limit=20
)

# Add any additional parameters
query.parameters.parameters["since_date"] = "2023-01-01"
```

### Aggregation Queries

```python
from src.utils.query_builder_utils import build_property_aggregation_query

query = build_property_aggregation_query(
    entity_label="Product",
    group_by_property="category",
    aggregate_property="price",
    aggregation_type="avg",
    filters=["e.active = true"],
    having_clause="agg_value > 50",
    limit=10
)
```

### Path Finding

```python
from src.utils.query_builder_utils import build_path_finding_query

query = build_path_finding_query(
    start_label="Person",
    start_property="name",
    start_value="Alice",
    end_label="Person",
    end_property="name",
    end_value="Bob",
    max_depth=3,
    relationship_types=["FRIEND", "COLLEAGUE"]
)
```

## Important Considerations

### String Content Validation

The validation system will reject queries containing destructive operations when in read-only mode (the default). These include:

```python
destructive_operations = {
    "CREATE", "DELETE", "REMOVE", "SET", "MERGE", "DROP",
    "CALL DB.INDEX", "CALL APOC", "WITH [0] AS FOO", "LOAD CSV",
    "PERIODIC COMMIT", "FOREACH"
}
```

The validator has been improved to correctly identify these as actual operations, not just strings:

1. **✅ Now Supported**: 
   - Past tense forms in relationship types (e.g., "CREATED", "DELETED")
   - Properties containing these words (e.g., "created_at", "deleted_by")
   - Variable names with these substrings

2. **❌ Still Rejected**: 
   - Actual Cypher operations (e.g., `CREATE`, `DELETE`, `SET`)
   - Destructive procedure calls (e.g., `CALL APOC.CREATE`)

Examples of what's allowed:

```python
# NOW VALID: These won't trigger validation errors
rel = RelationshipPattern(variable="r", type="CREATED", direction="OUTGOING")
prop_name = "created_at"
builder.return_fields = ["p.created_at"]
```

### Query Format Requirements

All read-only queries must:

1. **Start with a valid read operation**:
   - Must begin with keywords like MATCH, RETURN, WITH, etc.
   - Comments at the beginning of the query will cause validation errors

```python
# AVOID: This will fail validation due to initial comments
query = """
// This is a comment
MATCH (n:Node) RETURN n
"""

# USE INSTEAD:
query = """
MATCH (n:Node) RETURN n
// This comment is fine here
"""
```

2. **Contain no destructive operations**:
   - Even in non-executed code paths or comments
   - The validator scans the entire query text

### Multiple Paths

When working with multiple paths, be cautious of how they connect:

1. **Connected Paths**: Paths that share nodes should use the same node instances
2. **Unrelated Paths**: Consider using separate QueryBuilder instances or custom queries

```python
# CORRECT: Use separate builders for unrelated paths
builder1 = QueryBuilder(match_patterns=[path1], ...)
query1 = builder1.to_cypher_query()

builder2 = QueryBuilder(match_patterns=[path2], ...)
query2 = builder2.to_cypher_query()
```

### Write Operations

For operations that require writing to the database:

```python
from src.utils.neo4j_query_utils import validate_query

# Create a query string
create_query = """
MATCH (u:User {id: $user_id})
CREATE (p:Post {
    id: randomUUID(),
    title: $title, 
    content: $content,
    date_published: datetime()  /* Use date_published instead of created_at */
})
CREATE (u)-[r:AUTHORED {at: datetime()}]->(p)
RETURN p
"""

# Validate for write operations
validated_query = validate_query(
    query=create_query,
    parameters={
        "user_id": "user-123",
        "title": "My Post",
        "content": "Post content here"
    },
    read_only=False  # Important: set to False for write operations
)

# Execute the validated query
result = safe_execute_validated_query(manager, validated_query)
```

## Error Handling

Always handle validation errors appropriately:

```python
try:
    query = builder.to_cypher_query()
    results = safe_execute_validated_query(manager, query)
    # Process results
except ValueError as e:
    if "Destructive operation" in str(e):
        logger.error(f"Query contains a destructive operation: {str(e)}")
        # Check for problematic strings in your query
    elif "Read-only query must start with" in str(e):
        logger.error(f"Query format error: {str(e)}")
        # Ensure query starts with a valid read operation
    else:
        logger.error(f"Query validation error: {str(e)}")
    # Handle validation error
except Exception as e:
    logger.error(f"Query execution error: {str(e)}")
    # Handle other errors
```

## Debugging Query Validation Issues

If you encounter validation errors:

1. **Check for reserved words**:
   - Examine all property names, relationship types, and variable names
   - Replace words like "CREATE", "DELETE", "SET", etc. with alternatives

2. **Verify query structure**:
   - Ensure queries start with valid operations (MATCH, RETURN, etc.)
   - Remove comments from the beginning of the query

3. **Use validation tracing**:
   - To see exactly what's causing the validation error, you can use:
   
   ```python
   import logging
   logging.getLogger('src.models.neo4j_queries').setLevel(logging.DEBUG)
   ```

4. **Consider using direct Cypher**:
   - For complex cases, bypass QueryBuilder and use validate_query directly
   - Set read_only=False when appropriate

## Performance Tips

1. **Limit Results**: Always set appropriate limits for queries
2. **Skip for Pagination**: Use the skip parameter for pagination
3. **Selective Fields**: Specify only needed fields in return_fields
4. **Index Usage**: Ensure properties used in WHERE clauses have indexes
5. **Query Profiling**: Use Neo4j's PROFILE command to analyze queries

## Security Best Practices

1. **Parameter Validation**: Always validate user input before using in parameters
2. **Read-Only Mode**: Use read_only=True for queries that should not modify the database
3. **Authorized Access**: Ensure users only have access to data they're authorized to see
4. **Error Handling**: Don't expose raw error messages to users
5. **Logging**: Log query validation failures for security monitoring

## Complete Example

```python
def find_recommended_posts(manager, user_id, limit=5):
    """Find posts recommended for a user based on interests."""
    try:
        # Create a custom query for recommendations
        query = """
        MATCH (u:User {id: $user_id})-[:INTERESTED_IN]->(t:Topic)<-[:ABOUT]-(p:Post)
        WHERE NOT (u)-[:VIEWED]->(p)
        WITH p, count(t) as shared_topics
        ORDER BY shared_topics DESC, p.date_published DESC
        RETURN p.id as id, p.title as title, p.summary as summary, 
               p.date_published as date_published, shared_topics
        LIMIT $limit
        """
        
        # Validate the query
        validated_query = validate_query(
            query=query,
            parameters={"user_id": user_id, "limit": limit},
            read_only=True
        )
        
        # Execute the query
        results = safe_execute_validated_query(manager, validated_query)
        
        # Process and return the results
        recommendations = []
        for record in results:
            recommendations.append({
                "id": record.get("id"),
                "title": record.get("title"),
                "summary": record.get("summary"),
                "date_published": record.get("date_published"),
                "relevance_score": record.get("shared_topics")
            })
        
        return recommendations
        
    except ValueError as e:
        logger.error(f"Query validation error: {str(e)}")
        raise ValueError(f"Invalid recommendation query: {str(e)}")
    except Exception as e:
        logger.error(f"Recommendation query error: {str(e)}")
        raise Exception(f"Error fetching recommendations: {str(e)}") 