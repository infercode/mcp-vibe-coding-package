# Neo4j Query Validation Integration Guide

This guide explains how to integrate the Neo4j query validation utilities into existing code. The utilities provide type safety, validation, and security for Neo4j Cypher queries.

## Overview

The new Neo4j query validation system consists of:

1. **Pydantic Models** (`src/models/neo4j_queries.py`): Define schemas for Neo4j queries, parameters, and results
2. **Utility Functions** (`src/utils/neo4j_query_utils.py`): Provide helper functions for common query operations
3. **Example Usage** (`src/examples/neo4j_query_validation_example.py`): Demonstrate how to use the utilities

## Integration Progress

| Component | Status |
|-----------|--------|
| BaseManager query validation methods | ✅ Completed |
| EntityManager validated methods | ✅ Completed |
| RelationManager validated methods | ✅ Completed |
| SearchManager validated methods | ✅ Completed |
| ObservationManager validated methods | ✅ Completed |
| ProjectMemoryManager validated methods | ✅ Completed |
| LessonMemoryManager validated methods | ✅ Completed |

## Benefits

- **Type Safety**: Runtime validation of query structures and parameters
- **Security**: Prevention of unsafe operations in read-only contexts
- **Standardization**: Consistent patterns for building and executing queries
- **Error Handling**: Clear error messages for invalid queries or parameters

## Integration Steps

### 1. Basic Parameter Validation ✅

Replace direct parameter usage with validated parameters:

```python
# Before
def get_entities(self, entity_type: str, properties: Dict[str, Any]):
    query = f"MATCH (n:{entity_type}) WHERE n.name = $name RETURN n"
    return self.execute_read_query(query, properties)

# After
from src.utils.neo4j_query_utils import sanitize_query_parameters

def get_entities(self, entity_type: str, properties: Dict[str, Any]):
    sanitized_params = sanitize_query_parameters(properties)
    query = f"MATCH (n:{entity_type}) WHERE n.name = $name RETURN n"
    return self.execute_read_query(query, sanitized_params)
```

### 2. Query Validation ✅

Add validation before executing queries:

```python
# Before
def get_entity(self, entity_type: str, entity_id: str):
    query = f"MATCH (n:{entity_type}) WHERE n.id = $id RETURN n"
    return self.execute_read_query(query, {"id": entity_id})

# After
from src.utils.neo4j_query_utils import validate_query, safe_execute_validated_query

def get_entity(self, entity_type: str, entity_id: str):
    query = f"MATCH (n:{entity_type}) WHERE n.id = $id RETURN n"
    validated_query = validate_query(
        query=query,
        parameters={"id": entity_id},
        read_only=True
    )
    return safe_execute_validated_query(self, validated_query)
```

### 3. Using Pre-Built Query Functions

Use utility functions for common query patterns:

```python
# Before
def create_entity(self, entity_type: str, properties: Dict[str, Any]):
    props_str = ", ".join([f"{k}: ${k}" for k in properties.keys()])
    query = f"CREATE (n:{entity_type} {{{props_str}}}) RETURN n"
    return self.execute_write_query(query, properties)

# After
from src.utils.neo4j_query_utils import create_node_query, safe_execute_validated_query

def create_entity(self, entity_type: str, properties: Dict[str, Any]):
    query_model = create_node_query(entity_type, properties)
    return safe_execute_validated_query(self, query_model)
```

### 4. Using the Query Builder for Complex Queries

For more complex queries, use the QueryBuilder:

```python
# Before
def find_related_entities(self, entity_type: str, properties: Dict[str, Any], 
                          related_type: str, relationship_type: str):
    query = f"""
    MATCH (a:{entity_type})-[r:{relationship_type}]->(b:{related_type})
    WHERE a.name = $name
    RETURN a, r, b
    """
    return self.execute_read_query(query, properties)

# After
from src.models.neo4j_queries import QueryBuilder, NodePattern, RelationshipPattern, PathPattern
from src.utils.neo4j_query_utils import sanitize_query_parameters, safe_execute_validated_query

def find_related_entities(self, entity_type: str, properties: Dict[str, Any], 
                          related_type: str, relationship_type: str):
    # Create node patterns
    from_node = NodePattern(variable="a", labels=[entity_type], properties={"name": "$name"})
    to_node = NodePattern(variable="b", labels=[related_type], properties={})
    
    # Create relationship pattern
    rel = RelationshipPattern(variable="r", type=relationship_type, properties={}, direction="OUTGOING")
    
    # Create path pattern
    path = PathPattern(nodes=[from_node, to_node], relationships=[rel])
    
    # Build the query
    builder = QueryBuilder(
        match_patterns=[path],
        where_clauses=[],
        return_fields=["a", "r", "b"],
        parameters=sanitize_query_parameters(properties)
    )
    
    # Convert to CypherQuery and execute
    query_model = builder.to_cypher_query()
    return safe_execute_validated_query(self, query_model)
```

## Integration with BaseManager ✅

The `BaseManager` class has been extended with validation methods:

```python
# New methods in BaseManager

def safe_execute_read_query(self, query: str, parameters: Optional[Dict[str, Any]] = None):
    """Validate and execute a read-only query."""
    # Sanitize parameters
    if parameters:
        sanitized_params = sanitize_query_parameters(parameters)
    else:
        sanitized_params = None
        
    # Validate query
    validated_query = validate_query(
        query=query,
        parameters=sanitized_params,
        read_only=True
    )
    
    # Execute the validated query
    records, _ = self._execute_validated_query(validated_query)
    return records

def safe_execute_write_query(self, query: str, parameters: Optional[Dict[str, Any]] = None):
    """Validate and execute a write query."""
    # Similar implementation to safe_execute_read_query but with read_only=False
```

## Integration with EntityManager ✅

The `EntityManager` class has been updated to use validated queries:

```python
# Example of updated EntityManager methods

def get_entity(self, entity_name: str) -> str:
    """Get an entity from the knowledge graph."""
    try:
        self.base_manager.ensure_initialized()
        
        query = """
        MATCH (e:Entity {name: $name})
        RETURN e
        """
        
        # Use safe_execute_read_query for validation
        records = self.base_manager.safe_execute_read_query(
            query,
            {"name": entity_name}
        )
        
        # Process results...
        
    except ValueError as e:
        # Handle validation errors
        error_msg = f"Query validation error: {str(e)}"
        self.logger.error(error_msg)
        return dict_to_json({"error": error_msg})
    except Exception as e:
        # Handle other errors
        error_msg = f"Error retrieving entity: {str(e)}"
        self.logger.error(error_msg)
        return dict_to_json({"error": error_msg})
```

## Integration with RelationManager ✅

The `RelationManager` class has been updated to use validated queries:

```python
# Example of updated RelationManager methods

def get_relations(self, entity_name: Optional[str] = None) -> str:
    """Get relationships from the knowledge graph."""
    try:
        self.base_manager.ensure_initialized()
        
        # Build query
        query = """
        MATCH (e1:Entity {name: $entity_name})-[r]->(e2:Entity)
        RETURN e1.name as from, TYPE(r) as type, e2.name as to, properties(r) as properties
        """
        
        try:
            # Use safe_execute_read_query for validation
            records = self.base_manager.safe_execute_read_query(
                query,
                {"entity_name": entity_name}
            )
            
            # Process results...
            
        except ValueError as e:
            # Handle validation errors
            error_msg = f"Query validation error: {str(e)}"
            self.logger.error(error_msg)
            return dict_to_json({"error": error_msg})
            
    except Exception as e:
        # Handle other errors
        error_msg = f"Error retrieving relations: {str(e)}"
        self.logger.error(error_msg)
        return dict_to_json({"error": error_msg})
```

## Integration with SearchManager ✅

The `SearchManager` class has been updated to use validated queries:

```python
# Example of updated SearchManager methods

def search_entities(self, search_term: str, limit: int = 10, 
                    entity_types: Optional[List[str]] = None,
                    semantic: bool = False) -> str:
    """Search for entities in the knowledge graph."""
    try:
        self.base_manager.ensure_initialized()
        
        # Prepare query
        query_parts = ["MATCH (e:Entity)"]
        params = {}
        
        if search_term:
            query_parts.append("WHERE e.name CONTAINS $search_term")
            params["search_term"] = search_term
        
        # Complete the query with ordering and limit
        query_parts.append("RETURN e")
        query_parts.append("ORDER BY e.name")
        query_parts.append(f"LIMIT {min(limit, 100)}")
        
        query = " ".join(query_parts)
        
        try:
            # Sanitize parameters
            sanitized_params = sanitize_query_parameters(params) if params else None
            
            # Execute read query with validation
            records = self.base_manager.safe_execute_read_query(
                query,
                sanitized_params
            )
            
            # Process results...
            
        except ValueError as e:
            # Handle validation errors
            error_msg = f"Query validation error: {str(e)}"
            self.logger.error(error_msg)
            return dict_to_json({"error": error_msg})
            
    except Exception as e:
        # Handle other errors
        error_msg = f"Error searching entities: {str(e)}"
        self.logger.error(error_msg)
        return dict_to_json({"error": error_msg})
```

## Integration with ObservationManager ✅

The `ObservationManager` class has been updated to use validated queries:

```python
# Example of updated ObservationManager methods

def get_entity_observations(self, entity_name: str) -> str:
    """Get all observations for an entity."""
    try:
        self.base_manager.ensure_initialized()
        
        # First, verify entity exists
        entity_query = """
            MATCH (e:Entity {name: $name})
            RETURN e
        """
        
        # Use safe_execute_read_query for validation
        entity_records = self.base_manager.safe_execute_read_query(
            entity_query,
            {"name": entity_name}
        )
        
        if not entity_records:
            error_msg = f"Entity '{entity_name}' not found"
            self.logger.error(error_msg)
            return dict_to_json({"error": error_msg})
        
        # Query to get observations
        query = """
                MATCH (e:Entity {name: $name})-[:HAS_OBSERVATION]->(o:Observation)

                RETURN o.id as id, o.content as content, o.type as type,
                       o.created as created, o.lastUpdated as lastUpdated
                """
        
        # Execute validated read query
        records = self.base_manager.safe_execute_read_query(
            query,
            {"name": entity_name}
        )
        
        # Process results...
        
    except ValueError as e:
        # Handle validation errors
        error_msg = f"Query validation error: {str(e)}"
        self.logger.error(error_msg)
        return dict_to_json({"error": error_msg})
    except Exception as e:
        # Handle other errors
        error_msg = f"Error retrieving observations: {str(e)}"
        self.logger.error(error_msg)
        return dict_to_json({"error": error_msg})

def add_observations(self, observations: List[Dict[str, Any]]) -> str:
    """Add observations to entities in the knowledge graph."""
    try:
        self.base_manager.ensure_initialized()
        
        results = []
        errors = []
        
        for obs in observations:
            entity_name = obs.get("entity", None)
            
            if not entity_name:
                errors.append({"error": "Missing entity name", "observation": obs})
                continue
                
            # Verify entity exists
            entity_query = """
                MATCH (e:Entity {name: $name})
                RETURN e
            """
            
            # Validate and execute query
            entity_records = self.base_manager.safe_execute_read_query(
                entity_query,
                {"name": entity_name}
            )
            
            if not entity_records:
                errors.append({"error": f"Entity '{entity_name}' not found", "observation": obs})
                continue
                
            # Add observation with validated query
            try:
                # Call internal method to add observation
                result = self._add_observation_to_entity(
                    entity_name, 
                    obs.get("content", ""),
                    obs.get("type", "observation")
                )
                
                # Process result...
                
            except ValueError as e:
                # Handle validation errors in the internal method
                errors.append({"error": str(e), "observation": obs})
                
        # Return combined results
        return dict_to_json({"added": results, "errors": errors})
            
    except Exception as e:
        # Handle other errors
        error_msg = f"Error adding observations: {str(e)}"
        self.logger.error(error_msg)
        return dict_to_json({"error": error_msg})
```

## Error Handling

The validation utilities provide clear error messages that can be used in error handling:

```python
from src.utils.neo4j_query_utils import validate_query
from src.models.responses import create_error_response, create_success_response

def search_entities(self, entity_type: str, search_params: Dict[str, Any]):
    try:
        query = f"MATCH (n:{entity_type}) WHERE n.name CONTAINS $search_term RETURN n"
        validated_query = validate_query(query, search_params, read_only=True)
        results = safe_execute_validated_query(self, validated_query)
        
        return create_success_response(
            message="Search completed successfully",
            data={"results": results}
        )
    except ValueError as e:
        return create_error_response(
            message=f"Invalid search parameters: {str(e)}",
            code="invalid_search_parameters"
        )
    except Exception as e:
        return create_error_response(
            message=f"Error executing search: {str(e)}",
            code="search_execution_error"
        )
```

## Security Considerations

- Always use `read_only=True` for queries that should not modify the database
- Validate all user-provided parameters with `sanitize_query_parameters`
- Use the query validation to prevent injection attacks and unauthorized operations

## Examples

See the following examples for detailed usage:
- `src/examples/neo4j_query_validation_example.py`: Basic validation examples
- `src/examples/neo4j_validation_integration_example.py`: Integration with managers
- `src/examples/entity_manager_validation_test.py`: EntityManager integration tests
- `src/examples/relation_manager_validation_test.py`: RelationManager integration tests
- `src/examples/search_manager_validation_test.py`: SearchManager integration tests

## Phased Integration Approach

1. ✅ Start by adding parameter validation to existing code
2. ✅ Add query validation to critical paths
3. ✅ Replace direct query execution with validated query execution
4. 📝 Refactor complex queries to use the QueryBuilder pattern

## Next Steps

1. ✅ Complete integration with all manager classes:
   - ✅ EntityManager (Completed)
   - ✅ RelationManager (Completed)
   - ✅ SearchManager (Completed)
   - ✅ ObservationManager (Completed)
2. 📝 Add integration tests for database operations with real Neo4j instance
3. 📝 Update documentation and examples as the system evolves 
4. 📝 Add examples of using the QueryBuilder for complex queries 