# Graph Memory System Integration Guide

## Overview

This guide provides instructions for migrating from the original monolithic `graph_manager.py` implementation to the refactored Graph Memory System. The refactored system offers improved maintainability, extensibility, and performance while maintaining backward compatibility with the original API.

## Table of Contents

- [Graph Memory System Integration Guide](#graph-memory-system-integration-guide)
  - [Overview](#overview)
  - [Table of Contents](#table-of-contents)
  - [Quick Start](#quick-start)
  - [Using the Facade API](#using-the-facade-api)
  - [Migrating to Direct Component Usage](#migrating-to-direct-component-usage)
  - [Specialized Memory Systems](#specialized-memory-systems)
    - [Lesson Memory System](#lesson-memory-system)
    - [Project Memory System](#project-memory-system)
  - [Common Use Cases](#common-use-cases)
    - [Semantic Search](#semantic-search)
    - [Knowledge Graph Visualization](#knowledge-graph-visualization)
    - [Batch Operations](#batch-operations)
  - [Performance Considerations](#performance-considerations)
  - [Error Handling](#error-handling)
  - [Best Practices](#best-practices)

## Quick Start

The fastest way to migrate is to replace imports of the original `GraphManager` with the new facade class:

```python
# Old import
from src.graph_manager import GraphManager

# New import
from src.graph_memory import GraphMemoryManager
```

The `GraphMemoryManager` facade implements the same API as the original `GraphManager`, allowing for a seamless transition:

```python
# Initialize memory system (same method signature)
memory_manager = GraphMemoryManager(logger)
memory_manager.initialize()

# Create entities (same method signature)
entities = memory_manager.create_entities([
    {"name": "entity1", "entityType": "Type1"},
    {"name": "entity2", "entityType": "Type2"}
])
```

## Using the Facade API

The `GraphMemoryManager` facade is the main entry point for compatibility with existing code. It delegates operations to the appropriate specialized components:

```python
# Initialize the manager
memory_manager = GraphMemoryManager(logger)
memory_manager.initialize()

# Entity operations (delegated to EntityManager)
entity = memory_manager.create_entity({
    "name": "example",
    "entityType": "Concept",
    "description": "An example entity"
})

# Relationship operations (delegated to RelationManager)
relation = memory_manager.create_relationship(
    "entity1",
    "entity2",
    "RELATED_TO"
)

# Observation operations (delegated to ObservationManager)
observation = memory_manager.add_observation(
    "entity1",
    "DESCRIPTION",
    "Detailed description content"
)

# Search operations (delegated to SearchManager)
results = memory_manager.search_entities(search_term="example")
```

## Migrating to Direct Component Usage

For new code or when refactoring existing code, consider using the specialized components directly for improved clarity and access to advanced features:

```python
from src.graph_memory.base_manager import BaseManager
from src.graph_memory.entity_manager import EntityManager
from src.graph_memory.relation_manager import RelationManager
from src.graph_memory.search_manager import SearchManager

# Initialize base manager
base_manager = BaseManager(logger)
base_manager.initialize()

# Create specialized managers
entity_manager = EntityManager(base_manager)
relation_manager = RelationManager(base_manager)
search_manager = SearchManager(base_manager)

# Use specialized methods
entity = entity_manager.create_entity({
    "name": "example",
    "entityType": "Concept",
    "description": "An example entity"
})

relation = relation_manager.create_relationship(
    "entity1",
    "entity2",
    "RELATED_TO",
    {"strength": 0.8}
)

results = search_manager.semantic_search(
    "concepts related to knowledge",
    threshold=0.7
)
```

## Specialized Memory Systems

### Lesson Memory System

For applications focused on lesson management, the dedicated Lesson Memory System provides specialized functionality:

```python
from src.lesson_memory import LessonMemoryManager

# Initialize lesson memory system
lesson_memory = LessonMemoryManager(logger)
lesson_memory.initialize()

# Create a lesson container
container = lesson_memory.create_lesson_container({
    "name": "Programming Concepts",
    "description": "Fundamental programming concepts"
})

# Create a lesson with structured observations
lesson = lesson_memory.create_lesson(
    {
        "name": "Variables",
        "description": "Understanding programming variables",
        "difficulty": "beginner"
    },
    "Programming Concepts",
    {
        "PREREQUISITES": "Basic computer knowledge",
        "APPLICATION_CONTEXT": "All programming languages",
        "EFFECTIVENESS": "High for beginners"
    }
)

# Track knowledge evolution
evolution = lesson_memory.track_knowledge_evolution("Variables")

# Find similar lessons for consolidation
similar_lessons = lesson_memory.find_similar_lessons("Variables")
```

### Project Memory System

For project-specific memory management, the dedicated Project Memory System offers specialized functionality:

```python
from src.project_memory import ProjectMemoryManager

# Initialize project memory system
project_memory = ProjectMemoryManager(logger)
project_memory.initialize()

# Create a project with domains and components
project = project_memory.create_project(
    {
        "name": "Web Application",
        "description": "Customer portal application",
        "status": "active"
    },
    [
        {"name": "Frontend", "description": "UI components"},
        {"name": "Backend", "description": "Server-side logic"}
    ],
    [
        {"name": "AuthService", "description": "Authentication service", "domain": "Backend"},
        {"name": "UserDashboard", "description": "User dashboard UI", "domain": "Frontend"}
    ]
)

# Record a design decision
decision = project_memory.record_decision(
    {
        "title": "Authentication Method",
        "description": "Will use JWT for authentication",
        "rationale": "Better for stateless architecture"
    },
    "Web Application",
    ["AuthService"]
)

# Get component dependencies
dependencies = project_memory.get_component_dependencies("AuthService")

# Transfer knowledge between similar components
transfer_results = project_memory.transfer_component_knowledge(
    "LegacyAuthService",
    "AuthService"
)
```

## Common Use Cases

### Semantic Search

```python
# Basic semantic search
results = memory_manager.semantic_search("concept related to programming")

# Advanced semantic search with direct component
from src.graph_memory.search_manager import SearchManager
from src.graph_memory.embedding_adapter import EmbeddingAdapter

search_manager = SearchManager(base_manager, EmbeddingAdapter(base_manager))
results = search_manager.combined_search(
    text_query="programming",
    semantic_query="fundamental concepts in software development",
    entity_type="Lesson",
    text_weight=0.3,
    semantic_weight=0.7
)
```

### Knowledge Graph Visualization

```python
# Get lesson knowledge graph for visualization
lesson_relations = lesson_memory.get_lesson_knowledge_graph(
    "Programming Concepts",
    max_depth=2
)

# Convert to visualization format (example for D3.js)
visualization_data = {
    "nodes": [{"id": entity["name"], "group": entity["entityType"]} for entity in lesson_relations["entities"]],
    "links": [{"source": rel["source"], "target": rel["target"], "value": 1} for rel in lesson_relations["relationships"]]
}
```

### Batch Operations

```python
# Batch entity creation for performance
entities = [
    {"name": f"entity_{i}", "entityType": "Type", "description": f"Description {i}"}
    for i in range(100)
]
created_entities = entity_manager.create_entities(entities)

# Batch embedding generation
from src.graph_memory.embedding_adapter import EmbeddingAdapter

embedding_adapter = EmbeddingAdapter(base_manager)
texts = ["Text 1", "Text 2", "Text 3", "Text 4", "Text 5"]
embeddings = embedding_adapter.batch_generate_embeddings(texts)
```

## Performance Considerations

The refactored system includes several performance optimizations:

1. **Batch Operations**: Use batch methods like `create_entities()` instead of multiple calls to `create_entity()`
2. **Query Optimization**: Specialized components use optimized Cypher queries
3. **Connection Pooling**: The BaseManager handles efficient connection pooling
4. **Embedding Caching**: The EmbeddingAdapter caches embeddings to reduce API calls
5. **Pagination**: Use paginated search methods for large result sets

Example of paginated search:

```python
page = 1
page_size = 20
has_more = True

while has_more:
    results, has_more = search_manager.paginated_search(
        search_term="example",
        page=page,
        page_size=page_size
    )
    
    # Process results
    for result in results:
        process_entity(result)
    
    page += 1
```

## Error Handling

The refactored system includes comprehensive error handling:

```python
try:
    entity = entity_manager.create_entity({
        "name": "example",
        "entityType": "Concept"
    })
    
    if entity is None:
        # Handle entity creation failure
        logger.warning("Entity creation failed")
except ValueError as e:
    # Handle validation errors
    logger.error(f"Validation error: {str(e)}")
except Exception as e:
    # Handle unexpected errors
    logger.error(f"Unexpected error: {str(e)}")
```

## Best Practices

1. **Use Specialized Components**: Prefer direct use of specialized components over the facade when possible
2. **Batch Operations**: Use batch operations for better performance
3. **Connection Management**: Initialize the base manager once and share across components
4. **Error Handling**: Implement proper error handling for each operation
5. **Container Organization**: Use containers to organize entities in the lesson and project memory systems
6. **Regular Consolidation**: Periodically consolidate similar lessons to optimize memory
7. **Type Consistency**: Maintain consistent entity types and relationship types

Example of proper component initialization and sharing:

```python
# Initialize base manager once
base_manager = BaseManager(logger)
base_manager.initialize()

# Share base manager across components
entity_manager = EntityManager(base_manager)
relation_manager = RelationManager(base_manager)
observation_manager = ObservationManager(base_manager)
search_manager = SearchManager(base_manager)

# Initialize specialized memory systems with shared base manager
lesson_memory = LessonMemoryManager(base_manager=base_manager)
project_memory = ProjectMemoryManager(base_manager=base_manager)

# Now use the components as needed
# ...

# Close connection when done
base_manager.close()
``` 