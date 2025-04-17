# Lesson Memory API Guide

## Overview

This document provides an overview of the Lesson Memory API, which enables AI agents to interact with the memory system in a human-like way. The system is designed to capture, retrieve, and evolve experiential knowledge through lessons.

## Access Patterns

The Lesson Memory system can be accessed through three complementary patterns:

1. **Direct Property Access**: Full unrestricted access to all LessonMemoryManager methods
2. **Operation Categories**: Simplified task-oriented interface with standard operations
3. **Context Management**: Transaction-like operations for complex workflows

Each pattern is optimized for different usage scenarios, from low-level direct control to high-level contextual operations.

## Operation Categories API

The Operation Categories API provides a simplified interface through a single dispatcher method with standardized operations.

### Basic Usage

```python
# Using the GraphMemoryManager's lesson_operation method
result = graph_manager.lesson_operation(
    "create",
    name="ErrorHandlingPatterns", 
    lesson_type="BestPractice",
    container_name="PythonLessons"
)
```

### Available Operations

| Operation | Description | Required Parameters | Optional Parameters |
|-----------|-------------|---------------------|---------------------|
| create | Create new lesson entities | name, lesson_type | container_name, observations, metadata |
| observe | Add structured observations | entity_name, what_was_learned | container_name, why_it_matters, confidence, metadata |
| relate | Create relationships between lessons | source_name, target_name, relationship_type | container_name, properties |
| search | Find relevant lessons | query | container_name, limit, search_type |
| track | Record where lessons were applied | entity_name, application_context | container_name, success_score, notes |
| consolidate | Combine related lessons | source_names, target_name | container_name, reason |
| evolve | Track lesson knowledge evolution | source_name, target_name | container_name, reason |
| update | Modify existing lessons | entity_name, updates | container_name |

### Operation Examples

#### Creating a Lesson

```python
result = graph_manager.lesson_operation(
    "create",
    name="ErrorHandlingPatterns",
    lesson_type="BestPractice",
    container_name="PythonLessons"
)
```

#### Adding Observations

```python
result = graph_manager.lesson_operation(
    "observe",
    entity_name="ErrorHandlingPatterns",
    what_was_learned="Always catch specific exceptions rather than using a generic except block",
    why_it_matters="This makes debugging easier and prevents masking other issues",
    confidence=0.9,
    container_name="PythonLessons"
)
```

#### Searching for Lessons

```python
result = graph_manager.lesson_operation(
    "search",
    query="best practices for error handling in Python",
    limit=5
)
```

#### Creating Relationships

```python
result = graph_manager.lesson_operation(
    "relate",
    source_name="ErrorHandlingPatterns",
    target_name="ExceptionDesignPatterns",
    relationship_type="BUILDS_ON",
    properties={"strength": 0.8}
)
```

## Context Management API

The Context Management API provides a context manager for transaction-like operations with preserved container and project context.

### Basic Usage

```python
# Using the context manager for a series of operations
with graph_manager.lesson_context("ProjectName", "PythonLessons") as context:
    # Create a lesson in this context
    create_result = context.create("ErrorHandlingPatterns", "BestPractice")
    
    # Add observations to the same lesson
    observe_result = context.observe(
        "ErrorHandlingPatterns", 
        what_was_learned="Specific best practice observation",
        why_it_matters="Importance explanation"
    )
```

### Available Context Methods

| Method | Description | Required Parameters | Optional Parameters |
|--------|-------------|---------------------|---------------------|
| create | Create a lesson in the current context | name, lesson_type | observations, metadata |
| observe | Add observations to a lesson | entity_name, what_was_learned | why_it_matters, confidence, metadata |
| relate | Create a relationship between lessons | source_name, target_name, relationship_type | properties |
| search | Search for lessons within the context | query | limit, search_type |
| track | Record lesson application | entity_name, application_context | success_score, notes |
| update | Update a lesson within the context | entity_name, updates | |

### Context Management Examples

#### Multiple Operations in a Single Context

```python
with graph_manager.lesson_context("AI Project", "DebuggingLessons") as context:
    # Create a new lesson
    context.create("ModelDebugging", "Technique")
    
    # Add multiple observations
    context.observe(
        "ModelDebugging",
        what_was_learned="Visualize attention patterns to understand model focus",
        why_it_matters="Helps identify what parts of input the model emphasizes"
    )
    
    context.observe(
        "ModelDebugging",
        what_was_learned="Log intermediate activations during problematic inputs",
        why_it_matters="Helps pinpoint where the model's reasoning diverges"
    )
    
    # Create a related lesson
    context.create("ActivationAnalysis", "Technique")
    
    # Establish a relationship
    context.relate(
        "ModelDebugging",
        "ActivationAnalysis",
        "RELATED_TO"
    )
```

#### Temporary Project Context

```python
# Original project context is preserved after the block executes
current_project = graph_manager.default_project_name

with graph_manager.lesson_context("TemporaryProject", "TempLessons"):
    # Operations in temporary project
    context.create("QuickNote", "Observation")

# After the block, project context is restored to the original
assert graph_manager.default_project_name == current_project
```

## Direct Access API

For complete access to all LessonMemoryManager functionality, you can use the direct property access pattern.

```python
# Direct access to the underlying LessonMemoryManager
result = graph_manager.lesson_memory.create_lesson_entity(
    "PythonLessons",
    "ErrorHandlingPatterns",
    "BestPractice",
    [{"what_was_learned": "Initial observation"}],
    {"importance": "high"}
)
```

Refer to the LessonMemoryManager API documentation for complete details on all available methods.

## Response Format

All operations return a JSON string with a standardized format:

```json
{
  "status": "success",
  "entity": {
    "name": "ErrorHandlingPatterns",
    "type": "BestPractice",
    "id": "lesson-123"
  }
}
```

Error responses follow this format:

```json
{
  "status": "error",
  "error": "Detailed error message",
  "code": "error_type_code"
}
```

## Common Error Codes

| Error Code | Description |
|------------|-------------|
| validation_error | Invalid parameters provided |
| not_found | Entity or container not found |
| duplicate | Entity already exists |
| operation_error | General operation failure |
| invalid_operation | Unknown operation type |

## Performance Considerations

- The layered interfaces (Operation Categories and Context Management) add minimal overhead to direct LessonMemoryManager calls.
- For bulk operations, the Context Management API is more efficient as it maintains context across multiple calls.
- For performance-critical code, direct property access provides the lowest overhead but requires more careful parameter management.

## Backward Compatibility

The new access patterns are designed to work alongside existing code that uses direct LessonMemoryManager methods. You can freely mix and match access patterns based on your needs.

## Complete Workflow Example

```python
# Example workflow integrating different access patterns

# 1. Check if container exists using direct access
if not graph_manager.lesson_memory.container_exists("PythonLessons"):
    graph_manager.lesson_memory.create_lesson_container("PythonLessons", "Lessons about Python")

# 2. Create a new lesson using operation categories
graph_manager.lesson_operation(
    "create",
    name="AsyncPatterns",
    lesson_type="BestPractice",
    container_name="PythonLessons"
)

# 3. Add multiple observations and relationships using context manager
with graph_manager.lesson_context("CodeProject", "PythonLessons") as ctx:
    # Add observations
    ctx.observe(
        "AsyncPatterns",
        what_was_learned="Use asyncio.gather for concurrent tasks",
        why_it_matters="Improves performance for I/O-bound operations"
    )
    
    # Create related lesson
    ctx.create("AsyncAntiPatterns", "BestPractice")
    
    # Relate the lessons
    ctx.relate(
        "AsyncPatterns",
        "AsyncAntiPatterns",
        "CONTRASTS_WITH"
    )

# 4. Search for relevant lessons using operation categories
search_result = graph_manager.lesson_operation(
    "search",
    query="async programming patterns",
    limit=10
)
``` 