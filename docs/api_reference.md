# Graph Memory System API Reference

## Table of Contents

- [Introduction](#introduction)
- [Core Graph Memory API](#core-graph-memory-api)
  - [BaseManager](#basemanager)
  - [EntityManager](#entitymanager)
  - [RelationManager](#relationmanager)
  - [ObservationManager](#observationmanager)
  - [SearchManager](#searchmanager)
  - [EmbeddingAdapter](#embeddingadapter)
- [Lesson Memory API](#lesson-memory-api)
  - [LessonContainer](#lessoncontainer)
  - [LessonEntity](#lessonentity)
  - [LessonRelation](#lessonrelation)
  - [LessonObservation](#lessonobservation)
  - [EvolutionTracker](#evolutiontracker)
  - [Consolidation](#consolidation)
- [Project Memory API](#project-memory-api)
  - [ProjectContainer](#projectcontainer)
  - [DomainManager](#domainmanager)
  - [ComponentManager](#componentmanager)
  - [FeatureManager](#featuremanager)
  - [DecisionManager](#decisionmanager)
  - [KnowledgeTransfer](#knowledgetransfer)
- [Facade API](#facade-api)
  - [GraphMemoryManager](#graphmemorymanager)
  - [LessonMemoryManager](#lessonmemorymanager)
  - [ProjectMemoryManager](#projectmemorymanager)

## Introduction

The Graph Memory System provides a comprehensive API for interacting with the MCP Neo4j-based memory architecture. This document details all public interfaces for the refactored components, organized by functional domain.

Each component includes:
- Method signatures with parameter types
- Return value descriptions
- Error handling information
- Example usage

## Core Graph Memory API

### BaseManager

Core functionality for Neo4j database connectivity and query execution.

#### `initialize() -> bool`

Initializes the Neo4j connection using environment variables.

**Returns:**
- `bool`: True if connection successful, False otherwise

**Example:**
```python
base_manager = BaseManager(logger)
success = base_manager.initialize()
if success:
    print("Connected to Neo4j database")
```

#### `execute_query(query: str, params: dict = None) -> list`

Executes a Cypher query against the Neo4j database.

**Parameters:**
- `query` (str): The Cypher query to execute
- `params` (dict, optional): Parameters for the query

**Returns:**
- `list`: List of query results as dictionaries

**Example:**
```python
results = base_manager.execute_query(
    "MATCH (n:Entity {name: $name}) RETURN n",
    {"name": "example_entity"}
)
```

#### `setup_vector_index() -> bool`

Creates vector index for semantic search if it doesn't exist.

**Returns:**
- `bool`: True if index created or already exists, False otherwise

**Example:**
```python
success = base_manager.setup_vector_index()
```

#### `close()`

Closes the Neo4j connection and releases resources.

**Example:**
```python
base_manager.close()
```

### EntityManager

Handles entity CRUD operations in the graph database.

#### `create_entity(entity_data: dict) -> dict`

Creates a new entity in the graph database.

**Parameters:**
- `entity_data` (dict): Entity properties including name, entityType, description, and optional tags

**Returns:**
- `dict`: Created entity with Neo4j ID

**Raises:**
- `ValueError`: If entity data is missing required fields

**Example:**
```python
entity = entity_manager.create_entity({
    "name": "example",
    "entityType": "Concept",
    "description": "An example entity"
})
```

#### `create_entities(entities: list) -> list`

Creates multiple entities in a batch operation.

**Parameters:**
- `entities` (list): List of entity dictionaries

**Returns:**
- `list`: List of created entities with Neo4j IDs

**Example:**
```python
entities = entity_manager.create_entities([
    {"name": "entity1", "entityType": "Type1"},
    {"name": "entity2", "entityType": "Type2"}
])
```

#### `get_entity(entity_name: str) -> dict`

Retrieves an entity by name.

**Parameters:**
- `entity_name` (str): Name of the entity to retrieve

**Returns:**
- `dict`: Entity data if found, None otherwise

**Example:**
```python
entity = entity_manager.get_entity("example")
```

#### `update_entity(entity_name: str, update_data: dict) -> dict`

Updates an entity's properties.

**Parameters:**
- `entity_name` (str): Name of the entity to update
- `update_data` (dict): Properties to update

**Returns:**
- `dict`: Updated entity data

**Example:**
```python
updated = entity_manager.update_entity(
    "example",
    {"description": "Updated description"}
)
```

#### `delete_entity(entity_name: str) -> bool`

Deletes an entity from the database.

**Parameters:**
- `entity_name` (str): Name of the entity to delete

**Returns:**
- `bool`: True if deleted, False otherwise

**Example:**
```python
success = entity_manager.delete_entity("example")
```

#### `get_entities_by_type(entity_type: str) -> list`

Retrieves all entities of a specific type.

**Parameters:**
- `entity_type` (str): The entity type to filter by

**Returns:**
- `list`: List of matching entities

**Example:**
```python
concepts = entity_manager.get_entities_by_type("Concept")
```

#### `add_tags_to_entity(entity_name: str, tags: list) -> dict`

Adds tags to an entity.

**Parameters:**
- `entity_name` (str): Name of the entity to tag
- `tags` (list): List of tag strings

**Returns:**
- `dict`: Updated entity with new tags

**Example:**
```python
tagged = entity_manager.add_tags_to_entity(
    "example",
    ["important", "verified"]
)
```

#### `search_entities(entity_type: str = None, search_term: str = None, tags: list = None) -> list`

Searches for entities using various filters.

**Parameters:**
- `entity_type` (str, optional): Entity type to filter by
- `search_term` (str, optional): Text to search for in name/description
- `tags` (list, optional): Tags to filter by

**Returns:**
- `list`: List of matching entities

**Example:**
```python
results = entity_manager.search_entities(
    entity_type="Concept",
    search_term="example",
    tags=["important"]
)
```

### RelationManager

Manages relationships between entities.

#### `create_relationship(source_entity: str, target_entity: str, relation_type: str, properties: dict = None) -> dict`

Creates a relationship between two entities.

**Parameters:**
- `source_entity` (str): Name of the source entity
- `target_entity` (str): Name of the target entity
- `relation_type` (str): Type of relationship
- `properties` (dict, optional): Relationship properties

**Returns:**
- `dict`: Created relationship data

**Example:**
```python
relation = relation_manager.create_relationship(
    "entity1",
    "entity2",
    "RELATED_TO",
    {"strength": 0.8}
)
```

#### `get_relationships(entity_name: str, direction: str = "OUTGOING") -> list`

Gets all relationships for an entity.

**Parameters:**
- `entity_name` (str): Name of the entity
- `direction` (str, optional): "OUTGOING", "INCOMING", or "BOTH"

**Returns:**
- `list`: List of relationships

**Example:**
```python
relations = relation_manager.get_relationships("entity1")
```

#### `get_relationships_by_type(relation_type: str) -> list`

Gets all relationships of a specific type.

**Parameters:**
- `relation_type` (str): Type of relationships to retrieve

**Returns:**
- `list`: List of relationships

**Example:**
```python
relations = relation_manager.get_relationships_by_type("RELATED_TO")
```

#### `update_relationship(source_entity: str, target_entity: str, relation_type: str, properties: dict) -> dict`

Updates a relationship's properties.

**Parameters:**
- `source_entity` (str): Name of the source entity
- `target_entity` (str): Name of the target entity
- `relation_type` (str): Type of relationship
- `properties` (dict): Updated properties

**Returns:**
- `dict`: Updated relationship data

**Example:**
```python
updated = relation_manager.update_relationship(
    "entity1",
    "entity2",
    "RELATED_TO",
    {"strength": 0.9}
)
```

#### `delete_relationship(source_entity: str, target_entity: str, relation_type: str) -> bool`

Deletes a relationship.

**Parameters:**
- `source_entity` (str): Name of the source entity
- `target_entity` (str): Name of the target entity
- `relation_type` (str): Type of relationship

**Returns:**
- `bool`: True if deleted, False otherwise

**Example:**
```python
success = relation_manager.delete_relationship(
    "entity1",
    "entity2",
    "RELATED_TO"
)
```

#### `delete_relationships_by_type(relation_type: str) -> tuple`

Deletes all relationships of a specific type.

**Parameters:**
- `relation_type` (str): Type of relationships to delete

**Returns:**
- `tuple`: (success_boolean, count_deleted)

**Example:**
```python
success, count = relation_manager.delete_relationships_by_type("OUTDATED")
```

#### `find_path_between_entities(source_entity: str, target_entity: str, max_depth: int = 5) -> dict`

Finds a path between two entities.

**Parameters:**
- `source_entity` (str): Starting entity name
- `target_entity` (str): Ending entity name
- `max_depth` (int, optional): Maximum path length to search

**Returns:**
- `dict`: Path information if found, None otherwise

**Example:**
```python
path = relation_manager.find_path_between_entities(
    "entity1",
    "entity2",
    max_depth=3
)
```

### ObservationManager

Manages observations attached to entities.

#### `add_observation(entity_name: str, observation_type: str, content: str, metadata: dict = None) -> dict`

Adds an observation to an entity.

**Parameters:**
- `entity_name` (str): Name of the entity
- `observation_type` (str): Type of observation
- `content` (str): Observation content
- `metadata` (dict, optional): Additional metadata

**Returns:**
- `dict`: Added observation data

**Example:**
```python
observation = observation_manager.add_observation(
    "entity1",
    "DESCRIPTION",
    "Detailed description content",
    {"confidence": 0.9}
)
```

#### `get_entity_observations(entity_name: str) -> list`

Gets all observations for an entity.

**Parameters:**
- `entity_name` (str): Name of the entity

**Returns:**
- `list`: List of observations

**Example:**
```python
observations = observation_manager.get_entity_observations("entity1")
```

#### `get_observations_by_type(entity_name: str, observation_type: str) -> list`

Gets entity observations of a specific type.

**Parameters:**
- `entity_name` (str): Name of the entity
- `observation_type` (str): Type of observations to retrieve

**Returns:**
- `list`: List of observations of specified type

**Example:**
```python
descriptions = observation_manager.get_observations_by_type(
    "entity1",
    "DESCRIPTION"
)
```

#### `update_observation(entity_name: str, observation_id: int, update_data: dict) -> dict`

Updates an observation.

**Parameters:**
- `entity_name` (str): Name of the entity
- `observation_id` (int): ID of the observation
- `update_data` (dict): Properties to update

**Returns:**
- `dict`: Updated observation data

**Example:**
```python
updated = observation_manager.update_observation(
    "entity1",
    123,
    {"content": "Updated content"}
)
```

#### `delete_observation(entity_name: str, observation_id: int) -> bool`

Deletes an observation.

**Parameters:**
- `entity_name` (str): Name of the entity
- `observation_id` (int): ID of the observation

**Returns:**
- `bool`: True if deleted, False otherwise

**Example:**
```python
success = observation_manager.delete_observation("entity1", 123)
```

#### `get_latest_observation(entity_name: str, observation_type: str) -> dict`

Gets the most recent observation of a specific type.

**Parameters:**
- `entity_name` (str): Name of the entity
- `observation_type` (str): Type of observation

**Returns:**
- `dict`: Latest observation data

**Example:**
```python
latest = observation_manager.get_latest_observation(
    "entity1",
    "STATUS"
)
```

### SearchManager

Handles advanced search operations across the graph.

#### `search_by_text(search_term: str, entity_type: str = None, limit: int = 20) -> list`

Searches entities by text content.

**Parameters:**
- `search_term` (str): Text to search for
- `entity_type` (str, optional): Entity type to filter by
- `limit` (int, optional): Maximum results to return

**Returns:**
- `list`: Matching entities

**Example:**
```python
results = search_manager.search_by_text("specific feature")
```

#### `semantic_search(query_text: str, limit: int = 10, threshold: float = 0.6) -> list`

Searches entities by semantic similarity.

**Parameters:**
- `query_text` (str): Query text to find similar entities
- `limit` (int, optional): Maximum results to return
- `threshold` (float, optional): Minimum similarity score

**Returns:**
- `list`: Semantically similar entities with scores

**Example:**
```python
results = search_manager.semantic_search(
    "concepts related to machine learning",
    limit=5,
    threshold=0.7
)
```

#### `combined_search(text_query: str, semantic_query: str, entity_type: str = None, text_weight: float = 0.3, semantic_weight: float = 0.7, limit: int = 10) -> list`

Performs hybrid text and semantic search.

**Parameters:**
- `text_query` (str): Text to search for
- `semantic_query` (str): Query for semantic similarity
- `entity_type` (str, optional): Entity type to filter by
- `text_weight` (float, optional): Weight for text search
- `semantic_weight` (float, optional): Weight for semantic search
- `limit` (int, optional): Maximum results to return

**Returns:**
- `list`: Ranked search results with scores

**Example:**
```python
results = search_manager.combined_search(
    "important",
    "concepts related to knowledge representation",
    entity_type="Concept",
    text_weight=0.4,
    semantic_weight=0.6
)
```

### EmbeddingAdapter

Handles embedding generation and vector operations.

#### `generate_embedding(text: str) -> list`

Generates an embedding vector for text.

**Parameters:**
- `text` (str): Text to generate embedding for

**Returns:**
- `list`: Embedding vector (floating-point values)

**Example:**
```python
embedding = embedding_adapter.generate_embedding(
    "This is the text to embed"
)
```

#### `get_entity_embedding(entity_name: str) -> list`

Gets the embedding for an entity.

**Parameters:**
- `entity_name` (str): Name of the entity

**Returns:**
- `list`: Entity's embedding vector

**Example:**
```python
embedding = embedding_adapter.get_entity_embedding("entity1")
```

#### `batch_generate_embeddings(texts: list) -> list`

Generates embeddings for multiple texts.

**Parameters:**
- `texts` (list): List of text strings

**Returns:**
- `list`: List of embedding vectors

**Example:**
```python
embeddings = embedding_adapter.batch_generate_embeddings(
    ["Text 1", "Text 2", "Text 3"]
)
```

#### `calculate_similarity(embedding1: list, embedding2: list) -> float`

Calculates cosine similarity between embeddings.

**Parameters:**
- `embedding1` (list): First embedding vector
- `embedding2` (list): Second embedding vector

**Returns:**
- `float`: Similarity score (0-1)

**Example:**
```python
similarity = embedding_adapter.calculate_similarity(
    embedding1,
    embedding2
)
```

## Lesson Memory API

### LessonContainer

Manages lesson containers and collections.

#### `create_lesson_container(container_data: dict) -> dict`

Creates a new lesson container.

**Parameters:**
- `container_data` (dict): Container properties including name, description, and owner

**Returns:**
- `dict`: Created container with Neo4j ID

**Example:**
```python
container = lesson_container.create_lesson_container({
    "name": "Programming Basics",
    "description": "Foundational programming concepts",
    "owner": "system"
})
```

#### `get_lesson_container(container_name: str) -> dict`

Retrieves a lesson container by name.

**Parameters:**
- `container_name` (str): Name of the container

**Returns:**
- `dict`: Container data with entity counts

**Example:**
```python
container = lesson_container.get_lesson_container("Programming Basics")
```

#### `list_lesson_containers(limit: int = 100, sort_by: str = "name") -> list`

Lists all lesson containers.

**Parameters:**
- `limit` (int, optional): Maximum containers to return
- `sort_by` (str, optional): Property to sort by

**Returns:**
- `list`: List of container data

**Example:**
```python
containers = lesson_container.list_lesson_containers(sort_by="created_date")
```

#### `add_lesson_to_container(lesson_name: str, container_name: str) -> bool`

Adds a lesson to a container.

**Parameters:**
- `lesson_name` (str): Name of the lesson entity
- `container_name` (str): Name of the container

**Returns:**
- `bool`: True if added, False otherwise

**Example:**
```python
success = lesson_container.add_lesson_to_container(
    "Variables Introduction",
    "Programming Basics"
)
```

#### `remove_lesson_from_container(lesson_name: str, container_name: str) -> bool`

Removes a lesson from a container.

**Parameters:**
- `lesson_name` (str): Name of the lesson entity
- `container_name` (str): Name of the container

**Returns:**
- `bool`: True if removed, False otherwise

**Example:**
```python
success = lesson_container.remove_lesson_from_container(
    "Variables Introduction",
    "Programming Basics"
)
```

### LessonEntity

Manages lesson-specific entity operations.

#### `create_lesson_entity(entity_data: dict, container_name: str = None) -> dict`

Creates a lesson entity and adds it to a container.

**Parameters:**
- `entity_data` (dict): Entity properties
- `container_name` (str, optional): Container to add lesson to

**Returns:**
- `dict`: Created lesson entity

**Example:**
```python
lesson = lesson_entity.create_lesson_entity(
    {
        "name": "Variables Introduction",
        "description": "Introduction to programming variables",
        "content": "Variables are used to store data...",
        "difficulty": "beginner"
    },
    "Programming Basics"
)
```

#### `get_lesson_entity(entity_name: str, container_name: str = None) -> dict`

Retrieves a lesson entity.

**Parameters:**
- `entity_name` (str): Name of the lesson
- `container_name` (str, optional): Container to verify membership

**Returns:**
- `dict`: Lesson entity data

**Example:**
```python
lesson = lesson_entity.get_lesson_entity("Variables Introduction")
```

#### `update_lesson_entity(entity_name: str, update_data: dict) -> dict`

Updates a lesson entity.

**Parameters:**
- `entity_name` (str): Name of the lesson
- `update_data` (dict): Properties to update

**Returns:**
- `dict`: Updated lesson data

**Example:**
```python
updated = lesson_entity.update_lesson_entity(
    "Variables Introduction",
    {"difficulty": "intermediate", "content": "Updated content..."}
)
```

#### `search_lesson_entities(container_name: str = None, search_term: str = None, entity_type: str = None, tags: list = None) -> list`

Searches for lesson entities.

**Parameters:**
- `container_name` (str, optional): Container to search within
- `search_term` (str, optional): Text to search for
- `entity_type` (str, optional): Specific lesson type
- `tags` (list, optional): Tags to filter by

**Returns:**
- `list`: Matching lesson entities

**Example:**
```python
results = lesson_entity.search_lesson_entities(
    container_name="Programming Basics",
    search_term="variables",
    tags=["beginner"]
)
```

### LessonRelation

Manages relationships between lesson entities.

#### `create_lesson_relation(source_lesson: str, target_lesson: str, relation_type: str, container_name: str = None, properties: dict = None) -> dict`

Creates a relationship between lesson entities.

**Parameters:**
- `source_lesson` (str): Source lesson name
- `target_lesson` (str): Target lesson name
- `relation_type` (str): Type of relationship
- `container_name` (str, optional): Container to verify membership
- `properties` (dict, optional): Relationship properties

**Returns:**
- `dict`: Created relationship data

**Example:**
```python
relation = lesson_relation.create_lesson_relation(
    "Variables Introduction",
    "Data Types",
    "PREREQUISITE_FOR",
    "Programming Basics",
    {"importance": "high"}
)
```

#### `get_lesson_knowledge_graph(container_name: str, max_depth: int = 3) -> dict`

Gets a knowledge graph of lessons and relationships.

**Parameters:**
- `container_name` (str): Container name
- `max_depth` (int, optional): Maximum relationship depth

**Returns:**
- `dict`: Knowledge graph with nodes and links

**Example:**
```python
graph = lesson_relation.get_lesson_knowledge_graph(
    "Programming Basics",
    max_depth=2
)
```

#### `track_lesson_application(lesson_name: str, context: str, success: bool, metadata: dict = None) -> dict`

Records a lesson application to a specific context.

**Parameters:**
- `lesson_name` (str): Lesson that was applied
- `context` (str): Context of application
- `success` (bool): Whether application was successful
- `metadata` (dict, optional): Additional metadata

**Returns:**
- `dict`: Application tracking data

**Example:**
```python
tracking = lesson_relation.track_lesson_application(
    "Error Handling",
    "Project XYZ Bug Fix",
    True,
    {"confidence": 0.95, "impact": "high"}
)
```

### LessonObservation

Manages structured observations for lessons.

#### `add_lesson_observation(lesson_name: str, observation_type: str, content: str, container_name: str = None, metadata: dict = None) -> dict`

Adds a structured observation to a lesson.

**Parameters:**
- `lesson_name` (str): Name of the lesson
- `observation_type` (str): Type of observation
- `content` (str): Observation content
- `container_name` (str, optional): Container to verify membership
- `metadata` (dict, optional): Additional metadata

**Returns:**
- `dict`: Added observation data

**Example:**
```python
observation = lesson_observation.add_lesson_observation(
    "Error Handling",
    "APPLICATION_CONTEXT",
    "Most effective for web API development",
    "Programming Basics",
    {"confidence": 0.9}
)
```

#### `check_observation_completeness(lesson_name: str, container_name: str = None) -> dict`

Checks which structured observation types are present.

**Parameters:**
- `lesson_name` (str): Name of the lesson
- `container_name` (str, optional): Container to verify membership

**Returns:**
- `dict`: Completeness data with score and missing types

**Example:**
```python
completeness = lesson_observation.check_observation_completeness(
    "Error Handling",
    "Programming Basics"
)
```

### EvolutionTracker

Tracks knowledge evolution over time.

#### `track_knowledge_evolution(lesson_name: str, time_period: str = "ALL") -> dict`

Analyzes how a lesson has evolved over time.

**Parameters:**
- `lesson_name` (str): Name of the lesson
- `time_period` (str, optional): Time period for analysis

**Returns:**
- `dict`: Evolution data with changes and metrics

**Example:**
```python
evolution = evolution_tracker.track_knowledge_evolution(
    "Error Handling",
    time_period="LAST_MONTH"
)
```

#### `get_confidence_trend(lesson_name: str, observation_type: str = None) -> list`

Gets confidence trend for a lesson over time.

**Parameters:**
- `lesson_name` (str): Name of the lesson
- `observation_type` (str, optional): Observation type to track

**Returns:**
- `list`: Confidence scores with timestamps

**Example:**
```python
confidence_trend = evolution_tracker.get_confidence_trend(
    "Error Handling",
    observation_type="EFFECTIVENESS"
)
```

### Consolidation

Manages memory consolidation and optimization.

#### `find_similar_lessons(lesson_name: str, threshold: float = 0.7) -> list`

Finds similar lessons for potential consolidation.

**Parameters:**
- `lesson_name` (str): Name of the reference lesson
- `threshold` (float, optional): Similarity threshold

**Returns:**
- `list`: Similar lessons with similarity scores

**Example:**
```python
similar = consolidation.find_similar_lessons(
    "Exception Handling",
    threshold=0.8
)
```

#### `merge_lessons(source_lesson: str, target_lesson: str, merge_strategy: str = "PRESERVE_BOTH") -> dict`

Merges two similar lessons.

**Parameters:**
- `source_lesson` (str): Source lesson name
- `target_lesson` (str): Target lesson name
- `merge_strategy` (str, optional): How to handle conflicts

**Returns:**
- `dict`: Merged lesson data

**Example:**
```python
merged = consolidation.merge_lessons(
    "Exception Handling",
    "Error Handling",
    merge_strategy="KEEP_TARGET"
)
```

## Project Memory API

### ProjectContainer

Manages project containers and hierarchies.

#### `create_project_container(container_data: dict) -> dict`

Creates a new project container.

**Parameters:**
- `container_data` (dict): Container properties

**Returns:**
- `dict`: Created container with Neo4j ID

**Example:**
```python
container = project_container.create_project_container({
    "name": "Web Application",
    "description": "Customer-facing web portal",
    "status": "active"
})
```

#### `get_container_components(container_name: str, component_type: str = None) -> list`

Gets all components in a project container.

**Parameters:**
- `container_name` (str): Name of the container
- `component_type` (str, optional): Type of components to filter by

**Returns:**
- `list`: List of container components

**Example:**
```python
components = project_container.get_container_components(
    "Web Application",
    component_type="Feature"
)
```

### DomainManager

Manages project domain entities.

#### `create_domain(domain_data: dict, container_name: str) -> dict`

Creates a domain entity within a project.

**Parameters:**
- `domain_data` (dict): Domain properties
- `container_name` (str): Project container name

**Returns:**
- `dict`: Created domain entity

**Example:**
```python
domain = domain_manager.create_domain(
    {
        "name": "Authentication",
        "description": "User authentication system",
        "priority": "high"
    },
    "Web Application"
)
```

#### `get_domains_by_container(container_name: str) -> list`

Gets all domains for a project.

**Parameters:**
- `container_name` (str): Project container name

**Returns:**
- `list`: List of domain entities

**Example:**
```python
domains = domain_manager.get_domains_by_container("Web Application")
```

### ComponentManager

Manages project component entities.

#### `create_component(component_data: dict, container_name: str, domain_name: str = None) -> dict`

Creates a component entity within a project.

**Parameters:**
- `component_data` (dict): Component properties
- `container_name` (str): Project container name
- `domain_name` (str, optional): Domain to associate with

**Returns:**
- `dict`: Created component entity

**Example:**
```python
component = component_manager.create_component(
    {
        "name": "LoginService",
        "description": "Handles user login and session management",
        "component_type": "Service",
        "tech_stack": ["Node.js", "Express"]
    },
    "Web Application",
    "Authentication"
)
```

#### `get_component_dependencies(component_name: str) -> list`

Gets all dependencies for a component.

**Parameters:**
- `component_name` (str): Component name

**Returns:**
- `list`: List of dependencies with relationship details

**Example:**
```python
dependencies = component_manager.get_component_dependencies("LoginService")
```

### FeatureManager

Manages project feature entities.

#### `create_feature(feature_data: dict, container_name: str, component_name: str = None) -> dict`

Creates a feature entity within a project.

**Parameters:**
- `feature_data` (dict): Feature properties
- `container_name` (str): Project container name
- `component_name` (str, optional): Component to associate with

**Returns:**
- `dict`: Created feature entity

**Example:**
```python
feature = feature_manager.create_feature(
    {
        "name": "2FA",
        "description": "Two-factor authentication",
        "status": "in_progress",
        "priority": "high"
    },
    "Web Application",
    "LoginService"
)
```

#### `update_feature_status(feature_name: str, status: str, metadata: dict = None) -> dict`

Updates a feature's status.

**Parameters:**
- `feature_name` (str): Feature name
- `status` (str): New status value
- `metadata` (dict, optional): Additional metadata

**Returns:**
- `dict`: Updated feature data

**Example:**
```python
updated = feature_manager.update_feature_status(
    "2FA",
    "completed",
    {"completion_date": "2025-03-15"}
)
```

### DecisionManager

Manages project decision entities.

#### `record_decision(decision_data: dict, container_name: str, related_components: list = None) -> dict`

Records a project decision.

**Parameters:**
- `decision_data` (dict): Decision properties
- `container_name` (str): Project container name
- `related_components` (list, optional): Components affected by decision

**Returns:**
- `dict`: Created decision entity

**Example:**
```python
decision = decision_manager.record_decision(
    {
        "title": "OAuth Implementation",
        "description": "Will use OAuth 2.0 with JWT tokens",
        "rationale": "Industry standard with good library support",
        "decision_maker": "Tech Lead",
        "decision_date": "2025-03-10"
    },
    "Web Application",
    ["LoginService", "UserService"]
)
```

#### `get_component_decisions(component_name: str) -> list`

Gets all decisions affecting a component.

**Parameters:**
- `component_name` (str): Component name

**Returns:**
- `list`: List of related decisions

**Example:**
```python
decisions = decision_manager.get_component_decisions("LoginService")
```

### KnowledgeTransfer

Facilitates cross-project knowledge transfer.

#### `transfer_component_knowledge(source_component: str, target_component: str, knowledge_types: list = None) -> dict`

Transfers knowledge between similar components.

**Parameters:**
- `source_component` (str): Source component name
- `target_component` (str): Target component name
- `knowledge_types` (list, optional): Types of knowledge to transfer

**Returns:**
- `dict`: Transfer results with statistics

**Example:**
```python
transfer = knowledge_transfer.transfer_component_knowledge(
    "LegacyLoginService",
    "NewLoginService",
    ["lessons", "decisions", "observations"]
)
```

#### `find_cross_project_patterns(entity_type: str, min_occurrences: int = 3) -> list`

Identifies patterns across multiple projects.

**Parameters:**
- `entity_type` (str): Entity type to analyze
- `min_occurrences` (int, optional): Minimum occurrences to qualify as pattern

**Returns:**
- `list`: Identified patterns with statistics

**Example:**
```python
patterns = knowledge_transfer.find_cross_project_patterns(
    "Decision",
    min_occurrences=2
)
```

## Facade API

### GraphMemoryManager

Main facade providing backward compatibility with original API.

#### `initialize() -> bool`

Initializes all memory systems.

**Returns:**
- `bool`: True if initialization successful

**Example:**
```python
memory_manager = GraphMemoryManager()
success = memory_manager.initialize()
```

#### `create_entities(entities: list) -> list`

Creates multiple entities (facade method).

**Parameters:**
- `entities` (list): List of entity dictionaries

**Returns:**
- `list`: Created entities with IDs

**Example:**
```python
entities = memory_manager.create_entities([
    {"name": "entity1", "entityType": "Type1"},
    {"name": "entity2", "entityType": "Type2"}
])
```

### LessonMemoryManager

Unified facade for lesson memory operations.

#### `initialize(base_manager = None) -> bool`

Initializes lesson memory components.

**Parameters:**
- `base_manager` (BaseManager, optional): Existing base manager to use

**Returns:**
- `bool`: True if initialization successful

**Example:**
```python
lesson_memory = LessonMemoryManager()
success = lesson_memory.initialize()
```

#### `create_lesson(lesson_data: dict, container_name: str = None, observations: dict = None) -> dict`

Comprehensive lesson creation with optional observations.

**Parameters:**
- `lesson_data` (dict): Lesson entity properties
- `container_name` (str, optional): Container to add lesson to
- `observations` (dict, optional): Initial observations to add

**Returns:**
- `dict`: Created lesson with observation IDs

**Example:**
```python
lesson = lesson_memory.create_lesson(
    {
        "name": "Debugging Techniques",
        "description": "Advanced debugging approaches",
        "difficulty": "intermediate"
    },
    "Programming Skills",
    {
        "APPLICATION_CONTEXT": "Most useful for complex systems",
        "EFFECTIVENESS": "Reduces debugging time by 50%"
    }
)
```

### ProjectMemoryManager

Unified facade for project memory operations.

#### `initialize(base_manager = None) -> bool`

Initializes project memory components.

**Parameters:**
- `base_manager` (BaseManager, optional): Existing base manager to use

**Returns:**
- `bool`: True if initialization successful

**Example:**
```python
project_memory = ProjectMemoryManager()
success = project_memory.initialize()
```

#### `create_project(project_data: dict, domains: list = None, components: list = None) -> dict`

Comprehensive project creation with optional domains and components.

**Parameters:**
- `project_data` (dict): Project container properties
- `domains` (list, optional): Initial domains to create
- `components` (list, optional): Initial components to create

**Returns:**
- `dict`: Created project with domain and component IDs

**Example:**
```python
project = project_memory.create_project(
    {
        "name": "Mobile App",
        "description": "iOS/Android mobile application",
        "status": "planning"
    },
    [
        {"name": "Frontend", "description": "UI/UX components"},
        {"name": "Backend", "description": "API services"}
    ],
    [
        {"name": "AuthModule", "description": "Authentication module", "domain": "Backend"},
        {"name": "ProfileScreen", "description": "User profile UI", "domain": "Frontend"}
    ]
)
``` 