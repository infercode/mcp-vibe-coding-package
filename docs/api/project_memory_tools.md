# Project Memory Tools API Documentation

## Overview

The Project Memory Tools provide a simplified interface for AI agents to interact with the Project Memory System. These tools allow agents to manage hierarchical project knowledge in a structured way, including creating projects, components, and domain entities, establishing relationships between them, and searching for relevant information.

## Tools

### project_memory_tool

```python
@project_memory_tool(operation_type: str, **kwargs) -> str
```

Manage project memory with a unified interface.

#### Parameters

- `operation_type`: The type of operation to perform
  - `create_project`: Create a new project container
  - `create_component`: Create a component within a project
  - `create_domain_entity`: Create a domain entity
  - `relate_entities`: Create relationships between entities
  - `search`: Find relevant project entities
  - `get_structure`: Retrieve project hierarchy
  - `add_observation`: Add observations to entities
  - `update`: Update existing entities
- `**kwargs`: Operation-specific parameters (see below)
  
#### Operation-Specific Parameters

##### create_project
- `name` (str, required): Name of the project to create
- `description` (str, optional): Description of the project
- `tags` (list, optional): List of tags for the project

##### create_component
- `project_id` (str, required): ID or name of the project
- `name` (str, required): Name of the component
- `component_type` (str, required): Type of component (e.g., "MICROSERVICE", "LIBRARY")

##### create_domain_entity
- `project_id` (str, required): ID or name of the project
- `entity_type` (str, required): Type of domain entity (e.g., "DECISION", "REQUIREMENT")
- `name` (str, required): Name of the entity

##### relate_entities
- `source_id` (str, required): ID or name of the source entity
- `target_id` (str, required): ID or name of the target entity
- `relationship_type` (str, required): Type of relationship (e.g., "DEPENDS_ON", "IMPLEMENTS")

##### search
- `query` (str, required): Search query
- `project_id` (str, optional): ID or name of the project to search within
- `entity_types` (list, optional): List of entity types to filter by
- `limit` (int, optional): Maximum number of results to return

##### get_structure
- `project_id` (str, required): ID or name of the project

##### add_observation
- `entity_id` (str, required): ID or name of the entity
- `content` (str, required): Content of the observation
- `observation_type` (str, optional): Type of observation

##### update
- `entity_id` (str, required): ID or name of the entity to update
- `updates` (dict, required): Dictionary of fields to update

#### Return Value

A JSON string with the following structure:

```json
{
  "status": "success" | "error",
  "message" | "error": "Description of result or error",
  "data" | "container" | "component" | "entity" | "results": {...}  // Operation-specific data
}
```

#### Examples

Create a new project:

```python
result = @project_memory_tool({
    "operation_type": "create_project",
    "name": "E-commerce Platform",
    "description": "Online store with microservices architecture"
})
```

Create a component within a project:

```python
result = @project_memory_tool({
    "operation_type": "create_component",
    "project_id": "E-commerce Platform",
    "name": "Authentication Service",
    "component_type": "MICROSERVICE"
})
```

Search for entities:

```python
result = @project_memory_tool({
    "operation_type": "search",
    "query": "authentication patterns",
    "project_id": "E-commerce Platform",
    "limit": 5
})
```

### project_memory_context

```python
@project_memory_context(context_data: Dict[str, Any], client_id: Optional[str] = None) -> str
```

Create a context for batch project memory operations.

#### Parameters

- `context_data`: Dictionary containing context information
  - `project_name` (str, required): Project name to set as context
- `client_id` (str, optional): Client ID for identifying the connection

#### Return Value

A JSON string with the following structure:

```json
{
  "status": "success" | "error",
  "message" | "error": "Description of result or error",
  "context": {  // Only present on success
    "project_name": "ProjectName",
    "created_at": "2023-07-15T10:30:45.123456",
    "operations_available": ["create_component", "create_domain_entity", "relate_entities", "search", "get_structure", "add_observation", "update"],
    "usage": "Use this context information with any project memory operation by including it in the operation's context parameter"
  }
}
```

#### Example

Create a context for a project:

```python
context = @project_memory_context({
    "project_name": "E-commerce Platform"
})
```

Use the context with an operation:

```python
result = @project_memory_tool({
    "operation_type": "create_component",
    "name": "Payment Processor",
    "component_type": "SERVICE",
    "context": context["context"]  // Pass the context object
})
```

## Working with Project Memory

### Project Structure

The Project Memory System uses a hierarchical structure:

1. **Project Containers**: Top-level entities that represent entire projects
2. **Components**: Functional units within projects (services, libraries, etc.)
3. **Domain Entities**: Project-specific knowledge items (decisions, requirements, etc.)
4. **Relationships**: Connections between entities showing dependencies and structure

### Best Practices

- Always create a project container before adding components or domain entities
- Use descriptive names for all entities to aid in search and retrieval
- Establish relationships between related components to document dependencies
- Add observations to entities to track decisions and progress
- Use context management for batch operations within the same project

### Usage Patterns

#### Creating a Project with Multiple Components

```python
# Create project
project_result = @project_memory_tool({
    "operation_type": "create_project",
    "name": "Customer Portal",
    "description": "Web application for customer self-service"
})

# Get context for batch operations
context = @project_memory_context({
    "project_name": "Customer Portal"
})

# Create multiple components with context
components = ["Frontend", "API Gateway", "User Service", "Order Service"]
for component in components:
    result = @project_memory_tool({
        "operation_type": "create_component",
        "name": component,
        "component_type": "SERVICE",
        "context": context["context"]
    })
```

#### Documenting Architecture Decisions

```python
# Create a decision entity
decision = @project_memory_tool({
    "operation_type": "create_domain_entity",
    "project_id": "Customer Portal",
    "entity_type": "DECISION",
    "name": "Use GraphQL for API"
})

# Add an observation explaining the decision
observation = @project_memory_tool({
    "operation_type": "add_observation",
    "entity_id": "Use GraphQL for API",
    "content": "Decided to use GraphQL instead of REST to allow flexible queries and reduce over-fetching of data.",
    "observation_type": "RATIONALE"
})
```

#### Establishing Component Dependencies

```python
# Create relationships between components
dependency = @project_memory_tool({
    "operation_type": "relate_entities",
    "source_id": "Frontend",
    "target_id": "API Gateway",
    "relationship_type": "DEPENDS_ON"
})

# Create another dependency
dependency = @project_memory_tool({
    "operation_type": "relate_entities",
    "source_id": "API Gateway",
    "target_id": "User Service",
    "relationship_type": "ROUTES_TO"
})
```

## Advanced Usage

### Semantic Search

The search operation supports semantic (meaning-based) search when available:

```python
search_results = @project_memory_tool({
    "operation_type": "search",
    "query": "components handling user authentication and authorization",
    "project_id": "Customer Portal",
    "semantic": true
})
```

### Project Structure Visualization

Retrieve the full project structure for visualization:

```python
structure = @project_memory_tool({
    "operation_type": "get_structure",
    "project_id": "Customer Portal"
})
```

The returned structure contains all components, domain entities, and their relationships, which can be used to generate architectural diagrams. 