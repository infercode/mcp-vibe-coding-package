# Project Memory Tools API Documentation

*Version: 1.2.0 - Last Updated: August 2023*

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
  - `delete_entity`: Delete a project entity (project, domain, component, or observation)
  - `delete_relationship`: Delete a relationship between entities
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
- `domain_name` (str, optional): Name of the domain to place component in

##### create_domain_entity
- `project_id` (str, required): ID or name of the project
- `entity_type` (str, required): Type of domain entity (e.g., "DECISION", "REQUIREMENT")
- `name` (str, required): Name of the entity

##### relate_entities
- `source_name` (str, required): Name of the source entity
- `target_name` (str, required): Name of the target entity
- `relation_type` (str, required): Type of relationship (e.g., "DEPENDS_ON", "IMPLEMENTS")
- `project_id` (str, required): ID or name of the project
- `domain_name` (str, optional): Domain name if entities are components in a domain

##### search
- `query` (str, required): Search query
- `project_id` (str, required): ID or name of the project to search within
- `entity_types` (list, optional): List of entity types to filter by
- `limit` (int, optional): Maximum number of results to return
- `semantic` (bool, optional): Whether to use semantic search (default: true)

##### get_structure
- `project_id` (str, required): ID or name of the project

##### add_observation
- `entity_name` (str, required): Name of the entity
- `content` (str, required): Content of the observation
- `observation_type` (str, optional): Type of observation

##### update
- `entity_name` (str, required): Name of the entity to update
- `updates` (dict, required): Dictionary of fields to update
- `project_id` (str, optional): ID or name of the project
- `entity_type` (str, optional): Type of entity being updated
- `domain_name` (str, optional): Domain name if updating a component

##### delete_entity
- `entity_name` (str, required): Name of the entity to delete
- `entity_type` (str, required): Type of entity to delete ("project", "domain", "component", or "observation")
- `project_id` (str, optional): ID or name of the project (required for domains and components)
- `domain_name` (str, optional): Domain name (required for components)
- `delete_contents` (bool, optional): Whether to delete contained entities (default: false)
- `observation_content` (str, optional): Content of observation to delete (alternative to observation_id)
- `observation_id` (str, optional): ID of the observation (alternative to content)

##### delete_relationship
- `source_name` (str, required): Name of the source entity
- `target_name` (str, required): Name of the target entity
- `relationship_type` (str, required): Type of relationship to delete
- `project_id` (str, optional): ID or name of the project
- `domain_name` (str, optional): Domain name if entities are components in a domain

#### Return Value

A JSON string with the following structure:

```json
{
  "status": "success" | "error",
  "message" | "error": "Description of result or error",
  "data" | "container" | "component" | "entity" | "results": {...}  // Operation-specific data
}
```

#### Error Handling

Common error codes that may be returned:

- `invalid_operation`: The operation type is not supported
- `missing_parameter`: A required parameter is missing
- `entity_not_found`: The specified entity could not be found
- `container_not_found`: The specified project container could not be found
- `validation_error`: The provided data does not pass validation
- `operation_error`: A general error occurred during the operation

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
    "component_type": "MICROSERVICE",
    "domain_name": "Backend"
})
```

Create a relationship between entities:

```python
result = @project_memory_tool({
    "operation_type": "relate_entities",
    "source_name": "Authentication Service",
    "target_name": "User Database",
    "relation_type": "DEPENDS_ON",
    "project_id": "E-commerce Platform"
})
```

Search for entities:

```python
result = @project_memory_tool({
    "operation_type": "search",
    "query": "authentication patterns",
    "project_id": "E-commerce Platform",
    "limit": 5,
    "semantic": true
})
```

Add an observation to an entity:

```python
result = @project_memory_tool({
    "operation_type": "add_observation",
    "entity_name": "Authentication Service",
    "content": "Implemented JWT-based authentication with 24-hour token expiry",
    "observation_type": "IMPLEMENTATION_DETAIL"
})
```

Update an entity:

```python
result = @project_memory_tool({
    "operation_type": "update",
    "entity_name": "Authentication Service",
    "updates": {
        "description": "Service handling user authentication and authorization"
    },
    "project_id": "E-commerce Platform",
    "entity_type": "component"
})
```

Delete a component:

```python
result = @project_memory_tool({
    "operation_type": "delete_entity",
    "entity_name": "Payment Gateway",
    "entity_type": "component",
    "project_id": "E-commerce Platform",
    "domain_name": "Payment"
})
```

Delete a relationship:

```python
result = @project_memory_tool({
    "operation_type": "delete_relationship",
    "source_name": "Authentication Service",
    "target_name": "User Database",
    "relationship_type": "DEPENDS_ON",
    "project_id": "E-commerce Platform",
    "domain_name": "Backend"
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
    "operations_available": ["create_component", "create_domain_entity", "relate_entities", "search", "get_structure", "add_observation", "update", "delete_entity", "delete_relationship"],
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
- Clean up mistakenly created entities or incorrect relationships promptly using delete operations

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
    "entity_name": "Use GraphQL for API",
    "content": "Decided to use GraphQL instead of REST to allow flexible queries and reduce over-fetching of data.",
    "observation_type": "RATIONALE"
})
```

#### Establishing Component Dependencies

```python
# Create relationships between components
dependency = @project_memory_tool({
    "operation_type": "relate_entities",
    "source_name": "Frontend",
    "target_name": "API Gateway",
    "relation_type": "DEPENDS_ON",
    "project_id": "Customer Portal"
})

# Create another dependency
dependency = @project_memory_tool({
    "operation_type": "relate_entities",
    "source_name": "API Gateway",
    "target_name": "User Service",
    "relation_type": "ROUTES_TO",
    "project_id": "Customer Portal"
})
```

#### Cleaning Up Mistakenly Created Entities

```python
# Delete a component that was created by mistake
result = @project_memory_tool({
    "operation_type": "delete_entity",
    "entity_name": "Shopping Cart Service",
    "entity_type": "component",
    "project_id": "Customer Portal",
    "domain_name": "Backend"
})

# Remove an incorrect relationship
result = @project_memory_tool({
    "operation_type": "delete_relationship",
    "source_name": "Frontend",
    "target_name": "Order Service",
    "relationship_type": "DEPENDS_ON",
    "project_id": "Customer Portal"
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