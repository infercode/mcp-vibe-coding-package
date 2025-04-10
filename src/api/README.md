# Memory System API

A FastAPI-based REST API for interacting with the graph-based memory system. This API provides comprehensive endpoints for managing entities, relations, observations, projects, lessons, and search functionality in the knowledge graph.

## Features

- Entity Management (CRUD operations)
- Relation Management (including bulk operations)
- Observation Tracking for both Projects and Lessons
- Project Memory System with Component Management
- Lesson Memory System with Section Management
- Advanced Search Capabilities (semantic, neighborhood, path finding)
- Custom Cypher Query Support

## Setup

1. **Clone the repository**

```bash
git clone <repository-url>
cd <repository-name>
```

2. **Install dependencies**

First, ensure you have `uv` installed:
```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

Then install the project dependencies:
```bash
uv sync
```

3. **Configure environment variables**

Copy the example environment file and update it with your settings:

```bash
cp src/api/.env.example src/api/.env
```

Edit the `.env` file with your configuration:

```env
# Neo4j Settings
NEO4J_URI=bolt://localhost:7687
NEO4J_USER=neo4j
NEO4J_PASSWORD=your_password_here
NEO4J_DATABASE=neo4j

# Embedding Settings
EMBEDDING_PROVIDER=openai
EMBEDDING_MODEL=text-embedding-ada-002
EMBEDDING_API_KEY=your_api_key_here
```

4. **Run the API**

```bash
uv run src.api.main
```

The API will be available at `http://localhost:8000`.

## API Documentation

Once the server is running, you can access:

- Interactive API documentation (Swagger UI): `http://localhost:8000/docs`
- Alternative API documentation (ReDoc): `http://localhost:8000/redoc`

## API Endpoints

### Entity Management

- `POST /entities/` - Create a new entity
- `GET /entities/{entity_name}` - Get entity details
- `PUT /entities/{entity_name}` - Update an entity
- `DELETE /entities/{entity_name}` - Delete an entity
- `GET /core/memories` - Get all entities in the knowledge graph

### Relation Management

#### Core Relations
- `POST /relations/` - Create a new relation
- `POST /relations/bulk` - Create multiple relations
- `GET /relations/{entity_name}` - Get relations for an entity
- `PUT /relations/{from_entity}/{to_entity}/{relation_type}` - Update a relation
- `DELETE /relations/{from_entity}/{to_entity}` - Delete a relation

#### Lesson Relations
- `POST /relations/lessons/{lesson_id}` - Create a lesson relation
- `POST /relations/lessons/{lesson_id}/bulk` - Create multiple lesson relations
- `GET /relations/lessons/{lesson_id}` - Get lesson relations
- `DELETE /relations/lessons/{lesson_id}/{to_lesson_id}` - Delete a lesson relation

#### Project Relations
- `POST /relations/projects/{project_id}` - Create a project relation
- `POST /relations/projects/{project_id}/bulk` - Create multiple project relations
- `GET /relations/projects/{project_id}` - Get project relations
- `DELETE /relations/projects/{project_id}/{to_project_id}` - Delete a project relation

### Observation Management

#### Lesson Observations
- `POST /observations/lessons/{lesson_id}` - Add lesson observation
- `GET /observations/lessons/{lesson_id}` - Get lesson observations
- `PUT /observations/lessons/{lesson_id}/{observation_id}` - Update lesson observation
- `DELETE /observations/lessons/{lesson_id}` - Delete lesson observation

#### Project Observations
- `POST /observations/projects/{project_id}` - Add project observation
- `GET /observations/projects/{project_id}` - Get project observations
- `PUT /observations/projects/{project_id}/{observation_id}` - Update project observation
- `DELETE /observations/projects/{project_id}` - Delete project observation

### Project Memory

- `POST /projects/` - Create a new project container
- `GET /projects/{project_id}` - Get project details
- `PUT /projects/{project_id}` - Update project information
- `DELETE /projects/{project_id}` - Delete a project
- `GET /projects/` - List all projects
- `GET /projects/{project_id}/status` - Get project status
- `GET /projects/memories` - Get all project entities with optional project name filter

#### Project Components
- `POST /projects/{project_id}/components` - Create a component
- `GET /projects/{project_id}/components` - List components
- `PUT /projects/{project_id}/components/{component_id}` - Update component
- `DELETE /projects/{project_id}/components/{component_id}` - Delete component

### Lesson Memory

- `POST /lessons/` - Create a new lesson
- `GET /lessons/{lesson_id}` - Get lesson details
- `PUT /lessons/{lesson_id}` - Update lesson
- `DELETE /lessons/{lesson_id}` - Delete lesson
- `GET /lessons/` - List all lessons
- `GET /lessons/memories` - Get all lesson entities with optional container name filter

#### Lesson Sections
- `POST /lessons/{lesson_id}/sections` - Add lesson section
- `GET /lessons/{lesson_id}/sections` - Get lesson sections
- `PUT /lessons/{lesson_id}/sections/{section_id}` - Update section
- `DELETE /lessons/{lesson_id}/sections/{section_id}` - Delete section

### Search

- `GET /search/nodes` - Search nodes with optional entity type filtering
- `POST /search/cypher` - Execute custom Cypher queries
- `GET /search/neighborhoods/{entity_name}` - Explore entity neighborhoods
- `POST /search/paths` - Find paths between entities
- `GET /search/lessons` - Search specifically for lessons
- `GET /search/projects` - Search specifically for projects

## Example Usage

### Creating a Project with Component

```bash
# Create project
curl -X POST "http://localhost:8000/projects/" \
     -H "Content-Type: application/json" \
     -d '{
           "name": "ExampleProject",
           "description": "A sample project"
         }'

# Add component
curl -X POST "http://localhost:8000/projects/ExampleProject/components" \
     -H "Content-Type: application/json" \
     -d '{
           "name": "AuthService",
           "component_type": "microservice",
           "description": "Authentication service"
         }'
```

### Creating Lesson Relations

```bash
curl -X POST "http://localhost:8000/relations/lessons/lesson1" \
     -H "Content-Type: application/json" \
     -d '{
           "to_lesson_id": "lesson2",
           "relation_type": "BUILDS_ON",
           "properties": {"confidence": 0.9}
         }'
```

### Advanced Search

```bash
# Neighborhood exploration
curl -X GET "http://localhost:8000/search/neighborhoods/ExampleProject?max_depth=2&max_nodes=50"

# Path finding
curl -X POST "http://localhost:8000/search/paths" \
     -H "Content-Type: application/json" \
     -d '{
           "from_entity": "ProjectA",
           "to_entity": "ProjectB",
           "max_depth": 4
         }'
```

## Error Handling

The API uses standard HTTP status codes:

- 200: Success
- 400: Bad Request
- 404: Not Found
- 500: Internal Server Error

Error responses include a detail message explaining what went wrong.

## Development

### Running in Debug Mode

To run the server in development mode with auto-reload:

```bash
uv run uvicorn src.api.main:app --reload --host 0.0.0.0 --port 8000
```

### Testing

```bash
pytest src/api/tests/
```

## License

[MIT License](LICENSE) 