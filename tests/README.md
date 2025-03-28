# Graph Memory Tests

This directory contains tests for the Graph Memory system. The tests are organized by module and functionality.

## Test Structure

- `conftest.py` - Contains common fixtures used across tests
- `/graph_memory/` - Tests for core graph memory components
- `/lesson_memory/` - Tests for lesson management functionality
- `/project_memory/` - Tests for project management functionality  
- `/integration/` - Integration tests covering multiple components

## Running Tests

To run all tests:

```bash
pytest
```

To run a specific test file:

```bash
pytest tests/graph_memory/test_entity_manager.py
```

To run a specific test function:

```bash
pytest tests/graph_memory/test_entity_manager.py::test_create_entity
```

To run tests with coverage:

```bash
pytest --cov=src
```

## Fixtures

Common fixtures are defined in `conftest.py` and are available to all test modules:

- `mock_logger` - Mock logger for testing
- `mock_neo4j_driver` - Mock Neo4j driver that returns a mock session
- `mock_base_manager` - Mock base manager with Neo4j driver
- `mock_entity_manager` - Mock entity manager
- `mock_relation_manager` - Mock relation manager
- `mock_observation_manager` - Mock observation manager
- `mock_embedding_adapter` - Mock embedding adapter
- `mock_search_manager` - Mock search manager
- `mock_lesson_manager` - Mock lesson manager
- `mock_project_manager` - Mock project manager
- `mock_graph_memory_manager` - Mock graph memory manager with all component managers

## Test Organization

The tests are organized by component, with each component having its own test file:

- `test_base_manager.py` - Tests for the BaseManager class
- `test_entity_manager.py` - Tests for the EntityManager class
- `test_relation_manager.py` - Tests for the RelationManager class
- `test_observation_manager.py` - Tests for the ObservationManager class
- `test_search_manager.py` - Tests for the SearchManager class
- `test_embedding_adapter.py` - Tests for the EmbeddingAdapter class
- `test_graph_memory_manager.py` - Tests for the GraphMemoryManager class
- `test_lesson_memory_manager.py` - Tests for the LessonMemoryManager class
- `test_project_memory_manager.py` - Tests for the ProjectMemoryManager class

Integration tests are located in `tests/integration/test_integration.py` and cover end-to-end functionality of multiple components working together.

## Mocking Approach

The tests use the `unittest.mock` library to mock external dependencies and isolate the components being tested. Each component is tested in isolation with its dependencies mocked.

For integration tests, the components are mocked but the interactions between them are tested to ensure they work together correctly.

## Adding New Tests

When adding new tests:

1. Identify the component to test
2. Create a new test file if necessary
3. Use the appropriate fixtures from `conftest.py`
4. Write test functions that test specific functionality
5. Mock any external dependencies
6. Use assertions to verify expected behavior 