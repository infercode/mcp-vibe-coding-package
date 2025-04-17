# Testing the Neo4j Graph Memory System

This directory contains tests for the Neo4j Graph Memory System. The tests are organized into unit tests and integration tests.

## Prerequisites

- Python 3.11+
- pytest
- pytest-asyncio
- Running Neo4j database (for integration tests)

## Test Structure

- `conftest.py`: Contains shared fixtures and utilities for tests
- `unit/`: Contains unit tests for individual components
  - `graph_memory/`: Tests for the graph memory system
  - `lesson_memory/`: Tests for the lesson memory system
  - `project_memory/`: Tests for the project memory system
  - `api/`: Tests for the API endpoints
  - `models/`: Tests for the data models
  - `registry/`: Tests for the function registry
  - `tools/`: Tests for the MCP tools
  - `utils/`: Tests for utility functions
- `integration/`: Contains integration tests for end-to-end functionality
- `test_session_manager.py`: Tests for the session manager
- `test_embedding_manager.py`: Tests for the embedding manager

## Running Tests

### Using uv (Recommended)

With uv installed, you can run the tests with:

```bash
uv run python -m pytest tests/
```

To run specific test files:

```bash
uv run python -m pytest tests/unit/graph_memory/
uv run python -m pytest tests/test_session_manager.py
```

To run tests with detailed output:

```bash
uv run python -m pytest tests/ -v
```

### Configuration

Tests use the settings in `.env` by default. You can create a test-specific `.env.test` file to use different settings for testing.

### Skipping Integration Tests

Integration tests require a running Neo4j database. If you don't have Neo4j running, you can skip these tests:

```bash
uv run python -m pytest tests/ -k "not integration"
```

### Test Coverage

To run tests with coverage:

```bash
uv run python -m pytest --cov=src tests/
```

To generate a coverage report:

```bash
uv run python -m pytest --cov=src --cov-report=html tests/
```

This will create an HTML report in the `htmlcov` directory.

## Writing Tests

### Unit Tests

- Focus on testing a single unit of functionality
- Use fixtures from `conftest.py` where possible
- Use mocks to avoid external dependencies
- Follow the pattern of other unit tests

### Integration Tests

- Test interactions between components
- Requires a running Neo4j database
- Focus on end-to-end functionality
- Use proper cleanup to maintain test isolation

## Test Environment

Tests will use the Neo4j connection information from the `.env` file. For testing purposes, consider using a dedicated test database to avoid affecting your development data. 