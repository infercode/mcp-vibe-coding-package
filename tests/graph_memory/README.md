# Neo4j Integration Tests for GraphMemoryManager

This directory contains integration tests that verify the GraphMemoryManager works correctly with a real Neo4j database. These tests are designed to create, query, and manipulate actual data in Neo4j, then clean up afterward.

## Test Setup

The tests are designed with the following characteristics:

1. **Isolation**: Each test run generates a unique prefix for all entities to prevent conflicts with existing data
2. **Cleanup**: All test data is automatically removed after tests complete
3. **Comprehensive Coverage**: Tests cover all major functionality of GraphMemoryManager
4. **Real Database**: Tests use an actual Neo4j instance to ensure real-world compatibility

## Prerequisites

Before running the integration tests, ensure you have:

1. A running Neo4j instance (local or remote)
2. Proper environment variables configured:
   - `NEO4J_URI` (default: bolt://localhost:7687)
   - `NEO4J_USER` (default: neo4j)
   - `NEO4J_PASSWORD` (default: password)
   - `OPENAI_API_KEY` (required for embedding-based tests)

## Running the Tests

### Using the Convenience Scripts

We provide two convenience scripts to run the integration tests:

**On Unix/Linux/macOS**:
```bash
./run_integration_tests.sh
```

**On Windows**:
```cmd
run_integration_tests.bat
```

These scripts check if Neo4j is running, set up the proper Python path, and run the tests.

### Manual Running

You can also run the tests manually:

```bash
# Set the Python path to include the project root
PYTHONPATH=. poetry run pytest -xvs tests/graph_memory/test_graph_memory_integration.py
```

## Test Cases

The integration tests include the following key test cases:

1. **Entity Operations**
   - Creating individual entities
   - Creating batches of entities
   - Retrieving entities

2. **Relationship Operations**
   - Creating relationships between entities
   - Querying relationships

3. **Observation Operations**
   - Adding observations to entities
   - Querying observations

4. **Search Functionality**
   - Semantic search across entities
   - Full text search

5. **Project Container Operations**
   - Creating project containers
   - Retrieving project information

6. **Lesson Container Operations**
   - Creating lesson containers
   - Creating sections within lessons
   - Creating relationships between lessons

7. **Complex Knowledge Graph**
   - Building a more comprehensive knowledge graph with multiple entity types
   - Creating a network of relationships
   - Adding observations to entities
   - Verifying relationship traversal

## Troubleshooting

If the tests fail, check the following:

1. **Neo4j Connection**: Ensure Neo4j is running and accessible with the provided credentials
2. **API Keys**: For embedding-based tests, ensure your OpenAI API key is valid
3. **Permission Issues**: Ensure the Neo4j user has write permissions
4. **Database State**: If the tests were interrupted, some test data might remain in the database

## Adding New Tests

When adding new integration tests:

1. Use the `TEST_PREFIX` variable to create unique entity names
2. Ensure your test cleans up after itself or relies on the fixture cleanup
3. Keep tests independent of each other
4. Consider the performance impact of large test data sets 