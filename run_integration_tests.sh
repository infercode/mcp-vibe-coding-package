#!/bin/bash

# Run Neo4j integration tests for GraphMemoryManager
# This script sets up the Python path and runs the integration tests

set -e  # Exit on error

# Configuration - update this with your Neo4j password
NEO4J_PASSWORD="P@ssW0rd2025!"

# Check if Neo4j is running
echo "Checking Neo4j connection..."
curl -s http://localhost:7474 > /dev/null
if [ $? -ne 0 ]; then
    echo "Error: Neo4j doesn't seem to be running at http://localhost:7474"
    echo "Please make sure Neo4j is running before executing the tests."
    exit 1
fi

# Set up environment variables
export NEO4J_URI="bolt://localhost:7687"
export NEO4J_USER="neo4j"
export NEO4J_PASSWORD="$NEO4J_PASSWORD"
export EMBEDDER_PROVIDER="none"  # Disable embeddings for the test
export PYTHONPATH=.

echo "Configuration:"
echo "  Neo4j URI: $NEO4J_URI"
echo "  Neo4j User: $NEO4J_USER"
echo "  Password: [configured]"
echo "  Embedder: $EMBEDDER_PROVIDER"

echo "Running integration tests..."
poetry run pytest -xvs tests/graph_memory/test_graph_memory_integration.py "$@"

# If we get here, the tests completed
echo "Integration tests completed." 