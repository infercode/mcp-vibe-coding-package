#!/bin/bash

# Debug version of Neo4j integration tests with detailed logging

echo "Setting Neo4j credentials..."
export NEO4J_PASSWORD="P@ssW0rd2025!"
export NEO4J_URI="bolt://localhost:7687"
export NEO4J_USER="neo4j"
export PYTHONPATH=.

echo "Checking Neo4j connection..."
echo "Using URI: $NEO4J_URI"
echo "Using Username: $NEO4J_USER"
echo "Password: [configured]"

# Test Neo4j connection with a simple command
echo "Testing Neo4j connection with direct Python script..."
poetry run python -c "
import sys
try:
    from neo4j import GraphDatabase
    print('Neo4j driver imported successfully')
    driver = GraphDatabase.driver('$NEO4J_URI', auth=('$NEO4J_USER', '$NEO4J_PASSWORD'))
    print('Driver created successfully')
    with driver.session() as session:
        result = session.run('RETURN 1 as test')
        for record in result:
            print(f'Connection test: {record[\"test\"]}')
    driver.close()
    print('Neo4j connection successful')
except Exception as e:
    print(f'Neo4j connection error: {str(e)}', file=sys.stderr)
    sys.exit(1)
"

if [ $? -ne 0 ]; then
    echo "Failed to connect to Neo4j. Please check your Neo4j instance and credentials."
    exit 1
fi

echo "Running integration tests with debug flags..."
poetry run pytest -xvs tests/graph_memory/test_graph_memory_integration.py

# If we get here, the tests completed
echo "Integration tests completed." 