@echo off
REM Run Neo4j integration tests for GraphMemoryManager
REM This script sets up the Python path and runs the integration tests

echo Checking Neo4j connection...
curl -s http://localhost:7474 >nul 2>&1
if %ERRORLEVEL% NEQ 0 (
    echo Error: Neo4j doesn't seem to be running at http://localhost:7474
    echo Please make sure Neo4j is running before executing the tests.
    exit /b 1
)

echo Running integration tests...
set PYTHONPATH=.
poetry run pytest -xvs tests/graph_memory/test_graph_memory_integration.py %*

REM If we get here, the tests completed
echo Integration tests completed. 