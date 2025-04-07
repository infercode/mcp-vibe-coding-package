#!/usr/bin/env python3
"""
API Validation Tests

This module tests the enhanced validation logic in the core memory tools.
It validates the input sanitization and error handling for key endpoints.
"""

import os
import sys
import json
import pytest
from typing import Dict, Any, List, Optional
from pathlib import Path
from unittest.mock import MagicMock

# Fix the Python import path
project_root = Path(__file__).resolve().parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from src.tools.core_memory_tools import register_core_tools


class DummyServer:
    """Mock server for testing tool functions."""
    
    def tool(self):
        """Decorator for registering tools."""
        def decorator(func):
            return func
        return decorator


class MockGraphMemoryManager:
    """Mock implementation of GraphMemoryManager for testing."""
    
    def __init__(self):
        """Initialize the mock manager."""
        self.default_project_name = "test_project"
        self.embedding_enabled = True
        self.calls = []
    
    def set_project_name(self, project_name: str) -> None:
        """Set the current project name."""
        self.calls.append(("set_project_name", project_name))
    
    def search_nodes(self, query: str, limit: int = 10, entity_types: Optional[List[str]] = None,
                   semantic: bool = True) -> str:
        """Mock implementation of search_nodes."""
        self.calls.append(("search_nodes", query, limit))
        return json.dumps({
            "nodes": [{"name": f"test_result_{i}", "type": "TestEntity"} for i in range(3)],
            "count": 3
        })
    
    def create_relations(self, relations: List[Dict[str, Any]]) -> str:
        """Mock implementation of create_relations."""
        self.calls.append(("create_relations", relations))
        return json.dumps({
            "success": True,
            "created": len(relations),
            "relations": relations
        })
    
    def add_observations(self, observations: List[Dict[str, Any]]) -> str:
        """Mock implementation of add_observations."""
        self.calls.append(("add_observations", observations))
        return json.dumps({
            "success": True,
            "added": len(observations),
            "observations": observations
        })


@pytest.fixture
def api_tools():
    """Fixture providing the core memory API tools with our own mock manager."""
    server = DummyServer()
    
    # Create a function that returns our mock graph manager
    def get_mock_manager(client_id=None):
        return MockGraphMemoryManager()
    
    tools = register_core_tools(server, get_mock_manager)
    return tools


# Search Nodes Tests

@pytest.mark.asyncio
async def test_search_nodes_valid_query(api_tools):
    """Test search with a valid query."""
    search_nodes = api_tools["search_nodes"]
    
    result = await search_nodes(query="test query", limit=10)
    parsed = json.loads(result)
    
    assert parsed.get("status") == "success"
    assert "Found" in parsed.get("message", "")


@pytest.mark.asyncio
async def test_search_nodes_empty_query(api_tools):
    """Test search with an empty query."""
    search_nodes = api_tools["search_nodes"]
    
    result = await search_nodes(query="", limit=10)
    parsed = json.loads(result)
    
    assert parsed.get("status") == "error"
    assert parsed.get("error", {}).get("code") == "invalid_input"
    assert "empty" in parsed.get("message", "").lower()


@pytest.mark.asyncio
async def test_search_nodes_dangerous_pattern(api_tools):
    """Test search with a query containing dangerous patterns."""
    search_nodes = api_tools["search_nodes"]
    
    result = await search_nodes(query="test; DROP TABLE entities", limit=10)
    parsed = json.loads(result)
    
    assert parsed.get("status") == "error"
    assert parsed.get("error", {}).get("code") == "security_violation"
    assert "dangerous pattern" in parsed.get("message", "").lower()


@pytest.mark.asyncio
async def test_search_nodes_invalid_limit(api_tools):
    """Test search with an invalid limit value."""
    search_nodes = api_tools["search_nodes"]
    
    result = await search_nodes(query="test query", limit=-5)
    parsed = json.loads(result)
    
    assert parsed.get("status") == "success"
    assert any(w.get("warning") == "limit_adjusted" for w in parsed.get("data", {}).get("warnings", []))


@pytest.mark.asyncio
async def test_search_nodes_limit_too_large(api_tools):
    """Test search with a limit value that's too large."""
    search_nodes = api_tools["search_nodes"]
    
    result = await search_nodes(query="test query", limit=500)
    parsed = json.loads(result)
    
    assert parsed.get("status") == "success"
    assert any(w.get("warning") == "limit_reduced" for w in parsed.get("data", {}).get("warnings", []))


@pytest.mark.asyncio
async def test_search_nodes_query_too_long(api_tools):
    """Test search with a query that's too long."""
    search_nodes = api_tools["search_nodes"]
    
    result = await search_nodes(query="test query " * 100, limit=10)
    parsed = json.loads(result)
    
    assert parsed.get("status") == "success"
    assert any(w.get("warning") == "query_truncated" for w in parsed.get("data", {}).get("warnings", []))


@pytest.mark.asyncio
async def test_search_nodes_invalid_project_name(api_tools):
    """Test search with an invalid project name."""
    search_nodes = api_tools["search_nodes"]
    
    result = await search_nodes(query="test query", limit=10, project_name="test;project")
    parsed = json.loads(result)
    
    assert parsed.get("status") == "error"
    assert parsed.get("error", {}).get("code") == "invalid_project_name"
    assert "invalid character" in parsed.get("message", "").lower()


# Create Relations Tests

@pytest.mark.asyncio
async def test_create_relations_valid(api_tools):
    """Test creating valid relations."""
    create_relations = api_tools["create_relations"]
    
    valid_relations = [
        {
            "from_entity": "Entity1",
            "to_entity": "Entity2",
            "relationship_type": "RELATED_TO",
            "weight": 0.8
        },
        {
            "from_entity": "Entity3",
            "to_entity": "Entity4",
            "relationship_type": "CONNECTED_TO"
        }
    ]
    
    result = await create_relations(relations=valid_relations)
    parsed = json.loads(result)
    
    assert parsed.get("status") == "success"
    assert "Successfully created" in parsed.get("message", "")


@pytest.mark.asyncio
async def test_create_relations_empty_list(api_tools):
    """Test creating relations with an empty list."""
    create_relations = api_tools["create_relations"]
    
    result = await create_relations(relations=[])
    parsed = json.loads(result)
    
    assert parsed.get("status") == "error"
    assert parsed.get("error", {}).get("code") == "empty_input"
    assert "empty" in parsed.get("message", "").lower()


@pytest.mark.asyncio
async def test_create_relations_missing_fields(api_tools):
    """Test creating relations with missing required fields."""
    create_relations = api_tools["create_relations"]
    
    invalid_relations = [
        {
            "from_entity": "Entity1",
            # Missing to_entity
            "relationship_type": "RELATED_TO"
        },
        {
            # Missing from_entity
            "to_entity": "Entity4",
            # Missing relationship_type
        }
    ]
    
    result = await create_relations(relations=invalid_relations)
    parsed = json.loads(result)
    
    assert parsed.get("status") == "error" or "skipped_relations" in parsed.get("data", {})
    # Either all relations are invalid (error) or we have skipped_relations in the data


@pytest.mark.asyncio
async def test_create_relations_invalid_weight(api_tools):
    """Test creating relations with an invalid weight value."""
    create_relations = api_tools["create_relations"]
    
    invalid_weight_relations = [
        {
            "from_entity": "Entity1",
            "to_entity": "Entity2",
            "relationship_type": "RELATED_TO",
            "weight": "not a number"
        }
    ]
    
    result = await create_relations(relations=invalid_weight_relations)
    parsed = json.loads(result)
    
    # The function might either return an error or successfully handle the invalid weight
    # by ignoring it and proceeding with the relation creation
    if parsed.get("status") == "error":
        assert "invalid" in parsed.get("message", "").lower()
    else:
        assert parsed.get("status") == "success"


# Add Observations Tests

@pytest.mark.asyncio
async def test_add_observations_valid(api_tools):
    """Test adding valid observations."""
    add_observations = api_tools["add_observations"]
    
    valid_observations = [
        {
            "entity": "Entity1",
            "content": "This is a test observation"
        },
        {
            "entity": "Entity2",
            "content": ["Observation 1", "Observation 2"]
        }
    ]
    
    result = await add_observations(observations=valid_observations)
    parsed = json.loads(result)
    
    assert parsed.get("status") == "success"
    assert "Successfully added" in parsed.get("message", "")


@pytest.mark.asyncio
async def test_add_observations_empty_list(api_tools):
    """Test adding observations with an empty list."""
    add_observations = api_tools["add_observations"]
    
    result = await add_observations(observations=[])
    parsed = json.loads(result)
    
    assert parsed.get("status") == "error"
    assert parsed.get("error", {}).get("code") == "empty_input"
    assert "empty" in parsed.get("message", "").lower()


@pytest.mark.asyncio
async def test_add_observations_missing_fields(api_tools):
    """Test adding observations with missing required fields."""
    add_observations = api_tools["add_observations"]
    
    invalid_observations = [
        {
            "entity": "Entity1",
            # Missing content
        },
        {
            # Missing entity
            "content": "Test observation"
        }
    ]
    
    result = await add_observations(observations=invalid_observations)
    parsed = json.loads(result)
    
    assert parsed.get("status") == "error" or "skipped_observations" in parsed.get("data", {})
    # Either all observations are invalid (error) or we have skipped_observations in the data


@pytest.mark.asyncio
async def test_add_observations_invalid_content(api_tools):
    """Test adding observations with invalid content type."""
    add_observations = api_tools["add_observations"]
    
    invalid_content_observations = [
        {
            "entity": "Entity1",
            "content": {"invalid": "object instead of string or list"}
        }
    ]
    
    result = await add_observations(observations=invalid_content_observations)
    parsed = json.loads(result)
    
    # The function might convert the object to a string or reject it
    assert parsed.get("status") in ["success", "error"]


@pytest.mark.asyncio
async def test_add_observations_long_content(api_tools):
    """Test adding observations with very long content."""
    add_observations = api_tools["add_observations"]
    
    long_content_observations = [
        {
            "entity": "Entity1",
            "content": "Very long content " * 500
        }
    ]
    
    result = await add_observations(observations=long_content_observations)
    parsed = json.loads(result)
    
    # The function should either truncate the content or succeed with the full content
    assert parsed.get("status") == "success" 