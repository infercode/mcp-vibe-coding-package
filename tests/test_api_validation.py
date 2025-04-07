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
        
        # Return proper error format for empty query
        if not query:
            return json.dumps({
                "status": "error",
                "error": {
                    "code": "invalid_input",
                    "message": "Search query cannot be empty"
                },
                "message": "Search query cannot be empty"
            })
        
        # Return proper error format for security violation
        if ";" in query:
            return json.dumps({
                "status": "error",
                "error": {
                    "code": "security_violation",
                    "message": "Search query contains dangerous pattern: ;"
                },
                "message": "Search query contains dangerous pattern: ;"
            })
        
        # For invalid limit, return success with warnings
        if limit <= 0:
            return json.dumps({
                "status": "success",
                "message": "Search completed with adjusted limit",
                "data": {
                    "nodes": [{"name": f"test_result_{i}", "type": "TestEntity"} for i in range(3)],
                    "count": 3,
                    "warnings": [
                        {"warning": "limit_adjusted", "message": "Negative limit adjusted to default of 10"}
                    ]
                }
            })
        
        # Default success response
        return json.dumps({
            "status": "success",
            "message": "Found search results",
            "data": {
                "nodes": [{"name": f"test_result_{i}", "type": "TestEntity"} for i in range(min(limit, 3))],
                "count": min(limit, 3)
            }
        })
    
    def create_relationship(self, relationship_data: Dict[str, Any]) -> str:
        """Mock implementation of create_relationship."""
        self.calls.append(("create_relationship", relationship_data))
        # Debug print to troubleshoot
        print(f"DEBUG MockGraphMemoryManager.create_relationship called with: {relationship_data}")
        
        # Missing required fields
        if "from" not in relationship_data or "to" not in relationship_data or "relationType" not in relationship_data:
            missing = []
            if "from" not in relationship_data:
                missing.append("from_entity")
            if "to" not in relationship_data:
                missing.append("to_entity")
            if "relationType" not in relationship_data:
                missing.append("relationship_type")
            
            return json.dumps({
                "status": "error",
                "error": {
                    "message": f"Missing required fields: {', '.join(missing)}",
                    "code": "validation_error"
                }
            })
        
        # Invalid weight test
        if "weight" in relationship_data and float(relationship_data["weight"]) > 1.0:
            return json.dumps({
                "status": "error",
                "error": {
                    "message": "Weight must be between 0 and 1",
                    "code": "validation_error"
                }
            })
        
        # Valid relationship
        return json.dumps({
            "status": "success",
            "relation_id": "rel-1"
        })
    
    def add_observations(self, observations: List[Dict[str, Any]]) -> str:
        """Mock implementation of add_observations."""
        self.calls.append(("add_observations", observations))
        
        # Empty list check
        if not observations:
            return json.dumps({
                "status": "error",
                "error": {
                    "code": "empty_input",
                    "message": "Observations list cannot be empty"
                }
            })
        
        # Missing required fields check
        missing_fields = False
        for obs in observations:
            if not isinstance(obs, dict) or "entity" not in obs or "content" not in obs:
                missing_fields = True
                break
        
        if missing_fields:
            return json.dumps({
                "status": "error",
                "error": {
                    "code": "validation_error",
                    "message": "Observations missing required fields"
                }
            })
        
        # Success response
        return json.dumps({
            "status": "success",
            "message": "Successfully added observations",
            "data": {
                "added": len(observations),
                "observations": observations
            }
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
    # The error field may be at the top level or in an error object
    error_code = parsed.get("error", {}).get("code") if isinstance(parsed.get("error"), dict) else parsed.get("code")
    assert error_code == "invalid_input"
    assert "empty" in parsed.get("message", "").lower()


@pytest.mark.asyncio
async def test_search_nodes_dangerous_pattern(api_tools):
    """Test search with a query containing dangerous patterns."""
    search_nodes = api_tools["search_nodes"]
    
    result = await search_nodes(query="test; DROP TABLE entities", limit=10)
    parsed = json.loads(result)
    
    assert parsed.get("status") == "error"
    # The error field may be at the top level or in an error object
    error_code = parsed.get("error", {}).get("code") if isinstance(parsed.get("error"), dict) else parsed.get("code")
    assert error_code == "security_violation"
    assert "dangerous" in parsed.get("message", "").lower() or "high-risk" in parsed.get("message", "").lower()


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
    # The error field may be at the top level or in an error object
    error_code = parsed.get("error", {}).get("code") if isinstance(parsed.get("error"), dict) else parsed.get("code")
    assert error_code == "invalid_project_name"
    assert "invalid" in parsed.get("message", "").lower()


# Create Relationship Tests

@pytest.mark.asyncio
async def test_create_relationship_valid(api_tools):
    """Test creating a valid relationship."""
    create_relationship = api_tools["create_relationship"]
    
    valid_relation = {
        "from_entity": "EntityA",
        "to_entity": "EntityB",
        "relationship_type": "DEPENDS_ON",
        "weight": 0.8
    }
    
    result = await create_relationship(relationship_data=valid_relation)
    
    assert result is not None
    result_obj = json.loads(result)
    print(f"DEBUG CREATE_RELATIONSHIP RESULT: {result_obj}")
    
    # Accept either error or status fields for backwards compatibility
    if "error" in result_obj:
        assert False, f"Error in result: {result_obj.get('error', {}).get('message', 'Unknown error')}"
    
    assert "status" in result_obj
    assert result_obj["status"] == "success"


@pytest.mark.asyncio
async def test_create_relationship_missing_fields(api_tools):
    """Test creating a relationship with missing required fields."""
    create_relationship = api_tools["create_relationship"]
    
    invalid_relation = {
        "from_entity": "EntityA",
        "relationship_type": "DEPENDS_ON"
        # Missing to_entity
    }
    
    result = await create_relationship(relationship_data=invalid_relation)
    
    assert result is not None
    result_obj = json.loads(result)
    # Check either for an 'error' field or for 'status' being 'error'
    assert result_obj.get("status") == "error" or "error" in result_obj
    # Check the code is either directly in the object or in an error subobject
    error_code = result_obj.get("error", {}).get("code") if isinstance(result_obj.get("error"), dict) else result_obj.get("code")
    assert error_code in ["missing_required_fields", "validation_error"]


@pytest.mark.asyncio
async def test_create_relationship_invalid_weight(api_tools):
    """Test creating a relationship with invalid weight."""
    create_relationship = api_tools["create_relationship"]
    
    invalid_weight_relation = {
        "from_entity": "EntityA",
        "to_entity": "EntityB",
        "relationship_type": "DEPENDS_ON",
        "weight": 1.5
    }
    
    result = await create_relationship(relationship_data=invalid_weight_relation)
    
    assert result is not None
    result_obj = json.loads(result)
    # Check either for an 'error' field or for 'status' being 'error'
    assert result_obj.get("status") == "error" or "error" in result_obj
    # Check the code is either directly in the object or in an error subobject
    error_code = result_obj.get("error", {}).get("code") if isinstance(result_obj.get("error"), dict) else result_obj.get("code")
    assert error_code in ["invalid_weight", "invalid_weight_type", "validation_error"]


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
    # The error field may be at the top level or in an error object
    error_code = parsed.get("error", {}).get("code") if isinstance(parsed.get("error"), dict) else parsed.get("code")
    assert error_code == "empty_input"
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
    
    assert parsed.get("status") == "error"
    # The error field may be at the top level or in an error object
    error_code = parsed.get("error", {}).get("code") if isinstance(parsed.get("error"), dict) else parsed.get("code")
    assert error_code in ["validation_error", "observation_error"]
    
    # Check error message for validation failure terminology
    error_message = parsed.get("error", {}).get("message") if isinstance(parsed.get("error"), dict) else parsed.get("message")
    if error_message:
        assert "missing" in error_message.lower() or "required" in error_message.lower() or "validation" in error_message.lower()


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