import json
import pytest
from unittest.mock import MagicMock, patch
from src.tools.core_memory_tools import register_core_tools


class MockServer:
    def __init__(self):
        self.tools = {}
    
    def tool(self):
        def decorator(func):
            self.tools[func.__name__] = func
            return func
        return decorator


class MockClientManager:
    def __init__(self):
        self.default_project_name = "test-project"
    
    def get_all_memories(self, project_name=None):
        return json.dumps({
            "status": "success",
            "memories": [
                {"name": "Entity1", "type": "Person"},
                {"name": "Entity2", "type": "Location"}
            ]
        })
    
    def delete_all_memories(self, project_name=None, **kwargs):
        return json.dumps({
            "status": "success",
            "message": "All memories deleted"
        })
    
    def debug_dump_neo4j(self, limit=100, query=None):
        return json.dumps({
            "nodes": [{"name": "entity1"}],
            "relationships": [{"type": "relation1"}],
            "statistics": {"node_count": 1, "relationship_count": 1}
        })
    
    def set_project_name(self, project_name):
        self.default_project_name = project_name


@pytest.fixture
def mock_server():
    return MockServer()


@pytest.fixture
def mock_client_manager():
    return MockClientManager()


@pytest.fixture
def core_tools(mock_server):
    def get_mock_client_manager(client_id=None):
        return MockClientManager()
    
    return register_core_tools(mock_server, get_mock_client_manager)


@pytest.mark.asyncio
async def test_get_all_memories_basic(core_tools):
    """Test get_all_memories with basic parameters"""
    result = await core_tools["get_all_memories"]()
    result_obj = json.loads(result)
    
    assert "status" in result_obj
    assert result_obj["status"] == "success"
    assert "memories" in result_obj
    assert len(result_obj["memories"]) == 2


@pytest.mark.asyncio
async def test_get_all_memories_with_parameters(core_tools):
    """Test get_all_memories with custom parameters"""
    result = await core_tools["get_all_memories"](
        random_string="test",
        client_id="test-client",
        limit=10,
        offset=0,
        project_name="test-project"
    )
    result_obj = json.loads(result)
    
    assert result_obj["status"] == "success"
    assert "memories" in result_obj


@pytest.mark.asyncio
async def test_get_all_memories_error_handling(core_tools):
    """Test get_all_memories error handling with invalid parameters"""
    result = await core_tools["get_all_memories"](limit="not-a-number")
    result_obj = json.loads(result)
    
    assert result_obj["status"] == "error"
    assert "code" in result_obj
    assert result_obj["code"] == "invalid_input"


@pytest.mark.asyncio
async def test_delete_all_memories_valid(core_tools):
    """Test delete_all_memories with valid parameters"""
    result = await core_tools["delete_all_memories"](
        random_string="CONFIRM_DELETE_20230101",
        project_name="test-project",
        double_confirm=True
    )
    result_obj = json.loads(result)
    
    assert result_obj["status"] == "success"
    assert "message" in result_obj
    assert "deleted" in result_obj["message"].lower()


@pytest.mark.asyncio
async def test_delete_all_memories_missing_confirmation(core_tools):
    """Test delete_all_memories without confirmation flag"""
    result = await core_tools["delete_all_memories"](
        random_string="CONFIRM_DELETE_20230101"
    )
    result_obj = json.loads(result)
    
    assert result_obj["status"] == "error"
    assert result_obj["code"] == "confirmation_required"


@pytest.mark.asyncio
async def test_delete_all_memories_invalid_string(core_tools):
    """Test delete_all_memories with invalid random string"""
    result = await core_tools["delete_all_memories"](
        random_string="invalid-string",
        double_confirm=True
    )
    result_obj = json.loads(result)
    
    assert result_obj["status"] == "error"
    assert "16 characters" in result_obj["message"] or "CONFIRM_DELETE_" in result_obj["message"]


@pytest.mark.asyncio
async def test_debug_dump_neo4j_valid(core_tools):
    """Test debug_dump_neo4j with valid parameters"""
    result = await core_tools["debug_dump_neo4j"](
        limit=50,
        confirm=True
    )
    result_obj = json.loads(result)
    
    assert "nodes" in result_obj
    assert "relationships" in result_obj
    assert "statistics" in result_obj


@pytest.mark.asyncio
async def test_debug_dump_neo4j_missing_confirmation(core_tools):
    """Test debug_dump_neo4j without confirmation flag"""
    result = await core_tools["debug_dump_neo4j"](limit=50)
    result_obj = json.loads(result)
    
    assert result_obj["status"] == "error"
    assert result_obj["code"] == "confirmation_required"


@pytest.mark.asyncio
async def test_debug_dump_neo4j_full_parameters(core_tools):
    """Test debug_dump_neo4j with all parameters"""
    result = await core_tools["debug_dump_neo4j"](
        limit=50,
        confirm=True,
        client_id="test-client",
        include_relationships=True,
        include_statistics=True
    )
    result_obj = json.loads(result)
    
    assert "nodes" in result_obj
    assert "relationships" in result_obj
    assert "statistics" in result_obj
    assert len(result_obj["relationships"]) == 1 