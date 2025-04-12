from typing import Dict, Any, Optional, List
from fastapi import APIRouter, HTTPException, Depends
from pydantic import BaseModel, Field
import json

from src.graph_memory import GraphMemoryManager
from src.api.dependencies import get_memory_manager

router = APIRouter(
    prefix="/core",
    tags=["core"]
)

class CoreNode(BaseModel):
    name: str
    type: Optional[str] = None
    description: Optional[str] = None
    properties: Optional[Dict[str, Any]] = None

class CoreUpdate(BaseModel):
    updates: Dict[str, Any] = Field(..., description="Updates to apply to the core node")

@router.get("/memories", response_model=Dict[str, Any])
async def get_all_memories(memory: GraphMemoryManager = Depends(get_memory_manager)):
    """Get all entities in the knowledge graph."""
    try:
        # Ensure manager is initialized
        if not memory.check_connection():
            raise HTTPException(status_code=503, detail="Memory system not initialized")
            
        result = memory.get_all_memories()
        return parse_response(result)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/create")
async def create_node(node: CoreNode, memory: GraphMemoryManager = Depends(get_memory_manager)):
    """Create a new node in the core memory graph."""
    try:
        # Ensure manager is initialized
        if not memory.check_connection():
            raise HTTPException(status_code=503, detail="Memory system not initialized")
            
        result = memory.create_entity(node.dict(exclude_none=True))
        return parse_response(result)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/batch")
async def create_nodes(nodes: List[CoreNode], memory: GraphMemoryManager = Depends(get_memory_manager)):
    """Create multiple nodes in the core memory graph."""
    try:
        # Ensure manager is initialized
        if not memory.check_connection():
            raise HTTPException(status_code=503, detail="Memory system not initialized")
            
        node_dicts = [n.dict(exclude_none=True) for n in nodes]
        result = memory.create_entities(node_dicts)
        return parse_response(result)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/{node_name}")
async def get_node(node_name: str, memory: GraphMemoryManager = Depends(get_memory_manager)):
    """Get a node from the core memory graph."""
    try:
        # Ensure manager is initialized
        if not memory.check_connection():
            raise HTTPException(status_code=503, detail="Memory system not initialized")
            
        result = memory.get_entity(node_name)
        return parse_response(result)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.put("/{node_name}")
async def update_node(node_name: str, update: CoreUpdate, memory: GraphMemoryManager = Depends(get_memory_manager)):
    """Update a node in the core memory graph."""
    try:
        # Ensure manager is initialized
        if not memory.check_connection():
            raise HTTPException(status_code=503, detail="Memory system not initialized")
            
        result = memory.update_entity(node_name, update.updates)
        return parse_response(result)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.delete("/{node_name}")
async def delete_node(node_name: str, memory: GraphMemoryManager = Depends(get_memory_manager)):
    """Delete a node from the core memory graph."""
    try:
        # Ensure manager is initialized
        if not memory.check_connection():
            raise HTTPException(status_code=503, detail="Memory system not initialized")
            
        result = memory.delete_entity(node_name)
        return parse_response(result)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/search/{query}")
async def search_nodes(query: str, limit: int = 10, memory: GraphMemoryManager = Depends(get_memory_manager)):
    """Search for nodes in the core memory graph."""
    try:
        # Ensure manager is initialized
        if not memory.check_connection():
            raise HTTPException(status_code=503, detail="Memory system not initialized")
            
        result = memory.search_nodes(query or "", limit)
        return parse_response(result)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/neighborhood/{node_name}")
async def get_node_neighborhood(
    node_name: str, 
    max_depth: int = 2, 
    max_nodes: int = 50, 
    memory: GraphMemoryManager = Depends(get_memory_manager)
):
    """Get the neighborhood of a node in the core memory graph."""
    try:
        # Ensure manager is initialized
        if not memory.check_connection():
            raise HTTPException(status_code=503, detail="Memory system not initialized")
            
        result = memory.search_entity_neighborhoods(node_name, max_depth, max_nodes)
        return parse_response(result)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/")
async def get_nodes(query: Optional[str] = None, limit: int = 10, memory: GraphMemoryManager = Depends(get_memory_manager)):
    """Get nodes from the core memory graph."""
    try:
        # Ensure manager is initialized
        if not memory.check_connection():
            raise HTTPException(status_code=503, detail="Memory system not initialized")
            
        # Use search_nodes instead of get_entities
        result = memory.search_nodes(query or "", limit)
        return parse_response(result)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# Helper function to parse JSON response
def parse_response(response):
    """Parse JSON response from GraphMemoryManager."""
    if isinstance(response, str):
        try:
            return json.loads(response)
        except json.JSONDecodeError:
            return {"result": response}
    elif isinstance(response, dict):
        return response
    else:
        return {"result": response} 