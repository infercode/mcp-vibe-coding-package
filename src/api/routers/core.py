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
        result = memory.get_all_memories()
        return json.loads(result)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/create")
async def create_node(node: CoreNode, memory: GraphMemoryManager = Depends(get_memory_manager)):
    """Create a new node in the core memory graph."""
    try:
        result = memory.create_entity(node.dict(exclude_none=True))
        return json.loads(result)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/batch")
async def create_nodes(nodes: List[CoreNode], memory: GraphMemoryManager = Depends(get_memory_manager)):
    """Create multiple nodes in the core memory graph."""
    try:
        node_dicts = [n.dict(exclude_none=True) for n in nodes]
        result = memory.create_entities(node_dicts)
        return json.loads(result)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/{node_name}")
async def get_node(node_name: str, memory: GraphMemoryManager = Depends(get_memory_manager)):
    """Get a node from the core memory graph."""
    try:
        result = memory.get_entity(node_name)
        return json.loads(result)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.put("/{node_name}")
async def update_node(node_name: str, update: CoreUpdate, memory: GraphMemoryManager = Depends(get_memory_manager)):
    """Update a node in the core memory graph."""
    try:
        result = memory.update_entity(node_name, update.updates)
        return json.loads(result)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.delete("/{node_name}")
async def delete_node(node_name: str, memory: GraphMemoryManager = Depends(get_memory_manager)):
    """Delete a node from the core memory graph."""
    try:
        result = memory.delete_entity(node_name)
        return json.loads(result)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/search/{query}")
async def search_nodes(query: str, limit: int = 10, memory: GraphMemoryManager = Depends(get_memory_manager)):
    """Search for nodes in the core memory graph."""
    try:
        result = memory.search_entities(query or "", limit)
        return json.loads(result)
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
        result = memory.search_entity_neighborhoods(node_name, max_depth, max_nodes)
        return json.loads(result)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/")
async def get_nodes(query: Optional[str] = None, limit: int = 10, memory: GraphMemoryManager = Depends(get_memory_manager)):
    """Get nodes from the core memory graph."""
    try:
        result = memory.get_entities(query, limit)
        return {"nodes": result}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e)) 