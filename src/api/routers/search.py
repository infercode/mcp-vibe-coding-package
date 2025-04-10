from typing import Dict, Any, List, Optional
from fastapi import APIRouter, HTTPException, Depends
from pydantic import BaseModel, Field

from src.graph_memory import GraphMemoryManager
from ..dependencies import get_memory_manager
from ..utils import parse_response

router = APIRouter(
    prefix="/search",
    tags=["search"],
    responses={404: {"description": "Not found"}}
)

class CypherQuery(BaseModel):
    """Model for executing a Cypher query."""
    query: str = Field(..., description="The Cypher query to execute")
    params: Optional[Dict[str, Any]] = Field(default=None, description="Optional parameters for the Cypher query")

class PathQuery(BaseModel):
    """Model for finding paths between entities."""
    from_entity: str = Field(..., description="The name of the starting entity")
    to_entity: str = Field(..., description="The name of the target entity")
    max_depth: int = Field(default=4, description="Maximum relationship depth to traverse")

@router.get("/nodes", response_model=Dict[str, Any])
async def search_nodes(
    query: str,
    limit: int = 10,
    entity_types: Optional[List[str]] = None,
    semantic: bool = True,
    memory: GraphMemoryManager = Depends(get_memory_manager)
):
    """
    Search for nodes in the knowledge graph.
    
    Args:
        query: The search query
        limit: Maximum number of results to return
        entity_types: List of entity types to filter by
        semantic: Whether to use semantic search (requires embeddings)
        memory: GraphMemoryManager instance
        
    Returns:
        JSON string with search results
    """
    try:
        result = memory.search_nodes(query, limit, entity_types, semantic)
        return parse_response(result)
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@router.post("/cypher", response_model=Dict[str, Any])
async def query_knowledge_graph(
    query: CypherQuery,
    memory: GraphMemoryManager = Depends(get_memory_manager)
):
    """
    Execute a custom Cypher query against the knowledge graph.
    
    Args:
        query: The Cypher query and optional parameters
        memory: GraphMemoryManager instance
        
    Returns:
        JSON string with the query results
    """
    try:
        result = memory.query_knowledge_graph(query.query, query.params)
        return parse_response(result)
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@router.get("/neighborhoods/{entity_name}", response_model=Dict[str, Any])
async def search_entity_neighborhoods(
    entity_name: str,
    max_depth: int = 2,
    max_nodes: int = 50,
    memory: GraphMemoryManager = Depends(get_memory_manager)
):
    """
    Search for entity neighborhoods (entity graph exploration).
    
    Args:
        entity_name: The name of the entity to start from
        max_depth: Maximum relationship depth to traverse
        max_nodes: Maximum number of nodes to return
        memory: GraphMemoryManager instance
        
    Returns:
        JSON string with the neighborhood graph
    """
    try:
        result = memory.search_entity_neighborhoods(entity_name, max_depth, max_nodes)
        return parse_response(result)
    except Exception as e:
        raise HTTPException(status_code=404, detail=str(e))

@router.post("/paths", response_model=Dict[str, Any])
async def find_paths_between_entities(
    path_query: PathQuery,
    memory: GraphMemoryManager = Depends(get_memory_manager)
):
    """
    Find all paths between two entities in the knowledge graph.
    
    Args:
        path_query: Query parameters for finding paths
        memory: GraphMemoryManager instance
        
    Returns:
        JSON string with all paths found
    """
    try:
        result = memory.find_paths_between_entities(
            path_query.from_entity,
            path_query.to_entity,
            path_query.max_depth
        )
        return parse_response(result)
    except Exception as e:
        raise HTTPException(status_code=404, detail=str(e))

# Specialized search endpoints for lessons and projects
@router.get("/lessons", response_model=Dict[str, Any])
async def search_lessons(
    query: str,
    limit: int = 10,
    semantic: bool = True,
    memory: GraphMemoryManager = Depends(get_memory_manager)
):
    """
    Search for lessons in the knowledge graph.
    
    Args:
        query: The search query
        limit: Maximum number of results to return
        semantic: Whether to use semantic search
        memory: GraphMemoryManager instance
        
    Returns:
        JSON string with search results
    """
    try:
        result = memory.search_nodes(query, limit, entity_types=["lesson"], semantic=semantic)
        return parse_response(result)
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@router.get("/projects", response_model=Dict[str, Any])
async def search_projects(
    query: str,
    limit: int = 10,
    semantic: bool = True,
    memory: GraphMemoryManager = Depends(get_memory_manager)
):
    """
    Search for projects in the knowledge graph.
    
    Args:
        query: The search query
        limit: Maximum number of results to return
        semantic: Whether to use semantic search
        memory: GraphMemoryManager instance
        
    Returns:
        JSON string with search results
    """
    try:
        result = memory.search_nodes(query, limit, entity_types=["project"], semantic=semantic)
        return parse_response(result)
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e)) 