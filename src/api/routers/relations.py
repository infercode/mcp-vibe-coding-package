from typing import Dict, Any, List, Optional
from fastapi import APIRouter, HTTPException, Depends
from pydantic import BaseModel, Field

from src.graph_memory import GraphMemoryManager
from ..dependencies import get_memory_manager
from ..utils import parse_response

router = APIRouter(
    prefix="/relations",
    tags=["relations"],
    responses={404: {"description": "Not found"}}
)

class Relationship(BaseModel):
    """Model for a relationship between entities."""
    from_entity: str = Field(..., description="Source entity name")
    to_entity: str = Field(..., description="Target entity name")
    relation_type: str = Field(..., description="Type of relationship")
    properties: Optional[Dict[str, Any]] = Field(default=None, description="Optional properties for the relationship")

class RelationshipUpdate(BaseModel):
    """Model for updating a relationship."""
    updates: Dict[str, Any] = Field(..., description="Updates to apply to the relationship")

class LessonRelationship(BaseModel):
    """Model for a relationship between lesson entities."""
    to_lesson_id: str = Field(..., description="Target lesson ID")
    relation_type: str = Field(..., description="Type of relationship (e.g., BUILDS_ON, SUPERSEDES, CONTRADICTS)")
    properties: Optional[Dict[str, Any]] = Field(default=None, description="Optional properties for the relationship")

class ProjectRelationship(BaseModel):
    """Model for a relationship between project entities."""
    to_project_id: str = Field(..., description="Target project ID")
    relation_type: str = Field(..., description="Type of relationship (e.g., DEPENDS_ON, ALTERNATIVE_TO)")
    component_id: Optional[str] = Field(None, description="Optional component ID if relationship is component-specific")
    properties: Optional[Dict[str, Any]] = Field(default=None, description="Optional properties for the relationship")

@router.post("/", response_model=Dict[str, Any])
async def create_relationship(
    relationship: Relationship,
    memory: GraphMemoryManager = Depends(get_memory_manager)
):
    """
    Create a new relationship between entities.
    
    Args:
        relationship: The relationship to create
        memory: GraphMemoryManager instance
        
    Returns:
        JSON string with operation result
    """
    try:
        relationship_data = relationship.dict(exclude_none=True)
        result = memory.create_relationship(relationship_data)
        return parse_response(result)
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@router.post("/bulk", response_model=Dict[str, Any])
async def create_relationships(
    relationships: List[Relationship],
    memory: GraphMemoryManager = Depends(get_memory_manager)
):
    """
    Create multiple relationships between entities.
    
    Args:
        relationships: List of relationships to create
        memory: GraphMemoryManager instance
        
    Returns:
        JSON string with operation result
    """
    try:
        relationships_data = [r.dict(exclude_none=True) for r in relationships]
        result = memory.create_relationships(relationships_data)
        return parse_response(result)
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@router.get("/{entity_name}", response_model=Dict[str, Any])
async def get_relationships(
    entity_name: str,
    relation_type: Optional[str] = None,
    memory: GraphMemoryManager = Depends(get_memory_manager)
):
    """
    Get relationships for an entity.
    
    Args:
        entity_name: The name of the entity
        relation_type: Optional type of relationship to filter by
        memory: GraphMemoryManager instance
        
    Returns:
        JSON string with the relationships
    """
    try:
        result = memory.get_relationships(entity_name, relation_type)
        return parse_response(result)
    except Exception as e:
        raise HTTPException(status_code=404, detail=str(e))

@router.put("/{from_entity}/{to_entity}/{relation_type}", response_model=Dict[str, Any])
async def update_relation(
    from_entity: str,
    to_entity: str,
    relation_type: str,
    updates: RelationshipUpdate,
    memory: GraphMemoryManager = Depends(get_memory_manager)
):
    """
    Update a relationship between entities.
    
    Args:
        from_entity: The name of the source entity
        to_entity: The name of the target entity
        relation_type: The type of the relationship
        updates: The updates to apply
        memory: GraphMemoryManager instance
        
    Returns:
        JSON string with the updated relationship
    """
    try:
        result = memory.update_relation(from_entity, to_entity, relation_type, updates.updates)
        return parse_response(result)
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@router.delete("/{from_entity}/{to_entity}", response_model=Dict[str, Any])
async def delete_relation(
    from_entity: str,
    to_entity: str,
    relation_type: Optional[str] = None,
    memory: GraphMemoryManager = Depends(get_memory_manager)
):
    """
    Delete a relationship between entities.
    
    Args:
        from_entity: The name of the source entity
        to_entity: The name of the target entity
        relation_type: Optional type of the relationship
        memory: GraphMemoryManager instance
        
    Returns:
        JSON string with the deletion result
    """
    try:
        result = memory.delete_relation(from_entity, to_entity, relation_type)
        return parse_response(result)
    except Exception as e:
        raise HTTPException(status_code=404, detail=str(e))

@router.post("/lessons/{lesson_id}", response_model=Dict[str, Any])
async def create_lesson_relationship(
    lesson_id: str,
    relationship: LessonRelationship,
    memory: GraphMemoryManager = Depends(get_memory_manager)
):
    """
    Create a relationship between lessons.
    
    Args:
        lesson_id: Source lesson ID
        relationship: The relationship to create
        memory: GraphMemoryManager instance
    """
    try:
        relationship_data = {
            "from_entity": f"lesson:{lesson_id}",
            "to_entity": f"lesson:{relationship.to_lesson_id}",
            "relation_type": relationship.relation_type,
            "properties": relationship.properties
        }
        result = memory.create_relationship(relationship_data)
        return parse_response(result)
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@router.get("/lessons/{lesson_id}", response_model=Dict[str, Any])
async def get_lesson_relationships(
    lesson_id: str,
    relation_type: Optional[str] = None,
    memory: GraphMemoryManager = Depends(get_memory_manager)
):
    """
    Get relationships for a lesson.
    
    Args:
        lesson_id: The lesson ID
        relation_type: Optional type of relationship to filter by
        memory: GraphMemoryManager instance
    """
    try:
        result = memory.get_relationships(f"lesson:{lesson_id}", relation_type)
        return parse_response(result)
    except Exception as e:
        raise HTTPException(status_code=404, detail=str(e))

@router.delete("/lessons/{lesson_id}/{to_lesson_id}", response_model=Dict[str, Any])
async def delete_lesson_relationship(
    lesson_id: str,
    to_lesson_id: str,
    relation_type: Optional[str] = None,
    memory: GraphMemoryManager = Depends(get_memory_manager)
):
    """
    Delete a relationship between lessons.
    
    Args:
        lesson_id: Source lesson ID
        to_lesson_id: Target lesson ID
        relation_type: Optional type of relationship
        memory: GraphMemoryManager instance
    """
    try:
        result = memory.delete_relation(
            f"lesson:{lesson_id}",
            f"lesson:{to_lesson_id}",
            relation_type
        )
        return parse_response(result)
    except Exception as e:
        raise HTTPException(status_code=404, detail=str(e))

@router.post("/projects/{project_id}", response_model=Dict[str, Any])
async def create_project_relationship(
    project_id: str,
    relationship: ProjectRelationship,
    memory: GraphMemoryManager = Depends(get_memory_manager)
):
    """
    Create a relationship between projects or project components.
    
    Args:
        project_id: Source project ID
        relationship: The relationship to create
        memory: GraphMemoryManager instance
    """
    try:
        from_entity = (f"project:{project_id}:{relationship.component_id}" 
                      if relationship.component_id 
                      else f"project:{project_id}")
        to_entity = f"project:{relationship.to_project_id}"
        
        relationship_data = {
            "from_entity": from_entity,
            "to_entity": to_entity,
            "relation_type": relationship.relation_type,
            "properties": relationship.properties
        }
        result = memory.create_relationship(relationship_data)
        return parse_response(result)
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@router.get("/projects/{project_id}", response_model=Dict[str, Any])
async def get_project_relationships(
    project_id: str,
    component_id: Optional[str] = None,
    relation_type: Optional[str] = None,
    memory: GraphMemoryManager = Depends(get_memory_manager)
):
    """
    Get relationships for a project or project component.
    
    Args:
        project_id: The project ID
        component_id: Optional component ID to get relationships for
        relation_type: Optional type of relationship to filter by
        memory: GraphMemoryManager instance
    """
    try:
        entity_name = (f"project:{project_id}:{component_id}" 
                      if component_id 
                      else f"project:{project_id}")
        result = memory.get_relationships(entity_name, relation_type)
        return parse_response(result)
    except Exception as e:
        raise HTTPException(status_code=404, detail=str(e))

@router.delete("/projects/{project_id}/{to_project_id}", response_model=Dict[str, Any])
async def delete_project_relationship(
    project_id: str,
    to_project_id: str,
    component_id: Optional[str] = None,
    relation_type: Optional[str] = None,
    memory: GraphMemoryManager = Depends(get_memory_manager)
):
    """
    Delete a relationship between projects or project components.
    
    Args:
        project_id: Source project ID
        to_project_id: Target project ID
        component_id: Optional component ID if relationship is component-specific
        relation_type: Optional type of relationship
        memory: GraphMemoryManager instance
    """
    try:
        from_entity = (f"project:{project_id}:{component_id}" 
                      if component_id 
                      else f"project:{project_id}")
        result = memory.delete_relation(
            from_entity,
            f"project:{to_project_id}",
            relation_type
        )
        return parse_response(result)
    except Exception as e:
        raise HTTPException(status_code=404, detail=str(e))

@router.post("/lessons/{lesson_id}/bulk", response_model=Dict[str, Any])
async def create_lesson_relationships(
    lesson_id: str,
    relationships: List[LessonRelationship],
    memory: GraphMemoryManager = Depends(get_memory_manager)
):
    """
    Create multiple relationships between lessons.
    
    Args:
        lesson_id: Source lesson ID
        relationships: List of relationships to create
        memory: GraphMemoryManager instance
        
    Returns:
        JSON string with operation result
    """
    try:
        relationships_data = [
            {
                "from_entity": f"lesson:{lesson_id}",
                "to_entity": f"lesson:{r.to_lesson_id}",
                "relation_type": r.relation_type,
                "properties": r.properties
            }
            for r in relationships
        ]
        result = memory.create_relationships(relationships_data)
        return parse_response(result)
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@router.post("/projects/{project_id}/bulk", response_model=Dict[str, Any])
async def create_project_relationships(
    project_id: str,
    relationships: List[ProjectRelationship],
    memory: GraphMemoryManager = Depends(get_memory_manager)
):
    """
    Create multiple relationships between projects or project components.
    
    Args:
        project_id: Source project ID
        relationships: List of relationships to create
        memory: GraphMemoryManager instance
        
    Returns:
        JSON string with operation result
    """
    try:
        relationships_data = [
            {
                "from_entity": (f"project:{project_id}:{r.component_id}" 
                              if r.component_id 
                              else f"project:{project_id}"),
                "to_entity": f"project:{r.to_project_id}",
                "relation_type": r.relation_type,
                "properties": r.properties
            }
            for r in relationships
        ]
        result = memory.create_relationships(relationships_data)
        return parse_response(result)
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e)) 