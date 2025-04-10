from typing import Dict, Any, List, Optional
from fastapi import APIRouter, HTTPException, Depends
from pydantic import BaseModel, Field

from src.graph_memory import GraphMemoryManager
from ..dependencies import get_memory_manager
from ..utils import parse_response

router = APIRouter(
    prefix="/observations",
    tags=["observations"],
    responses={404: {"description": "Not found"}}
)

class Observation(BaseModel):
    """Model for an observation."""
    entity: str = Field(..., description="Entity name")
    content: str = Field(..., description="Observation content")
    type: Optional[str] = Field(None, description="Type of observation")

class LessonObservation(BaseModel):
    """Model for a lesson-specific observation."""
    lesson_id: str = Field(..., description="ID of the lesson")
    content: str = Field(..., description="Observation content")
    type: str = Field(..., description="Type of observation (WhatWasLearned, WhyItMatters, HowToApply, RootCause, Evidence)")

class ProjectObservation(BaseModel):
    """Model for a project-specific observation."""
    project_id: str = Field(..., description="ID of the project")
    component_id: Optional[str] = Field(None, description="Optional ID of the specific component")
    content: str = Field(..., description="Observation content")
    type: str = Field(..., description="Type of observation")

# Core Observation APIs
@router.post("/", response_model=Dict[str, Any])
async def add_observation(observation: Observation, memory: GraphMemoryManager = Depends(get_memory_manager)):
    """Add an observation to an entity."""
    try:
        result = memory.add_observations([{
            "entity": observation.entity,
            "content": observation.content,
            "type": observation.type
        }])
        return parse_response(result)
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@router.get("/{entity_name}", response_model=Dict[str, Any])
async def get_observations(
    entity_name: str,
    observation_type: Optional[str] = None,
    memory: GraphMemoryManager = Depends(get_memory_manager)
):
    """
    Get observations for an entity from the knowledge graph.
    
    Args:
        entity_name: The name of the entity
        observation_type: Optional type of observation to filter by
        memory: GraphMemoryManager instance
        
    Returns:
        JSON string with the observations
    """
    try:
        result = memory.get_observations(entity_name, observation_type)
        return parse_response(result)
    except Exception as e:
        raise HTTPException(status_code=404, detail=str(e))

@router.put("/{entity_name}/{observation_id}", response_model=Dict[str, Any])
async def update_observation(
    entity_name: str,
    observation_id: str,
    content: str,
    observation_type: Optional[str] = None,
    memory: GraphMemoryManager = Depends(get_memory_manager)
):
    """
    Update an observation in the knowledge graph.
    
    Args:
        entity_name: The name of the entity
        observation_id: The ID of the observation
        content: The new content for the observation
        observation_type: Optional new type for the observation
        memory: GraphMemoryManager instance
        
    Returns:
        JSON string with the updated observation
    """
    try:
        result = memory.update_observation(entity_name, observation_id, content, observation_type)
        return parse_response(result)
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@router.delete("/{entity_name}", response_model=Dict[str, Any])
async def delete_observation(
    entity_name: str,
    observation_content: Optional[str] = None,
    observation_id: Optional[str] = None,
    memory: GraphMemoryManager = Depends(get_memory_manager)
):
    """
    Delete an observation from the knowledge graph.
    
    Args:
        entity_name: The name of the entity
        observation_content: The content of the observation to delete
        observation_id: The ID of the observation to delete
        memory: GraphMemoryManager instance
        
    Returns:
        JSON string with the result of the deletion
    """
    try:
        result = memory.delete_observation(entity_name, observation_content, observation_id)
        return parse_response(result)
    except Exception as e:
        raise HTTPException(status_code=404, detail=str(e))

# Lesson-specific Observation APIs
@router.post("/lessons/{lesson_id}", response_model=Dict[str, Any])
async def add_lesson_observation(
    lesson_id: str,
    observation: LessonObservation,
    memory: GraphMemoryManager = Depends(get_memory_manager)
):
    """
    Add an observation to a lesson.
    
    Args:
        lesson_id: ID of the lesson
        observation: The lesson observation to add
        memory: GraphMemoryManager instance
        
    Returns:
        JSON string with the added observation
    """
    try:
        # Format entity name for lesson observations
        entity_name = f"lesson_{lesson_id}"
        result = memory.add_observations([{
            "entity": entity_name,
            "content": observation.content,
            "type": observation.type,
            "metadata": {"lesson_id": lesson_id}
        }])
        return parse_response(result)
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@router.get("/lessons/{lesson_id}", response_model=Dict[str, Any])
async def get_lesson_observations(
    lesson_id: str,
    observation_type: Optional[str] = None,
    memory: GraphMemoryManager = Depends(get_memory_manager)
):
    """
    Get observations for a lesson.
    
    Args:
        lesson_id: ID of the lesson
        observation_type: Optional type of observation to filter by
        memory: GraphMemoryManager instance
        
    Returns:
        JSON string with the lesson observations
    """
    try:
        entity_name = f"lesson_{lesson_id}"
        result = memory.get_observations(entity_name, observation_type)
        return parse_response(result)
    except Exception as e:
        raise HTTPException(status_code=404, detail=str(e))

@router.put("/lessons/{lesson_id}/{observation_id}", response_model=Dict[str, Any])
async def update_lesson_observation(
    lesson_id: str,
    observation_id: str,
    content: str,
    observation_type: Optional[str] = None,
    memory: GraphMemoryManager = Depends(get_memory_manager)
):
    """
    Update a lesson observation.
    
    Args:
        lesson_id: ID of the lesson
        observation_id: ID of the observation to update
        content: New content for the observation
        observation_type: Optional new type for the observation
        memory: GraphMemoryManager instance
        
    Returns:
        JSON string with the updated observation
    """
    try:
        entity_name = f"lesson_{lesson_id}"
        result = memory.update_observation(entity_name, observation_id, content, observation_type)
        return parse_response(result)
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@router.delete("/lessons/{lesson_id}", response_model=Dict[str, Any])
async def delete_lesson_observation(
    lesson_id: str,
    observation_content: Optional[str] = None,
    observation_id: Optional[str] = None,
    memory: GraphMemoryManager = Depends(get_memory_manager)
):
    """
    Delete a lesson observation.
    
    Args:
        lesson_id: ID of the lesson
        observation_content: Optional content of the observation to delete
        observation_id: Optional ID of the observation to delete
        memory: GraphMemoryManager instance
        
    Returns:
        JSON string with the deletion result
    """
    try:
        entity_name = f"lesson_{lesson_id}"
        result = memory.delete_observation(entity_name, observation_content, observation_id)
        return parse_response(result)
    except Exception as e:
        raise HTTPException(status_code=404, detail=str(e))

# Project-specific Observation APIs
@router.post("/projects/{project_id}", response_model=Dict[str, Any])
async def add_project_observation(
    project_id: str,
    observation: ProjectObservation,
    memory: GraphMemoryManager = Depends(get_memory_manager)
):
    """
    Add an observation to a project or project component.
    
    Args:
        project_id: ID of the project
        observation: The project observation to add
        memory: GraphMemoryManager instance
        
    Returns:
        JSON string with the added observation
    """
    try:
        # Format entity name based on whether it's for a specific component
        entity_name = f"project_{project_id}"
        if observation.component_id:
            entity_name = f"{entity_name}_component_{observation.component_id}"
        
        result = memory.add_observations([{
            "entity": entity_name,
            "content": observation.content,
            "type": observation.type,
            "metadata": {
                "project_id": project_id,
                "component_id": observation.component_id
            }
        }])
        return parse_response(result)
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@router.get("/projects/{project_id}", response_model=Dict[str, Any])
async def get_project_observations(
    project_id: str,
    component_id: Optional[str] = None,
    observation_type: Optional[str] = None,
    memory: GraphMemoryManager = Depends(get_memory_manager)
):
    """
    Get observations for a project or project component.
    
    Args:
        project_id: ID of the project
        component_id: Optional ID of the specific component
        observation_type: Optional type of observation to filter by
        memory: GraphMemoryManager instance
        
    Returns:
        JSON string with the project observations
    """
    try:
        entity_name = f"project_{project_id}"
        if component_id:
            entity_name = f"{entity_name}_component_{component_id}"
        result = memory.get_observations(entity_name, observation_type)
        return parse_response(result)
    except Exception as e:
        raise HTTPException(status_code=404, detail=str(e))

@router.put("/projects/{project_id}/{observation_id}", response_model=Dict[str, Any])
async def update_project_observation(
    project_id: str,
    observation_id: str,
    content: str,
    component_id: Optional[str] = None,
    observation_type: Optional[str] = None,
    memory: GraphMemoryManager = Depends(get_memory_manager)
):
    """
    Update a project observation.
    
    Args:
        project_id: ID of the project
        observation_id: ID of the observation to update
        content: New content for the observation
        component_id: Optional ID of the specific component
        observation_type: Optional new type for the observation
        memory: GraphMemoryManager instance
        
    Returns:
        JSON string with the updated observation
    """
    try:
        entity_name = f"project_{project_id}"
        if component_id:
            entity_name = f"{entity_name}_component_{component_id}"
        result = memory.update_observation(entity_name, observation_id, content, observation_type)
        return parse_response(result)
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@router.delete("/projects/{project_id}", response_model=Dict[str, Any])
async def delete_project_observation(
    project_id: str,
    component_id: Optional[str] = None,
    observation_content: Optional[str] = None,
    observation_id: Optional[str] = None,
    memory: GraphMemoryManager = Depends(get_memory_manager)
):
    """
    Delete a project observation.
    
    Args:
        project_id: ID of the project
        component_id: Optional ID of the specific component
        observation_content: Optional content of the observation to delete
        observation_id: Optional ID of the observation to delete
        memory: GraphMemoryManager instance
        
    Returns:
        JSON string with the deletion result
    """
    try:
        entity_name = f"project_{project_id}"
        if component_id:
            entity_name = f"{entity_name}_component_{component_id}"
        result = memory.delete_observation(entity_name, observation_content, observation_id)
        return parse_response(result)
    except Exception as e:
        raise HTTPException(status_code=404, detail=str(e)) 