from typing import Dict, Any, Optional, List, Union
from fastapi import APIRouter, HTTPException, Depends
from pydantic import BaseModel, Field, validator, model_validator
import json

from src.graph_memory import GraphMemoryManager
from ..dependencies import get_memory_manager
from ..utils import parse_response

router = APIRouter(
    prefix="/lessons",
    tags=["lessons"],
    responses={404: {"description": "Not found"}},
)

class LessonSection(BaseModel):
    title: str = Field(..., description="Section title")
    content: str = Field(..., description="Section content")
    confidence: Optional[float] = Field(0.5, description="Confidence score (0.0-1.0)")
    tags: Optional[List[str]] = Field(None, description="Tags for the section")

class LessonContainer(BaseModel):
    description: Optional[str] = Field(None, description="Container description")
    metadata: Optional[Dict[str, Any]] = Field(None, description="Optional metadata")
    
    @model_validator(mode='after')
    def validate_metadata(self):
        """Validate that metadata only contains Neo4j-compatible primitive values."""
        if self.metadata:
            sanitized = {}
            for key, value in self.metadata.items():
                # Only include primitive types (strings, numbers, booleans) or lists of primitives
                if isinstance(value, (str, int, float, bool)) or (
                    isinstance(value, list) and all(isinstance(item, (str, int, float, bool)) for item in value)
                ):
                    sanitized[key] = value
            self.metadata = sanitized
        return self

class Lesson(BaseModel):
    title: str = Field(..., description="Lesson title")
    description: Optional[str] = Field(None, description="Lesson description")
    tags: Optional[List[str]] = Field(None, description="Tags for the lesson")

@router.post("/container", response_model=Dict[str, Any])
async def create_lesson_container(
    container: Optional[LessonContainer] = None,
    memory: GraphMemoryManager = Depends(get_memory_manager)
):
    """
    Create the lesson container if it doesn't exist.
    Only one container named 'Lessons' is allowed in the system.
    """
    try:
        # Prepare container data
        description = container.description if container else None
        
        # Sanitize metadata to ensure only primitive types are included
        metadata = None
        if container and container.metadata:
            metadata = {}
            for key, value in container.metadata.items():
                # Only include primitive types (strings, numbers, booleans) or lists of primitives
                if isinstance(value, (str, int, float, bool)) or (
                    isinstance(value, list) and all(isinstance(item, (str, int, float, bool)) for item in value)
                ):
                    metadata[key] = value
        
        container_data = {
            "description": description,
            "metadata": metadata
        }
        result = memory.create_lesson_container(container_data)
        return parse_response(result)
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@router.get("/container", response_model=Dict[str, Any])
async def get_lesson_container(
    memory: GraphMemoryManager = Depends(get_memory_manager)
):
    """
    Get the lesson container. There is only one container named 'Lessons'.
    """
    try:
        result = memory.get_lesson_container()
        return json.loads(result)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/memories", response_model=Dict[str, Any])
async def get_all_lesson_memories(
    container_name: Optional[str] = None,
    memory: GraphMemoryManager = Depends(get_memory_manager)
):
    """
    Get all lesson entities in the knowledge graph.
    
    Args:
        container_name: Optional container name to scope query (defaults to "Lessons")
    """
    try:
        result = memory.get_all_lesson_memories(container_name=container_name)
        return json.loads(result)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/", response_model=Dict[str, Any])
async def create_lesson(
    lesson: Lesson,
    memory: GraphMemoryManager = Depends(get_memory_manager)
):
    """Create a new lesson entity."""
    try:
        result = memory.create_lesson(
            name=lesson.title,
            problem_description=lesson.description or "",
            tags=lesson.tags
        )
        return parse_response(result)
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@router.get("/", response_model=Dict[str, Any])
async def get_lessons(
    search_term: Optional[str] = None,
    entity_type: Optional[str] = None,
    limit: Optional[int] = 50,
    memory: GraphMemoryManager = Depends(get_memory_manager)
):
    """Get lessons from the knowledge graph."""
    try:
        result = memory.get_lessons(
            search_term=search_term,
            entity_type=entity_type,
            limit=limit
        )
        return parse_response(result)
    except Exception as e:
        raise HTTPException(status_code=404, detail=str(e))

@router.put("/{lesson_name}", response_model=Dict[str, Any])
async def update_lesson(
    lesson_name: str,
    lesson: Lesson,
    memory: GraphMemoryManager = Depends(get_memory_manager)
):
    """Update a lesson's properties."""
    try:
        updates = {
            "title": lesson.title,
            "description": lesson.description,
            "tags": lesson.tags
        }
        result = memory.update_lesson(lesson_name, **updates)
        return parse_response(result)
    except Exception as e:
        raise HTTPException(status_code=404, detail=str(e))

@router.post("/{lesson_name}/apply", response_model=Dict[str, Any])
async def apply_lesson_to_context(
    lesson_name: str,
    context_entity: str,
    success_score: Optional[float] = 0.8,
    application_notes: Optional[str] = "",
    memory: GraphMemoryManager = Depends(get_memory_manager)
):
    """Apply a lesson to a context entity."""
    try:
        result = memory.apply_lesson_to_context(
            lesson_name,
            context_entity,
            success_score=success_score,
            application_notes=application_notes
        )
        return parse_response(result)
    except Exception as e:
        raise HTTPException(status_code=404, detail=str(e))

@router.post("/{lesson_id}/sections", response_model=Dict[str, Any])
async def add_lesson_section(
    lesson_id: str,
    section: LessonSection,
    memory: GraphMemoryManager = Depends(get_memory_manager)
):
    """Add a section as a relationship/observation to a lesson node."""
    try:
        # Create the section as an observation on the lesson node
        result = memory.add_observation({
            "entity_name": lesson_id,
            "content": section.content,
            "observation_type": "section",
            "metadata": {
                "title": section.title,
                "confidence": section.confidence,
                "tags": section.tags
            }
        })
        return parse_response(result)
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@router.get("/{lesson_id}", response_model=Dict[str, Any])
async def get_lesson(lesson_id: str, memory: GraphMemoryManager = Depends(get_memory_manager)):
    """Get a lesson node by ID."""
    try:
        result = memory.get_entity(lesson_id)
        return parse_response(result)
    except Exception as e:
        raise HTTPException(status_code=404, detail=str(e))

@router.get("/{lesson_id}/sections", response_model=Dict[str, Any])
async def get_lesson_sections(lesson_id: str, memory: GraphMemoryManager = Depends(get_memory_manager)):
    """Get all section observations connected to a lesson node."""
    try:
        result = memory.get_observations(lesson_id, observation_type="section")
        return parse_response(result)
    except Exception as e:
        raise HTTPException(status_code=404, detail=str(e))

@router.delete("/{lesson_id}/sections/{section_id}")
async def delete_lesson_section(
    lesson_id: str,
    section_id: str,
    memory: GraphMemoryManager = Depends(get_memory_manager)
):
    """Delete a section observation from a lesson node."""
    try:
        result = memory.delete_observation(lesson_id, observation_id=section_id)
        return parse_response(result)
    except Exception as e:
        raise HTTPException(status_code=404, detail=str(e))

@router.post("/extract", response_model=Dict[str, Any])
async def extract_potential_lessons(
    content: Dict[str, Any],
    memory: GraphMemoryManager = Depends(get_memory_manager)
):
    """Extract potential lessons from provided content."""
    try:
        result = memory.extract_potential_lessons(**content)
        return parse_response(result)
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@router.post("/consolidate", response_model=Dict[str, Any])
async def consolidate_related_lessons(
    lesson_ids: List[str],
    new_name: Optional[str] = None,
    merge_strategy: Optional[str] = "union",
    problem_description: Optional[str] = None,
    memory: GraphMemoryManager = Depends(get_memory_manager)
):
    """Consolidate related lessons into a new lesson."""
    try:
        result = memory.consolidate_related_lessons(
            lesson_ids,
            new_name=new_name,
            merge_strategy=merge_strategy,
            problem_description=problem_description
        )
        return parse_response(result)
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@router.get("/evolution", response_model=Dict[str, Any])
async def get_knowledge_evolution(
    entity_name: Optional[str] = None,
    lesson_type: Optional[str] = None,
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
    memory: GraphMemoryManager = Depends(get_memory_manager)
):
    """Track the evolution of knowledge in the lesson graph."""
    try:
        result = memory.get_knowledge_evolution(
            entity_name=entity_name,
            lesson_type=lesson_type,
            start_date=start_date,
            end_date=end_date
        )
        return parse_response(result)
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@router.get("/query", response_model=Dict[str, Any])
async def query_across_contexts(
    query_text: str,
    memory: GraphMemoryManager = Depends(get_memory_manager)
):
    """Query across multiple contexts."""
    try:
        result = memory.query_across_contexts(query_text)
        return parse_response(result)
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e)) 