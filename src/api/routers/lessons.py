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
        # Ensure manager is initialized
        if not memory.check_connection():
            raise HTTPException(status_code=503, detail="Memory system not initialized")
            
        # Prepare container data
        container_data = {}
        if container:
            container_data = {
                "description": container.description,
                "metadata": container.metadata
            }
        
        # Use lesson_operation for unified interface
        result = memory.lesson_operation(
            operation_type="create_container",
            **container_data
        )
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
        # Ensure manager is initialized
        if not memory.check_connection():
            raise HTTPException(status_code=503, detail="Memory system not initialized")
            
        # Use lesson_operation for unified interface
        result = memory.lesson_operation(
            operation_type="get_container"
        )
        return parse_response(result)
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
        # Ensure manager is initialized
        if not memory.check_connection():
            raise HTTPException(status_code=503, detail="Memory system not initialized")
            
        # Use get_all_memories with entity_types filter
        result = memory.get_all_memories()
        result_data = parse_response(result)
        
        # Filter for lesson entities
        if "entities" in result_data:
            result_data["entities"] = [
                entity for entity in result_data["entities"] 
                if entity.get("entity_type", "").lower() in ["lesson", "lessoncontainer"]
            ]
        
        return result_data
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/", response_model=Dict[str, Any])
async def create_lesson(
    lesson: Lesson,
    memory: GraphMemoryManager = Depends(get_memory_manager)
):
    """Create a new lesson entity."""
    try:
        # Ensure manager is initialized
        if not memory.check_connection():
            raise HTTPException(status_code=503, detail="Memory system not initialized")
            
        # Use lesson_operation for unified interface
        result = memory.lesson_operation(
            operation_type="create",
            name=lesson.title,
            lesson_type="Lesson",
            description=lesson.description or "",
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
        # Ensure manager is initialized
        if not memory.check_connection():
            raise HTTPException(status_code=503, detail="Memory system not initialized")
            
        # Use lesson_operation for unified interface
        result = memory.lesson_operation(
            operation_type="search",
            query=search_term or "",
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
        # Ensure manager is initialized
        if not memory.check_connection():
            raise HTTPException(status_code=503, detail="Memory system not initialized")
            
        # Use lesson_operation for unified interface
        updates = {
            "title": lesson.title,
            "description": lesson.description,
            "tags": lesson.tags
        }
        
        result = memory.lesson_operation(
            operation_type="update",
            entity_name=lesson_name,
            updates=updates
        )
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
        # Ensure manager is initialized
        if not memory.check_connection():
            raise HTTPException(status_code=503, detail="Memory system not initialized")
            
        # Use lesson_operation for unified interface
        result = memory.lesson_operation(
            operation_type="track",
            lesson_name=lesson_name,
            context_entity=context_entity,
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
        # Ensure manager is initialized
        if not memory.check_connection():
            raise HTTPException(status_code=503, detail="Memory system not initialized")
            
        # Use lesson_operation for unified interface
        result = memory.lesson_operation(
            operation_type="observe",
            entity_name=lesson_id,
            what_was_learned=section.content,
            observation_type="section",
            title=section.title,
            confidence=section.confidence,
            tags=section.tags
        )
        return parse_response(result)
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@router.get("/{lesson_id}", response_model=Dict[str, Any])
async def get_lesson(lesson_id: str, memory: GraphMemoryManager = Depends(get_memory_manager)):
    """Get a lesson by ID."""
    try:
        # Ensure manager is initialized
        if not memory.check_connection():
            raise HTTPException(status_code=503, detail="Memory system not initialized")
            
        result = memory.get_entity(lesson_id)
        return parse_response(result)
    except Exception as e:
        raise HTTPException(status_code=404, detail=str(e))

@router.get("/{lesson_id}/sections", response_model=Dict[str, Any])
async def get_lesson_sections(lesson_id: str, memory: GraphMemoryManager = Depends(get_memory_manager)):
    """Get all sections for a lesson."""
    try:
        # Ensure manager is initialized
        if not memory.check_connection():
            raise HTTPException(status_code=503, detail="Memory system not initialized")
            
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
    """Delete a section from a lesson."""
    try:
        # Ensure manager is initialized
        if not memory.check_connection():
            raise HTTPException(status_code=503, detail="Memory system not initialized")
            
        result = memory.delete_observation(lesson_id, observation_id=section_id)
        return parse_response(result)
    except Exception as e:
        raise HTTPException(status_code=404, detail=str(e))

@router.post("/extract", response_model=Dict[str, Any])
async def extract_potential_lessons(
    content: Dict[str, Any],
    memory: GraphMemoryManager = Depends(get_memory_manager)
):
    """Extract potential lessons from content."""
    try:
        # Ensure manager is initialized
        if not memory.check_connection():
            raise HTTPException(status_code=503, detail="Memory system not initialized")
            
        # This would require an AI-based extraction service
        raise HTTPException(status_code=501, detail="Lesson extraction not implemented in this version")
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
    """Consolidate multiple related lessons into a single lesson."""
    try:
        # Ensure manager is initialized
        if not memory.check_connection():
            raise HTTPException(status_code=503, detail="Memory system not initialized")
            
        # Use lesson_operation for unified interface
        result = memory.lesson_operation(
            operation_type="consolidate",
            source_lessons=lesson_ids,
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
    """Get knowledge evolution over time."""
    try:
        # Ensure manager is initialized
        if not memory.check_connection():
            raise HTTPException(status_code=503, detail="Memory system not initialized")
            
        # Use custom Cypher to get evolution data
        cypher_query = """
        MATCH (l:Lesson)
        WHERE l.created_at IS NOT NULL
        """
        
        if entity_name:
            cypher_query += f" AND l.name = '{entity_name}'"
        
        if lesson_type:
            cypher_query += f" AND l.lesson_type = '{lesson_type}'"
            
        cypher_query += """
        RETURN l.name as name, l.lesson_type as type, l.created_at as created, 
               l.confidence as confidence, l.tags as tags
        ORDER BY l.created_at
        """
        
        result = memory.query_knowledge_graph(cypher_query)
        return parse_response(result)
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@router.get("/query", response_model=Dict[str, Any])
async def query_across_contexts(
    query_text: str,
    memory: GraphMemoryManager = Depends(get_memory_manager)
):
    """Query lessons across different contexts."""
    try:
        # Ensure manager is initialized
        if not memory.check_connection():
            raise HTTPException(status_code=503, detail="Memory system not initialized")
            
        # Use semantic search for cross-context queries
        result = memory.search_nodes(query_text, limit=10, entity_types=["Lesson"], semantic=True)
        return parse_response(result)
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e)) 