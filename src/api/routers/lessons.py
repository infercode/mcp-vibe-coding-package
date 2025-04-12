from typing import Dict, Any, Optional, List, Union
from fastapi import APIRouter, HTTPException, Depends
from pydantic import BaseModel, Field, validator, model_validator
import json
import datetime

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
    Get the lesson container details.
    
    This endpoint returns information about the primary lesson container.
    If the container does not exist, it will return a 404 error.
    """
    try:
        # Ensure manager is initialized
        if not memory.check_connection():
            raise HTTPException(status_code=503, detail="Memory system not initialized")
            
        # Execute the get_container operation
        result = memory.lesson_operation(operation_type="get_container")
        result_data = parse_response(result)
        
        # Check if the result indicates no container exists
        if result_data.get("status") == "error" and result_data.get("code") == "container_not_found":
            raise HTTPException(
                status_code=404, 
                detail={"message": "No lesson container found", "exists": False}
            )
        
        # If we get here with a success status and container data, return it
        if result_data.get("status") == "success" and result_data.get("exists") == True:
            # Extract the container data from the nested structure if needed
            if "data" in result_data and "container" in result_data["data"]:
                return {
                    "status": "success",
                    "data": result_data["data"],
                    "exists": True
                }
        
        # Return the parsed response data
        return result_data
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/containers", response_model=Dict[str, Any])
async def list_lesson_containers(
    limit: int = 100,
    sort_by: str = "created",
    memory: GraphMemoryManager = Depends(get_memory_manager)
):
    """
    List all lesson containers.
    
    Args:
        limit: Maximum number of containers to return
        sort_by: Field to sort results by
    
    Returns:
        JSON response with list of containers
    """
    try:
        # Ensure manager is initialized
        if not memory.check_connection():
            raise HTTPException(status_code=503, detail="Memory system not initialized")
        
        # Execute the regular list_containers operation
        result = memory.lesson_operation(
            operation_type="list_containers",
            limit=limit,
            sort_by=sort_by
        )
        result_data = parse_response(result)
        
        # If we didn't get any containers, try to get the main container directly
        found_containers = []
        get_container_result = None
        
        if "containers" not in result_data or not result_data.get("containers"):
            # Try to get the container directly
            get_result = memory.lesson_operation(operation_type="get_container")
            get_container_result = parse_response(get_result)
            
            # Check if we found a container
            if (get_container_result.get("status") == "success" and 
                "container" in str(get_container_result).lower()):
                
                # Extract the container data
                container_data = None
                
                if "container" in get_container_result:
                    container_data = get_container_result["container"]
                elif "data" in get_container_result and "container" in get_container_result["data"]:
                    container_data = get_container_result["data"]["container"]
                
                if container_data:
                    found_containers = [container_data]
        
        # Also check the container exists endpoint
        exists_result = memory.lesson_operation(
            operation_type="container_exists",
            container_name="Lessons"
        )
        exists_data = parse_response(exists_result)
        
        # Build a comprehensive response with all the information
        containers = result_data.get("containers", []) or found_containers
        
        return {
            "status": "success",
            "containers": containers,
            "count": len(containers),
            "message": f"Found {len(containers)} containers",
            "debug": {
                "list_result": result_data,
                "get_container_result": get_container_result,
                "exists_result": exists_data,
                "api_version": "1.0.0",
                "timestamp": str(datetime.datetime.now())
            }
        }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/container/exists", response_model=Dict[str, Any])
async def check_lesson_container_exists(
    container_name: str = "Lessons",
    memory: GraphMemoryManager = Depends(get_memory_manager)
):
    """
    Check if a lesson container exists.
    
    Args:
        container_name: Name of the container to check
    
    Returns:
        JSON response with existence status (exists: true/false)
    """
    try:
        # Ensure manager is initialized
        if not memory.check_connection():
            raise HTTPException(status_code=503, detail="Memory system not initialized")
            
        # Execute the container_exists operation
        result = memory.lesson_operation(
            operation_type="container_exists",
            container_name=container_name
        )
        result_data = parse_response(result)
        
        # Always return a 200 OK for this endpoint, with exists: true/false in the body
        return {
            "status": "success",
            "exists": result_data.get("exists", False),
            "container_name": container_name,
            "message": result_data.get("message", f"Container '{container_name}' existence check completed")
        }
    except HTTPException:
        raise
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

# Direct access to lesson_operation and lesson_context methods
class LessonOperation(BaseModel):
    """Model for direct access to lesson_operation method."""
    operation_type: str = Field(..., description="Type of operation to perform")
    parameters: Dict[str, Any] = Field(default_factory=dict, description="Parameters for the operation")

@router.post("/operation", response_model=Dict[str, Any])
async def direct_lesson_operation(
    operation: LessonOperation,
    memory: GraphMemoryManager = Depends(get_memory_manager)
):
    """
    Direct access to lesson_operation method.
    
    Args:
        operation: Operation type and parameters
        memory: GraphMemoryManager instance
    """
    try:
        # Ensure manager is initialized
        if not memory.check_connection():
            raise HTTPException(status_code=503, detail="Memory system not initialized")
            
        # Execute the lesson operation with provided parameters
        result = memory.lesson_operation(
            operation_type=operation.operation_type,
            **operation.parameters
        )
        return parse_response(result)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

class LessonContextStart(BaseModel):
    """Model for starting a lesson context."""
    container_name: Optional[str] = Field(None, description="Name of the container")

@router.post("/context/start", response_model=Dict[str, Any])
async def start_lesson_context(
    context: LessonContextStart,
    memory: GraphMemoryManager = Depends(get_memory_manager)
):
    """
    Start a lesson context for sequential operations.
    
    This simulates entering a lesson_context context manager.
    """
    try:
        # Ensure manager is initialized
        if not memory.check_connection():
            raise HTTPException(status_code=503, detail="Memory system not initialized")
            
        # Create a session ID for the context
        session_id = f"lesson-context-{context.container_name or 'default'}"
        
        # Set up the context
        # In a real implementation, we would need to store context state,
        # but for demonstration we'll just return the session ID
        
        return {
            "status": "success",
            "message": f"Lesson context started for container {context.container_name or 'default'}",
            "session_id": session_id,
            "container_name": context.container_name
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

class LessonContextOperation(BaseModel):
    """Model for operations within a lesson context."""
    session_id: str = Field(..., description="Session ID from start_lesson_context")
    operation_type: str = Field(..., description="Type of operation to perform")
    parameters: Dict[str, Any] = Field(default_factory=dict, description="Parameters for the operation")

@router.post("/context/operation", response_model=Dict[str, Any])
async def lesson_context_operation(
    operation: LessonContextOperation,
    memory: GraphMemoryManager = Depends(get_memory_manager)
):
    """
    Execute an operation within a lesson context.
    
    This simulates operations performed inside a lesson_context context manager.
    """
    try:
        # Ensure manager is initialized
        if not memory.check_connection():
            raise HTTPException(status_code=503, detail="Memory system not initialized")
        
        # Check if the session ID is valid
        if not operation.session_id.startswith("lesson-context-"):
            raise HTTPException(status_code=400, detail="Invalid session ID")
            
        # Extract container name from session ID
        container_name = operation.session_id.replace("lesson-context-", "", 1)
        if container_name == "default":
            container_name = None
            
        # Execute the lesson operation with provided parameters
        # Add container_name to parameters if it's not None
        parameters = dict(operation.parameters)
        if container_name:
            parameters["container_name"] = container_name
            
        result = memory.lesson_operation(
            operation_type=operation.operation_type,
            **parameters
        )
        return parse_response(result)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/context/end", response_model=Dict[str, Any])
async def end_lesson_context(
    context: LessonContextOperation,
    memory: GraphMemoryManager = Depends(get_memory_manager)
):
    """
    End a lesson context.
    
    This simulates exiting a lesson_context context manager.
    """
    try:
        # Ensure manager is initialized
        if not memory.check_connection():
            raise HTTPException(status_code=503, detail="Memory system not initialized")
            
        # Check if the session ID is valid
        if not context.session_id.startswith("lesson-context-"):
            raise HTTPException(status_code=400, detail="Invalid session ID")
            
        # In a real implementation, we would clean up the context
        
        return {
            "status": "success",
            "message": f"Lesson context ended for session {context.session_id}"
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# Advanced API for bulk operations in lesson context
class BulkLessonOperations(BaseModel):
    """Model for executing multiple operations in a lesson context."""
    container_name: Optional[str] = Field(None, description="Name of the container")
    operations: List[Dict[str, Any]] = Field(..., description="List of operations to perform")

@router.post("/bulk", response_model=Dict[str, Any])
async def bulk_lesson_operations(
    bulk: BulkLessonOperations,
    memory: GraphMemoryManager = Depends(get_memory_manager)
):
    """
    Execute multiple operations in a lesson context.
    
    This simulates using a lesson_context context manager for multiple operations.
    """
    try:
        # Ensure manager is initialized
        if not memory.check_connection():
            raise HTTPException(status_code=503, detail="Memory system not initialized")
            
        # Execute all operations
        results = []
        for operation in bulk.operations:
            operation_type = operation.pop("operation_type", None)
            if not operation_type:
                raise HTTPException(status_code=400, detail="Missing operation_type in operation")
                
            # Add container_name to parameters if it's provided
            if bulk.container_name:
                operation["container_name"] = bulk.container_name
                
            result = memory.lesson_operation(
                operation_type=operation_type,
                **operation
            )
            results.append(parse_response(result))
            
        return {
            "status": "success",
            "message": f"Executed {len(results)} operations in lesson context for container {bulk.container_name or 'default'}",
            "results": results
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/debug/container_check", response_model=Dict[str, Any])
async def debug_container_check(
    memory: GraphMemoryManager = Depends(get_memory_manager)
):
    """
    Directly check Neo4j for lesson containers - for debugging only.
    """
    try:
        # Ensure manager is initialized
        if not memory.check_connection():
            raise HTTPException(status_code=503, detail="Memory system not initialized")
            
        # Direct Cypher query to check for containers
        query = """
        MATCH (c:LessonContainer)
        RETURN c
        """
        
        # Use query_knowledge_graph for direct database access
        result = memory.query_knowledge_graph(query)
        
        # Parse the response
        result_data = parse_response(result)
        
        # Add debug information
        debug_info = {
            "raw_query_result": json.dumps(result_data),
            "check_time": str(datetime.datetime.now())
        }
        
        # Check if we found any containers
        records = result_data.get("records", [])
        if records:
            containers = []
            for record in records:
                if "c" in record:
                    container_props = record["c"]
                    containers.append(container_props)
            
            return {
                "status": "success",
                "containers_found": len(containers),
                "containers": containers,
                "debug": debug_info
            }
        else:
            return {
                "status": "success",
                "containers_found": 0,
                "message": "No containers found in Neo4j",
                "debug": debug_info
            }
            
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/debug/create_container", response_model=Dict[str, Any])
async def debug_create_container(
    memory: GraphMemoryManager = Depends(get_memory_manager)
):
    """
    Directly create a lesson container in Neo4j - for debugging only.
    """
    try:
        # Ensure manager is initialized
        if not memory.check_connection():
            raise HTTPException(status_code=503, detail="Memory system not initialized")
            
        # Direct Cypher query to create a container
        query = """
        CREATE (c:LessonContainer {name: 'Lessons', description: 'Default lesson container', created: datetime()})
        RETURN c
        """
        
        # Use query_knowledge_graph for direct database access
        result = memory.query_knowledge_graph(query)
        
        # Parse the response
        result_data = parse_response(result)
        
        # Check if we created the container
        records = result_data.get("records", [])
        if records and len(records) > 0 and "c" in records[0]:
            container = records[0]["c"]
            return {
                "status": "success",
                "message": "Container created successfully",
                "container": container
            }
        else:
            return {
                "status": "error",
                "message": "Failed to create container",
                "raw_result": result_data
            }
            
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/debug/direct_container_list", response_model=Dict[str, Any])
async def debug_direct_container_list(
    memory: GraphMemoryManager = Depends(get_memory_manager)
):
    """
    Directly query Neo4j for lesson containers - for debugging.
    """
    try:
        # Ensure manager is initialized
        if not memory.check_connection():
            raise HTTPException(status_code=503, detail="Memory system not initialized")
            
        # Direct Cypher query to list containers
        query = """
        MATCH (c:LessonContainer)
        RETURN c
        """
        
        # Use query_knowledge_graph for direct database access
        result = memory.query_knowledge_graph(query)
        result_data = parse_response(result)
        
        containers = []
        
        # Extract container data from records
        if "records" in result_data and result_data["records"]:
            for record in result_data["records"]:
                if "c" in record:
                    container_data = record["c"]
                    containers.append(container_data)
        
        return {
            "status": "success",
            "message": f"Found {len(containers)} containers via direct query",
            "containers": containers
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/debug/node_count", response_model=Dict[str, Any])
async def debug_node_count(
    memory: GraphMemoryManager = Depends(get_memory_manager)
):
    """
    Get a count of all nodes in the database - for debugging.
    """
    try:
        # Ensure manager is initialized
        if not memory.check_connection():
            raise HTTPException(status_code=503, detail="Memory system not initialized")
            
        # Direct Cypher query to count all nodes
        query = """
        MATCH (n)
        RETURN count(n) as node_count, 
               count(n:LessonContainer) as lesson_container_count,
               count(n:Entity) as entity_count
        """
        
        # Use query_knowledge_graph for direct database access
        result = memory.query_knowledge_graph(query)
        result_data = parse_response(result)
        
        # Extract the counts
        counts = {}
        if "records" in result_data and result_data["records"]:
            counts = result_data["records"][0]
            
        return {
            "status": "success", 
            "message": "Database node counts",
            "counts": counts
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e)) 