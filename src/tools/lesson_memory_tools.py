#!/usr/bin/env python3
import json
import datetime
from typing import Dict, List, Any, Optional, Union

from src.logger import get_logger
from src.utils import dict_to_json

# Initialize logger
logger = get_logger()

class ErrorResponse:
    @staticmethod
    def create(message: str, code: str = "internal_error", details: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Create a standardized error response."""
        response = {
            "status": "error",
            "error": {
                "code": code,
                "message": message
            },
            "timestamp": datetime.datetime.now().isoformat()
        }
        if details:
            response["error"]["details"] = details
        return response

def register_lesson_tools(server, graph_manager):
    """Register lesson memory tools with the server."""
    
    # Lesson Container Management Tools
    @server.tool()
    async def create_lesson_container(lesson_data: Dict[str, Any]) -> str:
        """
        Create a new lesson container in the knowledge graph.
        
        Args:
            lesson_data: Dictionary containing lesson information
                - title: Required. The title of the lesson
                - description: Optional. Description of the lesson
                - metadata: Optional. Additional metadata for the lesson
                - tags: Optional. List of tags for categorizing the lesson
                - lesson_id: Optional. Custom ID for the lesson (generated if not provided)
                - visibility: Optional. Visibility setting for the lesson ("public", "private", etc.)
                - difficulty: Optional. Difficulty level of the lesson
                - prerequisites: Optional. List of prerequisite lessons
                - estimated_time: Optional. Estimated time to complete the lesson
                
        Returns:
            JSON response with operation result
        """
        try:
            # Extract required fields
            if "title" not in lesson_data:
                return dict_to_json(ErrorResponse.create(
                    message="Missing required field: title",
                    code="missing_required_field"
                ))
                
            result = graph_manager.create_lesson_container(lesson_data)
            
            if result["status"] == "error":
                return dict_to_json(ErrorResponse.create(
                    message=f"Failed to create lesson: {result.get('message', 'Unknown error')}",
                    code="lesson_creation_error"
                ))
                
            return dict_to_json({
                "status": "success",
                "message": f"Lesson container '{lesson_data['title']}' created successfully",
                "lesson_id": result.get("lesson_id", ""),
                "timestamp": datetime.datetime.now().isoformat()
            })
            
        except Exception as e:
            logger.error(f"Error creating lesson container: {str(e)}")
            error_response = ErrorResponse.create(
                message=f"Failed to create lesson container: {str(e)}",
                code="lesson_creation_error"
            )
            return dict_to_json(error_response)
            
    @server.tool()
    async def get_lesson_container(lesson_id: str) -> str:
        """
        Retrieve a lesson container by ID or title.
        
        Args:
            lesson_id: The ID or title of the lesson container
                
        Returns:
            JSON response with lesson container data
        """
        try:
            result = graph_manager.get_lesson_container(lesson_id)
            
            if result["status"] == "error":
                return dict_to_json(ErrorResponse.create(
                    message=f"Lesson container not found: {lesson_id}",
                    code="lesson_not_found"
                ))
                
            return dict_to_json({
                "status": "success",
                "lesson": result["lesson"],
                "timestamp": datetime.datetime.now().isoformat()
            })
            
        except Exception as e:
            logger.error(f"Error retrieving lesson container: {str(e)}")
            error_response = ErrorResponse.create(
                message=f"Failed to retrieve lesson container: {str(e)}",
                code="lesson_retrieval_error"
            )
            return dict_to_json(error_response)
            
    @server.tool()
    async def update_lesson_container(lesson_data: Dict[str, Any]) -> str:
        """
        Update an existing lesson container.
        
        Args:
            lesson_data: Dictionary containing lesson information
                - id: Required. The ID of the lesson container to update
                - title: Optional. New title for the lesson
                - description: Optional. New description for the lesson
                - metadata: Optional. Updated metadata for the lesson
                - tags: Optional. Updated list of tags
                - visibility: Optional. Updated visibility setting
                - difficulty: Optional. Updated difficulty level
                - prerequisites: Optional. Updated list of prerequisite lessons
                - estimated_time: Optional. Updated estimated completion time
                
        Returns:
            JSON response with operation result
        """
        try:
            # Extract required fields
            if "id" not in lesson_data:
                return dict_to_json(ErrorResponse.create(
                    message="Missing required field: id",
                    code="missing_required_field"
                ))
                
            result = graph_manager.update_lesson_container(lesson_data)
            
            if result["status"] == "error":
                return dict_to_json(ErrorResponse.create(
                    message=f"Failed to update lesson: {result.get('message', 'Unknown error')}",
                    code="lesson_update_error"
                ))
                
            return dict_to_json({
                "status": "success",
                "message": f"Lesson container '{lesson_data.get('title', lesson_data['id'])}' updated successfully",
                "timestamp": datetime.datetime.now().isoformat()
            })
            
        except Exception as e:
            logger.error(f"Error updating lesson container: {str(e)}")
            error_response = ErrorResponse.create(
                message=f"Failed to update lesson container: {str(e)}",
                code="lesson_update_error"
            )
            return dict_to_json(error_response)
            
    @server.tool()
    async def delete_lesson_container(lesson_id: str) -> str:
        """
        Delete a lesson container and all its associated entities.
        
        Args:
            lesson_id: The ID or title of the lesson container to delete
                
        Returns:
            JSON response with operation result
        """
        try:
            result = graph_manager.delete_lesson_container(lesson_id)
            
            if result["status"] == "error":
                return dict_to_json(ErrorResponse.create(
                    message=f"Failed to delete lesson: {result.get('message', 'Unknown error')}",
                    code="lesson_deletion_error"
                ))
                
            return dict_to_json({
                "status": "success",
                "message": f"Lesson container '{lesson_id}' deleted successfully",
                "timestamp": datetime.datetime.now().isoformat()
            })
            
        except Exception as e:
            logger.error(f"Error deleting lesson container: {str(e)}")
            error_response = ErrorResponse.create(
                message=f"Failed to delete lesson container: {str(e)}",
                code="lesson_deletion_error"
            )
            return dict_to_json(error_response)
            
    @server.tool()
    async def list_lesson_containers(tags: Optional[List[str]] = None, 
                                    difficulty: Optional[str] = None, 
                                    visibility: Optional[str] = None) -> str:
        """
        List lesson containers, optionally filtered by tags, difficulty, and/or visibility.
        
        Args:
            tags: Optional. List of tags to filter by
            difficulty: Optional. Difficulty level to filter by
            visibility: Optional. Visibility setting to filter by
                
        Returns:
            JSON response with list of lesson containers
        """
        try:
            # Build filter parameters
            filter_params = {}
            if tags:
                filter_params["tags"] = tags
            if difficulty:
                filter_params["difficulty"] = difficulty
            if visibility:
                filter_params["visibility"] = visibility
                
            result = graph_manager.list_lesson_containers(filter_params)
            
            if result["status"] == "error":
                return dict_to_json(ErrorResponse.create(
                    message=f"Failed to list lessons: {result.get('message', 'Unknown error')}",
                    code="lesson_list_error"
                ))
                
            return dict_to_json({
                "status": "success",
                "lessons": result["lessons"],
                "count": len(result["lessons"]),
                "filters": filter_params,
                "timestamp": datetime.datetime.now().isoformat()
            })
            
        except Exception as e:
            logger.error(f"Error listing lesson containers: {str(e)}")
            error_response = ErrorResponse.create(
                message=f"Failed to list lesson containers: {str(e)}",
                code="lesson_list_error"
            )
            return dict_to_json(error_response)
    
    # Lesson Section Management
    @server.tool()
    async def create_lesson_section(section_data: Dict[str, Any]) -> str:
        """
        Create a new section within a lesson.
        
        Args:
            section_data: Dictionary containing section information
                - lesson_id: Required. The ID of the lesson this section belongs to
                - title: Required. The title of the section
                - content: Optional. The content of the section
                - order: Optional. The order of the section within the lesson
                - section_type: Optional. The type of section (e.g., "introduction", "explanation", "exercise")
                - metadata: Optional. Additional metadata for the section
                
        Returns:
            JSON response with operation result
        """
        try:
            # Extract required fields
            required_fields = ["lesson_id", "title"]
            missing_fields = [field for field in required_fields if field not in section_data]
            
            if missing_fields:
                return dict_to_json(ErrorResponse.create(
                    message=f"Missing required fields: {', '.join(missing_fields)}",
                    code="missing_required_fields"
                ))
                
            result = graph_manager.create_lesson_section(section_data)
            
            if result["status"] == "error":
                return dict_to_json(ErrorResponse.create(
                    message=f"Failed to create section: {result.get('message', 'Unknown error')}",
                    code="section_creation_error"
                ))
                
            return dict_to_json({
                "status": "success",
                "message": f"Section '{section_data['title']}' created successfully",
                "section_id": result.get("section_id", ""),
                "timestamp": datetime.datetime.now().isoformat()
            })
            
        except Exception as e:
            logger.error(f"Error creating lesson section: {str(e)}")
            error_response = ErrorResponse.create(
                message=f"Failed to create lesson section: {str(e)}",
                code="section_creation_error"
            )
            return dict_to_json(error_response)
            
    @server.tool()
    async def update_lesson_section(section_data: Dict[str, Any]) -> str:
        """
        Update an existing lesson section.
        
        Args:
            section_data: Dictionary containing section information
                - id: Required. The ID of the section to update
                - title: Optional. New title for the section
                - content: Optional. Updated content for the section
                - order: Optional. Updated order within the lesson
                - section_type: Optional. Updated section type
                - metadata: Optional. Updated metadata
                
        Returns:
            JSON response with operation result
        """
        try:
            # Extract required fields
            if "id" not in section_data:
                return dict_to_json(ErrorResponse.create(
                    message="Missing required field: id",
                    code="missing_required_field"
                ))
                
            result = graph_manager.update_lesson_section(section_data)
            
            if result["status"] == "error":
                return dict_to_json(ErrorResponse.create(
                    message=f"Failed to update section: {result.get('message', 'Unknown error')}",
                    code="section_update_error"
                ))
                
            return dict_to_json({
                "status": "success",
                "message": f"Section '{section_data.get('title', section_data['id'])}' updated successfully",
                "timestamp": datetime.datetime.now().isoformat()
            })
            
        except Exception as e:
            logger.error(f"Error updating lesson section: {str(e)}")
            error_response = ErrorResponse.create(
                message=f"Failed to update lesson section: {str(e)}",
                code="section_update_error"
            )
            return dict_to_json(error_response)
    
    # Module Management Tools
    @server.tool()
    async def create_module(module_data: Dict[str, Any]) -> str:
        """
        Create a new module (grouping of lessons).
        
        Args:
            module_data: Dictionary containing module information
                - title: Required. The title of the module
                - description: Optional. Description of the module
                - lessons: Optional. List of lesson IDs to include in the module
                - metadata: Optional. Additional metadata for the module
                - tags: Optional. List of tags for categorizing the module
                - visibility: Optional. Visibility setting for the module
                
        Returns:
            JSON response with operation result
        """
        try:
            # Extract required fields
            if "title" not in module_data:
                return dict_to_json(ErrorResponse.create(
                    message="Missing required field: title",
                    code="missing_required_field"
                ))
                
            result = graph_manager.create_module(module_data)
            
            if result["status"] == "error":
                return dict_to_json(ErrorResponse.create(
                    message=f"Failed to create module: {result.get('message', 'Unknown error')}",
                    code="module_creation_error"
                ))
                
            return dict_to_json({
                "status": "success",
                "message": f"Module '{module_data['title']}' created successfully",
                "module_id": result.get("module_id", ""),
                "timestamp": datetime.datetime.now().isoformat()
            })
            
        except Exception as e:
            logger.error(f"Error creating module: {str(e)}")
            error_response = ErrorResponse.create(
                message=f"Failed to create module: {str(e)}",
                code="module_creation_error"
            )
            return dict_to_json(error_response)
    
    # Lesson Relationship Tools
    @server.tool()
    async def create_lesson_relationship(relationship_data: Dict[str, Any]) -> str:
        """
        Create a relationship between two lessons.
        
        Args:
            relationship_data: Dictionary containing relationship information
                - source_id: Required. The ID of the source lesson
                - target_id: Required. The ID of the target lesson
                - relationship_type: Required. The type of relationship (e.g., "prerequisite", "related", "next")
                - metadata: Optional. Metadata for the relationship
                
        Returns:
            JSON response with operation result
        """
        try:
            # Extract required fields
            required_fields = ["source_id", "target_id", "relationship_type"]
            missing_fields = [field for field in required_fields if field not in relationship_data]
            
            if missing_fields:
                return dict_to_json(ErrorResponse.create(
                    message=f"Missing required fields: {', '.join(missing_fields)}",
                    code="missing_required_fields"
                ))
                
            result = graph_manager.create_lesson_relationship(relationship_data)
            
            if result["status"] == "error":
                return dict_to_json(ErrorResponse.create(
                    message=f"Failed to create lesson relationship: {result.get('message', 'Unknown error')}",
                    code="relationship_creation_error"
                ))
                
            return dict_to_json({
                "status": "success",
                "message": f"Relationship '{relationship_data['relationship_type']}' created successfully between lessons '{relationship_data['source_id']}' and '{relationship_data['target_id']}'",
                "relationship_id": result.get("relationship_id", ""),
                "timestamp": datetime.datetime.now().isoformat()
            })
            
        except Exception as e:
            logger.error(f"Error creating lesson relationship: {str(e)}")
            error_response = ErrorResponse.create(
                message=f"Failed to create lesson relationship: {str(e)}",
                code="relationship_creation_error"
            )
            return dict_to_json(error_response)

    # Lesson Content Analysis Tools
    @server.tool()
    async def analyze_lesson_prerequisites(lesson_id: str) -> str:
        """
        Analyze a lesson's content to identify potential prerequisites.
        
        Args:
            lesson_id: The ID of the lesson to analyze
                
        Returns:
            JSON response with analysis results
        """
        try:
            result = graph_manager.analyze_lesson_prerequisites(lesson_id)
            
            if result["status"] == "error":
                return dict_to_json(ErrorResponse.create(
                    message=f"Failed to analyze lesson prerequisites: {result.get('message', 'Unknown error')}",
                    code="analysis_error"
                ))
                
            return dict_to_json({
                "status": "success",
                "prerequisites": result["prerequisites"],
                "confidence_scores": result.get("confidence_scores", {}),
                "timestamp": datetime.datetime.now().isoformat()
            })
            
        except Exception as e:
            logger.error(f"Error analyzing lesson prerequisites: {str(e)}")
            error_response = ErrorResponse.create(
                message=f"Failed to analyze lesson prerequisites: {str(e)}",
                code="analysis_error"
            )
            return dict_to_json(error_response)
            
    @server.tool()
    async def find_related_lessons(lesson_id: str, similarity_threshold: float = 0.7) -> str:
        """
        Find lessons related to the specified lesson based on content similarity.
        
        Args:
            lesson_id: The ID of the lesson to find related lessons for
            similarity_threshold: Minimum similarity score threshold (0.0 to 1.0)
                
        Returns:
            JSON response with related lessons
        """
        try:
            result = graph_manager.find_related_lessons(lesson_id, similarity_threshold)
            
            if result["status"] == "error":
                return dict_to_json(ErrorResponse.create(
                    message=f"Failed to find related lessons: {result.get('message', 'Unknown error')}",
                    code="related_lessons_error"
                ))
                
            return dict_to_json({
                "status": "success",
                "related_lessons": result["related_lessons"],
                "similarity_scores": result.get("similarity_scores", {}),
                "timestamp": datetime.datetime.now().isoformat()
            })
            
        except Exception as e:
            logger.error(f"Error finding related lessons: {str(e)}")
            error_response = ErrorResponse.create(
                message=f"Failed to find related lessons: {str(e)}",
                code="related_lessons_error"
            )
            return dict_to_json(error_response)
            
    @server.tool()
    async def consolidate_related_lessons(lesson_ids: List[str]) -> str:
        """
        Consolidate multiple related lessons into a structured knowledge graph.
        
        Args:
            lesson_ids: List of lesson IDs to consolidate
                
        Returns:
            JSON response with consolidated structure
        """
        try:
            if not lesson_ids or not isinstance(lesson_ids, list):
                return dict_to_json(ErrorResponse.create(
                    message="Missing or invalid lesson_ids parameter",
                    code="invalid_parameter"
                ))
                
            result = graph_manager.consolidate_related_lessons(lesson_ids)
            
            if result["status"] == "error":
                return dict_to_json(ErrorResponse.create(
                    message=f"Failed to consolidate lessons: {result.get('message', 'Unknown error')}",
                    code="consolidation_error"
                ))
                
            return dict_to_json({
                "status": "success",
                "consolidated_structure": result["consolidated_structure"],
                "relationships": result.get("relationships", []),
                "timestamp": datetime.datetime.now().isoformat()
            })
            
        except Exception as e:
            logger.error(f"Error consolidating lessons: {str(e)}")
            error_response = ErrorResponse.create(
                message=f"Failed to consolidate lessons: {str(e)}",
                code="consolidation_error"
            )
            return dict_to_json(error_response)
            
    # Learning Path Tools
    @server.tool()
    async def generate_learning_path(topic: str, difficulty: Optional[str] = None, 
                                    max_lessons: int = 10) -> str:
        """
        Generate a personalized learning path for a given topic.
        
        Args:
            topic: The topic to generate a learning path for
            difficulty: Optional. Desired difficulty level
            max_lessons: Maximum number of lessons to include in the path
                
        Returns:
            JSON response with generated learning path
        """
        try:
            # Build parameters
            params = {
                "topic": topic,
                "max_lessons": max_lessons
            }
            if difficulty:
                params["difficulty"] = difficulty
                
            result = graph_manager.generate_learning_path(params)
            
            if result["status"] == "error":
                return dict_to_json(ErrorResponse.create(
                    message=f"Failed to generate learning path: {result.get('message', 'Unknown error')}",
                    code="learning_path_error"
                ))
                
            return dict_to_json({
                "status": "success",
                "learning_path": result["learning_path"],
                "estimated_completion_time": result.get("estimated_completion_time"),
                "difficulty_progression": result.get("difficulty_progression", []),
                "timestamp": datetime.datetime.now().isoformat()
            })
            
        except Exception as e:
            logger.error(f"Error generating learning path: {str(e)}")
            error_response = ErrorResponse.create(
                message=f"Failed to generate learning path: {str(e)}",
                code="learning_path_error"
            )
            return dict_to_json(error_response) 