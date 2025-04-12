#!/usr/bin/env python3
"""
Lesson Memory Tools

This module provides MCP tools for interacting with the Lesson Memory System
using the new layered approach in GraphMemoryManager.
"""

import json
from typing import Any, Dict, List, Optional, Union
from datetime import datetime

from src.logger import get_logger
from src.graph_memory import GraphMemoryManager

# Initialize logger
logger = get_logger()

# Initialize GraphMemoryManager singleton
_graph_manager = None

def get_graph_manager() -> GraphMemoryManager:
    """Get or initialize the GraphMemoryManager singleton instance."""
    global _graph_manager
    if _graph_manager is None:
        _graph_manager = GraphMemoryManager()
        _graph_manager.initialize()
    return _graph_manager

def register_lesson_tools(server, get_client_manager):
    """Register lesson memory tools with the server."""
    
    @server.tool()
    async def lesson_memory_tool(operation_type: str, **kwargs) -> str:
        """
        Manage lesson memory with a unified interface
        
        This tool provides a simplified interface to the Lesson Memory System,
        allowing AI agents to store and retrieve experiential knowledge in a
        structured way.
        
        Args:
            operation_type: The type of operation to perform
              - create_container: Create a lesson container
              - create: Create a new lesson
              - observe: Add structured observations to a lesson
              - relate: Create relationships between lessons
              - search: Find relevant lessons
              - track: Track lesson application
              - consolidate: Combine related lessons
              - evolve: Track lesson knowledge evolution
              - update: Update existing lessons
            **kwargs: Operation-specific parameters
                
        Returns:
            JSON response with operation results
            
        Required parameters by operation_type:
            - create_container: description (str, optional), metadata (dict, optional)
            - create: name (str), lesson_type (str), container_name (str, optional)
            - observe: entity_name (str), what_was_learned (str), why_it_matters (str), how_to_apply (str), container_name (str, optional), confidence (float, optional)
            - relate: source_name (str), target_name (str), relationship_type (str), container_name (str, optional)
            - search: query (str), limit (int, optional), container_name (str, optional)
            - track: lesson_name (str), project_name (str), success_score (float), application_notes (str)
            - update: name (str), properties (dict), container_name (str, optional)
            - consolidate: primary_lesson (str), lessons_to_consolidate (list), container_name (str, optional)
            - evolve: original_lesson (str), new_understanding (str), container_name (str, optional)
            
        Response format:
            All operations return a JSON string with at minimum:
            - status: "success" or "error"
            - message or error: Description of result or error

        Examples:
            ```
            # Create a new lesson container
            @lesson_memory_tool({
                "operation_type": "create_container",
                "description": "Container for React-related lessons",
                "metadata": {"category": "frontend", "framework": "react"}
            })
            
            # Create a new lesson
            @lesson_memory_tool({
                "operation_type": "create",
                "name": "ReactHookRules",
                "lesson_type": "BestPractice"
            })
            
            # Add observations to the lesson
            @lesson_memory_tool({
                "operation_type": "observe",
                "entity_name": "ReactHookRules",
                "what_was_learned": "React hooks must be called at the top level of components",
                "why_it_matters": "Hook call order must be consistent between renders",
                "how_to_apply": "Extract conditional logic into the hook implementation instead"
            })
            
            # Create a relationship
            @lesson_memory_tool({
                "operation_type": "relate",
                "source_name": "ReactHookRules",
                "target_name": "ReactPerformance",
                "relationship_type": "RELATED_TO"
            })
            
            # Search for lessons
            @lesson_memory_tool({
                "operation_type": "search",
                "query": "best practices for state management in React",
                "limit": 3
            })
            
            # Using with context
            # First get a context
            context = @lesson_memory_context({
                "project_name": "WebApp",
                "container_name": "ReactLessons"
            })
            
            # Then use context in operations
            @lesson_memory_tool({
                "operation_type": "create",
                "name": "StateManagementPatterns",
                "lesson_type": "Pattern",
                "context": context["context"]
            })
            ```
        """
        try:
            # Get the graph manager with the appropriate client context
            client_id = kwargs.pop("client_id", None)
            client_graph_manager = get_client_manager(client_id)
            
            # Validate operation type
            valid_operations = [
                "create_container", "create", "observe", "relate", "search", "track", 
                "consolidate", "evolve", "update"
            ]
            
            if operation_type not in valid_operations:
                logger.error(f"Invalid operation type: {operation_type}")
                return json.dumps({
                    "status": "error",
                    "error": f"Invalid operation type: {operation_type}. Valid operations are: {', '.join(valid_operations)}",
                    "code": "invalid_operation"
                })
            
            # Delegate to the GraphMemoryManager's lesson_operation method
            return client_graph_manager.lesson_operation(operation_type, **kwargs)
            
        except Exception as e:
            logger.error(f"Error in lesson_memory_tool: {str(e)}")
            return json.dumps({
                "status": "error",
                "error": f"Operation failed: {str(e)}",
                "code": "operation_error"
            })
    
    @server.tool()
    async def lesson_memory_context(context_data: Dict[str, Any], client_id: Optional[str] = None) -> str:
        """
        Create a context for batch lesson memory operations.
        
        This tool returns a context object that can be used for multiple
        lesson operations with shared project and container context.
        
        Args:
            context_data: Dictionary containing context information
                - project_name: Optional. Project name to set as context
                - container_name: Optional. Container name to use for operations (defaults to "Lessons")
            client_id: Optional client ID for identifying the connection
            
        Returns:
            JSON response with context information that includes:
            - status: "success" or "error"
            - message or error: Description of result or error
            - context: Context object with project_name, container_name, created_at timestamp,
                      available operations, and usage instructions
        
        Response structure:
            ```json
            {
                "status": "success",
                "message": "Lesson memory context created for project 'ProjectName' and container 'ContainerName'",
                "context": {
                    "project_name": "ProjectName",
                    "container_name": "ContainerName",
                    "created_at": "2023-07-15T10:30:45.123456",
                    "operations_available": ["create", "observe", "relate", "search", "track", "update", "consolidate", "evolve"],
                    "usage": "Use this context information with any lesson memory operation by including it in the operation's context parameter"
                }
            }
            ```
        
        Example:
            ```
            # Create a context for a specific project and container
            context = @lesson_memory_context({
                "project_name": "E-commerce Refactoring",
                "container_name": "PerformanceLessons"
            })
            
            # Use the context with another tool
            result = @lesson_memory_tool({
                "operation_type": "search",
                "query": "database optimization patterns",
                "context": context["context"]  # Pass the context object from the response
            })
            ```
        """
        try:
            # Get the graph manager with the appropriate client context
            # client_graph_manager = get_client_manager(client_id)
            
            # Extract context parameters
            project_name = context_data.get("project_name")
            container_name = context_data.get("container_name", "Lessons")
            
            # Create a context description rather than the actual context object
            # since we can't return Python objects through the API
            context_info = {
                "project_name": project_name,
                "container_name": container_name,
                "created_at": datetime.now().isoformat(),
                "operations_available": [
                    "create_container", "create", "observe", "relate", "search", 
                    "track", "update", "consolidate", "evolve"
                ],
                "usage": "Use this context information with any lesson memory operation by including it in the operation's context parameter"
            }
            
            # Create success response
            response = {
                "status": "success",
                "message": f"Lesson memory context created for project '{project_name}' and container '{container_name}'",
                "context": context_info
            }
            return json.dumps(response)
            
        except Exception as e:
            logger.error(f"Error creating lesson memory context: {str(e)}")
            return json.dumps({
                "status": "error",
                "error": f"Failed to create lesson memory context: {str(e)}",
                "code": "context_creation_error"
            })
    
    # Return a dictionary of all registered tools
    return {
        "lesson_memory_tool": lesson_memory_tool,
        "lesson_memory_context": lesson_memory_context
    } 