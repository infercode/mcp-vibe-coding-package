#!/usr/bin/env python3
"""
Lesson Memory Tools

This module provides MCP tools for interacting with the Lesson Memory System
using the new layered approach in GraphMemoryManager.
"""

import json
from typing import Any, Dict, List, Optional, Union

from src.graph_memory import GraphMemoryManager

# Initialize GraphMemoryManager singleton
_graph_manager = None

def get_graph_manager() -> GraphMemoryManager:
    """Get or initialize the GraphMemoryManager singleton instance."""
    global _graph_manager
    if _graph_manager is None:
        _graph_manager = GraphMemoryManager()
        _graph_manager.initialize()
    return _graph_manager

def lesson_memory_tool(operation_type: str, **kwargs) -> str:
    """
    Manage lesson memory with a unified interface
    
    This tool provides a simplified interface to the Lesson Memory System,
    allowing AI agents to store and retrieve experiential knowledge in a
    structured way.
    
    Args:
        operation_type: The type of operation to perform
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
        
    Examples:
        ```python
        # Create a new lesson
        lesson_memory_tool(
            operation_type="create",
            name="ReactHookRules",
            lesson_type="BestPractice"
        )
        
        # Add observations to the lesson
        lesson_memory_tool(
            operation_type="observe",
            entity_name="ReactHookRules",
            what_was_learned="React hooks must be called at the top level of components",
            why_it_matters="Hook call order must be consistent between renders",
            how_to_apply="Extract conditional logic into the hook implementation instead"
        )
        
        # Create a relationship
        lesson_memory_tool(
            operation_type="relate",
            source_name="ReactHookRules",
            target_name="ReactPerformance",
            relationship_type="RELATED_TO"
        )
        
        # Search for lessons
        lesson_memory_tool(
            operation_type="search",
            query="best practices for state management in React",
            limit=3
        )
        ```
    """
    graph_manager = get_graph_manager()
    
    # Delegate to the GraphMemoryManager's lesson_operation method
    return graph_manager.lesson_operation(operation_type, **kwargs)

def lesson_memory_context(project_name: Optional[str] = None, container_name: Optional[str] = None) -> Dict[str, Any]:
    """
    Start a lesson memory context for batch operations
    
    This tool is designed for use with the 'with' statement in Python to maintain
    consistent context across multiple lesson memory operations.
    
    Args:
        project_name: Optional project name to set as context
        container_name: Optional container name to use
        
    Returns:
        A context object with bound methods for lesson operations
        
    Note:
        This tool is primarily intended for use in scripts rather than as a direct
        MCP tool due to the context manager pattern, but is provided for completeness.
    """
    graph_manager = get_graph_manager()
    
    # Create an info response about the context object
    return {
        "message": "Lesson context created",
        "project_name": project_name or graph_manager.default_project_name,
        "container_name": container_name or "Lessons",
        "usage": "This is meant to be used with Python's 'with' statement in scripts."
    }