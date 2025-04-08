#!/usr/bin/env python3
"""
MCP Memory Tools

This module contains tools for interacting with the MCP memory systems,
including core memory, lesson memory, project memory, and configuration tools.
"""

from src.tools.core_memory_tools import register_core_tools
from src.tools.lesson_memory_tools import register_lesson_tools
from src.tools.project_memory_tools import register_project_tools
from src.tools.config_tools import register_config_tools

# Import registry tools
from src.registry.registry_tools import register_registry_tools

__all__ = [
    "register_core_tools",
    "register_lesson_tools", 
    "register_project_tools",
    "register_config_tools",
    "register_registry_tools",
    "register_all_tools"
]

def register_all_tools(server, manager_or_getter):
    """
    Register all memory tools with the server.
    
    Args:
        server: The server instance to register tools with
        manager_or_getter: Either a GraphMemoryManager instance or a function 
                         that returns a GraphMemoryManager
        
    Returns:
        None
    """
    # Check if manager_or_getter is a callable (function that returns a GraphMemoryManager)
    if callable(manager_or_getter):
        # manager_or_getter is a getter function that returns the appropriate manager
        register_core_tools(server, manager_or_getter)
        register_lesson_tools(server, manager_or_getter)
        register_project_tools(server, manager_or_getter)
        register_config_tools(server, manager_or_getter)
        register_registry_tools(server, manager_or_getter)
    else:
        # manager_or_getter is a direct GraphMemoryManager instance
        # Create a simple getter function that always returns this manager
        def get_fixed_manager():
            return manager_or_getter
            
        register_core_tools(server, get_fixed_manager)
        register_lesson_tools(server, get_fixed_manager)
        register_project_tools(server, get_fixed_manager)
        register_config_tools(server, get_fixed_manager)
        register_registry_tools(server, get_fixed_manager) 