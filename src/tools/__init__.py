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

# Import consolidated tools
from src.registry.consolidated_tools import register_consolidated_registry_tools

# Import essential registry tools
from src.registry.core_registry_tools import register_essential_registry_tools

__all__ = [
    "register_core_tools",
    "register_lesson_tools", 
    "register_project_tools",
    "register_config_tools",
    "register_registry_tools",
    "register_all_tools",
    "register_consolidated_tools",
    "register_essential_tools"
]

def register_all_tools(server, manager_or_getter):
    """
    Register all memory tools with the server.
    
    This registers every individual function as a separate tool,
    which can lead to a large number of tools.
    
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

def register_consolidated_tools(server, manager_or_getter):
    """
    Register consolidated tools with the server.
    
    This uses the category-based approach that groups functions by namespace,
    significantly reducing the total number of exposed tools.
    
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
        # Use consolidated registry tools instead of individual registry tools
        register_consolidated_registry_tools(server, manager_or_getter)
    else:
        # manager_or_getter is a direct GraphMemoryManager instance
        # Create a simple getter function that always returns this manager
        def get_fixed_manager():
            return manager_or_getter
            
        register_core_tools(server, get_fixed_manager)
        register_lesson_tools(server, get_fixed_manager)
        register_project_tools(server, get_fixed_manager)
        register_config_tools(server, get_fixed_manager)
        # Use consolidated registry tools instead of individual registry tools
        register_consolidated_registry_tools(server, get_fixed_manager)

def register_essential_tools(server, manager_or_getter):
    """
    Register only the essential registry tools with the server.
    
    This exposes the absolute minimum number of tools needed for
    the Function Registry Pattern to work, while allowing access to
    all other functions through these tools.
    
    Args:
        server: The server instance to register tools with
        manager_or_getter: Either a GraphMemoryManager instance or a function 
                         that returns a GraphMemoryManager
        
    Returns:
        None
    """
    # Only register the essential registry tools
    # All other functionality will be accessible through these tools
    if callable(manager_or_getter):
        register_essential_registry_tools(server, manager_or_getter)
    else:
        def get_fixed_manager():
            return manager_or_getter
        register_essential_registry_tools(server, get_fixed_manager) 