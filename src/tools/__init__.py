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

__all__ = [
    "register_core_tools",
    "register_lesson_tools", 
    "register_project_tools",
    "register_config_tools",
    "register_all_tools"
]

def register_all_tools(server, graph_manager):
    """
    Register all memory tools with the server.
    
    Args:
        server: The server instance to register tools with
        graph_manager: The GraphMemoryManager instance
        
    Returns:
        None
    """
    register_core_tools(server, graph_manager)
    register_lesson_tools(server, graph_manager)
    register_project_tools(server, graph_manager)
    register_config_tools(server, graph_manager) 