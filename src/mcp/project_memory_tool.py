"""
Project Memory Tool for MCP.

This module provides a unified interface for interacting with project memory
through the GraphMemoryManager. It enables AI agents to create, query, and
manage hierarchical project knowledge in a structured way.
"""

import json
from typing import Any, Dict, List, Optional, Union

# Import the GraphMemoryManager
from src.graph_memory import GraphMemoryManager

# Singleton instance of GraphMemoryManager
_graph_manager = None

def _get_graph_manager() -> GraphMemoryManager:
    """
    Get or initialize the GraphMemoryManager singleton.
    
    Returns:
        The initialized GraphMemoryManager instance
    """
    global _graph_manager
    if _graph_manager is None:
        _graph_manager = GraphMemoryManager()
        _graph_manager.initialize()
    return _graph_manager

def project_memory_tool(operation_type: str, **kwargs) -> str:
    """
    Manage project memory with a unified interface.
    
    Args:
        operation_type: The type of operation to perform
          - create_project: Create a new project
          - create_component: Create a component within a project
          - create_domain_entity: Create a domain entity
          - relate_entities: Create relationships between entities
          - search: Find relevant project entities
          - get_structure: Retrieve project hierarchy
          - add_observation: Add observations to entities
          - update: Update existing entities
        **kwargs: Operation-specific parameters
          
    Returns:
        JSON response string with operation results
    """
    gm = _get_graph_manager()
    return gm.project_operation(operation_type, **kwargs)

def project_context(project_name: str):
    """
    Context manager for performing multiple operations within a project context.
    
    Args:
        project_name: Name of the project to use as context
        
    Returns:
        A context manager that yields a ProjectContext object
        
    Example:
        ```python
        with project_context("MyProject") as project:
            # Create a domain
            domain_result = project.create_domain("Authentication")
            
            # Create a component in that domain
            component_result = project.create_component(
                "AuthService", 
                "Microservice", 
                "Authentication",
                description="Handles user authentication"
            )
            
            # Create a relationship
            relation_result = project.relate(
                "AuthService", 
                "UserDatabase", 
                "DEPENDS_ON",
                entity_type="component", 
                domain_name="Authentication"
            )
        ```
    """
    gm = _get_graph_manager()
    return gm.project_context(project_name)

# Example usage functions

def create_project_example(name: str, description: Optional[str] = None) -> str:
    """
    Example of creating a project.
    
    Args:
        name: Name of the project to create
        description: Optional description
        
    Returns:
        JSON response with created project
    """
    return project_memory_tool(
        operation_type="create_project",
        name=name,
        description=description
    )

def create_domain_example(project_id: str, name: str, description: Optional[str] = None) -> str:
    """
    Example of creating a domain within a project.
    
    Args:
        project_id: Name of the project
        name: Name of the domain to create
        description: Optional description
        
    Returns:
        JSON response with created domain
    """
    return project_memory_tool(
        operation_type="create_domain_entity",
        name=name,
        entity_type="Domain",
        project_id=project_id,
        description=description
    )

def create_component_example(project_id: str, domain_name: str, name: str, 
                           component_type: str, description: Optional[str] = None) -> str:
    """
    Example of creating a component within a domain.
    
    Args:
        project_id: Name of the project
        domain_name: Name of the domain
        name: Name of the component to create
        component_type: Type of the component (e.g., 'Service', 'Module')
        description: Optional description
        
    Returns:
        JSON response with created component
    """
    return project_memory_tool(
        operation_type="create_component",
        name=name,
        component_type=component_type,
        project_id=project_id,
        domain_name=domain_name,
        description=description
    )

def search_example(project_id: str, query: str, semantic: bool = False) -> str:
    """
    Example of searching for entities within a project.
    
    Args:
        project_id: Name of the project
        query: Search query text
        semantic: Whether to use semantic search
        
    Returns:
        JSON response with search results
    """
    return project_memory_tool(
        operation_type="search",
        query=query,
        project_id=project_id,
        semantic=semantic
    )

def project_context_example() -> None:
    """
    Example of using the project context for multiple operations.
    
    This function demonstrates how to use the project_context context manager
    to perform multiple operations within a project context.
    """
    with project_context("ExampleProject") as project:
        # Create a domain
        domain_result = project.create_domain("Backend")
        
        # Create a component in that domain
        component_result = project.create_component(
            "ApiService", 
            "Microservice", 
            "Backend",
            description="Handles API requests"
        )
        
        # Create another component
        db_result = project.create_component(
            "Database", 
            "Infrastructure", 
            "Backend",
            description="Stores application data"
        )
        
        # Create a relationship between components
        relation_result = project.relate(
            "ApiService", 
            "Database", 
            "DEPENDS_ON",
            entity_type="component", 
            domain_name="Backend"
        )
        
        # Search for components
        search_result = project.search(
            "API",
            entity_types=["Component"],
            semantic=True
        )
        
        # Print the results (in a real application, you would parse and use these results)
        print("Domain created:", domain_result)
        print("API Service created:", component_result)
        print("Database created:", db_result)
        print("Relationship created:", relation_result)
        print("Search results:", search_result) 