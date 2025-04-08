#!/usr/bin/env python3
"""
Example Functions for Registry Pattern

This module contains example functions that demonstrate how to register
functions with the registry.
"""

from typing import Dict, List, Any, Optional
from src.registry.registry_manager import register_function
from src.registry.function_models import FunctionResult

# Memory Functions
@register_function(namespace="memory")
def get_entity(entity_name: str, include_observations: bool = False) -> Dict[str, Any]:
    """
    Get an entity from the knowledge graph.
    
    Args:
        entity_name: Name of the entity to retrieve
        include_observations: Whether to include observations
        
    Returns:
        Entity data
    """
    # This is just an example - in a real implementation, this would
    # call the actual memory system
    return {
        "entity_id": entity_name,
        "entity_type": "EXAMPLE",
        "observations": [] if not include_observations else [
            {"content": "This is an example observation"}
        ]
    }

@register_function(namespace="memory")
def search_entities(query: str, limit: int = 10) -> Dict[str, Any]:
    """
    Search for entities in the knowledge graph.
    
    Args:
        query: Search query
        limit: Maximum number of results
        
    Returns:
        Search results
    """
    # Example implementation
    return {
        "query": query,
        "results": [
            {"entity_id": f"Result {i}", "score": 0.9 - (i * 0.1)}
            for i in range(min(5, limit))
        ],
        "total": min(5, limit)
    }

# Project Functions
@register_function(namespace="project")
def get_project(project_name: str) -> Dict[str, Any]:
    """
    Get a project from the project memory.
    
    Args:
        project_name: Name of the project
        
    Returns:
        Project data
    """
    return {
        "project_id": project_name,
        "components": ["Example Component 1", "Example Component 2"]
    }

@register_function(namespace="project")
def list_projects(limit: int = 10) -> Dict[str, Any]:
    """
    List projects in the project memory.
    
    Args:
        limit: Maximum number of projects to return
        
    Returns:
        List of projects
    """
    return {
        "projects": [
            {"project_id": f"Project {i}"}
            for i in range(min(3, limit))
        ],
        "total": min(3, limit)
    }

# Utility Functions
@register_function(namespace="utils")
def format_json(data: Dict[str, Any], pretty: bool = True) -> Dict[str, Any]:
    """
    Format JSON data.
    
    Args:
        data: Data to format
        pretty: Whether to format prettily
        
    Returns:
        Formatted data
    """
    return {
        "original": data,
        "formatted": "Pretty" if pretty else "Compact",
        "size": len(str(data))
    }

@register_function(namespace="utils")
def validate_query(query: str) -> Dict[str, Any]:
    """
    Validate a search query.
    
    Args:
        query: Query to validate
        
    Returns:
        Validation result
    """
    is_valid = len(query) > 3 and ";" not in query
    
    return {
        "query": query,
        "is_valid": is_valid,
        "issues": [] if is_valid else ["Query too short or contains invalid characters"]
    }

# Config Functions
@register_function(namespace="config")
def get_config(config_key: str) -> Dict[str, Any]:
    """
    Get a configuration value.
    
    Args:
        config_key: Key to get
        
    Returns:
        Configuration value
    """
    # Example configs
    configs = {
        "search.limit": 10,
        "search.threshold": 0.7,
        "memory.embeddings.model": "text-embedding-ada-002"
    }
    
    if config_key in configs:
        return {
            "key": config_key,
            "value": configs[config_key]
        }
    else:
        return FunctionResult.error(
            message=f"Config key '{config_key}' not found",
            error_code="config_not_found"
        ) 