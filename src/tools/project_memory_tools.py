#!/usr/bin/env python3
"""
Project Memory Tools

This module provides MCP tools for interacting with the Project Memory System
using the layered approach in GraphMemoryManager.
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

def register_project_tools(server, get_client_manager):
    """Register project memory tools with the server."""
    
    @server.tool()
    async def project_memory_tool(operation_type: str, **kwargs) -> str:
        """
        Manage project memory with a unified interface
        
        This tool provides a simplified interface to the Project Memory System,
        allowing AI agents to store and retrieve structured project knowledge.
        
        Args:
            operation_type: The type of operation to perform
              - create_project: Create a new project container
              - create_component: Create a component within a project
              - create_domain_entity: Create a domain entity
              - relate_entities: Create relationships between entities
              - search: Find relevant project entities
              - get_structure: Retrieve project hierarchy
              - add_observation: Add observations to entities
              - update: Update existing entities
              - delete_entity: Delete a project entity (project, domain, component, or observation)
              - delete_relationship: Delete a relationship between entities
            **kwargs: Operation-specific parameters
                
        Returns:
            JSON response with operation results
            
        Required parameters by operation_type:
            - create_project: name (str)
            - create_component: name (str), component_type (str), project_id (str)
            - create_domain_entity: name (str), entity_type (str), project_id (str)
            - relate_entities: source_name (str), target_name (str), relation_type (str), project_id (str)
            - search: query (str), project_id (str)
            - get_structure: project_id (str)
            - add_observation: entity_name (str), content (str)
            - update: entity_name (str), updates (dict)
            - delete_entity: entity_name (str), entity_type (str)
            - delete_relationship: source_name (str), target_name (str), relationship_type (str)
            
        Optional parameters by operation_type:
            - create_project: description (str), metadata (dict), tags (list)
            - create_component: domain_name (str), description (str), content (str), metadata (dict)
            - create_domain_entity: description (str), properties (dict)
            - relate_entities: domain_name (str), entity_type (str), properties (dict)
            - search: entity_types (list), limit (int), semantic (bool), domain_name (str)
            - get_structure: include_components (bool), include_domains (bool), include_relationships (bool), max_depth (int)
            - add_observation: project_id (str), observation_type (str), entity_type (str), domain_name (str)
            - update: project_id (str), entity_type (str), domain_name (str)
            - delete_entity: container_name (str), domain_name (str), delete_contents (bool), observation_id (str)
            - delete_relationship: container_name (str), domain_name (str), relationship_type (str)
            
        Response format:
            All operations return a JSON string with at minimum:
            - status: "success" or "error"
            - message or error: Description of result or error

        Examples:
            ```
            # Create a new project
            @project_memory_tool({
                "operation_type": "create_project",
                "name": "E-commerce Platform",
                "description": "Online store with microservices architecture"
            })
            
            # Create a component within the project
            @project_memory_tool({
                "operation_type": "create_component",
                "project_id": "E-commerce Platform",
                "name": "Authentication Service",
                "component_type": "MICROSERVICE"
            })
            
            # Create a domain entity
            @project_memory_tool({
                "operation_type": "create_domain_entity",
                "project_id": "E-commerce Platform",
                "entity_type": "DECISION",
                "name": "Use JWT for Auth"
            })
            
            # Create a relationship
            @project_memory_tool({
                "operation_type": "relate_entities",
                "source_name": "Authentication Service",
                "target_name": "User Database",
                "relationship_type": "DEPENDS_ON",
                "project_id": "E-commerce Platform"
            })
            
            # Search for entities
            @project_memory_tool({
                "operation_type": "search",
                "query": "authentication patterns",
                "project_id": "E-commerce Platform",
                "limit": 5
            })
            
            # Delete a component
            @project_memory_tool({
                "operation_type": "delete_entity",
                "entity_name": "Payment Gateway",
                "entity_type": "component",
                "container_name": "E-commerce Platform",
                "domain_name": "Payment"
            })
            
            # Delete a relationship
            @project_memory_tool({
                "operation_type": "delete_relationship",
                "source_name": "Authentication Service",
                "target_name": "User Database",
                "relationship_type": "DEPENDS_ON",
                "container_name": "E-commerce Platform",
                "domain_name": "Backend"
            })
            
            # Using with context
            # First get a context
            context = @project_memory_context({
                "project_name": "E-commerce Platform"
            })
            
            # Then use context in operations
            @project_memory_tool({
                "operation_type": "create_component",
                "name": "Payment Processor",
                "component_type": "SERVICE",
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
                "create_project", "create_component", "create_domain_entity", 
                "relate_entities", "search", "get_structure", 
                "add_observation", "update"
            ]
            
            if operation_type not in valid_operations:
                logger.error(f"Invalid operation type: {operation_type}")
                return json.dumps({
                    "status": "error",
                    "error": f"Invalid operation type: {operation_type}. Valid operations are: {', '.join(valid_operations)}",
                    "code": "invalid_operation"
                })
            
            # Delegate to the GraphMemoryManager's project_operation method
            return client_graph_manager.project_operation(operation_type, **kwargs)
            
        except Exception as e:
            logger.error(f"Error in project_memory_tool: {str(e)}")
            return json.dumps({
                "status": "error",
                "error": f"Operation failed: {str(e)}",
                "code": "operation_error"
            })
    
    @server.tool()
    async def project_memory_context(context_data: Dict[str, Any], client_id: Optional[str] = None) -> str:
        """
        Create a context for batch project memory operations.
        
        This tool returns a context object that can be used for multiple
        project operations with shared project context.
        
        Args:
            context_data: Dictionary containing context information
                - project_name: Project name to set as context
            client_id: Optional client ID for identifying the connection
            
        Returns:
            JSON response with context information that includes:
            - status: "success" or "error"
            - message or error: Description of result or error
            - context: Context object with project_name, created_at timestamp,
                      available operations, and usage instructions
        
        Response structure:
            ```json
            {
                "status": "success",
                "message": "Project memory context created for project 'ProjectName'",
                "context": {
                    "project_name": "ProjectName",
                    "created_at": "2023-07-15T10:30:45.123456",
                    "operations_available": ["create_component", "create_domain_entity", "relate_entities", "search", "get_structure", "add_observation", "update"],
                    "usage": "Use this context information with any project memory operation by including it in the operation's context parameter"
                }
            }
            ```
        
        Example:
            ```
            # Create a context for a specific project
            context = @project_memory_context({
                "project_name": "E-commerce Platform"
            })
            
            # Use the context with another tool
            result = @project_memory_tool({
                "operation_type": "search",
                "query": "authentication patterns",
                "context": context["context"]  # Pass the context object from the response
            })
            ```
        """
        try:
            # Extract context parameters
            project_name = context_data.get("project_name")
            
            if not project_name:
                return json.dumps({
                    "status": "error",
                    "error": "Project name is required for creating a project memory context",
                    "code": "missing_project_name"
                })
            
            # Create a context description rather than the actual context object
            # since we can't return Python objects through the API
            context_info = {
                "project_name": project_name,
                "created_at": datetime.now().isoformat(),
                "operations_available": [
                    "create_component", "create_domain_entity", "relate_entities", 
                    "search", "get_structure", "add_observation", "update"
                ],
                "usage": "Use this context information with any project memory operation by including it in the operation's context parameter"
            }
            
            # Create success response
            response = {
                "status": "success",
                "message": f"Project memory context created for project '{project_name}'",
                "context": context_info
            }
            return json.dumps(response)
            
        except Exception as e:
            logger.error(f"Error creating project memory context: {str(e)}")
            return json.dumps({
                "status": "error",
                "error": f"Failed to create project memory context: {str(e)}",
                "code": "context_creation_error"
            })
    
    # Return a dictionary of all registered tools
    return {
        "project_memory_tool": project_memory_tool,
        "project_memory_context": project_memory_context
    } 