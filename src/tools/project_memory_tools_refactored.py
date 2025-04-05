#!/usr/bin/env python3
"""
Project Memory Tools with Pydantic Integration

This module implements MCP tools for the project memory system using
Pydantic models for validation and serialization.
"""

from typing import Dict, List, Any, Optional, Union, TypeVar, Type, cast
from datetime import datetime
import json

from src.logger import get_logger
from src.models.project_memory import (
    ProjectContainerCreate, ProjectContainerUpdate, ComponentCreate, ComponentUpdate,
    DomainEntityCreate, RelationshipCreate, SearchQuery,
    ProjectContainerResponse, ComponentResponse, SearchResponse
)
from src.models.responses import (
    create_error_response, create_success_response,
    model_to_json, model_to_dict
)

# Initialize logger
logger = get_logger()


def register_project_tools(server, get_client_manager):
    """Register project memory tools with the server."""
    
    # Project Container Management Tools
    @server.tool()
    async def create_project_container(project_data: Dict[str, Any], client_id: Optional[str] = None) -> str:
        """
        Create a new project container in the knowledge graph.
        
        IMPORTANT: Always check first if a project container with the same name 
        already exists using list_project_containers() before creating a new one. Creating
        multiple containers for the same project leads to fragmented knowledge and confusion.
        
        Args:
            project_data: Dictionary containing project information
                - name: Required. The name of the project container
                - description: Optional. Description of the project
                - metadata: Optional. Additional metadata for the project
                - tags: Optional. List of tags for categorizing the project
            client_id: Optional client ID for identifying the connection (must match the ID used in set_project_name)
                
        Returns:
            JSON response with operation result
        """
        try:
            # Validate input using Pydantic model
            try:
                # Extract client_id from metadata if present and not provided directly
                metadata_client_id = None
                if "metadata" in project_data and isinstance(project_data["metadata"], dict):
                    metadata_client_id = project_data["metadata"].get("client_id")
                
                # Use the explicitly provided client_id, or the one from metadata
                effective_client_id = client_id or metadata_client_id
                
                # Add client_id to metadata if it doesn't exist but was provided
                if effective_client_id:
                    if "metadata" not in project_data:
                        project_data["metadata"] = {}
                    project_data["metadata"]["client_id"] = effective_client_id
                
                # Create Pydantic model for validation
                project_container = ProjectContainerCreate(**project_data)
            except Exception as e:
                logger.error(f"Validation error for project data: {str(e)}")
                error_response = create_error_response(
                    message=f"Invalid project data: {str(e)}",
                    code="validation_error"
                )
                return model_to_json(error_response)
            
            # Get the graph manager with the appropriate client context
            client_graph_manager = get_client_manager(effective_client_id)
            
            # Call the create method with validated data
            result = client_graph_manager.create_project_container(model_to_dict(project_container))
            
            # Process the result
            try:
                # Handle string result (legacy format)
                if isinstance(result, str):
                    try:
                        result = json.loads(result)
                    except json.JSONDecodeError:
                        result = {"error": f"Invalid JSON result: {result}"}
                
                # Check for error in result
                if isinstance(result, dict) and "status" in result and result["status"] == "error":
                    logger.error(f"Error from graph manager: {result.get('error', 'Unknown error')}")
                    error_response = create_error_response(
                        message=result.get("error", "Unknown error"),
                        code="project_creation_error"
                    )
                    return model_to_json(error_response)
                
                # Extract project_id
                project_id = ""
                container_data = {}
                
                if isinstance(result, dict):
                    project_id = result.get("project_id", "")
                    if "container" in result and isinstance(result["container"], dict):
                        project_id = project_id or result["container"].get("id", "")
                        container_data = result["container"]
                
                # Create success response
                response = ProjectContainerResponse(
                    status="success",
                    message=f"Project container '{project_container.name}' created successfully",
                    timestamp=datetime.now(),
                    project_id=project_id,
                    project=container_data
                )
                return model_to_json(response)
                
            except Exception as e:
                logger.error(f"Error processing result: {str(e)}")
                error_response = create_error_response(
                    message=f"Error processing result: {str(e)}",
                    code="result_processing_error"
                )
                return model_to_json(error_response)
            
        except Exception as e:
            logger.error(f"Error creating project container: {str(e)}")
            error_response = create_error_response(
                message=f"Failed to create project container: {str(e)}",
                code="project_creation_error"
            )
            return model_to_json(error_response)
            
    @server.tool()
    async def get_project_container(project_id: str, client_id: Optional[str] = None) -> str:
        """
        Retrieve a project container by ID or name.
        
        Args:
            project_id: The ID or name of the project container
            client_id: Optional client ID for identifying the connection (must match the ID used in set_project_name)
                
        Returns:
            JSON response with project container data
        """
        try:
            # Get the graph manager with the appropriate client context
            client_graph_manager = get_client_manager(client_id)
            
            # Get project container
            result = client_graph_manager.get_project_container(project_id)
            
            # Check for error
            if isinstance(result, dict) and "status" in result and result["status"] == "error":
                error_response = create_error_response(
                    message=f"Project container not found: {project_id}",
                    code="project_not_found"
                )
                return model_to_json(error_response)
            
            # Create success response
            project_data = {}
            if isinstance(result, dict) and "project" in result:
                project_data = result["project"] if isinstance(result["project"], dict) else {}
            
            response = ProjectContainerResponse(
                status="success",
                message=f"Project container retrieved successfully",
                timestamp=datetime.now(),
                project_id=project_id,
                project=project_data
            )
            return model_to_json(response)
            
        except Exception as e:
            logger.error(f"Error retrieving project container: {str(e)}")
            error_response = create_error_response(
                message=f"Failed to retrieve project container: {str(e)}",
                code="project_retrieval_error"
            )
            return model_to_json(error_response)
            
    @server.tool()
    async def update_project_container(project_data: Dict[str, Any]) -> str:
        """
        Update an existing project container.
        
        Args:
            project_data: Dictionary containing project information
                - id: Required. The ID of the project container to update
                - name: Optional. New name for the project
                - description: Optional. New description for the project
                - metadata: Optional. Updated metadata for the project
                - tags: Optional. Updated list of tags
                
        Returns:
            JSON response with operation result
        """
        try:
            # Validate input using Pydantic model
            try:
                # Create Pydantic model for validation
                project_update = ProjectContainerUpdate(**project_data)
            except Exception as e:
                logger.error(f"Validation error for project update data: {str(e)}")
                error_response = create_error_response(
                    message=f"Invalid project update data: {str(e)}",
                    code="validation_error"
                )
                return model_to_json(error_response)
            
            # Get the graph manager
            result = get_client_manager(project_data.get("id")).update_project_container(
                model_to_dict(project_update)
            )
            
            # Check for error
            if isinstance(result, dict) and "status" in result and result["status"] == "error":
                error_response = create_error_response(
                    message=f"Failed to update project: {result.get('message', 'Unknown error')}",
                    code="project_update_error"
                )
                return model_to_json(error_response)
            
            # Create success response
            project_name = project_data.get("name", project_data.get("id", ""))
            project_id = project_data.get("id", "")
            
            project_data = {}
            if isinstance(result, dict) and "project" in result:
                project_data = result["project"] if isinstance(result["project"], dict) else {}
                
            response = ProjectContainerResponse(
                status="success",
                message=f"Project container '{project_name}' updated successfully",
                timestamp=datetime.now(),
                project_id=project_id,
                project=project_data
            )
            return model_to_json(response)
            
        except Exception as e:
            logger.error(f"Error updating project container: {str(e)}")
            error_response = create_error_response(
                message=f"Failed to update project container: {str(e)}",
                code="project_update_error"
            )
            return model_to_json(error_response)
            
    @server.tool()
    async def list_project_containers(client_id: Optional[str] = None) -> str:
        """
        List all project containers in the knowledge graph.
        
        Args:
            client_id: Optional client ID for identifying the connection
                
        Returns:
            JSON response with list of project containers
        """
        try:
            # Get the graph manager
            client_graph_manager = get_client_manager(client_id)
            
            # Get project containers
            result = client_graph_manager.list_project_containers()
            
            # Check for error
            if isinstance(result, dict) and "status" in result and result["status"] == "error":
                error_response = create_error_response(
                    message=f"Failed to list project containers: {result.get('message', 'Unknown error')}",
                    code="project_list_error"
                )
                return model_to_json(error_response)
            
            # Create success response
            projects = []
            if isinstance(result, dict) and "projects" in result:
                projects = result["projects"] if isinstance(result["projects"], list) else []
                
            response = create_success_response(
                message=f"Retrieved {len(projects)} project containers",
                data={"projects": projects}
            )
            return model_to_json(response)
            
        except Exception as e:
            logger.error(f"Error listing project containers: {str(e)}")
            error_response = create_error_response(
                message=f"Failed to list project containers: {str(e)}",
                code="project_list_error"
            )
            return model_to_json(error_response)
    
    # Additional component, domain entity, and relationship tools would follow
    # the same pattern - using Pydantic models for validation and response formatting
    
    return {
        "create_project_container": create_project_container,
        "get_project_container": get_project_container,
        "update_project_container": update_project_container,
        "list_project_containers": list_project_containers,
    } 