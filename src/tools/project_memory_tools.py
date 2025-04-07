#!/usr/bin/env python3
"""
Project Memory Tools with Pydantic Integration

This module implements MCP tools for the project memory system using
Pydantic models for validation and serialization.
"""

from typing import Dict, List, Any, Optional, Union, TypeVar, Type, cast
from datetime import datetime
import json
import re

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
            # Parameter validation
            if not project_id:
                error_response = create_error_response(
                    message="Project ID is required",
                    code="missing_parameter"
                )
                return model_to_json(error_response)
                
            # Sanitize input
            project_id = str(project_id).strip()
            
            # Reject certain dangerous inputs
            restricted_names = ["all", "*", "database", "system", "admin"]
            if project_id.lower() in restricted_names:
                error_response = create_error_response(
                    message=f"Invalid project ID: {project_id}",
                    code="invalid_parameter"
                )
                return model_to_json(error_response)
                
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
                project_container = ProjectContainerUpdate(**project_data)
            except Exception as e:
                logger.error(f"Validation error for project update data: {str(e)}")
                error_response = create_error_response(
                    message=f"Invalid project update data: {str(e)}",
                    code="validation_error"
                )
                return model_to_json(error_response)
                
            # Get the client ID from metadata if present
            client_id = None
            if project_data.get("metadata") and isinstance(project_data["metadata"], dict):
                client_id = project_data["metadata"].get("client_id")
            
            # Get the graph manager with the appropriate client context
            client_graph_manager = get_client_manager(client_id)
            
            # Call the update method with validated data
            result = client_graph_manager.update_project_container(model_to_dict(project_container))
            
            # Process the result
            if isinstance(result, str):
                try:
                    result = json.loads(result)
                except json.JSONDecodeError:
                    result = {"error": f"Invalid JSON result: {result}"}
            
            # Check for error in result
            if "error" in result or (isinstance(result, dict) and result.get("status") == "error"):
                error_message = result.get("error", result.get("message", "Unknown error"))
                error_response = create_error_response(
                    message=f"Failed to update project: {error_message}",
                    code="project_update_error"
                )
                return model_to_json(error_response)
            
            # Create success response
            response = create_success_response(
                message=f"Project container '{project_data.get('name', project_data['id'])}' updated successfully",
                data={}
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
            # Get the graph manager with the appropriate client context
            client_graph_manager = get_client_manager(client_id)
            
            # List project containers
            result = client_graph_manager.list_project_containers()
            
            # Process the result
            if isinstance(result, str):
                try:
                    result = json.loads(result)
                except json.JSONDecodeError:
                    result = {"error": f"Invalid JSON result: {result}"}
            
            # Check for error in result
            if "error" in result or (isinstance(result, dict) and result.get("status") == "error"):
                error_message = result.get("error", result.get("message", "Unknown error"))
                error_response = create_error_response(
                    message=f"Failed to list project containers: {error_message}",
                    code="project_listing_error"
                )
                return model_to_json(error_response)
            
            # Extract container data
            containers = []
            if isinstance(result, dict) and "projects" in result:
                containers = result["projects"] if isinstance(result["projects"], list) else []
            
            # Create success response
            response = create_success_response(
                message=f"Found {len(containers)} project containers",
                data={"projects": containers}
            )
            return model_to_json(response)
            
        except Exception as e:
            logger.error(f"Error listing project containers: {str(e)}")
            error_response = create_error_response(
                message=f"Failed to list project containers: {str(e)}",
                code="project_listing_error"
            )
            return model_to_json(error_response)
    
    @server.tool()
    async def delete_project_container(project_id: str, client_id: Optional[str] = None) -> str:
        """
        Delete a project container and all its associated entities.
        
        Args:
            project_id: The ID or name of the project container to delete
            client_id: Optional client ID for identifying the connection (must match the ID used in set_project_name)
                
        Returns:
            JSON response with operation result
        """
        try:
            # Parameter validation
            if not project_id:
                error_response = create_error_response(
                    message="Project ID is required",
                    code="missing_parameter"
                )
                return model_to_json(error_response)
                
            # Sanitize input
            project_id = str(project_id).strip()
            
            # Reject certain dangerous inputs
            restricted_names = ["all", "*", "database", "system", "admin"]
            if project_id.lower() in restricted_names:
                error_response = create_error_response(
                    message=f"Invalid project ID: {project_id}. Cannot delete protected resources.",
                    code="invalid_parameter"
                )
                return model_to_json(error_response)
                
            # Get the graph manager with the appropriate client context
            client_graph_manager = get_client_manager(client_id)
            
            # Delete project container
            result = client_graph_manager.delete_project_container(project_id)
            
            # Check for error
            if isinstance(result, dict) and "status" in result and result["status"] == "error":
                error_response = create_error_response(
                    message=f"Failed to delete project container: {result.get('message', 'Unknown error')}",
                    code="project_deletion_error"
                )
                return model_to_json(error_response)
            
            # Create success response
            response = create_success_response(
                message=f"Project container '{project_id}' deleted successfully",
                data={}
            )
            return model_to_json(response)
            
        except Exception as e:
            logger.error(f"Error deleting project container: {str(e)}")
            error_response = create_error_response(
                message=f"Failed to delete project container: {str(e)}",
                code="project_deletion_error"
            )
            return model_to_json(error_response)
            
    @server.tool()
    async def create_component(component_data: Dict[str, Any], client_id: Optional[str] = None) -> str:
        """
        Create a new component in a project.
        
        Args:
            component_data: Dictionary containing component information
                - project_id: Required. The ID or name of the project this component belongs to
                - name: Required. Name of the component
                - component_type: Required. Type of component (e.g., "class", "function", "module")
                - description: Optional. Description of the component
                - content: Optional. Content of the component (e.g., code)
                - metadata: Optional. Additional metadata for the component
                - dependencies: Optional. List of dependencies for this component
            client_id: Optional client ID for identifying the connection
                
        Returns:
            JSON response with operation result
        """
        try:
            # Validate input using Pydantic model
            try:
                # Add client_id to metadata if provided
                if client_id:
                    if "metadata" not in component_data:
                        component_data["metadata"] = {}
                    component_data["metadata"]["client_id"] = client_id
                
                # Create Pydantic model for validation
                component_model = ComponentCreate(**component_data)
            except Exception as e:
                logger.error(f"Validation error for component data: {str(e)}")
                error_response = create_error_response(
                    message=f"Invalid component data: {str(e)}",
                    code="validation_error"
                )
                return model_to_json(error_response)
            
            # Get the graph manager with the appropriate client context
            client_graph_manager = get_client_manager(client_id)
            
            # Call the project memory domain manager to create the component
            result = client_graph_manager.project_memory.domain_manager.create_component(model_to_dict(component_model))
            
            # Check for error
            if isinstance(result, dict) and "status" in result and result["status"] == "error":
                error_response = create_error_response(
                    message=f"Failed to create component: {result.get('message', 'Unknown error')}",
                    code="component_creation_error"
                )
                return model_to_json(error_response)
            
            # Extract component data
            component_id = ""
            component_data = {}
            
            if isinstance(result, dict):
                component_id = result.get("component_id", "")
                if "component" in result and isinstance(result["component"], dict):
                    component_id = component_id or result["component"].get("id", "")
                    component_data = result["component"]
            
            # Create success response
            response = ComponentResponse(
                status="success",
                message=f"Component '{component_model.name}' created successfully",
                timestamp=datetime.now(),
                component_id=component_id,
                component=component_data
            )
            return model_to_json(response)
            
        except Exception as e:
            logger.error(f"Error creating component: {str(e)}")
            error_response = create_error_response(
                message=f"Failed to create component: {str(e)}",
                code="component_creation_error"
            )
            return model_to_json(error_response)
    
    @server.tool()
    async def get_component(component_id: str, project_id: Optional[str] = None, client_id: Optional[str] = None) -> str:
        """
        Retrieve a component by ID or name.
        
        Args:
            component_id: The ID or name of the component
            project_id: Optional ID or name of the project to scope the search
            client_id: Optional client ID for identifying the connection
                
        Returns:
            JSON response with component data
        """
        try:
            # Parameter validation
            if not component_id:
                error_response = create_error_response(
                    message="Component ID is required",
                    code="missing_parameter"
                )
                return model_to_json(error_response)
                
            # Sanitize inputs
            component_id = str(component_id).strip()
            
            # Sanitize project_id if provided
            if project_id:
                project_id = str(project_id).strip()
                
                # Reject certain dangerous inputs for project_id
                restricted_names = ["all", "*", "database", "system", "admin"]
                if project_id.lower() in restricted_names:
                    error_response = create_error_response(
                        message=f"Invalid project ID: {project_id}",
                        code="invalid_parameter"
                    )
                    return model_to_json(error_response)
            
            # Get the graph manager with the appropriate client context
            client_graph_manager = get_client_manager(client_id)
            
            # If project_id is provided, set it as the current project
            if project_id:
                client_graph_manager.set_project_name(project_id)
            
            # Get component from project memory
            result = client_graph_manager.project_memory.domain_manager.get_component(component_id)
            
            # Check for error
            if isinstance(result, dict) and "status" in result and result["status"] == "error":
                error_response = create_error_response(
                    message=f"Component not found: {component_id}",
                    code="component_not_found"
                )
                return model_to_json(error_response)
            
            # Extract component data
            component_data = {}
            if isinstance(result, dict) and "component" in result:
                component_data = result["component"] if isinstance(result["component"], dict) else {}
            
            # Create success response
            response = ComponentResponse(
                status="success",
                message=f"Component retrieved successfully",
                timestamp=datetime.now(),
                component_id=component_id,
                component=component_data
            )
            return model_to_json(response)
            
        except Exception as e:
            logger.error(f"Error retrieving component: {str(e)}")
            error_response = create_error_response(
                message=f"Failed to retrieve component: {str(e)}",
                code="component_retrieval_error"
            )
            return model_to_json(error_response)
    
    @server.tool()
    async def update_component(component_data: Dict[str, Any], client_id: Optional[str] = None) -> str:
        """
        Update an existing component.
        
        Args:
            component_data: Dictionary containing component information
                - id: Required. The ID of the component to update
                - project_id: Optional. The ID or name of the project this component belongs to
                - name: Optional. Updated name for the component
                - component_type: Optional. Updated type of component
                - description: Optional. Updated description of the component
                - content: Optional. Updated content of the component
                - metadata: Optional. Updated metadata for the component
                - dependencies: Optional. Updated list of dependencies
            client_id: Optional client ID for identifying the connection
                
        Returns:
            JSON response with operation result
        """
        try:
            # Validate input using Pydantic model
            try:
                # Add client_id to metadata if provided
                if client_id:
                    if "metadata" not in component_data:
                        component_data["metadata"] = {}
                    component_data["metadata"]["client_id"] = client_id
                
                # Get the project_id from the input data if it exists
                project_id = component_data.get("project_id")
                
                # Create Pydantic model for validation
                component_model = ComponentUpdate(**component_data)
            except Exception as e:
                logger.error(f"Validation error for component update data: {str(e)}")
                error_response = create_error_response(
                    message=f"Invalid component update data: {str(e)}",
                    code="validation_error"
                )
                return model_to_json(error_response)
            
            # Get the graph manager with the appropriate client context
            client_graph_manager = get_client_manager(client_id)
            
            # If project_id is provided, set it as the current project
            if project_id:
                client_graph_manager.set_project_name(project_id)
            
            # Call the project memory domain manager to update the component
            result = client_graph_manager.project_memory.domain_manager.update_component(model_to_dict(component_model))
            
            # Check for error
            if isinstance(result, dict) and "status" in result and result["status"] == "error":
                error_response = create_error_response(
                    message=f"Failed to update component: {result.get('message', 'Unknown error')}",
                    code="component_update_error"
                )
                return model_to_json(error_response)
            
            # Extract updated component data
            component_data = {}
            if isinstance(result, dict) and "component" in result:
                component_data = result["component"] if isinstance(result["component"], dict) else {}
            
            # Create success response
            response = ComponentResponse(
                status="success",
                message=f"Component updated successfully",
                timestamp=datetime.now(),
                component_id=component_model.id,
                component=component_data
            )
            return model_to_json(response)
            
        except Exception as e:
            logger.error(f"Error updating component: {str(e)}")
            error_response = create_error_response(
                message=f"Failed to update component: {str(e)}",
                code="component_update_error"
            )
            return model_to_json(error_response)
    
    @server.tool()
    async def delete_component(component_id: str, project_id: Optional[str] = None, client_id: Optional[str] = None) -> str:
        """
        Delete a component from a project.
        
        Args:
            component_id: The ID or name of the component to delete
            project_id: Optional ID or name of the project to scope the search
            client_id: Optional client ID for identifying the connection
                
        Returns:
            JSON response with operation result
        """
        try:
            # Parameter validation
            if not component_id:
                error_response = create_error_response(
                    message="Component ID is required",
                    code="missing_parameter"
                )
                return model_to_json(error_response)
                
            # Sanitize inputs
            component_id = str(component_id).strip()
            
            # Sanitize project_id if provided
            if project_id:
                project_id = str(project_id).strip()
                
                # Reject certain dangerous inputs for project_id
                restricted_names = ["all", "*", "database", "system", "admin"]
                if project_id.lower() in restricted_names:
                    error_response = create_error_response(
                        message=f"Invalid project ID: {project_id}",
                        code="invalid_parameter"
                    )
                    return model_to_json(error_response)
            
            # Get the graph manager with the appropriate client context
            client_graph_manager = get_client_manager(client_id)
            
            # If project_id is provided, set it as the current project
            if project_id:
                client_graph_manager.set_project_name(project_id)
            
            # Delete component
            result = client_graph_manager.project_memory.domain_manager.delete_component(component_id)
            
            # Check for error
            if isinstance(result, dict) and "status" in result and result["status"] == "error":
                error_response = create_error_response(
                    message=f"Failed to delete component: {result.get('message', 'Unknown error')}",
                    code="component_deletion_error"
                )
                return model_to_json(error_response)
            
            # Create success response
            response = create_success_response(
                message=f"Component '{component_id}' deleted successfully",
                data={}
            )
            return model_to_json(response)
            
        except Exception as e:
            logger.error(f"Error deleting component: {str(e)}")
            error_response = create_error_response(
                message=f"Failed to delete component: {str(e)}",
                code="component_deletion_error"
            )
            return model_to_json(error_response)
    
    @server.tool()
    async def list_components(
        project_id: str, 
        component_type: Optional[str] = None, 
        name_contains: Optional[str] = None,
        limit: Optional[int] = None,
        client_id: Optional[str] = None
    ) -> str:
        """
        List components in a project, optionally filtered by type or name.
        
        Args:
            project_id: The ID or name of the project
            component_type: Optional component type to filter by (e.g., "class", "function")
            name_contains: Optional string to search in component names
            limit: Optional maximum number of results to return
            client_id: Optional client ID for identifying the connection
                
        Returns:
            JSON response with components matching the criteria
        """
        try:
            # Parameter validation
            if not project_id:
                error_response = create_error_response(
                    message="Project ID is required",
                    code="missing_parameter"
                )
                return model_to_json(error_response)
                
            # Sanitize inputs
            project_id = str(project_id).strip()
            
            # Reject certain dangerous inputs for project_id
            restricted_names = ["all", "*", "database", "system", "admin"]
            if project_id.lower() in restricted_names:
                error_response = create_error_response(
                    message=f"Invalid project ID: {project_id}",
                    code="invalid_parameter"
                )
                return model_to_json(error_response)
                
            # Sanitize optional parameters
            if component_type:
                component_type = str(component_type).strip()
                
            if name_contains:
                name_contains = str(name_contains).strip()
                
            if limit is not None:
                try:
                    limit = int(limit)
                    if limit < 1:
                        limit = 10  # Use default if negative
                    elif limit > 1000:
                        limit = 1000  # Cap to prevent performance issues
                except (ValueError, TypeError):
                    limit = 100  # Default if conversion fails
            
            # Get the graph manager with the appropriate client context
            client_graph_manager = get_client_manager(client_id)
            
            # Set the project context
            client_graph_manager.set_project_name(project_id)
            
            # Prepare filters
            filters = {}
            if component_type:
                filters["component_type"] = component_type
            if name_contains:
                filters["name_contains"] = name_contains
            if limit:
                filters["limit"] = limit
            
            # List components
            result = client_graph_manager.project_memory.domain_manager.list_components(filters)
            
            # Check for error
            if isinstance(result, dict) and "status" in result and result["status"] == "error":
                error_response = create_error_response(
                    message=f"Failed to list components: {result.get('message', 'Unknown error')}",
                    code="component_listing_error"
                )
                return model_to_json(error_response)
            
            # Extract components
            components = []
            if isinstance(result, dict) and "components" in result:
                components = result["components"] if isinstance(result["components"], list) else []
            
            # Create success response
            response = create_success_response(
                message=f"Found {len(components)} components in project '{project_id}'",
                data={"components": components}
            )
            return model_to_json(response)
            
        except Exception as e:
            logger.error(f"Error listing components: {str(e)}")
            error_response = create_error_response(
                message=f"Failed to list components: {str(e)}",
                code="component_listing_error"
            )
            return model_to_json(error_response)

    @server.tool()
    async def create_domain_entity(entity_data: Dict[str, Any], client_id: Optional[str] = None) -> str:
        """
        Create a domain entity in the project.
        
        Args:
            entity_data: Dictionary containing entity information
                - project_id: Required. The ID or name of the project
                - name: Required. Name of the entity
                - entity_type: Required. Type of entity
                - properties: Optional. Additional properties for the entity
                - metadata: Optional. Additional metadata
            client_id: Optional client ID for identifying the connection
                
        Returns:
            JSON response with operation result
        """
        try:
            # Validate input using Pydantic model
            try:
                # Add client_id to metadata if provided
                if client_id:
                    if "metadata" not in entity_data:
                        entity_data["metadata"] = {}
                    entity_data["metadata"]["client_id"] = client_id
                
                # Extract project_id before validation
                project_id = entity_data.get("project_id")
                
                # Create Pydantic model for validation
                entity_model = DomainEntityCreate(**entity_data)
            except Exception as e:
                logger.error(f"Validation error for domain entity data: {str(e)}")
                error_response = create_error_response(
                    message=f"Invalid domain entity data: {str(e)}",
                    code="validation_error"
                )
                return model_to_json(error_response)
            
            # Get the graph manager with the appropriate client context
            client_graph_manager = get_client_manager(client_id)
            
            # Set the project context
            if project_id:
                client_graph_manager.set_project_name(project_id)
            
            # Call the project memory domain manager to create the entity
            result = client_graph_manager.project_memory.domain_manager.create_domain_entity(model_to_dict(entity_model))
            
            # Check for error
            if isinstance(result, dict) and "status" in result and result["status"] == "error":
                error_response = create_error_response(
                    message=f"Failed to create domain entity: {result.get('message', 'Unknown error')}",
                    code="entity_creation_error"
                )
                return model_to_json(error_response)
            
            # Extract entity data
            entity_id = ""
            entity_data = {}
            
            if isinstance(result, dict):
                entity_id = result.get("entity_id", "")
                if "entity" in result and isinstance(result["entity"], dict):
                    entity_id = entity_id or result["entity"].get("id", "")
                    entity_data = result["entity"]
            
            # Create success response
            response = create_success_response(
                message=f"Domain entity '{entity_model.name}' created successfully",
                data={"entity_id": entity_id, "entity": entity_data}
            )
            return model_to_json(response)
            
        except Exception as e:
            logger.error(f"Error creating domain entity: {str(e)}")
            error_response = create_error_response(
                message=f"Failed to create domain entity: {str(e)}",
                code="entity_creation_error"
            )
            return model_to_json(error_response)
    
    @server.tool()
    async def create_domain_relationship(relationship_data: Dict[str, Any], client_id: Optional[str] = None) -> str:
        """
        Create a relationship between domain entities.
        
        Args:
            relationship_data: Dictionary containing relationship information
                - project_id: Required. The ID or name of the project
                - from_entity: Required. Name or ID of the source entity
                - to_entity: Required. Name or ID of the target entity
                - relationship_type: Required. Type of relationship
                - properties: Optional. Additional properties for the relationship
                - metadata: Optional. Additional metadata
            client_id: Optional client ID for identifying the connection
                
        Returns:
            JSON response with operation result
        """
        try:
            # Validate input using Pydantic model
            try:
                # Add client_id to metadata if provided
                if client_id:
                    if "metadata" not in relationship_data:
                        relationship_data["metadata"] = {}
                    relationship_data["metadata"]["client_id"] = client_id
                
                # Extract project_id before validation
                project_id = relationship_data.get("project_id")
                
                # Create Pydantic model for validation
                relationship_model = RelationshipCreate(**relationship_data)
            except Exception as e:
                logger.error(f"Validation error for relationship data: {str(e)}")
                error_response = create_error_response(
                    message=f"Invalid relationship data: {str(e)}",
                    code="validation_error"
                )
                return model_to_json(error_response)
            
            # Get the graph manager with the appropriate client context
            client_graph_manager = get_client_manager(client_id)
            
            # Set the project context
            if project_id:
                client_graph_manager.set_project_name(project_id)
            
            # Call the project memory domain manager to create the relationship
            result = client_graph_manager.project_memory.domain_manager.create_domain_relationship(model_to_dict(relationship_model))
            
            # Check for error
            if isinstance(result, dict) and "status" in result and result["status"] == "error":
                error_response = create_error_response(
                    message=f"Failed to create relationship: {result.get('message', 'Unknown error')}",
                    code="relationship_creation_error"
                )
                return model_to_json(error_response)
            
            # Extract relationship data
            relationship_id = ""
            relationship_data = {}
            
            if isinstance(result, dict):
                relationship_id = result.get("relationship_id", "")
                if "relationship" in result and isinstance(result["relationship"], dict):
                    relationship_id = relationship_id or result["relationship"].get("id", "")
                    relationship_data = result["relationship"]
            
            # Create success response
            response = create_success_response(
                message=f"Domain relationship created successfully",
                data={"relationship_id": relationship_id, "relationship": relationship_data}
            )
            return model_to_json(response)
            
        except Exception as e:
            logger.error(f"Error creating domain relationship: {str(e)}")
            error_response = create_error_response(
                message=f"Failed to create domain relationship: {str(e)}",
                code="relationship_creation_error"
            )
            return model_to_json(error_response)

    @server.tool()
    async def create_component_relationship(relationship_data: Dict[str, Any], client_id: Optional[str] = None) -> str:
        """
        Create a relationship between components in a project.
        
        Args:
            relationship_data: Dictionary containing relationship information
                - source_id: Required. ID or name of the source component
                - target_id: Required. ID or name of the target component
                - relationship_type: Required. Type of relationship (e.g., "DEPENDS_ON", "IMPORTS", "CALLS")
                - project_id: Optional. ID or name of the project (extracted from source/target if not provided)
                - properties: Optional. Additional properties for the relationship
                - metadata: Optional. Additional metadata
            client_id: Optional client ID for identifying the connection
                
        Returns:
            JSON response with operation result
        """
        try:
            # Validate required fields
            if not relationship_data:
                error_response = create_error_response(
                    message="Relationship data is required",
                    code="missing_data"
                )
                return model_to_json(error_response)
                
            # Check for required fields
            for field in ["source_id", "target_id", "relationship_type"]:
                if field not in relationship_data or not relationship_data[field]:
                    error_response = create_error_response(
                        message=f"Missing required field: {field}",
                        code="missing_field"
                    )
                    return model_to_json(error_response)
            
            # Sanitize inputs
            relationship_data["source_id"] = str(relationship_data["source_id"]).strip()
            relationship_data["target_id"] = str(relationship_data["target_id"]).strip()
            relationship_data["relationship_type"] = str(relationship_data["relationship_type"]).strip()
            
            # Validate relationship_type format
            if not re.match(r'^[A-Za-z0-9_]+$', relationship_data["relationship_type"]):
                error_response = create_error_response(
                    message="Invalid relationship_type format. Use only alphanumeric characters and underscores.",
                    code="invalid_format"
                )
                return model_to_json(error_response)
                
            # Add client_id to metadata if provided
            if client_id:
                if "metadata" not in relationship_data:
                    relationship_data["metadata"] = {}
                relationship_data["metadata"]["client_id"] = client_id
            
            # Extract project_id for context setting
            project_id = relationship_data.get("project_id")
            if project_id:
                relationship_data["project_id"] = str(project_id).strip()
            
            # Get the graph manager with the appropriate client context
            client_graph_manager = get_client_manager(client_id)
            
            # Set the project context if project_id is provided
            if project_id:
                client_graph_manager.set_project_name(relationship_data["project_id"])
            
            # Call the create method
            result = client_graph_manager.project_memory.domain_manager.create_component_relationship(relationship_data)
            
            # Check for error
            if isinstance(result, dict) and "status" in result and result["status"] == "error":
                error_response = create_error_response(
                    message=f"Failed to create component relationship: {result.get('message', 'Unknown error')}",
                    code="relationship_creation_error"
                )
                return model_to_json(error_response)
            
            # Extract relationship data
            relationship_id = ""
            relationship_data = {}
            
            if isinstance(result, dict):
                relationship_id = result.get("relationship_id", "")
                if "relationship" in result and isinstance(result["relationship"], dict):
                    relationship_id = relationship_id or result["relationship"].get("id", "")
                    relationship_data = result["relationship"]
            
            # Create success response
            response = create_success_response(
                message=f"Component relationship created successfully",
                data={"relationship_id": relationship_id, "relationship": relationship_data}
            )
            return model_to_json(response)
            
        except Exception as e:
            logger.error(f"Error creating component relationship: {str(e)}")
            error_response = create_error_response(
                message=f"Failed to create component relationship: {str(e)}",
                code="relationship_creation_error"
            )
            return model_to_json(error_response)

    return {
        "create_project_container": create_project_container,
        "get_project_container": get_project_container,
        "update_project_container": update_project_container,
        "list_project_containers": list_project_containers,
        "delete_project_container": delete_project_container,
        "create_component": create_component,
        "get_component": get_component,
        "update_component": update_component,
        "delete_component": delete_component,
        "list_components": list_components,
        "create_domain_entity": create_domain_entity,
        "create_domain_relationship": create_domain_relationship,
        "create_component_relationship": create_component_relationship,
    } 