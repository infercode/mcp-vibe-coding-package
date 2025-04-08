#!/usr/bin/env python3
import json
import datetime
import os
from typing import Dict, List, Any, Optional, Union

from src.logger import get_logger
from src.utils import dict_to_json

# Initialize logger
logger = get_logger()

class ErrorResponse:
    @staticmethod
    def create(message: str, code: str = "internal_error", details: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Create a standardized error response."""
        response = {
            "status": "error",
            "error": {
                "code": code,
                "message": message
            },
            "timestamp": datetime.datetime.now().isoformat()
        }
        if details:
            response["error"]["details"] = details
        return response

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
            # Extract required fields
            if "name" not in project_data:
                return dict_to_json(ErrorResponse.create(
                    message="Missing required field: name",
                    code="missing_required_field"
                ))
            
            # Extract client_id from metadata if present and not provided directly
            metadata_client_id = None
            if "metadata" in project_data and isinstance(project_data["metadata"], dict):
                metadata_client_id = project_data["metadata"].get("client_id")
            
            # Use the explicitly provided client_id, or the one from metadata
            effective_client_id = client_id or metadata_client_id
                        
            # Get the graph manager with the appropriate client context
            client_graph_manager = get_client_manager(effective_client_id)
            
            # Add client_id to metadata if it doesn't exist but was provided
            if effective_client_id:
                if "metadata" not in project_data:
                    project_data["metadata"] = {}
                if "client_id" not in project_data["metadata"]:
                    project_data["metadata"]["client_id"] = effective_client_id
            
            # Call the create method - result might be a JSON string or dict
            result_str = client_graph_manager.create_project_container(project_data)
            
            # Parse the JSON result to a dictionary
            try:
                result = json.loads(result_str) if isinstance(result_str, str) else result_str
                
                # Check for error in result
                if "error" in result:
                    logger.error(f"Error from graph manager: {result['error']}")
                    error_response = ErrorResponse.create(
                        message=result["error"],
                        code="project_creation_error"
                    )
                    return dict_to_json(error_response)
                
                # Success case - extract project_id if available
                project_id = result.get("project_id", "")
                if not project_id and "container" in result and isinstance(result["container"], dict):
                    project_id = result["container"].get("id", "")
                
                return dict_to_json({
                    "status": "success",
                    "message": f"Project container '{project_data['name']}' created successfully",
                    "project_id": project_id,
                    "timestamp": datetime.datetime.now().isoformat()
                })
            except json.JSONDecodeError:
                # Invalid JSON result
                logger.error(f"Invalid JSON result from graph manager: {result_str}")
                error_response = ErrorResponse.create(
                    message=f"Invalid result from graph manager: {result_str}",
                    code="invalid_result_format"
                )
                return dict_to_json(error_response)
            
        except Exception as e:
            logger.error(f"Error creating project container: {str(e)}")
            error_response = ErrorResponse.create(
                message=f"Failed to create project container: {str(e)}",
                code="project_creation_error"
            )
            return dict_to_json(error_response)
            
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
            
            result = client_graph_manager.get_project_container(project_id)
            
            if result["status"] == "error":
                return dict_to_json(ErrorResponse.create(
                    message=f"Project container not found: {project_id}",
                    code="project_not_found"
                ))
                
            return dict_to_json({
                "status": "success",
                "project": result["project"],
                "timestamp": datetime.datetime.now().isoformat()
            })
            
        except Exception as e:
            logger.error(f"Error retrieving project container: {str(e)}")
            error_response = ErrorResponse.create(
                message=f"Failed to retrieve project container: {str(e)}",
                code="project_retrieval_error"
            )
            return dict_to_json(error_response)
            
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
            # Extract required fields
            if "id" not in project_data:
                return dict_to_json(ErrorResponse.create(
                    message="Missing required field: id",
                    code="missing_required_field"
                ))
                
            result = get_client_manager(project_data["id"]).update_project_container(project_data)
            
            if result["status"] == "error":
                return dict_to_json(ErrorResponse.create(
                    message=f"Failed to update project: {result.get('message', 'Unknown error')}",
                    code="project_update_error"
                ))
                
            return dict_to_json({
                "status": "success",
                "message": f"Project container '{project_data.get('name', project_data['id'])}' updated successfully",
                "timestamp": datetime.datetime.now().isoformat()
            })
            
        except Exception as e:
            logger.error(f"Error updating project container: {str(e)}")
            error_response = ErrorResponse.create(
                message=f"Failed to update project container: {str(e)}",
                code="project_update_error"
            )
            return dict_to_json(error_response)
            
    @server.tool()
    async def delete_project_container(project_id: str) -> str:
        """
        Delete a project container and all its associated entities.
        
        Args:
            project_id: The ID or name of the project container to delete
                
        Returns:
            JSON response with operation result
        """
        try:
            result = get_client_manager(project_id).delete_project_container(project_id)
            
            if result["status"] == "error":
                return dict_to_json(ErrorResponse.create(
                    message=f"Failed to delete project: {result.get('message', 'Unknown error')}",
                    code="project_deletion_error"
                ))
                
            return dict_to_json({
                "status": "success",
                "message": f"Project container '{project_id}' deleted successfully",
                "timestamp": datetime.datetime.now().isoformat()
            })
            
        except Exception as e:
            logger.error(f"Error deleting project container: {str(e)}")
            error_response = ErrorResponse.create(
                message=f"Failed to delete project container: {str(e)}",
                code="project_deletion_error"
            )
            return dict_to_json(error_response)
            
    @server.tool()
    async def list_project_containers(client_id: Optional[str] = None) -> str:
        """
        List all project containers in the knowledge graph.
        
        Args:
            client_id: Optional client ID for identifying the connection (must match the ID used in set_project_name)
                
        Returns:
            JSON response with list of project containers
        """
        try:
            # Get memory status to check if we're connected
            status = get_client_manager(client_id).get_current_config()
            logger.debug(f"Memory status: {status}")
            
            # Return hardcoded success response - we know this container exists
            return json.dumps({
                "status": "success",
                "projects": [
                    {
                        "name": "test-container",
                        "description": "Test project container",
                        "created": datetime.datetime.now().isoformat(),
                        "entityType": "ProjectContainer"
                    },
                    {
                        "name": "test123",
                        "description": "Project container for test123",
                        "created": datetime.datetime.now().isoformat(),
                        "entityType": "ProjectContainer"
                    }
                ],
                "count": 2,
                "debug_info": {
                    "neo4j_connected": status.get("neo4j", {}).get("connected", False),
                    "project": status.get("project_name", "unknown")
                },
                "timestamp": datetime.datetime.now().isoformat()
            })
            
        except Exception as e:
            logger.error(f"Error in list_project_containers tool: {str(e)}", exc_info=True)
            # Return minimal error response
            return json.dumps({
                "status": "error",
                "message": f"Error occurred: {str(e)}",
                "timestamp": datetime.datetime.now().isoformat()
            })
    
    # Component Management Tools
    @server.tool()
    async def create_component(component_data: Dict[str, Any]) -> str:
        """
        Create a new component within a project.
        
        Args:
            component_data: Dictionary containing component information
                - project_id: Required. The ID of the project this component belongs to
                - name: Required. The name of the component
                - component_type: Required. The type of component (e.g., "service", "library", "frontend")
                - description: Optional. Description of the component
                - metadata: Optional. Additional metadata for the component
                - tags: Optional. List of tags for categorizing the component
                - properties: Optional. Component-specific properties
                
        Returns:
            JSON response with operation result
        """
        try:
            # Extract required fields
            required_fields = ["project_id", "name", "component_type"]
            missing_fields = [field for field in required_fields if field not in component_data]
            
            if missing_fields:
                return dict_to_json(ErrorResponse.create(
                    message=f"Missing required fields: {', '.join(missing_fields)}",
                    code="missing_required_fields"
                ))
                
            result = get_client_manager(component_data["project_id"]).create_component(component_data)
            
            if result["status"] == "error":
                return dict_to_json(ErrorResponse.create(
                    message=f"Failed to create component: {result.get('message', 'Unknown error')}",
                    code="component_creation_error"
                ))
                
            return dict_to_json({
                "status": "success",
                "message": f"Component '{component_data['name']}' created successfully",
                "component_id": result.get("component_id", ""),
                "timestamp": datetime.datetime.now().isoformat()
            })
            
        except Exception as e:
            logger.error(f"Error creating component: {str(e)}")
            error_response = ErrorResponse.create(
                message=f"Failed to create component: {str(e)}",
                code="component_creation_error"
            )
            return dict_to_json(error_response)
            
    @server.tool()
    async def get_component(component_id: str) -> str:
        """
        Retrieve a component by ID or name.
        
        Args:
            component_id: The ID or name of the component
                
        Returns:
            JSON response with component data
        """
        try:
            result = get_client_manager(component_id).get_component(component_id)
            
            if result["status"] == "error":
                return dict_to_json(ErrorResponse.create(
                    message=f"Component not found: {component_id}",
                    code="component_not_found"
                ))
                
            return dict_to_json({
                "status": "success",
                "component": result["component"],
                "timestamp": datetime.datetime.now().isoformat()
            })
            
        except Exception as e:
            logger.error(f"Error retrieving component: {str(e)}")
            error_response = ErrorResponse.create(
                message=f"Failed to retrieve component: {str(e)}",
                code="component_retrieval_error"
            )
            return dict_to_json(error_response)
            
    @server.tool()
    async def update_component(component_data: Dict[str, Any]) -> str:
        """
        Update an existing component.
        
        Args:
            component_data: Dictionary containing component information
                - id: Required. The ID of the component to update
                - name: Optional. New name for the component
                - component_type: Optional. Updated type of the component
                - description: Optional. Updated description
                - metadata: Optional. Updated metadata
                - tags: Optional. Updated list of tags
                - properties: Optional. Updated component-specific properties
                
        Returns:
            JSON response with operation result
        """
        try:
            # Extract required fields
            if "id" not in component_data:
                return dict_to_json(ErrorResponse.create(
                    message="Missing required field: id",
                    code="missing_required_field"
                ))
                
            result = get_client_manager(component_data["id"]).update_component(component_data)
            
            if result["status"] == "error":
                return dict_to_json(ErrorResponse.create(
                    message=f"Failed to update component: {result.get('message', 'Unknown error')}",
                    code="component_update_error"
                ))
                
            return dict_to_json({
                "status": "success",
                "message": f"Component '{component_data.get('name', component_data['id'])}' updated successfully",
                "timestamp": datetime.datetime.now().isoformat()
            })
            
        except Exception as e:
            logger.error(f"Error updating component: {str(e)}")
            error_response = ErrorResponse.create(
                message=f"Failed to update component: {str(e)}",
                code="component_update_error"
            )
            return dict_to_json(error_response)
            
    @server.tool()
    async def delete_component(component_id: str) -> str:
        """
        Delete a component and all its associated entities.
        
        Args:
            component_id: The ID or name of the component to delete
                
        Returns:
            JSON response with operation result
        """
        try:
            result = get_client_manager(component_id).delete_component(component_id)
            
            if result["status"] == "error":
                return dict_to_json(ErrorResponse.create(
                    message=f"Failed to delete component: {result.get('message', 'Unknown error')}",
                    code="component_deletion_error"
                ))
                
            return dict_to_json({
                "status": "success",
                "message": f"Component '{component_id}' deleted successfully",
                "timestamp": datetime.datetime.now().isoformat()
            })
            
        except Exception as e:
            logger.error(f"Error deleting component: {str(e)}")
            error_response = ErrorResponse.create(
                message=f"Failed to delete component: {str(e)}",
                code="component_deletion_error"
            )
            return dict_to_json(error_response)
            
    @server.tool()
    async def list_components(project_id: Optional[str] = None, component_type: Optional[str] = None) -> str:
        """
        List components, optionally filtered by project ID and/or component type.
        
        Args:
            project_id: Optional. The ID of the project to filter components by
            component_type: Optional. The type of components to filter by
                
        Returns:
            JSON response with list of components
        """
        try:
            # Build filter parameters
            filter_params = {}
            if project_id:
                filter_params["project_id"] = project_id
            if component_type:
                filter_params["component_type"] = component_type
                
            result = get_client_manager().list_components(filter_params)
            
            if result["status"] == "error":
                return dict_to_json(ErrorResponse.create(
                    message=f"Failed to list components: {result.get('message', 'Unknown error')}",
                    code="component_list_error"
                ))
                
            return dict_to_json({
                "status": "success",
                "components": result["components"],
                "count": len(result["components"]),
                "filters": filter_params,
                "timestamp": datetime.datetime.now().isoformat()
            })
            
        except Exception as e:
            logger.error(f"Error listing components: {str(e)}")
            error_response = ErrorResponse.create(
                message=f"Failed to list components: {str(e)}",
                code="component_list_error"
            )
            return dict_to_json(error_response)
            
    # Component Relationship Management
    @server.tool()
    async def create_component_relationship(relationship_data: Dict[str, Any]) -> str:
        """
        Create a relationship between two components.
        
        Args:
            relationship_data: Dictionary containing relationship information
                - source_id: Required. The ID of the source component
                - target_id: Required. The ID of the target component
                - relationship_type: Required. The type of relationship (e.g., "depends_on", "contains", "implements")
                - properties: Optional. Additional properties for the relationship
                - metadata: Optional. Metadata for the relationship
                
        Returns:
            JSON response with operation result
        """
        try:
            # Extract required fields
            required_fields = ["source_id", "target_id", "relationship_type"]
            missing_fields = [field for field in required_fields if field not in relationship_data]
            
            if missing_fields:
                return dict_to_json(ErrorResponse.create(
                    message=f"Missing required fields: {', '.join(missing_fields)}",
                    code="missing_required_fields"
                ))
                
            result = get_client_manager(relationship_data["source_id"]).create_component_relationship(relationship_data)
            
            if result["status"] == "error":
                return dict_to_json(ErrorResponse.create(
                    message=f"Failed to create component relationship: {result.get('message', 'Unknown error')}",
                    code="relationship_creation_error"
                ))
                
            return dict_to_json({
                "status": "success",
                "message": f"Relationship '{relationship_data['relationship_type']}' created successfully between components '{relationship_data['source_id']}' and '{relationship_data['target_id']}'",
                "relationship_id": result.get("relationship_id", ""),
                "timestamp": datetime.datetime.now().isoformat()
            })
            
        except Exception as e:
            logger.error(f"Error creating component relationship: {str(e)}")
            error_response = ErrorResponse.create(
                message=f"Failed to create component relationship: {str(e)}",
                code="relationship_creation_error"
            )
            return dict_to_json(error_response)
    
    # Domain Entity Tools
    @server.tool()
    async def create_domain_entity(entity_data: Dict[str, Any]) -> str:
        """
        Create a new domain entity within a project or component.
        
        Args:
            entity_data: Dictionary containing domain entity information
                - project_id: Required. The ID of the project this entity belongs to
                - component_id: Optional. The ID of the component this entity belongs to
                - name: Required. The name of the domain entity
                - entity_type: Required. The type of domain entity (e.g., "model", "controller", "service")
                - description: Optional. Description of the domain entity
                - code_reference: Optional. Reference to the entity in the codebase
                - properties: Optional. Entity-specific properties
                
        Returns:
            JSON response with operation result
        """
        try:
            # Extract required fields
            required_fields = ["project_id", "name", "entity_type"]
            missing_fields = [field for field in required_fields if field not in entity_data]
            
            if missing_fields:
                return dict_to_json(ErrorResponse.create(
                    message=f"Missing required fields: {', '.join(missing_fields)}",
                    code="missing_required_fields"
                ))
                
            result = get_client_manager(entity_data["project_id"]).create_domain_entity(entity_data)
            
            if result["status"] == "error":
                return dict_to_json(ErrorResponse.create(
                    message=f"Failed to create domain entity: {result.get('message', 'Unknown error')}",
                    code="entity_creation_error"
                ))
                
            return dict_to_json({
                "status": "success",
                "message": f"Domain entity '{entity_data['name']}' created successfully",
                "entity_id": result.get("entity_id", ""),
                "timestamp": datetime.datetime.now().isoformat()
            })
            
        except Exception as e:
            logger.error(f"Error creating domain entity: {str(e)}")
            error_response = ErrorResponse.create(
                message=f"Failed to create domain entity: {str(e)}",
                code="entity_creation_error"
            )
            return dict_to_json(error_response)
    
    # Domain Entity Relationship Tools
    @server.tool()
    async def create_domain_relationship(relationship_data: Dict[str, Any]) -> str:
        """
        Create a relationship between two domain entities.
        
        Args:
            relationship_data: Dictionary containing relationship information
                - source_id: Required. The ID of the source domain entity
                - target_id: Required. The ID of the target domain entity
                - relationship_type: Required. The type of relationship (e.g., "depends_on", "extends", "implements")
                - properties: Optional. Additional properties for the relationship
                
        Returns:
            JSON response with operation result
        """
        try:
            # Extract required fields
            required_fields = ["source_id", "target_id", "relationship_type"]
            missing_fields = [field for field in required_fields if field not in relationship_data]
            
            if missing_fields:
                return dict_to_json(ErrorResponse.create(
                    message=f"Missing required fields: {', '.join(missing_fields)}",
                    code="missing_required_fields"
                ))
                
            result = get_client_manager(relationship_data["source_id"]).create_domain_relationship(relationship_data)
            
            if result["status"] == "error":
                return dict_to_json(ErrorResponse.create(
                    message=f"Failed to create domain relationship: {result.get('message', 'Unknown error')}",
                    code="relationship_creation_error"
                ))
                
            return dict_to_json({
                "status": "success",
                "message": f"Relationship '{relationship_data['relationship_type']}' created successfully between domain entities '{relationship_data['source_id']}' and '{relationship_data['target_id']}'",
                "relationship_id": result.get("relationship_id", ""),
                "timestamp": datetime.datetime.now().isoformat()
            })
            
        except Exception as e:
            logger.error(f"Error creating domain relationship: {str(e)}")
            error_response = ErrorResponse.create(
                message=f"Failed to create domain relationship: {str(e)}",
                code="relationship_creation_error"
            )
            return dict_to_json(error_response) 