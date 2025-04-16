from typing import Any, Dict, List, Optional, Union
import time
import json
from datetime import datetime

from src.utils import dict_to_json, generate_id
from src.graph_memory.base_manager import BaseManager
from src.models.project_memory import (
    ProjectContainer as ProjectContainerModel,
    ProjectContainerCreate,
    ProjectContainerUpdate,
    ComponentCreate,
    ComponentUpdate,
    DomainEntityCreate,
    RelationshipCreate,
    SearchQuery,
    ErrorResponse,
    ErrorDetail,
    SuccessResponse,
    ProjectContainerResponse
)

class ProjectContainer:
    """
    Manager for project containers in the memory system.
    Handles creation, retrieval, and management of project containers.
    """
    
    def __init__(self, base_manager: BaseManager):
        """
        Initialize the project container manager.
        
        Args:
            base_manager: Base manager for graph operations
        """
        self.base_manager = base_manager
        self.logger = base_manager.logger
    
    def create_project_container(self, project_data: ProjectContainerCreate) -> str:
        """
        Create a new project container.
        
        Args:
            project_data: ProjectContainerCreate model containing project information
                - name: Required. The name of the project container (unique identifier)
                - description: Optional. Description of the project
                - metadata: Optional. Additional metadata for the project
                - tags: Optional. List of tags for categorizing the project
            
        Returns:
            JSON string with the created container
        """
        try:
            self.base_manager.ensure_initialized()
            
            name = project_data.name
            
            # Check if a container with this name already exists
            check_query = """
            MATCH (c:Entity {name: $name, entityType: 'ProjectContainer'})
            RETURN c
            """
            
            # Use safe_execute_read_query for validation (this is a read-only operation)
            records = self.base_manager.safe_execute_read_query(
                check_query,
                {"name": name}
            )
            
            if records and len(records) > 0:
                return dict_to_json({
                    "error": f"A project container with name '{name}' already exists"
                })
            
            # Generate a unique ID for the container
            container_id = generate_id("prj")
            
            # Prepare container properties
            container_properties = {
                "id": container_id,
                "name": name,
                "entityType": "ProjectContainer",
                "domain": "project",
                "status": "active"
            }
            
            # Add description if provided
            if project_data.description:
                container_properties["description"] = project_data.description
                
            # Add metadata if provided
            if project_data.metadata:
                metadata_dict = project_data.metadata.model_dump(exclude_none=True)
                for key, value in metadata_dict.items():
                    container_properties[key] = value
                    
            # Add tags if provided
            if project_data.tags:
                container_properties["tags"] = json.dumps(project_data.tags)
            
            # Create container
            create_query = """
            CREATE (c:Entity $properties)
            SET c.created = datetime(),
                c.lastUpdated = datetime()
            RETURN c
            """
            
            # Use safe_execute_write_query for validation (this is a write operation)
            create_records = self.base_manager.safe_execute_write_query(
                create_query,
                {"properties": container_properties}
            )
            
            if not create_records or len(create_records) == 0:
                return dict_to_json({
                    "error": "Failed to create project container"
                })
            
            # Return created container
            container = dict(create_records[0]["c"].items())
            
            # Create response using Pydantic response model
            response = ProjectContainerResponse(
                message=f"Project container '{name}' created successfully",
                project_id=container_id,
                project=container,
                timestamp=datetime.now()
            )
            
            return dict_to_json(response.model_dump(exclude_none=True))
                
        except Exception as e:
            error_msg = f"Error creating project container: {str(e)}"
            self.logger.error(error_msg)
            error_response = ErrorResponse(
                error=ErrorDetail(
                    code="ERROR",
                    message=error_msg,
                    details={}
                ),
                timestamp=datetime.now()
            )
            return dict_to_json(error_response.model_dump())
    
    def get_project_container(self, name: str) -> str:
        """
        Retrieve a project container by name.
        
        Args:
            name: Name of the project container
            
        Returns:
            JSON string with the container details
        """
        try:
            self.base_manager.ensure_initialized()
            
            # Get container query
            query = """
            MATCH (c:Entity {name: $name, entityType: 'ProjectContainer'})
            OPTIONAL MATCH (c)<-[:PART_OF]-(component:Entity)
            RETURN c, count(component) as component_count
            """
            
            # Use safe_execute_read_query for validation (this is a read-only operation)
            records = self.base_manager.safe_execute_read_query(
                query,
                {"name": name}
            )
            
            if not records or len(records) == 0:
                error_response = ErrorResponse(
                    error=ErrorDetail(
                        code="NOT_FOUND",
                        message=f"Project container '{name}' not found",
                        details={}
                    ),
                    timestamp=datetime.now()
                )
                return dict_to_json(error_response.model_dump())
            
            # Extract container info
            record = records[0]
            container = dict(record["c"].items())
            component_count = record["component_count"]
            
            # Get domain counts
            domain_query = """
            MATCH (c:Entity {name: $name, entityType: 'ProjectContainer'})
            MATCH (c)<-[:PART_OF]-(component:Entity)
            RETURN component.entityType as type, count(component) as count
            """
            
            # Use safe_execute_read_query for validation (this is a read-only operation)
            domain_records = self.base_manager.safe_execute_read_query(
                domain_query,
                {"name": name}
            )
            
            # Process domain counts
            domain_counts = {}
            if domain_records:
                for record in domain_records:
                    entity_type = record["type"]
                    count = record["count"]
                    if entity_type:
                        domain_counts[entity_type] = count
            
            # Add component statistics
            container["component_count"] = component_count
            container["domain_counts"] = domain_counts
            
            # Create response using Pydantic model
            response = ProjectContainerResponse(
                message=f"Project container '{name}' retrieved successfully",
                project_id=container.get("id"),
                project=container,
                timestamp=datetime.now()
            )
            
            return dict_to_json(response.model_dump(exclude_none=True))
                
        except Exception as e:
            error_msg = f"Error retrieving project container: {str(e)}"
            self.logger.error(error_msg)
            error_response = ErrorResponse(
                error=ErrorDetail(
                    code="ERROR",
                    message=error_msg,
                    details={}
                ),
                timestamp=datetime.now()
            )
            return dict_to_json(error_response.model_dump())
    
    def update_project_container(self, name: str, updates: ProjectContainerUpdate) -> str:
        """
        Update a project container's properties.
        
        Args:
            name: Name of the project container
            updates: ProjectContainerUpdate model with properties to update
            
        Returns:
            JSON string with the updated container
        """
        try:
            self.base_manager.ensure_initialized()
            
            # Check if container exists
            check_query = """
            MATCH (c:Entity {name: $name, entityType: 'ProjectContainer'})
            RETURN c
            """
            
            # Use safe_execute_read_query for validation (this is a read-only operation)
            records = self.base_manager.safe_execute_read_query(
                check_query,
                {"name": name}
            )
            
            if not records or len(records) == 0:
                error_response = ErrorResponse(
                    error=ErrorDetail(
                        code="NOT_FOUND",
                        message=f"Project container '{name}' not found",
                        details={}
                    ),
                    timestamp=datetime.now()
                )
                return dict_to_json(error_response.model_dump())
            
            # Get the existing container ID
            container_id = dict(records[0]["c"].items()).get("id")
            
            # Extract validated updates
            update_data = updates.model_dump(exclude={"id"}, exclude_none=True)
            
            # Validate updates - prevent changing core properties
            protected_fields = ["id", "name", "entityType", "domain", "created"]
            invalid_updates = [field for field in update_data if field in protected_fields]
            
            if invalid_updates:
                error_response = ErrorResponse(
                    error=ErrorDetail(
                        code="INVALID_UPDATE",
                        message=f"Cannot update protected fields: {', '.join(invalid_updates)}",
                        details={"invalid_fields": invalid_updates}
                    ),
                    timestamp=datetime.now()
                )
                return dict_to_json(error_response.model_dump())
            
            # Handle special fields
            if "tags" in update_data and isinstance(update_data["tags"], list):
                update_data["tags"] = json.dumps(update_data["tags"])
                
            if "metadata" in update_data and update_data["metadata"]:
                # Flatten metadata into top-level properties
                metadata = update_data.pop("metadata")
                if isinstance(metadata, dict):
                    for key, value in metadata.items():
                        update_data[key] = value
            
            # Prepare update parts
            set_parts = []
            for key, value in update_data.items():
                set_parts.append(f"c.{key} = ${key}")
            
            # Build update query
            update_query = f"""
            MATCH (c:Entity {{name: $name, entityType: 'ProjectContainer'}})
            SET {', '.join(set_parts)}, c.lastUpdated = datetime()
            RETURN c
            """
            
            # Add name to updates for the query
            params = {"name": name, **update_data}
            
            # Use safe_execute_write_query for validation (this is a write operation)
            update_records = self.base_manager.safe_execute_write_query(
                update_query,
                params
            )
            
            if not update_records or len(update_records) == 0:
                error_response = ErrorResponse(
                    error=ErrorDetail(
                        code="UPDATE_FAILED",
                        message="Failed to update project container",
                        details={}
                    ),
                    timestamp=datetime.now()
                )
                return dict_to_json(error_response.model_dump())
            
            # Return updated container
            container = dict(update_records[0]["c"].items())
            
            # Create response using Pydantic model
            response = ProjectContainerResponse(
                message=f"Project container '{name}' updated successfully",
                project_id=container.get("id"),
                project=container,
                timestamp=datetime.now()
            )
            
            return dict_to_json(response.model_dump(exclude_none=True))
                
        except Exception as e:
            error_msg = f"Error updating project container: {str(e)}"
            self.logger.error(error_msg)
            error_response = ErrorResponse(
                error=ErrorDetail(
                    code="ERROR",
                    message=error_msg,
                    details={}
                ),
                timestamp=datetime.now()
            )
            return dict_to_json(error_response.model_dump())
    
    def delete_project_container(self, name: str, delete_contents: bool = False) -> str:
        """
        Delete a project container and optionally its contents.
        
        Args:
            name: Name of the project container
            delete_contents: If True, delete all components in the container
            
        Returns:
            JSON string with the deletion result
        """
        try:
            self.base_manager.ensure_initialized()
            
            # Check if container exists
            check_query = """
            MATCH (c:Entity {name: $name, entityType: 'ProjectContainer'})
            RETURN c
            """
            
            records = self.base_manager.safe_execute_read_query(
                check_query,
                {"name": name}
            )
            
            if not records or len(records) == 0:
                error_response = ErrorResponse(
                    error=ErrorDetail(
                        code="NOT_FOUND",
                        message=f"Project container '{name}' not found",
                        details={}
                    ),
                    timestamp=datetime.now()
                )
                return dict_to_json(error_response.model_dump())
            
            if delete_contents:
                # Delete all components in the container along with their relationships and observations
                delete_components_query = """
                MATCH (c:Entity {name: $name, entityType: 'ProjectContainer'})
                MATCH (c)<-[:PART_OF]-(component:Entity)
                
                // Delete component relationships
                OPTIONAL MATCH (component)-[r]-()
                DELETE r
                
                // Delete component observations
                OPTIONAL MATCH (component)-[:HAS_OBSERVATION]->(o:Observation)
                DELETE o
                
                // Delete components
                DELETE component
                
                RETURN count(component) as deleted_count
                """
                
                delete_records = self.base_manager.safe_execute_write_query(
                    delete_components_query,
                    {"name": name}
                )
                
                deleted_count = 0
                if delete_records and len(delete_records) > 0:
                    deleted_count = delete_records[0]["deleted_count"]
                
                # Now delete the container itself
                delete_container_query = """
                MATCH (c:Entity {name: $name, entityType: 'ProjectContainer'})
                DELETE c
                """
                
                self.base_manager.safe_execute_write_query(
                    delete_container_query,
                    {"name": name}
                )
                
                response = SuccessResponse(
                    message=f"Project container '{name}' and {deleted_count} components deleted successfully",
                    timestamp=datetime.now()
                )
                return dict_to_json(response.model_dump())
            else:
                # Check if container has components
                check_components_query = """
                MATCH (c:Entity {name: $name, entityType: 'ProjectContainer'})
                MATCH (c)<-[:PART_OF]-(component:Entity)
                RETURN count(component) as component_count
                """
                
                component_records = self.base_manager.safe_execute_read_query(
                    check_components_query,
                    {"name": name}
                )
                
                component_count = 0
                if component_records and len(component_records) > 0:
                    component_count = component_records[0]["component_count"]
                
                if component_count > 0:
                    error_response = ErrorResponse(
                        error=ErrorDetail(
                            code="COMPONENTS_EXIST",
                            message=f"Cannot delete project container '{name}' with {component_count} components. Set delete_contents=True to delete the container and its contents.",
                            details={"component_count": component_count}
                        ),
                        timestamp=datetime.now()
                    )
                    return dict_to_json(error_response.model_dump())
                
                # Delete the container (which has no components)
                delete_container_query = """
                MATCH (c:Entity {name: $name, entityType: 'ProjectContainer'})
                DELETE c
                """
                
                self.base_manager.safe_execute_write_query(
                    delete_container_query,
                    {"name": name}
                )
                
                response = SuccessResponse(
                    message=f"Project container '{name}' deleted successfully",
                    timestamp=datetime.now()
                )
                return dict_to_json(response.model_dump())
                
        except Exception as e:
            error_msg = f"Error deleting project container: {str(e)}"
            self.logger.error(error_msg)
            error_response = ErrorResponse(
                error=ErrorDetail(
                    code="ERROR",
                    message=error_msg,
                    details={}
                ),
                timestamp=datetime.now()
            )
            return dict_to_json(error_response.model_dump())
    
    def list_project_containers(self, sort_by: str = "name", limit: int = 100) -> str:
        """
        List all project containers.
        
        Args:
            sort_by: Field to sort by ('name', 'created', or 'lastUpdated')
            limit: Maximum number of containers to return
            
        Returns:
            JSON string with the list of containers
        """
        try:
            self.base_manager.ensure_initialized()
            
            # Validate sort_by
            valid_sort_fields = ["name", "created", "lastUpdated"]
            if sort_by not in valid_sort_fields:
                sort_by = "name"
            
            # Build query
            query = f"""
            MATCH (c:Entity {{entityType: 'ProjectContainer'}})
            OPTIONAL MATCH (c)<-[:PART_OF]-(component:Entity)
            WITH c, count(component) as component_count
            RETURN c, component_count
            ORDER BY c.{sort_by}
            LIMIT $limit
            """
            
            records = self.base_manager.safe_execute_read_query(
                query,
                {"limit": limit}
            )
            
            # Process results
            containers = []
            if records:
                for record in records:
                    container = dict(record["c"].items())
                    container["component_count"] = record["component_count"]
                    containers.append(container)
            
            # Create response using Pydantic model
            response = SuccessResponse(
                message=f"Retrieved {len(containers)} project containers",
                timestamp=datetime.now()
            )
            
            # Add container data to response
            response_dict = response.model_dump()
            response_dict["containers"] = containers
            response_dict["count"] = len(containers)
            
            return dict_to_json(response_dict)
                
        except Exception as e:
            error_msg = f"Error listing project containers: {str(e)}"
            self.logger.error(error_msg)
            error_response = ErrorResponse(
                error=ErrorDetail(
                    code="ERROR",
                    message=error_msg,
                    details={}
                ),
                timestamp=datetime.now()
            )
            return dict_to_json(error_response.model_dump())
    
    def add_component_to_container(self, container_name: str, entity_name: str) -> str:
        """
        Add a component entity to a project container.
        
        Args:
            container_name: Name of the project container
            entity_name: Name of the entity to add
            
        Returns:
            JSON string with the result
        """
        try:
            self.base_manager.ensure_initialized()
            
            # Check if container exists
            container_query = """
            MATCH (c:Entity {name: $container_name, entityType: 'ProjectContainer'})
            RETURN c
            """
            
            container_records = self.base_manager.safe_execute_read_query(
                container_query,
                {"container_name": container_name}
            )
            
            if not container_records or len(container_records) == 0:
                error_response = ErrorResponse(
                    error=ErrorDetail(
                        code="NOT_FOUND",
                        message=f"Project container '{container_name}' not found",
                        details={}
                    ),
                    timestamp=datetime.now()
                )
                return dict_to_json(error_response.model_dump())
            
            # Check if entity exists
            entity_query = """
            MATCH (e:Entity {name: $entity_name})
            RETURN e
            """
            
            entity_records = self.base_manager.safe_execute_read_query(
                entity_query,
                {"entity_name": entity_name}
            )
            
            if not entity_records or len(entity_records) == 0:
                error_response = ErrorResponse(
                    error=ErrorDetail(
                        code="NOT_FOUND",
                        message=f"Entity '{entity_name}' not found",
                        details={}
                    ),
                    timestamp=datetime.now()
                )
                return dict_to_json(error_response.model_dump())
            
            # Check if entity is already in the container
            check_query = """
            MATCH (c:Entity {name: $container_name, entityType: 'ProjectContainer'})
            MATCH (e:Entity {name: $entity_name})
            MATCH (e)-[:PART_OF]->(c)
            RETURN e
            """
            
            check_records = self.base_manager.safe_execute_read_query(
                check_query,
                {"container_name": container_name, "entity_name": entity_name}
            )
            
            if check_records and len(check_records) > 0:
                # Already in container, return success
                response = SuccessResponse(
                    message=f"Entity '{entity_name}' is already in project container '{container_name}'",
                    timestamp=datetime.now()
                )
                return dict_to_json(response.model_dump())
            
            # Add entity to container
            add_query = """
            MATCH (c:Entity {name: $container_name, entityType: 'ProjectContainer'})
            MATCH (e:Entity {name: $entity_name})
            CREATE (e)-[:PART_OF {created: datetime()}]->(c)
            RETURN e
            """
            
            add_records = self.base_manager.safe_execute_write_query(
                add_query,
                {"container_name": container_name, "entity_name": entity_name}
            )
            
            if not add_records or len(add_records) == 0:
                error_response = ErrorResponse(
                    error=ErrorDetail(
                        code="UPDATE_FAILED",
                        message=f"Failed to add entity '{entity_name}' to project container '{container_name}'",
                        details={}
                    ),
                    timestamp=datetime.now()
                )
                return dict_to_json(error_response.model_dump())
            
            response = SuccessResponse(
                message=f"Entity '{entity_name}' added to project container '{container_name}'",
                timestamp=datetime.now()
            )
            return dict_to_json(response.model_dump())
                
        except Exception as e:
            error_msg = f"Error adding component to container: {str(e)}"
            self.logger.error(error_msg)
            error_response = ErrorResponse(
                error=ErrorDetail(
                    code="ERROR",
                    message=error_msg,
                    details={}
                ),
                timestamp=datetime.now()
            )
            return dict_to_json(error_response.model_dump())
    
    def remove_component_from_container(self, container_name: str, entity_name: str) -> str:
        """
        Remove a component entity from a project container.
        
        Args:
            container_name: Name of the project container
            entity_name: Name of the entity to remove
            
        Returns:
            JSON string with the result
        """
        try:
            self.base_manager.ensure_initialized()
            
            # Check if the relationship exists
            check_query = """
            MATCH (c:Entity {name: $container_name, entityType: 'ProjectContainer'})
            MATCH (e:Entity {name: $entity_name})
            MATCH (e)-[r:PART_OF]->(c)
            RETURN r
            """
            
            check_records = self.base_manager.safe_execute_read_query(
                check_query,
                {"container_name": container_name, "entity_name": entity_name}
            )
            
            if not check_records or len(check_records) == 0:
                error_response = ErrorResponse(
                    error=ErrorDetail(
                        code="NOT_FOUND",
                        message=f"Entity '{entity_name}' is not in project container '{container_name}'",
                        details={}
                    ),
                    timestamp=datetime.now()
                )
                return dict_to_json(error_response.model_dump())
            
            # Remove entity from container
            remove_query = """
            MATCH (c:Entity {name: $container_name, entityType: 'ProjectContainer'})
            MATCH (e:Entity {name: $entity_name})
            MATCH (e)-[r:PART_OF]->(c)
            DELETE r
            """
            
            self.base_manager.safe_execute_write_query(
                remove_query,
                {"container_name": container_name, "entity_name": entity_name}
            )
            
            response = SuccessResponse(
                message=f"Entity '{entity_name}' removed from project container '{container_name}'",
                timestamp=datetime.now()
            )
            return dict_to_json(response.model_dump())
                
        except Exception as e:
            error_msg = f"Error removing component from container: {str(e)}"
            self.logger.error(error_msg)
            error_response = ErrorResponse(
                error=ErrorDetail(
                    code="ERROR",
                    message=error_msg,
                    details={}
                ),
                timestamp=datetime.now()
            )
            return dict_to_json(error_response.model_dump())
    
    def get_container_components(self, container_name: str, entity_type: Optional[str] = None) -> str:
        """
        Get all components in a project container.
        
        Args:
            container_name: Name of the project container
            entity_type: Optional entity type to filter by
            
        Returns:
            JSON string with the components
        """
        try:
            self.base_manager.ensure_initialized()
            
            # Check if container exists
            container_query = """
            MATCH (c:Entity {name: $container_name, entityType: 'ProjectContainer'})
            RETURN c
            """
            
            container_records = self.base_manager.safe_execute_read_query(
                container_query,
                {"container_name": container_name}
            )
            
            if not container_records or len(container_records) == 0:
                error_response = ErrorResponse(
                    error=ErrorDetail(
                        code="NOT_FOUND",
                        message=f"Project container '{container_name}' not found",
                        details={}
                    ),
                    timestamp=datetime.now()
                )
                return dict_to_json(error_response.model_dump())
            
            # Build query based on entity_type
            if entity_type:
                query = """
                MATCH (c:Entity {name: $container_name, entityType: 'ProjectContainer'})
                MATCH (e:Entity)-[:PART_OF]->(c)
                WHERE e.entityType = $entity_type
                RETURN e
                ORDER BY e.name
                """
                
                params = {"container_name": container_name, "entity_type": entity_type}
            else:
                query = """
                MATCH (c:Entity {name: $container_name, entityType: 'ProjectContainer'})
                MATCH (e:Entity)-[:PART_OF]->(c)
                RETURN e
                ORDER BY e.name
                """
                
                params = {"container_name": container_name}
            
            records = self.base_manager.safe_execute_read_query(
                query,
                params
            )
            
            # Process results
            components = []
            if records:
                for record in records:
                    component = dict(record["e"].items())
                    components.append(component)
            
            # Create response using Pydantic model
            response = SuccessResponse(
                message=f"Retrieved {len(components)} components from project container '{container_name}'",
                timestamp=datetime.now()
            )
            
            # Add component data to response
            response_dict = response.model_dump()
            response_dict["container"] = container_name
            response_dict["component_count"] = len(components)
            response_dict["entity_type_filter"] = entity_type
            response_dict["components"] = components
            
            return dict_to_json(response_dict)
                
        except Exception as e:
            error_msg = f"Error retrieving container components: {str(e)}"
            self.logger.error(error_msg)
            error_response = ErrorResponse(
                error=ErrorDetail(
                    code="ERROR",
                    message=error_msg,
                    details={}
                ),
                timestamp=datetime.now()
            )
            return dict_to_json(error_response.model_dump())
    
    def change_container_status(self, container_name: str, status: str) -> str:
        """
        Change the status of a project container.
        
        Args:
            container_name: Name of the project container
            status: New status ('active', 'archived', 'completed')
            
        Returns:
            JSON string with the result
        """
        try:
            self.base_manager.ensure_initialized()
            
            # Validate status
            valid_statuses = ["active", "archived", "completed"]
            if status not in valid_statuses:
                error_response = ErrorResponse(
                    error=ErrorDetail(
                        code="INVALID_STATUS",
                        message=f"Invalid status '{status}'. Valid values are: {', '.join(valid_statuses)}",
                        details={"valid_statuses": valid_statuses}
                    ),
                    timestamp=datetime.now()
                )
                return dict_to_json(error_response.model_dump())
            
            # Update container status
            update_query = """
            MATCH (c:Entity {name: $container_name, entityType: 'ProjectContainer'})
            SET c.status = $status,
                c.lastUpdated = datetime()
            RETURN c
            """
            
            update_records = self.base_manager.safe_execute_write_query(
                update_query,
                {"container_name": container_name, "status": status}
            )
            
            if not update_records or len(update_records) == 0:
                error_response = ErrorResponse(
                    error=ErrorDetail(
                        code="NOT_FOUND",
                        message=f"Project container '{container_name}' not found",
                        details={}
                    ),
                    timestamp=datetime.now()
                )
                return dict_to_json(error_response.model_dump())
            
            # Return updated container
            container = dict(update_records[0]["c"].items())
            
            # Create response using Pydantic model
            response = ProjectContainerResponse(
                message=f"Project container '{container_name}' status changed to '{status}'",
                project_id=container.get("id"),
                project=container,
                timestamp=datetime.now()
            )
            
            return dict_to_json(response.model_dump(exclude_none=True))
                
        except Exception as e:
            error_msg = f"Error changing container status: {str(e)}"
            self.logger.error(error_msg)
            error_response = ErrorResponse(
                error=ErrorDetail(
                    code="ERROR",
                    message=error_msg,
                    details={}
                ),
                timestamp=datetime.now()
            )
            return dict_to_json(error_response.model_dump())
    
    def get_container_stats(self, container_name: str) -> str:
        """
        Get detailed statistics for a project container.
        
        Args:
            container_name: Name of the project container
            
        Returns:
            JSON string with container statistics
        """
        try:
            self.base_manager.ensure_initialized()
            
            # Check if container exists
            container_query = """
            MATCH (c:Entity {name: $container_name, entityType: 'ProjectContainer'})
            RETURN c
            """
            
            container_records = self.base_manager.safe_execute_read_query(
                container_query,
                {"container_name": container_name}
            )
            
            if not container_records or len(container_records) == 0:
                error_response = ErrorResponse(
                    error=ErrorDetail(
                        code="NOT_FOUND",
                        message=f"Project container '{container_name}' not found",
                        details={}
                    ),
                    timestamp=datetime.now()
                )
                return dict_to_json(error_response.model_dump())
            
            # Get container info
            container = dict(container_records[0]["c"].items())
            
            # Get entity type counts
            type_query = """
            MATCH (c:Entity {name: $container_name, entityType: 'ProjectContainer'})
            MATCH (e:Entity)-[:PART_OF]->(c)
            RETURN e.entityType as type, count(e) as count
            """
            
            type_records = self.base_manager.safe_execute_read_query(
                type_query,
                {"container_name": container_name}
            )
            
            # Process entity type stats
            entity_types = {}
            if type_records:
                for record in type_records:
                    entity_type = record["type"]
                    count = record["count"]
                    if entity_type:
                        entity_types[entity_type] = count
            
            # Get relationship stats
            rel_query = """
            MATCH (c:Entity {name: $container_name, entityType: 'ProjectContainer'})
            MATCH (e1:Entity)-[:PART_OF]->(c)
            MATCH (e1)-[r]->(e2:Entity)
            WHERE (e2)-[:PART_OF]->(c)
            RETURN type(r) as type, count(r) as count
            """
            
            rel_records = self.base_manager.safe_execute_read_query(
                rel_query,
                {"container_name": container_name}
            )
            
            # Process relationship stats
            relationship_types = {}
            if rel_records:
                for record in rel_records:
                    rel_type = record["type"]
                    count = record["count"]
                    if rel_type:
                        relationship_types[rel_type] = count
            
            # Get observation stats
            obs_query = """
            MATCH (c:Entity {name: $container_name, entityType: 'ProjectContainer'})
            MATCH (e:Entity)-[:PART_OF]->(c)
            MATCH (e)-[:HAS_OBSERVATION]->(o:Observation)
            RETURN o.type as type, count(o) as count
            """
            
            obs_records = self.base_manager.safe_execute_read_query(
                obs_query,
                {"container_name": container_name}
            )
            
            # Process observation stats
            observation_types = {}
            if obs_records:
                for record in obs_records:
                    obs_type = record["type"]
                    count = record["count"]
                    if obs_type:
                        observation_types[obs_type] = count
            
            # Calculate total counts
            entity_count = sum(entity_types.values()) if entity_types else 0
            relationship_count = sum(relationship_types.values()) if relationship_types else 0
            observation_count = sum(observation_types.values()) if observation_types else 0
            
            # Create response using Pydantic model
            response = SuccessResponse(
                message=f"Retrieved statistics for project container '{container_name}'",
                timestamp=datetime.now()
            )
            
            # Add stats to response
            response_dict = response.model_dump()
            response_dict["container"] = container
            response_dict["total_entities"] = entity_count
            response_dict["entity_types"] = entity_types
            response_dict["total_relationships"] = relationship_count
            response_dict["relationship_types"] = relationship_types
            response_dict["total_observations"] = observation_count
            response_dict["observation_types"] = observation_types
            
            return dict_to_json(response_dict)
                
        except Exception as e:
            error_msg = f"Error retrieving container stats: {str(e)}"
            self.logger.error(error_msg)
            error_response = ErrorResponse(
                error=ErrorDetail(
                    code="ERROR",
                    message=error_msg,
                    details={}
                ),
                timestamp=datetime.now()
            )
            return dict_to_json(error_response.model_dump()) 