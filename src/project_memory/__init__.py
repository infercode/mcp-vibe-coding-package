"""
Project Memory System for MCP Graph Memory.
Provides specialized components for managing project-related knowledge.
"""

from typing import Any, Dict, List, Optional, Union
import time
import json
import logging
import re
from datetime import datetime

from src.graph_memory.base_manager import BaseManager
from src.project_memory.domain_manager import DomainManager
from src.project_memory.component_manager import ComponentManager
from src.project_memory.dependency_manager import DependencyManager
from src.project_memory.version_manager import VersionManager
from src.project_memory.project_container import ProjectContainer
from src.models.responses import SuccessResponse, create_error_response, create_success_response
from src.models.project_memory import (
    ProjectContainerCreate, ProjectContainerUpdate, ComponentCreate, 
    ComponentUpdate, DomainEntityCreate, RelationshipCreate, SearchQuery, VersionCreate, VersionGetRequest,
    VersionCompareRequest, TagCreate, SyncRequest, CommitData, ErrorDetail, ErrorResponse
)
from src.graph_memory.search_manager import SearchManager

class ProjectMemoryManager:
    """
    Unified facade for the Project Memory System.
    Provides access to all project memory components through a single interface.
    """
    
    def __init__(self, base_manager: Optional[BaseManager] = None):
        """
        Initialize the Project Memory Manager.
        
        Args:
            base_manager: Optional base manager for graph operations.
                          If not provided, a new one will be created.
        """
        # Initialize or use provided base manager
        self.base_manager = base_manager or BaseManager()
        
        # Initialize component managers
        self.domain_manager = DomainManager(self.base_manager)
        self.component_manager = ComponentManager(self.base_manager)
        self.dependency_manager = DependencyManager(self.base_manager)
        self.version_manager = VersionManager(self.base_manager)
        self.project_container = ProjectContainer(self.base_manager)
        self.search_manager = SearchManager(self.base_manager)
        
        # Logger for the project memory manager
        self.logger = self.base_manager.logger or logging.getLogger(__name__)
    
    def _standardize_response(self, result_json: str, success_message: str, error_code: str) -> str:
        """
        Standardize API responses.
        
        Args:
            result_json: JSON result string from a manager
            success_message: Message to include in successful responses
            error_code: Error code to use for error responses
            
        Returns:
            Standardized JSON response string
        """
        try:
            # Parse the result JSON
            result_data = json.loads(result_json)
            
            # If the result already has a Pydantic-style error structure, return it directly
            if isinstance(result_data, dict) and result_data.get("status") == "error" and "error" in result_data:
                return result_json
                
            # If result contains an "error" key, convert to proper error response
            if isinstance(result_data, dict) and "error" in result_data:
                from src.models.project_memory import ErrorDetail, ErrorResponse
                error = ErrorDetail(
                    code=error_code,
                    message=result_data["error"],
                    details=None
                )
                error_response = ErrorResponse(
                    status="error",
                    timestamp=datetime.now(),
                    error=error
                )
                return error_response.model_dump_json()
                
            # Otherwise, it's a success response - create proper Pydantic response
            from src.models.project_memory import SuccessResponse
            
            # Create a standard response with just the message and timestamp
            success_response = SuccessResponse(
                message=success_message,
                timestamp=datetime.now()
            )
            
            # Convert the response to a dict and merge with result data
            response_dict = success_response.model_dump(exclude_none=True)
            if isinstance(result_data, dict):
                response_dict.update(result_data)
            
            # Return as JSON
            return json.dumps(response_dict, default=str)
            
        except Exception as e:
            # Something went wrong while processing the response
            self.logger.error(f"Error standardizing response: {str(e)}")
            from src.models.project_memory import ErrorDetail, ErrorResponse
            error = ErrorDetail(
                code="response_processing_error",
                message=f"Error processing response: {str(e)}",
                details={"original_response": result_json}
            )
            error_response = ErrorResponse(
                status="error",
                timestamp=datetime.now(),
                error=error
            )
            return error_response.model_dump_json()
    
    # ============================================================================
    # Domain management methods
    # ============================================================================
    
    def create_project_domain(self, name: str, container_name: str, 
                    description: Optional[str] = None,
                    properties: Optional[Dict[str, Any]] = None) -> str:
        """
        Create a new project domain.
        
        Args:
            name: Name of the domain
            container_name: Name of the project container
            description: Optional. Description of the domain
            properties: Optional. Additional properties for the domain
            
        Returns:
            JSON string with the created domain
        """
        try:
            # Use Pydantic v2 validation with modern error handling
            from pydantic import ValidationError
            
            try:
                # Prepare domain data for the Pydantic model
                domain_data = {
                    "name": name,
                    "project_id": container_name,
                    "description": description,
                    "metadata": properties,
                    "type": "Domain"  # Required by DomainEntityCreate
                }
                
                # Create and validate with Pydantic
                domain_model = DomainEntityCreate(**domain_data)
                
                # We can access computed fields or perform model validation
                if hasattr(domain_model, 'entity_identifier'):
                    domain_id = domain_model.entity_identifier
                    self.logger.debug(f"Creating domain with generated ID: {domain_id}")
                
            except ValidationError as ve:
                # Enhanced validation error handling
                self.logger.error(f"Validation error for domain: {ve}")
                error = ErrorDetail(
                    code="validation_error",
                    message="Invalid domain data",
                    details={"validation_errors": ve.errors()}
                )
                error_response = ErrorResponse(
                    status="error",
                    timestamp=datetime.now(),
                    error=error
                )
                return error_response.model_dump_json()
            
            # Call the domain manager with the validated model
            result = self.domain_manager.create_domain(domain_model)
            
            # Process and standardize the response
            result_data = json.loads(result)
            
            # Handle errors in the domain manager response
            if "error" in result_data:
                # We can directly return since it's already in error format
                return result
            
            # We can transform to a proper response type if needed
            # For now, standardize using our helper method
            return self._standardize_response(
                result,
                f"Successfully created domain '{name}' in project '{container_name}'",
                "domain_creation_error"
            )
            
        except Exception as e:
            # Centralized error handling with Pydantic models
            self.logger.error(f"Error creating project domain: {str(e)}")
            error = ErrorDetail(
                code="domain_creation_error",
                message=f"Failed to create project domain: {str(e)}",
                details=None
            )
            error_response = ErrorResponse(
                status="error",
                timestamp=datetime.now(),
                error=error
            )
            return error_response.model_dump_json()
    
    def get_project_domain(self, name: str, container_name: str) -> str:
        """
        Retrieve a domain from a project container.
        
        Args:
            name: Name of the domain
            container_name: Name of the project container
            
        Returns:
            JSON string with the domain details
        """
        try:
            result = self.domain_manager.get_domain(name, container_name)
            return self._standardize_response(
                result, 
                f"Retrieved domain '{name}' from project '{container_name}'",
                "domain_retrieval_error"
            )
        except Exception as e:
            self.logger.error(f"Error retrieving project domain: {str(e)}")
            return json.dumps(create_error_response(
                message=f"Failed to retrieve domain: {str(e)}",
                code="domain_retrieval_error"
            ).model_dump(), default=str)
    
    def update_project_domain(self, name: str, container_name: str, 
                    updates: Dict[str, Any]) -> str:
        """
        Update a domain's properties.
        
        Args:
            name: Name of the domain
            container_name: Name of the project container
            updates: Dictionary of properties to update
            
        Returns:
            JSON string with the updated domain
        """
        try:
            result = self.domain_manager.update_domain(name, container_name, updates)
            return self._standardize_response(
                result,
                f"Updated domain '{name}' in project '{container_name}'",
                "domain_update_error"
            )
        except Exception as e:
            self.logger.error(f"Error updating project domain: {str(e)}")
            return json.dumps(create_error_response(
                message=f"Failed to update domain: {str(e)}",
                code="domain_update_error"
            ).model_dump(), default=str)
    
    def delete_project_domain(self, name: str, container_name: str, 
                    delete_components: bool = False) -> str:
        """
        Delete a domain from a project container.
        
        Args:
            name: Name of the domain
            container_name: Name of the project container
            delete_components: If True, delete all components belonging to the domain
            
        Returns:
            JSON string with the deletion result
        """
        try:
            result = self.domain_manager.delete_domain(
                name, container_name, delete_components
            )
            return self._standardize_response(
                result,
                f"Deleted domain '{name}' from project '{container_name}'",
                "domain_deletion_error"
            )
        except Exception as e:
            self.logger.error(f"Error deleting project domain: {str(e)}")
            return json.dumps(create_error_response(
                message=f"Failed to delete domain: {str(e)}",
                code="domain_deletion_error"
            ).model_dump(), default=str)
    
    def list_project_domains(self, container_name: str, sort_by: str = "name", 
                   limit: int = 100) -> str:
        """
        List all domains in a project container.
        
        Args:
            container_name: Name of the project container
            sort_by: Field to sort by ('name', 'created', or 'lastUpdated')
            limit: Maximum number of domains to return
            
        Returns:
            JSON string with the list of domains
        """
        try:
            result = self.domain_manager.list_domains(container_name, sort_by, limit)
            return self._standardize_response(
                result,
                f"Listed domains in project '{container_name}'",
                "domain_list_error"
            )
        except Exception as e:
            self.logger.error(f"Error listing project domains: {str(e)}")
            return json.dumps(create_error_response(
                message=f"Failed to list domains: {str(e)}",
                code="domain_list_error"
            ).model_dump(), default=str)
    
    def add_entity_to_project_domain(self, domain_name: str, container_name: str, 
                          entity_name: str) -> str:
        """
        Add an entity to a domain.
        
        Args:
            domain_name: Name of the domain
            container_name: Name of the project container
            entity_name: Name of the entity to add
            
        Returns:
            JSON string with the result
        """
        try:
            result = self.domain_manager.add_entity_to_domain(
                domain_name, container_name, entity_name
            )
            return self._standardize_response(
                result,
                f"Added entity '{entity_name}' to domain '{domain_name}' in project '{container_name}'",
                "entity_add_error"
            )
        except Exception as e:
            self.logger.error(f"Error adding entity to project domain: {str(e)}")
            return json.dumps(create_error_response(
                message=f"Failed to add entity to domain: {str(e)}",
                code="entity_add_error"
            ).model_dump(), default=str)
    
    def remove_entity_from_project_domain(self, domain_name: str, container_name: str, 
                               entity_name: str) -> str:
        """
        Remove an entity from a domain.
        
        Args:
            domain_name: Name of the domain
            container_name: Name of the project container
            entity_name: Name of the entity to remove
            
        Returns:
            JSON string with the result
        """
        try:
            result = self.domain_manager.remove_entity_from_domain(
                domain_name, container_name, entity_name
            )
            return self._standardize_response(
                result,
                f"Removed entity '{entity_name}' from domain '{domain_name}' in project '{container_name}'",
                "entity_remove_error"
            )
        except Exception as e:
            self.logger.error(f"Error removing entity from project domain: {str(e)}")
            return json.dumps(create_error_response(
                message=f"Failed to remove entity from domain: {str(e)}",
                code="entity_remove_error"
            ).model_dump(), default=str)
    
    def get_project_domain_entities(self, domain_name: str, container_name: str, 
                         entity_type: Optional[str] = None) -> str:
        """
        Get all entities in a domain.
        
        Args:
            domain_name: Name of the domain
            container_name: Name of the project container
            entity_type: Optional entity type to filter by
            
        Returns:
            JSON string with the entities
        """
        try:
            result = self.domain_manager.get_domain_entities(
                domain_name, container_name, entity_type
            )
            return self._standardize_response(
                result,
                f"Retrieved entities from domain '{domain_name}' in project '{container_name}'",
                "entity_retrieval_error"
            )
        except Exception as e:
            self.logger.error(f"Error retrieving project domain entities: {str(e)}")
            return json.dumps(create_error_response(
                message=f"Failed to retrieve domain entities: {str(e)}",
                code="entity_retrieval_error"
            ).model_dump(), default=str)
    
    def create_project_domain_relationship(self, from_domain: str, to_domain: str, 
                                container_name: str, relation_type: str,
                                properties: Optional[Dict[str, Any]] = None) -> str:
        """
        Create a relationship between two project domains.
        
        Args:
            from_domain: Name of the source domain
            to_domain: Name of the target domain
            container_name: Name of the project container
            relation_type: Type of relationship
            properties: Optional. Additional properties for the relationship
            
        Returns:
            JSON string with the created relationship
        """
        try:
            # Validate relationship data using Pydantic model
            try:
                relationship_data = {
                    "source_id": from_domain,
                    "target_id": to_domain,
                    "relationship_type": relation_type,
                    "properties": properties
                }
                relationship_model = RelationshipCreate(**relationship_data)
                validated_data = relationship_model.model_dump()
            except Exception as ve:
                self.logger.error(f"Validation error for domain relationship: {str(ve)}")
                return json.dumps(create_error_response(
                    message=f"Invalid relationship data: {str(ve)}",
                    code="validation_error"
                ).model_dump(), default=str)
            
            # Now proceed with the domain manager call
            result = self.domain_manager.create_domain_relationship(
                relationship=relationship_model,
                container_name=container_name
            )
            
            return self._standardize_response(
                result,
                f"Created {validated_data['relationship_type']} relationship from domain '{from_domain}' to '{to_domain}' in project '{container_name}'",
                "domain_relationship_error"
            )
        except Exception as e:
            self.logger.error(f"Error creating project domain relationship: {str(e)}")
            return json.dumps(create_error_response(
                message=f"Failed to create domain relationship: {str(e)}",
                code="domain_relationship_error"
            ).model_dump(), default=str)
    
    # ============================================================================
    # Component management methods
    # ============================================================================
    
    def create_project_component(self, name: str, component_type: str,
                      domain_name: str, container_name: str,
                      description: Optional[str] = None,
                      content: Optional[str] = None,
                      metadata: Optional[Dict[str, Any]] = None) -> str:
        """
        Create a new project component.
        
        Args:
            name: Name of the component
            component_type: Type of component (e.g., 'SERVICE', 'MODULE')
            domain_name: Name of the domain
            container_name: Name of the project container
            description: Optional. Description of the component
            content: Optional. Content or code associated with the component
            metadata: Optional. Additional metadata for the component
            
        Returns:
            JSON string with the created component
        """
        try:
            # Use Pydantic v2 validation and conversion
            from pydantic import ValidationError
            from src.models.project_memory import ComponentResponse
            
            try:
                # Convert parameters to a dictionary for component construction
                component_data = {
                    "name": name,
                    "type": component_type,  # component_manager expects 'type'
                    "description": description,
                    "project_id": container_name,  # required by ComponentCreate model
                    "metadata": metadata
                }
                
                # Create the Pydantic model with validation
                component_model = ComponentCreate(**component_data)
                
                # Use the model's computed fields if needed
                component_id = component_model.component_identifier
                self.logger.debug(f"Creating component with ID: {component_id}")
                
            except ValidationError as ve:
                # Handle validation errors with proper Pydantic error format
                self.logger.error(f"Validation error for component: {ve}")
                error = ErrorDetail(
                    code="validation_error",
                    message="Invalid component data",
                    details={"validation_errors": ve.errors()}
                )
                error_response = ErrorResponse(
                    status="error",
                    timestamp=datetime.now(),
                    error=error
                )
                return error_response.model_dump_json()
            
            # Call the component manager with validated Pydantic model
            result = self.component_manager.create_component(
                component=component_model,
                domain_name=domain_name,
                container_name=container_name
            )
            
            # Process the response with Pydantic
            try:
                # Try to parse directly as a Pydantic ComponentResponse
                response_obj = ComponentResponse.model_validate_json(result)
                return response_obj.model_dump_json()
            except Exception:
                # Fallback to standard response
                return self._standardize_response(
                    result,
                    f"Created {component_type} component '{name}' in domain '{domain_name}' of project '{container_name}'",
                    "component_creation_error"
                )
                
        except Exception as e:
            self.logger.error(f"Error creating project component: {str(e)}")
            error = ErrorDetail(
                code="component_creation_error",
                message=f"Failed to create component: {str(e)}",
                details=None
            )
            error_response = ErrorResponse(
                status="error",
                timestamp=datetime.now(),
                error=error
            )
            return error_response.model_dump_json()
    
    def get_project_component(self, name: str, domain_name: str, 
                    container_name: str) -> str:
        """
        Retrieve a component from a domain.
        
        Args:
            name: Name of the component
            domain_name: Name of the domain
            container_name: Name of the project container
            
        Returns:
            JSON string with the component details
        """
        try:
            result = self.component_manager.get_component(
                name, domain_name, container_name
            )
            return self._standardize_response(
                result,
                f"Retrieved component '{name}' from domain '{domain_name}' in project '{container_name}'",
                "component_retrieval_error"
            )
        except Exception as e:
            self.logger.error(f"Error retrieving project component: {str(e)}")
            return json.dumps(create_error_response(
                message=f"Failed to retrieve component: {str(e)}",
                code="component_retrieval_error"
            ).model_dump(), default=str)
    
    def update_project_component(self, name: str, container_name: str, updates: Dict[str, Any], domain_name: Optional[str] = None) -> str:
        """
        Update a project component.
        
        Args:
            name: Name of the component to update
            container_name: Name of the project container
            updates: Dictionary with updates to apply
            domain_name: Optional. Name of the domain
            
        Returns:
            JSON string with the updated component
        """
        try:
            # Use Pydantic v2 validation and conversion
            from pydantic import ValidationError
            from src.models.project_memory import ComponentResponse
            
            try:
                # Add the required ID field to the updates
                updates_with_id = {"id": name, **updates}
                
                # Create and validate the ComponentUpdate model
                component_update = ComponentUpdate(**updates_with_id)
                
                # We can access computed fields if they exist
                if hasattr(component_update, 'update_count'):
                    self.logger.debug(f"Updating {component_update.update_count} fields for component '{name}'")
                
            except ValidationError as ve:
                # Enhanced validation error handling
                self.logger.error(f"Validation error for component update: {ve}")
                error = ErrorDetail(
                    code="validation_error",
                    message="Invalid update data",
                    details={"validation_errors": ve.errors()}
                )
                error_response = ErrorResponse(
                    status="error",
                    timestamp=datetime.now(),
                    error=error
                )
                return error_response.model_dump_json()
            
            # Pass an empty string when domain_name is None for API compatibility
            domain_name_to_use = domain_name if domain_name else ""
            
            # Use the proper structure for the update_component call
            update_result = self.component_manager.update_component(
                component_update=component_update,
                domain_name=domain_name_to_use, 
                container_name=container_name
            )
            
            # Process the response with Pydantic
            try:
                # Try to parse as a Pydantic response model
                response_obj = ComponentResponse.model_validate_json(update_result)
                return response_obj.model_dump_json()
            except Exception:
                # Domain description for message
                domain_desc = f" in domain '{domain_name}'" if domain_name else ""
                
                # Fallback to standard response handling
                return self._standardize_response(
                    update_result,
                    f"Updated component '{name}'{domain_desc} in project '{container_name}'",
                    "component_update_error"
                )
                
        except Exception as e:
            self.logger.error(f"Error updating project component: {str(e)}")
            error = ErrorDetail(
                code="component_update_error",
                message=f"Failed to update component: {str(e)}",
                details=None
            )
            error_response = ErrorResponse(
                status="error",
                timestamp=datetime.now(),
                error=error
            )
            return error_response.model_dump_json()
    
    def delete_project_component(self, component_id: str, domain_name: Optional[str] = None, 
                      container_name: Optional[str] = None) -> str:
        """
        Delete a component.
        
        Args:
            component_id: ID or name of the component to delete
            domain_name: Optional name of the domain
            container_name: Optional name of the project container
            
        Returns:
            JSON string with the deletion result
        """
        try:
            # Create validation data
            delete_data = {
                "component_id": component_id
            }
            
            if domain_name:
                delete_data["domain_name"] = domain_name
            
            if container_name:
                delete_data["container_name"] = container_name
                
            # Validate the input
            # While there's no specific ComponentDelete model, we can validate the ID format
            # and check if the component exists
            
            # For test_delete_component, use the component_manager directly
            if domain_name is not None and container_name is not None:
                # Validate component exists before deletion
                comp_query = """
                MATCH (comp:Entity {name: $component_id})
                MATCH (d:Entity {name: $domain_name, entityType: 'Domain'})
                MATCH (c:Entity {name: $container_name, entityType: 'ProjectContainer'})
                WHERE (comp)-[:BELONGS_TO]->(d) AND (d)-[:PART_OF]->(c)
                RETURN comp
                """
                
                records = self.base_manager.safe_execute_read_query(
                    comp_query,
                    {"component_id": component_id, "domain_name": domain_name, "container_name": container_name}
                )
                
                if not records or len(records) == 0:
                    return json.dumps(create_error_response(
                        message=f"Component '{component_id}' not found in domain '{domain_name}' of project '{container_name}'",
                        code="component_not_found"
                    ).model_dump(), default=str)
                
                # Use the component_manager's delete_component method
                result = self.component_manager.delete_component(component_id, domain_name, container_name)
                return self._standardize_response(
                    result,
                    f"Deleted component '{component_id}' from domain '{domain_name}' in project '{container_name}'",
                    "component_deletion_error"
                )
                
            # First check if the component exists
            query = """
            MATCH (comp:Entity {id: $component_id})
            RETURN comp
            """
            
            # If component_id looks like a name rather than an ID, search by name
            if not component_id.startswith("cmp-"):
                query = """
                MATCH (comp:Entity {name: $component_id})
                RETURN comp
                """
            
            records = self.base_manager.safe_execute_read_query(
                query,
                {"component_id": component_id}
            )
            
            if not records or len(records) == 0:
                error_msg = f"Component '{component_id}' not found{' in domain ' + domain_name if domain_name else ''}"
                return json.dumps(create_error_response(
                    message=error_msg,
                    code="component_not_found"
                ).model_dump(), default=str)
            
            # Get the component data - Neo4j records need to be accessed properly
            component_data = {}
            if isinstance(records[0], dict) and "comp" in records[0]:
                # If records[0] is a dict with comp key
                comp_node = records[0]["comp"]
                if hasattr(comp_node, "items"):
                    component_data = dict(comp_node.items())
                elif isinstance(comp_node, dict):
                    component_data = comp_node
            
            component_id_actual = component_data.get("id", component_id)
            component_name = component_data.get("name", component_id)
            
            # Check for relationships
            relation_query = """
            MATCH (comp:Entity {id: $component_id})-[r]-(other:Entity)
            WHERE type(r) <> 'BELONGS_TO' AND type(r) <> 'PART_OF' 
                AND type(r) <> 'HAS_OBSERVATION'
            RETURN count(r) as rel_count
            """
            
            rel_records = self.base_manager.safe_execute_read_query(
                relation_query,
                {"component_id": component_id_actual}
            )
            
            rel_count = 0
            if rel_records and len(rel_records) > 0:
                # Access Neo4j record properly
                if isinstance(rel_records[0], dict) and "rel_count" in rel_records[0]:
                    rel_count = rel_records[0]["rel_count"]
                else:
                    # Just try to extract the value using various methods
                    try:
                        # Different ways Neo4j records might be structured
                        if hasattr(rel_records[0], "get"):
                            rel_count = rel_records[0].get("rel_count", 0)
                        elif isinstance(rel_records[0], dict):
                            rel_count = rel_records[0].get("rel_count", 0)
                        # Just in case we have a list of results
                        elif len(rel_records) > 0 and isinstance(rel_records[0], (list, tuple)) and len(rel_records[0]) > 0:
                            rel_count = rel_records[0][0]  # Assuming first element is the count
                    except (TypeError, IndexError, AttributeError):
                        # If all else fails, default to 0
                        rel_count = 0
            
            if rel_count > 0:
                return json.dumps(create_error_response(
                    message=f"Cannot delete component with {rel_count} relationships. Remove the relationships first.",
                    code="component_has_relationships"
                ).model_dump(), default=str)
            
            # Delete the component's observations
            delete_obs_query = """
            MATCH (comp:Entity {id: $component_id})-[:HAS_OBSERVATION]->(o:Observation)
            DELETE o
            """
            
            self.base_manager.safe_execute_write_query(
                delete_obs_query,
                {"component_id": component_id_actual}
            )
            
            # Delete the component's relationships
            delete_rel_query = """
            MATCH (comp:Entity {id: $component_id})-[r]-()
            DELETE r
            """
            
            self.base_manager.safe_execute_write_query(
                delete_rel_query,
                {"component_id": component_id_actual}
            )
            
            # Delete the component
            delete_query = """
            MATCH (comp:Entity {id: $component_id})
            DELETE comp
            """
            
            self.base_manager.safe_execute_write_query(
                delete_query,
                {"component_id": component_id_actual}
            )
            
            return json.dumps(create_success_response(
                message=f"Component '{component_name}' deleted successfully",
                data={"status": "success", "component_name": component_name}
            ).model_dump(), default=str)
        
        except Exception as e:
            self.logger.error(f"Error deleting project component: {str(e)}")
            return json.dumps(create_error_response(
                message=f"Failed to delete component: {str(e)}",
                code="component_deletion_error"
            ).model_dump(), default=str)
    
    def list_project_components(self, domain_name: str, container_name: str, 
                     component_type: Optional[str] = None,
                     sort_by: str = "name", limit: int = 100) -> str:
        """
        List all components in a domain.
        
        Args:
            domain_name: Name of the domain
            container_name: Name of the project container
            component_type: Optional component type to filter by
            sort_by: Field to sort by ('name', 'created', or 'lastUpdated')
            limit: Maximum number of components to return
            
        Returns:
            JSON string with the list of components
        """
        try:
            result = self.component_manager.list_components(
                domain_name, container_name, component_type, sort_by, limit
            )
            return self._standardize_response(
                result,
                f"Listed components in domain '{domain_name}' of project '{container_name}'",
                "component_list_error"
            )
        except Exception as e:
            self.logger.error(f"Error listing project components: {str(e)}")
            return json.dumps(create_error_response(
                message=f"Failed to list components: {str(e)}",
                code="component_list_error"
            ).model_dump(), default=str)
    
    def create_project_component_relationship(self, from_component: str, to_component: str, 
                                   domain_name: str, container_name: str,
                                   relation_type: str,
                                   properties: Optional[Dict[str, Any]] = None) -> str:
        """
        Create a relationship between two project components.
        
        Args:
            from_component: Name of the source component
            to_component: Name of the target component
            domain_name: Name of the domain
            container_name: Name of the project container
            relation_type: Type of relationship
            properties: Optional. Additional properties for the relationship
            
        Returns:
            JSON string with the created relationship
        """
        try:
            # Use Pydantic v2 validation with proper error handling
            from pydantic import ValidationError
            
            try:
                # Create the relationship data for the Pydantic model
                relationship_data = {
                    "source_id": from_component,
                    "target_id": to_component,
                    "relationship_type": relation_type,
                    "properties": properties
                }
                
                # Create and validate the RelationshipCreate model
                relationship_model = RelationshipCreate(**relationship_data)
                
                # We can access computed fields if needed
                if hasattr(relationship_model, 'relationship_label'):
                    relationship_label = relationship_model.relationship_label
                    self.logger.debug(f"Creating relationship with label: {relationship_label}")
                
            except ValidationError as ve:
                # Enhanced validation error handling
                self.logger.error(f"Validation error for component relationship: {ve}")
                error = ErrorDetail(
                    code="validation_error",
                    message="Invalid relationship data",
                    details={"validation_errors": ve.errors()}
                )
                error_response = ErrorResponse(
                    status="error",
                    timestamp=datetime.now(),
                    error=error
                )
                return error_response.model_dump_json()
            
            # Call the component manager with the validated model
            result = self.component_manager.create_component_relationship(
                relationship=relationship_model,
                domain_name=domain_name,
                container_name=container_name
            )
            
            # Process the response - use standard response
            # We don't have a RelationshipResponse model currently
            relationship_desc = relationship_model.relationship_type
            return self._standardize_response(
                result,
                f"Created {relationship_desc} relationship from component '{from_component}' to '{to_component}' in domain '{domain_name}'",
                "component_relationship_error"
            )
                
        except Exception as e:
            self.logger.error(f"Error creating project component relationship: {str(e)}")
            error = ErrorDetail(
                code="component_relationship_error",
                message=f"Failed to create component relationship: {str(e)}",
                details=None
            )
            error_response = ErrorResponse(
                status="error",
                timestamp=datetime.now(),
                error=error
            )
            return error_response.model_dump_json()
    
    # ============================================================================
    # Dependency management methods
    # ============================================================================
    
    def create_project_dependency(self, from_component: str, to_component: str,
                       domain_name: str, container_name: str,
                       dependency_type: str,
                       properties: Optional[Dict[str, Any]] = None) -> str:
        """
        Create a dependency relationship between two project components.
        
        Args:
            from_component: Name of the source component
            to_component: Name of the target component
            domain_name: Name of the domain
            container_name: Name of the project container
            dependency_type: Type of dependency
            properties: Optional. Additional properties for the dependency
            
        Returns:
            JSON string with the created dependency
        """
        try:
            # Validate relationship data using Pydantic model
            try:
                relationship_data = {
                    "source_id": from_component,
                    "target_id": to_component,
                    "relationship_type": dependency_type,
                    "properties": properties
                }
                relationship_model = RelationshipCreate(**relationship_data)
                validated_data = relationship_model.model_dump()
            except Exception as ve:
                self.logger.error(f"Validation error for dependency relationship: {str(ve)}")
                return json.dumps(create_error_response(
                    message=f"Invalid dependency data: {str(ve)}",
                    code="validation_error"
                ).model_dump(), default=str)
            
            # Now proceed with the dependency manager call
            result = self.dependency_manager.create_dependency(
                relationship=relationship_model,
                domain_name=domain_name,
                container_name=container_name
            )
            
            return self._standardize_response(
                result,
                f"Created {validated_data['relationship_type']} dependency from '{from_component}' to '{to_component}' in domain '{domain_name}'",
                "dependency_creation_error"
            )
        except Exception as e:
            self.logger.error(f"Error creating project dependency: {str(e)}")
            return json.dumps(create_error_response(
                message=f"Failed to create dependency: {str(e)}",
                code="dependency_creation_error"
            ).model_dump(), default=str)
    
    def get_project_dependencies(self, component_name: str, domain_name: str, 
                      container_name: str, direction: str = "outgoing",
                      dependency_type: Optional[str] = None) -> str:
        """
        Get dependencies for a component.
        
        Args:
            component_name: Name of the component
            domain_name: Name of the domain
            container_name: Name of the project container
            direction: Direction of dependencies ('outgoing', 'incoming', or 'both')
            dependency_type: Optional dependency type to filter by
            
        Returns:
            JSON string with the dependencies
        """
        try:
            result = self.dependency_manager.get_dependencies(
                component_name, domain_name, container_name, direction, dependency_type
            )
            type_filter = f" of type '{dependency_type}'" if dependency_type else ""
            dir_desc = "from" if direction == "incoming" else "to" if direction == "outgoing" else "connected to"
            return self._standardize_response(
                result,
                f"Retrieved {direction} dependencies{type_filter} {dir_desc} component '{component_name}'",
                "dependency_retrieval_error"
            )
        except Exception as e:
            self.logger.error(f"Error retrieving project dependencies: {str(e)}")
            return json.dumps(create_error_response(
                message=f"Failed to retrieve dependencies: {str(e)}",
                code="dependency_retrieval_error"
            ).model_dump(), default=str)
    
    def delete_project_dependency(self, from_component: str, to_component: str,
                       domain_name: str, container_name: str,
                       dependency_type: str) -> str:
        """
        Delete a dependency relationship between components.
        
        Args:
            from_component: Name of the source component
            to_component: Name of the target component
            domain_name: Name of the domain
            container_name: Name of the project container
            dependency_type: Type of dependency to delete
            
        Returns:
            JSON string with the result
        """
        try:
            result = self.dependency_manager.delete_dependency(
                from_component, to_component, domain_name, container_name, dependency_type
            )
            return self._standardize_response(
                result,
                f"Deleted {dependency_type} dependency from '{from_component}' to '{to_component}' in domain '{domain_name}'",
                "dependency_deletion_error"
            )
        except Exception as e:
            self.logger.error(f"Error deleting project dependency: {str(e)}")
            return json.dumps(create_error_response(
                message=f"Failed to delete dependency: {str(e)}",
                code="dependency_deletion_error"
            ).model_dump(), default=str)
    
    def analyze_project_dependency_graph(self, domain_name: str, 
                              container_name: str) -> str:
        """
        Analyze the dependency graph for a domain.
        
        Args:
            domain_name: Name of the domain
            container_name: Name of the project container
            
        Returns:
            JSON string with the dependency analysis
        """
        try:
            result = self.dependency_manager.analyze_dependency_graph(
                domain_name, container_name
            )
            return self._standardize_response(
                result,
                f"Analyzed dependency graph for domain '{domain_name}' in project '{container_name}'",
                "dependency_analysis_error"
            )
        except Exception as e:
            self.logger.error(f"Error analyzing project dependency graph: {str(e)}")
            return json.dumps(create_error_response(
                message=f"Failed to analyze dependency graph: {str(e)}",
                code="dependency_analysis_error"
            ).model_dump(), default=str)
    
    def find_project_dependency_path(self, from_component: str, to_component: str,
               domain_name: str, container_name: str,
               max_depth: int = 5) -> str:
        """
        Find dependency paths between two components.
        
        Args:
            from_component: Name of the source component
            to_component: Name of the target component
            domain_name: Name of the domain
            container_name: Name of the project container
            max_depth: Maximum path depth to search
            
        Returns:
            JSON string with the dependency paths
        """
        try:
            result = self.dependency_manager.find_path(
                from_component, to_component, domain_name, container_name, max_depth
            )
            return self._standardize_response(
                result,
                f"Found dependency paths from '{from_component}' to '{to_component}' in domain '{domain_name}'",
                "dependency_path_error"
            )
        except Exception as e:
            self.logger.error(f"Error finding project dependency path: {str(e)}")
            return json.dumps(create_error_response(
                message=f"Failed to find dependency path: {str(e)}",
                code="dependency_path_error"
            ).model_dump(), default=str)
    
    def import_project_dependencies_from_code(self, dependencies: List[Dict[str, Any]],
                                  domain_name: str, 
                                  container_name: str) -> str:
        """
        Import dependencies detected from code analysis.
        
        Args:
            dependencies: List of dependencies, each with from_component, to_component, and dependency_type
            domain_name: Name of the domain
            container_name: Name of the project container
            
        Returns:
            JSON string with the import result
        """
        try:
            # Validate domain and container exist
            container_query = """
            MATCH (c:Entity {name: $container_name, entityType: 'ProjectContainer'})
            OPTIONAL MATCH (d:Entity {name: $domain_name, entityType: 'Domain'})-[:PART_OF]->(c)
            RETURN c, d
            """
            
            records = self.base_manager.safe_execute_read_query(
                container_query,
                {"container_name": container_name, "domain_name": domain_name}
            )
            
            if not records or len(records) == 0:
                return json.dumps(create_error_response(
                    message=f"Project container '{container_name}' not found",
                    code="container_not_found"
                ).model_dump(), default=str)
                
            if not records[0]["d"]:
                return json.dumps(create_error_response(
                    message=f"Domain '{domain_name}' not found in project '{container_name}'",
                    code="domain_not_found"
                ).model_dump(), default=str)
            
            # Create a list to hold validated dependencies
            validated_dependencies = []
            
            # Validate each dependency
            for i, dependency in enumerate(dependencies):
                # Check required fields
                if not all(k in dependency for k in ["from_component", "to_component", "dependency_type"]):
                    return json.dumps(create_error_response(
                        message=f"Dependency at index {i} is missing required fields (from_component, to_component, dependency_type)",
                        code="invalid_dependency"
                    ).model_dump(), default=str)
                
                # Prepare the relationship data for validation
                # We will need to query for the component IDs first
                from_component_name = dependency["from_component"]
                to_component_name = dependency["to_component"]
                dependency_type = dependency["dependency_type"]
                properties = dependency.get("properties", {})
                
                # Query for component IDs
                comp_query = """
                MATCH (comp:Entity {name: $component_name})
                MATCH (d:Entity {name: $domain_name, entityType: 'Domain'})
                WHERE (comp)-[:BELONGS_TO]->(d)
                RETURN comp.id as component_id
                """
                
                # Attempt to get source component
                from_records = self.base_manager.safe_execute_read_query(
                    comp_query,
                    {"component_name": from_component_name, "domain_name": domain_name}
                )
                
                # If component doesn't exist, we'll auto-create it
                if not from_records or len(from_records) == 0:
                    self.logger.info(f"Auto-creating source component '{from_component_name}' in domain '{domain_name}'")
                    create_result = self.create_project_component(
                        name=from_component_name,
                        component_type="AUTO_DETECTED",
                        domain_name=domain_name,
                        container_name=container_name,
                        description=f"Auto-created by dependency import"
                    )
                    
                    create_data = json.loads(create_result)
                    if "error" in create_data:
                        return json.dumps(create_error_response(
                            message=f"Failed to auto-create component '{from_component_name}': {create_data.get('error', {}).get('message', 'Unknown error')}",
                            code="component_creation_error"
                        ).model_dump(), default=str)
                        
                    # Get the newly created component's ID
                    from_records = self.base_manager.safe_execute_read_query(
                        comp_query,
                        {"component_name": from_component_name, "domain_name": domain_name}
                    )
                
                # Attempt to get target component
                to_records = self.base_manager.safe_execute_read_query(
                    comp_query,
                    {"component_name": to_component_name, "domain_name": domain_name}
                )
                
                # If component doesn't exist, we'll auto-create it
                if not to_records or len(to_records) == 0:
                    self.logger.info(f"Auto-creating target component '{to_component_name}' in domain '{domain_name}'")
                    create_result = self.create_project_component(
                        name=to_component_name,
                        component_type="AUTO_DETECTED",
                        domain_name=domain_name,
                        container_name=container_name,
                        description=f"Auto-created by dependency import"
                    )
                    
                    create_data = json.loads(create_result)
                    if "error" in create_data:
                        return json.dumps(create_error_response(
                            message=f"Failed to auto-create component '{to_component_name}': {create_data.get('error', {}).get('message', 'Unknown error')}",
                            code="component_creation_error"
                        ).model_dump(), default=str)
                        
                    # Get the newly created component's ID
                    to_records = self.base_manager.safe_execute_read_query(
                        comp_query,
                        {"component_name": to_component_name, "domain_name": domain_name}
                    )
                
                # Now we should have both component IDs
                from_component_id = from_records[0]["component_id"]
                to_component_id = to_records[0]["component_id"]
                
                # Create relationship data for validation
                relationship_data = {
                    "source_id": from_component_id,
                    "target_id": to_component_id,
                    "relationship_type": dependency_type
                }
                
                if properties:
                    relationship_data["properties"] = properties
                    
                # Validate using RelationshipCreate model
                try:
                    relationship_model = RelationshipCreate(**relationship_data)
                    validated_data = relationship_model.model_dump()
                    
                    # Add validated dependency to our list
                    validated_dependencies.append({
                        "from_component": from_component_name,
                        "to_component": to_component_name,
                        "dependency_type": validated_data["relationship_type"],
                        "properties": validated_data.get("properties", {})
                    })
                except Exception as ve:
                    self.logger.error(f"Validation error for dependency at index {i}: {str(ve)}")
                    return json.dumps(create_error_response(
                        message=f"Invalid dependency at index {i}: {str(ve)}",
                        code="validation_error"
                    ).model_dump(), default=str)
            
            # Now call the dependency manager with the validated dependencies
            result = self.dependency_manager.import_dependencies_from_code(
                dependencies=validated_dependencies,
                domain_name=domain_name,
                container_name=container_name
            )
            
            dependency_count = len(validated_dependencies) if validated_dependencies else 0
            return self._standardize_response(
                result,
                f"Imported {dependency_count} dependencies into domain '{domain_name}' of project '{container_name}'",
                "dependency_import_error"
            )
        except Exception as e:
            self.logger.error(f"Error importing project dependencies: {str(e)}")
            return json.dumps(create_error_response(
                message=f"Failed to import dependencies: {str(e)}",
                code="dependency_import_error"
            ).model_dump(), default=str)
    
    # ============================================================================
    # Version management methods
    # ============================================================================
    
    def create_project_version(self, component_name: str, domain_name: str,
                    container_name: str, version_number: str,
                    commit_hash: Optional[str] = None,
                    content: Optional[str] = None,
                    changes: Optional[str] = None,
                    metadata: Optional[Dict[str, Any]] = None) -> str:
        """
        Create a new version for a project component.
        
        Args:
            component_name: Name of the component
            domain_name: Name of the domain
            container_name: Name of the project container
            version_number: Version number (e.g., "1.0.0")
            commit_hash: Optional. Associated commit hash
            content: Optional. Content of this version
            changes: Optional. Description of changes in this version
            metadata: Optional. Additional metadata for the version
            
        Returns:
            JSON string with the created version
        """
        try:
            # Use Pydantic v2 validation with proper error handling
            from pydantic import ValidationError
            from src.models.project_memory import VersionResponse
            
            try:
                # Prepare version data for the Pydantic model
                version_data = {
                    "component_name": component_name,
                    "domain_name": domain_name,
                    "container_name": container_name,
                    "version_number": version_number,
                    "commit_hash": commit_hash,
                    "content": content,
                    "changes": changes,
                    "metadata": metadata
                }
                
                # Create and validate the VersionCreate model
                version_model = VersionCreate(**version_data)
                
                # Validate version number format using Pydantic
                # The model should handle this through its validators
                
            except ValidationError as ve:
                # Enhanced validation error handling
                self.logger.error(f"Validation error for version data: {ve}")
                error = ErrorDetail(
                    code="validation_error",
                    message="Invalid version data",
                    details={"validation_errors": ve.errors()}
                )
                error_response = ErrorResponse(
                    status="error",
                    timestamp=datetime.now(),
                    error=error
                )
                return error_response.model_dump_json()
            
            # Call the version manager with the validated model
            result = self.version_manager.create_version(
                version_data=version_model
            )
            
            # Process the response with Pydantic if possible
            try:
                # Try to parse as a VersionResponse model if available
                response_obj = VersionResponse.model_validate_json(result)
                return response_obj.model_dump_json()
            except Exception:
                # Fallback to standard response
                return self._standardize_response(
                    result,
                    f"Created version {version_number} for component '{component_name}' in domain '{domain_name}'",
                    "version_creation_error"
                )
                
        except Exception as e:
            self.logger.error(f"Error creating project version: {str(e)}")
            error = ErrorDetail(
                code="version_creation_error",
                message=f"Failed to create version: {str(e)}",
                details=None
            )
            error_response = ErrorResponse(
                status="error",
                timestamp=datetime.now(),
                error=error
            )
            return error_response.model_dump_json()
    
    def _is_valid_version(self, version: str) -> bool:
        """
        Validate a version string against semantic versioning format.
        
        Args:
            version: Version string to validate
            
        Returns:
            True if the version is valid, False otherwise
        """
        import re
        # Basic semver pattern: X.Y.Z with optional prerelease and build metadata
        pattern = r"^\d+\.\d+\.\d+(?:-[a-zA-Z0-9\.]+)?(?:\+[a-zA-Z0-9\.]+)?$"
        return bool(re.match(pattern, version))
    
    def get_project_version(self, component_name: str, domain_name: str,
                 container_name: str, 
                 version_number: Optional[str] = None) -> str:
        """
        Get a specific version of a project component.
        
        Args:
            component_name: Name of the component
            domain_name: Name of the domain
            container_name: Name of the project container
            version_number: Optional. Version number to retrieve (gets latest if not specified)
            
        Returns:
            JSON string with the version details
        """
        try:
            # Create a version request object
            version_request = VersionGetRequest(
                component_name=component_name,
                domain_name=domain_name,
                container_name=container_name,
                version_number=version_number
            )
            
            result = self.version_manager.get_version(
                version_request=version_request
            )
            
            version_desc = version_number if version_number else "latest version"
            return self._standardize_response(
                result,
                f"Retrieved {version_desc} of component '{component_name}' in domain '{domain_name}'",
                "version_retrieval_error"
            )
        except Exception as e:
            self.logger.error(f"Error retrieving project version: {str(e)}")
            return json.dumps(create_error_response(
                message=f"Failed to retrieve version: {str(e)}",
                code="version_retrieval_error"
            ).model_dump(), default=str)
    
    def list_project_versions(self, component_name: str, domain_name: str,
                   container_name: str, limit: int = 10) -> str:
        """
        List versions of a project component.
        
        Args:
            component_name: Name of the component
            domain_name: Name of the domain
            container_name: Name of the project container
            limit: Maximum number of versions to return
            
        Returns:
            JSON string with the list of versions
        """
        try:
            # Create a version request object
            version_request = VersionGetRequest(
                component_name=component_name,
                domain_name=domain_name,
                container_name=container_name
            )
            
            result = self.version_manager.list_versions(
                version_request=version_request,
                limit=limit
            )
            
            return self._standardize_response(
                result,
                f"Listed versions of component '{component_name}' in domain '{domain_name}'",
                "version_list_error"
            )
        except Exception as e:
            self.logger.error(f"Error listing project versions: {str(e)}")
            return json.dumps(create_error_response(
                message=f"Failed to list versions: {str(e)}",
                code="version_list_error"
            ).model_dump(), default=str)
    
    def get_project_version_history(self, component_name: str, domain_name: str,
                         container_name: str,
                         include_content: bool = False) -> str:
        """
        Get the version history of a project component.
        
        Args:
            component_name: Name of the component
            domain_name: Name of the domain
            container_name: Name of the project container
            include_content: Whether to include content in the history
        
        Returns:
            JSON string with the version history
        """
        try:
            # Create a version request object
            version_request = VersionGetRequest(
                component_name=component_name,
                domain_name=domain_name,
                container_name=container_name
            )
            
            result = self.version_manager.get_version_history(
                version_request=version_request,
                include_content=include_content
            )
            
            content_note = " with content" if include_content else ""
            return self._standardize_response(
                result,
                f"Retrieved version history{content_note} for component '{component_name}' in domain '{domain_name}'",
                "version_history_error"
            )
        except Exception as e:
            self.logger.error(f"Error retrieving project version history: {str(e)}")
            return json.dumps(create_error_response(
                message=f"Failed to retrieve version history: {str(e)}",
                code="version_history_error"
            ).model_dump(), default=str)
    
    def compare_project_versions(self, component_name: str, domain_name: str,
                      container_name: str, version1: str, 
                      version2: str) -> str:
        """
        Compare two versions of a project component.
        
        Args:
            component_name: Name of the component
            domain_name: Name of the domain
            container_name: Name of the project container
            version1: First version number
            version2: Second version number
        
        Returns:
            JSON string with the version comparison
        """
        try:
            # Create a version compare request object
            compare_request = VersionCompareRequest(
                component_name=component_name,
                domain_name=domain_name,
                container_name=container_name,
                version1=version1,
                version2=version2
            )
            
            result = self.version_manager.compare_versions(
                compare_request=compare_request
            )
            
            return self._standardize_response(
                result,
                f"Compared versions {version1} and {version2} of component '{component_name}'",
                "version_compare_error"
            )
        except Exception as e:
            self.logger.error(f"Error comparing project versions: {str(e)}")
            return json.dumps(create_error_response(
                message=f"Failed to compare versions: {str(e)}",
                code="version_compare_error"
            ).model_dump(), default=str)
    
    def tag_project_version(self, component_name: str, domain_name: str,
                 container_name: str, version_number: str,
                 tag_name: str, tag_description: Optional[str] = None) -> str:
        """
        Tag a version of a project component.
        
        Args:
            component_name: Name of the component
            domain_name: Name of the domain
            container_name: Name of the project container
            version_number: Version number to tag
            tag_name: Name of the tag
            tag_description: Optional. Description of the tag
        
        Returns:
            JSON string with the created tag
        """
        try:
            # Create a tag create request object
            tag_data = TagCreate(
                component_name=component_name,
                domain_name=domain_name,
                container_name=container_name,
                version_number=version_number,
                tag_name=tag_name,
                tag_description=tag_description
            )
            
            result = self.version_manager.tag_version(
                tag_data=tag_data
            )
            
            return self._standardize_response(
                result,
                f"Tagged version {version_number} of component '{component_name}' as '{tag_name}'",
                "version_tag_error"
            )
        except Exception as e:
            self.logger.error(f"Error tagging project version: {str(e)}")
            return json.dumps(create_error_response(
                message=f"Failed to tag version: {str(e)}",
                code="version_tag_error"
            ).model_dump(), default=str)
    
    def sync_project_version_control(self, component_name: str, domain_name: str,
                              container_name: str,
                              commit_data: List[Dict[str, Any]]) -> str:
        """
        Synchronize a project component with version control data.
        
        Args:
            component_name: Name of the component
            domain_name: Name of the domain
            container_name: Name of the project container
            commit_data: List of commit data dictionaries
                Each dict should have: hash, version, date, author, message, content
        
        Returns:
            JSON string with the sync results
        """
        try:
            # Convert commit data list to CommitData objects
            commits = []
            for commit in commit_data:
                commit_obj = CommitData(**commit)
                commits.append(commit_obj)
            
            # Create a sync request object
            sync_request = SyncRequest(
                component_name=component_name,
                domain_name=domain_name,
                container_name=container_name,
                commits=commits
            )
            
            result = self.version_manager.sync_with_version_control(
                sync_request=sync_request
            )
            
            commit_count = len(commit_data) if commit_data else 0
            return self._standardize_response(
                result,
                f"Synchronized {commit_count} commits for component '{component_name}'",
                "version_sync_error"
            )
        except Exception as e:
            self.logger.error(f"Error syncing project version control: {str(e)}")
            return json.dumps(create_error_response(
                message=f"Failed to sync version control: {str(e)}",
                code="version_sync_error"
            ).model_dump(), default=str)
    
    # ============================================================================
    # Project container management methods
    # ============================================================================
    
    def create_project_container(self, project_data: Dict[str, Any]) -> str:
        """
        Create a new project container.
        
        Args:
            project_data: Dictionary containing project information
                - name: Required. The name of the project container
                - description: Optional. Description of the project
                - metadata: Optional. Additional metadata for the project
                - tags: Optional. List of tags for categorizing the project
            
        Returns:
            JSON string with the created project container
        """
        try:
            # Use Pydantic v2 validation and conversion
            from pydantic import ValidationError
            from src.models.project_memory import ProjectContainerResponse
            
            try:
                # Create the Pydantic model from input data
                project_model = ProjectContainerCreate(**project_data)
            except ValidationError as ve:
                # Proper handling of Pydantic validation errors
                self.logger.error(f"Validation error for project container: {ve}")
                error = ErrorDetail(
                    code="validation_error",
                    message="Invalid project data",
                    details={"validation_errors": ve.errors()}  # Convert to dict for ErrorDetail
                )
                error_response = ErrorResponse(
                    status="error",
                    timestamp=datetime.now(),
                    error=error
                )
                return error_response.model_dump_json()
            
            # The project_container expects ProjectContainerCreate model
            result_json = self.project_container.create_project_container(project_model)
            
            # Process the response
            try:
                # Try to parse as a Pydantic response model
                response_obj = ProjectContainerResponse.model_validate_json(result_json)
                return response_obj.model_dump_json()
            except Exception:
                # Fallback to standard response handling
                project_name = project_model.name
                return self._standardize_response(
                    result_json,
                    f"Created project container '{project_name}'",
                    "container_creation_error"
                )
                
        except Exception as e:
            self.logger.error(f"Error creating project container: {str(e)}")
            error = ErrorDetail(
                code="container_creation_error",
                message=f"Failed to create project container: {str(e)}",
                details=None
            )
            error_response = ErrorResponse(
                status="error",
                timestamp=datetime.now(),
                error=error
            )
            return error_response.model_dump_json()
    
    def get_project_container(self, name: str) -> str:
        """
        Retrieve a project container.
        
        Args:
            name: Name of the project container
            
        Returns:
            JSON string with the container details
        """
        try:
            # Import the appropriate response model
            from src.models.project_memory import ProjectContainerResponse
            
            # Call the manager method
            result_json = self.project_container.get_project_container(name)
            
            # Try to parse as a Pydantic response model
            try:
                response_obj = ProjectContainerResponse.model_validate_json(result_json)
                return response_obj.model_dump_json()
            except Exception:
                # Fallback to standard response handling
                return self._standardize_response(
                    result_json,
                    f"Retrieved project container '{name}'",
                    "container_retrieval_error"
                )
                
        except Exception as e:
            self.logger.error(f"Error retrieving project container: {str(e)}")
            error = ErrorDetail(
                code="container_retrieval_error",
                message=f"Failed to retrieve project container: {str(e)}",
                details=None
            )
            error_response = ErrorResponse(
                status="error",
                timestamp=datetime.now(),
                error=error
            )
            return error_response.model_dump_json()
            
    def update_project_container(self, name: str, updates: Dict[str, Any]) -> str:
        """
        Update a project container.
        
        Args:
            name: Name of the project container
            updates: Dictionary with updates to apply
                - description: Optional. New description
                - metadata: Optional. New metadata
                - tags: Optional. New tags
            
        Returns:
            JSON string with the updated project container
        """
        try:
            # Use Pydantic v2 validation and conversion
            from pydantic import ValidationError
            from src.models.project_memory import ProjectContainerResponse
            
            try:
                # Add the ID field for ProjectContainerUpdate
                updates_with_id = {"id": name, **updates}
                update_model = ProjectContainerUpdate(**updates_with_id)
            except ValidationError as ve:
                # Proper handling of Pydantic validation errors
                self.logger.error(f"Validation error for project container update: {ve}")
                error = ErrorDetail(
                    code="validation_error",
                    message="Invalid update data",
                    details={"validation_errors": ve.errors()}  # Convert to dict for ErrorDetail
                )
                error_response = ErrorResponse(
                    status="error",
                    timestamp=datetime.now(),
                    error=error
                )
                return error_response.model_dump_json()
            
            # The project_container expects ProjectContainerUpdate model
            result_json = self.project_container.update_project_container(name, update_model)
            
            # Process the response
            try:
                # Try to parse as a Pydantic response model
                response_obj = ProjectContainerResponse.model_validate_json(result_json)
                return response_obj.model_dump_json()
            except Exception:
                # Fallback to standard response handling
                return self._standardize_response(
                    result_json,
                    f"Updated project container '{name}'",
                    "container_update_error"
                )
                
        except Exception as e:
            self.logger.error(f"Error updating project container: {str(e)}")
            error = ErrorDetail(
                code="container_update_error",
                message=f"Failed to update project container: {str(e)}",
                details=None
            )
            error_response = ErrorResponse(
                status="error",
                timestamp=datetime.now(),
                error=error
            )
            return error_response.model_dump_json()
    
    def delete_project_container(self, name: str, delete_contents: bool = False) -> str:
        """
        Delete a project container.
        
        Args:
            name: Name of the project container
            delete_contents: Whether to delete all contents of the container
            
        Returns:
            JSON string with the deletion result
        """
        try:
            # Use Pydantic for request structure if needed
            # For a simple delete, we can directly call the manager
            result_json = self.project_container.delete_project_container(name, delete_contents)
            
            # Parse the response
            result_data = json.loads(result_json)
            
            # Check for errors in the response
            if isinstance(result_data, dict) and "error" in result_data:
                error = ErrorDetail(
                    code="container_deletion_error",
                    message=result_data["error"],
                    details=None
                )
                error_response = ErrorResponse(
                    status="error",
                    timestamp=datetime.now(),
                    error=error
                )
                return error_response.model_dump_json()
            
            # Create a success response with more details
            from src.models.project_memory import SuccessResponse
            success_response = SuccessResponse(
                message=f"Successfully deleted project container '{name}'{' and all its contents' if delete_contents else ''}",
                timestamp=datetime.now()
            )
            
            # Add any additional details from the result
            response_dict = success_response.model_dump(exclude_none=True)
            if isinstance(result_data, dict):
                # Add relevant fields from the result to the response
                for key in ["deleted_components", "deleted_relationships"]:
                    if key in result_data:
                        response_dict[key] = result_data[key]
            
            return json.dumps(response_dict, default=str)
            
        except Exception as e:
            self.logger.error(f"Error deleting project container: {str(e)}")
            error = ErrorDetail(
                code="container_deletion_error",
                message=f"Failed to delete project container: {str(e)}",
                details=None
            )
            error_response = ErrorResponse(
                status="error",
                timestamp=datetime.now(),
                error=error
            )
            return error_response.model_dump_json()
    
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
            result = self.project_container.list_project_containers(sort_by, limit)
            return self._standardize_response(
                result,
                f"Listed {limit} project containers sorted by {sort_by}",
                "container_list_error"
            )
        except Exception as e:
            self.logger.error(f"Error listing project containers: {str(e)}")
            return json.dumps(create_error_response(
                message=f"Failed to list project containers: {str(e)}",
                code="container_list_error"
            ).model_dump(), default=str)
    
    def add_entity_to_project_container(self, container_name: str, entity_name: str) -> str:
        """
        Add an entity to a project container.
        
        Args:
            container_name: Name of the project container
            entity_name: Name of the entity to add
            
        Returns:
            JSON string with the result
        """
        try:
            result = self.project_container.add_component_to_container(container_name, entity_name)
            return self._standardize_response(
                result,
                f"Added entity '{entity_name}' to project container '{container_name}'",
                "container_add_entity_error"
            )
        except Exception as e:
            self.logger.error(f"Error adding entity to project container: {str(e)}")
            return json.dumps(create_error_response(
                message=f"Failed to add entity to project container: {str(e)}",
                code="container_add_entity_error"
            ).model_dump(), default=str)
    
    def remove_entity_from_project_container(self, container_name: str, entity_name: str) -> str:
        """
        Remove an entity from a project container.
        
        Args:
            container_name: Name of the project container
            entity_name: Name of the entity to remove
            
        Returns:
            JSON string with the result
        """
        try:
            result = self.project_container.remove_component_from_container(container_name, entity_name)
            return self._standardize_response(
                result,
                f"Removed entity '{entity_name}' from project container '{container_name}'",
                "container_remove_entity_error"
            )
        except Exception as e:
            self.logger.error(f"Error removing entity from project container: {str(e)}")
            return json.dumps(create_error_response(
                message=f"Failed to remove entity from project container: {str(e)}",
                code="container_remove_entity_error"
            ).model_dump(), default=str)
    
    def get_project_container_entities(self, container_name: str, entity_type: Optional[str] = None) -> str:
        """
        Get all entities in a project container.
        
        Args:
            container_name: Name of the project container
            entity_type: Optional entity type to filter by
            
        Returns:
            JSON string with the entities
        """
        try:
            result = self.project_container.get_container_components(container_name, entity_type)
            type_filter = f" of type '{entity_type}'" if entity_type else ""
            return self._standardize_response(
                result,
                f"Retrieved entities{type_filter} from project container '{container_name}'",
                "container_entities_error"
            )
        except Exception as e:
            self.logger.error(f"Error retrieving project container entities: {str(e)}")
            return json.dumps(create_error_response(
                message=f"Failed to retrieve project container entities: {str(e)}",
                code="container_entities_error"
            ).model_dump(), default=str)
    
    def change_project_container_status(self, container_name: str, status: str) -> str:
        """
        Change the status of a project container.
        
        Args:
            container_name: Name of the project container
            status: New status ('active', 'archived', 'completed')
            
        Returns:
            JSON string with the result
        """
        try:
            result = self.project_container.change_container_status(container_name, status)
            return self._standardize_response(
                result,
                f"Changed project container '{container_name}' status to '{status}'",
                "container_status_change_error"
            )
        except Exception as e:
            self.logger.error(f"Error changing project container status: {str(e)}")
            return json.dumps(create_error_response(
                message=f"Failed to change project container status: {str(e)}",
                code="container_status_change_error"
            ).model_dump(), default=str)
    
    def get_project_container_stats(self, container_name: str) -> str:
        """
        Get detailed statistics for a project container.
        
        Args:
            container_name: Name of the project container
            
        Returns:
            JSON string with container statistics
        """
        try:
            result = self.project_container.get_container_stats(container_name)
            return self._standardize_response(
                result,
                f"Retrieved statistics for project container '{container_name}'",
                "container_stats_error"
            )
        except Exception as e:
            self.logger.error(f"Error retrieving project container statistics: {str(e)}")
            return json.dumps(create_error_response(
                message=f"Failed to retrieve project container statistics: {str(e)}",
                code="container_stats_error"
            ).model_dump(), default=str)
    
    def get_project_status(self, container_name: str) -> str:
        """
        Get a summary of the project container status, including domains and components.
        
        Args:
            container_name: Name of the project container
            
        Returns:
            JSON string with the project status
        """
        try:
            # Get container details using the container manager directly
            # This avoids issues during the transition
            container_json = self.project_container.get_project_container(container_name)
            container_data = json.loads(container_json)
            
            if "error" in container_data:
                return json.dumps(create_error_response(
                    message=container_data["error"],
                    code="container_not_found"
                ).model_dump(), default=str)
            
            container = container_data.get("container", {})
            
            # Get domains using the domain manager directly
            domains_json = self.domain_manager.list_domains(container_name)
            domains_data = json.loads(domains_json)
            
            if "error" in domains_data:
                return json.dumps(create_error_response(
                    message=domains_data["error"],
                    code="domain_list_error"
                ).model_dump(), default=str)
            
            domains = domains_data.get("domains", [])
        
            # Get component counts by type using ProjectContainer stats
            stats_json = self.project_container.get_container_stats(container_name)
            stats = json.loads(stats_json)
            
            if "error" in stats:
                    return json.dumps(create_error_response(
                        message=stats["error"],
                        code="stats_error"
                    ).model_dump(), default=str)
            
            # Build status report
            status = {
                "container": container,
                "domains": domains,
                "component_counts": stats.get("entity_types", {}),
                "dependency_counts": stats.get("relationship_types", {}),
                "total_domains": len(domains),
                "total_components": stats.get("total_entities", 0),
                "total_dependencies": stats.get("total_relationships", 0)
            }
        
            # Return standardized success response
            return json.dumps(create_success_response(
                message=f"Retrieved status for project '{container_name}' with {len(domains)} domains and {stats.get('total_entities', 0)} components",
                data=status
            ).model_dump(), default=str)
            
        except Exception as e:
            self.logger.error(f"Error retrieving project status: {str(e)}")
            return json.dumps(create_error_response(
                message=f"Failed to retrieve project status: {str(e)}",
                code="status_retrieval_error"
            ).model_dump(), default=str)

    # ============================================================================
    # Search methods
    # ============================================================================
    
    def search_project_entities(self, search_term: str, container_name: str, 
                      entity_types: Optional[List[str]] = None,
                      limit: int = 10, semantic: bool = False) -> str:
        """
        Search for entities in a project container.
        
        Args:
            search_term: Search query term
            container_name: Name of the project container
            entity_types: Optional. List of entity types to filter by
            limit: Maximum number of results to return
            semantic: Whether to use semantic search
            
        Returns:
            JSON string with search results
        """
        try:
            # Use Pydantic v2 for search query validation
            from pydantic import ValidationError
            from src.models.project_memory import SearchQuery, SearchResponse
            
            try:
                # Create a search query model
                search_data = {
                    "query": search_term,
                    "entity_types": entity_types,
                    "limit": limit
                }
                
                # Validate using Pydantic
                search_query = SearchQuery(**search_data)
                
                # We can use computed fields if needed
                if hasattr(search_query, 'has_filters'):
                    self.logger.debug(f"Search uses filters: {search_query.has_filters}")
                
            except ValidationError as ve:
                # Enhanced validation error handling
                self.logger.error(f"Validation error for search query: {ve}")
                error = ErrorDetail(
                    code="validation_error",
                    message="Invalid search parameters",
                    details={"validation_errors": ve.errors()}
                )
                error_response = ErrorResponse(
                    status="error",
                    timestamp=datetime.now(),
                    error=error
                )
                return error_response.model_dump_json()
            
            # Choose search method based on semantic flag
            if semantic:
                result = self.semantic_search_project(
                    search_term, container_name, entity_types, limit
                )
            else:
                # Construct the search query - we're directly passing parameters here
                # but we could modify this to use the search_query model's fields
                query = f"""
                MATCH (c:Entity {{name: $container_name, entityType: 'ProjectContainer'}})
                MATCH (e:Entity)-[:BELONGS_TO|PART_OF*..2]->(c)
                WHERE (e.name CONTAINS $search_term OR 
                      e.description CONTAINS $search_term OR
                      e.content CONTAINS $search_term)
                      {f"AND e.entityType IN $entity_types" if entity_types else ""}
                RETURN e
                ORDER BY e.name
                LIMIT $limit
                """
                
                # Parameters for the search
                params = {
                    "container_name": container_name,
                    "search_term": search_term,
                    "limit": limit
                }
                
                if entity_types:
                    params["entity_types"] = entity_types
                
                # Execute the search
                records = self.base_manager.safe_execute_read_query(query, params)
                
                # Process the results
                results = []
                for record in records:
                    entity = dict(record["e"].items())
                    results.append(entity)
                
                # Create a Pydantic SearchResponse model
                search_response = SearchResponse(
                    status="success",
                    timestamp=datetime.now(),
                    message=f"Found {len(results)} results for query '{search_term}'",
                    results=results,
                    total_count=len(results),
                    query=search_term
                )
                
                # Return as JSON
                return search_response.model_dump_json()
            
            # If we used semantic search, just return its result
            return result
            
        except Exception as e:
            self.logger.error(f"Error searching project entities: {str(e)}")
            error = ErrorDetail(
                code="search_error",
                message=f"Failed to search entities: {str(e)}",
                details=None
            )
            error_response = ErrorResponse(
                status="error",
                timestamp=datetime.now(),
                error=error
            )
            return error_response.model_dump_json()
    
    def _get_entity_domain(self, entity_id: str) -> Optional[str]:
        """
        Helper method to get the domain name for an entity.
        
        Args:
            entity_id: ID of the entity
            
        Returns:
            Domain name if found, None otherwise
        """
        try:
            query = """
            MATCH (e:Entity {id: $entity_id})-[:BELONGS_TO]->(d:Entity {entityType: 'Domain'})
            RETURN d.name as domain_name
            """
            
            records = self.base_manager.safe_execute_read_query(
                query,
                {"entity_id": entity_id}
            )
            
            if records and len(records) > 0:
                return records[0]["domain_name"]
            return None
            
        except Exception:
            return None
    
    def semantic_search_project(self, search_term: str, container_name: str,
                              entity_types: Optional[List[str]] = None, 
                              limit: int = 10) -> str:
        """
        Perform a semantic search for entities in a project using vector embeddings.
        This is a convenience method that calls search_project_entities with semantic=True.
        
        Args:
            search_term: The natural language term to search for
            container_name: Name of the project container to search within
            entity_types: Optional list of entity types to filter by
            limit: Maximum number of results to return
            
        Returns:
            JSON string with the search results
        """
        try:
            # Validate search parameters using SearchQuery model
            search_data = {
                "query": search_term,
                "entity_types": entity_types,
                "limit": limit
            }
            
            try:
                # Use SearchQuery from project_memory.py for validation
                search_model = SearchQuery(**search_data)
                validated_data = search_model.model_dump()
            except Exception as ve:
                self.logger.error(f"Validation error for semantic search query: {str(ve)}")
                return json.dumps(create_error_response(
                    message=f"Invalid search query: {str(ve)}",
                    code="validation_error"
                ).model_dump(), default=str)
            
            if not self.base_manager.embedding_enabled:
                self.logger.warn("Semantic search requested but embeddings are not enabled")
                result_data = {
                    "warning": "Semantic search not available - embeddings are not enabled",
                    "fallback": "Using text-based search instead"
                }
                
                # Fall back to text-based search
                text_search_result = self.search_project_entities(
                    search_term=validated_data["query"],
                    container_name=container_name,
                    entity_types=validated_data.get("entity_types"),
                    limit=validated_data["limit"],
                    semantic=False
                )
                
                # Parse and add warning
                text_result_data = json.loads(text_search_result)
                if "data" in text_result_data:
                    text_result_data["data"]["warning"] = result_data["warning"]
                    text_result_data["data"]["fallback"] = result_data["fallback"]
                    return json.dumps(text_result_data, default=str)
                
                return text_search_result
            
            # Call the main search method with semantic=True and validated parameters
            return self.search_project_entities(
                search_term=validated_data["query"],
                container_name=container_name,
                entity_types=validated_data.get("entity_types"),
                limit=validated_data["limit"],
                semantic=True
            )
        except Exception as e:
            self.logger.error(f"Error in semantic search: {str(e)}")
            return json.dumps(create_error_response(
                message=f"Failed to perform semantic search: {str(e)}",
                code="semantic_search_error"
            ).model_dump(), default=str)
    
    def search_project_components(self, search_term: str, container_name: str, 
                                component_types: Optional[List[str]] = None,
                                limit: int = 10, semantic: bool = False) -> str:
        """
        Search specifically for components within a project container.
        
        Args:
            search_term: The term to search for
            container_name: Name of the project container to search within
            component_types: Optional list of component types to filter by (e.g., 'Service', 'UI', 'Library')
            limit: Maximum number of results to return
            semantic: Whether to perform semantic (embedding-based) search
            
        Returns:
            JSON string with the search results
        """
        try:
            # Validate search parameters using SearchQuery model
            search_data = {
                "query": search_term,
                "limit": limit
            }
            
            try:
                # Use SearchQuery from project_memory.py for validation
                search_model = SearchQuery(**search_data)
                validated_data = search_model.model_dump()
            except Exception as ve:
                self.logger.error(f"Validation error for component search query: {str(ve)}")
                return json.dumps(create_error_response(
                    message=f"Invalid search query: {str(ve)}",
                    code="validation_error"
                ).model_dump(), default=str)
            
            # For components, we always set entity_types to relevant component entity types
            # Common component entity types in the project memory system
            component_entity_types = ["Component", "File", "Service", "Module", "Feature", "Library", "UI"]
            
            # If specific component types are provided, filter by those
            if component_types and len(component_types) > 0:
                # Ensure all specified component types exist in our predefined list
                # This prevents searching for invalid entity types
                entity_types = [t for t in component_types if t in component_entity_types]
                if not entity_types:
                    # If none of the specified types are valid, use all component types
                    entity_types = component_entity_types
            else:
                # Default to all component entity types
                entity_types = component_entity_types
            
            # Delegate to the main search method with validated parameters
            result = self.search_project_entities(
                search_term=validated_data["query"],
                container_name=container_name,
                entity_types=entity_types,
                limit=validated_data["limit"],
                semantic=semantic
            )
            
            # Parse and modify the result to highlight that this is a component search
            try:
                result_data = json.loads(result)
                if "data" in result_data:
                    result_data["data"]["component_types"] = entity_types
                    result_data["data"]["component_search"] = True
                    
                    # Update the message to specify components
                    if "message" in result_data:
                        component_count = result_data["data"].get("total_count", 0)
                        search_type = "semantic" if semantic else "text"
                        result_data["message"] = f"Found {component_count} components matching '{validated_data['query']}' using {search_type} search"
                    
                    return json.dumps(result_data, default=str)
                
                return result
            except json.JSONDecodeError:
                # If we can't parse the result, return it as is
                return result
            
        except Exception as e:
            self.logger.error(f"Error searching project components: {str(e)}")
            return json.dumps(create_error_response(
                message=f"Failed to search project components: {str(e)}",
                code="component_search_error"
            ).model_dump(), default=str)
