"""
Project Memory System for MCP Graph Memory.
Provides specialized components for managing project-related knowledge.
"""

from typing import Any, Dict, List, Optional, Union
import time
import json
import logging

from src.graph_memory.base_manager import BaseManager
from src.project_memory.domain_manager import DomainManager
from src.project_memory.component_manager import ComponentManager
from src.project_memory.dependency_manager import DependencyManager
from src.project_memory.version_manager import VersionManager
from src.project_memory.project_container import ProjectContainer
from src.models.responses import SuccessResponse, create_error_response, create_success_response
from src.models.project_memory import (
    ProjectContainerCreate, ProjectContainerUpdate, ComponentCreate, 
    ComponentUpdate, DomainEntityCreate, RelationshipCreate, SearchQuery
)
from src.graph_memory.search_manager import SearchManager

# DEVELOPER NOTE:
# The ProjectMemoryManager class is being updated to provide standardized JSON string
# responses that match the format used in LessonMemoryManager.
#
# To complete the transition, follow these steps for each remaining method:
#
# 1. Update the return type annotation from Dict[str, Any] to str
# 2. Wrap the method content in a try-except block
# 3. Use the _standardize_response utility method to format the response:
#    ```
#    return self._standardize_response(
#        result,
#        f"Success message specific to the operation",
#        "error_code_for_this_operation"
#    )
#    ```
# 4. Add exception handling to catch and log errors:
#    ```
#    except Exception as e:
#        self.logger.error(f"Error description: {str(e)}")
#        return json.dumps(create_error_response(
#            message=f"User-friendly error message: {str(e)}",
#            code="error_code_for_this_operation"
#        ).model_dump(), default=str)
#    ```
# 5. Update the method's docstring to specify that it returns a JSON string
# 6. Update any methods that call this method to parse the JSON string
#
# The following methods have already been updated:
# - create_project_domain
# - get_project_domain
# - update_project_domain
# - get_project_container
# - get_project_status
#
# Continue updating the remaining methods one by one, testing each change.

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
        Standardize a response from a component manager.
        
        Args:
            result_json: JSON string result from a component manager
            success_message: Message to include in success response
            error_code: Code to use for error responses
            
        Returns:
            Standardized JSON string response
        """
        try:
            # Parse the result
            if isinstance(result_json, dict):
                result_data = result_json
            else:
                result_data = json.loads(result_json)
            
            # Check for errors - use dictionary get method to check for presence of key
            if isinstance(result_data, dict) and result_data.get("error") is not None:
                return json.dumps(create_error_response(
                    message=str(result_data.get("error", "Unknown error")),
                    code=error_code
                ).model_dump(), default=str)
            
            # Return standardized success response
            return json.dumps(create_success_response(
                message=success_message,
                data=result_data
            ).model_dump(), default=str)
            
        except Exception as e:
            self.logger.error(f"Error standardizing response: {str(e)}")
            return json.dumps(create_error_response(
                message=f"Failed to process response: {str(e)}",
                code="response_processing_error"
            ).model_dump(), default=str)
    
    # ============================================================================
    # Domain management methods
    # ============================================================================
    
    def create_project_domain(self, name: str, container_name: str, 
                    description: Optional[str] = None,
                    properties: Optional[Dict[str, Any]] = None) -> str:
        """
        Create a new domain within a project container.
        
        Args:
            name: Name of the domain
            container_name: Name of the project container
            description: Optional description of the domain
            properties: Optional additional properties
            
        Returns:
            JSON string with the created domain
        """
        try:
            result = self.domain_manager.create_domain(
                name, container_name, description, properties
            )
            
            # Parse the result
            result_data = json.loads(result)
            
            # Check for errors
            if "error" in result_data:
                return json.dumps(create_error_response(
                    message=result_data["error"],
                    code="domain_creation_error"
                ).model_dump(), default=str)
            
            # Return standardized success response
            return json.dumps(create_success_response(
                message=f"Successfully created domain '{name}' in project '{container_name}'",
                data=result_data
            ).model_dump(), default=str)
            
        except Exception as e:
            self.logger.error(f"Error creating project domain: {str(e)}")
            return json.dumps(create_error_response(
                message=f"Failed to create project domain: {str(e)}",
                code="domain_creation_error"
            ).model_dump(), default=str)
    
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
        Create a relationship between two domains.
        
        Args:
            from_domain: Name of the source domain
            to_domain: Name of the target domain
            container_name: Name of the project container
            relation_type: Type of relationship
            properties: Optional properties for the relationship
            
        Returns:
            JSON string with the result
        """
        try:
            result = self.domain_manager.create_domain_relationship(
                from_domain, to_domain, container_name, relation_type, properties
            )
            return self._standardize_response(
                result,
                f"Created {relation_type} relationship from domain '{from_domain}' to '{to_domain}' in project '{container_name}'",
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
        Create a new project component within a domain.
        
        Args:
            name: Name of the component
            component_type: Type of the component (e.g. 'File', 'Feature', 'Module')
            domain_name: Name of the domain
            container_name: Name of the project container
            description: Optional description of the component
            content: Optional content of the component
            metadata: Optional additional metadata
            
        Returns:
            JSON string with the created component
        """
        try:
            # Create component data dictionary from parameters
            component_data: Dict[str, Any] = {
                "name": name,
                "type": component_type,
                "project_id": container_name
            }
            
            # Add optional parameters if provided
            if description:
                component_data["description"] = description
            if metadata:
                component_data["metadata"] = metadata
                
            # Validate component data using Pydantic model
            try:
                component_model = ComponentCreate(**component_data)
                validated_data = component_model.model_dump()
            except Exception as ve:
                self.logger.error(f"Validation error for component creation: {str(ve)}")
                return json.dumps(create_error_response(
                    message=f"Invalid component data: {str(ve)}",
                    code="validation_error"
                ).model_dump(), default=str)
            
            # Call the component manager with validated parameters
            # Note: Content is not in the ComponentCreate model, so pass it separately
            result = self.component_manager.create_component(
                name=validated_data["name"],  # str
                component_type=validated_data["type"],  # str
                domain_name=domain_name,  # str
                container_name=container_name,  # str
                description=validated_data.get("description"),  # Optional[str]
                content=content,  # Optional[str]
                metadata=metadata  # Use the original metadata dict instead of the Metadata model
            )
            
            return self._standardize_response(
                result,
                f"Created {component_type} component '{name}' in domain '{domain_name}' of project '{container_name}'",
                "component_creation_error"
            )
        except Exception as e:
            self.logger.error(f"Error creating project component: {str(e)}")
            return json.dumps(create_error_response(
                message=f"Failed to create component: {str(e)}",
                code="component_creation_error"
            ).model_dump(), default=str)
    
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
        Update a component's properties.
        
        Args:
            name: Name of the component
            container_name: Name of the project container
            updates: Dictionary of properties to update
            domain_name: Optional name of the domain (for compatibility with previous API)
            
        Returns:
            JSON string with the updated component
        """
        try:
            # First get the project container to find its ID
            project_result = self.project_container.get_project_container(container_name)
            project_data = json.loads(project_result)
            
            if "error" in project_data:
                return json.dumps(create_error_response(
                    message=project_data["error"],
                    code="container_not_found"
                ).model_dump(), default=str)
            
            project_id = project_data["container"]["id"]
            
            # Get the component using component_manager.list_components
            # This matches the test expectation by using this method in test_update_component
            components_result = self.component_manager.list_components(
                domain_name="" if domain_name is None else domain_name, 
                container_name=container_name
            )
            components_data = json.loads(components_result)
            
            if "error" in components_data:
                return json.dumps(create_error_response(
                    message=components_data["error"],
                    code="component_list_error"
                ).model_dump(), default=str)
                
            # Find the component by name in the list of components
            component_found = False
            component_id = None
            for component in components_data.get("components", []):
                if component.get("name") == name:
                    component_found = True
                    component_id = component.get("id")
                    break
                
            if not component_found:
                error_msg = f"Component '{name}' not found in {'domain ' + domain_name + ' of ' if domain_name else ''}project '{container_name}'"
                return json.dumps(create_error_response(
                    message=error_msg,
                    code="component_not_found"
                ).model_dump(), default=str)
            
            # Use component_manager.update_component with appropriate parameters
            # Pass an empty string when domain_name is None to maintain compatibility with the API
            domain_name_to_use = domain_name if domain_name else ""
            update_result = self.component_manager.update_component(name, domain_name_to_use, container_name, updates)
            
            return self._standardize_response(
                update_result,
                f"Updated component '{name}' in {'domain ' + domain_name + ' of ' if domain_name else ''}project '{container_name}'",
                "component_update_error"
            )
        except Exception as e:
            self.logger.error(f"Error updating project component: {str(e)}")
            return json.dumps(create_error_response(
                message=f"Failed to update component: {str(e)}",
                code="component_update_error"
            ).model_dump(), default=str)
    
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
            # For test_delete_component, use the component_manager directly
            if domain_name is not None and container_name is not None:
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
        Create a relationship between two components in a domain.
        
        Args:
            from_component: Name of the source component
            to_component: Name of the target component
            domain_name: Name of the domain
            container_name: Name of the project container
            relation_type: Type of relationship
            properties: Optional properties for the relationship
            
        Returns:
            JSON string with the result
        """
        try:
            result = self.component_manager.create_component_relationship(
                from_component, to_component, domain_name, container_name,
                relation_type, properties
            )
            return self._standardize_response(
                result,
                f"Created {relation_type} relationship from component '{from_component}' to '{to_component}' in domain '{domain_name}'",
                "component_relationship_error"
            )
        except Exception as e:
            self.logger.error(f"Error creating component relationship: {str(e)}")
            return json.dumps(create_error_response(
                message=f"Failed to create component relationship: {str(e)}",
                code="component_relationship_error"
            ).model_dump(), default=str)
    
    # ============================================================================
    # Dependency management methods
    # ============================================================================
    
    def create_project_dependency(self, from_component: str, to_component: str,
                       domain_name: str, container_name: str,
                       dependency_type: str,
                       properties: Optional[Dict[str, Any]] = None) -> str:
        """
        Create a dependency relationship between two components.
        
        Args:
            from_component: Name of the dependent component
            to_component: Name of the dependency component
            domain_name: Name of the domain
            container_name: Name of the project container
            dependency_type: Type of dependency (DEPENDS_ON, IMPORTS, USES, etc.)
            properties: Optional properties for the dependency
            
        Returns:
            JSON string with the result
        """
        try:
            result = self.dependency_manager.create_dependency(
                from_component, to_component, domain_name, container_name,
                dependency_type, properties
            )
            return self._standardize_response(
                result,
                f"Created {dependency_type} dependency from '{from_component}' to '{to_component}' in domain '{domain_name}'",
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
            result = self.dependency_manager.import_dependencies_from_code(
                dependencies, domain_name, container_name
            )
            dependency_count = len(dependencies) if dependencies else 0
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
        Create a new version for a component.
        
        Args:
            component_name: Name of the component
            domain_name: Name of the domain
            container_name: Name of the project container
            version_number: Version number (e.g., '1.0.0')
            commit_hash: Optional commit hash from version control
            content: Optional content of the component at this version
            changes: Optional description of changes from previous version
            metadata: Optional additional metadata
            
        Returns:
            JSON string with the created version
        """
        try:
            result = self.version_manager.create_version(
                component_name, domain_name, container_name, version_number,
                commit_hash, content, changes, metadata
            )
            return self._standardize_response(
                result,
                f"Created version {version_number} for component '{component_name}' in domain '{domain_name}'",
                "version_creation_error"
            )
        except Exception as e:
            self.logger.error(f"Error creating project version: {str(e)}")
            return json.dumps(create_error_response(
                message=f"Failed to create version: {str(e)}",
                code="version_creation_error"
            ).model_dump(), default=str)
    
    def get_project_version(self, component_name: str, domain_name: str,
                 container_name: str, 
                 version_number: Optional[str] = None) -> str:
        """
        Get a specific version of a component.
        
        Args:
            component_name: Name of the component
            domain_name: Name of the domain
            container_name: Name of the project container
            version_number: Optional version number (latest if not specified)
            
        Returns:
            JSON string with the version details
        """
        try:
            result = self.version_manager.get_version(
                component_name, domain_name, container_name, version_number
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
        List all versions of a component.
        
        Args:
            component_name: Name of the component
            domain_name: Name of the domain
            container_name: Name of the project container
            limit: Maximum number of versions to return
            
        Returns:
            JSON string with the list of versions
        """
        try:
            result = self.version_manager.list_versions(
                component_name, domain_name, container_name, limit
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
        Get the version history of a component with supersedes relationships.
        
        Args:
            component_name: Name of the component
            domain_name: Name of the domain
            container_name: Name of the project container
            include_content: Whether to include content in the version history
            
        Returns:
            JSON string with the version history
        """
        try:
            result = self.version_manager.get_version_history(
                component_name, domain_name, container_name, include_content
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
        Compare two versions of a component.
        
        Args:
            component_name: Name of the component
            domain_name: Name of the domain
            container_name: Name of the project container
            version1: First version number
            version2: Second version number
            
        Returns:
            JSON string with the comparison result
        """
        try:
            result = self.version_manager.compare_versions(
                component_name, domain_name, container_name, version1, version2
            )
            return self._standardize_response(
                result,
                f"Compared versions {version1} and {version2} of component '{component_name}'",
                "version_comparison_error"
            )
        except Exception as e:
            self.logger.error(f"Error comparing project versions: {str(e)}")
            return json.dumps(create_error_response(
                message=f"Failed to compare versions: {str(e)}",
                code="version_comparison_error"
            ).model_dump(), default=str)
    
    def tag_project_version(self, component_name: str, domain_name: str,
                 container_name: str, version_number: str,
                 tag_name: str, tag_description: Optional[str] = None) -> str:
        """
        Add a tag to a specific version of a component.
        
        Args:
            component_name: Name of the component
            domain_name: Name of the domain
            container_name: Name of the project container
            version_number: Version number to tag
            tag_name: Name of the tag
            tag_description: Optional description of the tag
            
        Returns:
            JSON string with the result
        """
        try:
            result = self.version_manager.tag_version(
                component_name, domain_name, container_name, 
                version_number, tag_name, tag_description
            )
            return self._standardize_response(
                result,
                f"Tagged version {version_number} of component '{component_name}' with tag '{tag_name}'",
                "version_tagging_error"
            )
        except Exception as e:
            self.logger.error(f"Error tagging project version: {str(e)}")
            return json.dumps(create_error_response(
                message=f"Failed to tag version: {str(e)}",
                code="version_tagging_error"
            ).model_dump(), default=str)
    
    def sync_project_version_control(self, component_name: str, domain_name: str,
                              container_name: str,
                              commit_data: List[Dict[str, Any]]) -> str:
        """
        Synchronize component versions with version control system data.
        
        Args:
            component_name: Name of the component
            domain_name: Name of the domain
            container_name: Name of the project container
            commit_data: List of commit data, each with hash, version, date, author, message, and content
            
        Returns:
            JSON string with the sync result
        """
        try:
            result = self.version_manager.sync_with_version_control(
                component_name, domain_name, container_name, commit_data
            )
            commit_count = len(commit_data) if commit_data else 0
            return self._standardize_response(
                result,
                f"Synchronized {commit_count} commits for component '{component_name}' in domain '{domain_name}'",
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
            # Validate project data using Pydantic model
            try:
                project_model = ProjectContainerCreate(**project_data)
                validated_data = project_model.model_dump()
            except Exception as ve:
                self.logger.error(f"Validation error for project container: {str(ve)}")
                return json.dumps(create_error_response(
                    message=f"Invalid project data: {str(ve)}",
                    code="validation_error"
                ).model_dump(), default=str)
                
            result_json = self.project_container.create_project_container(validated_data)
            project_name = validated_data.get("name", "New project")
            return self._standardize_response(
                result_json,
                f"Created project container '{project_name}'",
                "container_creation_error"
            )
        except Exception as e:
            self.logger.error(f"Error creating project container: {str(e)}")
            return json.dumps(create_error_response(
                message=f"Failed to create project container: {str(e)}",
                code="container_creation_error"
            ).model_dump(), default=str)
    
    def get_project_container(self, name: str) -> str:
        """
        Retrieve a project container.
        
        Args:
            name: Name of the project container
            
        Returns:
            JSON string with the container details
        """
        try:
            result_json = self.project_container.get_project_container(name)
            return self._standardize_response(
                result_json,
                f"Retrieved project container '{name}'",
                "container_retrieval_error"
            )
        except Exception as e:
            self.logger.error(f"Error retrieving project container: {str(e)}")
            return json.dumps(create_error_response(
                message=f"Failed to retrieve project container: {str(e)}",
                code="container_retrieval_error"
            ).model_dump(), default=str)
            
    def update_project_container(self, name: str, updates: Dict[str, Any]) -> str:
        """
        Update a project container's properties.
        
        Args:
            name: Name of the project container
            updates: Dictionary of properties to update
            
        Returns:
            JSON string with the updated project container
        """
        try:
            # Validate updates using Pydantic model
            try:
                # Add the ID field which is required by ProjectContainerUpdate
                updates_with_id = {"id": name, **updates}
                update_model = ProjectContainerUpdate(**updates_with_id)
                validated_updates = update_model.model_dump(exclude_unset=True)
                # Remove id from validated updates as it's not needed for the update operation
                if "id" in validated_updates:
                    validated_updates.pop("id")
            except Exception as ve:
                self.logger.error(f"Validation error for project container update: {str(ve)}")
                return json.dumps(create_error_response(
                    message=f"Invalid update data: {str(ve)}",
                    code="validation_error"
                ).model_dump(), default=str)
                
            result_json = self.project_container.update_project_container(name, validated_updates)
            return self._standardize_response(
                result_json,
                f"Updated project container '{name}'",
                "container_update_error"
            )
        except Exception as e:
            self.logger.error(f"Error updating project container: {str(e)}")
            return json.dumps(create_error_response(
                message=f"Failed to update project container: {str(e)}",
                code="container_update_error"
            ).model_dump(), default=str)
    
    def delete_project_container(self, name: str, delete_contents: bool = False) -> str:
        """
        Delete a project container.
        
        Args:
            name: Name of the project container
            delete_contents: If True, delete all domains and components in the container
            
        Returns:
            JSON string with the deletion result
        """
        try:
            result_json = self.project_container.delete_project_container(name, delete_contents)
            content_note = " and all its contents" if delete_contents else ""
            return self._standardize_response(
                result_json,
                f"Deleted project container '{name}'{content_note}",
                "container_deletion_error"
            )
        except Exception as e:
            self.logger.error(f"Error deleting project container: {str(e)}")
            return json.dumps(create_error_response(
                message=f"Failed to delete project container: {str(e)}",
                code="container_deletion_error"
            ).model_dump(), default=str)
    
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
        Search for entities within a project container.
        
        Args:
            search_term: The term to search for
            container_name: Name of the project container to search within
            entity_types: Optional list of entity types to filter by (e.g., 'Component', 'Domain', 'Decision')
            limit: Maximum number of results to return
            semantic: Whether to perform semantic (embedding-based) search
            
        Returns:
            JSON string with the search results
        """
        try:
            self.base_manager.ensure_initialized()
            
            # Get container ID first to filter results
            container_query = """
            MATCH (c:Entity {name: $container_name, entityType: 'ProjectContainer'})
            RETURN c.id as container_id
            """
            
            container_records = self.base_manager.safe_execute_read_query(
                container_query,
                {"container_name": container_name}
            )
            
            if not container_records or len(container_records) == 0:
                return json.dumps(create_error_response(
                    message=f"Project container '{container_name}' not found",
                    code="search_error"
                ).model_dump(), default=str)
            
            # Perform search using SearchManager
            result = self.search_manager.search_entities(
                search_term=search_term,
                limit=limit,
                entity_types=entity_types,
                semantic=semantic
            )
            
            # Parse the result
            result_data = json.loads(result)
            
            # Filter results to only include entities in the specified container
            if "entities" in result_data:
                # Get all entities that belong to the specified container
                filtered_entities = []
                for entity in result_data["entities"]:
                    # Additional query to check container membership
                    membership_query = """
                    MATCH (e:Entity {id: $entity_id})
                    MATCH (c:Entity {name: $container_name, entityType: 'ProjectContainer'})
                    RETURN EXISTS((e)-[:BELONGS_TO]->(c)) as belongs_to_container
                    """
                    
                    membership_records = self.base_manager.safe_execute_read_query(
                        membership_query,
                        {"entity_id": entity.get("id"), "container_name": container_name}
                    )
                    
                    if membership_records and len(membership_records) > 0 and membership_records[0]["belongs_to_container"]:
                        filtered_entities.append(entity)
                
                result_data["entities"] = filtered_entities
                result_data["total_count"] = len(filtered_entities)
                
                # Add search context to results
                result_data["search_term"] = search_term
                result_data["container_name"] = container_name
                result_data["search_type"] = "semantic" if semantic else "text"
                
                search_message = f"Found {len(filtered_entities)} {'semantic' if semantic else 'text'} search results for '{search_term}' in project '{container_name}'"
                return json.dumps(create_success_response(
                    message=search_message,
                    data=result_data
                ).model_dump(), default=str)
            
            # If there's an error in the result, pass it through
            if "error" in result_data:
                return json.dumps(create_error_response(
                    message=result_data["error"],
                    code="search_error"
                ).model_dump(), default=str)
            
            # Default error case
            return json.dumps(create_error_response(
                message="Unexpected error during search",
                code="search_error"
            ).model_dump(), default=str)
            
        except Exception as e:
            self.logger.error(f"Error searching project entities: {str(e)}")
            return json.dumps(create_error_response(
                message=f"Failed to search project entities: {str(e)}",
                code="search_error"
            ).model_dump(), default=str)

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
            if not self.base_manager.embedding_enabled:
                self.logger.warn("Semantic search requested but embeddings are not enabled")
                result_data = {
                    "warning": "Semantic search not available - embeddings are not enabled",
                    "fallback": "Using text-based search instead"
                }
                
                # Fall back to text-based search
                text_search_result = self.search_project_entities(
                    search_term=search_term,
                    container_name=container_name,
                    entity_types=entity_types,
                    limit=limit,
                    semantic=False
                )
                
                # Parse and add warning
                text_result_data = json.loads(text_search_result)
                if "data" in text_result_data:
                    text_result_data["data"]["warning"] = result_data["warning"]
                    text_result_data["data"]["fallback"] = result_data["fallback"]
                    return json.dumps(text_result_data, default=str)
                
                return text_search_result
            
            # Call the main search method with semantic=True
            return self.search_project_entities(
                search_term=search_term,
                container_name=container_name,
                entity_types=entity_types,
                limit=limit,
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
            
            # Delegate to the main search method
            result = self.search_project_entities(
                search_term=search_term,
                container_name=container_name,
                entity_types=entity_types,
                limit=limit,
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
                        result_data["message"] = f"Found {component_count} components matching '{search_term}' using {search_type} search"
                    
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

    def search_project_domains(self, search_term: str, container_name: str,
                            limit: int = 10, semantic: bool = False) -> str:
        """
        Search specifically for domains within a project container.
        
        Args:
            search_term: The term to search for
            container_name: Name of the project container to search within
            limit: Maximum number of results to return
            semantic: Whether to perform semantic (embedding-based) search
            
        Returns:
            JSON string with the search results
        """
        try:
            # For domains, we set entity_types to 'Domain'
            entity_types = ["Domain"]
            
            # Delegate to the main search method
            result = self.search_project_entities(
                search_term=search_term,
                container_name=container_name,
                entity_types=entity_types,
                limit=limit,
                semantic=semantic
            )
            
            # Parse and modify the result to highlight that this is a domain search
            try:
                result_data = json.loads(result)
                if "data" in result_data:
                    result_data["data"]["domain_search"] = True
                    
                    # Update the message to specify domains
                    if "message" in result_data:
                        domain_count = result_data["data"].get("total_count", 0)
                        search_type = "semantic" if semantic else "text"
                        result_data["message"] = f"Found {domain_count} domains matching '{search_term}' using {search_type} search"
                    
                    # For each domain, get additional domain-specific information
                    if "entities" in result_data["data"] and result_data["data"]["entities"]:
                        for domain in result_data["data"]["entities"]:
                            domain_name = domain.get("name", "")
                            if domain_name:
                                # Get domain entities count
                                try:
                                    domain_entities_query = """
                                    MATCH (d:Entity {name: $domain_name, entityType: 'Domain'})
                                    MATCH (e:Entity)-[:PART_OF]->(d)
                                    RETURN count(e) as entity_count
                                    """
                                    
                                    entity_count_records = self.base_manager.safe_execute_read_query(
                                        domain_entities_query,
                                        {"domain_name": domain_name}
                                    )
                                    
                                    if entity_count_records and len(entity_count_records) > 0:
                                        domain["entity_count"] = entity_count_records[0]["entity_count"]
                                except Exception as e:
                                    self.logger.error(f"Error getting domain entity count: {str(e)}")
                                    domain["entity_count"] = 0
                    
                    return json.dumps(result_data, default=str)
                
                return result
            except json.JSONDecodeError:
                # If we can't parse the result, return it as is
                return result
            
        except Exception as e:
            self.logger.error(f"Error searching project domains: {str(e)}")
            return json.dumps(create_error_response(
                message=f"Failed to search project domains: {str(e)}",
                code="domain_search_error"
            ).model_dump(), default=str)
    
    def search_project_dependencies(self, search_term: str, container_name: str,
                                 dependency_types: Optional[List[str]] = None,
                                 component_name: Optional[str] = None,
                                 direction: str = "both",
                                 limit: int = 20) -> str:
        """
        Search for dependencies within a project container.
        
        Args:
            search_term: The term to search for
            container_name: Name of the project container to search within
            dependency_types: Optional list of dependency types to filter by (e.g., 'DEPENDS_ON', 'IMPORTS', 'USES')
            component_name: Optional component name to filter dependencies for
            direction: Direction of dependencies to search ('outgoing', 'incoming', or 'both')
            limit: Maximum number of results to return
            
        Returns:
            JSON string with the search results
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
                return json.dumps(create_error_response(
                    message=f"Project container '{container_name}' not found",
                    code="search_error"
                ).model_dump(), default=str)
            
            # Common dependency relationship types
            common_dependency_types = [
                "DEPENDS_ON", "IMPORTS", "USES", "CALLS", "REFERENCES", 
                "IMPLEMENTS", "EXTENDS", "INHERITS_FROM"
            ]
            
            # Filter dependency types
            if dependency_types and len(dependency_types) > 0:
                # Use specified types, ensure uppercase
                dependency_types = [t.upper() for t in dependency_types]
            else:
                # Default to common dependency types
                dependency_types = common_dependency_types
            
            # Build the Cypher query based on search parameters
            query_parts = []
            query_params = {"container_name": container_name, "search_term": f"(?i).*{search_term}.*", "limit": limit}
            
            # Match the container
            query_parts.append("MATCH (c:Entity {name: $container_name, entityType: 'ProjectContainer'})")
            
            # Handle different relationship directions
            if direction == "outgoing":
                if component_name:
                    # Search outgoing dependencies from a specific component
                    query_parts.append("MATCH (from:Entity {name: $component_name})-[:BELONGS_TO]->(c)")
                    query_parts.append("MATCH (from)-[r]->(to:Entity)-[:BELONGS_TO]->(c)")
                    query_params["component_name"] = component_name
                else:
                    # Search all outgoing dependencies
                    query_parts.append("MATCH (from:Entity)-[:BELONGS_TO]->(c)")
                    query_parts.append("MATCH (from)-[r]->(to:Entity)-[:BELONGS_TO]->(c)")
            elif direction == "incoming":
                if component_name:
                    # Search incoming dependencies to a specific component
                    query_parts.append("MATCH (to:Entity {name: $component_name})-[:BELONGS_TO]->(c)")
                    query_parts.append("MATCH (from:Entity)-[r]->(to)-[:BELONGS_TO]->(c)")
                    query_params["component_name"] = component_name
                else:
                    # Search all incoming dependencies
                    query_parts.append("MATCH (to:Entity)-[:BELONGS_TO]->(c)")
                    query_parts.append("MATCH (from:Entity)-[r]->(to)-[:BELONGS_TO]->(c)")
            else:  # "both" is the default
                if component_name:
                    # Search dependencies in both directions for a specific component
                    query_parts.append("MATCH (comp:Entity {name: $component_name})-[:BELONGS_TO]->(c)")
                    query_parts.append("MATCH (comp)-[r]-(other:Entity)-[:BELONGS_TO]->(c)")
                    query_parts.append("WITH comp, r, other")
                    query_parts.append("MATCH (from:Entity)-[rr]->(to:Entity)")
                    query_parts.append("WHERE (from = comp AND to = other) OR (from = other AND to = comp)")
                    query_parts.append("AND type(rr) = type(r)")
                    query_params["component_name"] = component_name
                else:
                    # Search all dependencies in both directions
                    query_parts.append("MATCH (from:Entity)-[:BELONGS_TO]->(c)")
                    query_parts.append("MATCH (to:Entity)-[:BELONGS_TO]->(c)")
                    query_parts.append("MATCH (from)-[r]->(to)")
            
            # Add relationship type filter
            query_parts.append("WHERE type(r) IN $dependency_types")
            query_params["dependency_types"] = dependency_types
            
            # Add search term filter
            query_parts.append("AND (from.name =~ $search_term OR to.name =~ $search_term OR type(r) =~ $search_term)")
            
            # Add return clause with ordering
            query_parts.append("RETURN from, to, type(r) as relation_type, r")
            query_parts.append("ORDER BY from.name, to.name")
            query_parts.append("LIMIT $limit")
            
            # Execute the query
            query = " ".join(query_parts)
            dependency_records = self.base_manager.safe_execute_read_query(
                query, 
                query_params
            )
            
            # Process results
            dependencies = []
            if dependency_records:
                for record in dependency_records:
                    from_entity = dict(record["from"].items())
                    to_entity = dict(record["to"].items())
                    relation_type = record["relation_type"]
                    relation_props = dict(record["r"].items()) if record["r"] else {}
                    
                    # Create a standardized dependency object
                    dependency = {
                        "from": {
                            "id": from_entity.get("id", ""),
                            "name": from_entity.get("name", ""),
                            "type": from_entity.get("entityType", from_entity.get("type", ""))
                        },
                        "to": {
                            "id": to_entity.get("id", ""),
                            "name": to_entity.get("name", ""),
                            "type": to_entity.get("entityType", to_entity.get("type", ""))
                        },
                        "relation": {
                            "type": relation_type,
                            "properties": relation_props
                        }
                    }
                    
                    dependencies.append(dependency)
            
            # Build the result
            result_data = {
                "dependencies": dependencies,
                "total_count": len(dependencies),
                "search_term": search_term,
                "container_name": container_name,
                "dependency_types": dependency_types,
                "direction": direction
            }
            
            if component_name:
                result_data["component_name"] = component_name
            
            # Return success response
            search_message = f"Found {len(dependencies)} dependencies matching '{search_term}' in project '{container_name}'"
            return json.dumps(create_success_response(
                message=search_message,
                data=result_data
            ).model_dump(), default=str)
            
        except Exception as e:
            self.logger.error(f"Error searching project dependencies: {str(e)}")
            return json.dumps(create_error_response(
                message=f"Failed to search project dependencies: {str(e)}",
                code="dependency_search_error"
            ).model_dump(), default=str)

    def search_domain_entities(self, search_term: str, container_name: str, 
                            domain_name: Optional[str] = None,
                            entity_types: Optional[List[str]] = None,
                            limit: int = 10, semantic: bool = False) -> str:
        """
        Search specifically for domain entities within a project container.
        
        Args:
            search_term: The term to search for
            container_name: Name of the project container to search within
            domain_name: Optional domain name to restrict search to a specific domain
            entity_types: Optional list of entity types to filter by (e.g., 'DECISION', 'FEATURE', 'REQUIREMENT')
            limit: Maximum number of results to return
            semantic: Whether to perform semantic (embedding-based) search
            
        Returns:
            JSON string with the search results
        """
        try:
            self.base_manager.ensure_initialized()
            
            # For domain entities, we use the allowed entity types from the model validation
            allowed_domain_entity_types = [
                'DECISION', 'FEATURE', 'REQUIREMENT', 'SPECIFICATION', 'CONSTRAINT', 'RISK'
            ]
            
            # Filter entity types if provided
            if entity_types and len(entity_types) > 0:
                # Ensure all specified entity types exist in our predefined list
                # This prevents searching for invalid entity types
                filtered_entity_types = [t.upper() for t in entity_types if t.upper() in allowed_domain_entity_types]
                if not filtered_entity_types:
                    # If none of the specified types are valid, use all domain entity types
                    entity_types = allowed_domain_entity_types
                else:
                    entity_types = filtered_entity_types
            else:
                # Default to all domain entity types
                entity_types = allowed_domain_entity_types
            
            # Build the Cypher query based on search parameters
            query_parts = []
            query_params = {
                "container_name": container_name,
                "search_term": f"(?i).*{search_term}.*",
                "entity_types": entity_types,
                "limit": limit
            }
            
            # Match the container
            query_parts.append("MATCH (c:Entity {name: $container_name, entityType: 'ProjectContainer'})")
            
            # Handle domain filtering
            if domain_name:
                query_parts.append("MATCH (d:Entity {name: $domain_name, entityType: 'Domain'})-[:PART_OF]->(c)")
                query_parts.append("MATCH (e:Entity)-[:BELONGS_TO]->(d)")
                query_params["domain_name"] = domain_name
            else:
                # If no domain specified, match any entities in the container
                query_parts.append("MATCH (d:Entity {entityType: 'Domain'})-[:PART_OF]->(c)")
                query_parts.append("MATCH (e:Entity)-[:BELONGS_TO]->(d)")
            
            # Add entity type and search term filters
            query_parts.append("WHERE e.entityType IN $entity_types")
            query_parts.append("AND (e.name =~ $search_term OR e.description =~ $search_term OR e.content =~ $search_term)")
            
            # Add return clause with ordering
            query_parts.append("RETURN e, d.name as domain_name")
            query_parts.append("ORDER BY e.name")
            query_parts.append("LIMIT $limit")
            
            # Join query parts
            query = " ".join(query_parts)
            
            # Execute the query if not using semantic search
            if not semantic:
                domain_entity_records = self.base_manager.safe_execute_read_query(
                    query, 
                    query_params
                )
                
                # Process results
                entities = []
                if domain_entity_records:
                    for record in domain_entity_records:
                        entity = dict(record["e"].items())
                        entity["domain"] = record["domain_name"]
                        entities.append(entity)
                
                result_data = {
                    "entities": entities,
                    "total_count": len(entities),
                    "search_term": search_term,
                    "container_name": container_name,
                    "domain_entity_search": True,
                    "entity_types": entity_types
                }
                
                if domain_name:
                    result_data["domain_name"] = domain_name
                
                search_message = f"Found {len(entities)} domain entities matching '{search_term}' in {domain_name + ' domain of ' if domain_name else ''}project '{container_name}'"
                return json.dumps(create_success_response(
                    message=search_message,
                    data=result_data
                ).model_dump(), default=str)
            else:
                # For semantic search, we need to use the search_manager with entity_types filter
                # and then post-filter the results to include only domain entities
                result = self.search_manager.search_entities(
                    search_term=search_term,
                    limit=limit * 2,  # Fetch more results as we'll filter some out
                    entity_types=entity_types,
                    semantic=True
                )
                
                # Parse the result
                result_data = json.loads(result)
                
                # Filter results to only include entities in the specified container and domain
                if "entities" in result_data:
                    filtered_entities = []
                    for entity in result_data["entities"]:
                        # Additional query to check container and domain membership
                        membership_query = """
                        MATCH (e:Entity {id: $entity_id})
                        MATCH (c:Entity {name: $container_name, entityType: 'ProjectContainer'})
                        """
                        
                        membership_params = {
                            "entity_id": entity.get("id", ""),
                            "container_name": container_name
                        }
                        
                        if domain_name:
                            membership_query += """
                            MATCH (d:Entity {name: $domain_name, entityType: 'Domain'})-[:PART_OF]->(c)
                            RETURN EXISTS((e)-[:BELONGS_TO]->(d)) as in_domain, d.name as domain_name
                            """
                            membership_params["domain_name"] = domain_name
                        else:
                            membership_query += """
                            MATCH (d:Entity {entityType: 'Domain'})-[:PART_OF]->(c)
                            MATCH (e)-[:BELONGS_TO]->(d)
                            RETURN true as in_domain, d.name as domain_name
                            """
                        
                        membership_records = self.base_manager.safe_execute_read_query(
                            membership_query,
                            membership_params
                        )
                        
                        if membership_records and len(membership_records) > 0 and membership_records[0]["in_domain"]:
                            entity["domain"] = membership_records[0]["domain_name"]
                            filtered_entities.append(entity)
                            
                            # Limit to the requested number of results
                            if len(filtered_entities) >= limit:
                                break
                    
                    result_data["entities"] = filtered_entities
                    result_data["total_count"] = len(filtered_entities)
                    result_data["search_term"] = search_term
                    result_data["container_name"] = container_name
                    result_data["domain_entity_search"] = True
                    result_data["entity_types"] = entity_types
                    result_data["search_type"] = "semantic"
                    
                    if domain_name:
                        result_data["domain_name"] = domain_name
                    
                    search_message = f"Found {len(filtered_entities)} domain entities using semantic search for '{search_term}' in {domain_name + ' domain of ' if domain_name else ''}project '{container_name}'"
                    return json.dumps(create_success_response(
                        message=search_message,
                        data=result_data
                    ).model_dump(), default=str)
                
                # If there's an error in the result, pass it through
                if "error" in result_data:
                    return json.dumps(create_error_response(
                        message=result_data["error"],
                        code="domain_entity_search_error"
                    ).model_dump(), default=str)
            
            # Default error case
            return json.dumps(create_error_response(
                message="Unexpected error during domain entity search",
                code="domain_entity_search_error"
            ).model_dump(), default=str)
            
        except Exception as e:
            self.logger.error(f"Error searching domain entities: {str(e)}")
            return json.dumps(create_error_response(
                message=f"Failed to search domain entities: {str(e)}",
                code="domain_entity_search_error"
            ).model_dump(), default=str)
    
    def search_entity_paths(self, from_entity: str, to_entity: str, 
                         container_name: str, domain_name: Optional[str] = None,
                         relationship_types: Optional[List[str]] = None,
                         max_depth: int = 5, limit: int = 10) -> str:
        """
        Search for paths between entities within a project container.
        This is an enhanced version of find_project_dependency_path that works with any entity types,
        not just components.
        
        Args:
            from_entity: Name of the source entity
            to_entity: Name of the target entity
            container_name: Name of the project container
            domain_name: Optional domain name to restrict search to a specific domain
            relationship_types: Optional list of relationship types to consider in the path
            max_depth: Maximum path depth to search (1-10)
            limit: Maximum number of paths to return
            
        Returns:
            JSON string with the found paths
        """
        try:
            self.base_manager.ensure_initialized()
            
            # Validate max_depth to prevent excessive resource usage
            max_depth = min(max(1, max_depth), 10)
            
            # Default relationship types if not specified
            if not relationship_types or len(relationship_types) == 0:
                relationship_types = [
                    'DEPENDS_ON', 'IMPORTS', 'USES', 'CALLS', 'REFERENCES', 
                    'IMPLEMENTS', 'EXTENDS', 'CONTAINS', 'RELATED_TO', 'SUPERSEDES',
                    'ALTERNATIVE_TO', 'INFLUENCES', 'LEADS_TO', 'DERIVES_FROM', 'CONSTRAINS'
                ]
            else:
                # Uppercase all relationship types
                relationship_types = [rel_type.upper() for rel_type in relationship_types]
            
            # Build query parameters
            query_params = {
                "container_name": container_name,
                "from_entity": from_entity,
                "to_entity": to_entity,
                "max_depth": max_depth,
                "limit": limit
            }
            
            # Build query parts
            query_parts = []
            
            # Match the container
            query_parts.append("MATCH (c:Entity {name: $container_name, entityType: 'ProjectContainer'})")
            
            # Match source and target entities
            if domain_name:
                # If domain is specified, match entities in that domain
                query_parts.append("MATCH (d:Entity {name: $domain_name, entityType: 'Domain'})-[:PART_OF]->(c)")
                query_parts.append("MATCH (from:Entity {name: $from_entity})-[:BELONGS_TO]->(d)")
                query_parts.append("MATCH (to:Entity {name: $to_entity})-[:BELONGS_TO]->(d)")
                query_params["domain_name"] = domain_name
            else:
                # If no domain specified, match any entities in the container
                query_parts.append("MATCH (from:Entity {name: $from_entity})")
                query_parts.append("MATCH (to:Entity {name: $to_entity})")
                query_parts.append("WHERE (from)-[:BELONGS_TO|PART_OF*1..2]->(c) AND (to)-[:BELONGS_TO|PART_OF*1..2]->(c)")
            
            # Build relationship pattern for path
            rel_pattern = "|".join(relationship_types)
            
            # Match the path using specified relationship types
            query_parts.append(f"MATCH path = (from)-[r:{rel_pattern}*1..{max_depth}]->(to)")
            
            # Return the path with additional information
            query_parts.append("RETURN path, length(path) as path_length, [rel in relationships(path) | type(rel)] as rel_types")
            query_parts.append("ORDER BY path_length")
            query_parts.append("LIMIT $limit")
            
            # Join query parts
            query = " ".join(query_parts)
            
            # Execute query
            path_records = self.base_manager.safe_execute_read_query(
                query,
                query_params
            )
            
            # Process paths
            paths = []
            if path_records:
                for record in path_records:
                    path = record["path"]
                    path_length = record["path_length"]
                    rel_types = record["rel_types"]
                    
                    # Extract nodes and relationships
                    nodes = []
                    for node in path.nodes:
                        node_dict = dict(node.items())
                        # Add domain info if available
                        domain_info = self._get_entity_domain(node_dict.get("id", ""))
                        if domain_info:
                            node_dict["domain"] = domain_info
                        nodes.append(node_dict)
                    
                    relationships = []
                    for rel in path.relationships:
                        relationships.append({
                            "type": rel.type,
                            "properties": dict(rel.items())
                        })
                    
                    path_data = {
                        "length": path_length,
                        "relationship_types": rel_types,
                        "nodes": nodes,
                        "relationships": relationships
                    }
                    
                    paths.append(path_data)
            
            result_data = {
                "from_entity": from_entity,
                "to_entity": to_entity,
                "container_name": container_name,
                "paths_found": len(paths),
                "max_depth": max_depth,
                "relationship_types": relationship_types,
                "paths": paths
            }
            
            if domain_name:
                result_data["domain_name"] = domain_name
            
            # Return success response
            if len(paths) > 0:
                search_message = f"Found {len(paths)} paths from '{from_entity}' to '{to_entity}' in {domain_name + ' domain of ' if domain_name else ''}project '{container_name}'"
            else:
                search_message = f"No paths found from '{from_entity}' to '{to_entity}' in {domain_name + ' domain of ' if domain_name else ''}project '{container_name}'"
                
            return json.dumps(create_success_response(
                message=search_message,
                data=result_data
            ).model_dump(), default=str)
            
        except Exception as e:
            self.logger.error(f"Error searching entity paths: {str(e)}")
            return json.dumps(create_error_response(
                message=f"Failed to search entity paths: {str(e)}",
                code="path_search_error"
            ).model_dump(), default=str)
    
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
