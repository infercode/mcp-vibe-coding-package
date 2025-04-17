from typing import Any, Dict, List, Optional, Union
import time
import json
from datetime import datetime

from src.utils import dict_to_json, generate_id
from src.graph_memory.base_manager import BaseManager
from src.graph_memory.entity_manager import EntityManager
from src.models.project_memory import (
    ErrorResponse,
    ErrorDetail,
    SuccessResponse,
    Metadata,
    DomainEntityCreate,
    RelationshipCreate
)

class DomainManager:
    """
    Manager for project domain entities.
    Handles the creation and management of domain-level knowledge.
    """
    
    def __init__(self, base_manager: BaseManager):
        """
        Initialize the domain manager.
        
        Args:
            base_manager: Base manager for graph operations
        """
        self.base_manager = base_manager
        self.logger = base_manager.logger
        self.entity_manager = EntityManager(base_manager)
    
    def create_domain(self, domain: DomainEntityCreate) -> str:
        """
        Create a new domain entity within a project container.
        
        Args:
            domain: DomainEntityCreate Pydantic model with domain details
            
        Returns:
            JSON string with the created domain
        """
        try:
            self.base_manager.ensure_initialized()
            
            name = domain.name
            container_name = domain.project_id
            description = domain.description
            
            # Check if project container exists
            container_query = """
            MATCH (c:Entity {name: $container_name, entityType: 'ProjectContainer'})
            RETURN c
            """
            
            # Use safe_execute_read_query for validation (read-only operation)
            container_records = self.base_manager.safe_execute_read_query(
                container_query,
                {"container_name": container_name}
            )
            
            if not container_records or len(container_records) == 0:
                error = ErrorDetail(
                    code="CONTAINER_NOT_FOUND",
                    message=f"Project container '{container_name}' not found",
                    details=None
                )
                return dict_to_json(ErrorResponse(
                    status="error",
                    timestamp=datetime.now(),
                    error=error
                ).model_dump())
            
            # Check if domain with this name already exists in the container
            domain_check_query = """
            MATCH (c:Entity {name: $container_name, entityType: 'ProjectContainer'})
            MATCH (d:Entity {name: $name, entityType: 'Domain'})-[:PART_OF]->(c)
            RETURN d
            """
            
            # Use safe_execute_read_query for validation (read-only operation)
            domain_records = self.base_manager.safe_execute_read_query(
                domain_check_query,
                {"container_name": container_name, "name": name}
            )
            
            if domain_records and len(domain_records) > 0:
                error = ErrorDetail(
                    code="DOMAIN_EXISTS",
                    message=f"Domain '{name}' already exists in container '{container_name}'",
                    details=None
                )
                return dict_to_json(ErrorResponse(
                    status="error",
                    timestamp=datetime.now(),
                    error=error
                ).model_dump())
            
            # Generate domain ID
            domain_id = generate_id("dom")
            timestamp = time.time()
            
            # Prepare domain entity
            domain_entity = {
                "id": domain_id,
                "name": name,
                "entityType": "Domain",
                "domain": "project",
                "created": timestamp,
                "lastUpdated": timestamp
            }
            
            if description:
                domain_entity["description"] = description
                
            # Handle metadata if provided
            if domain.metadata:
                metadata_dict = domain.metadata.model_dump(exclude_none=True)
                for key, value in metadata_dict.items():
                    domain_entity[f"metadata_{key}"] = value
            
            # Handle tags if provided
            if domain.tags:
                domain_entity["tags"] = domain.tags
                
            # Create domain entity
            create_query = """
            CREATE (d:Entity $properties)
            RETURN d
            """
            
            # Use safe_execute_write_query for validation (write operation)
            create_records = self.base_manager.safe_execute_write_query(
                create_query,
                {"properties": domain_entity}
            )
            
            if not create_records or len(create_records) == 0:
                error = ErrorDetail(
                    code="DOMAIN_CREATION_FAILED",
                    message="Failed to create domain entity",
                    details=None
                )
                return dict_to_json(ErrorResponse(
                    status="error",
                    timestamp=datetime.now(),
                    error=error
                ).model_dump())
            
            # Add domain to project container
            add_to_container_query = """
            MATCH (c:Entity {name: $container_name, entityType: 'ProjectContainer'})
            MATCH (d:Entity {id: $domain_id})
            CREATE (d)-[:PART_OF {created: $timestamp}]->(c)
            RETURN d
            """
            
            # Use safe_execute_write_query for validation (write operation)
            add_records = self.base_manager.safe_execute_write_query(
                add_to_container_query,
                {"container_name": container_name, "domain_id": domain_id, "timestamp": timestamp}
            )
            
            if not add_records or len(add_records) == 0:
                # Attempt to clean up the created entity
                self.base_manager.safe_execute_write_query(
                    "MATCH (d:Entity {id: $domain_id}) DELETE d",
                    {"domain_id": domain_id}
                )
                
                error = ErrorDetail(
                    code="CONTAINER_LINKING_FAILED",
                    message=f"Failed to add domain '{name}' to container '{container_name}'",
                    details=None
                )
                return dict_to_json(ErrorResponse(
                    status="error",
                    timestamp=datetime.now(),
                    error=error
                ).model_dump())
            
            # Get created domain
            domain_data = dict(add_records[0]["d"].items())
            
            response = {
                "status": "success",
                "timestamp": datetime.now().isoformat(),
                "message": f"Domain '{name}' created successfully in container '{container_name}'",
                "domain_id": domain_id,
                "domain": domain_data
            }
            
            return dict_to_json(response)
                
        except Exception as e:
            error_msg = f"Error creating domain: {str(e)}"
            self.logger.error(error_msg)
            error = ErrorDetail(
                code="INTERNAL_ERROR",
                message=error_msg,
                details=None
            )
            return dict_to_json(ErrorResponse(
                status="error",
                timestamp=datetime.now(),
                error=error
            ).model_dump())
    
    def get_domain(self, name: str, container_name: str) -> str:
        """
        Retrieve a domain entity from a project container.
        
        Args:
            name: Name of the domain
            container_name: Name of the project container
            
        Returns:
            JSON string with the domain details
        """
        try:
            self.base_manager.ensure_initialized()
            
            # Get domain query
            query = """
            MATCH (c:Entity {name: $container_name, entityType: 'ProjectContainer'})
            MATCH (d:Entity {name: $name, entityType: 'Domain'})-[:PART_OF]->(c)
            OPTIONAL MATCH (d)<-[:BELONGS_TO]-(component:Entity)
            RETURN d, count(component) as component_count
            """
            
            # Use safe_execute_read_query for validation (read-only operation)
            records = self.base_manager.safe_execute_read_query(
                query,
                {"container_name": container_name, "name": name}
            )
            
            if not records or len(records) == 0:
                error = ErrorDetail(
                    code="DOMAIN_NOT_FOUND",
                    message=f"Domain '{name}' not found in container '{container_name}'",
                    details=None
                )
                return dict_to_json(ErrorResponse(
                    status="error",
                    timestamp=datetime.now(),
                    error=error
                ).model_dump())
            
            # Extract domain info
            record = records[0]
            domain = dict(record["d"].items())
            component_count = record["component_count"]
            
            # Get component counts by type
            component_query = """
            MATCH (c:Entity {name: $container_name, entityType: 'ProjectContainer'})
            MATCH (d:Entity {name: $name, entityType: 'Domain'})-[:PART_OF]->(c)
            MATCH (d)<-[:BELONGS_TO]-(component:Entity)
            RETURN component.entityType as type, count(component) as count
            """
            
            # Use safe_execute_read_query for validation (read-only operation)
            component_records = self.base_manager.safe_execute_read_query(
                component_query,
                {"container_name": container_name, "name": name}
            )
            
            # Process component counts
            component_types = {}
            if component_records:
                for record in component_records:
                    entity_type = record["type"]
                    count = record["count"]
                    if entity_type:
                        component_types[entity_type] = count
            
            # Add component statistics
            domain["component_count"] = component_count
            domain["component_types"] = component_types
            
            response = {
                "status": "success",
                "timestamp": datetime.now().isoformat(),
                "domain_id": domain.get("id"),
                "domain": domain
            }
            
            return dict_to_json(response)
                
        except Exception as e:
            error_msg = f"Error retrieving domain: {str(e)}"
            self.logger.error(error_msg)
            error = ErrorDetail(
                code="INTERNAL_ERROR",
                message=error_msg,
                details=None
            )
            return dict_to_json(ErrorResponse(
                status="error",
                timestamp=datetime.now(),
                error=error
            ).model_dump())
    
    def update_domain(self, domain_id: str, container_name: str, updates: Dict[str, Any]) -> str:
        """
        Update a domain entity's properties.
        
        Args:
            domain_id: ID of the domain to update
            container_name: Name of the project container
            updates: Dictionary of properties to update
            
        Returns:
            JSON string with the updated domain
        """
        try:
            self.base_manager.ensure_initialized()
            
            # Check if domain exists in container
            domain_query = """
            MATCH (c:Entity {name: $container_name, entityType: 'ProjectContainer'})
            MATCH (d:Entity {id: $domain_id, entityType: 'Domain'})-[:PART_OF]->(c)
            RETURN d
            """
            
            # Use safe_execute_read_query for validation (read-only operation)
            domain_records = self.base_manager.safe_execute_read_query(
                domain_query,
                {"container_name": container_name, "domain_id": domain_id}
            )
            
            if not domain_records or len(domain_records) == 0:
                error = ErrorDetail(
                    code="DOMAIN_NOT_FOUND",
                    message=f"Domain with ID '{domain_id}' not found in container '{container_name}'",
                    details=None
                )
                return dict_to_json(ErrorResponse(
                    status="error",
                    timestamp=datetime.now(),
                    error=error
                ).model_dump())
            
            # Get current domain data
            domain = dict(domain_records[0]["d"].items())
            domain_name = domain.get("name")
            
            # Validate updates - prevent changing core properties
            protected_fields = ["id", "name", "entityType", "domain", "created"]
            invalid_updates = [field for field in updates if field in protected_fields]
            
            if invalid_updates:
                error = ErrorDetail(
                    code="INVALID_UPDATE",
                    message=f"Cannot update protected fields: {', '.join(invalid_updates)}",
                    details=None
                )
                return dict_to_json(ErrorResponse(
                    status="error",
                    timestamp=datetime.now(),
                    error=error
                ).model_dump())
            
            # Add lastUpdated timestamp
            updates["lastUpdated"] = time.time()
            
            # Handle metadata updates
            if "metadata" in updates:
                metadata = updates.pop("metadata")
                if metadata:
                    metadata_dict = metadata
                    if hasattr(metadata, "model_dump"):
                        metadata_dict = metadata.model_dump(exclude_none=True)
                    for key, value in metadata_dict.items():
                        updates[f"metadata_{key}"] = value
            
            # Prepare update parts
            set_parts = []
            for key, value in updates.items():
                set_parts.append(f"d.{key} = ${key}")
            
            # If no updates provided, return success with current domain
            if not set_parts:
                return dict_to_json({
                    "status": "success",
                    "timestamp": datetime.now().isoformat(),
                    "message": "No updates provided for domain",
                    "domain_id": domain_id,
                    "domain": domain
                })
            
            # Build update query
            update_query = f"""
            MATCH (c:Entity {{name: $container_name, entityType: 'ProjectContainer'}})
            MATCH (d:Entity {{id: $domain_id, entityType: 'Domain'}})-[:PART_OF]->(c)
            SET {', '.join(set_parts)}
            RETURN d
            """
            
            # Add domain_id and container_name to updates for the query
            params = {"domain_id": domain_id, "container_name": container_name, **updates}
            
            # Use safe_execute_write_query for validation (write operation)
            update_records = self.base_manager.safe_execute_write_query(
                update_query,
                params
            )
            
            if not update_records or len(update_records) == 0:
                error = ErrorDetail(
                    code="UPDATE_FAILED",
                    message=f"Failed to update domain '{domain_name}'",
                    details=None
                )
                return dict_to_json(ErrorResponse(
                    status="error",
                    timestamp=datetime.now(),
                    error=error
                ).model_dump())
            
            # Return updated domain
            updated_domain = dict(update_records[0]["d"].items())
            
            response = {
                "status": "success",
                "timestamp": datetime.now().isoformat(),
                "message": f"Domain '{domain_name}' updated successfully",
                "domain_id": domain_id,
                "domain": updated_domain
            }
            
            return dict_to_json(response)
                
        except Exception as e:
            error_msg = f"Error updating domain: {str(e)}"
            self.logger.error(error_msg)
            error = ErrorDetail(
                code="INTERNAL_ERROR",
                message=error_msg,
                details=None
            )
            return dict_to_json(ErrorResponse(
                status="error",
                timestamp=datetime.now(),
                error=error
            ).model_dump())
    
    def delete_domain(self, domain_id: str, container_name: str, delete_components: bool = False) -> str:
        """
        Delete a domain entity from a project container.
        
        Args:
            domain_id: ID of the domain to delete
            container_name: Name of the project container
            delete_components: If True, delete all components belonging to the domain
            
        Returns:
            JSON string with the deletion result
        """
        try:
            self.base_manager.ensure_initialized()
            
            # Check if domain exists in container
            domain_query = """
            MATCH (c:Entity {name: $container_name, entityType: 'ProjectContainer'})
            MATCH (d:Entity {id: $domain_id, entityType: 'Domain'})-[:PART_OF]->(c)
            RETURN d
            """
            
            # Use safe_execute_read_query for validation (read-only operation)
            domain_records = self.base_manager.safe_execute_read_query(
                domain_query,
                {"container_name": container_name, "domain_id": domain_id}
            )
            
            if not domain_records or len(domain_records) == 0:
                error = ErrorDetail(
                    code="DOMAIN_NOT_FOUND",
                    message=f"Domain with ID '{domain_id}' not found in container '{container_name}'",
                    details=None
                )
                return dict_to_json(ErrorResponse(
                    status="error",
                    timestamp=datetime.now(),
                    error=error
                ).model_dump())
            
            # Get domain name for the message
            domain_name = domain_records[0]["d"]["name"]
            
            if delete_components:
                # Delete all components belonging to the domain
                delete_components_query = """
                MATCH (c:Entity {name: $container_name, entityType: 'ProjectContainer'})
                MATCH (d:Entity {id: $domain_id})-[:PART_OF]->(c)
                MATCH (component:Entity)-[:BELONGS_TO]->(d)
                
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
                
                # Use safe_execute_write_query for validation (write operation)
                delete_records = self.base_manager.safe_execute_write_query(
                    delete_components_query,
                    {"container_name": container_name, "domain_id": domain_id}
                )
                
                deleted_count = 0
                if delete_records and len(delete_records) > 0:
                    deleted_count = delete_records[0]["deleted_count"]
                
                # Delete domain relationships
                delete_domain_rels_query = """
                MATCH (d:Entity {id: $domain_id})
                OPTIONAL MATCH (d)-[r]-()
                DELETE r
                """
                
                # Use safe_execute_write_query for validation (write operation)
                self.base_manager.safe_execute_write_query(
                    delete_domain_rels_query,
                    {"domain_id": domain_id}
                )
                
                # Delete domain
                delete_domain_query = """
                MATCH (d:Entity {id: $domain_id})
                DELETE d
                """
                
                # Use safe_execute_write_query for validation (write operation)
                self.base_manager.safe_execute_write_query(
                    delete_domain_query,
                    {"domain_id": domain_id}
                )
                
                return dict_to_json(SuccessResponse(
                    status="success",
                    timestamp=datetime.now(),
                    message=f"Domain '{domain_name}' and {deleted_count} components deleted successfully"
                ).model_dump())
            else:
                # Check if domain has components
                check_components_query = """
                MATCH (d:Entity {id: $domain_id})
                MATCH (component:Entity)-[:BELONGS_TO]->(d)
                RETURN count(component) as component_count
                """
                
                # Use safe_execute_read_query for validation (read-only operation)
                component_records = self.base_manager.safe_execute_read_query(
                    check_components_query,
                    {"domain_id": domain_id}
                )
                
                component_count = 0
                if component_records and len(component_records) > 0:
                    component_count = component_records[0]["component_count"]
                
                if component_count > 0:
                    error = ErrorDetail(
                        code="DOMAIN_HAS_COMPONENTS",
                        message=f"Cannot delete domain '{domain_name}' with {component_count} components. Set delete_components=True to delete the domain and its components.",
                        details=None
                    )
                    return dict_to_json(ErrorResponse(
                        status="error",
                        timestamp=datetime.now(),
                        error=error
                    ).model_dump())
                
                # Delete domain relationships
                delete_domain_rels_query = """
                MATCH (d:Entity {id: $domain_id})
                OPTIONAL MATCH (d)-[r]-()
                DELETE r
                """
                
                # Use safe_execute_write_query for validation (write operation)
                self.base_manager.safe_execute_write_query(
                    delete_domain_rels_query,
                    {"domain_id": domain_id}
                )
                
                # Delete domain
                delete_domain_query = """
                MATCH (d:Entity {id: $domain_id})
                DELETE d
                """
                
                # Use safe_execute_write_query for validation (write operation)
                self.base_manager.safe_execute_write_query(
                    delete_domain_query,
                    {"domain_id": domain_id}
                )
                
                return dict_to_json(SuccessResponse(
                    status="success",
                    timestamp=datetime.now(),
                    message=f"Domain '{domain_name}' deleted successfully"
                ).model_dump())
                
        except Exception as e:
            error_msg = f"Error deleting domain: {str(e)}"
            self.logger.error(error_msg)
            error = ErrorDetail(
                code="INTERNAL_ERROR",
                message=error_msg,
                details=None
            )
            return dict_to_json(ErrorResponse(
                status="error",
                timestamp=datetime.now(),
                error=error
            ).model_dump())
    
    def list_domains(self, container_name: str, sort_by: str = "name", limit: int = 100) -> str:
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
            self.base_manager.ensure_initialized()
            
            # Check if container exists
            container_query = """
            MATCH (c:Entity {name: $container_name, entityType: 'ProjectContainer'})
            RETURN c
            """
            
            # Use safe_execute_read_query for validation (read-only operation)
            container_records = self.base_manager.safe_execute_read_query(
                container_query,
                {"container_name": container_name}
            )
            
            if not container_records or len(container_records) == 0:
                error = ErrorDetail(
                    code="CONTAINER_NOT_FOUND",
                    message=f"Project container '{container_name}' not found",
                    details=None
                )
                return dict_to_json(ErrorResponse(
                    status="error",
                    timestamp=datetime.now(),
                    error=error
                ).model_dump())
            
            # Validate sort_by
            valid_sort_fields = ["name", "created", "lastUpdated"]
            if sort_by not in valid_sort_fields:
                sort_by = "name"
            
            # Build query
            query = f"""
            MATCH (c:Entity {{name: $container_name, entityType: 'ProjectContainer'}})
            MATCH (d:Entity {{entityType: 'Domain'}})-[:PART_OF]->(c)
            OPTIONAL MATCH (d)<-[:BELONGS_TO]-(component:Entity)
            WITH d, count(component) as component_count
            RETURN d, component_count
            ORDER BY d.{sort_by}
            LIMIT $limit
            """
            
            # Use safe_execute_read_query for validation (read-only operation)
            records = self.base_manager.safe_execute_read_query(
                query,
                {"container_name": container_name, "limit": limit}
            )
            
            # Process results
            domains = []
            if records:
                for record in records:
                    domain = dict(record["d"].items())
                    domain["component_count"] = record["component_count"]
                    domains.append(domain)
            
            response = {
                "status": "success",
                "timestamp": datetime.now().isoformat(),
                "message": f"Found {len(domains)} domains in container '{container_name}'",
                "container": container_name,
                "domains": domains,
                "count": len(domains)
            }
            
            return dict_to_json(response)
                
        except Exception as e:
            error_msg = f"Error listing domains: {str(e)}"
            self.logger.error(error_msg)
            error = ErrorDetail(
                code="INTERNAL_ERROR",
                message=error_msg,
                details=None
            )
            return dict_to_json(ErrorResponse(
                status="error",
                timestamp=datetime.now(),
                error=error
            ).model_dump())
    
    def add_entity_to_domain(self, domain_id: str, container_name: str, 
                          entity_id: str) -> str:
        """
        Add an entity to a domain.
        
        Args:
            domain_id: ID of the domain
            container_name: Name of the project container
            entity_id: ID of the entity to add
            
        Returns:
            JSON string with the result
        """
        try:
            self.base_manager.ensure_initialized()
            
            # Check if domain exists in container
            domain_query = """
            MATCH (c:Entity {name: $container_name, entityType: 'ProjectContainer'})
            MATCH (d:Entity {id: $domain_id, entityType: 'Domain'})-[:PART_OF]->(c)
            RETURN d.name as domain_name
            """
            
            # Use safe_execute_read_query for validation (read-only operation)
            domain_records = self.base_manager.safe_execute_read_query(
                domain_query,
                {"container_name": container_name, "domain_id": domain_id}
            )
            
            if not domain_records or len(domain_records) == 0:
                error = ErrorDetail(
                    code="DOMAIN_NOT_FOUND",
                    message=f"Domain with ID '{domain_id}' not found in container '{container_name}'",
                    details=None
                )
                return dict_to_json(ErrorResponse(
                    status="error",
                    timestamp=datetime.now(),
                    error=error
                ).model_dump())
            
            domain_name = domain_records[0]["domain_name"]
            
            # Check if entity exists
            entity_query = """
            MATCH (e:Entity {id: $entity_id})
            RETURN e.name as entity_name
            """
            
            # Use safe_execute_read_query for validation (read-only operation)
            entity_records = self.base_manager.safe_execute_read_query(
                entity_query,
                {"entity_id": entity_id}
            )
            
            if not entity_records or len(entity_records) == 0:
                error = ErrorDetail(
                    code="ENTITY_NOT_FOUND",
                    message=f"Entity with ID '{entity_id}' not found",
                    details=None
                )
                return dict_to_json(ErrorResponse(
                    status="error",
                    timestamp=datetime.now(),
                    error=error
                ).model_dump())
            
            entity_name = entity_records[0]["entity_name"]
            
            # Check if entity is already in the domain
            check_query = """
            MATCH (c:Entity {name: $container_name, entityType: 'ProjectContainer'})
            MATCH (d:Entity {id: $domain_id, entityType: 'Domain'})-[:PART_OF]->(c)
            MATCH (e:Entity {id: $entity_id})
            MATCH (e)-[:BELONGS_TO]->(d)
            RETURN e
            """
            
            # Use safe_execute_read_query for validation (read-only operation)
            check_records = self.base_manager.safe_execute_read_query(
                check_query,
                {"container_name": container_name, "domain_id": domain_id, "entity_id": entity_id}
            )
            
            if check_records and len(check_records) > 0:
                return dict_to_json(SuccessResponse(
                    status="success",
                    timestamp=datetime.now(),
                    message=f"Entity '{entity_name}' is already in domain '{domain_name}'"
                ).model_dump())
            
            # Check if entity is in the project container
            container_check_query = """
            MATCH (c:Entity {name: $container_name, entityType: 'ProjectContainer'})
            MATCH (e:Entity {id: $entity_id})
            OPTIONAL MATCH (e)-[:PART_OF]->(c)
            RETURN exists((e)-[:PART_OF]->(c)) as in_container
            """
            
            # Use safe_execute_read_query for validation (read-only operation)
            container_check_records = self.base_manager.safe_execute_read_query(
                container_check_query,
                {"container_name": container_name, "entity_id": entity_id}
            )
            
            in_container = False
            if container_check_records and len(container_check_records) > 0:
                in_container = container_check_records[0]["in_container"]
            
            # If entity is not already in container, add it
            if not in_container:
                add_to_container_query = """
                MATCH (c:Entity {name: $container_name, entityType: 'ProjectContainer'})
                MATCH (e:Entity {id: $entity_id})
                CREATE (e)-[:PART_OF {created: $timestamp}]->(c)
                """
                
                # Use safe_execute_write_query for validation (write operation)
                self.base_manager.safe_execute_write_query(
                    add_to_container_query,
                    {"container_name": container_name, "entity_id": entity_id, "timestamp": time.time()}
                )
            
            # Add entity to domain
            add_query = """
            MATCH (c:Entity {name: $container_name, entityType: 'ProjectContainer'})
            MATCH (d:Entity {id: $domain_id, entityType: 'Domain'})-[:PART_OF]->(c)
            MATCH (e:Entity {id: $entity_id})
            CREATE (e)-[:BELONGS_TO {created: $timestamp}]->(d)
            RETURN e
            """
            
            # Use safe_execute_write_query for validation (write operation)
            add_records = self.base_manager.safe_execute_write_query(
                add_query,
                {
                    "container_name": container_name, 
                    "domain_id": domain_id, 
                    "entity_id": entity_id, 
                    "timestamp": time.time()
                }
            )
            
            if not add_records or len(add_records) == 0:
                error = ErrorDetail(
                    code="ADD_ENTITY_FAILED",
                    message=f"Failed to add entity '{entity_name}' to domain '{domain_name}'",
                    details=None
                )
                return dict_to_json(ErrorResponse(
                    status="error",
                    timestamp=datetime.now(),
                    error=error
                ).model_dump())
            
            return dict_to_json(SuccessResponse(
                status="success",
                timestamp=datetime.now(),
                message=f"Entity '{entity_name}' added to domain '{domain_name}'"
            ).model_dump())
                
        except Exception as e:
            error_msg = f"Error adding entity to domain: {str(e)}"
            self.logger.error(error_msg)
            error = ErrorDetail(
                code="INTERNAL_ERROR",
                message=error_msg,
                details=None
            )
            return dict_to_json(ErrorResponse(
                status="error",
                timestamp=datetime.now(),
                error=error
            ).model_dump())
    
    def remove_entity_from_domain(self, domain_id: str, container_name: str, 
                                entity_id: str) -> str:
        """
        Remove an entity from a domain.
        
        Args:
            domain_id: ID of the domain
            container_name: Name of the project container
            entity_id: ID of the entity to remove
            
        Returns:
            JSON string with the result
        """
        try:
            self.base_manager.ensure_initialized()
            
            # Get domain and entity names for better error messages
            names_query = """
            MATCH (c:Entity {name: $container_name, entityType: 'ProjectContainer'})
            MATCH (d:Entity {id: $domain_id, entityType: 'Domain'})-[:PART_OF]->(c)
            MATCH (e:Entity {id: $entity_id})
            RETURN d.name as domain_name, e.name as entity_name
            """
            
            # Use safe_execute_read_query for validation (read-only operation)
            names_records = self.base_manager.safe_execute_read_query(
                names_query,
                {"container_name": container_name, "domain_id": domain_id, "entity_id": entity_id}
            )
            
            if not names_records or len(names_records) == 0:
                error = ErrorDetail(
                    code="ENTITY_OR_DOMAIN_NOT_FOUND",
                    message=f"Domain with ID '{domain_id}' or entity with ID '{entity_id}' not found in container '{container_name}'",
                    details=None
                )
                return dict_to_json(ErrorResponse(
                    status="error",
                    timestamp=datetime.now(),
                    error=error
                ).model_dump())
                
            domain_name = names_records[0]["domain_name"]
            entity_name = names_records[0]["entity_name"]
            
            # Check if the relationship exists
            check_query = """
            MATCH (c:Entity {name: $container_name, entityType: 'ProjectContainer'})
            MATCH (d:Entity {id: $domain_id, entityType: 'Domain'})-[:PART_OF]->(c)
            MATCH (e:Entity {id: $entity_id})
            MATCH (e)-[r:BELONGS_TO]->(d)
            RETURN r
            """
            
            # Use safe_execute_read_query for validation (read-only operation)
            check_records = self.base_manager.safe_execute_read_query(
                check_query,
                {"container_name": container_name, "domain_id": domain_id, "entity_id": entity_id}
            )
            
            if not check_records or len(check_records) == 0:
                error = ErrorDetail(
                    code="ENTITY_NOT_IN_DOMAIN",
                    message=f"Entity '{entity_name}' is not in domain '{domain_name}'",
                    details=None
                )
                return dict_to_json(ErrorResponse(
                    status="error",
                    timestamp=datetime.now(),
                    error=error
                ).model_dump())
            
            # Remove entity from domain
            remove_query = """
            MATCH (c:Entity {name: $container_name, entityType: 'ProjectContainer'})
            MATCH (d:Entity {id: $domain_id, entityType: 'Domain'})-[:PART_OF]->(c)
            MATCH (e:Entity {id: $entity_id})
            MATCH (e)-[r:BELONGS_TO]->(d)
            DELETE r
            """
            
            # Use safe_execute_write_query for validation (write operation)
            self.base_manager.safe_execute_write_query(
                remove_query,
                {"container_name": container_name, "domain_id": domain_id, "entity_id": entity_id}
            )
            
            return dict_to_json(SuccessResponse(
                status="success",
                timestamp=datetime.now(),
                message=f"Entity '{entity_name}' removed from domain '{domain_name}'"
            ).model_dump())
                
        except Exception as e:
            error_msg = f"Error removing entity from domain: {str(e)}"
            self.logger.error(error_msg)
            error = ErrorDetail(
                code="INTERNAL_ERROR",
                message=error_msg,
                details=None
            )
            return dict_to_json(ErrorResponse(
                status="error",
                timestamp=datetime.now(),
                error=error
            ).model_dump())
    
    def get_domain_entities(self, domain_id: str, container_name: str, 
                         entity_type: Optional[str] = None) -> str:
        """
        Get all entities in a domain.
        
        Args:
            domain_id: ID of the domain
            container_name: Name of the project container
            entity_type: Optional entity type to filter by
            
        Returns:
            JSON string with the entities
        """
        try:
            self.base_manager.ensure_initialized()
            
            # Check if domain exists
            domain_query = """
            MATCH (c:Entity {name: $container_name, entityType: 'ProjectContainer'})
            MATCH (d:Entity {id: $domain_id, entityType: 'Domain'})-[:PART_OF]->(c)
            RETURN d.name as domain_name
            """
            
            # Use safe_execute_read_query for validation (read-only operation)
            domain_records = self.base_manager.safe_execute_read_query(
                domain_query,
                {"container_name": container_name, "domain_id": domain_id}
            )
            
            if not domain_records or len(domain_records) == 0:
                error = ErrorDetail(
                    code="DOMAIN_NOT_FOUND",
                    message=f"Domain with ID '{domain_id}' not found in container '{container_name}'",
                    details=None
                )
                return dict_to_json(ErrorResponse(
                    status="error",
                    timestamp=datetime.now(),
                    error=error
                ).model_dump())
            
            domain_name = domain_records[0]["domain_name"]
            
            # Build query based on entity_type
            if entity_type:
                query = """
                MATCH (c:Entity {name: $container_name, entityType: 'ProjectContainer'})
                MATCH (d:Entity {id: $domain_id, entityType: 'Domain'})-[:PART_OF]->(c)
                MATCH (e:Entity)-[:BELONGS_TO]->(d)
                WHERE e.entityType = $entity_type
                RETURN e
                ORDER BY e.name
                """
                
                params = {
                    "container_name": container_name, 
                    "domain_id": domain_id,
                    "entity_type": entity_type
                }
            else:
                query = """
                MATCH (c:Entity {name: $container_name, entityType: 'ProjectContainer'})
                MATCH (d:Entity {id: $domain_id, entityType: 'Domain'})-[:PART_OF]->(c)
                MATCH (e:Entity)-[:BELONGS_TO]->(d)
                RETURN e
                ORDER BY e.name
                """
                
                params = {"container_name": container_name, "domain_id": domain_id}
            
            # Use safe_execute_read_query for validation (read-only operation)
            records = self.base_manager.safe_execute_read_query(
                query,
                params
            )
            
            # Process results
            entities = []
            if records:
                for record in records:
                    entity = dict(record["e"].items())
                    entities.append(entity)
            
            response = {
                "status": "success",
                "timestamp": datetime.now().isoformat(),
                "message": f"Found {len(entities)} entities in domain '{domain_name}'",
                "domain_id": domain_id,
                "domain_name": domain_name,
                "container": container_name,
                "entity_count": len(entities),
                "entity_type_filter": entity_type,
                "entities": entities
            }
            
            return dict_to_json(response)
                
        except Exception as e:
            error_msg = f"Error retrieving domain entities: {str(e)}"
            self.logger.error(error_msg)
            error = ErrorDetail(
                code="INTERNAL_ERROR",
                message=error_msg,
                details=None
            )
            return dict_to_json(ErrorResponse(
                status="error",
                timestamp=datetime.now(),
                error=error
            ).model_dump())
    
    def create_domain_relationship(self, relationship: RelationshipCreate, 
                                container_name: str) -> str:
        """
        Create a relationship between two domains.
        
        Args:
            relationship: RelationshipCreate Pydantic model with relationship details
            container_name: Name of the project container
            
        Returns:
            JSON string with the result
        """
        try:
            self.base_manager.ensure_initialized()
            
            from_domain_id = relationship.source_id
            to_domain_id = relationship.target_id
            relation_type = relationship.relationship_type
            
            # Get domain names for better messages
            names_query = """
            MATCH (from:Entity {id: $from_id, entityType: 'Domain'})
            MATCH (to:Entity {id: $to_id, entityType: 'Domain'})
            RETURN from.name as from_name, to.name as to_name
            """
            
            # Use safe_execute_read_query for validation (read-only operation)
            names_records = self.base_manager.safe_execute_read_query(
                names_query,
                {"from_id": from_domain_id, "to_id": to_domain_id}
            )
            
            if not names_records or len(names_records) == 0:
                error = ErrorDetail(
                    code="DOMAINS_NOT_FOUND",
                    message="One or both domains not found",
                    details=None
                )
                return dict_to_json(ErrorResponse(
                    status="error",
                    timestamp=datetime.now(),
                    error=error
                ).model_dump())
                
            from_domain_name = names_records[0]["from_name"]
            to_domain_name = names_records[0]["to_name"]
            
            # Check if domains exist in container
            domains_query = """
            MATCH (c:Entity {name: $container_name, entityType: 'ProjectContainer'})
            MATCH (from:Entity {id: $from_id, entityType: 'Domain'})-[:PART_OF]->(c)
            MATCH (to:Entity {id: $to_id, entityType: 'Domain'})-[:PART_OF]->(c)
            RETURN from, to
            """
            
            # Use safe_execute_read_query for validation (read-only operation)
            domains_records = self.base_manager.safe_execute_read_query(
                domains_query,
                {"container_name": container_name, "from_id": from_domain_id, "to_id": to_domain_id}
            )
            
            if not domains_records or len(domains_records) == 0:
                error = ErrorDetail(
                    code="DOMAINS_NOT_IN_CONTAINER",
                    message=f"One or both domains not found in container '{container_name}'",
                    details=None
                )
                return dict_to_json(ErrorResponse(
                    status="error",
                    timestamp=datetime.now(),
                    error=error
                ).model_dump())
            
            # Check if relationship already exists
            check_query = """
            MATCH (c:Entity {name: $container_name, entityType: 'ProjectContainer'})
            MATCH (from:Entity {id: $from_id, entityType: 'Domain'})-[:PART_OF]->(c)
            MATCH (to:Entity {id: $to_id, entityType: 'Domain'})-[:PART_OF]->(c)
            MATCH (from)-[r]->(to)
            WHERE type(r) = $relation_type
            RETURN r
            """
            
            # Use safe_execute_read_query for validation (read-only operation)
            check_records = self.base_manager.safe_execute_read_query(
                check_query,
                {
                    "container_name": container_name, 
                    "from_id": from_domain_id, 
                    "to_id": to_domain_id,
                    "relation_type": relation_type
                }
            )
            
            if check_records and len(check_records) > 0:
                return dict_to_json(SuccessResponse(
                    status="success",
                    timestamp=datetime.now(),
                    message=f"Relationship of type '{relation_type}' already exists between domains '{from_domain_name}' and '{to_domain_name}'"
                ).model_dump())
            
            # Prepare relationship properties
            relation_props = {}
            if relationship.metadata:
                metadata_dict = relationship.metadata.model_dump(exclude_none=True)
                for key, value in metadata_dict.items():
                    relation_props[f"metadata_{key}"] = value
                    
            relation_props["created"] = time.time()
            relation_props["domain"] = "project"
            
            # Create dynamic relationship
            create_query = f"""
            MATCH (c:Entity {{name: $container_name, entityType: 'ProjectContainer'}})
            MATCH (from:Entity {{id: $from_id, entityType: 'Domain'}})-[:PART_OF]->(c)
            MATCH (to:Entity {{id: $to_id, entityType: 'Domain'}})-[:PART_OF]->(c)
            CREATE (from)-[r:{relation_type} $properties]->(to)
            RETURN r
            """
            
            # Use safe_execute_write_query for validation (write operation)
            create_records = self.base_manager.safe_execute_write_query(
                create_query,
                {
                    "container_name": container_name, 
                    "from_id": from_domain_id, 
                    "to_id": to_domain_id,
                    "properties": relation_props
                }
            )
            
            if not create_records or len(create_records) == 0:
                error = ErrorDetail(
                    code="RELATIONSHIP_CREATION_FAILED",
                    message=f"Failed to create relationship between domains '{from_domain_name}' and '{to_domain_name}'",
                    details=None
                )
                return dict_to_json(ErrorResponse(
                    status="error",
                    timestamp=datetime.now(),
                    error=error
                ).model_dump())
            
            return dict_to_json(SuccessResponse(
                status="success",
                timestamp=datetime.now(),
                message=f"Relationship of type '{relation_type}' created between domains '{from_domain_name}' and '{to_domain_name}'"
            ).model_dump())
                
        except Exception as e:
            error_msg = f"Error creating domain relationship: {str(e)}"
            self.logger.error(error_msg)
            error = ErrorDetail(
                code="INTERNAL_ERROR",
                message=error_msg,
                details=None
            )
            return dict_to_json(ErrorResponse(
                status="error",
                timestamp=datetime.now(),
                error=error
            ).model_dump()) 