from typing import Any, Dict, List, Optional, Union
import time
import json
import logging
from datetime import datetime

from src.utils import dict_to_json, generate_id
from src.graph_memory.base_manager import BaseManager
from src.graph_memory.entity_manager import EntityManager
from src.models.project_memory import (
    ComponentCreate, 
    ComponentUpdate,
    ComponentResponse,
    Metadata,
    ErrorResponse,
    ErrorDetail,
    SuccessResponse,
    RelationshipCreate
)

class ComponentManager:
    """
    Manager for project components.
    Handles creation and management of project components within domains.
    """
    
    def __init__(self, base_manager: BaseManager):
        """
        Initialize the component manager.
        
        Args:
            base_manager: Base manager for graph operations
        """
        self.base_manager = base_manager
        self.logger = logging.getLogger(__name__)
        self.entity_manager = EntityManager(base_manager)
    
    def create_component(self, 
                      component: ComponentCreate,
                      domain_name: str,
                      container_name: str) -> str:
        """
        Create a new project component within a domain.
        
        Args:
            component: ComponentCreate Pydantic model with component details
            domain_name: Name of the domain
            container_name: Name of the project container
            
        Returns:
            JSON string with the created component
        """
        try:
            self.base_manager.ensure_initialized()
            
            # Check if domain exists in container
            domain_query = """
            MATCH (c:Entity {name: $container_name, entityType: 'ProjectContainer'})
            MATCH (d:Entity {name: $domain_name, entityType: 'Domain'})-[:PART_OF]->(c)
            RETURN d
            """
            
            domain_records = self.base_manager.safe_execute_read_query(
                domain_query,
                {"container_name": container_name, "domain_name": domain_name}
            )
            
            if not domain_records or len(domain_records) == 0:
                error = ErrorDetail(
                    code="DOMAIN_NOT_FOUND",
                    message=f"Domain '{domain_name}' not found in container '{container_name}'",
                    details=None
                )
                return dict_to_json(ErrorResponse(
                    status="error",
                    timestamp=datetime.now(),
                    error=error
                ).model_dump())
            
            # Check if component already exists in domain
            component_check_query = """
            MATCH (c:Entity {name: $container_name, entityType: 'ProjectContainer'})
            MATCH (d:Entity {name: $domain_name, entityType: 'Domain'})-[:PART_OF]->(c)
            MATCH (comp:Entity {name: $name})-[:BELONGS_TO]->(d)
            RETURN comp
            """
            
            component_records = self.base_manager.safe_execute_read_query(
                component_check_query,
                {"container_name": container_name, "domain_name": domain_name, "name": component.name}
            )
            
            if component_records and len(component_records) > 0:
                error = ErrorDetail(
                    code="COMPONENT_EXISTS",
                    message=f"Component '{component.name}' already exists in domain '{domain_name}'",
                    details=None
                )
                return dict_to_json(ErrorResponse(
                    status="error",
                    timestamp=datetime.now(),
                    error=error
                ).model_dump())
            
            # Generate component ID
            component_id = generate_id("cmp")
            
            # Prepare component entity
            component_entity = {
                "id": component_id,
                "name": component.name,
                "entityType": component.type,
                "domain": "project",
                "created": time.time(),
                "lastUpdated": time.time()
            }
            
            if component.description:
                component_entity["description"] = component.description
                
            if component.metadata:
                metadata_dict = component.metadata.model_dump(exclude_none=True)
                for key, value in metadata_dict.items():
                    component_entity[f"metadata_{key}"] = value
            
            # Create component query
            create_query = f"""
            MATCH (d:Entity {{name: $domain_name, entityType: 'Domain'}})
            MATCH (c:Entity {{name: $container_name, entityType: 'ProjectContainer'}})
            CREATE (e:Entity $properties)
            CREATE (e)-[:PART_OF]->(d)
            CREATE (e)-[:BELONGS_TO]->(c)
            SET e.created = datetime(),
                e.lastUpdated = datetime()
            RETURN e
            """
            
            create_records = self.base_manager.safe_execute_write_query(
                create_query,
                {"properties": component_entity}
            )
            
            if not create_records or len(create_records) == 0:
                error = ErrorDetail(
                    code="CREATION_FAILED",
                    message="Failed to create component entity",
                    details=None
                )
                return dict_to_json(ErrorResponse(
                    status="error",
                    timestamp=datetime.now(),
                    error=error
                ).model_dump())
            
            # Get created component
            component_data = dict(create_records[0]["e"].items())
            
            # Create response
            response = ComponentResponse(
                status="success",
                timestamp=datetime.now(),
                message=f"Component '{component.name}' created successfully in domain '{domain_name}'",
                component_id=component_id,
                component=component_data
            )
            
            return dict_to_json(response.model_dump(exclude_none=True))
                
        except Exception as e:
            error_msg = f"Error creating component: {str(e)}"
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
    
    def get_component(self, name: str, domain_name: str, container_name: str) -> str:
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
            self.base_manager.ensure_initialized()
            
            # Get component query
            query = """
            MATCH (c:Entity {name: $container_name, entityType: 'ProjectContainer'})
            MATCH (d:Entity {name: $domain_name, entityType: 'Domain'})-[:PART_OF]->(c)
            MATCH (comp:Entity {name: $name})-[:BELONGS_TO]->(d)
            OPTIONAL MATCH (comp)-[:HAS_OBSERVATION]->(o:Observation)
            RETURN comp, count(o) as observation_count
            """
            
            records = self.base_manager.safe_execute_read_query(
                query,
                {"container_name": container_name, "domain_name": domain_name, "name": name}
            )
            
            if not records or len(records) == 0:
                return dict_to_json({
                    "error": f"Component '{name}' not found in domain '{domain_name}'"
                })
            
            # Extract component info
            record = records[0]
            component = dict(record["comp"].items())
            observation_count = record["observation_count"]
            
            # Add observation count
            component["observation_count"] = observation_count
            
            # Get relationships with other components
            relations_query = """
            MATCH (c:Entity {name: $container_name, entityType: 'ProjectContainer'})
            MATCH (d:Entity {name: $domain_name, entityType: 'Domain'})-[:PART_OF]->(c)
            MATCH (comp:Entity {name: $name})-[:BELONGS_TO]->(d)
            MATCH (comp)-[r]-(other:Entity)
            WHERE type(r) <> 'BELONGS_TO' AND type(r) <> 'PART_OF' AND type(r) <> 'HAS_OBSERVATION'
            RETURN type(r) as relation_type, count(r) as count
            """
            
            relation_records = self.base_manager.safe_execute_read_query(
                relations_query,
                {"container_name": container_name, "domain_name": domain_name, "name": name}
            )
            
            # Process relation counts
            relation_types = {}
            if relation_records:
                for record in relation_records:
                    rel_type = record["relation_type"]
                    count = record["count"]
                    if rel_type:
                        relation_types[rel_type] = count
            
            # Add relation statistics
            component["relation_types"] = relation_types
            
            return dict_to_json({
                "component": component
            })
                
        except Exception as e:
            error_msg = f"Error retrieving component: {str(e)}"
            self.logger.error(error_msg)
            return dict_to_json({"error": error_msg})
    
    def update_component(self, 
                      component_update: ComponentUpdate,
                      domain_name: str, 
                      container_name: str) -> str:
        """
        Update a component's properties.
        
        Args:
            component_update: ComponentUpdate Pydantic model with updated properties
            domain_name: Name of the domain
            container_name: Name of the project container
            
        Returns:
            JSON string with the updated component
        """
        try:
            self.base_manager.ensure_initialized()
            
            # Check if component exists in domain
            component_query = """
            MATCH (c:Entity {name: $container_name, entityType: 'ProjectContainer'})
            MATCH (d:Entity {name: $domain_name, entityType: 'Domain'})-[:PART_OF]->(c)
            MATCH (comp:Entity {id: $id})-[:BELONGS_TO]->(d)
            RETURN comp
            """
            
            component_records = self.base_manager.safe_execute_read_query(
                component_query,
                {"container_name": container_name, "domain_name": domain_name, "id": component_update.id}
            )
            
            if not component_records or len(component_records) == 0:
                error = ErrorDetail(
                    code="COMPONENT_NOT_FOUND",
                    message=f"Component with ID '{component_update.id}' not found in domain '{domain_name}'",
                    details=None
                )
                return dict_to_json(ErrorResponse(
                    status="error",
                    timestamp=datetime.now(),
                    error=error
                ).model_dump())
            
            # Get current component data
            component = dict(component_records[0]["comp"].items())
            
            # Validate updates - prevent changing core properties
            protected_fields = ["id", "domain", "created", "entityType"]
            updates = component_update.model_dump(exclude_none=True)
            
            # Remove ID as it's used for identification
            if "id" in updates:
                updates.pop("id")
            
            # Check for protected fields
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
                set_parts.append(f"comp.{key} = ${key}")
            
            # If no updates, return success
            if not set_parts:
                response = ComponentResponse(
                    status="success",
                    timestamp=datetime.now(),
                    message=f"No updates provided for component",
                    component_id=component_update.id,
                    component=component
                )
                return dict_to_json(response.model_dump(exclude_none=True))
            
            # Update entity
            update_query = f"""
            MATCH (comp:Entity {{id: $id}})
            WHERE EXISTS((comp)-[:BELONGS_TO]->(:Entity {{name: $domain_name, entityType: 'Domain'}}))
            AND EXISTS((comp)-[:PART_OF]->(:Entity {{name: $container_name, entityType: 'ProjectContainer'}}))
            SET {', '.join(set_parts)}, comp.lastUpdated = datetime()
            RETURN comp
            """
            
            # Add component id, domain_name and container_name to updates for the query
            params = {"id": component_update.id, "domain_name": domain_name, "container_name": container_name, **updates}
            
            update_records = self.base_manager.safe_execute_write_query(
                update_query,
                params
            )
            
            if not update_records or len(update_records) == 0:
                error = ErrorDetail(
                    code="UPDATE_FAILED",
                    message=f"Failed to update component '{component_update.id}'",
                    details=None
                )
                return dict_to_json(ErrorResponse(
                    status="error",
                    timestamp=datetime.now(),
                    error=error
                ).model_dump())
            
            # Return updated component
            updated_component = dict(update_records[0]["comp"].items())
            
            response = ComponentResponse(
                status="success",
                timestamp=datetime.now(),
                message=f"Component updated successfully",
                component_id=component_update.id,
                component=updated_component
            )
            
            return dict_to_json(response.model_dump(exclude_none=True))
                
        except Exception as e:
            error_msg = f"Error updating component: {str(e)}"
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
    
    def delete_component(self, name: str, domain_name: str, container_name: str) -> str:
        """
        Delete a component from a domain.
        
        Args:
            name: Name of the component
            domain_name: Name of the domain
            container_name: Name of the project container
            
        Returns:
            JSON string with the deletion result
        """
        try:
            self.base_manager.ensure_initialized()
            
            # Check if component exists in domain
            component_query = """
            MATCH (c:Entity {name: $container_name, entityType: 'ProjectContainer'})
            MATCH (d:Entity {name: $domain_name, entityType: 'Domain'})-[:PART_OF]->(c)
            MATCH (comp:Entity {name: $name})-[:BELONGS_TO]->(d)
            RETURN comp
            """
            
            component_records = self.base_manager.safe_execute_read_query(
                component_query,
                {"container_name": container_name, "domain_name": domain_name, "name": name}
            )
            
            if not component_records or len(component_records) == 0:
                error = ErrorDetail(
                    code="COMPONENT_NOT_FOUND",
                    message=f"Component '{name}' not found in domain '{domain_name}'",
                    details=None
                )
                return dict_to_json(ErrorResponse(
                    status="error",
                    timestamp=datetime.now(),
                    error=error
                ).model_dump())
            
            # Get component ID
            component_id = component_records[0]["comp"]["id"]
            
            # Check if component has relationships with other components
            check_relations_query = """
            MATCH (comp:Entity {id: $component_id})
            MATCH (comp)-[r]-(other:Entity)
            WHERE type(r) <> 'BELONGS_TO' AND type(r) <> 'PART_OF' AND type(r) <> 'HAS_OBSERVATION'
            RETURN count(r) as relation_count
            """
            
            relation_records = self.base_manager.safe_execute_read_query(
                check_relations_query,
                {"component_id": component_id}
            )
            
            relation_count = 0
            if relation_records and len(relation_records) > 0:
                relation_count = relation_records[0]["relation_count"]
            
            if relation_count > 0:
                error = ErrorDetail(
                    code="COMPONENT_HAS_RELATIONSHIPS",
                    message=f"Cannot delete component '{name}' with {relation_count} relationships. Remove relationships first.",
                    details=None
                )
                return dict_to_json(ErrorResponse(
                    status="error",
                    timestamp=datetime.now(),
                    error=error
                ).model_dump())
            
            # Delete component observations
            delete_observations_query = """
            MATCH (comp:Entity {id: $component_id})
            MATCH (comp)-[:HAS_OBSERVATION]->(o:Observation)
            DELETE o
            """
            
            self.base_manager.safe_execute_write_query(
                delete_observations_query,
                {"component_id": component_id}
            )
            
            # Delete component relationships
            delete_rels_query = """
            MATCH (comp:Entity {id: $component_id})
            MATCH (comp)-[r]-()
            DELETE r
            """
            
            self.base_manager.safe_execute_write_query(
                delete_rels_query,
                {"component_id": component_id}
            )
            
            # Delete component
            delete_query = """
            MATCH (comp:Entity {id: $component_id})
            DELETE comp
            """
            
            self.base_manager.safe_execute_write_query(
                delete_query,
                {"component_id": component_id}
            )
            
            return dict_to_json(SuccessResponse(
                status="success",
                timestamp=datetime.now(),
                message=f"Component '{name}' deleted successfully"
            ).model_dump(exclude_none=True))
                
        except Exception as e:
            error_msg = f"Error deleting component: {str(e)}"
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
    
    def list_components(self, domain_name: str, container_name: str, 
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
            self.base_manager.ensure_initialized()
            
            # Check if domain exists
            domain_query = """
            MATCH (c:Entity {name: $container_name, entityType: 'ProjectContainer'})
            MATCH (d:Entity {name: $domain_name, entityType: 'Domain'})-[:PART_OF]->(c)
            RETURN d
            """
            
            domain_records = self.base_manager.safe_execute_read_query(
                domain_query,
                {"container_name": container_name, "domain_name": domain_name}
            )
            
            if not domain_records or len(domain_records) == 0:
                error = ErrorDetail(
                    code="DOMAIN_NOT_FOUND",
                    message=f"Domain '{domain_name}' not found in container '{container_name}'",
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
            
            # Build query based on component_type
            if component_type:
                query = f"""
                MATCH (c:Entity {{name: $container_name, entityType: 'ProjectContainer'}})
                MATCH (d:Entity {{name: $domain_name, entityType: 'Domain'}})-[:PART_OF]->(c)
                MATCH (comp:Entity)-[:BELONGS_TO]->(d)
                WHERE comp.entityType = $component_type
                RETURN comp
                ORDER BY comp.{sort_by}
                LIMIT $limit
                """
                
                params = {
                    "container_name": container_name, 
                    "domain_name": domain_name,
                    "component_type": component_type,
                    "limit": limit
                }
            else:
                query = f"""
                MATCH (c:Entity {{name: $container_name, entityType: 'ProjectContainer'}})
                MATCH (d:Entity {{name: $domain_name, entityType: 'Domain'}})-[:PART_OF]->(c)
                MATCH (comp:Entity)-[:BELONGS_TO]->(d)
                RETURN comp
                ORDER BY comp.{sort_by}
                LIMIT $limit
                """
                
                params = {
                    "container_name": container_name, 
                    "domain_name": domain_name,
                    "limit": limit
                }
            
            records = self.base_manager.safe_execute_read_query(
                query,
                params
            )
            
            # Process results
            components = []
            if records:
                for record in records:
                    component = dict(record["comp"].items())
                    components.append(component)
            
            # Create custom response structure since there's no specific model for listing components
            response_data = {
                "status": "success",
                "timestamp": datetime.now().isoformat(),
                "message": f"Found {len(components)} components in domain '{domain_name}'",
                "domain": domain_name,
                "container": container_name,
                "component_count": len(components),
                "component_type_filter": component_type,
                "components": components
            }
            
            return dict_to_json(response_data)
                
        except Exception as e:
            error_msg = f"Error listing components: {str(e)}"
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
    
    def create_component_relationship(self, 
                                  relationship: RelationshipCreate,
                                  domain_name: str, 
                                  container_name: str) -> str:
        """
        Create a relationship between two components in a domain.
        
        Args:
            relationship: RelationshipCreate Pydantic model with relationship details
            domain_name: Name of the domain
            container_name: Name of the project container
            
        Returns:
            JSON string with the result
        """
        try:
            self.base_manager.ensure_initialized()
            
            # Split component ids to get names
            # Assuming component ids are in the format of 'cmp-<name>'
            from_id = relationship.source_id
            to_id = relationship.target_id
            
            # Extract component names from ids if needed
            # This depends on the format of your IDs
            from_component = from_id
            to_component = to_id
            
            # Check if components exist in domain
            components_query = """
            MATCH (c:Entity {name: $container_name, entityType: 'ProjectContainer'})
            MATCH (d:Entity {name: $domain_name, entityType: 'Domain'})-[:PART_OF]->(c)
            MATCH (from:Entity {id: $from_id})-[:BELONGS_TO]->(d)
            MATCH (to:Entity {id: $to_id})-[:BELONGS_TO]->(d)
            RETURN from.name as from_name, to.name as to_name
            """
            
            components_records = self.base_manager.safe_execute_read_query(
                components_query,
                {
                    "container_name": container_name, 
                    "domain_name": domain_name, 
                    "from_id": from_id, 
                    "to_id": to_id
                }
            )
            
            if not components_records or len(components_records) == 0:
                error = ErrorDetail(
                    code="COMPONENTS_NOT_FOUND",
                    message=f"One or both components not found in domain '{domain_name}'",
                    details=None
                )
                return dict_to_json(ErrorResponse(
                    status="error",
                    timestamp=datetime.now(),
                    error=error
                ).model_dump())
            
            # Get component names for more descriptive messages
            from_name = components_records[0]["from_name"]
            to_name = components_records[0]["to_name"]
            
            # Check if relationship already exists
            check_query = """
            MATCH (c:Entity {name: $container_name, entityType: 'ProjectContainer'})
            MATCH (d:Entity {name: $domain_name, entityType: 'Domain'})-[:PART_OF]->(c)
            MATCH (from:Entity {id: $from_id})-[:BELONGS_TO]->(d)
            MATCH (to:Entity {id: $to_id})-[:BELONGS_TO]->(d)
            MATCH (from)-[r]->(to)
            WHERE type(r) = $relation_type
            RETURN r
            """
            
            check_records = self.base_manager.safe_execute_read_query(
                check_query,
                {
                    "container_name": container_name, 
                    "domain_name": domain_name, 
                    "from_id": from_id, 
                    "to_id": to_id,
                    "relation_type": relationship.relationship_type
                }
            )
            
            if check_records and len(check_records) > 0:
                return dict_to_json(SuccessResponse(
                    status="success",
                    timestamp=datetime.now(),
                    message=f"Relationship of type '{relationship.relationship_type}' already exists between components '{from_name}' and '{to_name}'"
                ).model_dump(exclude_none=True))
            
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
            MATCH (d:Entity {{name: $domain_name, entityType: 'Domain'}})-[:PART_OF]->(c)
            MATCH (from:Entity {{id: $from_id}})-[:BELONGS_TO]->(d)
            MATCH (to:Entity {{id: $to_id}})-[:BELONGS_TO]->(d)
            CREATE (from)-[r:{relationship.relationship_type} $properties]->(to)
            RETURN r
            """
            
            create_records = self.base_manager.safe_execute_write_query(
                create_query,
                {
                    "container_name": container_name, 
                    "domain_name": domain_name, 
                    "from_id": from_id, 
                    "to_id": to_id,
                    "properties": relation_props
                }
            )
            
            if not create_records or len(create_records) == 0:
                error = ErrorDetail(
                    code="RELATIONSHIP_CREATION_FAILED",
                    message=f"Failed to create relationship between components '{from_name}' and '{to_name}'",
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
                message=f"Relationship of type '{relationship.relationship_type}' created between components '{from_name}' and '{to_name}'"
            ).model_dump(exclude_none=True))
                
        except Exception as e:
            error_msg = f"Error creating component relationship: {str(e)}"
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