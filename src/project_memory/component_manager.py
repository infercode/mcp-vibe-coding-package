from typing import Any, Dict, List, Optional, Union
import time
import json
import logging

from src.utils import dict_to_json, generate_id
from src.graph_memory.base_manager import BaseManager
from src.graph_memory.entity_manager import EntityManager

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
    
    def create_component(self, name: str, 
                      component_type: str,
                      domain_name: str,
                      container_name: str,
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
                return dict_to_json({
                    "error": f"Domain '{domain_name}' not found in container '{container_name}'"
                })
            
            # Check if component already exists in domain
            component_check_query = """
            MATCH (c:Entity {name: $container_name, entityType: 'ProjectContainer'})
            MATCH (d:Entity {name: $domain_name, entityType: 'Domain'})-[:PART_OF]->(c)
            MATCH (comp:Entity {name: $name})-[:BELONGS_TO]->(d)
            RETURN comp
            """
            
            component_records = self.base_manager.safe_execute_read_query(
                component_check_query,
                {"container_name": container_name, "domain_name": domain_name, "name": name}
            )
            
            if component_records and len(component_records) > 0:
                return dict_to_json({
                    "error": f"Component '{name}' already exists in domain '{domain_name}'"
                })
            
            # Generate component ID
            component_id = generate_id("cmp")
            timestamp = time.time()
            
            # Prepare component entity
            component_entity = {
                "id": component_id,
                "name": name,
                "entityType": component_type,
                "domain": "project",
                "created": timestamp,
                "lastUpdated": timestamp
            }
            
            if description:
                component_entity["description"] = description
                
            if content:
                component_entity["content"] = content
                
            if metadata:
                for key, value in metadata.items():
                    if key not in component_entity:
                        component_entity[key] = value
            
            # Create component entity
            create_query = """
            CREATE (comp:Entity $properties)
            RETURN comp
            """
            
            create_records = self.base_manager.safe_execute_write_query(
                create_query,
                {"properties": component_entity}
            )
            
            if not create_records or len(create_records) == 0:
                return dict_to_json({
                    "error": "Failed to create component entity"
                })
            
            # Add component to project container
            add_to_container_query = """
            MATCH (c:Entity {name: $container_name, entityType: 'ProjectContainer'})
            MATCH (comp:Entity {id: $component_id})
            CREATE (comp)-[:PART_OF {created: $timestamp}]->(c)
            RETURN comp
            """
            
            add_records = self.base_manager.safe_execute_write_query(
                add_to_container_query,
                {"container_name": container_name, "component_id": component_id, "timestamp": timestamp}
            )
            
            if not add_records or len(add_records) == 0:
                # Attempt to clean up the created entity
                self.base_manager.safe_execute_write_query(
                    "MATCH (comp:Entity {id: $component_id}) DELETE comp",
                    {"component_id": component_id}
                )
                
                return dict_to_json({
                    "error": f"Failed to add component '{name}' to container '{container_name}'"
                })
            
            # Add component to domain
            add_to_domain_query = """
            MATCH (d:Entity {name: $domain_name, entityType: 'Domain'})
            MATCH (comp:Entity {id: $component_id})
            CREATE (comp)-[:BELONGS_TO {created: $timestamp}]->(d)
            RETURN comp
            """
            
            domain_add_records = self.base_manager.safe_execute_write_query(
                add_to_domain_query,
                {"domain_name": domain_name, "component_id": component_id, "timestamp": timestamp}
            )
            
            if not domain_add_records or len(domain_add_records) == 0:
                # Attempt to clean up the created relationships and entity
                self.base_manager.safe_execute_write_query(
                    """
                    MATCH (comp:Entity {id: $component_id})
                    OPTIONAL MATCH (comp)-[r]-()
                    DELETE r, comp
                    """,
                    {"component_id": component_id}
                )
                
                return dict_to_json({
                    "error": f"Failed to add component '{name}' to domain '{domain_name}'"
                })
            
            # Get created component
            component = dict(domain_add_records[0]["comp"].items())
            
            return dict_to_json({
                "status": "success",
                "message": f"Component '{name}' created successfully in domain '{domain_name}'",
                "component": component
            })
                
        except Exception as e:
            error_msg = f"Error creating component: {str(e)}"
            self.logger.error(error_msg)
            return dict_to_json({"error": error_msg})
    
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
    
    def update_component(self, name: str, domain_name: str, container_name: str, 
                      updates: Dict[str, Any]) -> str:
        """
        Update a component's properties.
        
        Args:
            name: Name of the component
            domain_name: Name of the domain
            container_name: Name of the project container
            updates: Dictionary of properties to update
            
        Returns:
            JSON string with the updated component
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
                return dict_to_json({
                    "error": f"Component '{name}' not found in domain '{domain_name}'"
                })
            
            # Validate updates - prevent changing core properties
            protected_fields = ["id", "name", "domain", "created", "entityType"]
            invalid_updates = [field for field in updates if field in protected_fields]
            
            if invalid_updates:
                return dict_to_json({
                    "error": f"Cannot update protected fields: {', '.join(invalid_updates)}"
                })
            
            # Add lastUpdated timestamp
            updates["lastUpdated"] = time.time()
            
            # Prepare update parts
            set_parts = []
            for key, value in updates.items():
                set_parts.append(f"comp.{key} = ${key}")
            
            # Build update query
            update_query = f"""
            MATCH (c:Entity {{name: $container_name, entityType: 'ProjectContainer'}})
            MATCH (d:Entity {{name: $domain_name, entityType: 'Domain'}})-[:PART_OF]->(c)
            MATCH (comp:Entity {{name: $name}})-[:BELONGS_TO]->(d)
            SET {', '.join(set_parts)}
            RETURN comp
            """
            
            # Add name, domain_name and container_name to updates for the query
            params = {"name": name, "domain_name": domain_name, "container_name": container_name, **updates}
            
            update_records = self.base_manager.safe_execute_write_query(
                update_query,
                params
            )
            
            if not update_records or len(update_records) == 0:
                return dict_to_json({
                    "error": f"Failed to update component '{name}'"
                })
            
            # Return updated component
            component = dict(update_records[0]["comp"].items())
            
            return dict_to_json({
                "status": "success",
                "message": f"Component '{name}' updated successfully",
                "component": component
            })
                
        except Exception as e:
            error_msg = f"Error updating component: {str(e)}"
            self.logger.error(error_msg)
            return dict_to_json({"error": error_msg})
    
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
                return dict_to_json({
                    "error": f"Component '{name}' not found in domain '{domain_name}'"
                })
            
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
                return dict_to_json({
                    "error": f"Cannot delete component '{name}' with {relation_count} relationships. Remove relationships first."
                })
            
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
            
            return dict_to_json({
                "status": "success",
                "message": f"Component '{name}' deleted successfully"
            })
                
        except Exception as e:
            error_msg = f"Error deleting component: {str(e)}"
            self.logger.error(error_msg)
            return dict_to_json({"error": error_msg})
    
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
                return dict_to_json({
                    "error": f"Domain '{domain_name}' not found in container '{container_name}'"
                })
            
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
            
            return dict_to_json({
                "domain": domain_name,
                "container": container_name,
                "component_count": len(components),
                "component_type_filter": component_type,
                "components": components
            })
                
        except Exception as e:
            error_msg = f"Error listing components: {str(e)}"
            self.logger.error(error_msg)
            return dict_to_json({"error": error_msg})
    
    def create_component_relationship(self, from_component: str, to_component: str, 
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
            self.base_manager.ensure_initialized()
            
            # Check if components exist in domain
            components_query = """
            MATCH (c:Entity {name: $container_name, entityType: 'ProjectContainer'})
            MATCH (d:Entity {name: $domain_name, entityType: 'Domain'})-[:PART_OF]->(c)
            MATCH (from:Entity {name: $from_component})-[:BELONGS_TO]->(d)
            MATCH (to:Entity {name: $to_component})-[:BELONGS_TO]->(d)
            RETURN from, to
            """
            
            components_records = self.base_manager.safe_execute_read_query(
                components_query,
                {
                    "container_name": container_name, 
                    "domain_name": domain_name, 
                    "from_component": from_component, 
                    "to_component": to_component
                }
            )
            
            if not components_records or len(components_records) == 0:
                return dict_to_json({
                    "error": f"One or both components not found in domain '{domain_name}'"
                })
            
            # Check if relationship already exists
            check_query = """
            MATCH (c:Entity {name: $container_name, entityType: 'ProjectContainer'})
            MATCH (d:Entity {name: $domain_name, entityType: 'Domain'})-[:PART_OF]->(c)
            MATCH (from:Entity {name: $from_component})-[:BELONGS_TO]->(d)
            MATCH (to:Entity {name: $to_component})-[:BELONGS_TO]->(d)
            MATCH (from)-[r]->(to)
            WHERE type(r) = $relation_type
            RETURN r
            """
            
            check_records = self.base_manager.safe_execute_read_query(
                check_query,
                {
                    "container_name": container_name, 
                    "domain_name": domain_name, 
                    "from_component": from_component, 
                    "to_component": to_component,
                    "relation_type": relation_type
                }
            )
            
            if check_records and len(check_records) > 0:
                return dict_to_json({
                    "status": "success",
                    "message": f"Relationship of type '{relation_type}' already exists between components '{from_component}' and '{to_component}'"
                })
            
            # Prepare relationship properties
            relation_props = properties or {}
            relation_props["created"] = time.time()
            relation_props["domain"] = "project"
            
            # Create dynamic relationship
            create_query = f"""
            MATCH (c:Entity {{name: $container_name, entityType: 'ProjectContainer'}})
            MATCH (d:Entity {{name: $domain_name, entityType: 'Domain'}})-[:PART_OF]->(c)
            MATCH (from:Entity {{name: $from_component}})-[:BELONGS_TO]->(d)
            MATCH (to:Entity {{name: $to_component}})-[:BELONGS_TO]->(d)
            CREATE (from)-[r:{relation_type} $properties]->(to)
            RETURN r
            """
            
            create_records = self.base_manager.safe_execute_write_query(
                create_query,
                {
                    "container_name": container_name, 
                    "domain_name": domain_name, 
                    "from_component": from_component, 
                    "to_component": to_component,
                    "properties": relation_props
                }
            )
            
            if not create_records or len(create_records) == 0:
                return dict_to_json({
                    "error": f"Failed to create relationship between components '{from_component}' and '{to_component}'"
                })
            
            return dict_to_json({
                "status": "success",
                "message": f"Relationship of type '{relation_type}' created between components '{from_component}' and '{to_component}'"
            })
                
        except Exception as e:
            error_msg = f"Error creating component relationship: {str(e)}"
            self.logger.error(error_msg)
            return dict_to_json({"error": error_msg}) 