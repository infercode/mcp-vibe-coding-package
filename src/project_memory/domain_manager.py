from typing import Any, Dict, List, Optional, Union
import time
import json

from src.utils import dict_to_json, generate_id
from src.graph_memory.base_manager import BaseManager
from src.graph_memory.entity_manager import EntityManager

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
    
    def create_domain(self, name: str, 
                    container_name: str,
                    description: Optional[str] = None,
                    properties: Optional[Dict[str, Any]] = None) -> str:
        """
        Create a new domain entity within a project container.
        
        Args:
            name: Name of the domain
            container_name: Name of the project container
            description: Optional description of the domain
            properties: Optional additional properties
            
        Returns:
            JSON string with the created domain
        """
        try:
            self.base_manager.ensure_initialized()
            
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
                return dict_to_json({
                    "error": f"Project container '{container_name}' not found"
                })
            
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
                return dict_to_json({
                    "error": f"Domain '{name}' already exists in container '{container_name}'"
                })
            
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
                
            if properties:
                for key, value in properties.items():
                    if key not in domain_entity:
                        domain_entity[key] = value
            
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
                return dict_to_json({
                    "error": "Failed to create domain entity"
                })
            
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
                
                return dict_to_json({
                    "error": f"Failed to add domain '{name}' to container '{container_name}'"
                })
            
            # Get created domain
            domain = dict(add_records[0]["d"].items())
            
            return dict_to_json({
                "status": "success",
                "message": f"Domain '{name}' created successfully in container '{container_name}'",
                "domain": domain
            })
                
        except Exception as e:
            error_msg = f"Error creating domain: {str(e)}"
            self.logger.error(error_msg)
            return dict_to_json({"error": error_msg})
    
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
                return dict_to_json({
                    "error": f"Domain '{name}' not found in container '{container_name}'"
                })
            
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
            
            return dict_to_json({
                "domain": domain
            })
                
        except Exception as e:
            error_msg = f"Error retrieving domain: {str(e)}"
            self.logger.error(error_msg)
            return dict_to_json({"error": error_msg})
    
    def update_domain(self, name: str, container_name: str, updates: Dict[str, Any]) -> str:
        """
        Update a domain entity's properties.
        
        Args:
            name: Name of the domain
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
            MATCH (d:Entity {name: $name, entityType: 'Domain'})-[:PART_OF]->(c)
            RETURN d
            """
            
            # Use safe_execute_read_query for validation (read-only operation)
            domain_records = self.base_manager.safe_execute_read_query(
                domain_query,
                {"container_name": container_name, "name": name}
            )
            
            if not domain_records or len(domain_records) == 0:
                return dict_to_json({
                    "error": f"Domain '{name}' not found in container '{container_name}'"
                })
            
            # Validate updates - prevent changing core properties
            protected_fields = ["id", "name", "entityType", "domain", "created"]
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
                set_parts.append(f"d.{key} = ${key}")
            
            # Build update query
            update_query = f"""
            MATCH (c:Entity {{name: $container_name, entityType: 'ProjectContainer'}})
            MATCH (d:Entity {{name: $name, entityType: 'Domain'}})-[:PART_OF]->(c)
            SET {', '.join(set_parts)}
            RETURN d
            """
            
            # Add name and container_name to updates for the query
            params = {"name": name, "container_name": container_name, **updates}
            
            # Use safe_execute_write_query for validation (write operation)
            update_records = self.base_manager.safe_execute_write_query(
                update_query,
                params
            )
            
            if not update_records or len(update_records) == 0:
                return dict_to_json({
                    "error": f"Failed to update domain '{name}'"
                })
            
            # Return updated domain
            domain = dict(update_records[0]["d"].items())
            
            return dict_to_json({
                "status": "success",
                "message": f"Domain '{name}' updated successfully",
                "domain": domain
            })
                
        except Exception as e:
            error_msg = f"Error updating domain: {str(e)}"
            self.logger.error(error_msg)
            return dict_to_json({"error": error_msg})
    
    def delete_domain(self, name: str, container_name: str, delete_components: bool = False) -> str:
        """
        Delete a domain entity from a project container.
        
        Args:
            name: Name of the domain
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
            MATCH (d:Entity {name: $name, entityType: 'Domain'})-[:PART_OF]->(c)
            RETURN d
            """
            
            # Use safe_execute_read_query for validation (read-only operation)
            domain_records = self.base_manager.safe_execute_read_query(
                domain_query,
                {"container_name": container_name, "name": name}
            )
            
            if not domain_records or len(domain_records) == 0:
                return dict_to_json({
                    "error": f"Domain '{name}' not found in container '{container_name}'"
                })
            
            # Get domain ID
            domain_id = domain_records[0]["d"]["id"]
            
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
                
                return dict_to_json({
                    "status": "success",
                    "message": f"Domain '{name}' and {deleted_count} components deleted successfully"
                })
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
                    return dict_to_json({
                        "error": f"Cannot delete domain '{name}' with {component_count} components. Set delete_components=True to delete the domain and its components."
                    })
                
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
                
                return dict_to_json({
                    "status": "success",
                    "message": f"Domain '{name}' deleted successfully"
                })
                
        except Exception as e:
            error_msg = f"Error deleting domain: {str(e)}"
            self.logger.error(error_msg)
            return dict_to_json({"error": error_msg})
    
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
                return dict_to_json({
                    "error": f"Project container '{container_name}' not found"
                })
            
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
            
            return dict_to_json({
                "container": container_name,
                "domains": domains,
                "count": len(domains)
            })
                
        except Exception as e:
            error_msg = f"Error listing domains: {str(e)}"
            self.logger.error(error_msg)
            return dict_to_json({"error": error_msg})
    
    def add_entity_to_domain(self, domain_name: str, container_name: str, 
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
            self.base_manager.ensure_initialized()
            
            # Check if domain exists in container
            domain_query = """
            MATCH (c:Entity {name: $container_name, entityType: 'ProjectContainer'})
            MATCH (d:Entity {name: $domain_name, entityType: 'Domain'})-[:PART_OF]->(c)
            RETURN d
            """
            
            # Use safe_execute_read_query for validation (read-only operation)
            domain_records = self.base_manager.safe_execute_read_query(
                domain_query,
                {"container_name": container_name, "domain_name": domain_name}
            )
            
            if not domain_records or len(domain_records) == 0:
                return dict_to_json({
                    "error": f"Domain '{domain_name}' not found in container '{container_name}'"
                })
            
            # Check if entity exists
            entity_query = """
            MATCH (e:Entity {name: $entity_name})
            RETURN e
            """
            
            # Use safe_execute_read_query for validation (read-only operation)
            entity_records = self.base_manager.safe_execute_read_query(
                entity_query,
                {"entity_name": entity_name}
            )
            
            if not entity_records or len(entity_records) == 0:
                return dict_to_json({
                    "error": f"Entity '{entity_name}' not found"
                })
            
            # Check if entity is already in the domain
            check_query = """
            MATCH (c:Entity {name: $container_name, entityType: 'ProjectContainer'})
            MATCH (d:Entity {name: $domain_name, entityType: 'Domain'})-[:PART_OF]->(c)
            MATCH (e:Entity {name: $entity_name})
            MATCH (e)-[:BELONGS_TO]->(d)
            RETURN e
            """
            
            # Use safe_execute_read_query for validation (read-only operation)
            check_records = self.base_manager.safe_execute_read_query(
                check_query,
                {"container_name": container_name, "domain_name": domain_name, "entity_name": entity_name}
            )
            
            if check_records and len(check_records) > 0:
                return dict_to_json({
                    "status": "success",
                    "message": f"Entity '{entity_name}' is already in domain '{domain_name}'"
                })
            
            # Check if entity is in the project container
            container_check_query = """
            MATCH (c:Entity {name: $container_name, entityType: 'ProjectContainer'})
            MATCH (e:Entity {name: $entity_name})
            OPTIONAL MATCH (e)-[:PART_OF]->(c)
            RETURN exists((e)-[:PART_OF]->(c)) as in_container
            """
            
            # Use safe_execute_read_query for validation (read-only operation)
            container_check_records = self.base_manager.safe_execute_read_query(
                container_check_query,
                {"container_name": container_name, "entity_name": entity_name}
            )
            
            in_container = False
            if container_check_records and len(container_check_records) > 0:
                in_container = container_check_records[0]["in_container"]
            
            # If entity is not already in container, add it
            if not in_container:
                add_to_container_query = """
                MATCH (c:Entity {name: $container_name, entityType: 'ProjectContainer'})
                MATCH (e:Entity {name: $entity_name})
                CREATE (e)-[:PART_OF {created: $timestamp}]->(c)
                """
                
                # Use safe_execute_write_query for validation (write operation)
                self.base_manager.safe_execute_write_query(
                    add_to_container_query,
                    {"container_name": container_name, "entity_name": entity_name, "timestamp": time.time()}
                )
            
            # Add entity to domain
            add_query = """
            MATCH (c:Entity {name: $container_name, entityType: 'ProjectContainer'})
            MATCH (d:Entity {name: $domain_name, entityType: 'Domain'})-[:PART_OF]->(c)
            MATCH (e:Entity {name: $entity_name})
            CREATE (e)-[:BELONGS_TO {created: $timestamp}]->(d)
            RETURN e
            """
            
            # Use safe_execute_write_query for validation (write operation)
            add_records = self.base_manager.safe_execute_write_query(
                add_query,
                {
                    "container_name": container_name, 
                    "domain_name": domain_name, 
                    "entity_name": entity_name, 
                    "timestamp": time.time()
                }
            )
            
            if not add_records or len(add_records) == 0:
                return dict_to_json({
                    "error": f"Failed to add entity '{entity_name}' to domain '{domain_name}'"
                })
            
            return dict_to_json({
                "status": "success",
                "message": f"Entity '{entity_name}' added to domain '{domain_name}'"
            })
                
        except Exception as e:
            error_msg = f"Error adding entity to domain: {str(e)}"
            self.logger.error(error_msg)
            return dict_to_json({"error": error_msg})
    
    def remove_entity_from_domain(self, domain_name: str, container_name: str, 
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
            self.base_manager.ensure_initialized()
            
            # Check if the relationship exists
            check_query = """
            MATCH (c:Entity {name: $container_name, entityType: 'ProjectContainer'})
            MATCH (d:Entity {name: $domain_name, entityType: 'Domain'})-[:PART_OF]->(c)
            MATCH (e:Entity {name: $entity_name})
            MATCH (e)-[r:BELONGS_TO]->(d)
            RETURN r
            """
            
            # Use safe_execute_read_query for validation (read-only operation)
            check_records = self.base_manager.safe_execute_read_query(
                check_query,
                {"container_name": container_name, "domain_name": domain_name, "entity_name": entity_name}
            )
            
            if not check_records or len(check_records) == 0:
                return dict_to_json({
                    "error": f"Entity '{entity_name}' is not in domain '{domain_name}'"
                })
            
            # Remove entity from domain
            remove_query = """
            MATCH (c:Entity {name: $container_name, entityType: 'ProjectContainer'})
            MATCH (d:Entity {name: $domain_name, entityType: 'Domain'})-[:PART_OF]->(c)
            MATCH (e:Entity {name: $entity_name})
            MATCH (e)-[r:BELONGS_TO]->(d)
            DELETE r
            """
            
            # Use safe_execute_write_query for validation (write operation)
            self.base_manager.safe_execute_write_query(
                remove_query,
                {"container_name": container_name, "domain_name": domain_name, "entity_name": entity_name}
            )
            
            return dict_to_json({
                "status": "success",
                "message": f"Entity '{entity_name}' removed from domain '{domain_name}'"
            })
                
        except Exception as e:
            error_msg = f"Error removing entity from domain: {str(e)}"
            self.logger.error(error_msg)
            return dict_to_json({"error": error_msg})
    
    def get_domain_entities(self, domain_name: str, container_name: str, 
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
            self.base_manager.ensure_initialized()
            
            # Check if domain exists
            domain_query = """
            MATCH (c:Entity {name: $container_name, entityType: 'ProjectContainer'})
            MATCH (d:Entity {name: $domain_name, entityType: 'Domain'})-[:PART_OF]->(c)
            RETURN d
            """
            
            # Use safe_execute_read_query for validation (read-only operation)
            domain_records = self.base_manager.safe_execute_read_query(
                domain_query,
                {"container_name": container_name, "domain_name": domain_name}
            )
            
            if not domain_records or len(domain_records) == 0:
                return dict_to_json({
                    "error": f"Domain '{domain_name}' not found in container '{container_name}'"
                })
            
            # Build query based on entity_type
            if entity_type:
                query = """
                MATCH (c:Entity {name: $container_name, entityType: 'ProjectContainer'})
                MATCH (d:Entity {name: $domain_name, entityType: 'Domain'})-[:PART_OF]->(c)
                MATCH (e:Entity)-[:BELONGS_TO]->(d)
                WHERE e.entityType = $entity_type
                RETURN e
                ORDER BY e.name
                """
                
                params = {
                    "container_name": container_name, 
                    "domain_name": domain_name,
                    "entity_type": entity_type
                }
            else:
                query = """
                MATCH (c:Entity {name: $container_name, entityType: 'ProjectContainer'})
                MATCH (d:Entity {name: $domain_name, entityType: 'Domain'})-[:PART_OF]->(c)
                MATCH (e:Entity)-[:BELONGS_TO]->(d)
                RETURN e
                ORDER BY e.name
                """
                
                params = {"container_name": container_name, "domain_name": domain_name}
            
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
            
            return dict_to_json({
                "domain": domain_name,
                "container": container_name,
                "entity_count": len(entities),
                "entity_type_filter": entity_type,
                "entities": entities
            })
                
        except Exception as e:
            error_msg = f"Error retrieving domain entities: {str(e)}"
            self.logger.error(error_msg)
            return dict_to_json({"error": error_msg})
    
    def create_domain_relationship(self, from_domain: str, to_domain: str, 
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
            self.base_manager.ensure_initialized()
            
            # Check if domains exist in container
            domains_query = """
            MATCH (c:Entity {name: $container_name, entityType: 'ProjectContainer'})
            MATCH (from:Entity {name: $from_domain, entityType: 'Domain'})-[:PART_OF]->(c)
            MATCH (to:Entity {name: $to_domain, entityType: 'Domain'})-[:PART_OF]->(c)
            RETURN from, to
            """
            
            # Use safe_execute_read_query for validation (read-only operation)
            domains_records = self.base_manager.safe_execute_read_query(
                domains_query,
                {"container_name": container_name, "from_domain": from_domain, "to_domain": to_domain}
            )
            
            if not domains_records or len(domains_records) == 0:
                return dict_to_json({
                    "error": f"One or both domains not found in container '{container_name}'"
                })
            
            # Check if relationship already exists
            check_query = """
            MATCH (c:Entity {name: $container_name, entityType: 'ProjectContainer'})
            MATCH (from:Entity {name: $from_domain, entityType: 'Domain'})-[:PART_OF]->(c)
            MATCH (to:Entity {name: $to_domain, entityType: 'Domain'})-[:PART_OF]->(c)
            MATCH (from)-[r]->(to)
            WHERE type(r) = $relation_type
            RETURN r
            """
            
            # Use safe_execute_read_query for validation (read-only operation)
            check_records = self.base_manager.safe_execute_read_query(
                check_query,
                {
                    "container_name": container_name, 
                    "from_domain": from_domain, 
                    "to_domain": to_domain,
                    "relation_type": relation_type
                }
            )
            
            if check_records and len(check_records) > 0:
                return dict_to_json({
                    "status": "success",
                    "message": f"Relationship of type '{relation_type}' already exists between domains '{from_domain}' and '{to_domain}'"
                })
            
            # Prepare relationship properties
            relation_props = properties or {}
            relation_props["created"] = time.time()
            relation_props["domain"] = "project"
            
            # Create dynamic relationship
            create_query = f"""
            MATCH (c:Entity {{name: $container_name, entityType: 'ProjectContainer'}})
            MATCH (from:Entity {{name: $from_domain, entityType: 'Domain'}})-[:PART_OF]->(c)
            MATCH (to:Entity {{name: $to_domain, entityType: 'Domain'}})-[:PART_OF]->(c)
            CREATE (from)-[r:{relation_type} $properties]->(to)
            RETURN r
            """
            
            # Use safe_execute_write_query for validation (write operation)
            create_records = self.base_manager.safe_execute_write_query(
                create_query,
                {
                    "container_name": container_name, 
                    "from_domain": from_domain, 
                    "to_domain": to_domain,
                    "properties": relation_props
                }
            )
            
            if not create_records or len(create_records) == 0:
                return dict_to_json({
                    "error": f"Failed to create relationship between domains '{from_domain}' and '{to_domain}'"
                })
            
            return dict_to_json({
                "status": "success",
                "message": f"Relationship of type '{relation_type}' created between domains '{from_domain}' and '{to_domain}'"
            })
                
        except Exception as e:
            error_msg = f"Error creating domain relationship: {str(e)}"
            self.logger.error(error_msg)
            return dict_to_json({"error": error_msg}) 