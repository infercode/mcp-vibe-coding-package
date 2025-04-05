from typing import Any, Dict, List, Optional, Union
import time
import json

from src.utils import dict_to_json, generate_id
from src.graph_memory.base_manager import BaseManager

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
    
    def create_project_container(self, project_data: Dict[str, Any]) -> str:
        """
        Create a new project container.
        
        Args:
            project_data: Dictionary containing project information
                - name: Required. The name of the project container (unique identifier)
                - description: Optional. Description of the project
                - metadata: Optional. Additional metadata for the project
                - tags: Optional. List of tags for categorizing the project
            
        Returns:
            JSON string with the created container
        """
        try:
            self.base_manager.ensure_initialized()
            
            # Validate required fields
            if "name" not in project_data:
                return dict_to_json({
                    "error": "Missing required field: name"
                })
                
            name = project_data["name"]
            
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
            timestamp = time.time()
            
            # Prepare container properties
            container_properties = {
                "id": container_id,
                "name": name,
                "entityType": "ProjectContainer",
                "domain": "project",
                "created": timestamp,
                "lastUpdated": timestamp,
                "status": "active"
            }
            
            # Add description if provided
            if "description" in project_data and project_data["description"]:
                container_properties["description"] = project_data["description"]
                
            # Add metadata if provided
            if "metadata" in project_data and isinstance(project_data["metadata"], dict):
                for key, value in project_data["metadata"].items():
                    container_properties[key] = value
                    
            # Add tags if provided
            if "tags" in project_data and isinstance(project_data["tags"], list):
                container_properties["tags"] = project_data["tags"]
            
            # Create container
            create_query = """
            CREATE (c:Entity $properties)
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
            
            return dict_to_json({
                "status": "success",
                "message": f"Project container '{name}' created successfully",
                "container": container
            })
                
        except Exception as e:
            error_msg = f"Error creating project container: {str(e)}"
            self.logger.error(error_msg)
            return dict_to_json({"error": error_msg})
    
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
                return dict_to_json({
                    "error": f"Project container '{name}' not found"
                })
            
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
            
            return dict_to_json({
                "container": container
            })
                
        except Exception as e:
            error_msg = f"Error retrieving project container: {str(e)}"
            self.logger.error(error_msg)
            return dict_to_json({"error": error_msg})
    
    def update_project_container(self, name: str, updates: Dict[str, Any]) -> str:
        """
        Update a project container's properties.
        
        Args:
            name: Name of the project container
            updates: Dictionary of properties to update
            
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
                return dict_to_json({
                    "error": f"Project container '{name}' not found"
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
                set_parts.append(f"c.{key} = ${key}")
            
            # Build update query
            update_query = f"""
            MATCH (c:Entity {{name: $name, entityType: 'ProjectContainer'}})
            SET {', '.join(set_parts)}
            RETURN c
            """
            
            # Add name to updates for the query
            params = {"name": name, **updates}
            
            # Use safe_execute_write_query for validation (this is a write operation)
            update_records = self.base_manager.safe_execute_write_query(
                update_query,
                params
            )
            
            if not update_records or len(update_records) == 0:
                return dict_to_json({
                    "error": "Failed to update project container"
                })
            
            # Return updated container
            container = dict(update_records[0]["c"].items())
            
            return dict_to_json({
                "status": "success",
                "message": f"Project container '{name}' updated successfully",
                "container": container
            })
                
        except Exception as e:
            error_msg = f"Error updating project container: {str(e)}"
            self.logger.error(error_msg)
            return dict_to_json({"error": error_msg})
    
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
                return dict_to_json({
                    "error": f"Project container '{name}' not found"
                })
            
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
                
                return dict_to_json({
                    "status": "success",
                    "message": f"Project container '{name}' and {deleted_count} components deleted successfully"
                })
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
                    return dict_to_json({
                        "error": f"Cannot delete project container '{name}' with {component_count} components. Set delete_contents=True to delete the container and its contents."
                    })
                
                # Delete the container (which has no components)
                delete_container_query = """
                MATCH (c:Entity {name: $name, entityType: 'ProjectContainer'})
                DELETE c
                """
                
                self.base_manager.safe_execute_write_query(
                    delete_container_query,
                    {"name": name}
                )
                
                return dict_to_json({
                    "status": "success",
                    "message": f"Project container '{name}' deleted successfully"
                })
                
        except Exception as e:
            error_msg = f"Error deleting project container: {str(e)}"
            self.logger.error(error_msg)
            return dict_to_json({"error": error_msg})
    
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
            
            return dict_to_json({
                "containers": containers,
                "count": len(containers)
            })
                
        except Exception as e:
            error_msg = f"Error listing project containers: {str(e)}"
            self.logger.error(error_msg)
            return dict_to_json({"error": error_msg})
    
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
                return dict_to_json({
                    "error": f"Project container '{container_name}' not found"
                })
            
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
                return dict_to_json({
                    "error": f"Entity '{entity_name}' not found"
                })
            
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
                return dict_to_json({
                    "status": "success",
                    "message": f"Entity '{entity_name}' is already in project container '{container_name}'"
                })
            
            # Add entity to container
            add_query = """
            MATCH (c:Entity {name: $container_name, entityType: 'ProjectContainer'})
            MATCH (e:Entity {name: $entity_name})
            CREATE (e)-[:PART_OF {created: $timestamp}]->(c)
            RETURN e
            """
            
            add_records = self.base_manager.safe_execute_write_query(
                add_query,
                {"container_name": container_name, "entity_name": entity_name, "timestamp": time.time()}
            )
            
            if not add_records or len(add_records) == 0:
                return dict_to_json({
                    "error": f"Failed to add entity '{entity_name}' to project container '{container_name}'"
                })
            
            return dict_to_json({
                "status": "success",
                "message": f"Entity '{entity_name}' added to project container '{container_name}'"
            })
                
        except Exception as e:
            error_msg = f"Error adding component to container: {str(e)}"
            self.logger.error(error_msg)
            return dict_to_json({"error": error_msg})
    
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
                return dict_to_json({
                    "error": f"Entity '{entity_name}' is not in project container '{container_name}'"
                })
            
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
            
            return dict_to_json({
                "status": "success",
                "message": f"Entity '{entity_name}' removed from project container '{container_name}'"
            })
                
        except Exception as e:
            error_msg = f"Error removing component from container: {str(e)}"
            self.logger.error(error_msg)
            return dict_to_json({"error": error_msg})
    
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
                return dict_to_json({
                    "error": f"Project container '{container_name}' not found"
                })
            
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
            
            return dict_to_json({
                "container": container_name,
                "component_count": len(components),
                "entity_type_filter": entity_type,
                "components": components
            })
                
        except Exception as e:
            error_msg = f"Error retrieving container components: {str(e)}"
            self.logger.error(error_msg)
            return dict_to_json({"error": error_msg})
    
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
                return dict_to_json({
                    "error": f"Invalid status '{status}'. Valid values are: {', '.join(valid_statuses)}"
                })
            
            # Update container status
            update_query = """
            MATCH (c:Entity {name: $container_name, entityType: 'ProjectContainer'})
            SET c.status = $status,
                c.lastUpdated = $timestamp
            RETURN c
            """
            
            update_records = self.base_manager.safe_execute_write_query(
                update_query,
                {"container_name": container_name, "status": status, "timestamp": time.time()}
            )
            
            if not update_records or len(update_records) == 0:
                return dict_to_json({
                    "error": f"Project container '{container_name}' not found"
                })
            
            # Return updated container
            container = dict(update_records[0]["c"].items())
            
            return dict_to_json({
                "status": "success",
                "message": f"Project container '{container_name}' status changed to '{status}'",
                "container": container
            })
                
        except Exception as e:
            error_msg = f"Error changing container status: {str(e)}"
            self.logger.error(error_msg)
            return dict_to_json({"error": error_msg})
    
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
                return dict_to_json({
                    "error": f"Project container '{container_name}' not found"
                })
            
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
            
            # Return stats
            return dict_to_json({
                "container": container,
                "total_entities": entity_count,
                "entity_types": entity_types,
                "total_relationships": relationship_count,
                "relationship_types": relationship_types,
                "total_observations": observation_count,
                "observation_types": observation_types
            })
                
        except Exception as e:
            error_msg = f"Error retrieving container stats: {str(e)}"
            self.logger.error(error_msg)
            return dict_to_json({"error": error_msg}) 