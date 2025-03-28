"""
Project Memory System for MCP Graph Memory.
Provides specialized components for managing project-related knowledge.
"""

from typing import Any, Dict, List, Optional, Union
import time
import json

from src.graph_memory.base_manager import BaseManager
from src.project_memory.domain_manager import DomainManager
from src.project_memory.component_manager import ComponentManager
from src.project_memory.dependency_manager import DependencyManager
from src.project_memory.version_manager import VersionManager

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
        
        # Logger for the project memory manager
        self.logger = self.base_manager.logger
    
    # ============================================================================
    # Domain management methods
    # ============================================================================
    
    def create_domain(self, name: str, container_name: str, 
                    description: Optional[str] = None,
                    properties: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Create a new domain within a project container.
        
        Args:
            name: Name of the domain
            container_name: Name of the project container
            description: Optional description of the domain
            properties: Optional additional properties
            
        Returns:
            Dictionary with the created domain
        """
        result = self.domain_manager.create_domain(
            name, container_name, description, properties
        )
        return json.loads(result)
    
    def get_domain(self, name: str, container_name: str) -> Dict[str, Any]:
        """
        Retrieve a domain from a project container.
        
        Args:
            name: Name of the domain
            container_name: Name of the project container
            
        Returns:
            Dictionary with the domain details
        """
        result = self.domain_manager.get_domain(name, container_name)
        return json.loads(result)
    
    def update_domain(self, name: str, container_name: str, 
                    updates: Dict[str, Any]) -> Dict[str, Any]:
        """
        Update a domain's properties.
        
        Args:
            name: Name of the domain
            container_name: Name of the project container
            updates: Dictionary of properties to update
            
        Returns:
            Dictionary with the updated domain
        """
        result = self.domain_manager.update_domain(name, container_name, updates)
        return json.loads(result)
    
    def delete_domain(self, name: str, container_name: str, 
                    delete_components: bool = False) -> Dict[str, Any]:
        """
        Delete a domain from a project container.
        
        Args:
            name: Name of the domain
            container_name: Name of the project container
            delete_components: If True, delete all components belonging to the domain
            
        Returns:
            Dictionary with the deletion result
        """
        result = self.domain_manager.delete_domain(
            name, container_name, delete_components
        )
        return json.loads(result)
    
    def list_domains(self, container_name: str, sort_by: str = "name", 
                   limit: int = 100) -> Dict[str, Any]:
        """
        List all domains in a project container.
        
        Args:
            container_name: Name of the project container
            sort_by: Field to sort by ('name', 'created', or 'lastUpdated')
            limit: Maximum number of domains to return
            
        Returns:
            Dictionary with the list of domains
        """
        result = self.domain_manager.list_domains(container_name, sort_by, limit)
        return json.loads(result)
    
    def add_entity_to_domain(self, domain_name: str, container_name: str, 
                          entity_name: str) -> Dict[str, Any]:
        """
        Add an entity to a domain.
        
        Args:
            domain_name: Name of the domain
            container_name: Name of the project container
            entity_name: Name of the entity to add
            
        Returns:
            Dictionary with the result
        """
        result = self.domain_manager.add_entity_to_domain(
            domain_name, container_name, entity_name
        )
        return json.loads(result)
    
    def remove_entity_from_domain(self, domain_name: str, container_name: str, 
                               entity_name: str) -> Dict[str, Any]:
        """
        Remove an entity from a domain.
        
        Args:
            domain_name: Name of the domain
            container_name: Name of the project container
            entity_name: Name of the entity to remove
            
        Returns:
            Dictionary with the result
        """
        result = self.domain_manager.remove_entity_from_domain(
            domain_name, container_name, entity_name
        )
        return json.loads(result)
    
    def get_domain_entities(self, domain_name: str, container_name: str, 
                         entity_type: Optional[str] = None) -> Dict[str, Any]:
        """
        Get all entities in a domain.
        
        Args:
            domain_name: Name of the domain
            container_name: Name of the project container
            entity_type: Optional entity type to filter by
            
        Returns:
            Dictionary with the entities
        """
        result = self.domain_manager.get_domain_entities(
            domain_name, container_name, entity_type
        )
        return json.loads(result)
    
    def create_domain_relationship(self, from_domain: str, to_domain: str, 
                                container_name: str, relation_type: str,
                                properties: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Create a relationship between two domains.
        
        Args:
            from_domain: Name of the source domain
            to_domain: Name of the target domain
            container_name: Name of the project container
            relation_type: Type of relationship
            properties: Optional properties for the relationship
            
        Returns:
            Dictionary with the result
        """
        result = self.domain_manager.create_domain_relationship(
            from_domain, to_domain, container_name, relation_type, properties
        )
        return json.loads(result)
    
    # ============================================================================
    # Component management methods
    # ============================================================================
    
    def create_component(self, name: str, component_type: str,
                      domain_name: str, container_name: str,
                      description: Optional[str] = None,
                      content: Optional[str] = None,
                      metadata: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
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
            Dictionary with the created component
        """
        result = self.component_manager.create_component(
            name, component_type, domain_name, container_name,
            description, content, metadata
        )
        return json.loads(result)
    
    def get_component(self, name: str, domain_name: str, 
                    container_name: str) -> Dict[str, Any]:
        """
        Retrieve a component from a domain.
        
        Args:
            name: Name of the component
            domain_name: Name of the domain
            container_name: Name of the project container
            
        Returns:
            Dictionary with the component details
        """
        result = self.component_manager.get_component(
            name, domain_name, container_name
        )
        return json.loads(result)
    
    def update_component(self, name: str, domain_name: str, 
                      container_name: str, updates: Dict[str, Any]) -> Dict[str, Any]:
        """
        Update a component's properties.
        
        Args:
            name: Name of the component
            domain_name: Name of the domain
            container_name: Name of the project container
            updates: Dictionary of properties to update
            
        Returns:
            Dictionary with the updated component
        """
        result = self.component_manager.update_component(
            name, domain_name, container_name, updates
        )
        return json.loads(result)
    
    def delete_component(self, name: str, domain_name: str, 
                      container_name: str) -> Dict[str, Any]:
        """
        Delete a component from a domain.
        
        Args:
            name: Name of the component
            domain_name: Name of the domain
            container_name: Name of the project container
            
        Returns:
            Dictionary with the deletion result
        """
        result = self.component_manager.delete_component(
            name, domain_name, container_name
        )
        return json.loads(result)
    
    def list_components(self, domain_name: str, container_name: str, 
                     component_type: Optional[str] = None,
                     sort_by: str = "name", limit: int = 100) -> Dict[str, Any]:
        """
        List all components in a domain.
        
        Args:
            domain_name: Name of the domain
            container_name: Name of the project container
            component_type: Optional component type to filter by
            sort_by: Field to sort by ('name', 'created', or 'lastUpdated')
            limit: Maximum number of components to return
            
        Returns:
            Dictionary with the list of components
        """
        result = self.component_manager.list_components(
            domain_name, container_name, component_type, sort_by, limit
        )
        return json.loads(result)
    
    def create_component_relationship(self, from_component: str, to_component: str, 
                                   domain_name: str, container_name: str,
                                   relation_type: str,
                                   properties: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
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
            Dictionary with the result
        """
        result = self.component_manager.create_component_relationship(
            from_component, to_component, domain_name, container_name,
            relation_type, properties
        )
        return json.loads(result)
    
    # ============================================================================
    # Dependency management methods
    # ============================================================================
    
    def create_dependency(self, from_component: str, to_component: str,
                       domain_name: str, container_name: str,
                       dependency_type: str,
                       properties: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
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
            Dictionary with the result
        """
        result = self.dependency_manager.create_dependency(
            from_component, to_component, domain_name, container_name,
            dependency_type, properties
        )
        return json.loads(result)
    
    def get_dependencies(self, component_name: str, domain_name: str, 
                      container_name: str, direction: str = "outgoing",
                      dependency_type: Optional[str] = None) -> Dict[str, Any]:
        """
        Get dependencies for a component.
        
        Args:
            component_name: Name of the component
            domain_name: Name of the domain
            container_name: Name of the project container
            direction: Direction of dependencies ('outgoing', 'incoming', or 'both')
            dependency_type: Optional dependency type to filter by
            
        Returns:
            Dictionary with the dependencies
        """
        result = self.dependency_manager.get_dependencies(
            component_name, domain_name, container_name, direction, dependency_type
        )
        return json.loads(result)
    
    def delete_dependency(self, from_component: str, to_component: str,
                       domain_name: str, container_name: str,
                       dependency_type: str) -> Dict[str, Any]:
        """
        Delete a dependency relationship between components.
        
        Args:
            from_component: Name of the source component
            to_component: Name of the target component
            domain_name: Name of the domain
            container_name: Name of the project container
            dependency_type: Type of dependency to delete
            
        Returns:
            Dictionary with the result
        """
        result = self.dependency_manager.delete_dependency(
            from_component, to_component, domain_name, container_name, dependency_type
        )
        return json.loads(result)
    
    def analyze_dependency_graph(self, domain_name: str, 
                              container_name: str) -> Dict[str, Any]:
        """
        Analyze the dependency graph for a domain.
        
        Args:
            domain_name: Name of the domain
            container_name: Name of the project container
            
        Returns:
            Dictionary with the dependency analysis
        """
        result = self.dependency_manager.analyze_dependency_graph(
            domain_name, container_name
        )
        return json.loads(result)
    
    def find_path(self, from_component: str, to_component: str,
               domain_name: str, container_name: str,
               max_depth: int = 5) -> Dict[str, Any]:
        """
        Find dependency paths between two components.
        
        Args:
            from_component: Name of the source component
            to_component: Name of the target component
            domain_name: Name of the domain
            container_name: Name of the project container
            max_depth: Maximum path depth to search
            
        Returns:
            Dictionary with the dependency paths
        """
        result = self.dependency_manager.find_path(
            from_component, to_component, domain_name, container_name, max_depth
        )
        return json.loads(result)
    
    def import_dependencies_from_code(self, dependencies: List[Dict[str, Any]],
                                  domain_name: str, 
                                  container_name: str) -> Dict[str, Any]:
        """
        Import dependencies detected from code analysis.
        
        Args:
            dependencies: List of dependencies, each with from_component, to_component, and dependency_type
            domain_name: Name of the domain
            container_name: Name of the project container
            
        Returns:
            Dictionary with the import result
        """
        result = self.dependency_manager.import_dependencies_from_code(
            dependencies, domain_name, container_name
        )
        return json.loads(result)
    
    # ============================================================================
    # Version management methods
    # ============================================================================
    
    def create_version(self, component_name: str, domain_name: str,
                    container_name: str, version_number: str,
                    commit_hash: Optional[str] = None,
                    content: Optional[str] = None,
                    changes: Optional[str] = None,
                    metadata: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
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
            Dictionary with the created version
        """
        result = self.version_manager.create_version(
            component_name, domain_name, container_name, version_number,
            commit_hash, content, changes, metadata
        )
        return json.loads(result)
    
    def get_version(self, component_name: str, domain_name: str,
                 container_name: str, 
                 version_number: Optional[str] = None) -> Dict[str, Any]:
        """
        Get a specific version of a component.
        
        Args:
            component_name: Name of the component
            domain_name: Name of the domain
            container_name: Name of the project container
            version_number: Optional version number (latest if not specified)
            
        Returns:
            Dictionary with the version details
        """
        result = self.version_manager.get_version(
            component_name, domain_name, container_name, version_number
        )
        return json.loads(result)
    
    def list_versions(self, component_name: str, domain_name: str,
                   container_name: str, limit: int = 10) -> Dict[str, Any]:
        """
        List all versions of a component.
        
        Args:
            component_name: Name of the component
            domain_name: Name of the domain
            container_name: Name of the project container
            limit: Maximum number of versions to return
            
        Returns:
            Dictionary with the list of versions
        """
        result = self.version_manager.list_versions(
            component_name, domain_name, container_name, limit
        )
        return json.loads(result)
    
    def get_version_history(self, component_name: str, domain_name: str,
                         container_name: str,
                         include_content: bool = False) -> Dict[str, Any]:
        """
        Get the version history of a component with supersedes relationships.
        
        Args:
            component_name: Name of the component
            domain_name: Name of the domain
            container_name: Name of the project container
            include_content: Whether to include content in the version history
            
        Returns:
            Dictionary with the version history
        """
        result = self.version_manager.get_version_history(
            component_name, domain_name, container_name, include_content
        )
        return json.loads(result)
    
    def compare_versions(self, component_name: str, domain_name: str,
                      container_name: str, version1: str, 
                      version2: str) -> Dict[str, Any]:
        """
        Compare two versions of a component.
        
        Args:
            component_name: Name of the component
            domain_name: Name of the domain
            container_name: Name of the project container
            version1: First version number
            version2: Second version number
            
        Returns:
            Dictionary with the comparison result
        """
        result = self.version_manager.compare_versions(
            component_name, domain_name, container_name, version1, version2
        )
        return json.loads(result)
    
    def tag_version(self, component_name: str, domain_name: str,
                 container_name: str, version_number: str,
                 tag_name: str, tag_description: Optional[str] = None) -> Dict[str, Any]:
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
            Dictionary with the result
        """
        result = self.version_manager.tag_version(
            component_name, domain_name, container_name, 
            version_number, tag_name, tag_description
        )
        return json.loads(result)
    
    def sync_with_version_control(self, component_name: str, domain_name: str,
                              container_name: str,
                              commit_data: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Synchronize component versions with version control system data.
        
        Args:
            component_name: Name of the component
            domain_name: Name of the domain
            container_name: Name of the project container
            commit_data: List of commit data, each with hash, version, date, author, message, and content
            
        Returns:
            Dictionary with the sync result
        """
        result = self.version_manager.sync_with_version_control(
            component_name, domain_name, container_name, commit_data
        )
        return json.loads(result)
    
    # ============================================================================
    # Project container management methods
    # ============================================================================
    
    def create_project_container(self, name: str, 
                              description: Optional[str] = None,
                              properties: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Create a new project container.
        
        Args:
            name: Name of the project container
            description: Optional description of the project
            properties: Optional additional properties
            
        Returns:
            Dictionary with the created project container
        """
        # Use the entity manager to create an entity of type 'ProjectContainer'
        timestamp = time.time()
        
        # Generate unique ID
        container_id = f"proj_{int(timestamp)}_{hash(name) % 10000:04d}"
        
        container_entity = {
            "id": container_id,
            "name": name,
            "entityType": "ProjectContainer",
            "created": timestamp,
            "lastUpdated": timestamp
        }
        
        if description:
            container_entity["description"] = description
            
        if properties:
            for key, value in properties.items():
                if key not in container_entity:
                    container_entity[key] = value
        
        # Create entity using the underlying entity manager from base manager
        entity_json = self.base_manager.entity_manager.create_entity(container_entity)
        entity_result = json.loads(entity_json)
        
        if "error" in entity_result:
            return {"error": entity_result["error"]}
        
        return {
            "status": "success",
            "message": f"Project container '{name}' created successfully",
            "container": entity_result.get("entity", {})
        }
    
    def get_project_container(self, name: str) -> Dict[str, Any]:
        """
        Retrieve a project container by name.
        
        Args:
            name: Name of the project container
            
        Returns:
            Dictionary with the project container details
        """
        # Get entity by name and type
        entity_json = self.base_manager.entity_manager.get_entity_by_name_and_type(
            name, "ProjectContainer"
        )
        entity_result = json.loads(entity_json)
        
        if "error" in entity_result:
            return {"error": entity_result["error"]}
        
        # Get domain and component counts
        container_id = entity_result.get("entity", {}).get("id")
        
        if not container_id:
            return {"error": f"Project container '{name}' found but has no ID"}
        
        # Get domain count
        domain_query = """
        MATCH (c:Entity {id: $container_id})
        MATCH (d:Entity {entityType: 'Domain'})-[:PART_OF]->(c)
        RETURN count(d) as domain_count
        """
        
        domain_records, _ = self.base_manager.safe_execute_query(
            domain_query,
            {"container_id": container_id}
        )
        
        domain_count = 0
        if domain_records and len(domain_records) > 0:
            domain_count = domain_records[0]["domain_count"]
        
        # Get component count
        component_query = """
        MATCH (c:Entity {id: $container_id})
        MATCH (d:Entity {entityType: 'Domain'})-[:PART_OF]->(c)
        MATCH (comp:Entity)-[:BELONGS_TO]->(d)
        RETURN count(comp) as component_count
        """
        
        component_records, _ = self.base_manager.safe_execute_query(
            component_query,
            {"container_id": container_id}
        )
        
        component_count = 0
        if component_records and len(component_records) > 0:
            component_count = component_records[0]["component_count"]
        
        # Add counts to result
        container = entity_result.get("entity", {})
        container["domain_count"] = domain_count
        container["component_count"] = component_count
        
        return {"container": container}
    
    def update_project_container(self, name: str, 
                              updates: Dict[str, Any]) -> Dict[str, Any]:
        """
        Update a project container's properties.
        
        Args:
            name: Name of the project container
            updates: Dictionary of properties to update
            
        Returns:
            Dictionary with the updated project container
        """
        # Get entity by name and type
        entity_json = self.base_manager.entity_manager.get_entity_by_name_and_type(
            name, "ProjectContainer"
        )
        entity_result = json.loads(entity_json)
        
        if "error" in entity_result:
            return {"error": entity_result["error"]}
        
        entity = entity_result.get("entity", {})
        entity_id = entity.get("id")
        
        if not entity_id:
            return {"error": f"Project container '{name}' found but has no ID"}
        
        # Update the entity
        updates["lastUpdated"] = time.time()
        
        # Use entity manager to update
        update_json = self.base_manager.entity_manager.update_entity(entity_id, updates)
        update_result = json.loads(update_json)
        
        if "error" in update_result:
            return {"error": update_result["error"]}
        
        return {
            "status": "success",
            "message": f"Project container '{name}' updated successfully",
            "container": update_result.get("entity", {})
        }
    
    def delete_project_container(self, name: str, 
                              delete_contents: bool = False) -> Dict[str, Any]:
        """
        Delete a project container.
        
        Args:
            name: Name of the project container
            delete_contents: If True, delete all domains and components in the container
            
        Returns:
            Dictionary with the deletion result
        """
        # Get entity by name and type
        entity_json = self.base_manager.entity_manager.get_entity_by_name_and_type(
            name, "ProjectContainer"
        )
        entity_result = json.loads(entity_json)
        
        if "error" in entity_result:
            return {"error": entity_result["error"]}
        
        entity = entity_result.get("entity", {})
        entity_id = entity.get("id")
        
        if not entity_id:
            return {"error": f"Project container '{name}' found but has no ID"}
        
        if delete_contents:
            # Delete all components in domains
            delete_components_query = """
            MATCH (c:Entity {id: $container_id})
            MATCH (d:Entity)-[:PART_OF]->(c)
            MATCH (comp:Entity)-[:BELONGS_TO]->(d)
            
            // Delete component relationships
            OPTIONAL MATCH (comp)-[r]-()
            DELETE r
            
            // Delete component observations
            OPTIONAL MATCH (comp)-[:HAS_OBSERVATION]->(o:Observation)
            DELETE o
            
            // Delete components
            DELETE comp
            
            RETURN count(comp) as deleted_count
            """
            
            comp_records, _ = self.base_manager.safe_execute_query(
                delete_components_query,
                {"container_id": entity_id}
            )
            
            deleted_components = 0
            if comp_records and len(comp_records) > 0:
                deleted_components = comp_records[0]["deleted_count"]
            
            # Delete all domains
            delete_domains_query = """
            MATCH (c:Entity {id: $container_id})
            MATCH (d:Entity)-[:PART_OF]->(c)
            
            // Delete domain relationships
            OPTIONAL MATCH (d)-[r]-()
            DELETE r
            
            // Delete domains
            DELETE d
            
            RETURN count(d) as deleted_count
            """
            
            domain_records, _ = self.base_manager.safe_execute_query(
                delete_domains_query,
                {"container_id": entity_id}
            )
            
            deleted_domains = 0
            if domain_records and len(domain_records) > 0:
                deleted_domains = domain_records[0]["deleted_count"]
        
        # Delete the container entity
        delete_json = self.base_manager.entity_manager.delete_entity(entity_id)
        delete_result = json.loads(delete_json)
        
        if "error" in delete_result:
            return {"error": delete_result["error"]}
        
        result = {
            "status": "success",
            "message": f"Project container '{name}' deleted successfully"
        }
        
        if delete_contents:
            result["deleted_domains"] = deleted_domains
            result["deleted_components"] = deleted_components
        
        return result
    
    def list_project_containers(self, sort_by: str = "name", 
                             limit: int = 100) -> Dict[str, Any]:
        """
        List all project containers.
        
        Args:
            sort_by: Field to sort by ('name', 'created', or 'lastUpdated')
            limit: Maximum number of containers to return
            
        Returns:
            Dictionary with the list of project containers
        """
        # Validate sort_by
        valid_sort_fields = ["name", "created", "lastUpdated"]
        if sort_by not in valid_sort_fields:
            sort_by = "name"
        
        # Query for containers with domain and component counts
        query = f"""
        MATCH (c:Entity {{entityType: 'ProjectContainer'}})
        OPTIONAL MATCH (d:Entity {{entityType: 'Domain'}})-[:PART_OF]->(c)
        WITH c, count(d) as domain_count
        
        OPTIONAL MATCH (c)<-[:PART_OF]-(d:Entity {{entityType: 'Domain'}})<-[:BELONGS_TO]-(comp:Entity)
        WITH c, domain_count, count(comp) as component_count
        
        RETURN c, domain_count, component_count
        ORDER BY c.{sort_by}
        LIMIT $limit
        """
        
        records, _ = self.base_manager.safe_execute_query(
            query,
            {"limit": limit}
        )
        
        containers = []
        if records:
            for record in records:
                container = dict(record["c"].items())
                container["domain_count"] = record["domain_count"]
                container["component_count"] = record["component_count"]
                containers.append(container)
        
        return {
            "containers": containers,
            "count": len(containers)
        }
    
    def get_project_status(self, container_name: str) -> Dict[str, Any]:
        """
        Get a summary of the project container status, including domains and components.
        
        Args:
            container_name: Name of the project container
            
        Returns:
            Dictionary with the project status
        """
        # Get container details
        container_result = self.get_project_container(container_name)
        
        if "error" in container_result:
            return {"error": container_result["error"]}
        
        container = container_result.get("container", {})
        
        # Get domains
        domains_result = self.list_domains(container_name)
        
        if "error" in domains_result:
            return {"error": domains_result["error"]}
        
        domains = domains_result.get("domains", [])
        
        # Get component counts by type
        component_query = """
        MATCH (c:Entity {name: $container_name, entityType: 'ProjectContainer'})
        MATCH (d:Entity {entityType: 'Domain'})-[:PART_OF]->(c)
        MATCH (comp:Entity)-[:BELONGS_TO]->(d)
        RETURN comp.entityType as type, count(comp) as count
        """
        
        component_records, _ = self.base_manager.safe_execute_query(
            component_query,
            {"container_name": container_name}
        )
        
        component_types = {}
        if component_records:
            for record in component_records:
                entity_type = record["type"]
                count = record["count"]
                if entity_type:
                    component_types[entity_type] = count
        
        # Get dependency counts
        dependency_query = """
        MATCH (c:Entity {name: $container_name, entityType: 'ProjectContainer'})
        MATCH (d:Entity {entityType: 'Domain'})-[:PART_OF]->(c)
        MATCH (from:Entity)-[:BELONGS_TO]->(d)
        MATCH (to:Entity)-[:BELONGS_TO]->(d)
        MATCH (from)-[r]->(to)
        WHERE type(r) IN ['DEPENDS_ON', 'IMPORTS', 'USES', 'EXTENDS', 'IMPLEMENTS', 'CALLS', 'REFERENCES']
        RETURN type(r) as type, count(r) as count
        """
        
        dependency_records, _ = self.base_manager.safe_execute_query(
            dependency_query,
            {"container_name": container_name}
        )
        
        dependency_types = {}
        if dependency_records:
            for record in dependency_records:
                rel_type = record["type"]
                count = record["count"]
                dependency_types[rel_type] = count
        
        # Build status report
        status = {
            "container": container,
            "domains": domains,
            "component_counts": component_types,
            "dependency_counts": dependency_types,
            "total_domains": len(domains),
            "total_components": sum(component_types.values()) if component_types else 0,
            "total_dependencies": sum(dependency_types.values()) if dependency_types else 0
        }
        
        return status
