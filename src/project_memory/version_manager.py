from typing import Any, Dict, List, Optional, Union
import time
import json

from src.utils import dict_to_json, generate_id
from src.graph_memory.base_manager import BaseManager
from src.graph_memory.entity_manager import EntityManager

class VersionManager:
    """
    Manager for tracking component versions and changes in the project memory system.
    Handles versioning, history tracking, and change integration.
    """
    
    def __init__(self, base_manager: BaseManager):
        """
        Initialize the version manager.
        
        Args:
            base_manager: Base manager for graph operations
        """
        self.base_manager = base_manager
        self.logger = base_manager.logger
        self.entity_manager = EntityManager(base_manager)
    
    def create_version(self, 
                    component_name: str,
                    domain_name: str,
                    container_name: str,
                    version_number: str,
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
            self.base_manager.ensure_initialized()
            
            # Check if component exists in domain
            component_query = """
            MATCH (c:Entity {name: $container_name, entityType: 'ProjectContainer'})
            MATCH (d:Entity {name: $domain_name, entityType: 'Domain'})-[:PART_OF]->(c)
            MATCH (comp:Entity {name: $component_name})-[:BELONGS_TO]->(d)
            RETURN comp
            """
            
            component_records, _ = self.base_manager.safe_execute_query(
                component_query,
                {"container_name": container_name, "domain_name": domain_name, "component_name": component_name}
            )
            
            if not component_records or len(component_records) == 0:
                return dict_to_json({
                    "error": f"Component '{component_name}' not found in domain '{domain_name}'"
                })
            
            component_id = component_records[0]["comp"]["id"]
            component_type = component_records[0]["comp"]["entityType"]
            
            # Check if version already exists
            version_check_query = """
            MATCH (comp:Entity {id: $component_id})
            MATCH (v:Entity {entityType: 'Version'})
            MATCH (comp)-[:HAS_VERSION]->(v)
            WHERE v.versionNumber = $version_number
            RETURN v
            """
            
            version_records, _ = self.base_manager.safe_execute_query(
                version_check_query,
                {"component_id": component_id, "version_number": version_number}
            )
            
            if version_records and len(version_records) > 0:
                return dict_to_json({
                    "error": f"Version '{version_number}' already exists for component '{component_name}'"
                })
            
            # Generate version entity ID
            version_id = generate_id("ver")
            timestamp = time.time()
            
            # Prepare version entity
            version_entity = {
                "id": version_id,
                "name": f"{component_name}_v{version_number}",
                "entityType": "Version",
                "versionNumber": version_number,
                "componentId": component_id,
                "domain": "project",
                "created": timestamp,
                "lastUpdated": timestamp
            }
            
            if commit_hash:
                version_entity["commitHash"] = commit_hash
                
            if content:
                version_entity["content"] = content
                
            if changes:
                version_entity["changes"] = changes
                
            if metadata:
                for key, value in metadata.items():
                    if key not in version_entity:
                        version_entity[key] = value
            
            # Create version entity
            create_query = """
            CREATE (v:Entity $properties)
            RETURN v
            """
            
            create_records, _ = self.base_manager.safe_execute_query(
                create_query,
                {"properties": version_entity}
            )
            
            if not create_records or len(create_records) == 0:
                return dict_to_json({
                    "error": "Failed to create version entity"
                })
            
            # Link version to component
            link_query = """
            MATCH (comp:Entity {id: $component_id})
            MATCH (v:Entity {id: $version_id})
            CREATE (comp)-[:HAS_VERSION {created: $timestamp}]->(v)
            RETURN v
            """
            
            link_records, _ = self.base_manager.safe_execute_query(
                link_query,
                {"component_id": component_id, "version_id": version_id, "timestamp": timestamp}
            )
            
            if not link_records or len(link_records) == 0:
                # Attempt to clean up the created entity
                self.base_manager.safe_execute_query(
                    "MATCH (v:Entity {id: $version_id}) DELETE v",
                    {"version_id": version_id}
                )
                
                return dict_to_json({
                    "error": f"Failed to link version to component '{component_name}'"
                })
            
            # Find previous version to create chain
            prev_version_query = """
            MATCH (comp:Entity {id: $component_id})
            MATCH (comp)-[:HAS_VERSION]->(prev:Entity {entityType: 'Version'})
            WHERE prev.id <> $version_id
            RETURN prev
            ORDER BY prev.versionNumber DESC
            LIMIT 1
            """
            
            prev_records, _ = self.base_manager.safe_execute_query(
                prev_version_query,
                {"component_id": component_id, "version_id": version_id}
            )
            
            # If previous version exists, create SUPERSEDES relationship
            if prev_records and len(prev_records) > 0:
                prev_version_id = prev_records[0]["prev"]["id"]
                prev_version_number = prev_records[0]["prev"]["versionNumber"]
                
                supersedes_query = """
                MATCH (curr:Entity {id: $version_id})
                MATCH (prev:Entity {id: $prev_version_id})
                CREATE (curr)-[:SUPERSEDES {created: $timestamp}]->(prev)
                """
                
                self.base_manager.safe_execute_query(
                    supersedes_query,
                    {"version_id": version_id, "prev_version_id": prev_version_id, "timestamp": timestamp}
                )
            
            # Get created version
            version = dict(link_records[0]["v"].items())
            
            return dict_to_json({
                "status": "success",
                "message": f"Version '{version_number}' created for component '{component_name}'",
                "version": version
            })
                
        except Exception as e:
            error_msg = f"Error creating version: {str(e)}"
            self.logger.error(error_msg)
            return dict_to_json({"error": error_msg})
    
    def get_version(self, 
                 component_name: str,
                 domain_name: str,
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
            self.base_manager.ensure_initialized()
            
            # Check if component exists in domain
            component_query = """
            MATCH (c:Entity {name: $container_name, entityType: 'ProjectContainer'})
            MATCH (d:Entity {name: $domain_name, entityType: 'Domain'})-[:PART_OF]->(c)
            MATCH (comp:Entity {name: $component_name})-[:BELONGS_TO]->(d)
            RETURN comp
            """
            
            component_records, _ = self.base_manager.safe_execute_query(
                component_query,
                {"container_name": container_name, "domain_name": domain_name, "component_name": component_name}
            )
            
            if not component_records or len(component_records) == 0:
                return dict_to_json({
                    "error": f"Component '{component_name}' not found in domain '{domain_name}'"
                })
            
            component_id = component_records[0]["comp"]["id"]
            
            if version_number:
                # Get specific version
                version_query = """
                MATCH (comp:Entity {id: $component_id})
                MATCH (v:Entity {entityType: 'Version'})
                MATCH (comp)-[:HAS_VERSION]->(v)
                WHERE v.versionNumber = $version_number
                RETURN v
                """
                
                version_records, _ = self.base_manager.safe_execute_query(
                    version_query,
                    {"component_id": component_id, "version_number": version_number}
                )
                
                if not version_records or len(version_records) == 0:
                    return dict_to_json({
                        "error": f"Version '{version_number}' not found for component '{component_name}'"
                    })
                
                version = dict(version_records[0]["v"].items())
                
            else:
                # Get latest version
                latest_query = """
                MATCH (comp:Entity {id: $component_id})
                MATCH (v:Entity {entityType: 'Version'})
                MATCH (comp)-[:HAS_VERSION]->(v)
                RETURN v
                ORDER BY v.versionNumber DESC
                LIMIT 1
                """
                
                latest_records, _ = self.base_manager.safe_execute_query(
                    latest_query,
                    {"component_id": component_id}
                )
                
                if not latest_records or len(latest_records) == 0:
                    return dict_to_json({
                        "error": f"No versions found for component '{component_name}'"
                    })
                
                version = dict(latest_records[0]["v"].items())
            
            return dict_to_json({
                "component": component_name,
                "domain": domain_name,
                "container": container_name,
                "version": version
            })
                
        except Exception as e:
            error_msg = f"Error retrieving version: {str(e)}"
            self.logger.error(error_msg)
            return dict_to_json({"error": error_msg})
    
    def list_versions(self, 
                   component_name: str,
                   domain_name: str,
                   container_name: str,
                   limit: int = 10) -> str:
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
            self.base_manager.ensure_initialized()
            
            # Check if component exists in domain
            component_query = """
            MATCH (c:Entity {name: $container_name, entityType: 'ProjectContainer'})
            MATCH (d:Entity {name: $domain_name, entityType: 'Domain'})-[:PART_OF]->(c)
            MATCH (comp:Entity {name: $component_name})-[:BELONGS_TO]->(d)
            RETURN comp
            """
            
            component_records, _ = self.base_manager.safe_execute_query(
                component_query,
                {"container_name": container_name, "domain_name": domain_name, "component_name": component_name}
            )
            
            if not component_records or len(component_records) == 0:
                return dict_to_json({
                    "error": f"Component '{component_name}' not found in domain '{domain_name}'"
                })
            
            component_id = component_records[0]["comp"]["id"]
            
            # Get all versions
            versions_query = """
            MATCH (comp:Entity {id: $component_id})
            MATCH (v:Entity {entityType: 'Version'})
            MATCH (comp)-[:HAS_VERSION]->(v)
            RETURN v
            ORDER BY v.versionNumber DESC
            LIMIT $limit
            """
            
            version_records, _ = self.base_manager.safe_execute_query(
                versions_query,
                {"component_id": component_id, "limit": limit}
            )
            
            versions = []
            if version_records:
                for record in version_records:
                    version = dict(record["v"].items())
                    versions.append(version)
            
            return dict_to_json({
                "component": component_name,
                "domain": domain_name,
                "container": container_name,
                "version_count": len(versions),
                "versions": versions
            })
                
        except Exception as e:
            error_msg = f"Error listing versions: {str(e)}"
            self.logger.error(error_msg)
            return dict_to_json({"error": error_msg})
    
    def get_version_history(self, 
                         component_name: str,
                         domain_name: str,
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
            self.base_manager.ensure_initialized()
            
            # Check if component exists in domain
            component_query = """
            MATCH (c:Entity {name: $container_name, entityType: 'ProjectContainer'})
            MATCH (d:Entity {name: $domain_name, entityType: 'Domain'})-[:PART_OF]->(c)
            MATCH (comp:Entity {name: $component_name})-[:BELONGS_TO]->(d)
            RETURN comp
            """
            
            component_records, _ = self.base_manager.safe_execute_query(
                component_query,
                {"container_name": container_name, "domain_name": domain_name, "component_name": component_name}
            )
            
            if not component_records or len(component_records) == 0:
                return dict_to_json({
                    "error": f"Component '{component_name}' not found in domain '{domain_name}'"
                })
            
            component_id = component_records[0]["comp"]["id"]
            
            # Get the latest version
            latest_query = """
            MATCH (comp:Entity {id: $component_id})
            MATCH (v:Entity {entityType: 'Version'})
            MATCH (comp)-[:HAS_VERSION]->(v)
            WHERE NOT EXISTS((ANY)-[:SUPERSEDES]->(v))
            RETURN v
            """
            
            latest_records, _ = self.base_manager.safe_execute_query(
                latest_query,
                {"component_id": component_id}
            )
            
            if not latest_records or len(latest_records) == 0:
                return dict_to_json({
                    "error": f"No versions found for component '{component_name}'"
                })
            
            latest_version = dict(latest_records[0]["v"].items())
            latest_version_id = latest_version["id"]
            
            # If we don't need content, exclude it from the results
            content_clause = "" if include_content else "REMOVE v.content"
            
            # Get the version chain
            history_query = f"""
            MATCH (start:Entity {{id: $latest_version_id}})
            MATCH path = (start)-[:SUPERSEDES*0..]->(v:Entity)
            WITH v, path
            ORDER BY length(path)
            WITH collect(v) as versions
            UNWIND versions as v
            {content_clause}
            RETURN v
            """
            
            history_records, _ = self.base_manager.safe_execute_query(
                history_query,
                {"latest_version_id": latest_version_id}
            )
            
            versions = []
            if history_records:
                for record in history_records:
                    version = dict(record["v"].items())
                    versions.append(version)
            
            return dict_to_json({
                "component": component_name,
                "domain": domain_name,
                "container": container_name,
                "latest_version": latest_version["versionNumber"],
                "version_count": len(versions),
                "versions": versions
            })
                
        except Exception as e:
            error_msg = f"Error retrieving version history: {str(e)}"
            self.logger.error(error_msg)
            return dict_to_json({"error": error_msg})
    
    def compare_versions(self, 
                      component_name: str,
                      domain_name: str,
                      container_name: str,
                      version1: str,
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
            self.base_manager.ensure_initialized()
            
            # Check if component exists in domain
            component_query = """
            MATCH (c:Entity {name: $container_name, entityType: 'ProjectContainer'})
            MATCH (d:Entity {name: $domain_name, entityType: 'Domain'})-[:PART_OF]->(c)
            MATCH (comp:Entity {name: $component_name})-[:BELONGS_TO]->(d)
            RETURN comp
            """
            
            component_records, _ = self.base_manager.safe_execute_query(
                component_query,
                {"container_name": container_name, "domain_name": domain_name, "component_name": component_name}
            )
            
            if not component_records or len(component_records) == 0:
                return dict_to_json({
                    "error": f"Component '{component_name}' not found in domain '{domain_name}'"
                })
            
            component_id = component_records[0]["comp"]["id"]
            
            # Get the two versions
            versions_query = """
            MATCH (comp:Entity {id: $component_id})
            MATCH (v:Entity {entityType: 'Version'})
            MATCH (comp)-[:HAS_VERSION]->(v)
            WHERE v.versionNumber IN [$version1, $version2]
            RETURN v
            """
            
            version_records, _ = self.base_manager.safe_execute_query(
                versions_query,
                {"component_id": component_id, "version1": version1, "version2": version2}
            )
            
            if not version_records or len(version_records) < 2:
                return dict_to_json({
                    "error": f"One or both versions not found for component '{component_name}'"
                })
            
            # Extract version data
            version_data = {}
            for record in version_records:
                v = dict(record["v"].items())
                version_data[v["versionNumber"]] = v
            
            v1 = version_data.get(version1)
            v2 = version_data.get(version2)
            
            if not v1 or not v2:
                return dict_to_json({
                    "error": f"One or both versions not found for component '{component_name}'"
                })
            
            # Determine which version is newer
            newer_version = v1 if v1["created"] > v2["created"] else v2
            older_version = v2 if newer_version == v1 else v1
            
            # Check if there's a direct supersedes path
            path_query = """
            MATCH (newer:Entity {id: $newer_id})
            MATCH (older:Entity {id: $older_id})
            MATCH path = (newer)-[:SUPERSEDES*]->(older)
            RETURN length(path) as distance
            """
            
            path_records, _ = self.base_manager.safe_execute_query(
                path_query,
                {"newer_id": newer_version["id"], "older_id": older_version["id"]}
            )
            
            direct_path_exists = False
            distance = None
            
            if path_records and len(path_records) > 0:
                direct_path_exists = True
                distance = path_records[0]["distance"]
            
            # Build comparison result
            comparison = {
                "component": component_name,
                "domain": domain_name,
                "container": container_name,
                "version1": {
                    "versionNumber": v1["versionNumber"],
                    "created": v1["created"],
                    "commitHash": v1.get("commitHash"),
                    "changes": v1.get("changes")
                },
                "version2": {
                    "versionNumber": v2["versionNumber"],
                    "created": v2["created"],
                    "commitHash": v2.get("commitHash"),
                    "changes": v2.get("changes")
                },
                "newer_version": newer_version["versionNumber"],
                "direct_path_exists": direct_path_exists,
                "distance": distance
            }
            
            # Include content if available
            if "content" in v1 and "content" in v2:
                comparison["version1"]["content"] = v1["content"]
                comparison["version2"]["content"] = v2["content"]
                
                # Simplistic diff (in real implementation, would use a proper diff algorithm)
                if v1["content"] == v2["content"]:
                    comparison["content_identical"] = True
                else:
                    comparison["content_identical"] = False
            
            return dict_to_json(comparison)
                
        except Exception as e:
            error_msg = f"Error comparing versions: {str(e)}"
            self.logger.error(error_msg)
            return dict_to_json({"error": error_msg})
    
    def tag_version(self, 
                 component_name: str,
                 domain_name: str,
                 container_name: str,
                 version_number: str,
                 tag_name: str,
                 tag_description: Optional[str] = None) -> str:
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
            self.base_manager.ensure_initialized()
            
            # Check if component exists in domain
            component_query = """
            MATCH (c:Entity {name: $container_name, entityType: 'ProjectContainer'})
            MATCH (d:Entity {name: $domain_name, entityType: 'Domain'})-[:PART_OF]->(c)
            MATCH (comp:Entity {name: $component_name})-[:BELONGS_TO]->(d)
            RETURN comp
            """
            
            component_records, _ = self.base_manager.safe_execute_query(
                component_query,
                {"container_name": container_name, "domain_name": domain_name, "component_name": component_name}
            )
            
            if not component_records or len(component_records) == 0:
                return dict_to_json({
                    "error": f"Component '{component_name}' not found in domain '{domain_name}'"
                })
            
            component_id = component_records[0]["comp"]["id"]
            
            # Check if version exists
            version_query = """
            MATCH (comp:Entity {id: $component_id})
            MATCH (v:Entity {entityType: 'Version'})
            MATCH (comp)-[:HAS_VERSION]->(v)
            WHERE v.versionNumber = $version_number
            RETURN v
            """
            
            version_records, _ = self.base_manager.safe_execute_query(
                version_query,
                {"component_id": component_id, "version_number": version_number}
            )
            
            if not version_records or len(version_records) == 0:
                return dict_to_json({
                    "error": f"Version '{version_number}' not found for component '{component_name}'"
                })
            
            version_id = version_records[0]["v"]["id"]
            
            # Check if tag already exists
            tag_check_query = """
            MATCH (v:Entity {id: $version_id})
            MATCH (t:Entity {entityType: 'Tag', name: $tag_name})
            MATCH (v)-[:HAS_TAG]->(t)
            RETURN t
            """
            
            tag_records, _ = self.base_manager.safe_execute_query(
                tag_check_query,
                {"version_id": version_id, "tag_name": tag_name}
            )
            
            if tag_records and len(tag_records) > 0:
                return dict_to_json({
                    "status": "success",
                    "message": f"Tag '{tag_name}' already exists for version '{version_number}'"
                })
            
            # Create tag entity
            tag_id = generate_id("tag")
            timestamp = time.time()
            
            tag_entity = {
                "id": tag_id,
                "name": tag_name,
                "entityType": "Tag",
                "domain": "project",
                "created": timestamp,
                "lastUpdated": timestamp
            }
            
            if tag_description:
                tag_entity["description"] = tag_description
            
            create_tag_query = """
            CREATE (t:Entity $properties)
            RETURN t
            """
            
            create_records, _ = self.base_manager.safe_execute_query(
                create_tag_query,
                {"properties": tag_entity}
            )
            
            if not create_records or len(create_records) == 0:
                return dict_to_json({
                    "error": "Failed to create tag entity"
                })
            
            # Link tag to version
            link_query = """
            MATCH (v:Entity {id: $version_id})
            MATCH (t:Entity {id: $tag_id})
            CREATE (v)-[:HAS_TAG {created: $timestamp}]->(t)
            RETURN t
            """
            
            link_records, _ = self.base_manager.safe_execute_query(
                link_query,
                {"version_id": version_id, "tag_id": tag_id, "timestamp": timestamp}
            )
            
            if not link_records or len(link_records) == 0:
                # Attempt to clean up the created entity
                self.base_manager.safe_execute_query(
                    "MATCH (t:Entity {id: $tag_id}) DELETE t",
                    {"tag_id": tag_id}
                )
                
                return dict_to_json({
                    "error": f"Failed to link tag to version '{version_number}'"
                })
            
            # Get created tag
            tag = dict(link_records[0]["t"].items())
            
            return dict_to_json({
                "status": "success",
                "message": f"Tag '{tag_name}' added to version '{version_number}' of component '{component_name}'",
                "tag": tag
            })
                
        except Exception as e:
            error_msg = f"Error tagging version: {str(e)}"
            self.logger.error(error_msg)
            return dict_to_json({"error": error_msg})
    
    def sync_with_version_control(self, 
                              component_name: str,
                              domain_name: str,
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
            self.base_manager.ensure_initialized()
            
            # Check if component exists in domain
            component_query = """
            MATCH (c:Entity {name: $container_name, entityType: 'ProjectContainer'})
            MATCH (d:Entity {name: $domain_name, entityType: 'Domain'})-[:PART_OF]->(c)
            MATCH (comp:Entity {name: $component_name})-[:BELONGS_TO]->(d)
            RETURN comp
            """
            
            component_records, _ = self.base_manager.safe_execute_query(
                component_query,
                {"container_name": container_name, "domain_name": domain_name, "component_name": component_name}
            )
            
            if not component_records or len(component_records) == 0:
                return dict_to_json({
                    "error": f"Component '{component_name}' not found in domain '{domain_name}'"
                })
            
            component_id = component_records[0]["comp"]["id"]
            
            # Get existing versions by commit hash
            existing_query = """
            MATCH (comp:Entity {id: $component_id})
            MATCH (v:Entity {entityType: 'Version'})
            MATCH (comp)-[:HAS_VERSION]->(v)
            WHERE v.commitHash IS NOT NULL
            RETURN v.commitHash as hash, v.versionNumber as version
            """
            
            existing_records, _ = self.base_manager.safe_execute_query(
                existing_query,
                {"component_id": component_id}
            )
            
            # Build map of existing commit hashes
            existing_commits = {}
            if existing_records:
                for record in existing_records:
                    commit_hash = record["hash"]
                    version = record["version"]
                    if commit_hash:
                        existing_commits[commit_hash] = version
            
            # Process each commit
            results = []
            success_count = 0
            error_count = 0
            
            for commit in commit_data:
                # Extract commit data
                commit_hash = commit.get("hash")
                version = commit.get("version")
                date = commit.get("date")
                author = commit.get("author")
                message = commit.get("message")
                content = commit.get("content")
                
                # Skip if already exists
                if commit_hash in existing_commits:
                    results.append({
                        "status": "skipped",
                        "message": f"Commit {commit_hash} already exists as version {existing_commits[commit_hash]}",
                        "commit": commit_hash,
                        "version": existing_commits[commit_hash]
                    })
                    continue
                
                # Prepare metadata
                metadata = {
                    "author": author,
                    "commit_date": date
                }
                
                # Create version
                result_json = self.create_version(
                    component_name=component_name,
                    domain_name=domain_name,
                    container_name=container_name,
                    version_number=version,
                    commit_hash=commit_hash,
                    content=content,
                    changes=message,
                    metadata=metadata
                )
                
                result = json.loads(result_json)
                
                if "error" in result:
                    results.append({
                        "status": "error",
                        "message": result["error"],
                        "commit": commit_hash,
                        "version": version
                    })
                    error_count += 1
                else:
                    results.append({
                        "status": "success",
                        "message": result["message"],
                        "commit": commit_hash,
                        "version": version
                    })
                    success_count += 1
            
            return dict_to_json({
                "component": component_name,
                "domain": domain_name,
                "container": container_name,
                "total_commits": len(commit_data),
                "success_count": success_count,
                "error_count": error_count,
                "skipped_count": len(commit_data) - success_count - error_count,
                "results": results
            })
                
        except Exception as e:
            error_msg = f"Error syncing with version control: {str(e)}"
            self.logger.error(error_msg)
            return dict_to_json({"error": error_msg}) 