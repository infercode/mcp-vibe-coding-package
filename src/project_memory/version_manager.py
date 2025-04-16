from typing import Any, Dict, List, Optional, Union
import time
import json
import logging
from datetime import datetime

from src.utils import dict_to_json, generate_id
from src.graph_memory.base_manager import BaseManager
from src.graph_memory.entity_manager import EntityManager
from pydantic import BaseModel

from src.models.project_memory import (
    Metadata, 
    ErrorResponse, 
    ErrorDetail,
    SuccessResponse,
    create_metadata,
    VersionCreate,
    VersionGetRequest,
    VersionResponse,
    VersionListResponse,
    VersionCompareRequest,
    TagCreate,
    TagResponse,
    CommitData,
    SyncRequest
)

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
        self.logger = logging.getLogger(__name__)
        self.entity_manager = EntityManager(base_manager)
    
    def create_version(self, version_data: VersionCreate) -> str:
        """
        Create a new version for a component.
        
        Args:
            version_data: Pydantic model containing version creation data
            
        Returns:
            JSON string with the created version
        """
        try:
            self.base_manager.ensure_initialized()
            
            # Extract values from the model
            component_name = version_data.component_name
            domain_name = version_data.domain_name
            container_name = version_data.container_name
            version_number = version_data.version_number
            commit_hash = version_data.commit_hash
            content = version_data.content
            changes = version_data.changes
            metadata = version_data.metadata
            
            # Check if component exists in domain
            component_query = """
            MATCH (c:Entity {name: $container_name, entityType: 'ProjectContainer'})
            MATCH (d:Entity {name: $domain_name, entityType: 'Domain'})-[:PART_OF]->(c)
            MATCH (comp:Entity {name: $component_name})-[:BELONGS_TO]->(d)
            RETURN comp
            """
            
            component_records = self.base_manager.safe_execute_read_query(
                component_query,
                {"container_name": container_name, "domain_name": domain_name, "component_name": component_name}
            )
            
            if not component_records or len(component_records) == 0:
                error = ErrorDetail(
                    code="NOT_FOUND",
                    message=f"Component '{component_name}' not found in domain '{domain_name}'",
                    details={}
                )
                return dict_to_json(ErrorResponse(
                    status="error",
                    timestamp=datetime.now(),
                    error=error
                ).model_dump())
            
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
            
            version_records = self.base_manager.safe_execute_read_query(
                version_check_query,
                {"component_id": component_id, "version_number": version_number}
            )
            
            if version_records and len(version_records) > 0:
                error = ErrorDetail(
                    code="ALREADY_EXISTS",
                    message=f"Version '{version_number}' already exists for component '{component_name}'",
                    details={}
                )
                return dict_to_json(ErrorResponse(
                    status="error",
                    timestamp=datetime.now(),
                    error=error
                ).model_dump())
            
            # Generate version entity ID
            version_id = generate_id("ver")
            timestamp = datetime.now()
            
            # Prepare version entity
            version_entity = {
                "id": version_id,
                "name": f"{component_name}_v{version_number}",
                "entityType": "Version",
                "versionNumber": version_number,
                "componentId": component_id,
                "domain": "project",
                "created": timestamp.timestamp(),
                "lastUpdated": timestamp.timestamp()
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
            
            create_records = self.base_manager.safe_execute_write_query(
                create_query,
                {"properties": version_entity}
            )
            
            if not create_records or len(create_records) == 0:
                error = ErrorDetail(
                    code="CREATION_FAILED",
                    message="Failed to create version entity",
                    details={}
                )
                return dict_to_json(ErrorResponse(
                    status="error",
                    timestamp=datetime.now(),
                    error=error
                ).model_dump())
            
            # Link version to component
            link_query = """
            MATCH (comp:Entity {id: $component_id})
            MATCH (v:Entity {id: $version_id})
            CREATE (comp)-[:HAS_VERSION {created: datetime()}]->(v)
            RETURN v
            """
            
            link_records = self.base_manager.safe_execute_write_query(
                link_query,
                {"component_id": component_id, "version_id": version_id}
            )
            
            if not link_records or len(link_records) == 0:
                # Attempt to clean up the created entity
                self.base_manager.safe_execute_write_query(
                    "MATCH (v:Entity {id: $version_id}) DELETE v",
                    {"version_id": version_id}
                )
                
                error = ErrorDetail(
                    code="LINK_FAILED",
                    message=f"Failed to link version to component '{component_name}'",
                    details={}
                )
                return dict_to_json(ErrorResponse(
                    status="error",
                    timestamp=datetime.now(),
                    error=error
                ).model_dump())
            
            # Find previous version to create chain
            prev_version_query = """
            MATCH (comp:Entity {id: $component_id})
            MATCH (comp)-[:HAS_VERSION]->(prev:Entity {entityType: 'Version'})
            WHERE prev.id <> $version_id
            RETURN prev
            ORDER BY prev.versionNumber DESC
            LIMIT 1
            """
            
            prev_records = self.base_manager.safe_execute_read_query(
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
                CREATE (curr)-[:SUPERSEDES {created: datetime()}]->(prev)
                """
                
                self.base_manager.safe_execute_write_query(
                    supersedes_query,
                    {"version_id": version_id, "prev_version_id": prev_version_id}
                )
            
            # Get created version
            version = dict(link_records[0]["v"].items())
            
            # Create response using Pydantic model
            response = VersionResponse(
                message=f"Version '{version_number}' created for component '{component_name}'",
                version_id=version_id,
                version=version,
                component_name=component_name,
                timestamp=datetime.now()
            )
            
            return dict_to_json(response.model_dump(exclude_none=True))
                
        except Exception as e:
            error_msg = f"Error creating version: {str(e)}"
            self.logger.error(error_msg)
            error = ErrorDetail(
                code="ERROR",
                message=error_msg,
                details={}
            )
            return dict_to_json(ErrorResponse(
                status="error",
                timestamp=datetime.now(),
                error=error
            ).model_dump())
    
    def get_version(self, version_request: VersionGetRequest) -> str:
        """
        Get a specific version of a component.
        
        Args:
            version_request: Pydantic model containing version request data
            
        Returns:
            JSON string with the version details
        """
        try:
            self.base_manager.ensure_initialized()
            
            # Extract values from model
            component_name = version_request.component_name
            domain_name = version_request.domain_name
            container_name = version_request.container_name
            version_number = version_request.version_number
            
            # Check if component exists in domain
            component_query = """
            MATCH (c:Entity {name: $container_name, entityType: 'ProjectContainer'})
            MATCH (d:Entity {name: $domain_name, entityType: 'Domain'})-[:PART_OF]->(c)
            MATCH (comp:Entity {name: $component_name})-[:BELONGS_TO]->(d)
            RETURN comp
            """
            
            component_records = self.base_manager.safe_execute_read_query(
                component_query,
                {"container_name": container_name, "domain_name": domain_name, "component_name": component_name}
            )
            
            if not component_records or len(component_records) == 0:
                error = ErrorDetail(
                    code="NOT_FOUND",
                    message=f"Component '{component_name}' not found in domain '{domain_name}'",
                    details={}
                )
                return dict_to_json(ErrorResponse(
                    status="error",
                    timestamp=datetime.now(),
                    error=error
                ).model_dump())
            
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
                
                version_records = self.base_manager.safe_execute_read_query(
                    version_query,
                    {"component_id": component_id, "version_number": version_number}
                )
                
                if not version_records or len(version_records) == 0:
                    error = ErrorDetail(
                        code="NOT_FOUND",
                        message=f"Version '{version_number}' not found for component '{component_name}'",
                        details={}
                    )
                    return dict_to_json(ErrorResponse(
                        status="error",
                        timestamp=datetime.now(),
                        error=error
                    ).model_dump())
                
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
                
                latest_records = self.base_manager.safe_execute_read_query(
                    latest_query,
                    {"component_id": component_id}
                )
                
                if not latest_records or len(latest_records) == 0:
                    error = ErrorDetail(
                        code="NOT_FOUND",
                        message=f"No versions found for component '{component_name}'",
                        details={}
                    )
                    return dict_to_json(ErrorResponse(
                        status="error",
                        timestamp=datetime.now(),
                        error=error
                    ).model_dump())
                
                version = dict(latest_records[0]["v"].items())
            
            # Create response using Pydantic model
            response = VersionResponse(
                message=f"Retrieved version{' ' + version_number if version_number else ''} of component '{component_name}'",
                version_id=version.get("id"),
                version=version,
                component_name=component_name,
                timestamp=datetime.now()
            )
            
            # Add additional context to response
            response_dict = response.model_dump(exclude_none=True)
            response_dict["domain"] = domain_name
            response_dict["container"] = container_name
            
            return dict_to_json(response_dict)
                
        except Exception as e:
            error_msg = f"Error retrieving version: {str(e)}"
            self.logger.error(error_msg)
            error = ErrorDetail(
                code="ERROR",
                message=error_msg,
                details={}
            )
            return dict_to_json(ErrorResponse(
                status="error",
                timestamp=datetime.now(),
                error=error
            ).model_dump())
    
    def list_versions(self, version_request: VersionGetRequest, limit: int = 10) -> str:
        """
        List all versions of a component.
        
        Args:
            version_request: Pydantic model containing version request data
            limit: Maximum number of versions to return
            
        Returns:
            JSON string with the list of versions
        """
        try:
            self.base_manager.ensure_initialized()
            
            # Extract values from model
            component_name = version_request.component_name
            domain_name = version_request.domain_name
            container_name = version_request.container_name
            
            # Validate limit
            if limit <= 0 or limit > 100:
                limit = 10  # Default to 10 for invalid limits
            
            # Check if component exists in domain
            component_query = """
            MATCH (c:Entity {name: $container_name, entityType: 'ProjectContainer'})
            MATCH (d:Entity {name: $domain_name, entityType: 'Domain'})-[:PART_OF]->(c)
            MATCH (comp:Entity {name: $component_name})-[:BELONGS_TO]->(d)
            RETURN comp
            """
            
            component_records = self.base_manager.safe_execute_read_query(
                component_query,
                {"container_name": container_name, "domain_name": domain_name, "component_name": component_name}
            )
            
            if not component_records or len(component_records) == 0:
                error = ErrorDetail(
                    code="NOT_FOUND",
                    message=f"Component '{component_name}' not found in domain '{domain_name}'",
                    details={}
                )
                return dict_to_json(ErrorResponse(
                    status="error",
                    timestamp=datetime.now(),
                    error=error
                ).model_dump())
            
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
            
            version_records = self.base_manager.safe_execute_read_query(
                versions_query,
                {"component_id": component_id, "limit": limit}
            )
            
            versions = []
            if version_records:
                for record in version_records:
                    version = dict(record["v"].items())
                    versions.append(version)
            
            # Create response using Pydantic model
            response = VersionListResponse(
                message=f"Retrieved {len(versions)} versions for component '{component_name}'",
                component_name=component_name,
                domain_name=domain_name,
                container_name=container_name,
                version_count=len(versions),
                versions=versions,
                timestamp=datetime.now()
            )
            
            return dict_to_json(response.model_dump(exclude_none=True))
                
        except Exception as e:
            error_msg = f"Error listing versions: {str(e)}"
            self.logger.error(error_msg)
            error = ErrorDetail(
                code="ERROR",
                message=error_msg,
                details={}
            )
            return dict_to_json(ErrorResponse(
                status="error",
                timestamp=datetime.now(),
                error=error
            ).model_dump())
    
    def get_version_history(self, version_request: VersionGetRequest, include_content: bool = False) -> str:
        """
        Get the version history of a component with supersedes relationships.
        
        Args:
            version_request: Pydantic model containing version request data
            include_content: Whether to include content in the version history
            
        Returns:
            JSON string with the version history
        """
        try:
            self.base_manager.ensure_initialized()
            
            # Extract values from model
            component_name = version_request.component_name
            domain_name = version_request.domain_name
            container_name = version_request.container_name
            
            # Check if component exists in domain
            component_query = """
            MATCH (c:Entity {name: $container_name, entityType: 'ProjectContainer'})
            MATCH (d:Entity {name: $domain_name, entityType: 'Domain'})-[:PART_OF]->(c)
            MATCH (comp:Entity {name: $component_name})-[:BELONGS_TO]->(d)
            RETURN comp
            """
            
            component_records = self.base_manager.safe_execute_read_query(
                component_query,
                {"container_name": container_name, "domain_name": domain_name, "component_name": component_name}
            )
            
            if not component_records or len(component_records) == 0:
                error = ErrorDetail(
                    code="NOT_FOUND",
                    message=f"Component '{component_name}' not found in domain '{domain_name}'",
                    details={}
                )
                return dict_to_json(ErrorResponse(
                    status="error",
                    timestamp=datetime.now(),
                    error=error
                ).model_dump())
            
            component_id = component_records[0]["comp"]["id"]
            
            # Get the latest version
            latest_query = """
            MATCH (comp:Entity {id: $component_id})
            MATCH (v:Entity {entityType: 'Version'})
            MATCH (comp)-[:HAS_VERSION]->(v)
            WHERE NOT EXISTS((ANY)-[:SUPERSEDES]->(v))
            RETURN v
            """
            
            latest_records = self.base_manager.safe_execute_read_query(
                latest_query,
                {"component_id": component_id}
            )
            
            if not latest_records or len(latest_records) == 0:
                error = ErrorDetail(
                    code="NOT_FOUND",
                    message=f"No versions found for component '{component_name}'",
                    details={}
                )
                return dict_to_json(ErrorResponse(
                    status="error",
                    timestamp=datetime.now(),
                    error=error
                ).model_dump())
            
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
            
            history_records = self.base_manager.safe_execute_read_query(
                history_query,
                {"latest_version_id": latest_version_id}
            )
            
            versions = []
            if history_records:
                for record in history_records:
                    version = dict(record["v"].items())
                    versions.append(version)
            
            # Create response using Pydantic model - extending the VersionListResponse
            response = VersionListResponse(
                message=f"Retrieved version history for component '{component_name}'",
                component_name=component_name,
                domain_name=domain_name,
                container_name=container_name,
                version_count=len(versions),
                versions=versions,
                timestamp=datetime.now()
            )
            
            # Add latest version info
            response_dict = response.model_dump(exclude_none=True)
            response_dict["latest_version"] = latest_version["versionNumber"]
            response_dict["include_content"] = include_content
            
            return dict_to_json(response_dict)
                
        except Exception as e:
            error_msg = f"Error retrieving version history: {str(e)}"
            self.logger.error(error_msg)
            error = ErrorDetail(
                code="ERROR",
                message=error_msg,
                details={}
            )
            return dict_to_json(ErrorResponse(
                status="error",
                timestamp=datetime.now(),
                error=error
            ).model_dump())
    
    def compare_versions(self, compare_request: VersionCompareRequest) -> str:
        """
        Compare two versions of a component.
        
        Args:
            compare_request: Pydantic model containing comparison request data
            
        Returns:
            JSON string with the comparison result
        """
        try:
            self.base_manager.ensure_initialized()
            
            # Extract values from model
            component_name = compare_request.component_name
            domain_name = compare_request.domain_name
            container_name = compare_request.container_name
            version1 = compare_request.version1
            version2 = compare_request.version2
            
            if version1 == version2:
                # This is a valid case, just note it
                self.logger.info(f"Comparing version '{version1}' to itself")
            
            # Check if component exists in domain
            component_query = """
            MATCH (c:Entity {name: $container_name, entityType: 'ProjectContainer'})
            MATCH (d:Entity {name: $domain_name, entityType: 'Domain'})-[:PART_OF]->(c)
            MATCH (comp:Entity {name: $component_name})-[:BELONGS_TO]->(d)
            RETURN comp
            """
            
            component_records = self.base_manager.safe_execute_read_query(
                component_query,
                {"container_name": container_name, "domain_name": domain_name, "component_name": component_name}
            )
            
            if not component_records or len(component_records) == 0:
                error = ErrorDetail(
                    code="NOT_FOUND",
                    message=f"Component '{component_name}' not found in domain '{domain_name}'",
                    details={}
                )
                return dict_to_json(ErrorResponse(
                    status="error",
                    timestamp=datetime.now(),
                    error=error
                ).model_dump())
            
            component_id = component_records[0]["comp"]["id"]
            
            # Get the two versions
            versions_query = """
            MATCH (comp:Entity {id: $component_id})
            MATCH (v:Entity {entityType: 'Version'})
            MATCH (comp)-[:HAS_VERSION]->(v)
            WHERE v.versionNumber IN [$version1, $version2]
            RETURN v
            """
            
            version_records = self.base_manager.safe_execute_read_query(
                versions_query,
                {"component_id": component_id, "version1": version1, "version2": version2}
            )
            
            if not version_records or len(version_records) < 2:
                error = ErrorDetail(
                    code="VERSION_NOT_FOUND",
                    message=f"One or both versions not found for component '{component_name}'",
                    details={}
                )
                return dict_to_json(ErrorResponse(
                    status="error",
                    timestamp=datetime.now(),
                    error=error
                ).model_dump())
            
            # Extract version data
            version_data = {}
            for record in version_records:
                v = dict(record["v"].items())
                version_data[v["versionNumber"]] = v
            
            v1 = version_data.get(version1)
            v2 = version_data.get(version2)
            
            if not v1 or not v2:
                error = ErrorDetail(
                    code="VERSION_NOT_FOUND",
                    message=f"One or both versions not found for component '{component_name}'",
                    details={}
                )
                return dict_to_json(ErrorResponse(
                    status="error",
                    timestamp=datetime.now(),
                    error=error
                ).model_dump())
            
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
            
            path_records = self.base_manager.safe_execute_read_query(
                path_query,
                {"newer_id": newer_version["id"], "older_id": older_version["id"]}
            )
            
            direct_path_exists = False
            distance = None
            
            if path_records and len(path_records) > 0:
                direct_path_exists = True
                distance = path_records[0]["distance"]
            
            # Create response
            response = SuccessResponse(
                message=f"Compared versions {version1} and {version2} of component '{component_name}'",
                timestamp=datetime.now()
            )
            
            # Add comparison data to response
            response_dict = response.model_dump(exclude_none=True)
            
            # Build comparison result
            comparison = {
                "component_name": component_name,
                "domain_name": domain_name,
                "container_name": container_name,
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
            
            # Merge comparison into response
            response_dict.update(comparison)
            
            return dict_to_json(response_dict)
                
        except Exception as e:
            error_msg = f"Error comparing versions: {str(e)}"
            self.logger.error(error_msg)
            error = ErrorDetail(
                code="ERROR",
                message=error_msg,
                details={}
            )
            return dict_to_json(ErrorResponse(
                status="error",
                timestamp=datetime.now(),
                error=error
            ).model_dump())
    
    def tag_version(self, tag_data: TagCreate) -> str:
        """
        Add a tag to a specific version of a component.
        
        Args:
            tag_data: Pydantic model containing tag data
            
        Returns:
            JSON string with the result
        """
        try:
            self.base_manager.ensure_initialized()
            
            # Extract values from model
            component_name = tag_data.component_name
            domain_name = tag_data.domain_name
            container_name = tag_data.container_name
            version_number = tag_data.version_number
            tag_name = tag_data.tag_name
            tag_description = tag_data.tag_description
            
            # Check if component exists in domain
            component_query = """
            MATCH (c:Entity {name: $container_name, entityType: 'ProjectContainer'})
            MATCH (d:Entity {name: $domain_name, entityType: 'Domain'})-[:PART_OF]->(c)
            MATCH (comp:Entity {name: $component_name})-[:BELONGS_TO]->(d)
            RETURN comp
            """
            
            component_records = self.base_manager.safe_execute_read_query(
                component_query,
                {"container_name": container_name, "domain_name": domain_name, "component_name": component_name}
            )
            
            if not component_records or len(component_records) == 0:
                error = ErrorDetail(
                    code="NOT_FOUND",
                    message=f"Component '{component_name}' not found in domain '{domain_name}'",
                    details={}
                )
                return dict_to_json(ErrorResponse(
                    status="error",
                    timestamp=datetime.now(),
                    error=error
                ).model_dump())
            
            component_id = component_records[0]["comp"]["id"]
            
            # Check if version exists
            version_query = """
            MATCH (comp:Entity {id: $component_id})
            MATCH (v:Entity {entityType: 'Version'})
            MATCH (comp)-[:HAS_VERSION]->(v)
            WHERE v.versionNumber = $version_number
            RETURN v
            """
            
            version_records = self.base_manager.safe_execute_read_query(
                version_query,
                {"component_id": component_id, "version_number": version_number}
            )
            
            if not version_records or len(version_records) == 0:
                error = ErrorDetail(
                    code="VERSION_NOT_FOUND",
                    message=f"Version '{version_number}' not found for component '{component_name}'",
                    details={}
                )
                return dict_to_json(ErrorResponse(
                    status="error",
                    timestamp=datetime.now(),
                    error=error
                ).model_dump())
            
            version_id = version_records[0]["v"]["id"]
            
            # Check if tag already exists
            tag_check_query = """
            MATCH (v:Entity {id: $version_id})
            MATCH (t:Entity {entityType: 'Tag', name: $tag_name})
            MATCH (v)-[:HAS_TAG]->(t)
            RETURN t
            """
            
            tag_records = self.base_manager.safe_execute_read_query(
                tag_check_query,
                {"version_id": version_id, "tag_name": tag_name}
            )
            
            if tag_records and len(tag_records) > 0:
                # Tag already exists, return success
                tag = dict(tag_records[0]["t"].items())
                response = TagResponse(
                    message=f"Tag '{tag_name}' already exists for version '{version_number}'",
                    tag_id=tag.get("id"),
                    tag=tag,
                    component_name=component_name,
                    version_number=version_number,
                    timestamp=datetime.now()
                )
                return dict_to_json(response.model_dump(exclude_none=True))
            
            # Create tag entity
            tag_id = generate_id("tag")
            timestamp = datetime.now()
            
            tag_entity = {
                "id": tag_id,
                "name": tag_name,
                "entityType": "Tag",
                "domain": "project",
                "created": timestamp.timestamp(),
                "lastUpdated": timestamp.timestamp()
            }
            
            if tag_description:
                tag_entity["description"] = tag_description
            
            create_tag_query = """
            CREATE (t:Entity $properties)
            RETURN t
            """
            
            create_records = self.base_manager.safe_execute_write_query(
                create_tag_query,
                {"properties": tag_entity}
            )
            
            if not create_records or len(create_records) == 0:
                error = ErrorDetail(
                    code="CREATION_FAILED",
                    message="Failed to create tag entity",
                    details={}
                )
                return dict_to_json(ErrorResponse(
                    status="error",
                    timestamp=datetime.now(),
                    error=error
                ).model_dump())
            
            # Link tag to version
            link_query = """
            MATCH (v:Entity {id: $version_id})
            MATCH (t:Entity {id: $tag_id})
            CREATE (v)-[:HAS_TAG {created: datetime()}]->(t)
            RETURN t
            """
            
            link_records = self.base_manager.safe_execute_write_query(
                link_query,
                {"version_id": version_id, "tag_id": tag_id}
            )
            
            if not link_records or len(link_records) == 0:
                # Attempt to clean up the created entity
                self.base_manager.safe_execute_write_query(
                    "MATCH (t:Entity {id: $tag_id}) DELETE t",
                    {"tag_id": tag_id}
                )
                
                error = ErrorDetail(
                    code="LINK_FAILED",
                    message=f"Failed to link tag to version '{version_number}'",
                    details={}
                )
                return dict_to_json(ErrorResponse(
                    status="error",
                    timestamp=datetime.now(),
                    error=error
                ).model_dump())
            
            # Get created tag
            tag = dict(link_records[0]["t"].items())
            
            # Create response
            response = TagResponse(
                message=f"Tag '{tag_name}' added to version '{version_number}' of component '{component_name}'",
                tag_id=tag_id,
                tag=tag,
                component_name=component_name,
                version_number=version_number,
                timestamp=datetime.now()
            )
            
            return dict_to_json(response.model_dump(exclude_none=True))
                
        except Exception as e:
            error_msg = f"Error tagging version: {str(e)}"
            self.logger.error(error_msg)
            error = ErrorDetail(
                code="ERROR",
                message=error_msg,
                details={}
            )
            return dict_to_json(ErrorResponse(
                status="error",
                timestamp=datetime.now(),
                error=error
            ).model_dump())
    
    def sync_with_version_control(self, sync_request: SyncRequest) -> str:
        """
        Synchronize component versions with version control system data.
        
        Args:
            sync_request: Pydantic model containing sync request data
            
        Returns:
            JSON string with the sync result
        """
        try:
            self.base_manager.ensure_initialized()
            
            # Extract values from model
            component_name = sync_request.component_name
            domain_name = sync_request.domain_name
            container_name = sync_request.container_name
            commit_data = sync_request.commits
            
            # Check if component exists in domain
            component_query = """
            MATCH (c:Entity {name: $container_name, entityType: 'ProjectContainer'})
            MATCH (d:Entity {name: $domain_name, entityType: 'Domain'})-[:PART_OF]->(c)
            MATCH (comp:Entity {name: $component_name})-[:BELONGS_TO]->(d)
            RETURN comp
            """
            
            component_records = self.base_manager.safe_execute_read_query(
                component_query,
                {"container_name": container_name, "domain_name": domain_name, "component_name": component_name}
            )
            
            if not component_records or len(component_records) == 0:
                error = ErrorDetail(
                    code="NOT_FOUND",
                    message=f"Component '{component_name}' not found in domain '{domain_name}'",
                    details={}
                )
                return dict_to_json(ErrorResponse(
                    status="error",
                    timestamp=datetime.now(),
                    error=error
                ).model_dump())
            
            component_id = component_records[0]["comp"]["id"]
            
            # Get existing versions by commit hash
            existing_query = """
            MATCH (comp:Entity {id: $component_id})
            MATCH (v:Entity {entityType: 'Version'})
            MATCH (comp)-[:HAS_VERSION]->(v)
            WHERE v.commitHash IS NOT NULL
            RETURN v.commitHash as hash, v.versionNumber as version
            """
            
            existing_records = self.base_manager.safe_execute_read_query(
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
                commit_hash = commit.hash
                version = commit.version
                date = commit.date
                author = commit.author
                message = commit.message
                content = commit.content
                
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
                metadata = {}
                if author:
                    metadata["author"] = author
                if date:
                    metadata["commit_date"] = date
                
                # Create version
                version_create = VersionCreate(
                    component_name=component_name,
                    domain_name=domain_name,
                    container_name=container_name,
                    version_number=version,
                    commit_hash=commit_hash,
                    content=content,
                    changes=message,
                    metadata=metadata
                )
                
                result_json = self.create_version(version_create)
                
                result = json.loads(result_json)
                
                if "status" in result and result["status"] == "error":
                    results.append({
                        "status": "error",
                        "message": result.get("error", {}).get("message", "Unknown error"),
                        "commit": commit_hash,
                        "version": version
                    })
                    error_count += 1
                else:
                    results.append({
                        "status": "success",
                        "message": result.get("message", f"Version {version} created"),
                        "commit": commit_hash,
                        "version": version
                    })
                    success_count += 1
            
            # Create response
            response = SuccessResponse(
                message=f"Synced {len(commit_data)} commits for component '{component_name}'",
                timestamp=datetime.now()
            )
            
            # Add sync data to response
            response_dict = response.model_dump(exclude_none=True)
            response_dict.update({
                "component_name": component_name,
                "domain_name": domain_name,
                "container_name": container_name,
                "total_commits": len(commit_data),
                "success_count": success_count,
                "error_count": error_count,
                "skipped_count": len(commit_data) - success_count - error_count,
                "results": results
            })
            
            return dict_to_json(response_dict)
                
        except Exception as e:
            error_msg = f"Error syncing with version control: {str(e)}"
            self.logger.error(error_msg)
            error = ErrorDetail(
                code="ERROR",
                message=error_msg,
                details={}
            )
            return dict_to_json(ErrorResponse(
                status="error",
                timestamp=datetime.now(),
                error=error
            ).model_dump()) 