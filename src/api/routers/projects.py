from typing import Dict, Any, Optional, List
from fastapi import APIRouter, HTTPException, Depends
from pydantic import BaseModel, Field
import json

from src.graph_memory import GraphMemoryManager
from ..dependencies import get_memory_manager
from ..utils import parse_response

router = APIRouter(
    prefix="/projects",
    tags=["projects"],
    responses={404: {"description": "Not found"}}
)

class Project(BaseModel):
    name: str = Field(..., description="Project name")
    description: Optional[str] = Field(None, description="Project description")

class Domain(BaseModel):
    name: str = Field(..., description="Domain name")
    container_name: str = Field(..., description="Project container name")
    description: Optional[str] = Field(None, description="Domain description")
    properties: Optional[Dict[str, Any]] = None

class Component(BaseModel):
    name: str = Field(..., description="Component name")
    component_type: str = Field(..., description="Type of component (e.g., Module, Class, Function)")
    domain_name: str = Field(..., description="Domain name")
    container_name: str = Field(..., description="Project container name")
    description: Optional[str] = Field(None, description="Component description")
    content: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = None

class ComponentUpdate(BaseModel):
    updates: Dict[str, Any] = Field(..., description="Updates to apply to the component")

class ComponentRelationship(BaseModel):
    from_component: str
    to_component: str
    domain_name: str
    container_name: str
    relation_type: str
    properties: Optional[Dict[str, Any]] = None

class DomainRelationship(BaseModel):
    from_domain: str
    to_domain: str
    container_name: str
    relation_type: str
    properties: Optional[Dict[str, Any]] = None

class Version(BaseModel):
    component_name: str
    domain_name: str
    container_name: str
    version_number: str
    commit_hash: Optional[str] = None
    content: Optional[str] = None
    changes: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = None

class VersionTag(BaseModel):
    component_name: str
    domain_name: str
    container_name: str
    version_number: str
    tag_name: str
    tag_description: Optional[str] = None

class Dependency(BaseModel):
    from_component: str
    to_component: str
    domain_name: str
    container_name: str
    dependency_type: str
    properties: Optional[Dict[str, Any]] = None

class DomainEntity(BaseModel):
    entity_name: str
    entity_type: str
    domain_name: str
    container_name: str
    properties: Optional[Dict[str, Any]] = None

class VersionControlSync(BaseModel):
    component_name: str
    domain_name: str
    container_name: str
    commit_data: List[Dict[str, Any]]

# Project Container Management
@router.post("/", response_model=Dict[str, Any])
async def create_project(project: Project, memory: GraphMemoryManager = Depends(get_memory_manager)):
    """Create a new project container."""
    try:
        result = memory.create_project_container(project.dict())
        return json.loads(result)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/{project_name}", response_model=Dict[str, Any])
async def get_project(project_name: str, memory: GraphMemoryManager = Depends(get_memory_manager)):
    """Get project container details."""
    try:
        result = memory.get_project_container(project_name)
        return json.loads(result)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.put("/{project_name}", response_model=Dict[str, Any])
async def update_project(project_name: str, updates: Dict[str, Any], memory: GraphMemoryManager = Depends(get_memory_manager)):
    """Update a project container."""
    try:
        result = memory.update_project_container(project_name, updates)
        return json.loads(result)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.delete("/{project_name}", response_model=Dict[str, Any])
async def delete_project(project_name: str, memory: GraphMemoryManager = Depends(get_memory_manager)):
    """Delete a project container."""
    try:
        result = memory.delete_project_container(project_name)
        return json.loads(result)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/", response_model=Dict[str, Any])
async def list_projects(memory: GraphMemoryManager = Depends(get_memory_manager)):
    """List all project containers."""
    try:
        result = memory.list_project_containers()
        return json.loads(result)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/{project_name}/status", response_model=Dict[str, Any])
async def get_project_status(project_name: str, memory: GraphMemoryManager = Depends(get_memory_manager)):
    """Get status summary of a project container."""
    try:
        result = memory.get_project_status(project_name)
        return json.loads(result)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# Domain Management
@router.post("/domains", response_model=Dict[str, Any])
async def create_domain(domain: Domain, memory: GraphMemoryManager = Depends(get_memory_manager)):
    """Create a new domain within a project container."""
    try:
        result = memory.create_domain(domain.name, domain.container_name, domain.description, domain.properties)
        return json.loads(result)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/domains/{domain_name}/{container_name}", response_model=Dict[str, Any])
async def get_domain(domain_name: str, container_name: str, memory: GraphMemoryManager = Depends(get_memory_manager)):
    """Get domain details."""
    try:
        result = memory.get_domain(domain_name, container_name)
        return json.loads(result)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.put("/domains/{domain_name}/{container_name}", response_model=Dict[str, Any])
async def update_domain(
    domain_name: str,
    container_name: str,
    updates: Dict[str, Any],
    memory: GraphMemoryManager = Depends(get_memory_manager)
):
    """Update a domain's properties."""
    try:
        result = memory.update_domain(domain_name, container_name, updates)
        return json.loads(result)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.delete("/domains/{domain_name}/{container_name}", response_model=Dict[str, Any])
async def delete_domain(
    domain_name: str,
    container_name: str,
    delete_components: bool = False,
    memory: GraphMemoryManager = Depends(get_memory_manager)
):
    """Delete a domain from a project container."""
    try:
        result = memory.delete_domain(domain_name, container_name, delete_components)
        return json.loads(result)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/domains/{container_name}", response_model=Dict[str, Any])
async def list_domains(
    container_name: str,
    sort_by: str = "name",
    limit: int = 100,
    memory: GraphMemoryManager = Depends(get_memory_manager)
):
    """List all domains in a project container."""
    try:
        result = memory.list_domains(container_name, sort_by, limit)
        return json.loads(result)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# Component Management
@router.post("/components", response_model=Dict[str, Any])
async def create_component(component: Component, memory: GraphMemoryManager = Depends(get_memory_manager)):
    """Create a component within a project."""
    try:
        result = memory.create_component(
            component.name,
            component.component_type,
            component.domain_name,
            component.container_name,
            component.description,
            component.content,
            component.metadata
        )
        return json.loads(result)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/components/{name}/{domain_name}/{container_name}", response_model=Dict[str, Any])
async def get_component(
    name: str,
    domain_name: str,
    container_name: str,
    memory: GraphMemoryManager = Depends(get_memory_manager)
):
    """Get a component from a project."""
    try:
        result = memory.get_component(name, domain_name, container_name)
        return json.loads(result)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.put("/components/{name}/{container_name}", response_model=Dict[str, Any])
async def update_component(
    name: str,
    container_name: str,
    update: ComponentUpdate,
    domain_name: Optional[str] = None,
    memory: GraphMemoryManager = Depends(get_memory_manager)
):
    """Update a component's properties."""
    try:
        result = memory.update_component(name, container_name, update.updates, domain_name)
        return json.loads(result)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.delete("/components/{component_id}", response_model=Dict[str, Any])
async def delete_component(
    component_id: str,
    domain_name: Optional[str] = None,
    container_name: Optional[str] = None,
    memory: GraphMemoryManager = Depends(get_memory_manager)
):
    """Delete a component from a domain."""
    try:
        result = memory.delete_component(component_id, domain_name, container_name)
        return json.loads(result)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/components/{domain_name}/{container_name}", response_model=Dict[str, Any])
async def list_components(
    domain_name: str,
    container_name: str,
    component_type: Optional[str] = None,
    sort_by: str = "name",
    limit: int = 100,
    memory: GraphMemoryManager = Depends(get_memory_manager)
):
    """List all components in a domain."""
    try:
        result = memory.list_components(domain_name, container_name, component_type, sort_by, limit)
        return json.loads(result)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# Relationship Management
@router.post("/components/relationships", response_model=Dict[str, Any])
async def create_component_relationship(
    relationship: ComponentRelationship,
    memory: GraphMemoryManager = Depends(get_memory_manager)
):
    """Create a relationship between two components."""
    try:
        result = memory.create_component_relationship(
            relationship.from_component,
            relationship.to_component,
            relationship.domain_name,
            relationship.container_name,
            relationship.relation_type,
            relationship.properties
        )
        return json.loads(result)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/domains/relationships", response_model=Dict[str, Any])
async def create_domain_relationship(
    relationship: DomainRelationship,
    memory: GraphMemoryManager = Depends(get_memory_manager)
):
    """Create a relationship between two domains."""
    try:
        result = memory.create_domain_relationship(
            relationship.from_domain,
            relationship.to_domain,
            relationship.container_name,
            relationship.relation_type,
            relationship.properties
        )
        return json.loads(result)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# Version Management
@router.post("/versions", response_model=Dict[str, Any])
async def create_version(version: Version, memory: GraphMemoryManager = Depends(get_memory_manager)):
    """Create a new version for a component."""
    try:
        result = memory.create_version(
            version.component_name,
            version.domain_name,
            version.container_name,
            version.version_number,
            version.commit_hash,
            version.content,
            version.changes,
            version.metadata
        )
        return json.loads(result)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/versions/{component_name}/{domain_name}/{container_name}", response_model=Dict[str, Any])
async def get_version(
    component_name: str,
    domain_name: str,
    container_name: str,
    version_number: Optional[str] = None,
    memory: GraphMemoryManager = Depends(get_memory_manager)
):
    """Get a specific version of a component."""
    try:
        result = memory.get_version(component_name, domain_name, container_name, version_number)
        return json.loads(result)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/versions/{component_name}/{domain_name}/{container_name}/list", response_model=Dict[str, Any])
async def list_versions(
    component_name: str,
    domain_name: str,
    container_name: str,
    limit: int = 10,
    memory: GraphMemoryManager = Depends(get_memory_manager)
):
    """List all versions of a component."""
    try:
        result = memory.list_versions(component_name, domain_name, container_name, limit)
        return json.loads(result)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/versions/{component_name}/{domain_name}/{container_name}/history", response_model=Dict[str, Any])
async def get_version_history(
    component_name: str,
    domain_name: str,
    container_name: str,
    include_content: bool = False,
    memory: GraphMemoryManager = Depends(get_memory_manager)
):
    """Get the version history of a component."""
    try:
        result = memory.get_version_history(component_name, domain_name, container_name, include_content)
        return json.loads(result)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/versions/tag", response_model=Dict[str, Any])
async def tag_version(tag: VersionTag, memory: GraphMemoryManager = Depends(get_memory_manager)):
    """Add a tag to a specific version of a component."""
    try:
        result = memory.tag_version(
            tag.component_name,
            tag.domain_name,
            tag.container_name,
            tag.version_number,
            tag.tag_name,
            tag.tag_description
        )
        return json.loads(result)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/versions/{component_name}/{domain_name}/{container_name}/compare", response_model=Dict[str, Any])
async def compare_versions(
    component_name: str,
    domain_name: str,
    container_name: str,
    version1: str,
    version2: str,
    memory: GraphMemoryManager = Depends(get_memory_manager)
):
    """Compare two versions of a component."""
    try:
        result = memory.compare_versions(component_name, domain_name, container_name, version1, version2)
        return json.loads(result)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# Dependencies Management
@router.get("/dependencies/{component_name}/{domain_name}/{container_name}", response_model=Dict[str, Any])
async def get_dependencies(
    component_name: str,
    domain_name: str,
    container_name: str,
    direction: str = "outgoing",
    dependency_type: Optional[str] = None,
    memory: GraphMemoryManager = Depends(get_memory_manager)
):
    """Get dependencies for a component."""
    try:
        result = memory.get_dependencies(component_name, domain_name, container_name, direction, dependency_type)
        return json.loads(result)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.delete("/dependencies/{from_component}/{to_component}/{domain_name}/{container_name}/{dependency_type}", response_model=Dict[str, Any])
async def delete_dependency(
    from_component: str,
    to_component: str,
    domain_name: str,
    container_name: str,
    dependency_type: str,
    memory: GraphMemoryManager = Depends(get_memory_manager)
):
    """Delete a dependency relationship between components."""
    try:
        result = memory.delete_dependency(from_component, to_component, domain_name, container_name, dependency_type)
        return json.loads(result)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/dependencies/{domain_name}/{container_name}/analyze", response_model=Dict[str, Any])
async def analyze_dependency_graph(
    domain_name: str,
    container_name: str,
    memory: GraphMemoryManager = Depends(get_memory_manager)
):
    """Analyze the dependency graph for a domain."""
    try:
        result = memory.analyze_dependency_graph(domain_name, container_name)
        return json.loads(result)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/dependencies/{from_component}/{to_component}/{domain_name}/{container_name}/path", response_model=Dict[str, Any])
async def find_dependency_path(
    from_component: str,
    to_component: str,
    domain_name: str,
    container_name: str,
    max_depth: int = 5,
    memory: GraphMemoryManager = Depends(get_memory_manager)
):
    """Find dependency paths between two components."""
    try:
        result = memory.find_path(from_component, to_component, domain_name, container_name, max_depth)
        return json.loads(result)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/dependencies/{domain_name}/{container_name}/import", response_model=Dict[str, Any])
async def import_dependencies(
    domain_name: str,
    container_name: str,
    dependencies: List[Dict[str, Any]],
    memory: GraphMemoryManager = Depends(get_memory_manager)
):
    """Import dependencies detected from code analysis."""
    try:
        result = memory.import_dependencies_from_code(dependencies, domain_name, container_name)
        return json.loads(result)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/dependencies", response_model=Dict[str, Any])
async def create_dependency(dependency: Dependency, memory: GraphMemoryManager = Depends(get_memory_manager)):
    """Create a dependency between components."""
    try:
        result = memory.create_dependency(
            dependency.from_component,
            dependency.to_component,
            dependency.domain_name,
            dependency.container_name,
            dependency.dependency_type,
            dependency.properties
        )
        return json.loads(result)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# Entity Management
@router.get("/{project_name}/entities", response_model=Dict[str, Any])
async def get_project_entities(project_name: str, memory: GraphMemoryManager = Depends(get_memory_manager)):
    """Get all entities in a project."""
    try:
        result = memory.get_project_entities(project_name)
        return json.loads(result)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/domains/entities/add", response_model=Dict[str, Any])
async def add_entity_to_domain(entity: DomainEntity, memory: GraphMemoryManager = Depends(get_memory_manager)):
    """Add an entity to a domain."""
    try:
        result = memory.add_entity_to_domain(
            entity.entity_name,
            entity.entity_type,
            entity.domain_name,
            entity.container_name,
            entity.properties
        )
        return json.loads(result)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/domains/entities/remove", response_model=Dict[str, Any])
async def remove_entity_from_domain(entity: DomainEntity, memory: GraphMemoryManager = Depends(get_memory_manager)):
    """Remove an entity from a domain."""
    try:
        result = memory.remove_entity_from_domain(
            entity.entity_name,
            entity.entity_type,
            entity.domain_name,
            entity.container_name
        )
        return json.loads(result)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/domains/{domain_name}/{container_name}/entities", response_model=Dict[str, Any])
async def get_domain_entities(
    domain_name: str,
    container_name: str,
    entity_type: Optional[str] = None,
    memory: GraphMemoryManager = Depends(get_memory_manager)
):
    """Get all entities in a domain."""
    try:
        result = memory.get_domain_entities(domain_name, container_name, entity_type)
        return json.loads(result)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# Version Control Integration
@router.post("/versions/sync", response_model=Dict[str, Any])
async def sync_with_version_control(sync: VersionControlSync, memory: GraphMemoryManager = Depends(get_memory_manager)):
    """Synchronize component versions with version control system data."""
    try:
        result = memory.sync_with_version_control(
            sync.component_name,
            sync.domain_name,
            sync.container_name,
            sync.commit_data
        )
        return json.loads(result)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/memories", response_model=Dict[str, Any])
async def get_all_project_memories(
    project_name: Optional[str] = None,
    memory: GraphMemoryManager = Depends(get_memory_manager)
):
    """
    Get all project entities in the knowledge graph.
    
    Args:
        project_name: Optional project name to scope the query
    """
    try:
        result = memory.get_all_project_memories(project_name)
        return json.loads(result)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e)) 