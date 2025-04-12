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
        # Ensure manager is initialized
        if not memory.check_connection():
            raise HTTPException(status_code=503, detail="Memory system not initialized")
            
        # Use project_operation with create_project operation
        result = memory.project_operation(
            operation_type="create_project",
            name=project.name,
            description=project.description
        )
        return parse_response(result)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/{project_name}", response_model=Dict[str, Any])
async def get_project(project_name: str, memory: GraphMemoryManager = Depends(get_memory_manager)):
    """Get project container details."""
    try:
        # Ensure manager is initialized
        if not memory.check_connection():
            raise HTTPException(status_code=503, detail="Memory system not initialized")
            
        # Use project_operation for unified interface
        result = memory.project_operation(
            operation_type="get_structure",
            project_id=project_name
        )
        return parse_response(result)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.put("/{project_name}", response_model=Dict[str, Any])
async def update_project(project_name: str, updates: Dict[str, Any], memory: GraphMemoryManager = Depends(get_memory_manager)):
    """Update a project container."""
    try:
        # Ensure manager is initialized
        if not memory.check_connection():
            raise HTTPException(status_code=503, detail="Memory system not initialized")
            
        # Use project_operation for unified interface
        result = memory.project_operation(
            operation_type="update",
            entity_name=project_name,
            updates=updates
        )
        return parse_response(result)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.delete("/{project_name}", response_model=Dict[str, Any])
async def delete_project(project_name: str, memory: GraphMemoryManager = Depends(get_memory_manager)):
    """Delete a project container."""
    try:
        # Ensure manager is initialized
        if not memory.check_connection():
            raise HTTPException(status_code=503, detail="Memory system not initialized")
            
        # Use project_operation for unified interface
        result = memory.project_operation(
            operation_type="delete_entity",
            entity_name=project_name,
            entity_type="Project"
        )
        return parse_response(result)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/", response_model=Dict[str, Any])
async def list_projects(memory: GraphMemoryManager = Depends(get_memory_manager)):
    """List all project containers."""
    try:
        # Ensure manager is initialized
        if not memory.check_connection():
            raise HTTPException(status_code=503, detail="Memory system not initialized")
            
        # Use search_nodes to get all projects
        result = memory.search_nodes("", limit=100, entity_types=["Project"])
        return parse_response(result)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/{project_name}/status", response_model=Dict[str, Any])
async def get_project_status(project_name: str, memory: GraphMemoryManager = Depends(get_memory_manager)):
    """Get status summary of a project container."""
    try:
        # Ensure manager is initialized
        if not memory.check_connection():
            raise HTTPException(status_code=503, detail="Memory system not initialized")
            
        # Use project_operation to get structure which contains status information
        result = memory.project_operation(
            operation_type="get_structure",
            project_id=project_name
        )
        return parse_response(result)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# Domain Management
@router.post("/domains", response_model=Dict[str, Any])
async def create_domain(domain: Domain, memory: GraphMemoryManager = Depends(get_memory_manager)):
    """Create a new domain within a project container."""
    try:
        # Ensure manager is initialized
        if not memory.check_connection():
            raise HTTPException(status_code=503, detail="Memory system not initialized")
            
        # Use project_operation for domain creation
        result = memory.project_operation(
            operation_type="create_domain",
            name=domain.name,
            project_id=domain.container_name,
            description=domain.description,
            properties=domain.properties
        )
        return parse_response(result)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/domains/{domain_name}/{container_name}", response_model=Dict[str, Any])
async def get_domain(domain_name: str, container_name: str, memory: GraphMemoryManager = Depends(get_memory_manager)):
    """Get domain details."""
    try:
        # Ensure manager is initialized
        if not memory.check_connection():
            raise HTTPException(status_code=503, detail="Memory system not initialized")
            
        # Use entity search to get domain details
        # Cypher to retrieve domain with specific name in project
        cypher_query = f"""
        MATCH (d:Domain {{name: $domain_name}})-[:BELONGS_TO]->(p:Project {{name: $project_name}})
        RETURN d
        """
        params = {
            "domain_name": domain_name,
            "project_name": container_name
        }
        
        result = memory.query_knowledge_graph(cypher_query, params)
        return parse_response(result)
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
        # Ensure manager is initialized
        if not memory.check_connection():
            raise HTTPException(status_code=503, detail="Memory system not initialized")
            
        # Use project_operation for entity update
        result = memory.project_operation(
            operation_type="update",
            entity_name=domain_name,
            project_id=container_name,
            entity_type="Domain",
            updates=updates
        )
        return parse_response(result)
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
        # Ensure manager is initialized
        if not memory.check_connection():
            raise HTTPException(status_code=503, detail="Memory system not initialized")
            
        # Use project_operation for entity deletion
        result = memory.project_operation(
            operation_type="delete_entity",
            entity_name=domain_name,
            project_id=container_name,
            entity_type="Domain",
            cascade_delete=delete_components
        )
        return parse_response(result)
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
        # Ensure manager is initialized
        if not memory.check_connection():
            raise HTTPException(status_code=503, detail="Memory system not initialized")
            
        # Use project_operation to get project structure which includes domains
        result = memory.project_operation(
            operation_type="get_structure",
            project_id=container_name
        )
        
        # Extract domains from structure
        parsed_result = parse_response(result)
        domains = []
        
        if "domains" in parsed_result:
            domains = parsed_result["domains"]
        elif "structure" in parsed_result and "domains" in parsed_result["structure"]:
            domains = parsed_result["structure"]["domains"]
            
        return {"domains": domains[:limit]}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# Component Management
@router.post("/components", response_model=Dict[str, Any])
async def create_component(component: Component, memory: GraphMemoryManager = Depends(get_memory_manager)):
    """Create a new component in a domain."""
    try:
        # Ensure manager is initialized
        if not memory.check_connection():
            raise HTTPException(status_code=503, detail="Memory system not initialized")
            
        # Use project_operation with create_component operation
        result = memory.project_operation(
            operation_type="create_component",
            name=component.name,
            component_type=component.component_type,
            domain_name=component.domain_name,
            project_id=component.container_name,
            description=component.description,
            content=component.content,
            metadata=component.metadata
        )
        return parse_response(result)
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
        # Ensure manager is initialized
        if not memory.check_connection():
            raise HTTPException(status_code=503, detail="Memory system not initialized")
            
        # Use custom cypher query to get component details
        cypher_query = """
        MATCH (c:Component {name: $component_name})-[:BELONGS_TO]->(d:Domain {name: $domain_name})-[:BELONGS_TO]->(p:Project {name: $project_name})
        RETURN c
        """
        params = {
            "component_name": name,
            "domain_name": domain_name,
            "project_name": container_name
        }
        
        result = memory.query_knowledge_graph(cypher_query, params)
        return parse_response(result)
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
        # Ensure manager is initialized
        if not memory.check_connection():
            raise HTTPException(status_code=503, detail="Memory system not initialized")
            
        # Use project_operation for updating component
        result = memory.project_operation(
            operation_type="update",
            entity_name=name,
            project_id=container_name,
            domain_name=domain_name,
            entity_type="Component",
            updates=update.updates
        )
        return parse_response(result)
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
        # Ensure manager is initialized
        if not memory.check_connection():
            raise HTTPException(status_code=503, detail="Memory system not initialized")
            
        # Use project_operation for deleting component
        result = memory.project_operation(
            operation_type="delete_entity",
            entity_name=component_id,
            project_id=container_name,
            domain_name=domain_name,
            entity_type="Component"
        )
        return parse_response(result)
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
        # Ensure manager is initialized
        if not memory.check_connection():
            raise HTTPException(status_code=503, detail="Memory system not initialized")
            
        # Use custom cypher to get components in domain
        cypher_query = """
        MATCH (c:Component)-[:BELONGS_TO]->(d:Domain {name: $domain_name})-[:BELONGS_TO]->(p:Project {name: $project_name})
        """
        
        if component_type:
            cypher_query += " WHERE c.component_type = $component_type"
            
        cypher_query += """
        RETURN c
        ORDER BY c.name
        LIMIT $limit
        """
        
        params = {
            "domain_name": domain_name,
            "project_name": container_name,
            "component_type": component_type,
            "limit": limit
        }
        
        result = memory.query_knowledge_graph(cypher_query, params)
        return parse_response(result)
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
        # Ensure manager is initialized
        if not memory.check_connection():
            raise HTTPException(status_code=503, detail="Memory system not initialized")
            
        # Use project_operation for relationship creation
        result = memory.project_operation(
            operation_type="relate",
            source_name=relationship.from_component,
            target_name=relationship.to_component,
            domain_name=relationship.domain_name,
            project_id=relationship.container_name,
            relation_type=relationship.relation_type,
            properties=relationship.properties
        )
        return parse_response(result)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/domains/relationships", response_model=Dict[str, Any])
async def create_domain_relationship(
    relationship: DomainRelationship,
    memory: GraphMemoryManager = Depends(get_memory_manager)
):
    """Create a relationship between two domains."""
    try:
        # Ensure manager is initialized
        if not memory.check_connection():
            raise HTTPException(status_code=503, detail="Memory system not initialized")
            
        # Use project_operation for domain relationship creation
        # Use a custom cypher query for domain relationships
        cypher_query = """
        MATCH (d1:Domain {name: $from_domain})-[:BELONGS_TO]->(p:Project {name: $project_name})
        MATCH (d2:Domain {name: $to_domain})-[:BELONGS_TO]->(p)
        MERGE (d1)-[r:$relation_type]->(d2)
        SET r += $properties
        RETURN d1, r, d2
        """
        params = {
            "from_domain": relationship.from_domain,
            "to_domain": relationship.to_domain,
            "project_name": relationship.container_name,
            "relation_type": relationship.relation_type,
            "properties": relationship.properties or {}
        }
        
        result = memory.query_knowledge_graph(cypher_query, params)
        return parse_response(result)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# Version Management
@router.post("/versions", response_model=Dict[str, Any])
async def create_version(version: Version, memory: GraphMemoryManager = Depends(get_memory_manager)):
    """Create a new version for a component."""
    try:
        # Ensure manager is initialized
        if not memory.check_connection():
            raise HTTPException(status_code=503, detail="Memory system not initialized")
            
        # Use project_operation with add_observation to create version as observation
        metadata = {
            "version_number": version.version_number,
            "commit_hash": version.commit_hash,
            "changes": version.changes
        }
        
        if version.metadata:
            metadata.update(version.metadata)
            
        result = memory.project_operation(
            operation_type="add_observation",
            entity_name=version.component_name,
            project_id=version.container_name,
            domain_name=version.domain_name,
            content=version.content or "",
            observation_type="VERSION",
            metadata=metadata
        )
        return parse_response(result)
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
        # Ensure manager is initialized
        if not memory.check_connection():
            raise HTTPException(status_code=503, detail="Memory system not initialized")
            
        # Use cypher query to get specific version
        if version_number:
            cypher_query = """
            MATCH (c:Component {name: $component_name})-[:BELONGS_TO]->(:Domain {name: $domain_name})-[:BELONGS_TO]->(:Project {name: $project_name})
            MATCH (c)-[:HAS_OBSERVATION]->(o:Observation {observation_type: 'VERSION'})
            WHERE o.metadata.version_number = $version_number
            RETURN o
            """
            params = {
                "component_name": component_name,
                "domain_name": domain_name,
                "project_name": container_name,
                "version_number": version_number
            }
        else:
            # Get latest version if no specific version requested
            cypher_query = """
            MATCH (c:Component {name: $component_name})-[:BELONGS_TO]->(:Domain {name: $domain_name})-[:BELONGS_TO]->(:Project {name: $project_name})
            MATCH (c)-[:HAS_OBSERVATION]->(o:Observation {observation_type: 'VERSION'})
            RETURN o
            ORDER BY o.metadata.version_number DESC
            LIMIT 1
            """
            params = {
                "component_name": component_name,
                "domain_name": domain_name,
                "project_name": container_name
            }
        
        result = memory.query_knowledge_graph(cypher_query, params)
        return parse_response(result)
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
        # Ensure manager is initialized
        if not memory.check_connection():
            raise HTTPException(status_code=503, detail="Memory system not initialized")
            
        # Use cypher query to list versions
        cypher_query = """
        MATCH (c:Component {name: $component_name})-[:BELONGS_TO]->(:Domain {name: $domain_name})-[:BELONGS_TO]->(:Project {name: $project_name})
        MATCH (c)-[:HAS_OBSERVATION]->(o:Observation {observation_type: 'VERSION'})
        RETURN o
        ORDER BY o.metadata.version_number DESC
        LIMIT $limit
        """
        params = {
            "component_name": component_name,
            "domain_name": domain_name,
            "project_name": container_name,
            "limit": limit
        }
        
        result = memory.query_knowledge_graph(cypher_query, params)
        return parse_response(result)
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
        # Ensure manager is initialized
        if not memory.check_connection():
            raise HTTPException(status_code=503, detail="Memory system not initialized")
            
        # Use cypher query to get version history
        cypher_query = """
        MATCH (c:Component {name: $component_name})-[:BELONGS_TO]->(:Domain {name: $domain_name})-[:BELONGS_TO]->(:Project {name: $project_name})
        MATCH (c)-[:HAS_OBSERVATION]->(o:Observation {observation_type: 'VERSION'})
        RETURN o.metadata.version_number as version, o.created_at as date, o.metadata.commit_hash as commit,
               o.metadata.changes as changes
        """
        
        if include_content:
            cypher_query = cypher_query.replace("RETURN o.metadata", "RETURN o.content as content, o.metadata")
            
        cypher_query += " ORDER BY o.metadata.version_number DESC"
        
        params = {
            "component_name": component_name,
            "domain_name": domain_name,
            "project_name": container_name
        }
        
        result = memory.query_knowledge_graph(cypher_query, params)
        return parse_response(result)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/versions/tag", response_model=Dict[str, Any])
async def tag_version(tag: VersionTag, memory: GraphMemoryManager = Depends(get_memory_manager)):
    """Add a tag to a specific version of a component."""
    try:
        # Ensure manager is initialized
        if not memory.check_connection():
            raise HTTPException(status_code=503, detail="Memory system not initialized")
            
        # Use a custom cypher to add a tag to a version
        # Find the version observation and update its metadata to include the tag
        cypher_query = """
        MATCH (c:Component {name: $component_name})-[:BELONGS_TO]->(:Domain {name: $domain_name})-[:BELONGS_TO]->(:Project {name: $project_name})
        MATCH (c)-[:HAS_OBSERVATION]->(o:Observation {observation_type: 'VERSION'})
        WHERE o.metadata.version_number = $version_number
        SET o.metadata.tag_name = $tag_name, o.metadata.tag_description = $tag_description
        RETURN o
        """
        params = {
            "component_name": tag.component_name,
            "domain_name": tag.domain_name,
            "project_name": tag.container_name,
            "version_number": tag.version_number,
            "tag_name": tag.tag_name,
            "tag_description": tag.tag_description or ""
        }
        
        result = memory.query_knowledge_graph(cypher_query, params)
        return parse_response(result)
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
        # Ensure manager is initialized
        if not memory.check_connection():
            raise HTTPException(status_code=503, detail="Memory system not initialized")
            
        # Use cypher to get both versions
        cypher_query = """
        MATCH (c:Component {name: $component_name})-[:BELONGS_TO]->(:Domain {name: $domain_name})-[:BELONGS_TO]->(:Project {name: $project_name})
        MATCH (c)-[:HAS_OBSERVATION]->(o:Observation {observation_type: 'VERSION'})
        WHERE o.metadata.version_number IN [$version1, $version2]
        RETURN o.metadata.version_number as version, o.content as content, o.created_at as created_at
        """
        params = {
            "component_name": component_name,
            "domain_name": domain_name,
            "project_name": container_name,
            "version1": version1,
            "version2": version2
        }
        
        result = memory.query_knowledge_graph(cypher_query, params)
        versions_data = parse_response(result)
        
        # Basic diff comparison (simplified)
        # In a real implementation, you might want to use a proper diff algorithm
        # or return the raw content for client-side diffing
        comparison = {
            "versions": versions_data,
            "comparison_result": "Content comparison requires client-side processing"
        }
        
        return comparison
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
        # Ensure manager is initialized
        if not memory.check_connection():
            raise HTTPException(status_code=503, detail="Memory system not initialized")
            
        # Use cypher query to find dependencies
        if direction.lower() == "outgoing":
            direction_rel = "-[r]->"
        else:
            direction_rel = "<-[r]-"
            
        type_filter = ""
        if dependency_type:
            type_filter = f"AND type(r) = '{dependency_type}'"
            
        cypher_query = f"""
        MATCH (c:Component {{name: $component_name}})-[:BELONGS_TO]->(:Domain {{name: $domain_name}})-[:BELONGS_TO]->(:Project {{name: $project_name}})
        MATCH (c){direction_rel}(other:Component) {type_filter}
        RETURN c.name as source, type(r) as relationship, other.name as target, r as properties
        """
        params = {
            "component_name": component_name,
            "domain_name": domain_name,
            "project_name": container_name
        }
        
        result = memory.query_knowledge_graph(cypher_query, params)
        return parse_response(result)
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
        # Ensure manager is initialized
        if not memory.check_connection():
            raise HTTPException(status_code=503, detail="Memory system not initialized")
            
        # Use project_operation to delete a relationship
        result = memory.project_operation(
            operation_type="delete_relationship",
            source_name=from_component,
            target_name=to_component,
            domain_name=domain_name,
            project_id=container_name,
            relationship_type=dependency_type
        )
        return parse_response(result)
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
        # Ensure manager is initialized
        if not memory.check_connection():
            raise HTTPException(status_code=503, detail="Memory system not initialized")
            
        # Use cypher query to analyze dependencies
        # Count dependencies, find circular deps, etc.
        cypher_query = """
        MATCH (d:Domain {name: $domain_name})-[:BELONGS_TO]->(p:Project {name: $project_name})
        MATCH (c:Component)-[:BELONGS_TO]->(d)
        
        // Get dependency count
        OPTIONAL MATCH (c)-[r:DEPENDS_ON]->(other:Component)-[:BELONGS_TO]->(d)
        WITH d, count(r) as dependency_count, c, collect(other.name) as dependencies
        
        // Find components with most dependencies
        RETURN {
            domain_name: d.name,
            component_count: count(c),
            total_dependencies: dependency_count,
            components_with_dependencies: collect({
                name: c.name, 
                outgoing_dependencies: dependencies,
                dependency_count: size(dependencies)
            })
        } as analysis
        """
        
        params = {
            "domain_name": domain_name,
            "project_name": container_name
        }
        
        result = memory.query_knowledge_graph(cypher_query, params)
        return parse_response(result)
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
        # Ensure manager is initialized
        if not memory.check_connection():
            raise HTTPException(status_code=503, detail="Memory system not initialized")
            
        # Use cypher query to find paths
        cypher_query = """
        MATCH (c1:Component {name: $from_component})-[:BELONGS_TO]->(:Domain {name: $domain_name})-[:BELONGS_TO]->(:Project {name: $project_name}),
              (c2:Component {name: $to_component})-[:BELONGS_TO]->(:Domain {name: $domain_name})
        MATCH p = shortestPath((c1)-[*1..$max_depth]->(c2))
        RETURN [n in nodes(p) | n.name] as path,
               [r in relationships(p) | type(r)] as relationship_types,
               length(p) as path_length
        """
        params = {
            "from_component": from_component,
            "to_component": to_component,
            "domain_name": domain_name,
            "project_name": container_name,
            "max_depth": max_depth
        }
        
        result = memory.query_knowledge_graph(cypher_query, params)
        return parse_response(result)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/dependencies", response_model=Dict[str, Any])
async def create_dependency(dependency: Dependency, memory: GraphMemoryManager = Depends(get_memory_manager)):
    """Create a dependency between components."""
    try:
        # Ensure manager is initialized
        if not memory.check_connection():
            raise HTTPException(status_code=503, detail="Memory system not initialized")
            
        # Use project_operation to create relationship
        result = memory.project_operation(
            operation_type="relate",
            source_name=dependency.from_component,
            target_name=dependency.to_component,
            domain_name=dependency.domain_name,
            project_id=dependency.container_name,
            relation_type=dependency.dependency_type,
            properties=dependency.properties
        )
        return parse_response(result)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# Entity Management
@router.get("/{project_name}/entities", response_model=Dict[str, Any])
async def get_project_entities(project_name: str, memory: GraphMemoryManager = Depends(get_memory_manager)):
    """Get all entities in a project."""
    try:
        # Ensure manager is initialized
        if not memory.check_connection():
            raise HTTPException(status_code=503, detail="Memory system not initialized")
            
        # Use project_operation to get project structure
        result = memory.project_operation(
            operation_type="get_structure",
            project_id=project_name
        )
        return parse_response(result)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/domains/entities/add", response_model=Dict[str, Any])
async def add_entity_to_domain(entity: DomainEntity, memory: GraphMemoryManager = Depends(get_memory_manager)):
    """Add an entity to a domain."""
    try:
        # Ensure manager is initialized
        if not memory.check_connection():
            raise HTTPException(status_code=503, detail="Memory system not initialized")
            
        # Use project_operation to create domain entity
        result = memory.project_operation(
            operation_type="create_domain_entity",
            name=entity.entity_name,
            entity_type=entity.entity_type,
            domain_name=entity.domain_name,
            project_id=entity.container_name,
            properties=entity.properties
        )
        return parse_response(result)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/domains/entities/remove", response_model=Dict[str, Any])
async def remove_entity_from_domain(entity: DomainEntity, memory: GraphMemoryManager = Depends(get_memory_manager)):
    """Remove an entity from a domain."""
    try:
        # Ensure manager is initialized
        if not memory.check_connection():
            raise HTTPException(status_code=503, detail="Memory system not initialized")
            
        # Use cypher query to detach entity from domain without deleting it
        cypher_query = """
        MATCH (e {name: $entity_name})-[r:BELONGS_TO]->(d:Domain {name: $domain_name})-[:BELONGS_TO]->(p:Project {name: $project_name})
        WHERE e.entity_type = $entity_type
        DELETE r
        RETURN e
        """
        params = {
            "entity_name": entity.entity_name,
            "entity_type": entity.entity_type,
            "domain_name": entity.domain_name,
            "project_name": entity.container_name
        }
        
        result = memory.query_knowledge_graph(cypher_query, params)
        return parse_response(result)
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
        # Ensure manager is initialized
        if not memory.check_connection():
            raise HTTPException(status_code=503, detail="Memory system not initialized")
            
        # Use project_operation for each dependency in the list
        results = []
        for dep in dependencies:
            if "from_component" in dep and "to_component" in dep and "dependency_type" in dep:
                result = memory.project_operation(
                    operation_type="relate",
                    source_name=dep["from_component"],
                    target_name=dep["to_component"],
                    domain_name=domain_name,
                    project_id=container_name,
                    relation_type=dep["dependency_type"],
                    properties=dep.get("properties", {})
                )
                results.append(parse_response(result))
                
        return {"results": results, "imported_count": len(results)}
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
        # Ensure manager is initialized
        if not memory.check_connection():
            raise HTTPException(status_code=503, detail="Memory system not initialized")
            
        # Use cypher query to get domain entities
        type_filter = ""
        if entity_type:
            type_filter = f"AND e.entity_type = '{entity_type}'"
            
        cypher_query = f"""
        MATCH (e)-[:BELONGS_TO]->(d:Domain {{name: $domain_name}})-[:BELONGS_TO]->(p:Project {{name: $project_name}})
        WHERE e.name IS NOT NULL {type_filter}
        RETURN e
        """
        params = {
            "domain_name": domain_name,
            "project_name": container_name
        }
        
        result = memory.query_knowledge_graph(cypher_query, params)
        return parse_response(result)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# Version Control Integration
@router.post("/versions/sync", response_model=Dict[str, Any])
async def sync_with_version_control(sync: VersionControlSync, memory: GraphMemoryManager = Depends(get_memory_manager)):
    """Synchronize component versions with version control system data."""
    try:
        # Ensure manager is initialized
        if not memory.check_connection():
            raise HTTPException(status_code=503, detail="Memory system not initialized")
            
        # Process each commit as a version using add_observation
        results = []
        for commit in sync.commit_data:
            version_number = commit.get("commit_hash", "")[:8]  # Use short hash as version
            
            metadata = {
                "version_number": version_number,
                "commit_hash": commit.get("commit_hash", ""),
                "author": commit.get("author", ""),
                "date": commit.get("date", ""),
                "message": commit.get("message", "")
            }
            
            result = memory.project_operation(
                operation_type="add_observation",
                entity_name=sync.component_name,
                project_id=sync.container_name,
                domain_name=sync.domain_name,
                content=commit.get("content", ""),
                observation_type="VERSION",
                metadata=metadata
            )
            results.append(parse_response(result))
            
        return {"results": results, "synced_count": len(results)}
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
        # Ensure manager is initialized
        if not memory.check_connection():
            raise HTTPException(status_code=503, detail="Memory system not initialized")
            
        # Use search_nodes with entity type filter
        entity_types = ["Project", "Domain", "Component"]
        
        if project_name:
            # Use project_operation to get structure of specific project
            result = memory.project_operation(
                operation_type="get_structure",
                project_id=project_name
            )
        else:
            # Use search_nodes to get all project-related entities
            result = memory.search_nodes("", limit=1000, entity_types=entity_types)
            
        return parse_response(result)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# Direct access to project_operation and project_context methods
class ProjectOperation(BaseModel):
    """Model for direct access to project_operation method."""
    operation_type: str = Field(..., description="Type of operation to perform")
    parameters: Dict[str, Any] = Field(default_factory=dict, description="Parameters for the operation")

@router.post("/operation", response_model=Dict[str, Any])
async def direct_project_operation(
    operation: ProjectOperation,
    memory: GraphMemoryManager = Depends(get_memory_manager)
):
    """
    Direct access to project_operation method.
    
    Args:
        operation: Operation type and parameters
        memory: GraphMemoryManager instance
    """
    try:
        # Ensure manager is initialized
        if not memory.check_connection():
            raise HTTPException(status_code=503, detail="Memory system not initialized")
            
        # Execute the project operation with provided parameters
        result = memory.project_operation(
            operation_type=operation.operation_type,
            **operation.parameters
        )
        return parse_response(result)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

class ProjectContextStart(BaseModel):
    """Model for starting a project context."""
    project_name: str = Field(..., description="Name of the project")

@router.post("/context/start", response_model=Dict[str, Any])
async def start_project_context(
    context: ProjectContextStart,
    memory: GraphMemoryManager = Depends(get_memory_manager)
):
    """
    Start a project context for sequential operations.
    
    This simulates entering a project_context context manager.
    """
    try:
        # Ensure manager is initialized
        if not memory.check_connection():
            raise HTTPException(status_code=503, detail="Memory system not initialized")
            
        # Use a custom session ID for the context
        session_id = f"context-{context.project_name}"
        
        # Set the project name for the context
        memory.set_project_name(context.project_name)
        
        return {
            "status": "success",
            "message": f"Project context started for {context.project_name}",
            "session_id": session_id,
            "project_name": context.project_name
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

class ProjectContextOperation(BaseModel):
    """Model for operations within a project context."""
    session_id: str = Field(..., description="Session ID from start_project_context")
    operation_type: str = Field(..., description="Type of operation to perform")
    parameters: Dict[str, Any] = Field(default_factory=dict, description="Parameters for the operation")

@router.post("/context/operation", response_model=Dict[str, Any])
async def project_context_operation(
    operation: ProjectContextOperation,
    memory: GraphMemoryManager = Depends(get_memory_manager)
):
    """
    Execute an operation within a project context.
    
    This simulates operations performed inside a project_context context manager.
    """
    try:
        # Ensure manager is initialized
        if not memory.check_connection():
            raise HTTPException(status_code=503, detail="Memory system not initialized")
        
        # Check if the session ID is valid
        if not operation.session_id.startswith("context-"):
            raise HTTPException(status_code=400, detail="Invalid session ID")
            
        # Extract project name from session ID
        project_name = operation.session_id.replace("context-", "", 1)
        
        # Set the project name for the context
        memory.set_project_name(project_name)
        
        # Execute the project operation with provided parameters
        result = memory.project_operation(
            operation_type=operation.operation_type,
            **operation.parameters
        )
        return parse_response(result)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/context/end", response_model=Dict[str, Any])
async def end_project_context(
    context: ProjectContextOperation,
    memory: GraphMemoryManager = Depends(get_memory_manager)
):
    """
    End a project context.
    
    This simulates exiting a project_context context manager.
    """
    try:
        # Ensure manager is initialized
        if not memory.check_connection():
            raise HTTPException(status_code=503, detail="Memory system not initialized")
            
        # Check if the session ID is valid
        if not context.session_id.startswith("context-"):
            raise HTTPException(status_code=400, detail="Invalid session ID")
            
        # Reset project name to default (empty string instead of None)
        memory.set_project_name("")
        
        return {
            "status": "success",
            "message": f"Project context ended for session {context.session_id}"
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# Advanced API for bulk operations in project context
class BulkProjectOperations(BaseModel):
    """Model for executing multiple operations in a project context."""
    project_name: str = Field(..., description="Name of the project")
    operations: List[Dict[str, Any]] = Field(..., description="List of operations to perform")

@router.post("/bulk", response_model=Dict[str, Any])
async def bulk_project_operations(
    bulk: BulkProjectOperations,
    memory: GraphMemoryManager = Depends(get_memory_manager)
):
    """
    Execute multiple operations in a project context.
    
    This simulates using a project_context context manager for multiple operations.
    """
    try:
        # Ensure manager is initialized
        if not memory.check_connection():
            raise HTTPException(status_code=503, detail="Memory system not initialized")
            
        # Set the project name
        memory.set_project_name(bulk.project_name)
        
        # Execute all operations
        results = []
        for operation in bulk.operations:
            operation_type = operation.pop("operation_type", None)
            if not operation_type:
                raise HTTPException(status_code=400, detail="Missing operation_type in operation")
                
            result = memory.project_operation(
                operation_type=operation_type,
                **operation
            )
            results.append(parse_response(result))
            
        # Reset project name to empty string instead of None
        memory.set_project_name("")
        
        return {
            "status": "success",
            "message": f"Executed {len(results)} operations in project context for {bulk.project_name}",
            "results": results
        }
    except Exception as e:
        # Make sure to reset project name even if there's an error
        memory.set_project_name("")
        raise HTTPException(status_code=500, detail=str(e)) 