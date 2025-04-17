from typing import Any, Dict, List, Optional, Union
import time
import json
import logging
from datetime import datetime

from src.utils import dict_to_json, generate_id
from src.graph_memory.base_manager import BaseManager
from src.graph_memory.entity_manager import EntityManager
from src.graph_memory.relation_manager import RelationManager
from src.models.project_memory import (
    RelationshipCreate,
    ErrorResponse,
    ErrorDetail,
    SuccessResponse,
    Metadata
)

class DependencyManager:
    """
    Manager for project component dependencies.
    Handles tracking and analyzing dependencies between project components.
    """
    
    def __init__(self, base_manager: BaseManager):
        """
        Initialize the dependency manager.
        
        Args:
            base_manager: Base manager for graph operations
        """
        self.base_manager = base_manager
        self.logger = logging.getLogger(__name__)
        self.entity_manager = EntityManager(base_manager)
        self.relation_manager = RelationManager(base_manager)
        
        # Define dependency relationship types
        self.dependency_types = [
            "DEPENDS_ON",          # General dependency
            "IMPORTS",             # Code imports
            "USES",                # Component uses another
            "EXTENDS",             # Inheritance/extension
            "IMPLEMENTS",          # Interface implementation
            "CALLS",               # Function/method calls
            "REFERENCES"           # References/mentions
        ]
    
    def create_dependency(self, 
                       relationship: RelationshipCreate,
                       domain_name: str, 
                       container_name: str) -> str:
        """
        Create a dependency relationship between two components.
        
        Args:
            relationship: Relationship model containing source, target and type
            domain_name: Name of the domain
            container_name: Name of the project container
            
        Returns:
            JSON string with the result
        """
        try:
            self.base_manager.ensure_initialized()
            
            # Extract components from relationship
            from_id = relationship.source_id
            to_id = relationship.target_id
            dependency_type = relationship.relationship_type
            
            # Validate dependency type
            if dependency_type not in self.dependency_types:
                error = ErrorDetail(
                    code="INVALID_DEPENDENCY_TYPE",
                    message=f"Invalid dependency type: {dependency_type}. Valid types: {', '.join(self.dependency_types)}",
                    details=None
                )
                return dict_to_json(ErrorResponse(
                    status="error",
                    timestamp=datetime.now(),
                    error=error
                ).model_dump())
            
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
            
            # Check if dependency already exists
            check_query = """
            MATCH (c:Entity {name: $container_name, entityType: 'ProjectContainer'})
            MATCH (d:Entity {name: $domain_name, entityType: 'Domain'})-[:PART_OF]->(c)
            MATCH (from:Entity {id: $from_id})-[:BELONGS_TO]->(d)
            MATCH (to:Entity {id: $to_id})-[:BELONGS_TO]->(d)
            MATCH (from)-[r]->(to)
            WHERE type(r) = $dependency_type
            RETURN r
            """
            
            check_records = self.base_manager.safe_execute_read_query(
                check_query,
                {
                    "container_name": container_name, 
                    "domain_name": domain_name,
                    "from_id": from_id,
                    "to_id": to_id,
                    "dependency_type": dependency_type
                }
            )
            
            if check_records and len(check_records) > 0:
                return dict_to_json(SuccessResponse(
                    status="success",
                    timestamp=datetime.now(),
                    message=f"Dependency of type '{dependency_type}' already exists between components '{from_name}' and '{to_name}'"
                ).model_dump(exclude_none=True))
            
            # Prepare dependency properties
            dependency_props = {}
            if relationship.metadata:
                metadata_dict = relationship.metadata.model_dump(exclude_none=True)
                for key, value in metadata_dict.items():
                    dependency_props[f"metadata_{key}"] = value
            
            dependency_props["created"] = time.time()
            
            # Create dynamic relationship
            create_query = f"""
            MATCH (c:Entity {{name: $container_name, entityType: 'ProjectContainer'}})
            MATCH (d:Entity {{name: $domain_name, entityType: 'Domain'}})-[:PART_OF]->(c)
            MATCH (from:Entity {{id: $from_id}})-[:BELONGS_TO]->(d)
            MATCH (to:Entity {{id: $to_id}})-[:BELONGS_TO]->(d)
            CREATE (from)-[r:{dependency_type} $properties]->(to)
            RETURN r
            """
            
            create_records = self.base_manager.safe_execute_write_query(
                create_query,
                {
                    "container_name": container_name, 
                    "domain_name": domain_name,
                    "from_id": from_id,
                    "to_id": to_id,
                    "properties": dependency_props
                }
            )
            
            if not create_records or len(create_records) == 0:
                error = ErrorDetail(
                    code="DEPENDENCY_CREATION_FAILED",
                    message=f"Failed to create dependency between components '{from_name}' and '{to_name}'",
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
                message=f"Dependency of type '{dependency_type}' created between components '{from_name}' and '{to_name}'"
            ).model_dump(exclude_none=True))
                
        except Exception as e:
            error_msg = f"Error creating dependency: {str(e)}"
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
    
    def get_dependencies(self, component_id: str, domain_name: str, container_name: str,
                      direction: str = "outgoing", dependency_type: Optional[str] = None) -> str:
        """
        Get dependencies for a component.
        
        Args:
            component_id: ID of the component
            domain_name: Name of the domain
            container_name: Name of the project container
            direction: Direction of dependencies ('outgoing', 'incoming', or 'both')
            dependency_type: Optional dependency type to filter by
            
        Returns:
            JSON string with the dependencies
        """
        try:
            self.base_manager.ensure_initialized()
            
            # Validate direction
            valid_directions = ["outgoing", "incoming", "both"]
            if direction not in valid_directions:
                error = ErrorDetail(
                    code="INVALID_DIRECTION",
                    message=f"Invalid direction: {direction}. Valid values: {', '.join(valid_directions)}",
                    details=None
                )
                return dict_to_json(ErrorResponse(
                    status="error",
                    timestamp=datetime.now(),
                    error=error
                ).model_dump())
            
            # Validate dependency type if provided
            if dependency_type and dependency_type not in self.dependency_types:
                error = ErrorDetail(
                    code="INVALID_DEPENDENCY_TYPE",
                    message=f"Invalid dependency type: {dependency_type}. Valid types: {', '.join(self.dependency_types)}",
                    details=None
                )
                return dict_to_json(ErrorResponse(
                    status="error",
                    timestamp=datetime.now(),
                    error=error
                ).model_dump())
            
            # Get component name for better reporting
            name_query = """
            MATCH (c:Entity {name: $container_name, entityType: 'ProjectContainer'})
            MATCH (d:Entity {name: $domain_name, entityType: 'Domain'})-[:PART_OF]->(c)
            MATCH (comp:Entity {id: $component_id})-[:BELONGS_TO]->(d)
            RETURN comp.name as component_name
            """
            
            name_records = self.base_manager.safe_execute_read_query(
                name_query,
                {"container_name": container_name, "domain_name": domain_name, "component_id": component_id}
            )
            
            if not name_records or len(name_records) == 0:
                error = ErrorDetail(
                    code="COMPONENT_NOT_FOUND",
                    message=f"Component with ID '{component_id}' not found in domain '{domain_name}'",
                    details=None
                )
                return dict_to_json(ErrorResponse(
                    status="error",
                    timestamp=datetime.now(),
                    error=error
                ).model_dump())
            
            component_name = name_records[0]["component_name"]
            
            # Build query based on direction and dependency_type
            if direction == "outgoing":
                if dependency_type:
                    query = f"""
                    MATCH (c:Entity {{name: $container_name, entityType: 'ProjectContainer'}})
                    MATCH (d:Entity {{name: $domain_name, entityType: 'Domain'}})-[:PART_OF]->(c)
                    MATCH (from:Entity {{id: $component_id}})-[:BELONGS_TO]->(d)
                    MATCH (to:Entity)-[:BELONGS_TO]->(d)
                    MATCH (from)-[r:{dependency_type}]->(to)
                    RETURN to as other, type(r) as relation_type, r as relation, 'outgoing' as direction
                    """
                else:
                    query = """
                    MATCH (c:Entity {name: $container_name, entityType: 'ProjectContainer'})
                    MATCH (d:Entity {name: $domain_name, entityType: 'Domain'})-[:PART_OF]->(c)
                    MATCH (from:Entity {id: $component_id})-[:BELONGS_TO]->(d)
                    MATCH (to:Entity)-[:BELONGS_TO]->(d)
                    MATCH (from)-[r]->(to)
                    WHERE type(r) IN $dependency_types
                    RETURN to as other, type(r) as relation_type, r as relation, 'outgoing' as direction
                    """
            elif direction == "incoming":
                if dependency_type:
                    query = f"""
                    MATCH (c:Entity {{name: $container_name, entityType: 'ProjectContainer'}})
                    MATCH (d:Entity {{name: $domain_name, entityType: 'Domain'}})-[:PART_OF]->(c)
                    MATCH (to:Entity {{id: $component_id}})-[:BELONGS_TO]->(d)
                    MATCH (from:Entity)-[:BELONGS_TO]->(d)
                    MATCH (from)-[r:{dependency_type}]->(to)
                    RETURN from as other, type(r) as relation_type, r as relation, 'incoming' as direction
                    """
                else:
                    query = """
                    MATCH (c:Entity {name: $container_name, entityType: 'ProjectContainer'})
                    MATCH (d:Entity {name: $domain_name, entityType: 'Domain'})-[:PART_OF]->(c)
                    MATCH (to:Entity {id: $component_id})-[:BELONGS_TO]->(d)
                    MATCH (from:Entity)-[:BELONGS_TO]->(d)
                    MATCH (from)-[r]->(to)
                    WHERE type(r) IN $dependency_types
                    RETURN from as other, type(r) as relation_type, r as relation, 'incoming' as direction
                    """
            else:  # both
                if dependency_type:
                    query = f"""
                    MATCH (c:Entity {{name: $container_name, entityType: 'ProjectContainer'}})
                    MATCH (d:Entity {{name: $domain_name, entityType: 'Domain'}})-[:PART_OF]->(c)
                    MATCH (comp:Entity {{id: $component_id}})-[:BELONGS_TO]->(d)
                    MATCH (other:Entity)-[:BELONGS_TO]->(d)
                    MATCH (comp)-[r1:{dependency_type}]->(other)
                    RETURN other, type(r1) as relation_type, r1 as relation, 'outgoing' as direction
                    UNION
                    MATCH (c:Entity {{name: $container_name, entityType: 'ProjectContainer'}})
                    MATCH (d:Entity {{name: $domain_name, entityType: 'Domain'}})-[:PART_OF]->(c)
                    MATCH (comp:Entity {{id: $component_id}})-[:BELONGS_TO]->(d)
                    MATCH (other:Entity)-[:BELONGS_TO]->(d)
                    MATCH (other)-[r2:{dependency_type}]->(comp)
                    RETURN other, type(r2) as relation_type, r2 as relation, 'incoming' as direction
                    """
                else:
                    query = """
                    MATCH (c:Entity {name: $container_name, entityType: 'ProjectContainer'})
                    MATCH (d:Entity {name: $domain_name, entityType: 'Domain'})-[:PART_OF]->(c)
                    MATCH (comp:Entity {id: $component_id})-[:BELONGS_TO]->(d)
                    MATCH (other:Entity)-[:BELONGS_TO]->(d)
                    MATCH (comp)-[r1]->(other)
                    WHERE type(r1) IN $dependency_types
                    RETURN other, type(r1) as relation_type, r1 as relation, 'outgoing' as direction
                    UNION
                    MATCH (c:Entity {name: $container_name, entityType: 'ProjectContainer'})
                    MATCH (d:Entity {name: $domain_name, entityType: 'Domain'})-[:PART_OF]->(c)
                    MATCH (comp:Entity {id: $component_id})-[:BELONGS_TO]->(d)
                    MATCH (other:Entity)-[:BELONGS_TO]->(d)
                    MATCH (other)-[r2]->(comp)
                    WHERE type(r2) IN $dependency_types
                    RETURN other, type(r2) as relation_type, r2 as relation, 'incoming' as direction
                    """
            
            # Prepare query parameters
            params = {
                "container_name": container_name,
                "domain_name": domain_name,
                "component_id": component_id,
                "dependency_types": self.dependency_types
            }
            
            records = self.base_manager.safe_execute_read_query(query, params)
            
            # Process results
            dependencies = []
            if records:
                for record in records:
                    other_component = dict(record["other"].items())
                    relation_type = record["relation_type"]
                    relation = dict(record["relation"].items())
                    direction = record["direction"]
                    
                    dependency = {
                        "component": other_component,
                        "relation_type": relation_type,
                        "relation_properties": relation,
                        "direction": direction
                    }
                    
                    dependencies.append(dependency)
            
            response = {
                "status": "success",
                "timestamp": datetime.now().isoformat(),
                "message": f"Found {len(dependencies)} dependencies for component '{component_name}'",
                "component_id": component_id,
                "component_name": component_name,
                "domain": domain_name,
                "container": container_name,
                "dependency_count": len(dependencies),
                "dependencies": dependencies
            }
            
            if dependency_type:
                response["dependency_type_filter"] = dependency_type
                
            if direction != "both":
                response["direction_filter"] = direction
                
            return dict_to_json(response)
                
        except Exception as e:
            error_msg = f"Error retrieving dependencies: {str(e)}"
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
    
    def delete_dependency(self, from_component_id: str, to_component_id: str,
                       domain_name: str, container_name: str,
                       dependency_type: str) -> str:
        """
        Delete a dependency relationship between components.
        
        Args:
            from_component_id: ID of the source component
            to_component_id: ID of the target component
            domain_name: Name of the domain
            container_name: Name of the project container
            dependency_type: Type of dependency to delete
            
        Returns:
            JSON string with the result
        """
        try:
            self.base_manager.ensure_initialized()
            
            # Validate dependency type
            if dependency_type not in self.dependency_types:
                error = ErrorDetail(
                    code="INVALID_DEPENDENCY_TYPE",
                    message=f"Invalid dependency type: {dependency_type}. Valid types: {', '.join(self.dependency_types)}",
                    details=None
                )
                return dict_to_json(ErrorResponse(
                    status="error",
                    timestamp=datetime.now(),
                    error=error
                ).model_dump())
            
            # Get component names for more descriptive messages
            name_query = """
            MATCH (c:Entity {name: $container_name, entityType: 'ProjectContainer'})
            MATCH (d:Entity {name: $domain_name, entityType: 'Domain'})-[:PART_OF]->(c)
            MATCH (from:Entity {id: $from_id})-[:BELONGS_TO]->(d)
            MATCH (to:Entity {id: $to_id})-[:BELONGS_TO]->(d)
            RETURN from.name as from_name, to.name as to_name
            """
            
            name_records = self.base_manager.safe_execute_read_query(
                name_query,
                {
                    "container_name": container_name, 
                    "domain_name": domain_name,
                    "from_id": from_component_id,
                    "to_id": to_component_id
                }
            )
            
            from_name = "unknown"
            to_name = "unknown"
            if name_records and len(name_records) > 0:
                from_name = name_records[0]["from_name"]
                to_name = name_records[0]["to_name"]
            
            # Check if dependency exists
            check_query = """
            MATCH (c:Entity {name: $container_name, entityType: 'ProjectContainer'})
            MATCH (d:Entity {name: $domain_name, entityType: 'Domain'})-[:PART_OF]->(c)
            MATCH (from:Entity {id: $from_id})-[:BELONGS_TO]->(d)
            MATCH (to:Entity {id: $to_id})-[:BELONGS_TO]->(d)
            MATCH (from)-[r]->(to)
            WHERE type(r) = $dependency_type
            RETURN r
            """
            
            check_records = self.base_manager.safe_execute_read_query(
                check_query,
                {
                    "container_name": container_name, 
                    "domain_name": domain_name,
                    "from_id": from_component_id,
                    "to_id": to_component_id,
                    "dependency_type": dependency_type
                }
            )
            
            if not check_records or len(check_records) == 0:
                error = ErrorDetail(
                    code="DEPENDENCY_NOT_FOUND",
                    message=f"Dependency of type '{dependency_type}' not found between '{from_name}' and '{to_name}'",
                    details=None
                )
                return dict_to_json(ErrorResponse(
                    status="error",
                    timestamp=datetime.now(),
                    error=error
                ).model_dump())
            
            # Delete dependency
            delete_query = f"""
            MATCH (c:Entity {{name: $container_name, entityType: 'ProjectContainer'}})
            MATCH (d:Entity {{name: $domain_name, entityType: 'Domain'}})-[:PART_OF]->(c)
            MATCH (from:Entity {{id: $from_id}})-[:BELONGS_TO]->(d)
            MATCH (to:Entity {{id: $to_id}})-[:BELONGS_TO]->(d)
            MATCH (from)-[r:{dependency_type}]->(to)
            DELETE r
            """
            
            self.base_manager.safe_execute_write_query(
                delete_query,
                {
                    "container_name": container_name, 
                    "domain_name": domain_name,
                    "from_id": from_component_id,
                    "to_id": to_component_id
                }
            )
            
            return dict_to_json(SuccessResponse(
                status="success",
                timestamp=datetime.now(),
                message=f"Dependency of type '{dependency_type}' deleted between '{from_name}' and '{to_name}'"
            ).model_dump(exclude_none=True))
                
        except Exception as e:
            error_msg = f"Error deleting dependency: {str(e)}"
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
    
    def analyze_dependency_graph(self, domain_name: str, container_name: str) -> str:
        """
        Analyze the dependency graph for a domain.
        
        Args:
            domain_name: Name of the domain
            container_name: Name of the project container
            
        Returns:
            JSON string with the dependency analysis
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
            
            # Count components in the domain
            component_count_query = """
            MATCH (c:Entity {name: $container_name, entityType: 'ProjectContainer'})
            MATCH (d:Entity {name: $domain_name, entityType: 'Domain'})-[:PART_OF]->(c)
            MATCH (comp:Entity)-[:BELONGS_TO]->(d)
            RETURN count(comp) as component_count
            """
            
            count_records = self.base_manager.safe_execute_read_query(
                component_count_query,
                {"container_name": container_name, "domain_name": domain_name}
            )
            
            component_count = count_records[0]["component_count"] if count_records else 0
            
            # Count dependencies by type
            dependency_count_query = """
            MATCH (c:Entity {name: $container_name, entityType: 'ProjectContainer'})
            MATCH (d:Entity {name: $domain_name, entityType: 'Domain'})-[:PART_OF]->(c)
            MATCH (from:Entity)-[:BELONGS_TO]->(d)
            MATCH (to:Entity)-[:BELONGS_TO]->(d)
            MATCH (from)-[r]->(to)
            WHERE type(r) IN $dependency_types
            RETURN type(r) as type, count(r) as count
            """
            
            dependency_records = self.base_manager.safe_execute_read_query(
                dependency_count_query,
                {"container_name": container_name, "domain_name": domain_name, "dependency_types": self.dependency_types}
            )
            
            # Process dependency counts
            dependency_counts = {}
            total_dependencies = 0
            
            if dependency_records:
                for record in dependency_records:
                    rel_type = record["type"]
                    count = record["count"]
                    dependency_counts[rel_type] = count
                    total_dependencies += count
            
            # Find components with most dependencies (outgoing)
            most_dependent_query = """
            MATCH (c:Entity {name: $container_name, entityType: 'ProjectContainer'})
            MATCH (d:Entity {name: $domain_name, entityType: 'Domain'})-[:PART_OF]->(c)
            MATCH (from:Entity)-[:BELONGS_TO]->(d)
            MATCH (to:Entity)-[:BELONGS_TO]->(d)
            MATCH (from)-[r]->(to)
            WHERE type(r) IN $dependency_types
            RETURN from.name as component, from.id as component_id, count(r) as dependency_count
            ORDER BY dependency_count DESC
            LIMIT 5
            """
            
            dependent_records = self.base_manager.safe_execute_read_query(
                most_dependent_query,
                {"container_name": container_name, "domain_name": domain_name, "dependency_types": self.dependency_types}
            )
            
            # Process most dependent components
            most_dependent = []
            if dependent_records:
                for record in dependent_records:
                    component = record["component"]
                    component_id = record["component_id"]
                    count = record["dependency_count"]
                    most_dependent.append({
                        "component_name": component, 
                        "component_id": component_id,
                        "outgoing_dependencies": count
                    })
            
            # Find components with most dependents (incoming)
            most_depended_query = """
            MATCH (c:Entity {name: $container_name, entityType: 'ProjectContainer'})
            MATCH (d:Entity {name: $domain_name, entityType: 'Domain'})-[:PART_OF]->(c)
            MATCH (from:Entity)-[:BELONGS_TO]->(d)
            MATCH (to:Entity)-[:BELONGS_TO]->(d)
            MATCH (from)-[r]->(to)
            WHERE type(r) IN $dependency_types
            RETURN to.name as component, to.id as component_id, count(r) as dependent_count
            ORDER BY dependent_count DESC
            LIMIT 5
            """
            
            depended_records = self.base_manager.safe_execute_read_query(
                most_depended_query,
                {"container_name": container_name, "domain_name": domain_name, "dependency_types": self.dependency_types}
            )
            
            # Process most depended upon components
            most_depended = []
            if depended_records:
                for record in depended_records:
                    component = record["component"]
                    component_id = record["component_id"]
                    count = record["dependent_count"]
                    most_depended.append({
                        "component_name": component, 
                        "component_id": component_id,
                        "incoming_dependencies": count
                    })
            
            # Detect cycles in the dependency graph
            cycle_detection_query = """
            MATCH (c:Entity {name: $container_name, entityType: 'ProjectContainer'})
            MATCH (d:Entity {name: $domain_name, entityType: 'Domain'})-[:PART_OF]->(c)
            MATCH path = (comp:Entity)-[:BELONGS_TO]->(d)
                         -[:PART_OF]->(c)<-[:PART_OF]-(d)<-[:BELONGS_TO]-(comp2:Entity)
                         -[r1:DEPENDS_ON|IMPORTS|USES|EXTENDS|IMPLEMENTS|CALLS|REFERENCES*1..3]->(comp)
            RETURN comp.name as start, comp.id as start_id, [rel in relationships(path) | type(rel)] as rel_types, 
                   [node in nodes(path) | node.name] as cycle_path,
                   length(path) as path_length
            LIMIT 10
            """
            
            cycle_records = self.base_manager.safe_execute_read_query(
                cycle_detection_query,
                {"container_name": container_name, "domain_name": domain_name}
            )
            
            # Process cycle detection
            dependency_cycles = []
            if cycle_records:
                for record in cycle_records:
                    start = record["start"]
                    start_id = record["start_id"]
                    rel_types = record["rel_types"]
                    cycle_path = record["cycle_path"]
                    path_length = record["path_length"]
                    
                    cycle = {
                        "start_component": start,
                        "start_component_id": start_id,
                        "cycle_path": cycle_path,
                        "relationship_types": rel_types,
                        "path_length": path_length
                    }
                    
                    dependency_cycles.append(cycle)
            
            # Calculate dependency density
            density = 0
            if component_count > 1:  # Avoid division by zero
                max_possible_dependencies = component_count * (component_count - 1)
                density = total_dependencies / max_possible_dependencies if max_possible_dependencies > 0 else 0
            
            # Compile results
            analysis = {
                "status": "success",
                "timestamp": datetime.now().isoformat(),
                "message": f"Analyzed dependency graph for domain '{domain_name}' with {component_count} components and {total_dependencies} dependencies",
                "domain": domain_name,
                "container": container_name,
                "component_count": component_count,
                "total_dependencies": total_dependencies,
                "dependency_counts_by_type": dependency_counts,
                "dependency_density": density,
                "most_dependent_components": most_dependent,
                "most_depended_upon_components": most_depended,
                "dependency_cycles": dependency_cycles,
                "dependency_types_analyzed": self.dependency_types
            }
            
            return dict_to_json(analysis)
                
        except Exception as e:
            error_msg = f"Error analyzing dependency graph: {str(e)}"
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
    
    def find_path(self, from_component_id: str, to_component_id: str,
               domain_name: str, container_name: str,
               max_depth: int = 5) -> str:
        """
        Find dependency paths between two components.
        
        Args:
            from_component_id: ID of the source component
            to_component_id: ID of the target component
            domain_name: Name of the domain
            container_name: Name of the project container
            max_depth: Maximum path depth to search
            
        Returns:
            JSON string with the dependency paths
        """
        try:
            self.base_manager.ensure_initialized()
            
            # Get component names for better reporting
            name_query = """
            MATCH (c:Entity {name: $container_name, entityType: 'ProjectContainer'})
            MATCH (d:Entity {name: $domain_name, entityType: 'Domain'})-[:PART_OF]->(c)
            MATCH (from:Entity {id: $from_id})-[:BELONGS_TO]->(d)
            MATCH (to:Entity {id: $to_id})-[:BELONGS_TO]->(d)
            RETURN from.name as from_name, to.name as to_name
            """
            
            name_records = self.base_manager.safe_execute_read_query(
                name_query,
                {
                    "container_name": container_name, 
                    "domain_name": domain_name,
                    "from_id": from_component_id,
                    "to_id": to_component_id
                }
            )
            
            if not name_records or len(name_records) == 0:
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
            
            from_name = name_records[0]["from_name"]
            to_name = name_records[0]["to_name"]
            
            # Find paths between components
            path_query = f"""
            MATCH (c:Entity {{name: $container_name, entityType: 'ProjectContainer'}})
            MATCH (d:Entity {{name: $domain_name, entityType: 'Domain'}})-[:PART_OF]->(c)
            MATCH (from:Entity {{id: $from_id}})-[:BELONGS_TO]->(d)
            MATCH (to:Entity {{id: $to_id}})-[:BELONGS_TO]->(d)
            MATCH path = (from)-[r:DEPENDS_ON|IMPORTS|USES|EXTENDS|IMPLEMENTS|CALLS|REFERENCES*1..{max_depth}]->(to)
            RETURN path, length(path) as path_length
            ORDER BY path_length
            LIMIT 10
            """
            
            path_records = self.base_manager.safe_execute_read_query(
                path_query,
                {
                    "container_name": container_name, 
                    "domain_name": domain_name,
                    "from_id": from_component_id,
                    "to_id": to_component_id
                }
            )
            
            # Process paths
            paths = []
            if path_records:
                for record in path_records:
                    path = record["path"]
                    path_length = record["path_length"]
                    
                    # Extract nodes and relationships
                    nodes = [dict(node.items()) for node in path.nodes]
                    relationships = [
                        {
                            "type": rel.type,
                            "properties": dict(rel.items())
                        } for rel in path.relationships
                    ]
                    
                    path_data = {
                        "length": path_length,
                        "nodes": nodes,
                        "relationships": relationships
                    }
                    
                    paths.append(path_data)
            
            response = {
                "status": "success",
                "timestamp": datetime.now().isoformat(),
                "message": f"Found {len(paths)} dependency paths from '{from_name}' to '{to_name}'",
                "from_component_id": from_component_id,
                "from_component_name": from_name,
                "to_component_id": to_component_id,
                "to_component_name": to_name,
                "domain": domain_name,
                "container": container_name,
                "max_depth": max_depth,
                "paths_found": len(paths),
                "paths": paths
            }
            
            return dict_to_json(response)
                
        except Exception as e:
            error_msg = f"Error finding dependency path: {str(e)}"
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
    
    def import_dependencies_from_code(self, 
                                   dependencies: List[Dict[str, Any]],
                                   domain_name: str, 
                                   container_name: str) -> str:
        """
        Import dependencies detected from code analysis.
        
        Args:
            dependencies: List of dependencies, each with from_component, to_component, and dependency_type
            domain_name: Name of the domain
            container_name: Name of the project container
            
        Returns:
            JSON string with the import result
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
            
            # Process each dependency
            results = []
            success_count = 0
            error_count = 0
            
            for dependency in dependencies:
                # Extract dependency data
                from_component = dependency.get("from_component")
                to_component = dependency.get("to_component")
                dependency_type = dependency.get("dependency_type", "DEPENDS_ON")
                properties = dependency.get("properties", {})
                
                # Validate dependency data
                if not from_component or not to_component:
                    results.append({
                        "status": "error",
                        "message": "Missing from_component or to_component",
                        "dependency": dependency
                    })
                    error_count += 1
                    continue
                
                # Validate dependency type
                if dependency_type not in self.dependency_types:
                    results.append({
                        "status": "error",
                        "message": f"Invalid dependency type: {dependency_type}",
                        "dependency": dependency
                    })
                    error_count += 1
                    continue
                
                # Create a RelationshipCreate model for the dependency
                relationship = RelationshipCreate(
                    source_id=from_component,
                    target_id=to_component,
                    relationship_type=dependency_type
                )
                
                # Create dependency
                result_json = self.create_dependency(
                    relationship, 
                    domain_name,
                    container_name
                )
                
                result = json.loads(result_json)
                
                if "error" in result:
                    results.append({
                        "status": "error",
                        "message": result["error"]["message"] if isinstance(result["error"], dict) and "message" in result["error"] else str(result["error"]),
                        "dependency": dependency
                    })
                    error_count += 1
                else:
                    results.append({
                        "status": "success",
                        "message": result["message"] if "message" in result else "Dependency created",
                        "dependency": dependency
                    })
                    success_count += 1
            
            return dict_to_json({
                "status": "success",
                "timestamp": datetime.now().isoformat(),
                "domain": domain_name,
                "container": container_name,
                "total_dependencies": len(dependencies),
                "success_count": success_count,
                "error_count": error_count,
                "results": results
            })
                
        except Exception as e:
            error_msg = f"Error importing dependencies: {str(e)}"
            self.logger.error(error_msg)
            return dict_to_json({"error": error_msg}) 