from typing import Any, Dict, List, Optional, Union
import time
import json

from src.utils import dict_to_json, generate_id
from src.graph_memory.base_manager import BaseManager
from src.graph_memory.relation_manager import RelationManager

class LessonRelation:
    """
    Manager for lesson-specific relationships.
    Extends relationship operations with lesson-specific semantics.
    """
    
    # Define lesson-specific relationship types
    LESSON_RELATION_TYPES = [
        "ORIGINATED_FROM",  # Lesson originated from component/feature/project
        "SOLVED_WITH",      # Solution technique/pattern/tool
        "PREVENTS",         # Problem prevention
        "BUILDS_ON",        # Learning progression
        "APPLIES_TO",       # Application context
        "SUPERSEDES",       # Version relationship
        "CONFLICTS_WITH",   # Conflicting perspectives
        "PRECEDED_BY",      # Temporal connections
        "CONSOLIDATES",     # Consolidation relationship
        "APPLIED_TO"        # Application tracking
    ]
    
    def __init__(self, base_manager: BaseManager):
        """
        Initialize the lesson relation manager.
        
        Args:
            base_manager: Base manager for graph operations
        """
        self.base_manager = base_manager
        self.logger = base_manager.logger
        self.relation_manager = RelationManager(base_manager)
    
    def create_lesson_relation(self, from_entity: str, to_entity: str, relation_type: str,
                             container_name: Optional[str] = None,
                             properties: Optional[Dict[str, Any]] = None) -> str:
        """
        Create a lesson-specific relationship between entities.
        
        Args:
            from_entity: Source entity name
            to_entity: Target entity name
            relation_type: Type of relationship (should be a lesson relation type)
            container_name: Optional container to verify entity membership
            properties: Optional properties for the relationship
            
        Returns:
            JSON string with the created relation
        """
        try:
            self.base_manager.ensure_initialized()
            
            # Validate relation type
            if relation_type not in self.LESSON_RELATION_TYPES:
                self.logger.error(f"Relation type '{relation_type}' is not a standard lesson relation type")
            
            # Check if source entity exists and is in the container
            if container_name:
                from_container_query = """
                MATCH (c:LessonContainer {name: $container_name})-[:CONTAINS]->(e:Entity {name: $entity_name})
                RETURN e
                """
                
                from_records, _ = self.base_manager.safe_execute_query(
                    from_container_query,
                    {"container_name": container_name, "entity_name": from_entity}
                )
                
                if not from_records or len(from_records) == 0:
                    return dict_to_json({
                        "error": f"Entity '{from_entity}' not found in container '{container_name}'"
                    })
            
            # Prepare relation properties
            relation_properties = properties or {}
            relation_properties["created"] = time.time()  # Add timestamp
            relation_properties["domain"] = "lesson"      # Mark as lesson relation
            
            # Create the relationship
            relation_dict = {
                "from": from_entity,
                "to": to_entity,
                "type": relation_type,
                **relation_properties
            }
            
            relation_result = self.relation_manager.create_relations([relation_dict])
            
            return relation_result
                
        except Exception as e:
            error_msg = f"Error creating lesson relation: {str(e)}"
            self.logger.error(error_msg)
            return dict_to_json({"error": error_msg})
    
    def get_lesson_relations(self, entity_name: Optional[str] = None,
                           relation_type: Optional[str] = None,
                           container_name: Optional[str] = None,
                           direction: str = "both") -> str:
        """
        Get lesson relations, optionally filtering by entity, type, and container.
        
        Args:
            entity_name: Optional entity name to filter by
            relation_type: Optional relation type to filter by
            container_name: Optional container to filter entities by
            direction: Direction of relations ('outgoing', 'incoming', or 'both')
            
        Returns:
            JSON string with the relations
        """
        try:
            self.base_manager.ensure_initialized()
            
            # If container specified but no entity, get all relations in container
            if container_name and not entity_name:
                return self._get_container_relations(container_name, relation_type)
            
            # If entity specified with container, verify membership
            if container_name and entity_name:
                membership_query = """
                MATCH (c:LessonContainer {name: $container_name})-[:CONTAINS]->(e:Entity {name: $entity_name})
                RETURN e
                """
                
                membership_records, _ = self.base_manager.safe_execute_query(
                    membership_query,
                    {"container_name": container_name, "entity_name": entity_name}
                )
                
                if not membership_records or len(membership_records) == 0:
                    return dict_to_json({
                        "error": f"Entity '{entity_name}' not found in container '{container_name}'"
                    })
            
            # Build query based on direction
            if direction == "outgoing":
                query_parts = [
                    "MATCH (e:Entity {name: $entity_name})-[r]->(target:Entity)"
                ]
            elif direction == "incoming":
                query_parts = [
                    "MATCH (source:Entity)-[r]->(e:Entity {name: $entity_name})"
                ]
            else:  # 'both' - default
                query_parts = [
                    "MATCH (e:Entity {name: $entity_name})",
                    "OPTIONAL MATCH (e)-[out]->(outTarget:Entity)",
                    "OPTIONAL MATCH (inSource:Entity)-[in]->(e)",
                    "WITH e, collect(out) as outgoing, collect(in) as incoming",
                    "UNWIND outgoing + incoming as r",
                    "WITH e, startNode(r) as source, endNode(r) as target, r"
                ]
            
            # Add filters
            params = {}
            if entity_name:
                params["entity_name"] = entity_name
            
            if relation_type:
                if direction == "both":
                    query_parts.append("WHERE type(r) = $relation_type")
                else:
                    query_parts.append("WHERE type(r) = $relation_type")
                params["relation_type"] = relation_type
            
            # Add domain filter to get lesson-specific relations
            if direction == "both":
                query_parts.append("AND (r.domain = 'lesson' OR type(r) IN ['ORIGINATED_FROM', 'SOLVED_WITH', 'PREVENTS', 'BUILDS_ON', 'APPLIES_TO', 'SUPERSEDES', 'CONFLICTS_WITH', 'PRECEDED_BY', 'CONSOLIDATES', 'APPLIED_TO'])")
            else:
                if relation_type:
                    query_parts.append("AND (r.domain = 'lesson' OR type(r) IN ['ORIGINATED_FROM', 'SOLVED_WITH', 'PREVENTS', 'BUILDS_ON', 'APPLIES_TO', 'SUPERSEDES', 'CONFLICTS_WITH', 'PRECEDED_BY', 'CONSOLIDATES', 'APPLIED_TO'])")
                else:
                    query_parts.append("WHERE (r.domain = 'lesson' OR type(r) IN ['ORIGINATED_FROM', 'SOLVED_WITH', 'PREVENTS', 'BUILDS_ON', 'APPLIES_TO', 'SUPERSEDES', 'CONFLICTS_WITH', 'PRECEDED_BY', 'CONSOLIDATES', 'APPLIED_TO'])")
            
            # Complete query
            if direction == "both":
                query_parts.append("RETURN source.name as from, target.name as to, type(r) as type, properties(r) as properties")
            elif direction == "outgoing":
                query_parts.append("RETURN e.name as from, target.name as to, type(r) as type, properties(r) as properties")
            else:  # 'incoming'
                query_parts.append("RETURN source.name as from, e.name as to, type(r) as type, properties(r) as properties")
            
            query = "\n".join(query_parts)
            
            # Execute query
            records, _ = self.base_manager.safe_execute_query(
                query,
                params
            )
            
            # Process results
            relations = []
            if records:
                for record in records:
                    relation = {
                        "from": record.get("from"),
                        "to": record.get("to"),
                        "type": record.get("type")
                    }
                    
                    # Add properties
                    properties = record.get("properties", {})
                    if properties:
                        for key, value in properties.items():
                            relation[key] = value
                    
                    relations.append(relation)
            
            return dict_to_json({"relations": relations})
                
        except Exception as e:
            error_msg = f"Error retrieving lesson relations: {str(e)}"
            self.logger.error(error_msg)
            return dict_to_json({"error": error_msg})
    
    def update_lesson_relation(self, from_entity: str, to_entity: str, relation_type: str,
                             updates: Dict[str, Any]) -> str:
        """
        Update a lesson relation.
        
        Args:
            from_entity: Source entity name
            to_entity: Target entity name
            relation_type: Type of relationship
            updates: Dictionary of property updates
            
        Returns:
            JSON string with the updated relation
        """
        try:
            self.base_manager.ensure_initialized()
            
            # Add lastUpdated timestamp
            updates["lastUpdated"] = time.time()
            
            # Use the relation manager's update method
            return self.relation_manager.update_relation(from_entity, to_entity, relation_type, updates)
                
        except Exception as e:
            error_msg = f"Error updating lesson relation: {str(e)}"
            self.logger.error(error_msg)
            return dict_to_json({"error": error_msg})
    
    def delete_lesson_relation(self, from_entity: str, to_entity: str, 
                             relation_type: Optional[str] = None) -> str:
        """
        Delete a lesson relation.
        
        Args:
            from_entity: Source entity name
            to_entity: Target entity name
            relation_type: Optional type of relationship
            
        Returns:
            JSON string with the deletion result
        """
        try:
            self.base_manager.ensure_initialized()
            
            # Verify it's a lesson relation before deletion
            if relation_type and relation_type not in self.LESSON_RELATION_TYPES:
                # Get the relation to check if it has the lesson domain
                query = """
                MATCH (from:Entity {name: $from_name})-[r:RELATES_TO]->(to:Entity {name: $to_name})
                WHERE type(r) = $relation_type
                RETURN r.domain as domain
                """
                
                records, _ = self.base_manager.safe_execute_query(
                    query,
                    {"from_name": from_entity, "to_name": to_entity, "relation_type": relation_type}
                )
                
                is_lesson_relation = False
                if records and len(records) > 0:
                    domain = records[0].get("domain")
                    is_lesson_relation = domain == "lesson"
                
                if not is_lesson_relation:
                    return dict_to_json({
                        "error": f"Relation between '{from_entity}' and '{to_entity}' is not a lesson relation"
                    })
            
            # Use the relation manager's delete method
            if relation_type:
                return self.relation_manager.delete_relation(from_entity, to_entity, relation_type)
            else:
                # When no relation type is specified, only delete lesson relations
                query = """
                MATCH (from:Entity {name: $from_name})-[r]->(to:Entity {name: $to_name})
                WHERE r.domain = 'lesson' OR type(r) IN ['ORIGINATED_FROM', 'SOLVED_WITH', 'PREVENTS', 'BUILDS_ON', 'APPLIES_TO', 'SUPERSEDES', 'CONFLICTS_WITH', 'PRECEDED_BY', 'CONSOLIDATES', 'APPLIED_TO']
                DELETE r
                RETURN count(r) as deleted_count
                """
                
                records, _ = self.base_manager.safe_execute_query(
                    query,
                    {
                        "from_name": from_entity, 
                        "to_name": to_entity
                    }
                )
                
                deleted_count = 0
                if records and len(records) > 0:
                    deleted_count = records[0].get("deleted_count", 0)
                
                if deleted_count > 0:
                    self.logger.info(f"Deleted {deleted_count} lesson relations between '{from_entity}' and '{to_entity}'")
                    return dict_to_json({
                        "status": "success",
                        "message": f"Deleted {deleted_count} lesson relations between '{from_entity}' and '{to_entity}'"
                    })
                
                return dict_to_json({
                    "status": "success",
                    "message": f"No lesson relations found between '{from_entity}' and '{to_entity}'"
                })
                
        except Exception as e:
            error_msg = f"Error deleting lesson relation: {str(e)}"
            self.logger.error(error_msg)
            return dict_to_json({"error": error_msg})
    
    def get_lesson_knowledge_graph(self, container_name: str, depth: int = 2) -> str:
        """
        Get knowledge graph of all entities and relations in a lesson container.
        
        Args:
            container_name: Name of the container
            depth: Maximum relationship depth to include
            
        Returns:
            JSON string with nodes and relationships
        """
        try:
            self.base_manager.ensure_initialized()
            
            # Validate depth
            if depth < 1:
                depth = 1
            elif depth > 3:  # Limit depth for performance
                depth = 3
            
            # Check if container exists
            container_query = """
            MATCH (c:LessonContainer {name: $name})
            RETURN c
            """
            
            container_records, _ = self.base_manager.safe_execute_query(
                container_query,
                {"name": container_name}
            )
            
            if not container_records or len(container_records) == 0:
                return dict_to_json({"error": f"Lesson container '{container_name}' not found"})
            
            # Get all entities in the container
            nodes_query = """
            MATCH (c:LessonContainer {name: $container_name})-[:CONTAINS]->(e:Entity)
            RETURN e
            """
            
            # Get all relationships between container entities
            relationships_query = f"""
            MATCH (c:LessonContainer {{name: $container_name}})-[:CONTAINS]->(e:Entity)
            MATCH path = (e)-[r*1..{depth}]-(related:Entity)
            WHERE (r[0].domain = 'lesson' OR type(r[0]) IN ['ORIGINATED_FROM', 'SOLVED_WITH', 'PREVENTS', 'BUILDS_ON', 'APPLIES_TO', 'SUPERSEDES', 'CONFLICTS_WITH', 'PRECEDED_BY', 'CONSOLIDATES', 'APPLIED_TO'])
            UNWIND r as rel
            WITH DISTINCT rel
            RETURN startNode(rel).name as from, endNode(rel).name as to, type(rel) as type, properties(rel) as properties
            """
            
            # Execute queries
            nodes_records, _ = self.base_manager.safe_execute_query(
                nodes_query,
                {"container_name": container_name}
            )
            
            relationships_records, _ = self.base_manager.safe_execute_query(
                relationships_query,
                {"container_name": container_name}
            )
            
            # Process results
            nodes = []
            if nodes_records:
                for record in nodes_records:
                    entity = record.get("e")
                    if entity:
                        entity_dict = dict(entity.items())
                        nodes.append(entity_dict)
            
            relationships = []
            if relationships_records:
                for record in relationships_records:
                    relation = {
                        "from": record.get("from"),
                        "to": record.get("to"),
                        "type": record.get("type")
                    }
                    
                    # Add properties
                    properties = record.get("properties", {})
                    if properties:
                        for key, value in properties.items():
                            relation[key] = value
                    
                    relationships.append(relation)
            
            return dict_to_json({
                "container": container_name,
                "nodes": nodes,
                "relationships": relationships
            })
                
        except Exception as e:
            error_msg = f"Error retrieving lesson knowledge graph: {str(e)}"
            self.logger.error(error_msg)
            return dict_to_json({"error": error_msg})
    
    def create_supersedes_relation(self, new_lesson: str, old_lesson: str) -> str:
        """
        Create a SUPERSEDES relationship between a new lesson version and old version.
        
        Args:
            new_lesson: New lesson entity name
            old_lesson: Old lesson entity name
            
        Returns:
            JSON string with the created relation
        """
        try:
            # Create a SUPERSEDES relation with timestamp
            relation_properties = {
                "created": time.time(),
                "superseded_date": time.time(),
                "domain": "lesson"
            }
            
            # Use the create_lesson_relation method
            return self.create_lesson_relation(
                new_lesson,
                old_lesson,
                "SUPERSEDES",
                properties=relation_properties
            )
                
        except Exception as e:
            error_msg = f"Error creating supersedes relation: {str(e)}"
            self.logger.error(error_msg)
            return dict_to_json({"error": error_msg})
    
    def track_lesson_application(self, lesson_name: str, context_entity: str,
                               application_notes: Optional[str] = None,
                               success_score: Optional[float] = None) -> str:
        """
        Record that a lesson was applied to a specific context.
        
        Args:
            lesson_name: Name of the lesson
            context_entity: Entity the lesson was applied to
            application_notes: Optional notes about the application
            success_score: Optional score (0.0-1.0) indicating application success
            
        Returns:
            JSON string with the application tracking result
        """
        try:
            self.base_manager.ensure_initialized()
            
            # Verify lesson exists
            lesson_query = """
            MATCH (l:Entity {name: $lesson_name, domain: 'lesson'})
            RETURN l
            """
            
            lesson_records, _ = self.base_manager.safe_execute_query(
                lesson_query,
                {"lesson_name": lesson_name}
            )
            
            if not lesson_records or len(lesson_records) == 0:
                return dict_to_json({"error": f"Lesson '{lesson_name}' not found"})
            
            # Verify context entity exists
            context_query = """
            MATCH (c:Entity {name: $context_name})
            RETURN c
            """
            
            context_records, _ = self.base_manager.safe_execute_query(
                context_query,
                {"context_name": context_entity}
            )
            
            if not context_records or len(context_records) == 0:
                return dict_to_json({"error": f"Context entity '{context_entity}' not found"})
            
            # Create APPLIED_TO relationship
            relation_properties = {
                "applied_date": time.time(),
                "domain": "lesson"
            }
            
            if application_notes:
                relation_properties["application_notes"] = application_notes
                
            if success_score is not None:
                if 0.0 <= success_score <= 1.0:
                    relation_properties["success_score"] = success_score
                else:
                    self.logger.error(f"Success score {success_score} outside valid range (0.0-1.0), clamping")
                    relation_properties["success_score"] = max(0.0, min(1.0, success_score))
            
            # Use create_lesson_relation to create the relationship
            application_result = self.create_lesson_relation(
                lesson_name,
                context_entity,
                "APPLIED_TO",
                properties=relation_properties
            )
            
            # Update lesson's relevance score based on application
            update_query = """
            MATCH (l:Entity {name: $lesson_name})
            SET l.lastRefreshed = datetime(),
                l.relevanceScore = CASE
                    WHEN l.relevanceScore IS NULL THEN $base_score
                    ELSE (l.relevanceScore + $base_score) / 2
                END
            RETURN l
            """
            
            base_score = 0.7
            if success_score is not None:
                base_score = 0.5 + (success_score * 0.5)  # Scale to 0.5-1.0 range
            
            self.base_manager.safe_execute_query(
                update_query,
                {"lesson_name": lesson_name, "base_score": base_score}
            )
            
            return application_result
                
        except Exception as e:
            error_msg = f"Error tracking lesson application: {str(e)}"
            self.logger.error(error_msg)
            return dict_to_json({"error": error_msg})
    
    def _get_container_relations(self, container_name: str, relation_type: Optional[str] = None) -> str:
        """
        Get all relations between entities in a lesson container.
        
        Args:
            container_name: Name of the container
            relation_type: Optional relation type to filter by
            
        Returns:
            JSON string with the relations
        """
        try:
            # Check if container exists
            container_query = """
            MATCH (c:LessonContainer {name: $name})
            RETURN c
            """
            
            container_records, _ = self.base_manager.safe_execute_query(
                container_query,
                {"name": container_name}
            )
            
            if not container_records or len(container_records) == 0:
                return dict_to_json({"error": f"Lesson container '{container_name}' not found"})
            
            # Build query for relations
            query_parts = [
                "MATCH (c:LessonContainer {name: $container_name})-[:CONTAINS]->(from:Entity)",
                "MATCH (c)-[:CONTAINS]->(to:Entity)",
                "MATCH (from)-[r]->(to)"
            ]
            
            params = {"container_name": container_name}
            
            # Add relation type filter if specified
            if relation_type:
                query_parts.append("WHERE type(r) = $relation_type")
                params["relation_type"] = relation_type
            else:
                # Use a different approach that avoids list parameters in the WHERE clause
                query_parts.append("WHERE r.domain = 'lesson' OR type(r) IN ['ORIGINATED_FROM', 'SOLVED_WITH', 'PREVENTS', 'BUILDS_ON', 'APPLIES_TO', 'SUPERSEDES', 'CONFLICTS_WITH', 'PRECEDED_BY', 'CONSOLIDATES', 'APPLIED_TO']")
            
            # Complete query
            query_parts.append("RETURN from.name as from, to.name as to, type(r) as type, properties(r) as properties")
            
            query = "\n".join(query_parts)
            
            # Execute query
            records, _ = self.base_manager.safe_execute_query(
                query,
                params
            )
            
            # Process results
            relations = []
            if records:
                for record in records:
                    relation = {
                        "from": record.get("from"),
                        "to": record.get("to"),
                        "type": record.get("type")
                    }
                    
                    # Add properties
                    properties = record.get("properties", {})
                    if properties:
                        for key, value in properties.items():
                            relation[key] = value
                    
                    relations.append(relation)
            
            return dict_to_json({"relations": relations})
                
        except Exception as e:
            error_msg = f"Error retrieving container relations: {str(e)}"
            self.logger.error(error_msg)
            return dict_to_json({"error": error_msg}) 