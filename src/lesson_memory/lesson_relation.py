from typing import Any, Dict, List, Optional, Union
import time
import json
import logging

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
        self.logger = logging.getLogger(__name__)
        self.relation_manager = RelationManager(base_manager)
    
    def create_lesson_relation(self, container_name: str, from_entity: str, to_entity: str,
                            relation_type: str, properties: Optional[Dict[str, Any]] = None) -> str:
        """
        Create a relationship between two entities in a lesson container.
        
        Args:
            container_name: Name of the container
            from_entity: Name of the source entity
            to_entity: Name of the target entity
            relation_type: Type of the relationship
            properties: Optional properties for the relationship
            
        Returns:
            JSON string with the created relationship
        """
        try:
            self.base_manager.ensure_initialized()
            
            # Check if container exists
            container_query = """
            MATCH (c:LessonContainer {name: $container_name})
            RETURN c
            """
            
            container_records = self.base_manager.safe_execute_read_query(
                container_query,
                {"container_name": container_name}
            )
            
            if not container_records or len(container_records) == 0:
                return dict_to_json({"error": f"Lesson container '{container_name}' not found"})
            
            # Check if entities exist in container
            entities_query = """
            MATCH (c:LessonContainer {name: $container_name})
            MATCH (c)-[:CONTAINS]->(from:Entity {name: $from_entity})
            MATCH (c)-[:CONTAINS]->(to:Entity {name: $to_entity})
            RETURN from, to
            """
            
            # Use safe_execute_read_query for validation (read-only operation)
            entities_records = self.base_manager.safe_execute_read_query(
                entities_query,
                {
                    "container_name": container_name,
                    "from_entity": from_entity,
                    "to_entity": to_entity
                }
            )
            
            if not entities_records or len(entities_records) == 0:
                return dict_to_json({
                    "error": f"Either '{from_entity}' or '{to_entity}' not found in container '{container_name}'"
                })
            
            # Check if relationship already exists
            relation_check_query = """
            MATCH (c:LessonContainer {name: $container_name})
            MATCH (c)-[:CONTAINS]->(from:Entity {name: $from_entity})
            MATCH (c)-[:CONTAINS]->(to:Entity {name: $to_entity})
            MATCH (from)-[r]->(to)
            WHERE type(r) = $relation_type
            RETURN r
            """
            
            # Use safe_execute_read_query for validation (read-only operation)
            relation_records = self.base_manager.safe_execute_read_query(
                relation_check_query,
                {
                    "container_name": container_name,
                    "from_entity": from_entity,
                    "to_entity": to_entity,
                    "relation_type": relation_type
                }
            )
            
            if relation_records and len(relation_records) > 0:
                existing_relation = relation_records[0].get("r")
                if existing_relation:
                    relation_dict = dict(existing_relation.items())
                    relation_dict["type"] = relation_type
                    relation_dict["fromEntity"] = from_entity
                    relation_dict["toEntity"] = to_entity
                    
                    return dict_to_json({
                        "relation": relation_dict,
                        "message": f"Relation of type '{relation_type}' already exists from '{from_entity}' to '{to_entity}'"
                    })
            
            # Prepare relation properties
            relation_props = properties or {}
            relation_props["domain"] = "lesson"
            relation_props["created"] = time.time()
            
            # Create relationship
            create_query = f"""
            MATCH (c:LessonContainer {{name: $container_name}})
            MATCH (c)-[:CONTAINS]->(from:Entity {{name: $from_entity}})
            MATCH (c)-[:CONTAINS]->(to:Entity {{name: $to_entity}})
            CREATE (from)-[r:{relation_type} $properties]->(to)
            RETURN r
            """
            
            # Use safe_execute_write_query for validation (write operation)
            create_records = self.base_manager.safe_execute_write_query(
                create_query,
                {
                    "container_name": container_name,
                    "from_entity": from_entity,
                    "to_entity": to_entity,
                    "properties": relation_props
                }
            )
            
            if create_records and len(create_records) > 0:
                relation = create_records[0].get("r")
                if relation:
                    relation_dict = dict(relation.items())
                    relation_dict["type"] = relation_type
                    relation_dict["fromEntity"] = from_entity
                    relation_dict["toEntity"] = to_entity
                    
                    self.logger.info(f"Created lesson relation: {from_entity} -{relation_type}-> {to_entity}")
                    return dict_to_json({
                        "relation": relation_dict,
                        "message": f"Relation '{relation_type}' created from '{from_entity}' to '{to_entity}'"
                    })
            
            return dict_to_json({"error": f"Failed to create relation from '{from_entity}' to '{to_entity}'"})
                
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
                
                membership_records = self.base_manager.safe_execute_read_query(
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
            records = self.base_manager.safe_execute_read_query(
                query,
                params
            )
            
            # Process results
            relations = []
            if records:
                for record in records:
                    # Safely access record values using dict access pattern
                    relation = {
                        "from": record["from"] if "from" in record else None,
                        "to": record["to"] if "to" in record else None,
                        "type": record["type"] if "type" in record else None
                    }
                    
                    # Add properties
                    properties = record["properties"] if "properties" in record else {}
                    if properties:
                        for key, value in properties.items():
                            relation[key] = value
                    
                    relations.append(relation)
            
            return dict_to_json({"relations": relations})
                
        except Exception as e:
            error_msg = f"Error retrieving lesson relations: {str(e)}"
            self.logger.error(error_msg)
            return dict_to_json({"error": error_msg})
    
    def update_lesson_relation(self, container_name: str, from_entity: str, to_entity: str,
                            relation_type: str, updates: Dict[str, Any]) -> str:
        """
        Update a relationship between two entities.
        
        Args:
            container_name: Name of the container
            from_entity: Name of the source entity
            to_entity: Name of the target entity
            relation_type: Type of the relationship
            updates: Dictionary of property updates
            
        Returns:
            JSON string with the updated relationship
        """
        try:
            self.base_manager.ensure_initialized()
            
            # Check if relationship exists
            relation_query = """
            MATCH (c:LessonContainer {name: $container_name})
            MATCH (c)-[:CONTAINS]->(from:Entity {name: $from_entity})
            MATCH (c)-[:CONTAINS]->(to:Entity {name: $to_entity})
            MATCH (from)-[r]->(to)
            WHERE type(r) = $relation_type
            RETURN r
            """
            
            # Use safe_execute_read_query for validation (read-only operation)
            relation_records = self.base_manager.safe_execute_read_query(
                relation_query,
                {
                    "container_name": container_name,
                    "from_entity": from_entity,
                    "to_entity": to_entity,
                    "relation_type": relation_type
                }
            )
            
            if not relation_records or len(relation_records) == 0:
                return dict_to_json({
                    "error": f"Relation of type '{relation_type}' not found from '{from_entity}' to '{to_entity}'"
                })
            
            # Validate updates - prevent changing core properties
            protected_fields = ["domain", "created", "type", "fromEntity", "toEntity"]
            for field in protected_fields:
                if field in updates:
                    del updates[field]
            
            if not updates:
                # No valid updates provided
                relation = relation_records[0].get("r")
                if relation:
                    relation_dict = dict(relation.items())
                    relation_dict["type"] = relation_type
                    relation_dict["fromEntity"] = from_entity
                    relation_dict["toEntity"] = to_entity
                    
                    return dict_to_json({
                        "relation": relation_dict,
                        "message": "No valid updates provided for the relation"
                    })
                
                return dict_to_json({"error": "Failed to get current relation details"})
            
            # Add lastUpdated timestamp
            updates["lastUpdated"] = time.time()
            
            # Prepare update query with dynamic SET clauses
            set_clauses = []
            for key, value in updates.items():
                set_clauses.append(f"r.{key} = ${key}")
            
            update_query = f"""
            MATCH (c:LessonContainer {{name: $container_name}})
            MATCH (c)-[:CONTAINS]->(from:Entity {{name: $from_entity}})
            MATCH (c)-[:CONTAINS]->(to:Entity {{name: $to_entity}})
            MATCH (from)-[r:{relation_type}]->(to)
            SET {", ".join(set_clauses)}
            RETURN r
            """
            
            # Prepare parameters
            params = {
                "container_name": container_name,
                "from_entity": from_entity,
                "to_entity": to_entity
            }
            for key, value in updates.items():
                params[key] = value
            
            # Use safe_execute_write_query for validation (write operation)
            update_records = self.base_manager.safe_execute_write_query(
                update_query,
                params
            )
            
            if update_records and len(update_records) > 0:
                updated_relation = update_records[0].get("r")
                if updated_relation:
                    relation_dict = dict(updated_relation.items())
                    relation_dict["type"] = relation_type
                    relation_dict["fromEntity"] = from_entity
                    relation_dict["toEntity"] = to_entity
                    
                    self.logger.info(f"Updated lesson relation: {from_entity} -{relation_type}-> {to_entity}")
                    return dict_to_json({
                        "relation": relation_dict,
                        "message": f"Relation '{relation_type}' updated from '{from_entity}' to '{to_entity}'"
                    })
            
            return dict_to_json({"error": f"Failed to update relation"})
                
        except Exception as e:
            error_msg = f"Error updating lesson relation: {str(e)}"
            self.logger.error(error_msg)
            return dict_to_json({"error": error_msg})
    
    def delete_lesson_relation(self, container_name: str, from_entity: str, to_entity: str,
                            relation_type: str) -> str:
        """
        Delete a relationship between two entities.
        
        Args:
            container_name: Name of the container
            from_entity: Name of the source entity
            to_entity: Name of the target entity
            relation_type: Type of the relationship
            
        Returns:
            JSON string with the deletion result
        """
        try:
            self.base_manager.ensure_initialized()
            
            # Check if relationship exists
            relation_query = """
            MATCH (c:LessonContainer {name: $container_name})
            MATCH (c)-[:CONTAINS]->(from:Entity {name: $from_entity})
            MATCH (c)-[:CONTAINS]->(to:Entity {name: $to_entity})
            MATCH (from)-[r]->(to)
            WHERE type(r) = $relation_type
            RETURN r
            """
            
            # Use safe_execute_read_query for validation (read-only operation)
            relation_records = self.base_manager.safe_execute_read_query(
                relation_query,
                {
                    "container_name": container_name,
                    "from_entity": from_entity,
                    "to_entity": to_entity,
                    "relation_type": relation_type
                }
            )
            
            if not relation_records or len(relation_records) == 0:
                return dict_to_json({
                    "error": f"Relation of type '{relation_type}' not found from '{from_entity}' to '{to_entity}'"
                })
            
            # Delete the relationship
            delete_query = f"""
            MATCH (c:LessonContainer {{name: $container_name}})
            MATCH (c)-[:CONTAINS]->(from:Entity {{name: $from_entity}})
            MATCH (c)-[:CONTAINS]->(to:Entity {{name: $to_entity}})
            MATCH (from)-[r:{relation_type}]->(to)
            DELETE r
            """
            
            # Use safe_execute_write_query for validation (write operation)
            self.base_manager.safe_execute_write_query(
                delete_query,
                {
                    "container_name": container_name,
                    "from_entity": from_entity,
                    "to_entity": to_entity
                }
            )
            
            self.logger.info(f"Deleted lesson relation: {from_entity} -{relation_type}-> {to_entity}")
            return dict_to_json({
                "status": "success",
                "message": f"Relation '{relation_type}' deleted from '{from_entity}' to '{to_entity}'"
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
            MATCH (c:LessonContainer {name: $container_name})
            RETURN c
            """
            
            container_records = self.base_manager.safe_execute_read_query(
                container_query,
                {"container_name": container_name}
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
            nodes_records = self.base_manager.safe_execute_read_query(
                nodes_query,
                {"container_name": container_name}
            )
            
            relationships_records = self.base_manager.safe_execute_read_query(
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
            
            lesson_records = self.base_manager.safe_execute_read_query(
                lesson_query,
                {"lesson_name": lesson_name}
            )
            
            if not lesson_records or len(lesson_records) == 0:
                return dict_to_json({"error": f"Lesson '{lesson_name}' not found"})
            
            # Verify context entity exists
            context_query = """
            MATCH (c:Entity {name: $context_entity})
            RETURN c
            """
            
            context_records = self.base_manager.safe_execute_read_query(
                context_query,
                {"context_entity": context_entity}
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
            
            self.base_manager.safe_execute_write_query(
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
            MATCH (c:LessonContainer {name: $container_name})
            RETURN c
            """
            
            container_records = self.base_manager.safe_execute_read_query(
                container_query,
                {"container_name": container_name}
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
            records = self.base_manager.safe_execute_read_query(
                query,
                params
            )
            
            # Process results
            relations = []
            if records:
                for record in records:
                    # Safely access record values using dict access pattern
                    relation = {
                        "from": record["from"] if "from" in record else None,
                        "to": record["to"] if "to" in record else None,
                        "type": record["type"] if "type" in record else None
                    }
                    
                    # Add properties
                    properties = record["properties"] if "properties" in record else {}
                    if properties:
                        for key, value in properties.items():
                            relation[key] = value
                    
                    relations.append(relation)
            
            return dict_to_json({"relations": relations})
                
        except Exception as e:
            error_msg = f"Error retrieving container relations: {str(e)}"
            self.logger.error(error_msg)
            return dict_to_json({"error": error_msg}) 