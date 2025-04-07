from typing import Any, Dict, List, Optional, Union

from src.utils import dict_to_json
from src.graph_memory.base_manager import BaseManager

class RelationManager:
    """Manager for relationship CRUD operations in the knowledge graph."""
    
    def __init__(self, base_manager: BaseManager):
        """
        Initialize the relationship manager.
        
        Args:
            base_manager: The base manager instance for database operations
        """
        self.base_manager = base_manager
        self.logger = base_manager.logger
    
    def create_relationship(self, relation: Dict[str, Any]) -> str:
        """
        Create a relationship in the knowledge graph.
        
        Args:
            relation: Dictionary with relationship information
            
        Returns:
            JSON string with the created relationship
        """
        try:
            self.base_manager.ensure_initialized()
            
            # Standardize field names to support multiple conventions
            standardized_relation = {}
            
            # Source entity - could be 'from' or 'from_entity'
            if "from" in relation:
                standardized_relation["from"] = relation["from"]
            elif "from_entity" in relation:
                standardized_relation["from"] = relation["from_entity"]
            else:
                return dict_to_json({"error": "Missing source entity (from/from_entity)"})
            
            # Target entity - could be 'to' or 'to_entity'
            if "to" in relation:
                standardized_relation["to"] = relation["to"]
            elif "to_entity" in relation:
                standardized_relation["to"] = relation["to_entity"]
            else:
                return dict_to_json({"error": "Missing target entity (to/to_entity)"})
            
            # Relationship type - could be 'relationType', 'type', or 'relationship_type'
            if "relationType" in relation:
                standardized_relation["relationType"] = relation["relationType"]
            elif "type" in relation:
                standardized_relation["relationType"] = relation["type"]
            elif "relationship_type" in relation:
                standardized_relation["relationType"] = relation["relationship_type"]
            else:
                return dict_to_json({"error": "Missing relationship type (relationType/type/relationship_type)"})
            
            # Copy other properties
            for key, value in relation.items():
                if key not in ["from", "from_entity", "to", "to_entity", "relationType", "type", "relationship_type"]:
                    standardized_relation[key] = value
            
            # If properties are in a separate field, include them
            if "properties" in relation and isinstance(relation["properties"], dict):
                for prop_key, prop_value in relation["properties"].items():
                    if prop_key not in standardized_relation:
                        standardized_relation[prop_key] = prop_value
            
            # Call create_relations with the standardized relation
            return self.create_relations([standardized_relation])
        
        except Exception as e:
            error_msg = f"Error creating relationship: {str(e)}"
            self.logger.error(error_msg)
            return dict_to_json({"error": error_msg})
    
    def create_relations(self, relations: List[Union[Dict, Any]]) -> str:
        """
        Create multiple relationships in the knowledge graph.
        
        Args:
            relations: List of relationships to create
            
        Returns:
            JSON string with the created relationships
        """
        try:
            self.base_manager.ensure_initialized()
            
            created_relations = []
            errors = []
            
            for relation in relations:
                # Process relation
                relation_dict = self._convert_to_dict(relation)
                
                # Extract relation information
                from_entity = relation_dict.get("from", relation_dict.get("from_entity", ""))
                to_entity = relation_dict.get("to", relation_dict.get("to_entity", ""))
                relation_type = relation_dict.get("relationType", relation_dict.get("type", ""))
                
                # Additional properties
                properties = {}
                for key, value in relation_dict.items():
                    if key not in ["from", "from_entity", "to", "to_entity", "relationType", "type"]:
                        properties[key] = value
                
                if from_entity and to_entity and relation_type:
                    # Create the relation in Neo4j
                    try:
                        self._create_relation_in_neo4j(from_entity, to_entity, relation_type, properties)
                        
                        # Add standardized format to created_relations
                        created_relation = {
                            "from": from_entity,
                            "to": to_entity,
                            "relationType": relation_type
                        }
                        
                        if properties:
                            created_relation.update(properties)
                        
                        created_relations.append(created_relation)
                        self.logger.info(f"Created relation: {from_entity} -[{relation_type}]-> {to_entity}")
                    except Exception as e:
                        error = {
                            "from": from_entity,
                            "to": to_entity,
                            "relationType": relation_type,
                            "error": str(e)
                        }
                        errors.append(error)
                        self.logger.error(f"Failed to create relation: {from_entity} -[{relation_type}]-> {to_entity}: {str(e)}")
            
            result = {"created": created_relations}
            if errors:
                result["errors"] = errors
                
            return dict_to_json(result)
                
        except Exception as e:
            error_msg = f"Error creating relations: {str(e)}"
            self.logger.error(error_msg)
            return dict_to_json({"error": error_msg})
    
    def create_relationships(self, relationships: List[Union[Dict, Any]]) -> str:
        """
        Create multiple relationships in the knowledge graph.
        This is an alias for create_relations to maintain API compatibility.
        
        Args:
            relationships: List of relationships to create
            
        Returns:
            JSON string with the created relationships
        """
        return self.create_relations(relationships)
    
    def get_relationships(self, entity_name: Optional[str] = None, relation_type: Optional[str] = None) -> str:
        """Get relationships from the knowledge graph (alias for get_relations)."""
        return self.get_relations(entity_name, relation_type)
    
    def get_relations(self, entity_name: Optional[str] = None, relation_type: Optional[str] = None) -> str:
        """
        Get relationships from the knowledge graph.
        
        Args:
            entity_name: Optional entity name to filter relationships
            relation_type: Optional relationship type to filter relationships
            
        Returns:
            JSON string with the relationships
        """
        try:
            self.base_manager.ensure_initialized()
            
            # Build query based on filters
            query_parts = []
            params = {}
            
            if entity_name:
                query_parts.append("(e1:Entity {name: $entity_name})-[r]->(e2:Entity)")
                params["entity_name"] = entity_name
            else:
                query_parts.append("(e1:Entity)-[r]->(e2:Entity)")
            
            if relation_type:
                query_parts.append(f"TYPE(r) = $relation_type")
                params["relation_type"] = relation_type
            
            query = f"""
            MATCH {query_parts[0]}
            {' WHERE ' + ' AND '.join(query_parts[1:]) if len(query_parts) > 1 else ''}
            RETURN e1.name as from_entity, TYPE(r) as relation_type, e2.name as to_entity, 
                   properties(r) as properties
            """
            
            try:
                # Use safe_execute_read_query for validation
                records = self.base_manager.safe_execute_read_query(
                    query,
                    params
                )
                
                relations = []
                if records:
                    for record in records:
                        # Extract relationship data
                        from_entity = record.get("from_entity")
                        relation_type = record.get("relation_type")
                        to_entity = record.get("to_entity")
                        properties = record.get("properties", {})
                        
                        # Create relation object
                        relation = {
                            "from": from_entity,
                            "to": to_entity,
                            "relationType": relation_type
                        }
                        
                        # Add additional properties
                        if properties:
                            relation.update(properties)
                        
                        relations.append(relation)
                
                # Return with both 'relationships' and 'relations' keys for backward compatibility
                return dict_to_json({
                    "relationships": relations,
                    "relations": relations
                })
            except ValueError as e:
                error_msg = f"Query validation error: {str(e)}"
                self.logger.error(error_msg)
                return dict_to_json({"error": error_msg})
            
        except Exception as e:
            error_msg = f"Error retrieving relations: {str(e)}"
            self.logger.error(error_msg)
            return dict_to_json({"error": error_msg})
    
    def update_relation(self, from_entity: str, to_entity: str, relation_type: str, updates: Dict[str, Any]) -> str:
        """
        Update a relationship in the knowledge graph.
        
        Args:
            from_entity: The source entity name
            to_entity: The target entity name
            relation_type: The type of the relationship
            updates: Dictionary of property updates
            
        Returns:
            JSON string with the updated relationship
        """
        try:
            self.base_manager.ensure_initialized()
            
            # Check if relationship exists
            query = f"""
            MATCH (e1:Entity {{name: $from_entity}})-[r:{relation_type}]->(e2:Entity {{name: $to_entity}})
            RETURN r
            """
            
            # Parameters for the query
            params = {
                "from_entity": from_entity,
                "to_entity": to_entity
            }
            
            try:
                # Use safe_execute_read_query for validation
                records = self.base_manager.safe_execute_read_query(
                    query,
                    params
                )
                
                if not records or len(records) == 0:
                    return dict_to_json({
                        "error": f"Relationship from '{from_entity}' to '{to_entity}' with type '{relation_type}' not found"
                    })
                
                # Build update query with dynamic property updates
                update_set_clauses = []
                update_params = {
                    "from_entity": from_entity,
                    "to_entity": to_entity,
                }
                
                for key, value in updates.items():
                    update_set_clauses.append(f"r.{key} = ${key}")
                    update_params[key] = value
                
                if update_set_clauses:
                    update_query = f"""
                    MATCH (e1:Entity {{name: $from_entity}})-[r:{relation_type}]->(e2:Entity {{name: $to_entity}})
                    SET {', '.join(update_set_clauses)}
                    RETURN r
                    """
                    
                    # Use safe_execute_write_query for validation
                    self.base_manager.safe_execute_write_query(
                        update_query,
                        update_params
                    )
                
                # Get updated relationship
                get_updated_query = f"""
                MATCH (e1:Entity {{name: $from_entity}})-[r:{relation_type}]->(e2:Entity {{name: $to_entity}})
                RETURN e1.name as from_entity, TYPE(r) as relation_type, e2.name as to_entity, 
                       properties(r) as properties
                """
                
                # Use safe_execute_read_query for validation
                updated_records = self.base_manager.safe_execute_read_query(
                    get_updated_query,
                    {"from_entity": from_entity, "to_entity": to_entity}
                )
                
                if updated_records and len(updated_records) > 0:
                    record = updated_records[0]
                    
                    # Create relation object
                    relation = {
                        "from": record.get("from_entity"),
                        "to": record.get("to_entity"),
                        "relationType": record.get("relation_type")
                    }
                    
                    # Add additional properties
                    properties = record.get("properties", {})
                    if properties:
                        relation.update(properties)
                    
                    return dict_to_json({"relation": relation})
                
                return dict_to_json({"error": "Failed to get updated relationship"})
            except ValueError as e:
                error_msg = f"Query validation error: {str(e)}"
                self.logger.error(error_msg)
                return dict_to_json({"error": error_msg})
            
        except Exception as e:
            error_msg = f"Error updating relation: {str(e)}"
            self.logger.error(error_msg)
            return dict_to_json({"error": error_msg})
    
    def delete_relation(self, from_entity: str, to_entity: str, relation_type: str) -> str:
        """
        Delete a relationship from the knowledge graph.
        
        Args:
            from_entity: The source entity name
            to_entity: The target entity name
            relation_type: The type of the relationship
            
        Returns:
            JSON string with the result of the deletion
        """
        try:
            self.base_manager.ensure_initialized()
            
            # Delete relation query
            delete_query = f"""
            MATCH (e1:Entity {{name: $from_entity}})-[r:{relation_type}]->(e2:Entity {{name: $to_entity}})
            DELETE r
            RETURN count(r) as deleted_count
            """
            
            # Parameters for the query
            params = {
                "from_entity": from_entity,
                "to_entity": to_entity
            }
            
            try:
                # Use safe_execute_write_query for validation
                records = self.base_manager.safe_execute_write_query(
                    delete_query,
                    params
                )
                
                deleted_count = 0
                if records and len(records) > 0:
                    deleted_count = records[0].get("deleted_count", 0)
                
                if deleted_count > 0:
                    return dict_to_json({
                        "status": "success", 
                        "message": f"Relationship from '{from_entity}' to '{to_entity}' with type '{relation_type}' deleted"
                    })
                else:
                    return dict_to_json({
                        "status": "success", 
                        "message": f"Relationship from '{from_entity}' to '{to_entity}' with type '{relation_type}' not found"
                    })
            except ValueError as e:
                error_msg = f"Query validation error: {str(e)}"
                self.logger.error(error_msg)
                return dict_to_json({"error": error_msg})
            
        except Exception as e:
            error_msg = f"Error deleting relation: {str(e)}"
            self.logger.error(error_msg)
            return dict_to_json({"error": error_msg})
    
    def _create_relation_in_neo4j(self, from_entity: str, to_entity: str, relation_type: str, 
                                properties: Optional[Dict[str, Any]] = None) -> None:
        """Create a relationship directly in Neo4j."""
        if not self.base_manager.neo4j_driver:
            self.logger.debug("Neo4j driver not initialized, skipping relation creation")
            return
        
        try:
            # Create basic query
            if properties and len(properties) > 0:
                # Build property string for query
                property_clauses = []
                params = {
                    "from_entity": from_entity,
                    "to_entity": to_entity
                }
                
                for key, value in properties.items():
                    property_clauses.append(f"{key}: ${key}")
                    params[key] = value
                
                # Create relation with properties
                query = f"""
                MATCH (e1:Entity {{name: $from_entity}}), (e2:Entity {{name: $to_entity}})
                MERGE (e1)-[r:{relation_type} {{{', '.join(property_clauses)}}}]->(e2)
                RETURN r
                """
                
                # Use safe_execute_write_query for validation
                self.base_manager.safe_execute_write_query(
                    query,
                    params
                )
            else:
                # Create relation without properties
                query = f"""
                MATCH (e1:Entity {{name: $from_entity}}), (e2:Entity {{name: $to_entity}})
                MERGE (e1)-[r:{relation_type}]->(e2)
                RETURN r
                """
                
                # Use safe_execute_write_query for validation
                self.base_manager.safe_execute_write_query(
                    query,
                    {"from_entity": from_entity, "to_entity": to_entity}
                )
                
            self.logger.debug(f"Created relation in Neo4j: {from_entity} -[{relation_type}]-> {to_entity}")
                
        except ValueError as e:
            self.logger.error(f"Query validation error: {str(e)}")
            raise
        except Exception as e:
            self.logger.error(f"Error creating relation in Neo4j: {str(e)}")
            raise
    
    def _convert_to_dict(self, relation: Any) -> Dict[str, Any]:
        """Convert a relation object to a dictionary."""
        if isinstance(relation, dict):
            # Verify all values are serializable
            for key, value in relation.items():
                if isinstance(value, (list, tuple)):
                    # Check items in lists/tuples
                    for item in value:
                        if not self._is_serializable(item):
                            raise ValueError(f"Relation has non-serializable value in list for '{key}': {type(item)}")
                elif not self._is_serializable(value):
                    raise ValueError(f"Relation has non-serializable value for '{key}': {type(value)}")
            return relation
        
        try:
            # Try to convert to dict if it has __dict__ attribute
            result = relation.__dict__
            # Verify all values are serializable
            for key, value in result.items():
                if isinstance(value, (list, tuple)):
                    # Check items in lists/tuples
                    for item in value:
                        if not self._is_serializable(item):
                            raise ValueError(f"Relation has non-serializable value in list for '{key}': {type(item)}")
                elif not self._is_serializable(value):
                    raise ValueError(f"Relation has non-serializable value for '{key}': {type(value)}")
            return result
        except (AttributeError, TypeError):
            # If not dict-like, try to get basic attributes
            result = {}
            
            # Common attributes to try
            for attr in ["from", "from_entity", "to", "to_entity", "relationType", "type"]:
                try:
                    value = getattr(relation, attr)
                    if not self._is_serializable(value):
                        raise ValueError(f"Relation has non-serializable value for '{attr}': {type(value)}")
                    result[attr] = value
                except (AttributeError, TypeError):
                    pass
            
            return result
            
    def _is_serializable(self, value: Any) -> bool:
        """Check if a value is serializable for Neo4j."""
        if value is None:
            return True
        # Neo4j accepts these primitive types
        if isinstance(value, (str, int, float, bool)):
            return True
        # Lists and arrays are fine if their items are serializable
        if isinstance(value, (list, tuple)):
            return all(self._is_serializable(item) for item in value)
        # Small dictionaries with serializable values are ok
        if isinstance(value, dict):
            return all(isinstance(k, str) and self._is_serializable(v) for k, v in value.items())
        # Reject anything else
        return False 