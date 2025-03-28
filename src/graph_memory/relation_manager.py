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
            
            records, _ = self.base_manager.safe_execute_query(
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
            
            return dict_to_json({"relations": relations})
            
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
            query = """
            MATCH (e1:Entity {name: $from_entity})-[r:$relation_type]->(e2:Entity {name: $to_entity})
            RETURN r
            """
            
            # Convert relation_type to a parameter for safe_execute_query
            params = {
                "from_entity": from_entity,
                "to_entity": to_entity,
                "relation_type": relation_type
            }
            
            # Manually prepare the query with the relation type
            prepared_query = query.replace("$relation_type", relation_type)
            
            records, _ = self.base_manager.safe_execute_query(
                prepared_query,
                {k: v for k, v in params.items() if k != "relation_type"}
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
                
                self.base_manager.safe_execute_query(
                    update_query,
                    update_params
                )
            
            # Get updated relationship
            get_updated_query = f"""
            MATCH (e1:Entity {{name: $from_entity}})-[r:{relation_type}]->(e2:Entity {{name: $to_entity}})
            RETURN e1.name as from_entity, TYPE(r) as relation_type, e2.name as to_entity, 
                   properties(r) as properties
            """
            
            updated_records, _ = self.base_manager.safe_execute_query(
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
            
            return dict_to_json({
                "error": f"Failed to retrieve updated relationship"
            })
            
        except Exception as e:
            error_msg = f"Error updating relationship: {str(e)}"
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
            
            # Delete relationship
            delete_query = f"""
            MATCH (e1:Entity {{name: $from_entity}})-[r:{relation_type}]->(e2:Entity {{name: $to_entity}})
            DELETE r
            RETURN count(r) as deleted_count
            """
            
            records, _ = self.base_manager.safe_execute_query(
                delete_query,
                {"from_entity": from_entity, "to_entity": to_entity}
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
            
        except Exception as e:
            error_msg = f"Error deleting relationship: {str(e)}"
            self.logger.error(error_msg)
            return dict_to_json({"error": error_msg})
    
    def _create_relation_in_neo4j(self, from_entity: str, to_entity: str, relation_type: str, 
                                properties: Optional[Dict[str, Any]] = None) -> None:
        """
        Create a relationship directly in Neo4j.
        
        Args:
            from_entity: The source entity name
            to_entity: The target entity name
            relation_type: The type of the relationship
            properties: Optional dictionary of relationship properties
        """
        if not self.base_manager.neo4j_driver:
            self.logger.debug("Neo4j driver not initialized, skipping relation creation")
            return
        
        try:
            # Check if entities exist
            check_query = """
            MATCH (e1:Entity {name: $from_entity})
            MATCH (e2:Entity {name: $to_entity})
            RETURN e1, e2
            """
            
            records, _ = self.base_manager.safe_execute_query(
                check_query,
                {"from_entity": from_entity, "to_entity": to_entity}
            )
            
            if not records or len(records) == 0:
                raise ValueError(f"One or both entities '{from_entity}' and '{to_entity}' do not exist")
            
            # Build property string if properties exist
            property_string = ""
            params = {"from_entity": from_entity, "to_entity": to_entity}
            
            if properties:
                property_parts = []
                for key, value in properties.items():
                    property_parts.append(f"{key}: ${key}")
                    params[key] = value
                
                if property_parts:
                    property_string = f" {{ {', '.join(property_parts)} }}"
            
            # Create relationship
            create_query = f"""
            MATCH (e1:Entity {{name: $from_entity}})
            MATCH (e2:Entity {{name: $to_entity}})
            MERGE (e1)-[r:{relation_type}{property_string}]->(e2)
            RETURN r
            """
            
            self.base_manager.safe_execute_query(
                create_query,
                params
            )
            
        except Exception as e:
            self.logger.error(f"Error creating relationship in Neo4j: {str(e)}")
            raise
    
    def _convert_to_dict(self, relation: Any) -> Dict[str, Any]:
        """Convert a relation object to a dictionary."""
        if isinstance(relation, dict):
            return relation
        
        try:
            # Try to convert to dict if it has __dict__ attribute
            return relation.__dict__
        except (AttributeError, TypeError):
            # If not dict-like, try to get basic attributes
            result = {}
            
            # Common attributes to try
            for attr in ["from", "to", "relationType", "type"]:
                try:
                    value = getattr(relation, attr)
                    if attr == "type":
                        result["relationType"] = value
                    else:
                        result[attr] = value
                except (AttributeError, TypeError):
                    pass
            
            # Handle "from" attribute which is a reserved word in Python
            if hasattr(relation, "from_entity"):
                result["from"] = relation.from_entity
            
            return result 