from typing import Any, Dict, List, Optional, Union, cast
from datetime import datetime

from src.utils import dict_to_json
from src.utils.neo4j_query_utils import sanitize_query_parameters
from src.graph_memory.base_manager import BaseManager
from src.models.response_models import (
    SearchQueryBase, SearchResponse, SearchResultItem,
    ErrorDetail, ErrorResponse, VectorSearchOptions,
    SuccessResponse
)

class SearchManager:
    """Manager for search operations in the knowledge graph."""
    
    def __init__(self, base_manager: BaseManager):
        """
        Initialize the search manager.
        
        Args:
            base_manager: The base manager instance for database operations
        """
        self.base_manager = base_manager
        self.logger = base_manager.logger
    
    def search_entities(
        self, 
        search_term: str, 
        limit: int = 10, 
        entity_types: Optional[List[str]] = None,
        semantic: bool = False
    ) -> str:
        """
        Search for entities in the knowledge graph.
        
        Args:
            search_term: The term to search for
            limit: Maximum number of results to return
            entity_types: Optional list of entity types to filter by
            semantic: Whether to perform semantic (embedding-based) search
            
        Returns:
            JSON string with the search results
        """
        try:
            self.base_manager.ensure_initialized()
            
            # Create a validated search query model
            query_model = SearchQueryBase(
                query=search_term,
                limit=min(limit, 100),  # Cap limit for safety
                semantic=semantic,
                confidence_threshold=None  # Optional parameter
            )
            
            if semantic:
                return self.semantic_search_entities(
                    search_term=query_model.query or "", 
                    limit=query_model.limit,
                    entity_types=entity_types
                )
            
            # Prepare base query parts
            query_parts = ["MATCH (e:Entity)"]
            params = {}
            
            # Add search term filter
            if search_term:
                query_parts.append("WHERE e.name CONTAINS $search_term")
                params["search_term"] = search_term
            
            # Add entity type filter if provided
            if entity_types and len(entity_types) > 0:
                if "WHERE" in query_parts[-1]:
                    query_parts.append("AND e.type IN $entity_types")
                else:
                    query_parts.append("WHERE e.type IN $entity_types")
                params["entity_types"] = entity_types
            
            # Complete the query with ordering and limit
            query_parts.append("RETURN e")
            query_parts.append("ORDER BY e.name")
            query_parts.append(f"LIMIT {query_model.limit}")
            
            # Execute the query with validation
            query = " ".join(query_parts)
            try:
                # Sanitize parameters
                sanitized_params = sanitize_query_parameters(params) if params else None
                
                # Execute read query with validation
                records = self.base_manager.safe_execute_read_query(
                    query,
                    sanitized_params
                )
                
                # Process results into SearchResultItem objects
                results = []
                if records:
                    for record in records:
                        entity = record.get("e")
                        if entity:
                            # Convert Neo4j node to dict
                            entity_dict = dict(entity.items())
                            entity_id = str(entity.id)  # Add the Neo4j ID
                            
                            # Create a properly formatted search result item
                            result_item = SearchResultItem(
                                id=entity_id,
                                name=entity_dict.get("name", "Unknown"),
                                entity_type=entity_dict.get("type", "Entity"),
                                confidence=entity_dict.get("confidence"),
                                score=None,  # No similarity score for non-semantic search
                                snippet=None,  # No snippet for basic search
                                metadata={k: v for k, v in entity_dict.items() 
                                         if k not in ["name", "type", "confidence"]}
                            )
                            results.append(result_item)
                
                # Create a standardized search response
                response = SearchResponse(
                    status="success",
                    timestamp=datetime.now(),
                    message=f"Found {len(results)} entities matching '{search_term}'",
                    results=results,
                    total_count=len(results),
                    query=query_model.model_dump(),
                    is_semantic=False,
                    data=None  # Add missing data parameter
                )
                
                return response.model_dump_json()
                
            except ValueError as e:
                error = ErrorDetail(
                    code="query_validation_error",
                    message=f"Query validation error: {str(e)}",
                    details={"query": query}
                )
                error_response = ErrorResponse(
                    status="error",
                    timestamp=datetime.now(),
                    error=error
                )
                self.logger.error(f"Query validation error: {str(e)}")
                return error_response.model_dump_json()
            
        except Exception as e:
            error = ErrorDetail(
                code="search_error",
                message=f"Error searching entities: {str(e)}",
                details=None
            )
            error_response = ErrorResponse(
                status="error",
                timestamp=datetime.now(),
                error=error
            )
            self.logger.error(f"Error searching entities: {str(e)}")
            return error_response.model_dump_json()
    
    def semantic_search_entities(
        self, 
        search_term: str, 
        limit: int = 10,
        entity_types: Optional[List[str]] = None,
        vector_options: Optional[VectorSearchOptions] = None
    ) -> str:
        """
        Perform a semantic search for entities using vector embeddings.
        
        Args:
            search_term: The term to search for
            limit: Maximum number of results to return
            entity_types: Optional list of entity types to filter by
            vector_options: Optional configuration for vector search
            
        Returns:
            JSON string with the search results
        """
        try:
            if not search_term:
                error = ErrorDetail(
                    code="missing_search_term",
                    message="Search term is required for semantic search",
                    details=None
                )
                error_response = ErrorResponse(
                    status="error",
                    timestamp=datetime.now(),
                    error=error
                )
                return error_response.model_dump_json()
            
            self.base_manager.ensure_initialized()
            
            # Create a validated search query model with semantic=True
            query_model = SearchQueryBase(
                query=search_term,
                limit=min(limit, 100),  # Cap limit for safety
                semantic=True,
                confidence_threshold=None  # Optional parameter
            )
            
            # Use provided vector options or create default ones
            if vector_options is None:
                vector_options = VectorSearchOptions()
            
            # Generate embedding for search term
            embedding = self.base_manager.generate_embedding(search_term)
            if not embedding:
                error = ErrorDetail(
                    code="embedding_generation_error",
                    message="Failed to generate embedding for search term",
                    details={"search_term": search_term}
                )
                error_response = ErrorResponse(
                    status="error",
                    timestamp=datetime.now(),
                    error=error
                )
                self.logger.error(f"Failed to generate embedding for search term: {search_term}")
                return error_response.model_dump_json()
            
            # Prepare vector search query
            vector_query_parts = [
                "CALL db.index.vector.queryNodes($index_name, $k, $embedding)"
            ]
            
            # Add type filter if provided
            if entity_types and len(entity_types) > 0:
                vector_query_parts.append("YIELD node, score")
                vector_query_parts.append("WHERE node.type IN $entity_types")
                vector_query_parts.append("RETURN node as e, score")
            else:
                vector_query_parts.append("YIELD node, score")
                vector_query_parts.append("RETURN node as e, score")
            
            # Build final query
            vector_query = " ".join(vector_query_parts)
            
            # Prepare parameters
            params = {
                "index_name": getattr(self.base_manager, "embedding_index_name", "entity_embedding"),
                "k": query_model.limit,
                "embedding": embedding
            }
            
            if entity_types and len(entity_types) > 0:
                params["entity_types"] = entity_types
            
            try:
                # Sanitize parameters
                sanitized_params = sanitize_query_parameters(params)
                
                # Execute read query with validation
                records = self.base_manager.safe_execute_read_query(
                    vector_query,
                    sanitized_params
                )
                
                # Process results into SearchResultItem objects
                results = []
                if records:
                    for record in records:
                        entity = record.get("e")
                        score = record.get("score")
                        
                        if entity:
                            # Convert Neo4j node to dict
                            entity_dict = dict(entity.items())
                            entity_id = str(entity.id)
                            
                            # Generate a snippet if there's a description field
                            snippet = None
                            if "description" in entity_dict:
                                desc = entity_dict["description"]
                                if desc and isinstance(desc, str):
                                    max_len = 150
                                    snippet = desc if len(desc) <= max_len else desc[:max_len] + "..."
                            
                            # Create a properly formatted search result item
                            result_item = SearchResultItem(
                                id=entity_id,
                                name=entity_dict.get("name", "Unknown"),
                                entity_type=entity_dict.get("type", "Entity"),
                                score=score,  # Similarity score from vector search
                                confidence=entity_dict.get("confidence"),
                                snippet=snippet,  # Include a snippet if available
                                metadata={k: v for k, v in entity_dict.items() 
                                         if k not in ["name", "type", "confidence", "description"]}
                            )
                            results.append(result_item)
                
                # Create a standardized search response
                response = SearchResponse(
                    status="success",
                    timestamp=datetime.now(),
                    message=f"Found {len(results)} semantically similar entities for '{search_term}'",
                    results=results,
                    total_count=len(results),
                    query={
                        **query_model.model_dump(),
                        "vector_options": vector_options.model_dump()
                    },
                    is_semantic=True,
                    data=None  # Add missing data parameter
                )
                
                return response.model_dump_json()
                
            except ValueError as e:
                error = ErrorDetail(
                    code="vector_query_validation_error",
                    message=f"Vector query validation error: {str(e)}",
                    details={"query": vector_query}
                )
                error_response = ErrorResponse(
                    status="error",
                    timestamp=datetime.now(),
                    error=error
                )
                self.logger.error(f"Query validation error: {str(e)}")
                return error_response.model_dump_json()
            
        except Exception as e:
            error = ErrorDetail(
                code="semantic_search_error",
                message=f"Error in semantic search: {str(e)}",
                details=None
            )
            error_response = ErrorResponse(
                status="error",
                timestamp=datetime.now(),
                error=error
            )
            self.logger.error(f"Error in semantic search: {str(e)}")
            return error_response.model_dump_json()
    
    def query_knowledge_graph(self, cypher_query: str, params: Optional[Dict[str, Any]] = None) -> str:
        """
        Execute a custom Cypher query against the knowledge graph.
        
        Args:
            cypher_query: The Cypher query to execute
            params: Optional parameters for the Cypher query
            
        Returns:
            JSON string with the query results
        """
        try:
            self.base_manager.ensure_initialized()
            
            if not cypher_query:
                return dict_to_json({"error": "Cypher query is required"})
            
            # Sanitize query to prevent access to sensitive operations
            forbidden_terms = ["CREATE", "DELETE", "REMOVE", "SET", "MERGE", "DROP", "CALL db.index"]
            for term in forbidden_terms:
                if term in cypher_query.upper():
                    error_msg = f"Forbidden operation detected in query: {term}"
                    self.logger.error(error_msg)
                    return dict_to_json({"error": error_msg})
            
            try:
                # Sanitize parameters
                sanitized_params = sanitize_query_parameters(params or {}) if params else None
                
                # Execute read query with validation
                records = self.base_manager.safe_execute_read_query(
                    cypher_query,
                    sanitized_params
                )
                
                # Process results
                results = []
                if records:
                    for record in records:
                        # Convert record to a dictionary
                        record_dict = {}
                        for key in record.keys():
                            value = record.get(key)
                            
                            # Handle Neo4j Node objects
                            if value is not None and hasattr(value, "items"):
                                record_dict[key] = dict(value.items())
                            else:
                                record_dict[key] = value
                        
                        results.append(record_dict)
                
                # Add summary information
                summary_info = {"counters": {}, "database": None, "time": None}
                
                response = {
                    "results": results,
                    "summary": summary_info
                }
                
                return dict_to_json(response)
            except ValueError as e:
                error_msg = f"Query validation error: {str(e)}"
                self.logger.error(error_msg)
                return dict_to_json({"error": error_msg})
            
        except Exception as e:
            error_msg = f"Error executing custom query: {str(e)}"
            self.logger.error(error_msg)
            return dict_to_json({"error": error_msg})
    
    def search_entity_neighborhoods(self, entity_name: str, max_depth: int = 2, 
                                  max_nodes: int = 50) -> str:
        """
        Search for entity neighborhoods (entity graph exploration).
        
        Args:
            entity_name: The name of the entity to start from
            max_depth: Maximum relationship depth to traverse
            max_nodes: Maximum number of nodes to return
            
        Returns:
            JSON string with the neighborhood graph
        """
        try:
            self.base_manager.ensure_initialized()
            
            if not entity_name:
                return dict_to_json({"error": "Entity name is required"})
            
            # Validate max_depth
            if max_depth < 1:
                max_depth = 1
            elif max_depth > 3:  # Cap for performance
                max_depth = 3
            
            # Validate max_nodes
            if max_nodes < 1:
                max_nodes = 10
            elif max_nodes > 100:  # Cap for performance
                max_nodes = 100
            
            try:
                # Check if entity exists
                check_query = """
                MATCH (e:Entity {name: $name})
                RETURN e
                """
                
                # Sanitize parameters
                sanitized_params = sanitize_query_parameters({"name": entity_name})
                
                # Execute read query with validation
                records = self.base_manager.safe_execute_read_query(
                    check_query,
                    sanitized_params
                )
                
                if not records or len(records) == 0:
                    self.logger.error(f"Entity '{entity_name}' not found")
                    return dict_to_json({"error": f"Entity '{entity_name}' not found"})
                
                # Build neighborhood query
                neighborhood_query = f"""
                MATCH path = (start:Entity {{name: $name}})-[*1..{max_depth}]-(related)
                WHERE related:Entity
                WITH DISTINCT related, start
                LIMIT $max_nodes
                RETURN collect(distinct start) + collect(distinct related) as nodes
                """
                
                # Prepare params for neighborhood query
                neighborhood_params = {
                    "name": entity_name,
                    "max_nodes": max_nodes
                }
                
                # Sanitize parameters
                sanitized_neighborhood_params = sanitize_query_parameters(neighborhood_params)
                
                # Get nodes with validation
                node_records = self.base_manager.safe_execute_read_query(
                    neighborhood_query,
                    sanitized_neighborhood_params
                )
                
                # Get relationships
                relationship_query = f"""
                MATCH (start:Entity {{name: $name}})-[r:RELATES_TO*1..{max_depth}]-(related:Entity)
                WITH DISTINCT r
                LIMIT $max_rels
                RETURN collect(r) as relationships
                """
                
                # Prepare params for relationship query
                relationship_params = {
                    "name": entity_name,
                    "max_rels": max_nodes * 3  # Allow more relationships than nodes
                }
                
                # Sanitize parameters
                sanitized_relationship_params = sanitize_query_parameters(relationship_params)
                
                # Execute relationship query with validation
                rel_records = self.base_manager.safe_execute_read_query(
                    relationship_query,
                    sanitized_relationship_params
                )
                
                # Process results
                nodes = []
                if node_records and len(node_records) > 0:
                    node_list = node_records[0].get("nodes", [])
                    for node in node_list:
                        node_dict = dict(node.items())
                        node_dict["id"] = node.id
                        nodes.append(node_dict)
                
                relationships = []
                if rel_records and len(rel_records) > 0:
                    rel_list = rel_records[0].get("relationships", [])
                    for rel in rel_list:
                        # Extract relationship properties
                        rel_dict = dict(rel.items())
                        rel_dict["id"] = rel.id
                        rel_dict["start_node"] = rel.start_node.id
                        rel_dict["end_node"] = rel.end_node.id
                        rel_dict["type"] = rel.type
                        relationships.append(rel_dict)
                
                # Prepare response
                graph = {
                    "nodes": nodes,
                    "relationships": relationships,
                    "center_entity": entity_name,
                    "max_depth": max_depth,
                    "max_nodes": max_nodes
                }
                
                return dict_to_json({"graph": graph})
            except ValueError as e:
                error_msg = f"Query validation error: {str(e)}"
                self.logger.error(error_msg)
                return dict_to_json({"error": error_msg})
            
        except Exception as e:
            error_msg = f"Error exploring entity neighborhood: {str(e)}"
            self.logger.error(error_msg)
            return dict_to_json({"error": error_msg})
    
    def find_paths_between_entities(self, from_entity: str, to_entity: str, 
                                  max_depth: int = 4) -> str:
        """
        Find all paths between two entities in the knowledge graph.
        
        Args:
            from_entity: The name of the starting entity
            to_entity: The name of the target entity
            max_depth: Maximum relationship depth to traverse
            
        Returns:
            JSON string with all paths found
        """
        try:
            self.base_manager.ensure_initialized()
            
            if not from_entity or not to_entity:
                return dict_to_json({"error": "Both from_entity and to_entity are required"})
            
            # Validate max_depth
            if max_depth < 1:
                max_depth = 1
            elif max_depth > 6:  # Cap for performance
                max_depth = 6
            
            try:
                # Check if entities exist
                check_query = """
                MATCH (from:Entity {name: $from_name})
                MATCH (to:Entity {name: $to_name})
                RETURN from, to
                """
                
                # Prepare params
                check_params = {
                    "from_name": from_entity,
                    "to_name": to_entity
                }
                
                # Sanitize parameters
                sanitized_check_params = sanitize_query_parameters(check_params)
                
                # Execute check query with validation
                records = self.base_manager.safe_execute_read_query(
                    check_query,
                    sanitized_check_params
                )
                
                if not records or len(records) == 0:
                    return dict_to_json({"error": "One or both entities not found"})
                
                # Build path query
                path_query = f"""
                MATCH path = (from:Entity {{name: $from_name}})-[*1..{max_depth}]-(to:Entity {{name: $to_name}})
                WITH path, length(path) as path_length
                ORDER BY path_length ASC
                LIMIT 10
                RETURN path, path_length
                """
                
                # Prepare path query params
                path_params = {
                    "from_name": from_entity,
                    "to_name": to_entity
                }
                
                # Sanitize parameters
                sanitized_path_params = sanitize_query_parameters(path_params)
                
                # Execute path query with validation
                path_records = self.base_manager.safe_execute_read_query(
                    path_query,
                    sanitized_path_params
                )
                
                # Process results
                paths = []
                if path_records:
                    for record in path_records:
                        path = record.get("path")
                        path_length = record.get("path_length")
                        
                        if path:
                            # Extract nodes and relationships
                            path_nodes = []
                            for node in path.nodes:
                                node_dict = dict(node.items())
                                node_dict["id"] = node.id
                                path_nodes.append(node_dict)
                            
                            path_rels = []
                            for rel in path.relationships:
                                rel_dict = dict(rel.items())
                                rel_dict["id"] = rel.id
                                rel_dict["start_node"] = rel.start_node.id
                                rel_dict["end_node"] = rel.end_node.id
                                rel_dict["type"] = rel.type
                                path_rels.append(rel_dict)
                            
                            paths.append({
                                "nodes": path_nodes,
                                "relationships": path_rels,
                                "length": path_length
                            })
                
                return dict_to_json({
                    "paths": paths,
                    "from_entity": from_entity,
                    "to_entity": to_entity,
                    "max_depth": max_depth,
                    "path_count": len(paths)
                })
            except ValueError as e:
                error_msg = f"Query validation error: {str(e)}"
                self.logger.error(error_msg)
                return dict_to_json({"error": error_msg})
            
        except Exception as e:
            error_msg = f"Error finding paths between entities: {str(e)}"
            self.logger.error(error_msg)
            return dict_to_json({"error": error_msg}) 