#!/usr/bin/env python3
"""
Core Memory Tools with Pydantic Integration

This module implements MCP tools for the core memory system using
Pydantic models for validation and serialization.
"""

import json
import datetime
import os
import copy
from typing import Dict, List, Any, Optional, Union, cast

from src.logger import get_logger
from src.models.core_memory.models import (
    EntityCreate, EntitiesCreate, RelationshipCreate, RelationshipsCreate,
    EntityObservation, ObservationsCreate, SearchQuery, EntityDelete,
    RelationshipDelete, ObservationDelete, create_error_response,
    create_success_response, model_to_json, model_to_dict
)

# Initialize logger
logger = get_logger()

def register_core_tools(server, get_client_manager):
    """Register core memory tools with the server."""
    
    @server.tool()
    async def create_entities(entities: List[Dict[str, Any]], client_id: Optional[str] = None) -> str:
        """
        Create multiple new entities in the knowledge graph.
        
        Each entity should have a name, entity type, and optional observations.
        
        Args:
            entities: List of entity objects to create
            client_id: Optional client ID for identifying the connection
        
        Returns:
            JSON response with operation result
        """
        try:
            # Initialize error tracking
            invalid_entities = []
            
            # Basic input validation
            if entities is None:
                error_response = create_error_response(
                    message="Entities list cannot be None",
                    code="invalid_input",
                    details={"invalid_entities": invalid_entities}
                )
                return model_to_json(error_response)
                
            # Ensure entities is a list
            if not isinstance(entities, list):
                error_response = create_error_response(
                    message="Entities must be provided as a list",
                    code="invalid_input_type"
                )
                return model_to_json(error_response)
                
            # Check if list is empty
            if len(entities) == 0:
                error_response = create_error_response(
                    message="Entities list cannot be empty",
                    code="empty_input"
                )
                return model_to_json(error_response)
                
            # Validate list size to prevent excessive operations
            if len(entities) > 1000:
                error_response = create_error_response(
                    message=f"Too many entities provided ({len(entities)}). Maximum allowed is 1000.",
                    code="input_limit_exceeded"
                )
                return model_to_json(error_response)
            
            # Sanitize client_id if provided
            if client_id:
                client_id = str(client_id).strip()
                if len(client_id) > 100:  # Prevent excessively long IDs
                    logger.warn(f"Client ID too long, truncating: {client_id[:20]}...")
                    client_id = client_id[:100]
                
            logger.debug(f"Creating {len(entities)} entities", context={"entity_count": len(entities)})
        
            # Convert input to Pydantic models for validation
            try:
                # Format entities to match the expected structure
                formatted_entities = []
                max_name_length = 500  # Maximum reasonable length for entity names
                max_type_length = 100  # Maximum reasonable length for entity types
                max_observation_length = 10000  # Maximum length for observations
                
                for i, entity_data in enumerate(entities):
                    # Check if entity is a valid dictionary
                    if not isinstance(entity_data, dict):
                        invalid_entities.append({"index": i, "reason": "Entity must be an object"})
                        continue
                    
                    # Extract entity name with multiple field name options
                    entity_name = entity_data.get("entity_id") or entity_data.get("name")
                    if not entity_name:
                        invalid_entities.append({"index": i, "reason": "Entity name is required"})
                        continue
                        
                    # Sanitize name
                    entity_name = str(entity_name).strip()
                    if len(entity_name) == 0:
                        invalid_entities.append({"index": i, "reason": "Entity name cannot be empty"})
                        continue
                        
                    # Check name length
                    if len(entity_name) > max_name_length:
                        entity_name = entity_name[:max_name_length]
                        logger.warn(f"Entity name too long, truncating: {entity_name[:20]}...")
                        
                    # Extract entity type with multiple field name options
                    entity_type = entity_data.get("entity_type") or entity_data.get("entityType") or entity_data.get("type")
                    if not entity_type:
                        entity_type = "Unknown"  # Use default type if not provided
                    else:
                        entity_type = str(entity_type).strip()
                        if len(entity_type) == 0:
                            entity_type = "Unknown"
                            
                        # Check type length
                        if len(entity_type) > max_type_length:
                            entity_type = entity_type[:max_type_length]
                            logger.warn(f"Entity type too long, truncating: {entity_type[:20]}...")
                    
                    # Prepare entity object
                    entity = {
                        "name": entity_name,
                        "entity_type": entity_type,
                        "observations": []
                    }
                    
                    # Handle observations: could be in entity.observations or as separate metadata
                    if "observations" in entity_data:
                        if isinstance(entity_data["observations"], list):
                            # Process and sanitize each observation
                            sanitized_observations = []
                            for obs in entity_data["observations"]:
                                if obs and isinstance(obs, str):
                                    # Truncate if needed
                                    if len(obs) > max_observation_length:
                                        obs = obs[:max_observation_length]
                                    sanitized_observations.append(obs)
                            entity["observations"] = sanitized_observations
                        else:
                            logger.warn(f"Observations for entity {entity_name} is not a list, ignoring")
                    elif "content" in entity_data:
                        content = str(entity_data["content"])
                        if len(content) > max_observation_length:
                            content = content[:max_observation_length]
                        entity["observations"] = [content]
                        
                    # Add any metadata
                    if "metadata" in entity_data and isinstance(entity_data["metadata"], dict):
                        entity["metadata"] = entity_data["metadata"]
                    
                    # Add client_id to metadata if provided
                    if client_id:
                        if "metadata" not in entity:
                            entity["metadata"] = {}
                        entity["metadata"]["client_id"] = client_id
                    
                    formatted_entities.append(entity)
                
                # Check if any entities were valid after sanitization
                if not formatted_entities:
                    error_response = create_error_response(
                        message="No valid entities to create after validation",
                        code="validation_error",
                        details={"invalid_entities": invalid_entities}
                    )
                    return model_to_json(error_response)
                
                # If some entities were invalid, log them
                if invalid_entities:
                    logger.warn(f"Found {len(invalid_entities)} invalid entities out of {len(entities)} total")
                
                # Validate using Pydantic model
                entities_model = EntitiesCreate(entities=[EntityCreate(**entity) for entity in formatted_entities])
                
            except ValueError as e:
                logger.error(f"Validation error for entities: {str(e)}")
                error_response = create_error_response(
                    message=f"Invalid entity data: {str(e)}",
                    code="validation_error",
                    details={"invalid_entities": invalid_entities}
                )
                return model_to_json(error_response)
            
            # Get the client-specific graph manager
            client_graph_manager = get_client_manager(client_id)
            
            # Call the graph manager with the validated entities
            result = client_graph_manager.create_entities([model_to_dict(entity) for entity in entities_model.entities])
            
            # Parse the result
            try:
                if isinstance(result, str):
                    parsed_result = json.loads(result)
                else:
                    parsed_result = result
                
                if isinstance(parsed_result, dict) and parsed_result.get("status") == "error":
                    error_response = create_error_response(
                        message=parsed_result.get("message", "Unknown error"),
                        code="entity_creation_error"
                    )
                    return model_to_json(error_response)
                
                # Return the success response
                success_response = create_success_response(
                    message=f"Successfully created {len(formatted_entities)} entities",
                    data=parsed_result
                )
                
                # Include information about skipped entities if any
                if invalid_entities:
                    success_response.data["skipped_entities"] = invalid_entities
                
                return model_to_json(success_response)
                
            except json.JSONDecodeError:
                # If result is not valid JSON, return it as-is (legacy compatibility)
                return result
            
        except Exception as e:
            logger.error(f"Error creating entities: {str(e)}", exc_info=True)
            error_response = create_error_response(
                message=f"Error creating entities: {str(e)}",
                code="entity_creation_error",
                details={"entity_count": len(entities) if isinstance(entities, list) else 0}
            )
            return model_to_json(error_response)

    @server.tool()
    async def create_relations(relations: List[Dict[str, Any]], client_id: Optional[str] = None) -> str:
        """
        Create multiple new relations in the knowledge graph.
        
        Each relation connects a source entity to a target entity with a specific relationship type.
        
        Args:
            relations: List of relation objects
            client_id: Optional client ID for identifying the connection
        
        Returns:
            JSON response with operation result
        """
        try:
            # Initialize error tracking
            invalid_relations = []
            
            # Basic input validation
            if relations is None:
                error_response = create_error_response(
                    message="Relations list cannot be None",
                    code="invalid_input",
                    details={"invalid_relations": invalid_relations}
                )
                return model_to_json(error_response)
                
            # Ensure relations is a list
            if not isinstance(relations, list):
                error_response = create_error_response(
                    message="Relations must be provided as a list",
                    code="invalid_input_type"
                )
                return model_to_json(error_response)
                
            # Check if list is empty
            if len(relations) == 0:
                error_response = create_error_response(
                    message="Relations list cannot be empty",
                    code="empty_input"
                )
                return model_to_json(error_response)
                
            # Validate list size to prevent excessive operations
            if len(relations) > 1000:
                error_response = create_error_response(
                    message=f"Too many relations provided ({len(relations)}). Maximum allowed is 1000.",
                    code="input_limit_exceeded"
                )
                return model_to_json(error_response)
            
            # Sanitize client_id if provided
            if client_id:
                client_id = str(client_id).strip()
                if len(client_id) > 100:  # Prevent excessively long IDs
                    logger.warn(f"Client ID too long, truncating: {client_id[:20]}...")
                    client_id = client_id[:100]
            
            logger.debug(f"Creating {len(relations)} relations", context={"relation_count": len(relations)})
        
            # Convert input to Pydantic models for validation
            try:
                formatted_relations = []
                max_entity_length = 500  # Maximum length for entity references
                max_type_length = 100    # Maximum length for relationship types
                
                for i, relation_data in enumerate(relations):
                    # Check if relation is a valid dictionary
                    if not isinstance(relation_data, dict):
                        invalid_relations.append({"index": i, "reason": "Relation must be an object"})
                        continue
                    
                    # Extract source entity with multiple field name options
                    from_entity = relation_data.get("from_entity") or relation_data.get("fromEntity") or relation_data.get("source")
                    if not from_entity:
                        invalid_relations.append({"index": i, "reason": "Source entity is required"})
                        continue
                        
                    # Sanitize source entity
                    from_entity = str(from_entity).strip()
                    if len(from_entity) == 0:
                        invalid_relations.append({"index": i, "reason": "Source entity cannot be empty"})
                        continue
                        
                    # Check source entity length
                    if len(from_entity) > max_entity_length:
                        from_entity = from_entity[:max_entity_length]
                        logger.warn(f"Source entity name too long, truncating: {from_entity[:20]}...")
                    
                    # Extract target entity with multiple field name options
                    to_entity = relation_data.get("to_entity") or relation_data.get("toEntity") or relation_data.get("target")
                    if not to_entity:
                        invalid_relations.append({"index": i, "reason": "Target entity is required"})
                        continue
                        
                    # Sanitize target entity
                    to_entity = str(to_entity).strip()
                    if len(to_entity) == 0:
                        invalid_relations.append({"index": i, "reason": "Target entity cannot be empty"})
                        continue
                        
                    # Check target entity length
                    if len(to_entity) > max_entity_length:
                        to_entity = to_entity[:max_entity_length]
                        logger.warn(f"Target entity name too long, truncating: {to_entity[:20]}...")
                    
                    # Extract relationship type with multiple field name options
                    relationship_type = relation_data.get("relationship_type") or relation_data.get("relationType") or relation_data.get("type")
                    if not relationship_type:
                        invalid_relations.append({"index": i, "reason": "Relationship type is required"})
                        continue
                        
                    # Sanitize relationship type
                    relationship_type = str(relationship_type).strip()
                    if len(relationship_type) == 0:
                        invalid_relations.append({"index": i, "reason": "Relationship type cannot be empty"})
                        continue
                        
                    # Check relationship type length and format
                    if len(relationship_type) > max_type_length:
                        relationship_type = relationship_type[:max_type_length]
                        logger.warn(f"Relationship type too long, truncating: {relationship_type[:20]}...")
                    
                    # Convert to uppercase for consistency, if it's not already a complex identifier
                    if relationship_type.isalpha():
                        relationship_type = relationship_type.upper()
                    
                    # Prepare relation object with explicit type annotation to avoid linter errors
                    relation_dict: Dict[str, Any] = {
                        "from_entity": from_entity,
                        "to_entity": to_entity,
                        "relationship_type": relationship_type
                    }
                    
                    # Add optional fields if present
                    if "weight" in relation_data:
                        try:
                            weight = float(relation_data["weight"])
                            # Normalize weight to a reasonable range
                            if weight < 0:
                                weight = 0.0
                            elif weight > 1000:
                                weight = 1000.0
                            relation_dict["weight"] = weight
                        except (ValueError, TypeError):
                            # Skip invalid weights
                            logger.warn(f"Invalid weight for relation from {from_entity} to {to_entity}, ignoring")
                        
                    # Add any metadata
                    if "metadata" in relation_data and isinstance(relation_data["metadata"], dict):
                        # Create a new dict to avoid reference issues
                        relation_dict["metadata"] = dict(relation_data["metadata"])
                    
                    # Add client_id to metadata if provided
                    if client_id:
                        # Ensure metadata exists
                        if "metadata" not in relation_dict or relation_dict.get("metadata") is None:
                            relation_dict["metadata"] = {}
                        
                        # Get metadata dict reference
                        metadata_dict = relation_dict.get("metadata")
                        if isinstance(metadata_dict, dict):
                            metadata_dict["client_id"] = client_id
                    
                    formatted_relations.append(relation_dict)
                
                # Check if any relations were valid after sanitization
                if not formatted_relations:
                    error_response = create_error_response(
                        message="No valid relations to create after validation",
                        code="validation_error",
                        details={"invalid_relations": invalid_relations}
                    )
                    return model_to_json(error_response)
                
                # If some relations were invalid, log them
                if invalid_relations:
                    logger.warn(f"Found {len(invalid_relations)} invalid relations out of {len(relations)} total")
                
                # Validate using Pydantic model
                relations_model = RelationshipsCreate(relationships=[RelationshipCreate(**relation) for relation in formatted_relations])
                
            except ValueError as e:
                logger.error(f"Validation error for relations: {str(e)}")
                error_response = create_error_response(
                    message=f"Invalid relation data: {str(e)}",
                    code="validation_error",
                    details={"invalid_relations": invalid_relations}
                )
                return model_to_json(error_response)
            
            # Get the client-specific graph manager
            client_graph_manager = get_client_manager(client_id)
            
            # Call the graph manager with the validated relations
            result = client_graph_manager.create_relations([model_to_dict(relation) for relation in relations_model.relationships])
            
            # Parse the result
            try:
                if isinstance(result, str):
                    parsed_result = json.loads(result)
                else:
                    parsed_result = result
                
                if isinstance(parsed_result, dict) and parsed_result.get("status") == "error":
                    error_response = create_error_response(
                        message=parsed_result.get("message", "Unknown error"),
                        code="relation_creation_error"
                    )
                    return model_to_json(error_response)
                
                # Return the success response
                success_response = create_success_response(
                    message=f"Successfully created {len(formatted_relations)} relations",
                    data=parsed_result
                )
                
                # Include information about skipped relations if any
                if invalid_relations:
                    success_response.data["skipped_relations"] = invalid_relations
                    
                return model_to_json(success_response)
                
            except json.JSONDecodeError:
                # If result is not valid JSON, return it as-is (legacy compatibility)
                return result
            
        except Exception as e:
            logger.error(f"Error creating relations: {str(e)}", exc_info=True)
            error_response = create_error_response(
                message=f"Error creating relations: {str(e)}",
                code="relation_creation_error",
                details={"relation_count": len(relations) if isinstance(relations, list) else 0}
            )
            return model_to_json(error_response)

    @server.tool()
    async def add_observations(observations: List[Dict[str, Any]], client_id: Optional[str] = None) -> str:
        """
        Add multiple observations to entities in the knowledge graph.
        
        Observations are facts or properties about an entity.
        
        Args:
            observations: List of observation objects
            client_id: Optional client ID for identifying the connection
        
        Returns:
            JSON response with operation result
        """
        try:
            # Initialize error tracking
            invalid_observations = []
            
            # Basic input validation
            if observations is None:
                error_response = create_error_response(
                    message="Observations list cannot be None",
                    code="invalid_input"
                )
                return model_to_json(error_response)
                
            # Ensure observations is a list
            if not isinstance(observations, list):
                error_response = create_error_response(
                    message="Observations must be provided as a list",
                    code="invalid_input_type"
                )
                return model_to_json(error_response)
                
            # Check if list is empty
            if len(observations) == 0:
                error_response = create_error_response(
                    message="Observations list cannot be empty",
                    code="empty_input"
                )
                return model_to_json(error_response)
                
            # Validate list size to prevent excessive operations
            if len(observations) > 1000:
                error_response = create_error_response(
                    message=f"Too many observations provided ({len(observations)}). Maximum allowed is 1000.",
                    code="input_limit_exceeded"
                )
                return model_to_json(error_response)
            
            # Sanitize client_id if provided
            if client_id:
                client_id = str(client_id).strip()
                if len(client_id) > 100:  # Prevent excessively long IDs
                    logger.warn(f"Client ID too long, truncating: {client_id[:20]}...")
                    client_id = client_id[:100]
                
            logger.debug(f"Adding {len(observations)} observations", context={"observation_count": len(observations)})
            
            # Convert input to Pydantic models for validation
            try:
                formatted_observations = []
                max_entity_length = 500   # Maximum length for entity names
                max_content_length = 50000  # Maximum length for observation content
                
                for i, obs_data in enumerate(observations):
                    # Check if observation is a valid dictionary
                    if not isinstance(obs_data, dict):
                        invalid_observations.append({"index": i, "reason": "Observation must be an object"})
                        continue
                
                    # Extract the entity name or ID with multiple field name options
                    entity_name = obs_data.get("entity") or obs_data.get("entityName") or obs_data.get("entity_name") or obs_data.get("entity_id")
                    
                    if not entity_name:
                        invalid_observations.append({"index": i, "reason": "Entity reference is required"})
                        continue
                    
                    # Sanitize entity name
                    entity_name = str(entity_name).strip()
                    if len(entity_name) == 0:
                        invalid_observations.append({"index": i, "reason": "Entity name cannot be empty"})
                        continue
                        
                    # Check entity name length
                    if len(entity_name) > max_entity_length:
                        entity_name = entity_name[:max_entity_length]
                        logger.warn(f"Entity name too long, truncating: {entity_name[:20]}...")
                    
                    # Check for reserved names or dangerous patterns
                    restricted_names = ["all", "*", "database", "system", "admin"]
                    if entity_name.lower() in restricted_names:
                        invalid_observations.append({"index": i, "reason": f"Cannot add observation to restricted entity: {entity_name}"})
                        continue
                        
                    # Extract the content
                    content = obs_data.get("content") or obs_data.get("contents")
                    
                    if not content:
                        invalid_observations.append({"index": i, "reason": "Observation content is required"})
                        continue
                    
                    # Prepare the contents field as a list
                    contents_list = []
                    if isinstance(content, str):
                        # Sanitize string content
                        content = content.strip()
                        if len(content) == 0:
                            invalid_observations.append({"index": i, "reason": "Observation content cannot be empty"})
                            continue
                            
                        # Check content length
                        if len(content) > max_content_length:
                            content = content[:max_content_length]
                            logger.warn(f"Observation content too long, truncating: {content[:20]}...")
                            
                        contents_list = [content]
                    elif isinstance(content, list):
                        # Process and validate each item in the list
                        for item in content:
                            if not item:
                                continue  # Skip empty items
                                
                            if not isinstance(item, str):
                                try:
                                    item = str(item)  # Try to convert non-string items
                                except:
                                    continue  # Skip items that can't be converted
                            
                            # Sanitize string
                            item = item.strip()
                            if len(item) == 0:
                                continue  # Skip empty strings
                                
                            # Check content length
                            if len(item) > max_content_length:
                                item = item[:max_content_length]
                                logger.warn(f"Observation content item too long, truncating: {item[:20]}...")
                                
                            contents_list.append(item)
                            
                        # Check if we have at least one valid content item
                        if not contents_list:
                            invalid_observations.append({"index": i, "reason": "No valid content items in observation"})
                            continue
                    else:
                        # Unexpected type, try to convert to string
                        try:
                            content_str = str(content).strip()
                            if len(content_str) == 0:
                                invalid_observations.append({"index": i, "reason": "Observation content cannot be empty"})
                                continue
                                
                            # Check content length
                            if len(content_str) > max_content_length:
                                content_str = content_str[:max_content_length]
                                logger.warn(f"Observation content too long, truncating: {content_str[:20]}...")
                                
                            contents_list = [content_str]
                        except:
                            invalid_observations.append({"index": i, "reason": f"Invalid content type: {type(content)}"})
                            continue
                        
                    # Format the observation
                    observation = {
                        "entity_name": entity_name,
                        "contents": contents_list
                    }
                    
                    # Add metadata if available
                    if "metadata" in obs_data and isinstance(obs_data["metadata"], dict):
                        observation["metadata"] = obs_data["metadata"]
                    
                    # Add client_id to metadata if provided
                    if client_id:
                        if "metadata" not in observation:
                            observation["metadata"] = {}
                        observation["metadata"]["client_id"] = client_id
                    
                    formatted_observations.append(observation)
                
                # Check if any observations were valid after sanitization
                if not formatted_observations:
                    error_response = create_error_response(
                        message="No valid observations to add after validation",
                        code="validation_error",
                        details={"invalid_observations": invalid_observations}
                    )
                    return model_to_json(error_response)
                
                # If some observations were invalid, log them
                if invalid_observations:
                    logger.warn(f"Found {len(invalid_observations)} invalid observations out of {len(observations)} total")
                
                # Validate using Pydantic model
                observations_model = ObservationsCreate(observations=[EntityObservation(**obs) for obs in formatted_observations])
                
            except ValueError as e:
                logger.error(f"Validation error for observations: {str(e)}")
                error_response = create_error_response(
                    message=f"Invalid observation data: {str(e)}",
                    code="validation_error",
                    details={"invalid_observations": invalid_observations}
                )
                return model_to_json(error_response)
            
            # Get the client-specific graph manager
            client_graph_manager = get_client_manager(client_id)
            
            # Call the graph manager with the validated observations
            result = client_graph_manager.add_observations([model_to_dict(obs) for obs in observations_model.observations])
            
            # Parse the result
            try:
                if isinstance(result, str):
                    parsed_result = json.loads(result)
                else:
                    parsed_result = result
                
                if isinstance(parsed_result, dict) and parsed_result.get("status") == "error":
                    error_response = create_error_response(
                        message=parsed_result.get("message", "Unknown error"),
                        code="observation_creation_error"
                    )
                    return model_to_json(error_response)
                
                # Return the success response
                success_response = create_success_response(
                    message=f"Successfully added {len(formatted_observations)} observations",
                    data=parsed_result
                )
                
                # Include information about skipped observations if any
                if invalid_observations:
                    success_response.data["skipped_observations"] = invalid_observations
                    
                return model_to_json(success_response)
                
            except json.JSONDecodeError:
                # If result is not valid JSON, return it as-is (legacy compatibility)
                return result
            
        except Exception as e:
            logger.error(f"Error adding observations: {str(e)}", exc_info=True)
            error_response = create_error_response(
                message=f"Error adding observations: {str(e)}",
                code="observation_creation_error",
                details={"observation_count": len(observations) if isinstance(observations, list) else 0}
            )
            return model_to_json(error_response)

    @server.tool()
    async def search_nodes(query: str, limit: int = 10, project_name: str = "", client_id: Optional[str] = None, fuzzy_match: bool = False) -> str:
        """
        Search for nodes in the knowledge graph.
        
        Args:
            query: Search query
            limit: Maximum number of results to return
            project_name: Optional project name to scope the search
            client_id: Optional client ID for identifying the connection
            fuzzy_match: Whether to use fuzzy matching for the search
        
        Returns:
            JSON string with search results
        """
        try:
            # Initialize error tracking
            search_warnings = []
            
            # Validate the query parameter
            if query is None:
                error_response = create_error_response(
                    message="Search query cannot be None",
                    code="invalid_input"
                )
                return model_to_json(error_response)
                
            # Convert to string and sanitize
            original_query = query
            query = str(query).strip()
            
            # Check if sanitization changed the query
            if query != original_query:
                search_warnings.append({
                    "warning": "query_sanitized",
                    "message": "Query was sanitized by removing leading/trailing whitespace"
                })
            
            # Check for empty query after sanitization
            if not query:
                error_response = create_error_response(
                    message="Search query cannot be empty",
                    code="invalid_input"
                )
                return model_to_json(error_response)
            
            # Security check: Detect potentially dangerous patterns in the query
            # This helps protect against injection attempts
            dangerous_patterns = ["--", ";", "DROP", "DELETE", "MERGE", "CREATE", "REMOVE", "SET", "exec(", "eval("]
            high_risk_patterns = ["--", ";", "DROP", "DELETE"]
            medium_risk_patterns = ["MERGE", "CREATE", "REMOVE", "SET"]
            
            # Check for high risk patterns (immediately reject)
            for pattern in high_risk_patterns:
                if pattern.upper() in query.upper():
                    error_response = create_error_response(
                        message=f"Search query contains high-risk pattern: {pattern}",
                        code="security_violation",
                        details={"query": query, "pattern": pattern, "risk_level": "high"}
                    )
                    # Log security violations for monitoring
                    logger.warn(f"Security violation in search query: {pattern}", context={
                        "query": query,
                        "client_id": client_id or "default",
                        "pattern": pattern
                    })
                    return model_to_json(error_response)
            
            # Check for medium risk patterns (warn but allow with modification)
            for pattern in medium_risk_patterns:
                if pattern.upper() in query.upper():
                    # Replace potentially harmful patterns
                    original_query = query
                    query = query.upper().replace(pattern.upper(), f" {pattern} ")
                    search_warnings.append({
                        "warning": "query_modified",
                        "message": f"Query contains potentially risky pattern '{pattern}' and was modified",
                        "original": original_query,
                        "modified": query
                    })
                    logger.warn(f"Potentially risky pattern in search query: {pattern}", context={
                        "original_query": original_query,
                        "modified_query": query
                    })
                    
            # Limit query length to prevent abuse and excessive processing
            max_query_length = 500
            if len(query) > max_query_length:
                # Truncate long queries and warn
                original_length = len(query)
                query = query[:max_query_length]
                search_warnings.append({
                    "warning": "query_truncated",
                    "message": f"Query truncated from {original_length} to {max_query_length} characters"
                })
                logger.warn(f"Search query truncated from {original_length} to {max_query_length} characters")
                
            # Validate and sanitize limit parameter
            try:
                # Convert to integer if it's not already
                if not isinstance(limit, int):
                    original_limit = limit
                    limit = int(limit)
                    search_warnings.append({
                        "warning": "limit_converted", 
                        "message": f"Limit was converted from {original_limit} to integer {limit}"
                    })
                
                # Enforce reasonable bounds
                if limit < 1:
                    original_limit = limit
                    limit = 1
                    search_warnings.append({
                        "warning": "limit_adjusted",
                        "message": f"Limit was adjusted from {original_limit} to minimum value of 1"
                    })
                elif limit > 100:
                    original_limit = limit
                    limit = 100  # Cap to prevent performance issues
                    search_warnings.append({
                        "warning": "limit_reduced",
                        "message": f"Limit reduced from {original_limit} to maximum allowed value of 100"
                    })
            except (ValueError, TypeError):
                logger.warn(f"Invalid limit value: {limit}, using default of 10")
                limit = 10  # Default if conversion fails
                search_warnings.append({
                    "warning": "invalid_limit",
                    "message": "Invalid limit value, using default of 10"
                })
                
            # Validate and sanitize project_name parameter
            if project_name is not None:
                original_project_name = project_name
                project_name = str(project_name).strip()
                
                # Check if sanitization changed the project name
                if project_name != original_project_name:
                    search_warnings.append({
                        "warning": "project_name_sanitized",
                        "message": "Project name was sanitized by removing whitespace"
                    })
                
                # Check for invalid characters in project name
                if project_name:
                    invalid_chars = [';', '--', '/*', '*/', '@@', '@', '=', 'exec(', 'eval(']
                    for char in invalid_chars:
                        if char in project_name:
                            error_response = create_error_response(
                                message=f"Project name contains invalid character: {char}",
                                code="invalid_project_name",
                                details={"project_name": project_name, "invalid_character": char}
                            )
                            return model_to_json(error_response)
                    
                    # Limit project name length
                    max_project_length = 100
                    if len(project_name) > max_project_length:
                        original_length = len(project_name)
                        project_name = project_name[:max_project_length]
                        search_warnings.append({
                            "warning": "project_name_truncated",
                            "message": f"Project name truncated from {original_length} to {max_project_length} characters"
                        })
            
            # Validate and sanitize fuzzy_match parameter
            if not isinstance(fuzzy_match, bool):
                original_fuzzy_match = fuzzy_match
                # Convert to boolean if it's not already
                if isinstance(fuzzy_match, str):
                    fuzzy_match = fuzzy_match.lower() in ["true", "1", "yes", "y", "t"]
                else:
                    fuzzy_match = bool(fuzzy_match)
                    
                search_warnings.append({
                    "warning": "fuzzy_match_converted",
                    "message": f"Fuzzy match was converted from {original_fuzzy_match} to boolean {fuzzy_match}"
                })
            
            # Sanitize client_id if provided
            if client_id:
                original_client_id = client_id
                client_id = str(client_id).strip()
                
                # Check if sanitization changed the client ID
                if client_id != original_client_id:
                    search_warnings.append({
                        "warning": "client_id_sanitized",
                        "message": "Client ID was sanitized by removing whitespace"
                    })
                
                # Check client ID length
                if len(client_id) > 100:  # Prevent excessively long IDs
                    logger.warn(f"Client ID too long, truncating: {client_id[:20]}...")
                    client_id = client_id[:100]
                    search_warnings.append({
                        "warning": "client_id_truncated",
                        "message": "Client ID was too long and has been truncated"
                    })
                
                # Check for dangerous patterns in client ID
                for pattern in dangerous_patterns:
                    if pattern.upper() in client_id.upper():
                        error_response = create_error_response(
                            message=f"Client ID contains potentially dangerous pattern: {pattern}",
                            code="security_violation",
                            details={"pattern": pattern}
                        )
                        return model_to_json(error_response)
            
            # Validate input using Pydantic model
            try:
                search_model = SearchQuery(
                    query=query,
                    limit=limit,
                    project_name=project_name if project_name else None
                )
            except ValueError as e:
                logger.error(f"Validation error for search query: {str(e)}")
                error_response = create_error_response(
                    message=f"Invalid search query: {str(e)}",
                    code="validation_error",
                    details={"query": query, "limit": limit, "project_name": project_name}
                )
                return model_to_json(error_response)
            
            # Get the client-specific graph manager
            client_graph_manager = get_client_manager(client_id)
            
            # Set project name if provided
            if search_model.project_name:
                client_graph_manager.set_project_name(search_model.project_name)
            
            # Log search parameters
            logger.info(f"Searching nodes with query: '{search_model.query}'", context={
                "limit": search_model.limit,
                "project_name": search_model.project_name or client_graph_manager.default_project_name,
                "client_id": client_id or "default",
                "fuzzy_match": fuzzy_match
            })
            
            # Perform the search
            # Add fuzzy_match parameter if the API supports it
            try:
                if hasattr(client_graph_manager, "search_nodes_fuzzy") and fuzzy_match:
                    result = client_graph_manager.search_nodes_fuzzy(
                        search_model.query,
                        search_model.limit
                    )
                    search_warnings.append({
                        "warning": "fuzzy_search_used",
                        "message": "Fuzzy matching search algorithm was used"
                    })
                else:
                    # If fuzzy search was requested but not available, add a warning
                    if fuzzy_match:
                        search_warnings.append({
                            "warning": "fuzzy_search_unavailable",
                            "message": "Fuzzy matching was requested but is not available, using exact search instead"
                        })
                    result = client_graph_manager.search_nodes(
                        search_model.query,
                        search_model.limit
                    )
            except Exception as search_error:
                logger.error(f"Error executing search: {str(search_error)}", exc_info=True)
                error_response = create_error_response(
                    message=f"Error executing search: {str(search_error)}",
                    code="search_execution_error",
                    details={"query": search_model.query}
                )
                return model_to_json(error_response)
            
            # Parse the result
            try:
                if isinstance(result, str):
                    try:
                        parsed_result = json.loads(result)
                    except json.JSONDecodeError:
                        # If result is not valid JSON but still a string, return legacy format
                        logger.warn(f"Search result is not valid JSON, returning as legacy format")
                        return result
                else:
                    parsed_result = result
                
                if isinstance(parsed_result, dict) and parsed_result.get("status") == "error":
                    error_response = create_error_response(
                        message=parsed_result.get("message", "Unknown error retrieving memories"),
                        code="search_error",
                        details=parsed_result.get("details", {})
                    )
                    return model_to_json(error_response)
                
                # Extract search results
                if isinstance(parsed_result, dict):
                    nodes = []
                    
                    # Support multiple response formats for better compatibility
                    if "nodes" in parsed_result:
                        nodes = parsed_result["nodes"]
                    elif "results" in parsed_result:
                        nodes = parsed_result["results"]
                    elif "entities" in parsed_result:
                        nodes = parsed_result["entities"]
                    elif "matches" in parsed_result:
                        nodes = parsed_result["matches"]
                    
                    # Sanitize sensitive information in the response
                    for node in nodes:
                        if isinstance(node, dict) and "properties" in node:
                            for key in list(node["properties"].keys()):
                                if any(sensitive in key.lower() for sensitive in ["password", "secret", "token", "key", "auth"]):
                                    node["properties"][key] = "[REDACTED]"
                    
                    # Add diagnostic information if no results found
                    if not nodes:
                        logger.debug(f"No results found for query: '{search_model.query}'")
                        diagnostic_info = {
                            "timestamp": datetime.datetime.now().isoformat(),
                            "query": search_model.query,
                            "project": search_model.project_name or client_graph_manager.default_project_name,
                            "fuzzy_match": fuzzy_match
                        }
                        
                        # Return the success response with empty results and diagnostic info
                        success_response = create_success_response(
                            message=f"No results found for query '{search_model.query}'",
                            data={
                                "nodes": [],
                                "diagnostic": diagnostic_info
                            }
                        )
                        
                        # Add any warnings collected during processing
                        if search_warnings:
                            success_response.data["warnings"] = search_warnings
                            
                        return model_to_json(success_response)
                    
                    # Return the success response with results
                    success_response = create_success_response(
                        message=f"Found {len(nodes)} results for query '{search_model.query}'",
                        data={
                            "nodes": nodes,
                            "query": search_model.query,
                            "limit": search_model.limit,
                            "fuzzy_match": fuzzy_match
                        }
                    )
                    
                    # Add any warnings collected during processing
                    if search_warnings:
                        success_response.data["warnings"] = search_warnings
                        
                    return model_to_json(success_response)
                
                # If result has a different format, return it as-is for legacy compatibility
                logger.debug(f"Returning non-standard search result format")
                return result
                
            except Exception as e:
                logger.error(f"Error parsing search result: {str(e)}", exc_info=True)
                error_response = create_error_response(
                    message=f"Error parsing search result: {str(e)}",
                    code="result_parsing_error",
                    details={"query": search_model.query}
                )
                return model_to_json(error_response)
            
        except Exception as e:
            logger.error(f"Error searching nodes: {str(e)}", exc_info=True)
            error_response = create_error_response(
                message=f"Error searching nodes: {str(e)}",
                code="search_error",
                details={
                    "query": query if 'query' in locals() else None,
                    "project_name": project_name if 'project_name' in locals() else None
                }
            )
            return model_to_json(error_response)

    @server.tool()
    async def delete_entity(entity: str, client_id: Optional[str] = None) -> str:
        """
        Delete an entity from the knowledge graph.
        
        Args:
            entity: Name or ID of the entity to delete
            client_id: Optional client ID for identifying the connection
        
        Returns:
            JSON response with operation result
        """
        try:
            # Initialize warning tracking
            deletion_warnings = []
            
            # Basic input validation
            if entity is None:
                error_response = create_error_response(
                    message="Entity name/ID cannot be None",
                    code="invalid_input"
                )
                return model_to_json(error_response)
                
            # Ensure entity is a non-empty string
            if not entity:
                error_response = create_error_response(
                    message="Entity name/ID cannot be empty",
                    code="invalid_input"
                )
                return model_to_json(error_response)
                
            # Sanitize entity name - trim whitespace
            original_entity = entity
            entity = str(entity).strip()
            
            # Check if sanitization changed the entity name
            if entity != original_entity:
                deletion_warnings.append({
                    "warning": "entity_name_sanitized",
                    "message": "Entity name was sanitized by removing whitespace"
                })
            
            # Check entity name length
            max_entity_length = 500
            if len(entity) > max_entity_length:
                entity = entity[:max_entity_length]
                logger.warn(f"Entity name too long, truncating: {entity[:20]}...")
                deletion_warnings.append({
                    "warning": "entity_name_truncated",
                    "message": f"Entity name truncated to {max_entity_length} characters"
                })
                
            # Check for empty entity after sanitization
            if not entity:
                error_response = create_error_response(
                    message="Entity name/ID is empty after sanitization",
                    code="invalid_input"
                )
                return model_to_json(error_response)
            
            # Check for reserved names or dangerous patterns
            restricted_names = ["all", "*", "database", "system", "admin", "neo4j", "apoc"]
            if entity.lower() in restricted_names:
                error_response = create_error_response(
                    message=f"Cannot delete restricted entity: {entity}",
                    code="restricted_entity",
                    details={"entity": entity}
                )
                return model_to_json(error_response)
            
            # Check for dangerous patterns that might indicate injection attempts
            dangerous_patterns = [';', '--', '/*', '*/', '@@', 'exec(', 'eval(', 'MATCH', 'CREATE', 'DELETE', 'REMOVE']
            for pattern in dangerous_patterns:
                if pattern.upper() in entity.upper():
                    error_response = create_error_response(
                        message=f"Entity name contains potentially dangerous pattern: {pattern}",
                        code="security_violation",
                        details={"entity": entity, "pattern": pattern}
                    )
                    return model_to_json(error_response)
            
            # Sanitize client_id if provided
            if client_id:
                client_id = str(client_id).strip()
                if len(client_id) > 100:  # Prevent excessively long IDs
                    logger.warn(f"Client ID too long, truncating: {client_id[:20]}...")
                    client_id = client_id[:100]
                    deletion_warnings.append({
                        "warning": "client_id_truncated",
                        "message": "Client ID was too long and has been truncated"
                    })
                
            # Validate input using Pydantic model
            try:
                entity_model = EntityDelete(entity_name=entity)
            except ValueError as e:
                logger.error(f"Validation error for entity deletion: {str(e)}")
                error_response = create_error_response(
                    message=f"Invalid entity name: {str(e)}",
                    code="validation_error",
                    details={"entity": entity}
                )
                return model_to_json(error_response)
            
            # Log the deletion request
            logger.info(f"Deleting entity: {entity_model.entity_name}", context={
                "client_id": client_id or "default"
            })
            
            # Get the client-specific graph manager
            client_graph_manager = get_client_manager(client_id)
            
            # Check if entity exists before trying to delete it
            try:
                # This check is optional but provides better error messages
                exists_result = client_graph_manager.get_entity(entity_model.entity_name)
                try:
                    exists_parsed = json.loads(exists_result) if isinstance(exists_result, str) else exists_result
                    if isinstance(exists_parsed, dict) and exists_parsed.get("status") == "error":
                        # Entity doesn't exist, return a more specific error
                        error_response = create_error_response(
                            message=f"Entity '{entity_model.entity_name}' not found",
                            code="entity_not_found",
                            details={"entity": entity_model.entity_name}
                        )
                        return model_to_json(error_response)
                except (json.JSONDecodeError, AttributeError):
                    # Continue with deletion even if we can't parse the exists result
                    pass
            except Exception:
                # Continue with deletion even if the exists check fails
                pass
            
            # Delete the entity
            result = client_graph_manager.delete_entity(entity_model.entity_name)
            
            # Parse the result
            try:
                if isinstance(result, str):
                    parsed_result = json.loads(result)
                else:
                    parsed_result = result
                
                if isinstance(parsed_result, dict) and parsed_result.get("status") == "error":
                    error_response = create_error_response(
                        message=parsed_result.get("message", "Unknown error"),
                        code="entity_deletion_error",
                        details={"entity": entity_model.entity_name}
                    )
                    return model_to_json(error_response)
                
                # Return the success response
                success_response = create_success_response(
                    message=f"Successfully deleted entity '{entity_model.entity_name}'",
                    data=parsed_result
                )
                
                # Add any warnings collected during processing
                if deletion_warnings:
                    success_response.data["warnings"] = deletion_warnings
                    
                return model_to_json(success_response)
                
            except json.JSONDecodeError:
                # If result is not valid JSON, return it as-is (legacy compatibility)
                return result
            
        except Exception as e:
            logger.error(f"Error deleting entity: {str(e)}", exc_info=True)
            error_response = create_error_response(
                message=f"Error deleting entity: {str(e)}",
                code="entity_deletion_error",
                details={"entity": entity if 'entity' in locals() else None}
            )
            return model_to_json(error_response)

    @server.tool()
    async def delete_relation(from_entity: str, to_entity: str, relationship_type: str, client_id: Optional[str] = None) -> str:
        """
        Delete a relationship from the knowledge graph.
        
        Args:
            from_entity: Name or ID of the source entity
            to_entity: Name or ID of the target entity
            relationship_type: Type of the relationship to delete
            client_id: Optional client ID for identifying the connection
        
        Returns:
            JSON response with operation result
        """
        try:
            # Initialize warning tracking
            deletion_warnings = []
            
            # Basic input validation - check for None values
            if from_entity is None or to_entity is None or relationship_type is None:
                missing_fields = []
                if from_entity is None:
                    missing_fields.append("from_entity")
                if to_entity is None:
                    missing_fields.append("to_entity")
                if relationship_type is None:
                    missing_fields.append("relationship_type")
                    
                error_response = create_error_response(
                    message=f"Required fields cannot be None: {', '.join(missing_fields)}",
                    code="invalid_input",
                    details={"missing_fields": missing_fields}
                )
                return model_to_json(error_response)
            
            # Check for missing required fields (empty strings)
            if not from_entity or not to_entity or not relationship_type:
                missing_fields = []
                if not from_entity:
                    missing_fields.append("from_entity")
                if not to_entity:
                    missing_fields.append("to_entity")
                if not relationship_type:
                    missing_fields.append("relationship_type")
                    
                error_response = create_error_response(
                    message=f"Missing required fields: {', '.join(missing_fields)}",
                    code="missing_required_fields",
                    details={"missing_fields": missing_fields}
                )
                return model_to_json(error_response)
                
            # Sanitize inputs - trim whitespace
            original_from = from_entity
            original_to = to_entity
            original_rel_type = relationship_type
            
            from_entity = str(from_entity).strip()
            to_entity = str(to_entity).strip()
            relationship_type = str(relationship_type).strip()
            
            # Check if sanitization changed any values
            if from_entity != original_from:
                deletion_warnings.append({
                    "warning": "from_entity_sanitized",
                    "message": "Source entity name was sanitized by removing whitespace"
                })
                
            if to_entity != original_to:
                deletion_warnings.append({
                    "warning": "to_entity_sanitized",
                    "message": "Target entity name was sanitized by removing whitespace"
                })
                
            if relationship_type != original_rel_type:
                deletion_warnings.append({
                    "warning": "relationship_type_sanitized",
                    "message": "Relationship type was sanitized by removing whitespace"
                })
            
            # Check entity name lengths
            max_entity_length = 500
            if len(from_entity) > max_entity_length:
                from_entity = from_entity[:max_entity_length]
                logger.warn(f"Source entity name too long, truncating: {from_entity[:20]}...")
                deletion_warnings.append({
                    "warning": "from_entity_truncated",
                    "message": f"Source entity name truncated to {max_entity_length} characters"
                })
                
            if len(to_entity) > max_entity_length:
                to_entity = to_entity[:max_entity_length]
                logger.warn(f"Target entity name too long, truncating: {to_entity[:20]}...")
                deletion_warnings.append({
                    "warning": "to_entity_truncated",
                    "message": f"Target entity name truncated to {max_entity_length} characters"
                })
            
            # Check relationship type length
            max_rel_type_length = 100
            if len(relationship_type) > max_rel_type_length:
                relationship_type = relationship_type[:max_rel_type_length]
                logger.warn(f"Relationship type too long, truncating: {relationship_type[:20]}...")
                deletion_warnings.append({
                    "warning": "relationship_type_truncated",
                    "message": f"Relationship type truncated to {max_rel_type_length} characters"
                })
                
            # Check for empty values after sanitization
            empty_fields = []
            if not from_entity:
                empty_fields.append("from_entity")
            if not to_entity:
                empty_fields.append("to_entity")
            if not relationship_type:
                empty_fields.append("relationship_type")
                
            if empty_fields:
                error_response = create_error_response(
                    message=f"Fields empty after sanitization: {', '.join(empty_fields)}",
                    code="invalid_input",
                    details={"empty_fields": empty_fields}
                )
                return model_to_json(error_response)
            
            # Check for restricted entity names
            restricted_names = ["all", "*", "database", "system", "admin", "neo4j", "apoc"]
            if from_entity.lower() in restricted_names or to_entity.lower() in restricted_names:
                restricted_entities = []
                if from_entity.lower() in restricted_names:
                    restricted_entities.append(from_entity)
                if to_entity.lower() in restricted_names:
                    restricted_entities.append(to_entity)
                    
                error_response = create_error_response(
                    message="Cannot delete relations involving restricted entities",
                    code="restricted_entity",
                    details={"restricted_entities": restricted_entities}
                )
                return model_to_json(error_response)
            
            # Check for dangerous patterns that might indicate injection attempts
            dangerous_patterns = [';', '--', '/*', '*/', '@@', 'exec(', 'eval(', 'MATCH', 'CREATE', 'DELETE', 'REMOVE']
            for field, value in [("from_entity", from_entity), ("to_entity", to_entity), ("relationship_type", relationship_type)]:
                for pattern in dangerous_patterns:
                    if pattern.upper() in value.upper():
                        error_response = create_error_response(
                            message=f"Field '{field}' contains potentially dangerous pattern: {pattern}",
                            code="security_violation",
                            details={"field": field, "value": value, "pattern": pattern}
                        )
                        return model_to_json(error_response)
                
            # Check for valid relationship type format
            # Allow only alphanumeric characters and underscores
            if not all(c.isalnum() or c == '_' for c in relationship_type):
                error_response = create_error_response(
                    message="Relationship type should contain only alphanumeric characters and underscores",
                    code="invalid_relationship_type",
                    details={"relationship_type": relationship_type}
                )
                return model_to_json(error_response)
            
            # Sanitize client_id if provided
            if client_id:
                client_id = str(client_id).strip()
                if len(client_id) > 100:  # Prevent excessively long IDs
                    logger.warn(f"Client ID too long, truncating: {client_id[:20]}...")
                    client_id = client_id[:100]
                    deletion_warnings.append({
                        "warning": "client_id_truncated",
                        "message": "Client ID was too long and has been truncated"
                    })
            
            # Validate input using Pydantic model
            try:
                relationship_model = RelationshipDelete(
                    from_entity=from_entity,
                    to_entity=to_entity,
                    relationship_type=relationship_type
                )
            except ValueError as e:
                logger.error(f"Validation error for relationship deletion: {str(e)}")
                error_response = create_error_response(
                    message=f"Invalid relationship data: {str(e)}",
                    code="validation_error",
                    details={
                        "from_entity": from_entity, 
                        "to_entity": to_entity, 
                        "relationship_type": relationship_type
                    }
                )
                return model_to_json(error_response)
            
            # Log the deletion request
            logger.info(f"Deleting relationship: {from_entity} -[{relationship_type}]-> {to_entity}", context={
                "client_id": client_id or "default"
            })
            
            # Get the client-specific graph manager
            client_graph_manager = get_client_manager(client_id)
            
            # Check if the entities exist before trying to delete the relation
            try:
                for entity_name, entity_role in [(from_entity, "source"), (to_entity, "target")]:
                    exists_result = client_graph_manager.get_entity(entity_name)
                    try:
                        exists_parsed = json.loads(exists_result) if isinstance(exists_result, str) else exists_result
                        if isinstance(exists_parsed, dict) and exists_parsed.get("status") == "error":
                            # Entity doesn't exist, return a more specific error
                            error_response = create_error_response(
                                message=f"{entity_role.capitalize()} entity '{entity_name}' not found",
                                code="entity_not_found",
                                details={"entity": entity_name, "role": entity_role}
                            )
                            return model_to_json(error_response)
                    except (json.JSONDecodeError, AttributeError):
                        # Continue even if we can't parse the exists result
                        pass
            except Exception:
                # Continue with deletion even if the exists check fails
                pass
            
            # Delete the relationship
            result = client_graph_manager.delete_relationship(
                relationship_model.from_entity,
                relationship_model.to_entity,
                relationship_model.relationship_type
            )
            
            # Parse the result
            try:
                if isinstance(result, str):
                    parsed_result = json.loads(result)
                else:
                    parsed_result = result
                
                if isinstance(parsed_result, dict) and parsed_result.get("status") == "error":
                    error_response = create_error_response(
                        message=parsed_result.get("message", "Unknown error"),
                        code="relationship_deletion_error",
                        details={
                            "from_entity": relationship_model.from_entity,
                            "to_entity": relationship_model.to_entity,
                            "relationship_type": relationship_model.relationship_type
                        }
                    )
                    return model_to_json(error_response)
                
                # Return the success response
                success_response = create_success_response(
                    message=f"Successfully deleted relationship from '{relationship_model.from_entity}' to '{relationship_model.to_entity}' of type '{relationship_model.relationship_type}'",
                    data=parsed_result
                )
                
                # Add any warnings collected during processing
                if deletion_warnings:
                    success_response.data["warnings"] = deletion_warnings
                    
                return model_to_json(success_response)
                
            except json.JSONDecodeError:
                # If result is not valid JSON, return it as-is (legacy compatibility)
                return result
            
        except Exception as e:
            logger.error(f"Error deleting relationship: {str(e)}", exc_info=True)
            error_response = create_error_response(
                message=f"Error deleting relationship: {str(e)}",
                code="relationship_deletion_error",
                details={
                    "from_entity": from_entity if 'from_entity' in locals() else None,
                    "to_entity": to_entity if 'to_entity' in locals() else None,
                    "relationship_type": relationship_type if 'relationship_type' in locals() else None
                }
            )
            return model_to_json(error_response)

    @server.tool()
    async def delete_observation(entity: str, content: str, client_id: Optional[str] = None) -> str:
        """
        Delete an observation from an entity in the knowledge graph.
        
        Args:
            entity: Name or ID of the entity
            content: Content of the observation to delete
            client_id: Optional client ID for identifying the connection
        
        Returns:
            JSON response with operation result
        """
        try:
            # Initialize warning tracking
            deletion_warnings = []
            
            # Basic input validation
            if entity is None or content is None:
                missing_fields = []
                if entity is None:
                    missing_fields.append("entity")
                if content is None:
                    missing_fields.append("content")
                    
                error_response = create_error_response(
                    message=f"Required fields cannot be None: {', '.join(missing_fields)}",
                    code="invalid_input",
                    details={"missing_fields": missing_fields}
                )
                return model_to_json(error_response)
            
            # Check for empty required fields
            if not entity or not content:
                missing_fields = []
                if not entity:
                    missing_fields.append("entity")
                if not content:
                    missing_fields.append("content")
                    
                error_response = create_error_response(
                    message=f"Missing required fields: {', '.join(missing_fields)}",
                    code="missing_required_fields",
                    details={"missing_fields": missing_fields}
                )
                return model_to_json(error_response)
                
            # Sanitize entity name - trim whitespace
            original_entity = entity
            entity = str(entity).strip()
            
            # Check if sanitization changed the entity name
            if entity != original_entity:
                deletion_warnings.append({
                    "warning": "entity_name_sanitized",
                    "message": "Entity name was sanitized by removing whitespace"
                })
            
            # Check entity name length
            max_entity_length = 500
            if len(entity) > max_entity_length:
                entity = entity[:max_entity_length]
                logger.warn(f"Entity name too long, truncating: {entity[:20]}...")
                deletion_warnings.append({
                    "warning": "entity_name_truncated",
                    "message": f"Entity name truncated to {max_entity_length} characters"
                })
                
            # Check for empty entity after sanitization
            if not entity:
                error_response = create_error_response(
                    message="Entity name/ID is empty after sanitization",
                    code="invalid_input"
                )
                return model_to_json(error_response)
            
            # Check for restricted entity names
            restricted_names = ["all", "*", "database", "system", "admin", "neo4j", "apoc"]
            if entity.lower() in restricted_names:
                error_response = create_error_response(
                    message=f"Cannot delete observations for restricted entity: {entity}",
                    code="restricted_entity",
                    details={"entity": entity}
                )
                return model_to_json(error_response)
            
            # Check for dangerous patterns in entity name
            dangerous_patterns = [';', '--', '/*', '*/', '@@', 'exec(', 'eval(', 'MATCH', 'CREATE', 'DELETE', 'REMOVE']
            for pattern in dangerous_patterns:
                if pattern.upper() in entity.upper():
                    error_response = create_error_response(
                        message=f"Entity name contains potentially dangerous pattern: {pattern}",
                        code="security_violation",
                        details={"entity": entity, "pattern": pattern}
                    )
                    return model_to_json(error_response)
            
            # Special handling for content - preserve whitespace but validate
            # We don't want to modify content unnecessarily as it might need exact matching
            
            # But we need to check if it's excessive in length for security
            max_content_length = 50000  # 50KB should be plenty for legitimate observations
            if len(content) > max_content_length:
                logger.warn(f"Observation content too long ({len(content)} chars), truncating")
                content = content[:max_content_length]
                deletion_warnings.append({
                    "warning": "content_truncated",
                    "message": f"Observation content truncated to {max_content_length} characters"
                })
                
            # Check content for dangerous patterns
            # Generally less strict with content patterns since this is part of data, not a query
            high_risk_patterns = ['--', ';--', '/*', '*/']
            for pattern in high_risk_patterns:
                if pattern in content:
                    deletion_warnings.append({
                        "warning": "content_suspicious_pattern",
                        "message": f"Content contains potentially suspicious pattern: {pattern}"
                    })
                    logger.warn(f"Potentially suspicious pattern in observation content: {pattern}")
            
            # For logging, truncate the content if it's very long
            content_for_log = content
            if len(content) > 50:
                content_for_log = content[:47] + "..."
                
            # Sanitize client_id if provided
            if client_id:
                client_id = str(client_id).strip()
                if len(client_id) > 100:  # Prevent excessively long IDs
                    logger.warn(f"Client ID too long, truncating: {client_id[:20]}...")
                    client_id = client_id[:100]
                    deletion_warnings.append({
                        "warning": "client_id_truncated",
                        "message": "Client ID was too long and has been truncated"
                    })
                    
                # Check client_id for dangerous patterns
                for pattern in dangerous_patterns:
                    if pattern.upper() in client_id.upper():
                        error_response = create_error_response(
                            message=f"Client ID contains potentially dangerous pattern: {pattern}",
                            code="security_violation",
                            details={"pattern": pattern}
                        )
                        return model_to_json(error_response)
                
            logger.info(f"Deleting observation for entity '{entity}': {content_for_log}", context={
                "client_id": client_id or "default",
                "content_length": len(content)
            })
            
            # Validate input using Pydantic model
            try:
                observation_model = ObservationDelete(
                    entity_name=entity,
                    content=content
                )
            except ValueError as e:
                logger.error(f"Validation error for observation deletion: {str(e)}")
                error_response = create_error_response(
                    message=f"Invalid observation data: {str(e)}",
                    code="validation_error",
                    details={"entity": entity}
                )
                return model_to_json(error_response)
            
            # Get the client-specific graph manager
            client_graph_manager = get_client_manager(client_id)
            
            # Check if entity exists before trying to delete the observation
            try:
                # This check is optional but provides better error messages
                exists_result = client_graph_manager.get_entity(observation_model.entity_name)
                try:
                    exists_parsed = json.loads(exists_result) if isinstance(exists_result, str) else exists_result
                    if isinstance(exists_parsed, dict) and exists_parsed.get("status") == "error":
                        # Entity doesn't exist, return a more specific error
                        error_response = create_error_response(
                            message=f"Entity '{observation_model.entity_name}' not found",
                            code="entity_not_found",
                            details={"entity": observation_model.entity_name}
                        )
                        return model_to_json(error_response)
                except (json.JSONDecodeError, AttributeError):
                    # Continue with deletion even if we can't parse the exists result
                    pass
            except Exception:
                # Continue with deletion even if the exists check fails
                pass
            
            # Delete the observation
            result = client_graph_manager.delete_observation(
                observation_model.entity_name,
                observation_model.content
            )
            
            # Parse the result
            try:
                if isinstance(result, str):
                    parsed_result = json.loads(result)
                else:
                    parsed_result = result
                
                if isinstance(parsed_result, dict) and parsed_result.get("status") == "error":
                    error_response = create_error_response(
                        message=parsed_result.get("message", "Unknown error"),
                        code="observation_deletion_error",
                        details={"entity": observation_model.entity_name}
                    )
                    return model_to_json(error_response)
                
                # Return the success response
                success_response = create_success_response(
                    message=f"Successfully deleted observation from entity '{observation_model.entity_name}'",
                    data=parsed_result
                )
                
                # Add any warnings collected during processing
                if deletion_warnings:
                    success_response.data["warnings"] = deletion_warnings
                    
                return model_to_json(success_response)
                
            except json.JSONDecodeError:
                # If result is not valid JSON, return it as-is (legacy compatibility)
                return result
            
        except Exception as e:
            logger.error(f"Error deleting observation: {str(e)}", exc_info=True)
            error_response = create_error_response(
                message=f"Error deleting observation: {str(e)}",
                code="observation_deletion_error",
                details={
                    "entity": entity if 'entity' in locals() else None
                }
            )
            return model_to_json(error_response)

    @server.tool()
    async def get_all_memories(random_string: str = "", client_id: Optional[str] = None, limit: int = 1000, offset: int = 0, include_observations: bool = True) -> str:
        """
        Get all memories from the knowledge graph.
        
        WARNING: This is a potentially expensive operation that should be used with caution.
        
        Args:
            random_string: Random string to confirm intentional use of this tool
            client_id: Optional client ID for identifying the connection
            limit: Maximum number of entities to return (default: 1000)
            offset: Number of entities to skip (for pagination)
            include_observations: Whether to include observations in the results
        
        Returns:
            JSON response with all memories
        """
        try:
            # Initialize warning tracking
            retrieval_warnings = []
            
            # Basic input validation for random_string
            if random_string is None:
                error_response = create_error_response(
                    message="Random string cannot be None",
                    code="invalid_input"
                )
                return model_to_json(error_response)
            
            # Sanitize the random_string
            original_random_string = random_string
            random_string = str(random_string).strip()
            
            # Check if sanitization changed the random_string
            if random_string != original_random_string:
                retrieval_warnings.append({
                    "warning": "random_string_sanitized",
                    "message": "Random string was sanitized by removing whitespace"
                })
            
            # Validate the random_string parameter
            if len(random_string) < 8:
                error_response = create_error_response(
                    message="Please provide a random string with at least 8 characters to confirm this potentially expensive operation",
                    code="validation_error",
                    details={"min_length": 8, "provided_length": len(random_string)}
                )
                return model_to_json(error_response)
            
            # Check for dangerous patterns in random_string
            dangerous_patterns = [';', '--', '/*', '*/', 'exec(', 'eval(', 'DROP', 'DELETE', 'MATCH', 'MERGE']
            for pattern in dangerous_patterns:
                if pattern.upper() in random_string.upper():
                    error_response = create_error_response(
                        message=f"Random string contains potentially dangerous pattern: {pattern}",
                        code="security_violation",
                        details={"pattern": pattern}
                    )
                    return model_to_json(error_response)
            
            # Validate and sanitize limit parameter
            try:
                limit = int(limit)
                original_limit = limit
                
                # Enforce reasonable limits
                if limit < 1:
                    limit = 10  # Set a sensible minimum
                    retrieval_warnings.append({
                        "warning": "limit_adjusted",
                        "message": f"Limit was too small, adjusted from {original_limit} to {limit}"
                    })
                elif limit > 1000:
                    # Cap to prevent excessive load
                    limit = 1000
                    retrieval_warnings.append({
                        "warning": "limit_capped",
                        "message": f"Limit was too large, capped from {original_limit} to {limit}"
                    })
            except (ValueError, TypeError):
                limit = 1000  # Default if conversion fails
                retrieval_warnings.append({
                    "warning": "limit_invalid",
                    "message": f"Invalid limit value, using default of {limit}"
                })
            
            # Validate and sanitize offset parameter
            try:
                offset = int(offset)
                original_offset = offset
                
                # Enforce reasonable offset
                if offset < 0:
                    offset = 0  # Cannot be negative
                    retrieval_warnings.append({
                        "warning": "offset_adjusted",
                        "message": f"Offset was negative, adjusted from {original_offset} to {offset}"
                    })
            except (ValueError, TypeError):
                offset = 0  # Default if conversion fails
                retrieval_warnings.append({
                    "warning": "offset_invalid",
                    "message": f"Invalid offset value, using default of {offset}"
                })
            
            # Validate include_observations parameter
            if not isinstance(include_observations, bool):
                # Convert to boolean if possible
                if isinstance(include_observations, str):
                    include_observations = include_observations.lower() in ["true", "1", "yes"]
                else:
                    try:
                        include_observations = bool(include_observations)
                    except (ValueError, TypeError):
                        include_observations = True
                
                retrieval_warnings.append({
                    "warning": "include_observations_converted",
                    "message": f"include_observations parameter was not boolean, converted to {include_observations}"
                })
            
            # Sanitize client_id if provided
            if client_id is not None:
                client_id = str(client_id).strip()
                
                # Check client_id length
                if len(client_id) > 100:  # Prevent excessively long IDs
                    logger.warn(f"Client ID too long, truncating: {client_id[:20]}...")
                    client_id = client_id[:100]
                    retrieval_warnings.append({
                        "warning": "client_id_truncated",
                        "message": "Client ID was too long and has been truncated"
                    })
                
                # Check for dangerous patterns in client_id
                for pattern in dangerous_patterns:
                    if pattern.upper() in client_id.upper():
                        error_response = create_error_response(
                            message=f"Client ID contains potentially dangerous pattern: {pattern}",
                            code="security_violation",
                            details={"pattern": pattern}
                        )
                        return model_to_json(error_response)
            
            # Log the retrieval request
            logger.info(f"Retrieving all memories with limit={limit}, offset={offset}", context={
                "client_id": client_id or "default",
                "include_observations": include_observations
            })
            
            # Get the client-specific graph manager
            client_graph_manager = get_client_manager(client_id)
            
            # Get all memories - pass params to support pagination in future
            result = client_graph_manager.get_all_memories(random_string)
            
            # Parse the result for consistency
            try:
                if isinstance(result, str):
                    parsed_result = json.loads(result)
                    
                    # Return success response in standard format
                    if isinstance(parsed_result, dict) and "error" not in parsed_result:
                        # Apply client-side pagination if supported
                        try:
                            if isinstance(parsed_result.get("entities"), list):
                                # Get entities from response
                                entities = parsed_result["entities"]
                                
                                # Apply offset and limit
                                paginated_entities = entities[offset:offset + limit]
                                
                                # Update response with paginated data
                                parsed_result["entities"] = paginated_entities
                                parsed_result["pagination"] = {
                                    "total": len(entities),
                                    "offset": offset,
                                    "limit": limit, 
                                    "returned": len(paginated_entities)
                                }
                                
                                # Filter out observations if not requested
                                if not include_observations:
                                    for entity in paginated_entities:
                                        if "observations" in entity:
                                            # Keep count but remove content
                                            observation_count = len(entity.get("observations", []))
                                            entity["observations"] = []
                                            entity["observation_count"] = observation_count
                        except Exception as pagination_error:
                            logger.error(f"Error applying pagination: {str(pagination_error)}")
                            retrieval_warnings.append({
                                "warning": "pagination_error",
                                "message": f"Error applying pagination: {str(pagination_error)}"
                            })
                        
                        success_response = create_success_response(
                            message="Successfully retrieved memories",
                            data=parsed_result
                        )
                        
                        # Add any warnings collected during processing
                        if retrieval_warnings:
                            success_response.data["warnings"] = retrieval_warnings
                            
                        return model_to_json(success_response)
                    else:
                        # Return error in standard format
                        error_response = create_error_response(
                            message=parsed_result.get("error", "Unknown error retrieving memories"),
                            code="memory_retrieval_error",
                            details={"random_string": "[REDACTED]", "limit": limit, "offset": offset}
                        )
                        return model_to_json(error_response)
                
                # Return the result as-is if parsing fails
                return result
                
            except json.JSONDecodeError as json_error:
                logger.error(f"Error parsing response: {str(json_error)}")
                # If result is not valid JSON, standardize the error
                error_response = create_error_response(
                    message=f"Error parsing memory retrieval result: {str(json_error)}",
                    code="response_parsing_error"
                )
                return model_to_json(error_response)
            
        except Exception as e:
            logger.error(f"Error getting all memories: {str(e)}", exc_info=True)
            error_response = create_error_response(
                message=f"Error getting all memories: {str(e)}",
                code="memory_retrieval_error",
                details={
                    "random_string": "[REDACTED]", 
                    "limit": limit if 'limit' in locals() else None,
                    "offset": offset if 'offset' in locals() else None
                }
            )
            return model_to_json(error_response)

    @server.tool()
    async def delete_all_memories(random_string: str = "", project_name: str = "", client_id: Optional[str] = None) -> str:
        """
        Delete all memories from the knowledge graph.
        
        WARNING: This is a destructive operation that should be used with extreme caution.
        
        Args:
            random_string: Random string to confirm intentional use of this tool, must start with 'CONFIRM_DELETE_'
            project_name: Optional project name to restrict deletion to a specific project
            client_id: Optional client ID for identifying the connection
        
        Returns:
            JSON response with operation result
        """
        try:
            # Initialize warning tracking
            deletion_warnings = []
            
            # Basic input validation for random_string
            if random_string is None:
                error_response = create_error_response(
                    message="Random string cannot be None",
                    code="invalid_input"
                )
                return model_to_json(error_response)
            
            # Sanitize the random_string
            original_random_string = random_string
            random_string = str(random_string).strip()
            
            # Check if sanitization changed the random_string
            if random_string != original_random_string:
                deletion_warnings.append({
                    "warning": "random_string_sanitized",
                    "message": "Random string was sanitized by removing whitespace"
                })
            
            # Validate the random_string parameter - require at least 12 characters for this destructive operation
            if len(random_string) < 12:
                error_response = create_error_response(
                    message="Please provide a random string with at least 12 characters to confirm this destructive operation",
                    code="validation_error",
                    details={"min_length": 12, "provided_length": len(random_string)}
                )
                return model_to_json(error_response)
                
            # Add a specific confirmation prefix for extra safety
            if not random_string.startswith("CONFIRM_DELETE_"):
                error_response = create_error_response(
                    message="The random string must start with 'CONFIRM_DELETE_' to confirm this destructive operation",
                    code="validation_error",
                    details={"required_prefix": "CONFIRM_DELETE_"}
                )
                return model_to_json(error_response)
            
            # Check for dangerous patterns in random_string (beyond the required prefix)
            dangerous_patterns = [';', '--', '/*', '*/', '@@', 'exec(', 'eval(', 'DROP', 'MATCH', 'MERGE']
            random_string_content = random_string[14:]  # Skip the prefix for pattern checking
            for pattern in dangerous_patterns:
                if pattern.upper() in random_string_content.upper():
                    error_response = create_error_response(
                        message=f"Random string contains potentially dangerous pattern: {pattern}",
                        code="security_violation",
                        details={"pattern": pattern}
                    )
                    return model_to_json(error_response)
            
            # Validate project_name if provided
            if project_name:
                original_project_name = project_name
                project_name = str(project_name).strip()
                
                # Check if sanitization changed the project_name
                if project_name != original_project_name:
                    deletion_warnings.append({
                        "warning": "project_name_sanitized",
                        "message": "Project name was sanitized by removing whitespace"
                    })
                
                # Check project name length
                if len(project_name) > 100:
                    project_name = project_name[:100]
                    deletion_warnings.append({
                        "warning": "project_name_truncated",
                        "message": "Project name was truncated to 100 characters"
                    })
                
                # Check project_name for dangerous patterns
                for pattern in dangerous_patterns:
                    if pattern.upper() in project_name.upper():
                        error_response = create_error_response(
                            message=f"Project name contains potentially dangerous pattern: {pattern}",
                            code="security_violation",
                            details={"pattern": pattern}
                        )
                        return model_to_json(error_response)
                
                # Check for valid project name format
                if not all(c.isalnum() or c in '-_.' for c in project_name):
                    error_response = create_error_response(
                        message="Project name should contain only alphanumeric characters, hyphens, underscores, and periods",
                        code="invalid_project_name",
                        details={"project_name": project_name}
                    )
                    return model_to_json(error_response)
            
            # Sanitize client_id if provided
            if client_id is not None:
                client_id = str(client_id).strip()
                
                # Check client_id length
                if len(client_id) > 100:  # Prevent excessively long IDs
                    logger.warn(f"Client ID too long, truncating: {client_id[:20]}...")
                    client_id = client_id[:100]
                    deletion_warnings.append({
                        "warning": "client_id_truncated",
                        "message": "Client ID was too long and has been truncated"
                    })
                
                # Check for dangerous patterns in client_id
                for pattern in dangerous_patterns:
                    if pattern.upper() in client_id.upper():
                        error_response = create_error_response(
                            message=f"Client ID contains potentially dangerous pattern: {pattern}",
                            code="security_violation",
                            details={"pattern": pattern}
                        )
                        return model_to_json(error_response)
            
            # Add additional time-based protection for this destructive operation
            timestamp = datetime.datetime.now().strftime("%Y%m%d%H%M")
            expected_timestamp_pattern = datetime.datetime.now().strftime("%Y%m%d%H")
            
            # Check if timestamp information is included in the random string
            if expected_timestamp_pattern not in random_string:
                error_response = create_error_response(
                    message=f"For additional security, the random string must include the current hour in format YYYYMMDDHH (e.g. CONFIRM_DELETE_{expected_timestamp_pattern}...)",
                    code="security_requirement",
                    details={"current_hour": expected_timestamp_pattern}
                )
                return model_to_json(error_response)
            
            # Log the deletion attempt with extra visibility
            scope_message = f"all memories{f' for project {project_name}' if project_name else ''}"
            logger.warn(f"DESTRUCTIVE OPERATION: Attempting to delete {scope_message}", context={
                "client_id": client_id or "default",
                "timestamp": timestamp,
                "scope": scope_message
            })
            
            # Get the client-specific graph manager
            client_graph_manager = get_client_manager(client_id)
            
            # Delete all memories - pass both random string and project name
            params = {
                "confirmation": random_string
            }
            if project_name:
                params["project_name"] = project_name
                
            # Delete all memories
            result = client_graph_manager.delete_all_memories(**params)
            
            # Parse the result for consistency
            try:
                if isinstance(result, str):
                    parsed_result = json.loads(result)
                    
                    # Return success response in standard format
                    if "error" not in parsed_result:
                        success_response = create_success_response(
                            message=f"Successfully deleted {scope_message}",
                            data=parsed_result
                        )
                        
                        # Add any warnings collected during processing
                        if deletion_warnings:
                            success_response.data["warnings"] = deletion_warnings
                            
                        return model_to_json(success_response)
                    else:
                        # Return error in standard format
                        error_response = create_error_response(
                            message=parsed_result.get("error", "Unknown error deleting memories"),
                            code="memory_deletion_error",
                            details={"project_name": project_name if project_name else "all"}
                        )
                        return model_to_json(error_response)
                
                # Return the result as-is if parsing fails
                return result
                
            except json.JSONDecodeError as json_error:
                logger.error(f"Error parsing deletion response: {str(json_error)}")
                # If result is not valid JSON, standardize the error
                error_response = create_error_response(
                    message=f"Error parsing memory deletion result: {str(json_error)}",
                    code="response_parsing_error"
                )
                return model_to_json(error_response)
            
        except Exception as e:
            logger.error(f"Error deleting all memories: {str(e)}", exc_info=True)
            error_response = create_error_response(
                message=f"Error deleting all memories: {str(e)}",
                code="memory_deletion_error",
                details={
                    "project_name": project_name if 'project_name' in locals() and project_name else "all"
                }
            )
            return model_to_json(error_response)

    @server.tool()
    async def debug_dump_neo4j(limit: int = 100, confirm: bool = False, query: str = "", client_id: Optional[str] = None, 
                             include_relationships: bool = True, include_statistics: bool = True) -> str:
        """
        Dump Neo4j database contents for debugging purposes.
        
        This operation can be expensive on large databases.
        
        Args:
            limit: Maximum number of nodes to return (default: 100)
            confirm: Set to True to confirm you want to run this potentially expensive operation
            query: Optional Cypher query to restrict what is returned (advanced users only)
            client_id: Optional client ID for identifying the connection
            include_relationships: Whether to include relationships in the results (default: True)
            include_statistics: Whether to include database statistics in the results (default: True)
        
        Returns:
            JSON response with Neo4j database contents
        """
        try:
            # Initialize warning tracking
            dump_warnings = []
            
            # Validate and sanitize the limit parameter
            try:
                # Ensure limit is an integer
                if not isinstance(limit, int):
                    original_limit = limit
                    try:
                        limit = int(limit)
                        dump_warnings.append({
                            "warning": "limit_converted",
                            "message": f"Limit was converted from {original_limit} to integer {limit}"
                        })
                    except (ValueError, TypeError):
                        limit = 100  # Default if conversion fails
                        dump_warnings.append({
                            "warning": "limit_invalid",
                            "message": f"Invalid limit value '{original_limit}', using default of {limit}"
                        })
                
                # Enforce reasonable limit range
                if limit < 1:
                    original_limit = limit
                    limit = 10  # Use sensible minimum
                    dump_warnings.append({
                        "warning": "limit_adjusted",
                        "message": f"Limit was too small, adjusted from {original_limit} to {limit}"
                    })
                elif limit > 1000:
                    original_limit = limit
                    # Cap to prevent excessive load
                    limit = 1000
                    dump_warnings.append({
                        "warning": "limit_capped",
                        "message": f"Limit was too large, capped from {original_limit} to {limit}"
                    })
                    
                # Require confirmation for large dumps
                if not confirm and limit > 200:
                    error_response = create_error_response(
                        message=f"Requesting a large dump (limit={limit}). Set confirm=True to proceed with this potentially expensive operation.",
                        code="confirmation_required",
                        details={"limit": limit, "confirmation_required": True}
                    )
                    return model_to_json(error_response)
            except Exception as limit_error:
                # Handle any unexpected errors in limit parsing
                logger.error(f"Error processing limit parameter: {str(limit_error)}")
                limit = 100  # Default if processing fails
                dump_warnings.append({
                    "warning": "limit_error",
                    "message": f"Error processing limit parameter: {str(limit_error)}, using default of {limit}"
                })
            
            # Validate and sanitize the confirm parameter
            if not isinstance(confirm, bool):
                original_confirm = confirm
                try:
                    # Convert to boolean if possible
                    if isinstance(confirm, str):
                        confirm = confirm.lower() in ["true", "1", "yes", "y", "t"]
                    else:
                        confirm = bool(confirm)
                    
                    dump_warnings.append({
                        "warning": "confirm_converted",
                        "message": f"Confirm parameter was converted from {original_confirm} to boolean {confirm}"
                    })
                except (ValueError, TypeError):
                    confirm = False
                    dump_warnings.append({
                        "warning": "confirm_invalid",
                        "message": "Invalid confirm value, defaulting to False"
                    })
            
            # Validate and sanitize boolean parameters
            for param_name, param_value in [
                ("include_relationships", include_relationships),
                ("include_statistics", include_statistics)
            ]:
                if not isinstance(param_value, bool):
                    original_value = param_value
                    try:
                        # Convert to boolean if possible
                        if isinstance(param_value, str):
                            new_value = param_value.lower() in ["true", "1", "yes", "y", "t"]
                        else:
                            new_value = bool(param_value)
                        
                        # Update the actual variable using locals()
                        locals()[param_name] = new_value
                        
                        dump_warnings.append({
                            "warning": f"{param_name}_converted",
                            "message": f"{param_name.replace('_', ' ').capitalize()} parameter was converted from {original_value} to boolean {new_value}"
                        })
                    except (ValueError, TypeError):
                        # Use the original default (True)
                        locals()[param_name] = True
                        dump_warnings.append({
                            "warning": f"{param_name}_invalid",
                            "message": f"Invalid {param_name.replace('_', ' ')} value, defaulting to True"
                        })
            
            # Validate and sanitize the query parameter
            if query:
                original_query = query
                query = str(query).strip()
                
                # Check if sanitization changed the query
                if query != original_query:
                    dump_warnings.append({
                        "warning": "query_sanitized",
                        "message": "Query was sanitized by removing whitespace"
                    })
                
                # Check for empty query after sanitization
                if not query:
                    dump_warnings.append({
                        "warning": "query_empty",
                        "message": "Empty query provided, ignoring"
                    })
                else:
                    # Check query length
                    max_query_length = 1000  # Allow longer queries for debugging
                    if len(query) > max_query_length:
                        original_length = len(query)
                        query = query[:max_query_length]
                        dump_warnings.append({
                            "warning": "query_truncated",
                            "message": f"Query was truncated from {original_length} to {max_query_length} characters"
                        })
                    
                    # Check if query starts with MATCH clause (common requirement for valid Cypher)
                    if not query.upper().strip().startswith("MATCH") and not query.upper().strip().startswith("CALL"):
                        dump_warnings.append({
                            "warning": "query_syntax",
                            "message": "Query might not be valid Cypher - consider starting with MATCH or CALL"
                        })
                    
                    # Check for dangerous patterns in query with categorized risk levels
                    high_risk_patterns = ['DROP', 'DELETE', 'CREATE', 'MERGE', 'REMOVE', 'SET', 'exec(', 'eval(', 'DETACH']
                    medium_risk_patterns = [';', '--', '/*', '*/', 'UNION', 'FOREACH']
                    
                    # Check for high risk patterns (block these operations)
                    for pattern in high_risk_patterns:
                        if pattern.upper() in query.upper():
                            error_response = create_error_response(
                                message=f"Query contains prohibited operation: {pattern}",
                                code="security_violation",
                                details={
                                    "pattern": pattern, 
                                    "query": query,
                                    "risk_level": "high",
                                    "allowed_operations": "READ-ONLY queries (MATCH, RETURN, WITH, etc.)"
                                }
                            )
                            # Log security violations for monitoring
                            logger.warn(f"Security violation in debug dump query: {pattern}", context={
                                "query": query,
                                "client_id": client_id or "default",
                                "pattern": pattern
                            })
                            return model_to_json(error_response)
                    
                    # Check for medium risk patterns (warn only)
                    for pattern in medium_risk_patterns:
                        if pattern.upper() in query.upper():
                            dump_warnings.append({
                                "warning": "query_medium_risk_pattern",
                                "message": f"Query contains pattern with medium security risk: {pattern}",
                                "risk_level": "medium"
                            })
                    
                    # Require explicit confirmation for custom queries
                    if not confirm:
                        error_response = create_error_response(
                            message="Custom queries require explicit confirmation. Set confirm=True to proceed.",
                            code="confirmation_required",
                            details={"query": query, "confirmation_required": True}
                        )
                        return model_to_json(error_response)
            
            # Sanitize client_id if provided
            if client_id is not None:
                original_client_id = client_id
                client_id = str(client_id).strip()
                
                # Check if sanitization changed the client ID
                if client_id != original_client_id:
                    dump_warnings.append({
                        "warning": "client_id_sanitized",
                        "message": "Client ID was sanitized by removing whitespace"
                    })
                
                # Check client_id length
                if len(client_id) > 100:  # Prevent excessively long IDs
                    original_length = len(client_id)
                    logger.warn(f"Client ID too long, truncating: {client_id[:20]}...")
                    client_id = client_id[:100]
                    dump_warnings.append({
                        "warning": "client_id_truncated",
                        "message": f"Client ID was too long, truncated from {original_length} to 100 characters"
                    })
                
                # Check for dangerous patterns in client_id
                dangerous_patterns = [';', '--', '/*', '*/', '@@', 'exec(', 'eval(', 'DROP', 'DELETE', 'CREATE', 'MERGE']
                for pattern in dangerous_patterns:
                    if pattern.upper() in client_id.upper():
                        error_response = create_error_response(
                            message=f"Client ID contains potentially dangerous pattern: {pattern}",
                            code="security_violation",
                            details={"pattern": pattern, "client_id": client_id}
                        )
                        return model_to_json(error_response)
            
            # Check if this is a debug environment - we might want to limit this functionality in production
            is_debug_allowed = True  # Default to allowed
            
            # Check for environment-specific restrictions (could be expanded)
            env = os.getenv("MCP_ENVIRONMENT", "development").lower()
            if env == "production" and not os.getenv("MCP_ALLOW_DEBUG_DUMP", "").lower() in ["true", "1", "yes"]:
                is_debug_allowed = False
                
            if not is_debug_allowed:
                error_response = create_error_response(
                    message="Debug dump operations are disabled in this environment",
                    code="operation_disabled",
                    details={"environment": env}
                )
                return model_to_json(error_response)
            
            # Log the dump request
            logger.info(f"Executing Neo4j debug dump with limit={limit}", context={
                "client_id": client_id or "default",
                "custom_query": bool(query),
                "dump_limit": limit,
                "include_relationships": include_relationships,
                "include_statistics": include_statistics
            })
                
            # Get the client-specific graph manager
            client_graph_manager = get_client_manager(client_id)
            
            # Prepare parameters for debug dump
            dump_params = {
                "limit": limit,
                "include_relationships": include_relationships,
                "include_statistics": include_statistics
            }
            
            # Add query if provided
            if query:
                dump_params["query"] = query
            
            # Dump Neo4j database contents (support both parameter styles)
            try:
                # Call the function directly with individual parameters
                if query:
                    result = client_graph_manager.debug_dump_neo4j(limit=limit, query=query)
                else:
                    result = client_graph_manager.debug_dump_neo4j(limit=limit)
            except Exception as call_error:
                logger.warn(f"Error calling debug_dump_neo4j: {str(call_error)}")
                # Attempt with minimal parameters
                result = client_graph_manager.debug_dump_neo4j(limit=limit)
                dump_warnings.append({
                    "warning": "call_error",
                    "message": f"Error calling debug_dump_neo4j: {str(call_error)}, using minimal parameters"
                })
            
            # Process the result for consistency
            try:
                # Convert string result to parsed object if needed
                if isinstance(result, str):
                    try:
                        parsed_result = json.loads(result)
                    except json.JSONDecodeError as json_error:
                        logger.error(f"Error parsing dump response: {str(json_error)}")
                        error_response = create_error_response(
                            message=f"Error parsing Neo4j dump result: {str(json_error)}",
                            code="response_parsing_error",
                            details={
                                "error_position": getattr(json_error, "pos", None),
                                "error_line": getattr(json_error, "lineno", None),
                                "error_column": getattr(json_error, "colno", None)
                            }
                        )
                        return model_to_json(error_response)
                else:
                    parsed_result = result
                
                # Check if it's an error response
                if isinstance(parsed_result, dict) and parsed_result.get("status") == "error":
                    error_response = create_error_response(
                        message=parsed_result.get("error", "Unknown error during Neo4j dump"),
                        code="dump_error",
                        details={
                            "limit": limit, 
                            "query": query if query else None,
                            "include_relationships": include_relationships,
                            "include_statistics": include_statistics
                        }
                    )
                    return model_to_json(error_response)
                
                # Process successful result
                # Make a copy of the result data to avoid modifying the original
                sanitized_result = copy.deepcopy(parsed_result)
                
                # Sanitize sensitive information in the response
                if isinstance(sanitized_result, dict):
                    # Sanitize nodes
                    if "data" in sanitized_result and "nodes" in sanitized_result["data"]:
                        for node in sanitized_result["data"]["nodes"]:
                            if "properties" in node:
                                for key in list(node["properties"].keys()):
                                    if any(sensitive in key.lower() for sensitive in ["password", "secret", "token", "key", "auth", "credential"]):
                                        node["properties"][key] = "[REDACTED]"
                    
                    # Sanitize relationships if they contain properties
                    if "data" in sanitized_result and "relationships" in sanitized_result["data"]:
                        for rel in sanitized_result["data"]["relationships"]:
                            if "properties" in rel:
                                for key in list(rel["properties"].keys()):
                                    if any(sensitive in key.lower() for sensitive in ["password", "secret", "token", "key", "auth", "credential"]):
                                        rel["properties"][key] = "[REDACTED]"
                
                # Format success message with appropriate details
                query_info = f" with custom query" if query else ""
                relationship_info = "" if include_relationships else " (relationships excluded)"
                stats_info = "" if include_statistics else " (statistics excluded)"
                
                success_response = create_success_response(
                    message=f"Successfully dumped Neo4j database (limit: {limit}{query_info}{relationship_info}{stats_info})",
                    data=sanitized_result
                )
                
                # Add execution context information
                success_response.data["execution_context"] = {
                    "timestamp": datetime.datetime.now().isoformat(),
                    "parameters": {
                        "limit": limit,
                        "custom_query": bool(query),
                        "include_relationships": include_relationships,
                        "include_statistics": include_statistics
                    }
                }
                
                # Add any warnings collected during processing
                if dump_warnings:
                    success_response.data["warnings"] = dump_warnings
                    
                return model_to_json(success_response)
                
            except Exception as parse_error:
                logger.error(f"Error processing dump result: {str(parse_error)}", exc_info=True)
                error_response = create_error_response(
                    message=f"Error processing dump result: {str(parse_error)}",
                    code="result_processing_error",
                    details={"error": str(parse_error)}
                )
                return model_to_json(error_response)
            
        except Exception as e:
            logger.error(f"Error dumping Neo4j database: {str(e)}", exc_info=True)
            error_response = create_error_response(
                message=f"Error dumping Neo4j database: {str(e)}",
                code="dump_error",
                details={
                    "limit": limit if 'limit' in locals() else None, 
                    "query": query if 'query' in locals() and query else None,
                    "include_relationships": include_relationships if 'include_relationships' in locals() else None,
                    "include_statistics": include_statistics if 'include_statistics' in locals() else None
                }
            )
            return model_to_json(error_response)
    
    # Return the registered tools
    return {
        "create_entities": create_entities,
        "create_relations": create_relations,
        "add_observations": add_observations,
        "search_nodes": search_nodes,
        "delete_entity": delete_entity,
        "delete_relation": delete_relation,
        "delete_observation": delete_observation,
        "get_all_memories": get_all_memories,
        "delete_all_memories": delete_all_memories,
        "debug_dump_neo4j": debug_dump_neo4j
    } 