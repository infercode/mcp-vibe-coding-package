#!/usr/bin/env python3
"""
Lesson Memory Tools with Pydantic Integration

This module implements MCP tools for the lesson memory system using
Pydantic models for validation and serialization.
"""

from typing import Dict, List, Optional, Any, Union
from datetime import datetime
import json

from src.logger import get_logger
from src.models.lesson_memory import (
    LessonEntityCreate, LessonEntityUpdate, LessonObservationCreate, LessonObservationUpdate,
    StructuredLessonObservations, LessonRelationshipCreate, LessonContainerCreate,
    LessonContainerUpdate, SearchQuery
)
from src.models.lesson_responses import (
    create_entity_response, create_observation_response, create_container_response,
    create_relationship_response, create_search_response, create_lesson_error_response,
    parse_legacy_result, handle_lesson_response, model_to_json
)
from src.models.responses import model_to_dict

# Initialize logger
logger = get_logger()


def register_lesson_tools(server, get_client_manager):
    """Register lesson memory tools with the server."""
    
    # Lesson Container Management Tools
    @server.tool()
    async def create_lesson_container(container_data: Dict[str, Any], client_id: Optional[str] = None) -> str:
        """
        Create a new lesson container in the knowledge graph.
        
        Args:
            container_data: Dictionary containing container information
                - name: Required. The name of the container
                - description: Optional. Description of the container
                - tags: Optional. List of tags for categorizing the container
                - metadata: Optional. Additional metadata
            client_id: Optional client ID for identifying the connection
                
        Returns:
            JSON response with operation result
        """
        try:
            # Validate input using Pydantic model
            try:
                # Extract client_id from metadata if present and not provided directly
                metadata_client_id = None
                if "metadata" in container_data and isinstance(container_data["metadata"], dict):
                    metadata_client_id = container_data["metadata"].get("client_id")
                
                # Use the explicitly provided client_id, or the one from metadata
                effective_client_id = client_id or metadata_client_id
                
                # Add client_id to metadata if it doesn't exist but was provided
                if effective_client_id:
                    if "metadata" not in container_data:
                        container_data["metadata"] = {}
                    container_data["metadata"]["client_id"] = effective_client_id
                
                # Create Pydantic model for validation
                container_model = LessonContainerCreate(**container_data)
            except Exception as e:
                logger.error(f"Validation error for container data: {str(e)}")
                error_response = create_lesson_error_response(
                    message=f"Invalid container data: {str(e)}",
                    code="validation_error"
                )
                return model_to_json(error_response)
            
            # Get the graph manager with the appropriate client context
            client_graph_manager = get_client_manager(effective_client_id)
            
            # Call the create method with validated data
            result = client_graph_manager.lesson_memory.create_lesson_container(model_to_dict(container_model))
            
            # Handle the response
            def success_handler(data):
                container_data = data.get("container", {})
                return create_container_response(
                    container_data=container_data,
                    message=f"Lesson container '{container_model.name}' created successfully"
                )
                
            return handle_lesson_response(
                result=result,
                success_handler=success_handler,
                error_code="container_creation_error"
            )
            
        except Exception as e:
            logger.error(f"Error creating lesson container: {str(e)}")
            error_response = create_lesson_error_response(
                message=f"Failed to create lesson container: {str(e)}",
                code="container_creation_error"
            )
            return model_to_json(error_response)
    
    @server.tool()
    async def create_lesson_entity(entity_data: Dict[str, Any], client_id: Optional[str] = None) -> str:
        """
        Create a new lesson entity and add it to a container.
        
        Args:
            entity_data: Dictionary containing entity information
                - container_name: Required. Name of the container to add the entity to
                - entity_name: Required. Name of the entity
                - entity_type: Required. Type of the entity
                - observations: Optional. List of observations
                - metadata: Optional. Additional metadata
            client_id: Optional client ID for identifying the connection
                
        Returns:
            JSON response with operation result
        """
        try:
            # Validate input using Pydantic model
            try:
                # Add client_id to metadata if provided
                if client_id:
                    if "metadata" not in entity_data:
                        entity_data["metadata"] = {}
                    entity_data["metadata"]["client_id"] = client_id
                
                # Create Pydantic model for validation
                entity_model = LessonEntityCreate(**entity_data)
            except Exception as e:
                logger.error(f"Validation error for entity data: {str(e)}")
                error_response = create_lesson_error_response(
                    message=f"Invalid entity data: {str(e)}",
                    code="validation_error"
                )
                return model_to_json(error_response)
            
            # Get the graph manager
            client_graph_manager = get_client_manager(client_id)
            
            # Call the create method
            result = client_graph_manager.lesson_memory.create_lesson_entity(
                container_name=entity_model.container_name,
                entity_name=entity_model.entity_name,
                entity_type=entity_model.entity_type,
                observations=entity_model.observations,
                metadata=entity_model.metadata
            )
            
            # Handle the response
            def success_handler(data):
                entity_data = data.get("entity", {})
                return create_entity_response(
                    entity_data=entity_data,
                    message=f"Lesson entity '{entity_model.entity_name}' created successfully"
                )
                
            return handle_lesson_response(
                result=result,
                success_handler=success_handler,
                error_code="entity_creation_error"
            )
            
        except Exception as e:
            logger.error(f"Error creating lesson entity: {str(e)}")
            error_response = create_lesson_error_response(
                message=f"Failed to create lesson entity: {str(e)}",
                code="entity_creation_error"
            )
            return model_to_json(error_response)
    
    @server.tool()
    async def add_lesson_observation(observation_data: Dict[str, Any], client_id: Optional[str] = None) -> str:
        """
        Add a structured observation to a lesson entity.
        
        Args:
            observation_data: Dictionary containing observation information
                - entity_name: Required. Name of the entity to add observation to
                - content: Required. Content of the observation
                - observation_type: Required. Type of observation
                - container_name: Optional. Container to verify entity membership
            client_id: Optional client ID for identifying the connection
                
        Returns:
            JSON response with operation result
        """
        try:
            # Validate input using Pydantic model
            try:
                # Create Pydantic model for validation
                observation_model = LessonObservationCreate(**observation_data)
            except Exception as e:
                logger.error(f"Validation error for observation data: {str(e)}")
                error_response = create_lesson_error_response(
                    message=f"Invalid observation data: {str(e)}",
                    code="validation_error"
                )
                return model_to_json(error_response)
            
            # Get the graph manager
            client_graph_manager = get_client_manager(client_id)
            
            # Call the add method
            result = client_graph_manager.lesson_memory.add_lesson_observation(
                entity_name=observation_model.entity_name,
                content=observation_model.content,
                observation_type=observation_model.observation_type,
                container_name=observation_model.container_name
            )
            
            # Handle the response
            def success_handler(data):
                observations = data.get("observations", [])
                return create_observation_response(
                    entity_name=observation_model.entity_name,
                    observations=observations,
                    message=f"Observation added to entity '{observation_model.entity_name}'"
                )
                
            return handle_lesson_response(
                result=result,
                success_handler=success_handler,
                error_code="observation_creation_error"
            )
            
        except Exception as e:
            logger.error(f"Error adding lesson observation: {str(e)}")
            error_response = create_lesson_error_response(
                message=f"Failed to add lesson observation: {str(e)}",
                code="observation_creation_error"
            )
            return model_to_json(error_response)
    
    @server.tool()
    async def create_structured_lesson_observations(observation_data: Dict[str, Any], client_id: Optional[str] = None) -> str:
        """
        Create structured observations for a lesson entity.
        
        Args:
            observation_data: Dictionary containing structured observations
                - entity_name: Required. Name of the entity
                - what_was_learned: Optional. Factual knowledge gained
                - why_it_matters: Optional. Importance and consequences
                - how_to_apply: Optional. Application guidance
                - root_cause: Optional. Underlying causes
                - evidence: Optional. Examples and supporting data
                - container_name: Optional. Container to verify entity membership
            client_id: Optional client ID for identifying the connection
                
        Returns:
            JSON response with operation result
        """
        try:
            # Validate input using Pydantic model
            try:
                # Create Pydantic model for validation
                structured_model = StructuredLessonObservations(**observation_data)
            except Exception as e:
                logger.error(f"Validation error for structured observations: {str(e)}")
                error_response = create_lesson_error_response(
                    message=f"Invalid structured observations: {str(e)}",
                    code="validation_error"
                )
                return model_to_json(error_response)
            
            # Get the graph manager
            client_graph_manager = get_client_manager(client_id)
            
            # Call the create method
            result = client_graph_manager.lesson_memory.create_structured_lesson_observations(
                entity_name=structured_model.entity_name,
                what_was_learned=structured_model.what_was_learned,
                why_it_matters=structured_model.why_it_matters,
                how_to_apply=structured_model.how_to_apply,
                root_cause=structured_model.root_cause,
                evidence=structured_model.evidence,
                container_name=structured_model.container_name
            )
            
            # Handle the response
            def success_handler(data):
                observations = data.get("observations", [])
                observations_by_type = data.get("observations_by_type", {})
                return create_observation_response(
                    entity_name=structured_model.entity_name,
                    observations=observations,
                    observations_by_type=observations_by_type,
                    message=f"Structured observations added to entity '{structured_model.entity_name}'"
                )
                
            return handle_lesson_response(
                result=result,
                success_handler=success_handler,
                error_code="structured_observations_error"
            )
            
        except Exception as e:
            logger.error(f"Error creating structured observations: {str(e)}")
            error_response = create_lesson_error_response(
                message=f"Failed to create structured observations: {str(e)}",
                code="structured_observations_error"
            )
            return model_to_json(error_response)
    
    @server.tool()
    async def create_lesson_relationship(relationship_data: Dict[str, Any], client_id: Optional[str] = None) -> str:
        """
        Create a relationship between lesson entities.
        
        Args:
            relationship_data: Dictionary containing relationship information
                - source_name: Required. Name of the source entity
                - target_name: Required. Name of the target entity
                - relationship_type: Required. Type of relationship
                - properties: Optional. Additional properties for the relationship
            client_id: Optional client ID for identifying the connection
                
        Returns:
            JSON response with operation result
        """
        try:
            # Validate input using Pydantic model
            try:
                # Create Pydantic model for validation
                relationship_model = LessonRelationshipCreate(**relationship_data)
            except Exception as e:
                logger.error(f"Validation error for relationship data: {str(e)}")
                error_response = create_lesson_error_response(
                    message=f"Invalid relationship data: {str(e)}",
                    code="validation_error"
                )
                return model_to_json(error_response)
            
            # Get the graph manager
            client_graph_manager = get_client_manager(client_id)
            
            # Call the create method
            result = client_graph_manager.lesson_memory.create_lesson_relationship(
                source_name=relationship_model.source_name,
                target_name=relationship_model.target_name,
                relationship_type=relationship_model.relationship_type,
                properties=relationship_model.properties
            )
            
            # Handle the response
            def success_handler(data):
                relationship_data = data.get("relationship", {})
                return create_relationship_response(
                    relationship_data=relationship_data,
                    message=f"Relationship created between '{relationship_model.source_name}' and '{relationship_model.target_name}'"
                )
                
            return handle_lesson_response(
                result=result,
                success_handler=success_handler,
                error_code="relationship_creation_error"
            )
            
        except Exception as e:
            logger.error(f"Error creating lesson relationship: {str(e)}")
            error_response = create_lesson_error_response(
                message=f"Failed to create lesson relationship: {str(e)}",
                code="relationship_creation_error"
            )
            return model_to_json(error_response)
    
    @server.tool()
    async def search_lesson_entities(search_data: Dict[str, Any], client_id: Optional[str] = None) -> str:
        """
        Search for lesson entities based on various criteria.
        
        Args:
            search_data: Dictionary containing search parameters
                - container_name: Optional. Container to search within
                - search_term: Optional. Text to search for
                - entity_type: Optional. Entity type to filter by
                - tags: Optional. Tags to filter by
                - limit: Optional. Maximum number of results
                - semantic: Optional. Whether to use semantic search
            client_id: Optional client ID for identifying the connection
                
        Returns:
            JSON response with search results
        """
        try:
            # Validate input using Pydantic model
            try:
                # Create Pydantic model for validation
                search_model = SearchQuery(**search_data)
            except Exception as e:
                logger.error(f"Validation error for search query: {str(e)}")
                error_response = create_lesson_error_response(
                    message=f"Invalid search query: {str(e)}",
                    code="validation_error"
                )
                return model_to_json(error_response)
            
            # Get the graph manager
            client_graph_manager = get_client_manager(client_id)
            
            # Call the search method
            result = client_graph_manager.lesson_memory.search_lesson_entities(
                container_name=search_model.container_name,
                search_term=search_model.search_term,
                entity_type=search_model.entity_type,
                tags=search_model.tags,
                limit=search_model.limit,
                semantic=search_model.semantic
            )
            
            # Handle the response
            def success_handler(data):
                results = data.get("results", [])
                total = data.get("total", len(results))
                return create_search_response(
                    results=results,
                    total_count=total,
                    query_params=model_to_dict(search_model),
                    message=f"Found {len(results)} lesson entities"
                )
                
            return handle_lesson_response(
                result=result,
                success_handler=success_handler,
                error_code="search_error"
            )
            
        except Exception as e:
            logger.error(f"Error searching lesson entities: {str(e)}")
            error_response = create_lesson_error_response(
                message=f"Failed to search lesson entities: {str(e)}",
                code="search_error"
            )
            return model_to_json(error_response)
    
    # Additional tools would follow the same pattern

    # Return registered tools
    return {
        "create_lesson_container": create_lesson_container,
        "create_lesson_entity": create_lesson_entity,
        "add_lesson_observation": add_lesson_observation,
        "create_structured_lesson_observations": create_structured_lesson_observations,
        "create_lesson_relationship": create_lesson_relationship,
        "search_lesson_entities": search_lesson_entities,
    } 