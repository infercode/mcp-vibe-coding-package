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
    async def get_lesson_container(container_id: str, client_id: Optional[str] = None) -> str:
        """
        Retrieve a lesson container by ID or name.
        
        Args:
            container_id: The ID or name of the lesson container
            client_id: Optional client ID for identifying the connection
                
        Returns:
            JSON response with lesson container data
        """
        try:
            # Get the graph manager with the appropriate client context
            client_graph_manager = get_client_manager(client_id)
            
            # Get lesson container
            result = client_graph_manager.lesson_memory.get_lesson_container(container_id)
            
            # Handle the response
            def success_handler(data):
                container_data = data.get("container", {})
                return create_container_response(
                    container_data=container_data,
                    message=f"Lesson container retrieved successfully"
                )
                
            return handle_lesson_response(
                result=result,
                success_handler=success_handler,
                error_code="container_retrieval_error"
            )
            
        except Exception as e:
            logger.error(f"Error retrieving lesson container: {str(e)}")
            error_response = create_lesson_error_response(
                message=f"Failed to retrieve lesson container: {str(e)}",
                code="container_retrieval_error"
            )
            return model_to_json(error_response)
    
    @server.tool()
    async def update_lesson_container(container_data: Dict[str, Any], client_id: Optional[str] = None) -> str:
        """
        Update an existing lesson container.

        Args:
            container_data: Dictionary containing container data. Required fields:
                - container_name: Name of the container to update
                Optional fields:
                - description: Updated description for the container
                - metadata: Additional metadata for the container
            client_id: Optional client ID for identifying the connection

        Returns:
            JSON string containing updated container information with status, message, and container details
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
                container_model = LessonContainerUpdate(**container_data)
            except Exception as e:
                logger.error(f"Validation error for container update data: {str(e)}")
                error_response = create_lesson_error_response(
                    message=f"Invalid container update data: {str(e)}",
                    code="validation_error"
                )
                return model_to_json(error_response)
            
            # Get the graph manager with the appropriate client context
            client_graph_manager = get_client_manager(effective_client_id)
            
            # Call the update method with validated data
            result = client_graph_manager.lesson_memory.update_lesson_container(model_to_dict(container_model))
            
            # Handle the response
            def success_handler(data):
                container_data = data.get("container", {})
                return create_container_response(
                    container_data=container_data,
                    message=f"Lesson container updated successfully"
                )
                
            return handle_lesson_response(
                result=result,
                success_handler=success_handler,
                error_code="container_update_error"
            )
            
        except Exception as e:
            logger.error(f"Error updating lesson container: {str(e)}")
            error_response = create_lesson_error_response(
                message=f"Failed to update lesson container: {str(e)}",
                code="container_update_error"
            )
            return model_to_json(error_response)
    
    @server.tool()
    async def delete_lesson_container(container_id: str, client_id: Optional[str] = None) -> str:
        """
        Delete a lesson container and all its associated entities.
        
        Args:
            container_id: The ID or name of the lesson container to delete
            client_id: Optional client ID for identifying the connection
                
        Returns:
            JSON response with operation result
        """
        try:
            # Get the graph manager with the appropriate client context
            client_graph_manager = get_client_manager(client_id)
            
            # Delete lesson container
            result = client_graph_manager.lesson_memory.delete_lesson_container(container_id)
            
            # Handle the response
            def success_handler(data):
                return create_container_response(
                    container_data={},
                    message=f"Lesson container '{container_id}' deleted successfully"
                )
                
            return handle_lesson_response(
                result=result,
                success_handler=success_handler,
                error_code="container_deletion_error"
            )
            
        except Exception as e:
            logger.error(f"Error deleting lesson container: {str(e)}")
            error_response = create_lesson_error_response(
                message=f"Failed to delete lesson container: {str(e)}",
                code="container_deletion_error"
            )
            return model_to_json(error_response)
    
    @server.tool()
    async def list_lesson_containers(filters: Optional[Dict[str, Any]] = None, client_id: Optional[str] = None) -> str:
        """
        List all lesson containers or filter by specific criteria.
        
        Args:
            filters: Optional filters for searching containers
                - tags: Optional list of tags to filter by
                - name_contains: Optional string to search in container names
                - limit: Optional maximum number of results to return
            client_id: Optional client ID for identifying the connection
                
        Returns:
            JSON response with lesson containers matching the criteria
        """
        try:
            # Set default filters if none provided
            if filters is None:
                filters = {}
            
            # Get the graph manager with the appropriate client context
            client_graph_manager = get_client_manager(client_id)
            
            # List lesson containers
            result = client_graph_manager.lesson_memory.list_lesson_containers(filters)
            
            # Handle the response
            def success_handler(data):
                containers = data.get("containers", [])
                return create_lesson_error_response(
                    message=f"Found {len(containers)} lesson containers",
                    code="success",
                    details={"containers": containers}
                )
                
            return handle_lesson_response(
                result=result,
                success_handler=success_handler,
                error_code="container_listing_error"
            )
            
        except Exception as e:
            logger.error(f"Error listing lesson containers: {str(e)}")
            error_response = create_lesson_error_response(
                message=f"Failed to list lesson containers: {str(e)}",
                code="container_listing_error"
            )
            return model_to_json(error_response)
    
    @server.tool()
    async def create_lesson_entity(entity_data: Dict[str, Any], client_id: Optional[str] = None) -> str:
        """
        Create a lesson entity.
        
        Args:
            entity_data: Entity data including name, type, and container
            client_id: Optional client ID for identifying the connection
            
        Returns:
            JSON string with the created entity
        """
        try:
            # Sanitize and validate input
            if not isinstance(entity_data, dict):
                error_response = create_lesson_error_response(
                    message="Entity data must be a dictionary",
                    code="invalid_input"
                )
                return model_to_json(error_response)
                
            # Ensure required fields are present
            required_fields = ["name", "container"]
            for field in required_fields:
                if field not in entity_data or not entity_data[field]:
                    error_response = create_lesson_error_response(
                        message=f"Missing required field: {field}",
                        code="missing_field"
                    )
                    return model_to_json(error_response)
                    
            # Sanitize string fields
            for field in ["name", "container", "type", "description"]:
                if field in entity_data and entity_data[field] is not None:
                    entity_data[field] = str(entity_data[field]).strip()
            
            # Validate input using Pydantic model
            try:
                # Extract client_id from metadata if present and not provided directly
                metadata_client_id = None
                if "metadata" in entity_data and isinstance(entity_data["metadata"], dict):
                    metadata_client_id = entity_data["metadata"].get("client_id")
                
                # Use the explicitly provided client_id, or the one from metadata
                effective_client_id = client_id or metadata_client_id
                
                # Add client_id to metadata if it doesn't exist but was provided
                if effective_client_id:
                    if "metadata" not in entity_data:
                        entity_data["metadata"] = {}
                    entity_data["metadata"]["client_id"] = effective_client_id
                
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
            client_graph_manager = get_client_manager(effective_client_id)
            
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
    
    @server.tool()
    async def create_lesson_section(section_data: Dict[str, Any], client_id: Optional[str] = None) -> str:
        """
        Create a new section within a lesson container.
        
        Args:
            section_data: Dictionary containing section information
                - container_name: Required. Name of the lesson container
                - title: Required. Title of the section
                - order: Optional. Order of the section within the container (integer)
                - content: Optional. Content of the section
                - metadata: Optional. Additional metadata
            client_id: Optional client ID for identifying the connection
                
        Returns:
            JSON response with operation result
        """
        try:
            # Add client_id to metadata if provided
            if client_id:
                if "metadata" not in section_data:
                    section_data["metadata"] = {}
                section_data["metadata"]["client_id"] = client_id
            
            # Get the graph manager with the appropriate client context
            client_graph_manager = get_client_manager(client_id)
            
            # Call the create method
            result = client_graph_manager.lesson_memory.create_lesson_section(section_data)
            
            # Handle the response
            def success_handler(data):
                section_data = data.get("section", {})
                return create_lesson_error_response(
                    message=f"Lesson section '{section_data.get('title', '')}' created successfully",
                    code="success",
                    details={"section": section_data}
                )
                
            return handle_lesson_response(
                result=result,
                success_handler=success_handler,
                error_code="section_creation_error"
            )
            
        except Exception as e:
            logger.error(f"Error creating lesson section: {str(e)}")
            error_response = create_lesson_error_response(
                message=f"Failed to create lesson section: {str(e)}",
                code="section_creation_error"
            )
            return model_to_json(error_response)
    
    @server.tool()
    async def update_lesson_section(section_data: Dict[str, Any], client_id: Optional[str] = None) -> str:
        """
        Update an existing section within a lesson container.
        
        Args:
            section_data: Dictionary containing section information
                - id: Required. ID of the section to update
                - container_name: Optional. Name of the lesson container
                - title: Optional. Updated title of the section
                - order: Optional. Updated order of the section
                - content: Optional. Updated content of the section
                - metadata: Optional. Updated metadata
            client_id: Optional client ID for identifying the connection
                
        Returns:
            JSON response with operation result
        """
        try:
            # Add client_id to metadata if provided
            if client_id:
                if "metadata" not in section_data:
                    section_data["metadata"] = {}
                section_data["metadata"]["client_id"] = client_id
            
            # Get the graph manager with the appropriate client context
            client_graph_manager = get_client_manager(client_id)
            
            # Call the update method
            result = client_graph_manager.lesson_memory.update_lesson_section(section_data)
            
            # Handle the response
            def success_handler(data):
                section_data = data.get("section", {})
                return create_lesson_error_response(
                    message=f"Lesson section updated successfully",
                    code="success",
                    details={"section": section_data}
                )
                
            return handle_lesson_response(
                result=result,
                success_handler=success_handler,
                error_code="section_update_error"
            )
            
        except Exception as e:
            logger.error(f"Error updating lesson section: {str(e)}")
            error_response = create_lesson_error_response(
                message=f"Failed to update lesson section: {str(e)}",
                code="section_update_error"
            )
            return model_to_json(error_response)
    
    @server.tool()
    async def create_module(module_data: Dict[str, Any], client_id: Optional[str] = None) -> str:
        """
        Create a module that groups multiple lesson containers.
        
        Args:
            module_data: Dictionary containing module information
                - name: Required. Name of the module
                - description: Optional. Description of the module
                - lessons: Optional. List of lesson IDs or names to include in the module
                - prerequisites: Optional. List of prerequisite module IDs or names
                - metadata: Optional. Additional metadata
            client_id: Optional client ID for identifying the connection
                
        Returns:
            JSON response with operation result
        """
        try:
            # Add client_id to metadata if provided
            if client_id:
                if "metadata" not in module_data:
                    module_data["metadata"] = {}
                module_data["metadata"]["client_id"] = client_id
            
            # Get the graph manager with the appropriate client context
            client_graph_manager = get_client_manager(client_id)
            
            # Call the create method
            result = client_graph_manager.lesson_memory.create_module(module_data)
            
            # Handle the response
            def success_handler(data):
                module_data = data.get("module", {})
                return create_lesson_error_response(
                    message=f"Module '{module_data.get('name', '')}' created successfully",
                    code="success",
                    details={"module": module_data}
                )
                
            return handle_lesson_response(
                result=result,
                success_handler=success_handler,
                error_code="module_creation_error"
            )
            
        except Exception as e:
            logger.error(f"Error creating module: {str(e)}")
            error_response = create_lesson_error_response(
                message=f"Failed to create module: {str(e)}",
                code="module_creation_error"
            )
            return model_to_json(error_response)
    
    @server.tool()
    async def analyze_lesson_prerequisites(container_id: str, client_id: Optional[str] = None) -> str:
        """
        Analyze prerequisites for a lesson container.
        
        Args:
            container_id: The ID or name of the lesson container
            client_id: Optional client ID for identifying the connection
                
        Returns:
            JSON response with prerequisite analysis
        """
        try:
            # Get the graph manager with the appropriate client context
            client_graph_manager = get_client_manager(client_id)
            
            # Call the analysis method
            result = client_graph_manager.lesson_memory.analyze_lesson_prerequisites(container_id)
            
            # Handle the response
            def success_handler(data):
                prerequisites = data.get("prerequisites", [])
                return create_lesson_error_response(
                    message=f"Analyzed prerequisites for lesson '{container_id}'",
                    code="success",
                    details={"prerequisites": prerequisites}
                )
                
            return handle_lesson_response(
                result=result,
                success_handler=success_handler,
                error_code="prerequisite_analysis_error"
            )
            
        except Exception as e:
            logger.error(f"Error analyzing lesson prerequisites: {str(e)}")
            error_response = create_lesson_error_response(
                message=f"Failed to analyze lesson prerequisites: {str(e)}",
                code="prerequisite_analysis_error"
            )
            return model_to_json(error_response)
    
    @server.tool()
    async def find_related_lessons(container_id: str, similarity_threshold: float = 0.7, client_id: Optional[str] = None) -> str:
        """
        Find lessons related to the specified lesson container based on content similarity.
        
        Args:
            container_id: The ID or name of the lesson container
            similarity_threshold: Minimum similarity threshold (0.0-1.0, default 0.7)
            client_id: Optional client ID for identifying the connection
                
        Returns:
            JSON response with related lessons
        """
        try:
            # Validate input parameter
            if not 0.0 <= similarity_threshold <= 1.0:
                error_response = create_lesson_error_response(
                    message=f"Invalid similarity threshold: {similarity_threshold}. Must be between 0.0 and 1.0.",
                    code="validation_error"
                )
                return model_to_json(error_response)
            
            # Get the graph manager with the appropriate client context
            client_graph_manager = get_client_manager(client_id)
            
            # Call the search method
            result = client_graph_manager.lesson_memory.find_related_lessons(container_id, similarity_threshold)
            
            # Handle the response
            def success_handler(data):
                related_lessons = data.get("related_lessons", [])
                return create_lesson_error_response(
                    message=f"Found {len(related_lessons)} related lessons for '{container_id}'",
                    code="success",
                    details={"related_lessons": related_lessons}
                )
                
            return handle_lesson_response(
                result=result,
                success_handler=success_handler,
                error_code="related_lessons_error"
            )
            
        except Exception as e:
            logger.error(f"Error finding related lessons: {str(e)}")
            error_response = create_lesson_error_response(
                message=f"Failed to find related lessons: {str(e)}",
                code="related_lessons_error"
            )
            return model_to_json(error_response)
    
    @server.tool()
    async def generate_learning_path(
        topic: str, 
        difficulty: Optional[str] = None, 
        prerequisite_knowledge: Optional[List[str]] = None,
        max_lessons: int = 5,
        client_id: Optional[str] = None
    ) -> str:
        """
        Generate a learning path for a specific topic.
        
        Args:
            topic: The main topic for the learning path
            difficulty: Optional difficulty level (e.g., "beginner", "intermediate", "advanced")
            prerequisite_knowledge: Optional list of topics the user already knows
            max_lessons: Maximum number of lessons to include in the path (default 5)
            client_id: Optional client ID for identifying the connection
                
        Returns:
            JSON response with the generated learning path
        """
        try:
            # Prepare parameters
            params = {
                "topic": topic,
                "max_lessons": max_lessons
            }
            
            if difficulty:
                params["difficulty"] = difficulty
                
            if prerequisite_knowledge:
                params["prerequisite_knowledge"] = prerequisite_knowledge
            
            # Get the graph manager with the appropriate client context
            client_graph_manager = get_client_manager(client_id)
            
            # Call the generation method
            result = client_graph_manager.lesson_memory.generate_learning_path(params)
            
            # Handle the response
            def success_handler(data):
                learning_path = data.get("learning_path", [])
                return create_lesson_error_response(
                    message=f"Generated learning path for topic '{topic}'",
                    code="success",
                    details={"learning_path": learning_path}
                )
                
            return handle_lesson_response(
                result=result,
                success_handler=success_handler,
                error_code="learning_path_error"
            )
            
        except Exception as e:
            logger.error(f"Error generating learning path: {str(e)}")
            error_response = create_lesson_error_response(
                message=f"Failed to generate learning path: {str(e)}",
                code="learning_path_error"
            )
            return model_to_json(error_response)
    
    @server.tool()
    async def get_lesson_entity(entity_name: str, container_name: Optional[str] = None, client_id: Optional[str] = None) -> str:
        """
        Get details of a lesson entity.
        
        Args:
            entity_name: Name of the entity to retrieve
            container_name: Optional name of the container to narrow down the search
            client_id: Optional client ID for identifying the connection
                
        Returns:
            JSON response with entity details
        """
        try:
            # Get the graph manager with the appropriate client context
            client_graph_manager = get_client_manager(client_id)
            
            # Call the get entity method
            result = client_graph_manager.lesson_memory.get_lesson_entity(entity_name, container_name)
            
            # Handle the response
            def success_handler(data):
                entity_data = data.get("entity", {})
                return create_entity_response(
                    entity_data=entity_data,
                    message=f"Retrieved lesson entity '{entity_name}' successfully"
                )
                
            return handle_lesson_response(
                result=result,
                success_handler=success_handler,
                error_code="entity_retrieval_error"
            )
            
        except Exception as e:
            logger.error(f"Error retrieving lesson entity: {str(e)}")
            error_response = create_lesson_error_response(
                message=f"Failed to retrieve lesson entity: {str(e)}",
                code="entity_retrieval_error"
            )
            return model_to_json(error_response)
    
    @server.tool()
    async def get_lesson_observations(entity_name: str, container_name: Optional[str] = None, client_id: Optional[str] = None) -> str:
        """
        Get observations for a lesson entity.
        
        Args:
            entity_name: Name of the entity to retrieve observations for
            container_name: Optional name of the container to narrow down the search
            client_id: Optional client ID for identifying the connection
                
        Returns:
            JSON response with entity observations
        """
        try:
            # Get the graph manager with the appropriate client context
            client_graph_manager = get_client_manager(client_id)
            
            # Call the get observations method
            result = client_graph_manager.lesson_memory.get_lesson_observations(entity_name, container_name)
            
            # Handle the response
            def success_handler(data):
                observations = data.get("observations", [])
                return create_observation_response(
                    entity_name=entity_name,
                    observations=observations,
                    message=f"Retrieved {len(observations)} observations for entity '{entity_name}'"
                )
                
            return handle_lesson_response(
                result=result,
                success_handler=success_handler,
                error_code="observation_retrieval_error"
            )
            
        except Exception as e:
            logger.error(f"Error retrieving lesson observations: {str(e)}")
            error_response = create_lesson_error_response(
                message=f"Failed to retrieve lesson observations: {str(e)}",
                code="observation_retrieval_error"
            )
            return model_to_json(error_response)
    
    # Return registered tools
    return {
        "create_lesson_container": create_lesson_container,
        "get_lesson_container": get_lesson_container,
        "update_lesson_container": update_lesson_container,
        "delete_lesson_container": delete_lesson_container,
        "list_lesson_containers": list_lesson_containers,
        "create_lesson_entity": create_lesson_entity,
        "add_lesson_observation": add_lesson_observation,
        "create_structured_lesson_observations": create_structured_lesson_observations,
        "create_lesson_relationship": create_lesson_relationship,
        "search_lesson_entities": search_lesson_entities,
        "create_lesson_section": create_lesson_section,
        "update_lesson_section": update_lesson_section,
        "create_module": create_module,
        "analyze_lesson_prerequisites": analyze_lesson_prerequisites,
        "find_related_lessons": find_related_lessons,
        "generate_learning_path": generate_learning_path,
        "get_lesson_entity": get_lesson_entity,
        "get_lesson_observations": get_lesson_observations,
    } 