"""
Lesson Memory Response Utilities

This module provides standardized response handling for the lesson memory system.
"""

from typing import Dict, List, Optional, Any, Union, Callable, Annotated, Type, TypeVar
from datetime import datetime
import json

from pydantic import BaseModel, Field, ConfigDict

from src.models.lesson_memory import (
    EntityResponse, ObservationResponse, ContainerResponse,
    RelationshipResponse, SearchResponse, ErrorResponse, ErrorDetail
)

T = TypeVar('T', bound=BaseModel)


def create_entity_response(
    entity_data: Optional[Dict[str, Any]] = None,
    message: Optional[str] = None
) -> EntityResponse:
    """
    Create a standardized entity response.
    
    Args:
        entity_data: Optional entity data
        message: Optional success message
        
    Returns:
        EntityResponse model instance
    """
    return EntityResponse(
        status="success",
        timestamp=datetime.now(),
        message=message,
        entity=entity_data
    )


def create_observation_response(
    entity_name: Optional[str] = None,
    observations: Optional[List[Dict[str, Any]]] = None,
    observations_by_type: Optional[Dict[str, List[Dict[str, Any]]]] = None,
    message: Optional[str] = None
) -> ObservationResponse:
    """
    Create a standardized observation response.
    
    Args:
        entity_name: Optional entity name
        observations: Optional list of observations
        observations_by_type: Optional observations grouped by type
        message: Optional success message
        
    Returns:
        ObservationResponse model instance
    """
    return ObservationResponse(
        status="success",
        timestamp=datetime.now(),
        message=message,
        entity=entity_name,
        observations=observations,
        observations_by_type=observations_by_type
    )


def create_container_response(
    container_data: Optional[Dict[str, Any]] = None,
    message: Optional[str] = None
) -> ContainerResponse:
    """
    Create a standardized container response.
    
    Args:
        container_data: Optional container data
        message: Optional success message
        
    Returns:
        ContainerResponse model instance
    """
    return ContainerResponse(
        status="success",
        timestamp=datetime.now(),
        message=message,
        container=container_data
    )


def create_relationship_response(
    relationship_data: Optional[Dict[str, Any]] = None,
    message: Optional[str] = None
) -> RelationshipResponse:
    """
    Create a standardized relationship response.
    
    Args:
        relationship_data: Optional relationship data
        message: Optional success message
        
    Returns:
        RelationshipResponse model instance
    """
    return RelationshipResponse(
        status="success",
        timestamp=datetime.now(),
        message=message,
        relationship=relationship_data
    )


def create_search_response(
    results: List[Dict[str, Any]],
    total_count: int,
    query_params: Optional[Dict[str, Any]] = None,
    message: Optional[str] = None
) -> SearchResponse:
    """
    Create a standardized search response.
    
    Args:
        results: Search results
        total_count: Total number of results found
        query_params: Optional search query parameters
        message: Optional success message
        
    Returns:
        SearchResponse model instance
    """
    return SearchResponse(
        status="success",
        timestamp=datetime.now(),
        message=message,
        results=results,
        total_count=total_count,
        query=query_params
    )


def create_lesson_error_response(
    message: str, 
    code: str = "lesson_error", 
    details: Optional[Dict[str, Any]] = None
) -> ErrorResponse:
    """
    Create a standardized error response for lesson memory.
    
    Args:
        message: Error message
        code: Error code
        details: Optional additional error details
        
    Returns:
        ErrorResponse model instance
    """
    error_detail = ErrorDetail(
        code=code,
        message=message,
        details=details
    )
    
    return ErrorResponse(
        status="error",
        timestamp=datetime.now(),
        error=error_detail
    )


def parse_legacy_result(result: Union[str, Dict[str, Any]]) -> Dict[str, Any]:
    """
    Parse a legacy result (JSON string or dict) to a standardized dict.
    
    Args:
        result: Legacy result from graph manager
        
    Returns:
        Standardized dict with error handling
    """
    try:
        # Handle string result (legacy format)
        if isinstance(result, str):
            try:
                return json.loads(result)
            except json.JSONDecodeError:
                return {"error": f"Invalid JSON result: {result}"}
        
        # Return dict result as is
        return result
    except Exception as e:
        return {"error": f"Error parsing result: {str(e)}"}


def model_to_json(model: BaseModel) -> str:
    """
    Convert a Pydantic model to a JSON string.
    
    Args:
        model: Pydantic model instance
        
    Returns:
        JSON string representation
    """
    return model.model_dump_json()


def model_to_dict(model: BaseModel) -> Dict[str, Any]:
    """
    Convert a Pydantic model to a dictionary.
    
    Args:
        model: Pydantic model instance
        
    Returns:
        Dictionary representation
    """
    return model.model_dump()


def parse_json_to_model(json_str: str, model_class: Type[T]) -> T:
    """
    Parse a JSON string into a Pydantic model.
    
    Args:
        json_str: JSON string
        model_class: Pydantic model class
        
    Returns:
        Pydantic model instance
    """
    try:
        data = json.loads(json_str) if isinstance(json_str, str) else json_str
        return model_class.model_validate(data)
    except Exception as e:
        raise ValueError(f"Failed to parse JSON to {model_class.__name__}: {str(e)}")


def handle_lesson_response(
    result: Union[str, Dict[str, Any]],
    success_handler: Callable[[Dict[str, Any]], BaseModel],
    error_code: str = "lesson_error"
) -> str:
    """
    Handle a response from the lesson memory system with proper error handling.
    
    Args:
        result: Result from graph manager (JSON string or dict)
        success_handler: Function to call with parsed result on success
        error_code: Error code to use if result contains an error
        
    Returns:
        JSON string with standardized response
    """
    # Parse the result
    parsed_result = parse_legacy_result(result)
    
    # Check for error
    if "error" in parsed_result:
        error_response = create_lesson_error_response(
            message=parsed_result["error"],
            code=error_code
        )
        return model_to_json(error_response)
    
    # Handle success
    try:
        response = success_handler(parsed_result)
        return model_to_json(response)
    except Exception as e:
        error_response = create_lesson_error_response(
            message=f"Error processing result: {str(e)}",
            code="result_processing_error"
        )
        return model_to_json(error_response) 