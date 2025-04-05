"""
Response utilities for MCP Graph Memory

This module provides standardized response formatting using Pydantic models.
"""

import json
from typing import Any, Dict, Optional, TypeVar, Generic, Type
from datetime import datetime

from pydantic import BaseModel

from src.models.project_memory import ErrorDetail, ErrorResponse, SuccessResponse, BaseResponse

T = TypeVar('T', bound=BaseModel)


def create_error_response(
    message: str, 
    code: str = "internal_error", 
    details: Optional[Dict[str, Any]] = None
) -> ErrorResponse:
    """
    Create a standardized error response using Pydantic models.
    
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


def create_success_response(
    message: Optional[str] = None,
    data: Optional[Dict[str, Any]] = None
) -> SuccessResponse:
    """
    Create a standardized success response using Pydantic models.
    
    Args:
        message: Optional success message
        data: Optional additional data
        
    Returns:
        SuccessResponse model instance
    """
    response = SuccessResponse(
        status="success",
        timestamp=datetime.now(),
        message=message
    )
    
    # Add any additional data as attributes
    if data:
        for key, value in data.items():
            setattr(response, key, value)
    
    return response


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