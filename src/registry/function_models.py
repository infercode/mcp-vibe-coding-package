#!/usr/bin/env python3
"""
Function Registry Models

This module defines the data models used for function registry, including
metadata, parameters, and results.
"""

from typing import Any, Dict, List, Optional, Union, Callable, Awaitable
from pydantic import BaseModel, Field, validator
import inspect
import json


class FunctionParameters(BaseModel):
    """
    Parameters for a function in the registry.
    
    This model is used to validate and normalize parameters passed to functions.
    """
    function_name: str = Field(..., description="The name of the function to execute")
    params: Dict[str, Any] = Field(default_factory=dict, description="Parameters to pass to the function")
    
    class Config:
        arbitrary_types_allowed = True
        extra = "allow"


class FunctionResult(BaseModel):
    """
    Result of a function execution.
    
    This model standardizes the return format from all registered functions.
    """
    status: str = Field(..., description="Status of the execution (success/error)")
    message: str = Field(..., description="Human-readable message about the result")
    data: Optional[Dict[str, Any]] = Field(None, description="Result data if successful")
    error_code: Optional[str] = Field(None, description="Error code if failed")
    error_details: Optional[Dict[str, Any]] = Field(None, description="Detailed error information")
    
    @classmethod
    def success(cls, message: str, data: Optional[Dict[str, Any]] = None) -> 'FunctionResult':
        """Create a success result."""
        return cls(
            status="success",
            message=message,
            data=data or {}
        )
    
    @classmethod
    def error(cls, message: str, error_code: str = "unknown_error", 
              error_details: Optional[Dict[str, Any]] = None) -> 'FunctionResult':
        """Create an error result."""
        return cls(
            status="error",
            message=message,
            error_code=error_code,
            error_details=error_details or {}
        )
    
    def to_json(self) -> str:
        """Convert result to JSON string."""
        return json.dumps(self.dict(), indent=2)


class FunctionMetadata(BaseModel):
    """
    Metadata about a function in the registry.
    
    This model contains information about the function, its parameters,
    return type, and documentation.
    """
    name: str = Field(..., description="Full function name with namespace")
    short_name: str = Field(..., description="Short name without namespace")
    namespace: str = Field(..., description="Category/namespace the function belongs to")
    description: str = Field(..., description="Human-readable description of the function")
    parameters: Dict[str, Dict[str, Any]] = Field(
        default_factory=dict, 
        description="Parameters the function accepts"
    )
    return_type: str = Field(..., description="Return type of the function")
    is_async: bool = Field(False, description="Whether the function is asynchronous")
    examples: List[Dict[str, Any]] = Field(
        default_factory=list,
        description="Example usage of the function"
    )
    source_file: Optional[str] = Field(None, description="Source file where the function is defined")
    
    @validator('parameters')
    def validate_parameters(cls, v):
        """Ensure parameters are properly structured."""
        for param_name, param_info in v.items():
            # Ensure parameter info has required fields
            if 'type' not in param_info:
                param_info['type'] = 'any'
            if 'description' not in param_info:
                param_info['description'] = f"Parameter {param_name}"
            if 'required' not in param_info:
                param_info['required'] = False
        return v
    
    @classmethod
    def from_function(cls, namespace: str, name: str, func: Callable) -> 'FunctionMetadata':
        """
        Create function metadata by inspecting the function.
        
        Args:
            namespace: Category/namespace for the function
            name: Name of the function (without namespace)
            func: The function to inspect
        
        Returns:
            FunctionMetadata object
        """
        # Get function signature
        sig = inspect.signature(func)
        
        # Get docstring
        doc = inspect.getdoc(func) or "No documentation available"
        
        # Extract parameters
        parameters = {}
        for param_name, param in sig.parameters.items():
            if param_name == 'self' or param_name == 'cls':
                continue
                
            param_type = 'any'
            if param.annotation != inspect.Parameter.empty:
                if hasattr(param.annotation, '__name__'):
                    param_type = param.annotation.__name__
                else:
                    param_type = str(param.annotation)
                    
            has_default = param.default != inspect.Parameter.empty
            default_value = None if not has_default else param.default
            
            parameters[param_name] = {
                'type': param_type,
                'description': f"Parameter {param_name}",
                'required': not has_default,
                'default': default_value
            }
        
        # Get return type
        return_type = 'any'
        if sig.return_annotation != inspect.Signature.empty:
            if hasattr(sig.return_annotation, '__name__'):
                return_type = sig.return_annotation.__name__
            else:
                return_type = str(sig.return_annotation)
        
        # Check if async
        is_async = inspect.iscoroutinefunction(func)
        
        # Get source file
        try:
            source_file = inspect.getsourcefile(func)
        except:
            source_file = None
            
        # Create metadata
        return cls(
            name=f"{namespace}.{name}",
            short_name=name,
            namespace=namespace,
            description=doc.split("\n\n")[0] if doc else "No description",
            parameters=parameters,
            return_type=return_type,
            is_async=is_async,
            source_file=source_file
        )


class ValidationError(BaseModel):
    """
    Validation error for parameter validation.
    
    This model represents an error that occurred during parameter validation.
    """
    param_name: str = Field(..., description="Name of the parameter that failed validation")
    error_type: str = Field(..., description="Type of validation error")
    message: str = Field(..., description="Human-readable error message")


class ParameterInfo(BaseModel):
    """
    Information about a function parameter.
    
    This model contains metadata about a function parameter, including its type,
    description, and validation requirements.
    """
    type: str = Field(..., description="Data type of the parameter")
    description: str = Field(..., description="Human-readable description")
    required: bool = Field(False, description="Whether the parameter is required")
    default: Optional[Any] = Field(None, description="Default value if not provided")
    enum: Optional[List[Any]] = Field(None, description="List of allowed values")
    min_value: Optional[float] = Field(None, description="Minimum allowed value for numeric types")
    max_value: Optional[float] = Field(None, description="Maximum allowed value for numeric types")
    min_length: Optional[int] = Field(None, description="Minimum length for string/array types")
    max_length: Optional[int] = Field(None, description="Maximum length for string/array types")
    pattern: Optional[str] = Field(None, description="Regex pattern for string validation")
    format: Optional[str] = Field(None, description="Format specifier (e.g., 'date', 'email')")
    custom_validator: Optional[str] = Field(None, description="Name of custom validator function")
    
    class Config:
        arbitrary_types_allowed = True 