#!/usr/bin/env python3
"""
Parameter Helper System

This module provides utilities for parameter validation, type conversion,
and suggestion generation to help AI agents use functions correctly.
"""

import inspect
import json
import re
import datetime
from typing import Any, Dict, List, Optional, Union, Callable, TypeVar, Type, cast, get_type_hints
from enum import Enum

from src.registry.function_models import FunctionMetadata, FunctionResult
from src.logger import get_logger

# Initialize logger
logger = get_logger()

# Type conversion mapping
TYPE_CONVERTERS = {
    "str": str,
    "int": lambda v: int(float(v)) if isinstance(v, str) and v.strip() else int(v),
    "float": float,
    "bool": lambda v: str(v).lower() in ("true", "t", "yes", "y", "1") if isinstance(v, str) else bool(v),
    "list": lambda v: json.loads(v) if isinstance(v, str) else list(v),
    "dict": lambda v: json.loads(v) if isinstance(v, str) else dict(v),
    "datetime": lambda v: datetime.datetime.fromisoformat(v) if isinstance(v, str) else v
}

class ValidationError:
    """Represents a validation error for a parameter."""
    
    def __init__(self, param_name: str, error_message: str, error_code: str = "validation_error"):
        self.param_name = param_name
        self.error_message = error_message
        self.error_code = error_code
    
    def to_dict(self) -> Dict[str, str]:
        """Convert to dictionary representation."""
        return {
            "param_name": self.param_name,
            "error_message": self.error_message,
            "error_code": self.error_code
        }

class ParameterHelper:
    """
    Helper class for working with function parameters.
    
    This class provides utilities for validating, converting, and suggesting
    parameters for functions in the registry.
    """
    
    @staticmethod
    def validate_parameters(metadata: FunctionMetadata, params: Dict[str, Any]) -> List[ValidationError]:
        """
        Validate parameters against function metadata.
        
        Args:
            metadata: Function metadata containing parameter definitions
            params: Parameters to validate
            
        Returns:
            List of validation errors, empty if all parameters are valid
        """
        errors = []
        
        # Check for required parameters
        for param_name, param_info in metadata.parameters.items():
            if param_info.get("required", False) and param_name not in params:
                errors.append(ValidationError(
                    param_name=param_name,
                    error_message=f"Required parameter '{param_name}' is missing",
                    error_code="missing_required_parameter"
                ))
        
        # Validate parameter types
        for param_name, param_value in params.items():
            if param_name not in metadata.parameters:
                # Skip validation for unknown parameters
                continue
                
            param_info = metadata.parameters[param_name]
            param_type = param_info.get("type", "any")
            
            # Skip validation for 'any' type or None values for optional parameters
            if param_type == "any" or (param_value is None and not param_info.get("required", False)):
                continue
                
            # Attempt type validation
            try:
                ParameterHelper.convert_parameter_value(param_value, param_type)
            except Exception as e:
                errors.append(ValidationError(
                    param_name=param_name,
                    error_message=f"Invalid type for parameter '{param_name}': expected {param_type}, got {type(param_value).__name__}. Error: {str(e)}",
                    error_code="invalid_parameter_type"
                ))
        
        return errors
    
    @staticmethod
    def convert_parameters(metadata: FunctionMetadata, params: Dict[str, Any]) -> Dict[str, Any]:
        """
        Convert parameters to the correct types based on function metadata.
        
        Args:
            metadata: Function metadata containing parameter definitions
            params: Parameters to convert
            
        Returns:
            Converted parameters
        """
        converted_params = {}
        
        for param_name, param_value in params.items():
            if param_name not in metadata.parameters:
                # Pass through unknown parameters
                converted_params[param_name] = param_value
                continue
                
            param_info = metadata.parameters[param_name]
            param_type = param_info.get("type", "any")
            
            # Skip conversion for 'any' type or None values
            if param_type == "any" or param_value is None:
                converted_params[param_name] = param_value
                continue
                
            # Convert the parameter
            try:
                converted_params[param_name] = ParameterHelper.convert_parameter_value(param_value, param_type)
            except Exception as e:
                logger.warn(f"Could not convert parameter '{param_name}' to type {param_type}: {str(e)}")
                # Use original value if conversion fails
                converted_params[param_name] = param_value
        
        return converted_params
    
    @staticmethod
    def convert_parameter_value(value: Any, target_type: str) -> Any:
        """
        Convert a parameter value to the target type.
        
        Args:
            value: Value to convert
            target_type: Target type name
            
        Returns:
            Converted value
            
        Raises:
            ValueError: If conversion fails
        """
        # Handle None values
        if value is None:
            return None
            
        # Handle Union types (e.g., "Union[str, int]")
        if "Union[" in target_type:
            # Try each type in the union
            union_types = re.findall(r'[a-zA-Z0-9_]+', target_type)
            for utype in union_types:
                if utype in ("Union", "Optional"):
                    continue
                try:
                    return ParameterHelper.convert_parameter_value(value, utype)
                except:
                    continue
            # If all conversions fail, raise error
            raise ValueError(f"Could not convert to any type in union: {target_type}")
            
        # Handle Optional types (e.g., "Optional[str]")
        if "Optional[" in target_type:
            inner_type = re.search(r'Optional\[(.*)\]', target_type)
            if inner_type:
                return ParameterHelper.convert_parameter_value(value, inner_type.group(1))
                
        # Handle List types (e.g., "List[str]")
        if "List[" in target_type or "list[" in target_type:
            if not isinstance(value, list):
                if isinstance(value, str):
                    try:
                        value = json.loads(value)
                        if not isinstance(value, list):
                            value = [value]
                    except:
                        value = [value]
                else:
                    value = [value]
            
            # If we have a typed list, convert each element
            inner_type_match = re.search(r'[Ll]ist\[(.*)\]', target_type)
            if inner_type_match:
                inner_type = inner_type_match.group(1)
                return [ParameterHelper.convert_parameter_value(item, inner_type) for item in value]
            return value
            
        # Handle Dict types (e.g., "Dict[str, Any]")
        if "Dict[" in target_type or "dict[" in target_type:
            if not isinstance(value, dict):
                if isinstance(value, str):
                    try:
                        value = json.loads(value)
                    except:
                        raise ValueError(f"Could not convert string to dict: {value}")
                else:
                    raise ValueError(f"Could not convert to dict: {value}")
            return value
            
        # Use type converters for basic types
        if target_type in TYPE_CONVERTERS:
            return TYPE_CONVERTERS[target_type](value)
            
        # Default to returning the original value if we can't convert
        return value
    
    @staticmethod
    def generate_parameter_suggestions(metadata: FunctionMetadata) -> Dict[str, Any]:
        """
        Generate parameter suggestions for a function.
        
        Args:
            metadata: Function metadata
            
        Returns:
            Dictionary with parameter suggestions
        """
        suggestions = {}
        
        for param_name, param_info in metadata.parameters.items():
            param_type = param_info.get("type", "any")
            
            # Generate suggestion based on type
            if "default" in param_info and param_info["default"] is not None:
                # Use default value as suggestion
                suggestions[param_name] = param_info["default"]
            elif param_type == "str":
                suggestions[param_name] = f"<{param_name}>"
            elif param_type == "int":
                suggestions[param_name] = 0
            elif param_type == "float":
                suggestions[param_name] = 0.0
            elif param_type == "bool":
                suggestions[param_name] = False
            elif "List" in param_type or "list" in param_type:
                suggestions[param_name] = []
            elif "Dict" in param_type or "dict" in param_type:
                suggestions[param_name] = {}
            else:
                suggestions[param_name] = None
                
        return suggestions 