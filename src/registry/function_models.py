#!/usr/bin/env python3
"""
Tool Registry Models

This module defines the data models used for tool registry, including
metadata, parameters, and results.
"""

from typing import Any, Dict, List, Optional, Union, Callable, Awaitable, Annotated
from pydantic import BaseModel, Field, field_validator, ConfigDict
import inspect
import json


class ToolParameters(BaseModel):
    """
    Parameters for a tool in the registry.
    
    This model is used to validate and normalize parameters passed to tools.
    """
    tool_name: Annotated[str, Field(description="The name of the tool to execute")]
    params: Annotated[Dict[str, Any], Field(default_factory=dict, description="Parameters to pass to the tool")]
    
    model_config = ConfigDict(
        arbitrary_types_allowed=True,
        extra="allow"
    )


class FunctionResult(BaseModel):
    """
    Result of a tool execution.
    
    This model standardizes the return format from all registered tools.
    """
    status: Annotated[str, Field(description="Status of the execution (success/error)")]
    message: Annotated[str, Field(description="Human-readable message about the result")]
    data: Annotated[Optional[Dict[str, Any]], Field(None, description="Result data if successful")]
    error_code: Annotated[Optional[str], Field(None, description="Error code if failed")]
    error_details: Annotated[Optional[Dict[str, Any]], Field(None, description="Detailed error information")]
    
    @classmethod
    def success(cls, message: str, data: Optional[Dict[str, Any]] = None) -> 'FunctionResult':
        """Create a success result."""
        return cls(
            status="success",
            message=message,
            data=data or {},
            error_code=None,
            error_details=None
        )
    
    @classmethod
    def error(cls, message: str, error_code: str = "unknown_error", 
              error_details: Optional[Dict[str, Any]] = None) -> 'FunctionResult':
        """Create an error result."""
        return cls(
            status="error",
            message=message,
            error_code=error_code,
            error_details=error_details or {},
            data=None
        )
    
    def to_json(self) -> str:
        """Convert to JSON string."""
        return json.dumps(self.model_dump(), indent=2)


class ToolMetadata(BaseModel):
    """
    Metadata about a tool in the registry.
    
    This model contains information about the tool, its parameters,
    return type, and documentation.
    """
    name: Annotated[str, Field(description="Full tool name with namespace")]
    short_name: Annotated[str, Field(description="Short name without namespace")]
    namespace: Annotated[str, Field(description="Category/namespace the tool belongs to")]
    description: Annotated[str, Field(description="Human-readable description of the tool")]
    parameters: Annotated[Dict[str, Dict[str, Any]], Field(
        default_factory=dict, 
        description="Parameters the tool accepts"
    )]
    return_type: Annotated[str, Field(description="Return type of the tool")]
    is_async: Annotated[bool, Field(default=False, description="Whether the tool is asynchronous")]
    examples: Annotated[List[Dict[str, Any]], Field(
        default_factory=list,
        description="Example usage of the tool"
    )]
    source_file: Annotated[Optional[str], Field(None, description="Source file where the tool is defined")]
    
    @field_validator('parameters')
    @classmethod
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
    def from_function(cls, namespace: str, name: str, func: Callable) -> 'ToolMetadata':
        """
        Create tool metadata by inspecting the function.
        
        Args:
            namespace: Category/namespace for the tool
            name: Name of the tool (without namespace)
            func: The function to inspect
        
        Returns:
            ToolMetadata object
        """
        try:
            # Import our enhanced docstring parser
            from src.registry.docstring_parser import parse_docstring
            enhanced_docstring_parser_available = True
        except ImportError:
            enhanced_docstring_parser_available = False
        
        # Get function signature
        sig = inspect.signature(func)
        
        # Get docstring
        doc = inspect.getdoc(func) or "No documentation available"
        
        # Parse docstring with enhanced parser if available
        docstring_data = {}
        if enhanced_docstring_parser_available:
            try:
                from src.registry.docstring_parser import parse_docstring as parse_ds
                docstring_data = parse_ds(doc)
            except Exception as e:
                # Fall back to basic parsing if enhanced parser fails
                docstring_data = {}
        
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
            
            # Create basic parameter info
            parameters[param_name] = {
                'type': param_type,
                'description': f"Parameter {param_name}",
                'required': not has_default,
                'default': default_value
            }
            
            # Enhance with docstring data if available
            if enhanced_docstring_parser_available and 'parameters' in docstring_data and param_name in docstring_data['parameters']:
                param_data = docstring_data['parameters'][param_name]
                
                # Use description from docstring
                if param_data.get('description'):
                    parameters[param_name]['description'] = param_data['description']
                
                # Add nested field information for Dict parameters
                if param_type in ('Dict', 'dict', 'Dictionary', 'dictionary') and param_data.get('nested_fields'):
                    parameters[param_name]['nested_fields'] = param_data['nested_fields']
                    
                    # Create a more descriptive parameter description that includes field information
                    if param_data['nested_fields']:
                        field_descriptions = []
                        for field_name, field_info in param_data['nested_fields'].items():
                            required_text = "Required" if field_info.get('required', False) else "Optional"
                            field_descriptions.append(f"- {field_name}: {required_text}. {field_info['description']}")
                            
                        if field_descriptions:
                            full_description = parameters[param_name]['description']
                            full_description += "\nFields:\n" + "\n".join(field_descriptions)
                            parameters[param_name]['description'] = full_description
        
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
            
        # Extract examples from docstring
        examples = []
        if enhanced_docstring_parser_available and 'examples' in docstring_data:
            examples = docstring_data['examples']
            
        # Use the first paragraph of docstring as description, or the docstring_data description if available
        description = doc.split("\n\n")[0] if doc else "No description"
        if enhanced_docstring_parser_available and 'description' in docstring_data and docstring_data['description']:
            description = docstring_data['description']
            
        # Create metadata
        return cls(
            name=f"{namespace}.{name}",
            short_name=name,
            namespace=namespace,
            description=description,
            parameters=parameters,
            return_type=return_type,
            is_async=is_async,
            examples=examples,
            source_file=source_file
        )


class ValidationError(BaseModel):
    """
    Validation error for parameter validation.
    
    This model represents an error that occurred during parameter validation.
    """
    param_name: Annotated[str, Field(description="Name of the parameter that failed validation")]
    error_type: Annotated[str, Field(description="Type of validation error")]
    message: Annotated[str, Field(description="Human-readable error message")]
    
    model_config = ConfigDict(
        validate_assignment=True
    )


class ParameterInfo(BaseModel):
    """
    Information about a tool parameter.
    
    This model contains metadata about a tool parameter, including its type,
    description, and validation requirements.
    """
    type: Annotated[str, Field(description="Data type of the parameter")]
    description: Annotated[str, Field(description="Human-readable description")]
    required: Annotated[bool, Field(default=False, description="Whether the parameter is required")]
    default: Annotated[Optional[Any], Field(None, description="Default value if not provided")]
    enum: Annotated[Optional[List[Any]], Field(None, description="List of allowed values")]
    min_value: Annotated[Optional[float], Field(None, description="Minimum allowed value for numeric types")]
    max_value: Annotated[Optional[float], Field(None, description="Maximum allowed value for numeric types")]
    min_length: Annotated[Optional[int], Field(None, description="Minimum length for string/array types")]
    max_length: Annotated[Optional[int], Field(None, description="Maximum length for string/array types")]
    pattern: Annotated[Optional[str], Field(None, description="Regex pattern for string validation")]
    format: Annotated[Optional[str], Field(None, description="Format specifier (e.g., 'date', 'email')")]
    custom_validator: Annotated[Optional[str], Field(None, description="Name of custom validator function")]
    
    model_config = ConfigDict(
        arbitrary_types_allowed=True
    )


# Add backward compatibility aliases at the end of the file
FunctionMetadata = ToolMetadata
FunctionParameters = ToolParameters
FunctionInfo = ToolMetadata  # For performance_demo.py which imports this
# The registry will continue to work since we kept the FunctionResult name 