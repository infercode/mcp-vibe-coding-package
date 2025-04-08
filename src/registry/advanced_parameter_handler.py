#!/usr/bin/env python3
"""
Advanced Parameter Handler

This module provides advanced parameter handling capabilities for the Function Registry,
including flexible parsing, context-aware defaults, and validation middleware.
"""

import json
import re
import copy
import inspect
from typing import Dict, List, Any, Optional, Union, Callable, Tuple

from src.registry.function_models import FunctionMetadata, ParameterInfo, ValidationError
from src.registry.parameter_helper import ParameterHelper
from src.logger import get_logger

logger = get_logger()

class AdvancedParameterHandler:
    """
    Advanced parameter handling capabilities for the Function Registry.
    
    Features:
    - Flexible parameter parsing from various input formats
    - Context-aware parameter defaults
    - Middleware pipeline for parameter processing
    - Advanced type conversion and validation
    """
    
    def __init__(self):
        """Initialize the advanced parameter handler."""
        self.middleware_pipeline = []
        self.context_providers = {}
        self.type_converters = {}
        
        # Register default middleware
        self.register_middleware(self._apply_context_defaults, priority=1)
        self.register_middleware(self._apply_parameter_aliases, priority=2)
        self.register_middleware(self._normalize_parameter_types, priority=3)
        
        # Register default context providers
        self.register_context_provider("user", lambda: {"timezone": "UTC"})
        self.register_context_provider("system", lambda: {"timestamp": __import__("datetime").datetime.now().isoformat()})
        
        # Register custom type converters
        self.register_type_converter("date", self._convert_to_date)
        self.register_type_converter("datetime", self._convert_to_datetime)
        self.register_type_converter("duration", self._convert_to_duration)
        
    def register_middleware(self, middleware_func: Callable, priority: int = 10) -> None:
        """
        Register a middleware function for parameter processing.
        
        Middleware functions take (function_metadata, parameters, context) and return
        updated parameters. They're executed in order of priority (lower first).
        
        Args:
            middleware_func: The middleware function to register
            priority: Execution priority (lower values run first)
        """
        self.middleware_pipeline.append({
            "function": middleware_func,
            "priority": priority
        })
        
        # Sort middleware by priority
        self.middleware_pipeline.sort(key=lambda m: m["priority"])
        
    def register_context_provider(self, namespace: str, provider_func: Callable[[], Dict[str, Any]]) -> None:
        """
        Register a context provider function.
        
        Context providers supply context data for parameter processing, such as user
        preferences, system information, or session state.
        
        Args:
            namespace: The provider namespace
            provider_func: Function that returns context data
        """
        self.context_providers[namespace] = provider_func
        
    def register_type_converter(self, type_name: str, converter_func: Callable[[Any], Any]) -> None:
        """
        Register a custom type converter.
        
        Type converters allow handling custom types beyond basic Python types.
        
        Args:
            type_name: The name of the custom type
            converter_func: Function that converts input to the correct type
        """
        self.type_converters[type_name] = converter_func
        
    def parse_parameters(self, 
                         function_metadata: FunctionMetadata, 
                         raw_input: Union[Dict[str, Any], str, List[Any]],
                         context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Parse parameters from various input formats.
        
        This method can handle parameters provided as a dictionary, a JSON string,
        a natural language string, or a list of positional arguments.
        
        Args:
            function_metadata: Metadata for the target function
            raw_input: The raw parameter input
            context: Optional context data
            
        Returns:
            Dictionary of parsed parameters
        """
        context = context or {}
        
        # Handle different input formats
        if isinstance(raw_input, dict):
            # Already a dictionary
            parameters = copy.deepcopy(raw_input)
        elif isinstance(raw_input, str):
            # Try to parse as JSON
            try:
                parameters = json.loads(raw_input)
                if not isinstance(parameters, dict):
                    # Convert JSON array to positional arguments
                    if isinstance(parameters, list):
                        parameters = self._convert_positional_to_named(function_metadata, parameters)
                    else:
                        # Handle natural language input
                        parameters = self._extract_parameters_from_text(function_metadata, raw_input)
            except json.JSONDecodeError:
                # Handle natural language input
                parameters = self._extract_parameters_from_text(function_metadata, raw_input)
        elif isinstance(raw_input, list):
            # Convert positional arguments to named parameters
            parameters = self._convert_positional_to_named(function_metadata, raw_input)
        else:
            raise ValueError(f"Unsupported parameter format: {type(raw_input)}")
            
        # Apply middleware to transform parameters
        return self.apply_middleware(function_metadata, parameters, context)
        
    def apply_middleware(self, 
                         function_metadata: FunctionMetadata, 
                         parameters: Dict[str, Any],
                         context: Dict[str, Any]) -> Dict[str, Any]:
        """
        Apply the middleware pipeline to parameters.
        
        Args:
            function_metadata: Metadata for the function
            parameters: Initial parameters
            context: Context data
            
        Returns:
            Processed parameters
        """
        # Gather all context data
        full_context = self._gather_context(context)
        
        # Create a copy of parameters
        processed_params = copy.deepcopy(parameters)
        
        # Apply each middleware in the pipeline
        for middleware in self.middleware_pipeline:
            try:
                processed_params = middleware["function"](
                    function_metadata, processed_params, full_context
                )
            except Exception as e:
                logger.error(f"Error in middleware {middleware['function'].__name__}: {str(e)}")
                # Continue with next middleware
        
        return processed_params
    
    def validate_and_convert(self, 
                            function_metadata: FunctionMetadata, 
                            parameters: Dict[str, Any]) -> Tuple[Dict[str, Any], List[ValidationError]]:
        """
        Validate and convert parameters with enhanced functionality.
        
        This extends the basic validation from ParameterHelper with more
        advanced type conversion and validation.
        
        Args:
            function_metadata: Function metadata
            parameters: Parameters to validate and convert
            
        Returns:
            Tuple of (converted parameters, validation errors)
        """
        # First do basic validation
        validation_errors = ParameterHelper.validate_parameters(function_metadata, parameters)
        
        # If there are errors, return them
        if validation_errors:
            return parameters, validation_errors
        
        # Apply advanced conversion
        converted_params = {}
        new_errors = []
        
        for name, value in parameters.items():
            if name in function_metadata.parameters:
                param_info = function_metadata.parameters[name]
                
                try:
                    # Use custom converter if available for this type
                    if param_info.type in self.type_converters:
                        converted_params[name] = self.type_converters[param_info.type](value)
                    else:
                        # Fall back to basic conversion
                        converted_params[name] = self._convert_parameter_value(param_info, value)
                except Exception as e:
                    new_errors.append(ValidationError(
                        param_name=name,
                        error_type="conversion_error",
                        message=str(e)
                    ))
            else:
                # Pass through parameters not in metadata
                converted_params[name] = value
        
        return converted_params, new_errors
        
    def _gather_context(self, user_context: Dict[str, Any]) -> Dict[str, Any]:
        """Gather context data from all registered providers."""
        context = copy.deepcopy(user_context)
        
        # Add data from providers
        for namespace, provider_func in self.context_providers.items():
            try:
                context[namespace] = provider_func()
            except Exception as e:
                logger.error(f"Error in context provider {namespace}: {str(e)}")
                context[namespace] = {}
                
        return context
    
    def _convert_positional_to_named(self, 
                                    function_metadata: FunctionMetadata, 
                                    positional_args: List[Any]) -> Dict[str, Any]:
        """Convert positional arguments to named parameters."""
        parameters = {}
        
        # Get ordered list of parameter names (required first, then optional)
        param_names = [
            name for name, info in function_metadata.parameters.items()
            if info.required
        ] + [
            name for name, info in function_metadata.parameters.items()
            if not info.required
        ]
        
        # Map positional args to parameter names
        for i, value in enumerate(positional_args):
            if i < len(param_names):
                parameters[param_names[i]] = value
            else:
                break
                
        return parameters
    
    def _extract_parameters_from_text(self, 
                                     function_metadata: FunctionMetadata, 
                                     text: str) -> Dict[str, Any]:
        """
        Extract parameters from natural language text.
        
        This method looks for parameter mentions in text using patterns like:
        - "param_name: value"
        - "param_name = value"
        - "with param_name value"
        - "param_name as value"
        """
        parameters = {}
        
        # Create pattern for each parameter
        for param_name in function_metadata.parameters:
            # Define patterns to look for
            patterns = [
                rf'{param_name}\s*[:=]\s*([^,;]+)',  # param: value or param = value
                rf'{param_name}\s+(?:as|is|of|with)\s+([^,;]+)',  # param as/is/with value
                rf'(?:set|use|with)\s+{param_name}\s+(?:to|as|of)?\s+([^,;]+)',  # use param as value
                rf'["\']?{param_name}["\']?\s*[:=]\s*["\']?([^"\']+)["\']?',  # "param": "value"
            ]
            
            # Try each pattern
            for pattern in patterns:
                matches = re.search(pattern, text, re.IGNORECASE)
                if matches:
                    value = matches.group(1).strip()
                    
                    # Try to convert value to appropriate type
                    param_info = function_metadata.parameters[param_name]
                    parameters[param_name] = self._infer_parameter_type(param_info, value)
                    break
        
        return parameters
    
    def _infer_parameter_type(self, param_info: Dict[str, Any], value: str) -> Any:
        """Infer the type of a parameter from a string value."""
        param_type = param_info.get('type', 'str')
        
        if param_type == 'int':
            # Try to convert to int
            if value.isdigit() or (value.startswith('-') and value[1:].isdigit()):
                return int(value)
        elif param_type == 'float':
            # Try to convert to float
            try:
                return float(value)
            except ValueError:
                pass
        elif param_type == 'bool':
            # Check for boolean values
            if value.lower() in ('true', 'yes', 'y', '1'):
                return True
            elif value.lower() in ('false', 'no', 'n', '0'):
                return False
        elif param_type == 'list':
            # Try to parse as list
            if value.startswith('[') and value.endswith(']'):
                try:
                    return json.loads(value)
                except json.JSONDecodeError:
                    # Fall back to simple splitting
                    return [item.strip() for item in value[1:-1].split(',')]
            else:
                # Simple comma-separated list
                return [item.strip() for item in value.split(',')]
        elif param_type == 'dict':
            # Try to parse as dictionary
            if value.startswith('{') and value.endswith('}'):
                try:
                    return json.loads(value)
                except json.JSONDecodeError:
                    pass
        
        # Default to string
        return value
    
    def _convert_parameter_value(self, param_info: Dict[str, Any], value: Any) -> Any:
        """Convert a parameter value to the expected type."""
        param_type = param_info.get('type', 'str')
        
        if value is None:
            return None
            
        if param_type == 'str':
            return str(value)
        elif param_type == 'int':
            return int(value)
        elif param_type == 'float':
            return float(value)
        elif param_type == 'bool':
            if isinstance(value, str):
                return value.lower() in ('true', 'yes', 'y', '1')
            return bool(value)
        elif param_type == 'list':
            if isinstance(value, str):
                try:
                    return json.loads(value)
                except json.JSONDecodeError:
                    return [item.strip() for item in value.split(',')]
            return list(value)
        elif param_type == 'dict':
            if isinstance(value, str):
                try:
                    return json.loads(value)
                except json.JSONDecodeError:
                    return {}
            return dict(value)
        
        return value
    
    # Default middleware functions
    
    def _apply_context_defaults(self, 
                               function_metadata: FunctionMetadata, 
                               parameters: Dict[str, Any],
                               context: Dict[str, Any]) -> Dict[str, Any]:
        """
        Apply context-aware defaults to parameters.
        
        This middleware fills in missing parameters using context data.
        """
        result = copy.deepcopy(parameters)
        
        for param_name, param_info in function_metadata.parameters.items():
            # Skip if parameter is already provided
            if param_name in result:
                continue
                
            # Check for context mapping
            context_path = param_info.get('context_path')
            if context_path:
                # Parse the context path (e.g., "user.preferences.theme")
                parts = context_path.split('.')
                
                # Navigate the context object
                value = context
                for part in parts:
                    if isinstance(value, dict) and part in value:
                        value = value[part]
                    else:
                        value = None
                        break
                
                # If we found a value, use it
                if value is not None:
                    result[param_name] = value
        
        return result
    
    def _apply_parameter_aliases(self, 
                               function_metadata: FunctionMetadata, 
                               parameters: Dict[str, Any],
                               context: Dict[str, Any]) -> Dict[str, Any]:
        """
        Apply parameter aliases.
        
        This middleware handles alternate parameter names.
        """
        result = copy.deepcopy(parameters)
        
        # Check each parameter for aliases
        for param_name, param_info in function_metadata.parameters.items():
            # Skip if parameter is already provided
            if param_name in result:
                continue
                
            # Check for aliases
            aliases = param_info.get('aliases', [])
            for alias in aliases:
                if alias in result:
                    # Found a match, copy value and remove alias
                    result[param_name] = result[alias]
                    del result[alias]
                    break
        
        return result
    
    def _normalize_parameter_types(self, 
                                 function_metadata: FunctionMetadata, 
                                 parameters: Dict[str, Any],
                                 context: Dict[str, Any]) -> Dict[str, Any]:
        """
        Normalize parameter types.
        
        This middleware attempts basic type conversion.
        """
        result = copy.deepcopy(parameters)
        
        for param_name, value in result.items():
            if param_name in function_metadata.parameters:
                param_info = function_metadata.parameters[param_name]
                
                # Skip None values
                if value is None:
                    continue
                    
                param_type = param_info.get('type', 'str')
                
                # Perform basic type conversion
                if param_type == 'str' and not isinstance(value, str):
                    result[param_name] = str(value)
                elif param_type == 'int' and not isinstance(value, int):
                    try:
                        if isinstance(value, str) and value.strip():
                            result[param_name] = int(float(value))
                    except ValueError:
                        pass
                elif param_type == 'float' and not isinstance(value, float):
                    try:
                        if isinstance(value, str) and value.strip():
                            result[param_name] = float(value)
                    except ValueError:
                        pass
                elif param_type == 'bool' and not isinstance(value, bool):
                    if isinstance(value, str):
                        result[param_name] = value.lower() in ('true', 'yes', 'y', '1')
        
        return result
    
    # Custom type converters
    
    def _convert_to_date(self, value: Any) -> Any:
        """Convert value to a date object."""
        if value is None:
            return None
            
        import datetime
        
        if isinstance(value, datetime.date):
            return value
            
        if isinstance(value, datetime.datetime):
            return value.date()
            
        if isinstance(value, str):
            # Try different date formats
            formats = [
                "%Y-%m-%d",
                "%m/%d/%Y",
                "%d/%m/%Y",
                "%B %d, %Y",
                "%d %B %Y"
            ]
            
            for fmt in formats:
                try:
                    return datetime.datetime.strptime(value, fmt).date()
                except ValueError:
                    continue
            
            # If explicit formats fail, try dateutil
            try:
                from dateutil import parser
                return parser.parse(value).date()
            except:
                pass
        
        raise ValueError(f"Cannot convert {value} to date")
    
    def _convert_to_datetime(self, value: Any) -> Any:
        """Convert value to a datetime object."""
        if value is None:
            return None
            
        import datetime
        
        if isinstance(value, datetime.datetime):
            return value
            
        if isinstance(value, datetime.date):
            return datetime.datetime.combine(value, datetime.time.min)
            
        if isinstance(value, str):
            # Try different datetime formats
            formats = [
                "%Y-%m-%dT%H:%M:%S",
                "%Y-%m-%d %H:%M:%S",
                "%m/%d/%Y %H:%M:%S",
                "%d/%m/%Y %H:%M:%S"
            ]
            
            for fmt in formats:
                try:
                    return datetime.datetime.strptime(value, fmt)
                except ValueError:
                    continue
            
            # If explicit formats fail, try dateutil
            try:
                from dateutil import parser
                return parser.parse(value)
            except:
                pass
        
        raise ValueError(f"Cannot convert {value} to datetime")
    
    def _convert_to_duration(self, value: Any) -> Any:
        """Convert value to a duration (in seconds)."""
        if value is None:
            return None
            
        if isinstance(value, (int, float)):
            return value
            
        if isinstance(value, str):
            # Check for format like "1h 30m 15s"
            total_seconds = 0
            
            # Extract hours, minutes, seconds
            hours_match = re.search(r'(\d+)\s*h', value, re.IGNORECASE)
            if hours_match:
                total_seconds += int(hours_match.group(1)) * 3600
                
            minutes_match = re.search(r'(\d+)\s*m', value, re.IGNORECASE)
            if minutes_match:
                total_seconds += int(minutes_match.group(1)) * 60
                
            seconds_match = re.search(r'(\d+)\s*s', value, re.IGNORECASE)
            if seconds_match:
                total_seconds += int(seconds_match.group(1))
                
            if total_seconds > 0:
                return total_seconds
                
            # Try as simple number of seconds
            try:
                return float(value)
            except ValueError:
                pass
        
        raise ValueError(f"Cannot convert {value} to duration")


# Create a singleton instance
_handler = None

def get_handler() -> AdvancedParameterHandler:
    """Get the global parameter handler instance."""
    global _handler
    if _handler is None:
        _handler = AdvancedParameterHandler()
    return _handler


# Convenience functions
def parse_parameters(function_metadata: FunctionMetadata, 
                    raw_input: Union[Dict[str, Any], str, List[Any]],
                    context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """
    Parse parameters from various input formats.
    
    Args:
        function_metadata: Metadata for the target function
        raw_input: The raw parameter input
        context: Optional context data
        
    Returns:
        Dictionary of parsed parameters
    """
    return get_handler().parse_parameters(function_metadata, raw_input, context)


def validate_and_convert(function_metadata: FunctionMetadata, 
                       parameters: Dict[str, Any]) -> Tuple[Dict[str, Any], List[ValidationError]]:
    """
    Validate and convert parameters.
    
    Args:
        function_metadata: Function metadata
        parameters: Parameters to validate and convert
        
    Returns:
        Tuple of (converted parameters, validation errors)
    """
    return get_handler().validate_and_convert(function_metadata, parameters)


def register_middleware(middleware_func: Callable, priority: int = 10) -> None:
    """
    Register a middleware function for parameter processing.
    
    Args:
        middleware_func: The middleware function to register
        priority: Execution priority (lower values run first)
    """
    get_handler().register_middleware(middleware_func, priority)


def register_context_provider(namespace: str, provider_func: Callable[[], Dict[str, Any]]) -> None:
    """
    Register a context provider function.
    
    Args:
        namespace: The provider namespace
        provider_func: Function that returns context data
    """
    get_handler().register_context_provider(namespace, provider_func)


def register_type_converter(type_name: str, converter_func: Callable[[Any], Any]) -> None:
    """
    Register a custom type converter.
    
    Args:
        type_name: The name of the custom type
        converter_func: Function that converts input to the correct type
    """
    get_handler().register_type_converter(type_name, converter_func) 