#!/usr/bin/env python3
"""
Function Registry Manager

This module provides the core functionality for registering and executing
functions through a unified registry.
"""

import json
import inspect
import logging
import traceback
import asyncio
from typing import Any, Dict, List, Optional, Union, Callable, Awaitable, Set, Tuple
from functools import wraps

from src.registry.function_models import FunctionMetadata, FunctionResult, FunctionParameters
from src.logger import get_logger

# Initialize logger
logger = get_logger()

# Type alias for registered functions
RegisteredFunction = Callable[..., Union[Any, Awaitable[Any]]]


class FunctionRegistry:
    """
    Function Registry for managing and dispatching function calls.
    
    This class maintains a registry of functions organized by namespace,
    provides dispatch capabilities, and handles parameter validation.
    """
    
    def __init__(self):
        """Initialize the function registry."""
        # Main registry structure:
        # {namespace: {function_name: {metadata: FunctionMetadata, function: callable}}}
        self._registry: Dict[str, Dict[str, Dict[str, Any]]] = {}
        
        # Function name lookup to resolve full names
        self._function_lookup: Dict[str, Tuple[str, str]] = {}
        
        # Track registered namespaces
        self._namespaces: Set[str] = set()
        
        logger.info("Function Registry initialized")
    
    def register(self, namespace: str, name: str, func: RegisteredFunction) -> None:
        """
        Register a function with the registry.
        
        Args:
            namespace: Category/namespace for the function
            name: Name of the function (without namespace)
            func: The function to register
            
        Returns:
            None
        """
        # Initialize namespace if it doesn't exist
        if namespace not in self._registry:
            self._registry[namespace] = {}
            self._namespaces.add(namespace)
        
        # Create full function name
        full_name = f"{namespace}.{name}"
        
        # Create metadata
        metadata = FunctionMetadata.from_function(namespace, name, func)
        
        # Store in registry
        self._registry[namespace][name] = {
            "metadata": metadata,
            "function": func
        }
        
        # Add to lookup
        self._function_lookup[full_name] = (namespace, name)
        self._function_lookup[name] = (namespace, name)  # Allow short names if unambiguous
        
        logger.debug(f"Registered function: {full_name}")
    
    async def execute(self, function_name: str, **kwargs) -> FunctionResult:
        """
        Execute a registered function by name.
        
        Args:
            function_name: Full or short name of the function
            **kwargs: Parameters to pass to the function
            
        Returns:
            FunctionResult with the execution result
        """
        try:
            # Resolve function name
            if function_name not in self._function_lookup:
                return FunctionResult.error(
                    message=f"Function '{function_name}' not found in registry",
                    error_code="function_not_found"
                )
            
            namespace, name = self._function_lookup[function_name]
            full_name = f"{namespace}.{name}"
            
            # Get function and metadata
            func_entry = self._registry[namespace][name]
            func = func_entry["function"]
            metadata = func_entry["metadata"]
            
            logger.debug(f"Executing function: {full_name} with params: {kwargs}")
            
            # Call the function
            if metadata.is_async:
                result = await func(**kwargs)
            else:
                result = func(**kwargs)
                
            # Process the result
            if isinstance(result, FunctionResult):
                return result
            elif isinstance(result, dict):
                return FunctionResult.success(
                    message=f"Successfully executed {full_name}",
                    data=result
                )
            else:
                # Convert to dict if possible
                try:
                    if hasattr(result, "model_dump") and callable(result.model_dump):
                        result_dict = result.model_dump()
                    elif hasattr(result, "dict") and callable(result.dict):
                        # Fallback for older Pydantic models
                        result_dict = result.dict()
                    elif hasattr(result, "__dict__"):
                        result_dict = result.__dict__
                    else:
                        result_dict = {"result": result}
                        
                    return FunctionResult.success(
                        message=f"Successfully executed {full_name}",
                        data=result_dict
                    )
                except Exception as e:
                    logger.error(f"Could not convert result to dict: {str(e)}")
                    return FunctionResult.success(
                        message=f"Successfully executed {full_name}",
                        data={"result": str(result)}
                    )
                
        except Exception as e:
            # Capture traceback for debugging
            tb = traceback.format_exc()
            logger.error(f"Error executing function {function_name}: {str(e)}\n{tb}")
            
            return FunctionResult.error(
                message=f"Error executing function {function_name}: {str(e)}",
                error_code="execution_error",
                error_details={"traceback": tb}
            )
    
    def get_function_metadata(self, function_name: str) -> Optional[FunctionMetadata]:
        """
        Get metadata for a registered function.
        
        Args:
            function_name: Full or short name of the function
            
        Returns:
            FunctionMetadata or None if function not found
        """
        if function_name not in self._function_lookup:
            return None
            
        namespace, name = self._function_lookup[function_name]
        return self._registry[namespace][name]["metadata"]
    
    def get_all_functions(self) -> List[FunctionMetadata]:
        """
        Get metadata for all registered functions.
        
        Returns:
            List of FunctionMetadata objects
        """
        all_functions = []
        for namespace, functions in self._registry.items():
            for name, entry in functions.items():
                all_functions.append(entry["metadata"])
        return all_functions
    
    def get_functions_by_namespace(self, namespace: str) -> List[FunctionMetadata]:
        """
        Get metadata for all functions in a namespace.
        
        Args:
            namespace: Namespace to get functions for
            
        Returns:
            List of FunctionMetadata objects or empty list if namespace not found
        """
        if namespace not in self._registry:
            return []
            
        return [entry["metadata"] for _, entry in self._registry[namespace].items()]
    
    def get_namespaces(self) -> List[str]:
        """
        Get all registered namespaces.
        
        Returns:
            List of namespace strings
        """
        return list(self._namespaces)


# Global registry instance
_registry = FunctionRegistry()

def register_function(namespace: str, name: Optional[str] = None):
    """
    Decorator for registering functions with the registry.
    
    Args:
        namespace: Category/namespace for the function
        name: Optional name override (defaults to function name)
        
    Returns:
        Decorator function
    """
    def decorator(func):
        func_name = name or func.__name__
        _registry.register(namespace, func_name, func)
        
        @wraps(func)
        def wrapper(*args, **kwargs):
            return func(*args, **kwargs)
            
        return wrapper
        
    return decorator

# Export the global registry
get_registry = lambda: _registry 