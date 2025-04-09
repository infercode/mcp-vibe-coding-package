#!/usr/bin/env python3
"""
Tool Registry Manager

This module provides the core functionality for registering and executing
tools through a unified registry.
"""

import json
import inspect
import logging
import traceback
import asyncio
from typing import Any, Dict, List, Optional, Union, Callable, Awaitable, Set, Tuple
from functools import wraps

from src.registry.function_models import ToolMetadata, FunctionResult, ToolParameters
from src.logger import get_logger

# Initialize logger
logger = get_logger()

# Type alias for registered tools
RegisteredTool = Callable[..., Union[Any, Awaitable[Any]]]


class ToolRegistry:
    """
    Tool Registry for managing and dispatching tool calls.
    
    This class maintains a registry of tools organized by namespace,
    provides dispatch capabilities, and handles parameter validation.
    """
    
    def __init__(self):
        """Initialize the tool registry."""
        # Main registry structure:
        # {namespace: {tool_name: {metadata: ToolMetadata, function: callable}}}
        self._registry: Dict[str, Dict[str, Dict[str, Any]]] = {}
        
        # Tool name lookup to resolve full names
        self._tool_lookup: Dict[str, Tuple[str, str]] = {}
        
        # Track registered namespaces
        self._namespaces: Set[str] = set()
        
        logger.info("Tool Registry initialized")
    
    def register(self, namespace: str, name: str, func: RegisteredTool) -> None:
        """
        Register a tool with the registry.
        
        Args:
            namespace: Category/namespace for the tool
            name: Name of the tool (without namespace)
            func: The function to register
            
        Returns:
            None
        """
        # Initialize namespace if it doesn't exist
        if namespace not in self._registry:
            self._registry[namespace] = {}
            self._namespaces.add(namespace)
        
        # Create full tool name
        full_name = f"{namespace}.{name}"
        
        # Create metadata
        metadata = ToolMetadata.from_function(namespace, name, func)
        
        # Store in registry
        self._registry[namespace][name] = {
            "metadata": metadata,
            "function": func
        }
        
        # Add to lookup
        self._tool_lookup[full_name] = (namespace, name)
        self._tool_lookup[name] = (namespace, name)  # Allow short names if unambiguous
        
        logger.debug(f"Registered tool: {full_name}")
    
    async def execute(self, tool_name: str, **kwargs) -> FunctionResult:
        """
        Execute a registered tool by name.
        
        Args:
            tool_name: Full or short name of the tool
            **kwargs: Parameters to pass to the tool
            
        Returns:
            FunctionResult with the execution result
        """
        try:
            # Resolve tool name
            if tool_name not in self._tool_lookup:
                return FunctionResult.error(
                    message=f"Tool '{tool_name}' not found in registry",
                    error_code="tool_not_found"
                )
            
            namespace, name = self._tool_lookup[tool_name]
            full_name = f"{namespace}.{name}"
            
            # Get function and metadata
            func_entry = self._registry[namespace][name]
            func = func_entry["function"]
            metadata = func_entry["metadata"]
            
            logger.debug(f"Executing tool: {full_name} with params: {kwargs}")
            
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
            logger.error(f"Error executing tool {tool_name}: {str(e)}\n{tb}")
            
            return FunctionResult.error(
                message=f"Error executing tool {tool_name}: {str(e)}",
                error_code="execution_error",
                error_details={"traceback": tb}
            )
    
    def get_tool_metadata(self, tool_name: str) -> Optional[ToolMetadata]:
        """
        Get metadata for a registered tool.
        
        Args:
            tool_name: Full or short name of the tool
            
        Returns:
            ToolMetadata or None if tool not found
        """
        if tool_name not in self._tool_lookup:
            return None
            
        namespace, name = self._tool_lookup[tool_name]
        return self._registry[namespace][name]["metadata"]
    
    def get_all_tools(self) -> List[ToolMetadata]:
        """
        Get metadata for all registered tools.
        
        Returns:
            List of ToolMetadata objects
        """
        all_tools = []
        for namespace, tools in self._registry.items():
            for name, entry in tools.items():
                all_tools.append(entry["metadata"])
        return all_tools
    
    def get_tools_by_namespace(self, namespace: str) -> List[ToolMetadata]:
        """
        Get metadata for all tools in a namespace.
        
        Args:
            namespace: Namespace to get tools for
            
        Returns:
            List of ToolMetadata objects or empty list if namespace not found
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
    
    # Add backward compatibility methods
    def get_function_metadata(self, function_name: str) -> Optional[ToolMetadata]:
        """Backward compatibility for get_tool_metadata."""
        return self.get_tool_metadata(function_name)
    
    def get_all_functions(self) -> List[ToolMetadata]:
        """Backward compatibility for get_all_tools."""
        return self.get_all_tools()
    
    def get_functions_by_namespace(self, namespace: str) -> List[ToolMetadata]:
        """Backward compatibility for get_tools_by_namespace."""
        return self.get_tools_by_namespace(namespace)


# Global registry instance
_registry = ToolRegistry()

def register_tool(namespace: str, name: Optional[str] = None):
    """
    Decorator for registering tools with the registry.
    
    Args:
        namespace: Category/namespace for the tool
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

# For backwards compatibility
register_function = register_tool

# Add another backward compatibility alias for the class
FunctionRegistry = ToolRegistry

# Export the global registry
get_registry = lambda: _registry 