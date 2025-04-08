#!/usr/bin/env python3
"""
Core Registry Tools

This module implements the minimal set of essential registry tools needed
for the Function Registry Pattern. It provides only the core tools while
allowing access to all other functions through the registry.
"""

import json
from typing import Dict, List, Any, Optional, Callable, Union

from src.registry.registry_manager import get_registry
from src.registry.function_models import FunctionResult
from src.logger import get_logger
from src.registry.parameter_helper import ParameterHelper

# Initialize logger
logger = get_logger()

class CoreRegistryToolsManager:
    """
    Manager for the minimal set of essential registry tools.
    
    This class provides utilities for registering only the core registry
    tools with an MCP server, allowing all other functionality to be
    accessed through these tools.
    """
    
    def __init__(self):
        """Initialize the core registry tools manager."""
        self.registry = get_registry()
        
    def register_essential_tools(self, server, get_client_manager=None):
        """
        Register only the essential registry tools with the server.
        
        This method registers only the minimal set of tools needed for
        the Function Registry Pattern to work, significantly reducing
        the total number of exposed tools.
        
        Args:
            server: The MCP server instance
            get_client_manager: Optional function to get client manager
        """
        @server.tool()
        async def execute_function(function_name: str, **parameters) -> str:
            """
            Execute any registered function by name with provided parameters.
            
            This tool provides a unified interface to all registered functions, allowing
            access to the full range of functionality through a single entry point.
            
            Args:
                function_name: Full name of function (e.g., 'memory.create_entity')
                **parameters: Parameters to pass to the function
                
            Returns:
                JSON string with the function result
            """
            if not function_name:
                error_result = FunctionResult(
                    status="error",
                    data=None,
                    message="Missing required parameter 'function_name'",
                    error_code="MISSING_PARAMETER",
                    error_details={"message": "function_name parameter is required"}
                )
                return error_result.to_json()
                
            try:
                # Get function metadata for validation and conversion
                metadata = self.registry.get_function_metadata(function_name)
                if metadata is None:
                    error_result = FunctionResult(
                        status="error",
                        data=None,
                        message=f"Function '{function_name}' not found",
                        error_code="FUNCTION_NOT_FOUND",
                        error_details={
                            "available_namespaces": list(self.registry.get_namespaces())
                        }
                    )
                    return error_result.to_json()
                
                # Try to validate and convert parameters
                try:
                    # Basic parameter validation
                    validation_errors = ParameterHelper.validate_parameters(metadata, parameters)
                    if validation_errors:
                        error_details = {
                            "validation_errors": [str(error) for error in validation_errors]
                        }
                        error_result = FunctionResult(
                            status="error",
                            message=f"Parameter validation failed for function '{function_name}'",
                            data=None,
                            error_code="PARAMETER_VALIDATION_ERROR",
                            error_details=error_details
                        )
                        return error_result.to_json()
                    
                    # Convert parameters to the right types
                    parameters = ParameterHelper.convert_parameters(metadata, parameters)
                except Exception as e:
                    logger.error(f"Error processing parameters: {str(e)}")
                    error_result = FunctionResult(
                        status="error",
                        data=None,
                        message=f"Error processing parameters: {str(e)}",
                        error_code="PARAMETER_PROCESSING_ERROR",
                        error_details={"exception": str(e)}
                    )
                    return error_result.to_json()
                
                # Execute the function
                result = await self.registry.execute(function_name, **parameters)
                return result.to_json()
            except Exception as e:
                logger.error(f"Error executing function: {str(e)}")
                error_result = FunctionResult(
                    status="error",
                    data=None,
                    message=f"Error executing function: {str(e)}",
                    error_code="FUNCTION_EXECUTION_ERROR",
                    error_details={"exception": str(e)}
                )
                return error_result.to_json()
        
        @server.tool()
        async def list_available_functions(category: Optional[str] = None) -> str:
            """
            List all available functions, optionally filtered by category.
            
            This tool allows discovery of all registered functions and their documentation.
            
            Args:
                category: Optional category to filter by
                
            Returns:
                JSON string with function metadata
            """
            try:
                if category:
                    functions = self.registry.get_functions_by_namespace(category)
                else:
                    functions = self.registry.get_all_functions()
                    
                # Convert to dictionary for better serialization
                result = {
                    "functions": [f.dict() for f in functions],
                    "count": len(functions),
                    "categories": list(self.registry.get_namespaces())
                }
                
                return json.dumps(result, indent=2)
            except Exception as e:
                error_result = {
                    "error": str(e),
                    "functions": []
                }
                return json.dumps(error_result)
                
        @server.tool()
        async def list_function_categories() -> str:
            """
            List all available function categories/namespaces.
            
            Returns:
                JSON string with category information
            """
            try:
                namespaces = self.registry.get_namespaces()
                
                # Get function count per namespace
                category_counts = {}
                for ns in namespaces:
                    functions = self.registry.get_functions_by_namespace(ns)
                    category_counts[ns] = len(functions)
                
                result = {
                    "categories": list(namespaces),
                    "category_counts": category_counts,
                    "total_categories": len(namespaces)
                }
                
                return json.dumps(result, indent=2)
            except Exception as e:
                error_result = {
                    "error": str(e),
                    "categories": []
                }
                return json.dumps(error_result)
        
        logger.info(f"Registered essential registry tools")

# Create a singleton instance
core_registry_tools_manager = CoreRegistryToolsManager()

def register_essential_registry_tools(server, get_client_manager=None):
    """
    Register only the essential registry tools with the server.
    
    This is the main entry point for registering the minimal set of tools.
    
    Args:
        server: The MCP server instance
        get_client_manager: Optional function to get client manager
    """
    core_registry_tools_manager.register_essential_tools(server, get_client_manager) 