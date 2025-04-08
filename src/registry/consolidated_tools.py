#!/usr/bin/env python3
"""
Consolidated Category Tools

This module implements consolidated category-based tools for the Function Registry Pattern.
Instead of exposing every function as a separate tool, it groups functions by namespace
and exposes them as a smaller number of category tools, addressing IDE limitations.
"""

import json
from typing import Dict, List, Any, Optional, Callable, Union

from src.registry.registry_manager import get_registry
from src.registry.function_models import FunctionResult
from src.registry.ide_integration import IDEMetadataGenerator
from src.logger import get_logger

# Initialize logger
logger = get_logger()

class ConsolidatedToolsManager:
    """
    Manager for consolidated category-based tools.
    
    This class provides utilities for registering consolidated tools with
    an MCP server, reducing the total number of exposed tools.
    """
    
    def __init__(self):
        """Initialize the consolidated tools manager."""
        self.registry = get_registry()
        self.ide_generator = IDEMetadataGenerator()
        self.category_tools = self.ide_generator.generate_category_tools()
        
    def register_consolidated_tools(self, server, get_client_manager=None):
        """
        Register consolidated category tools with the server.
        
        Instead of registering every function as a separate tool, this method
        registers one tool per namespace/category, significantly reducing
        the total number of exposed tools.
        
        Args:
            server: The MCP server instance
            get_client_manager: Optional function to get client manager
        """
        # Register core registry tools that must remain separate
        self._register_core_tools(server)
        
        # Register one consolidated tool per namespace
        for namespace, tool_info in self.category_tools.items():
            self._register_category_tool(server, namespace, tool_info)
            
        logger.info(f"Registered {len(self.category_tools)} consolidated category tools")
        
    def _register_core_tools(self, server):
        """
        Register essential core tools that must remain separate.
        
        Some tools like list_available_functions and execute_function need
        to be registered separately from the category tools.
        
        Args:
            server: The MCP server instance
        """
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
                    "categories": list(self.category_tools.keys())
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
        
        logger.info(f"Registered core registry tools")
    
    def _register_category_tool(self, server, namespace, tool_info):
        """
        Register a category tool for a specific namespace.
        
        This creates a single tool that can execute any function within
        the given namespace, reducing the total number of tools.
        
        Args:
            server: The MCP server instance
            namespace: Namespace for the category tool
            tool_info: Tool information from IDEMetadataGenerator
        """
        # Register the category tool
        @server.tool()
        async def execute_category_function(**parameters) -> str:
            """
            Execute a function in the specified category.
            
            Args:
                command: The function to execute (required)
                **parameters: Parameters for the function
                
            Returns:
                JSON string with execution result
            """
            # Replace the dynamic function name with the actual namespace for documentation
            execute_category_function.__name__ = f"{namespace}_tool"
            execute_category_function.__doc__ = f"""
            Execute a function in the {namespace} category.
            
            Available functions: {', '.join(tool_info.get('function_names', []))}
            
            Args:
                command: The function to execute (required)
                **parameters: Parameters for the function
                
            Returns:
                JSON string with execution result
            """
            
            # Get the command (function name)
            if 'command' not in parameters:
                error_result = FunctionResult(
                    status="error",
                    message=f"Missing required parameter 'command' for {namespace}_tool",
                    data=None,
                    error_code="MISSING_COMMAND",
                    error_details={
                        "available_commands": tool_info.get('function_names', [])
                    }
                )
                return error_result.to_json()
            
            command = parameters.pop('command')
            
            # Construct the full function name
            function_name = f"{namespace}.{command}"
            
            # Extract parameters relevant to this function
            function_params = {}
            for key, value in parameters.items():
                # If parameter is prefixed with command name, extract it
                if key.startswith(f"{command}."):
                    param_name = key.split('.', 1)[1]
                    function_params[param_name] = value
                # Also include non-prefixed parameters
                else:
                    function_params[key] = value
            
            try:
                # Execute the function
                result = await self.registry.execute(function_name, **function_params)
                return result.to_json()
            except Exception as e:
                error_result = FunctionResult(
                    status="error",
                    message=f"Error executing {function_name}: {str(e)}",
                    data=None,
                    error_code="EXECUTION_ERROR",
                    error_details={"exception": str(e)}
                )
                return error_result.to_json()
        
        # Update the tool name for proper registration
        execute_category_function.__name__ = f"{namespace}_tool"
        
        logger.info(f"Registered category tool for namespace '{namespace}'")

# Create a singleton instance
consolidated_tools_manager = ConsolidatedToolsManager()

def register_consolidated_registry_tools(server, get_client_manager=None):
    """
    Register consolidated registry tools with the server.
    
    This is the main entry point for registering consolidated tools.
    
    Args:
        server: The MCP server instance
        get_client_manager: Optional function to get client manager
    """
    consolidated_tools_manager.register_consolidated_tools(server, get_client_manager) 