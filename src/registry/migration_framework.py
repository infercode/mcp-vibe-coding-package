#!/usr/bin/env python3
"""
Tool Migration Framework

This module provides utilities for migrating existing tools to the Function Registry Pattern.
It handles analyzing tool files, registering functions, and providing backward compatibility.
"""

import inspect
import importlib
import re
import os
import sys
import ast
import json
from typing import Dict, List, Any, Optional, Union, Callable, Set, Tuple, Type

from src.registry.registry_manager import register_function, get_registry
from src.registry.function_models import FunctionMetadata, FunctionResult
from src.logger import get_logger

# Initialize logger
logger = get_logger()

class ToolAnalyzer:
    """
    Analyzes existing tool modules to extract function signatures and documentation.
    
    This class uses AST and inspect to analyze Python modules and extract
    function information for migration to the registry.
    """
    
    def __init__(self):
        """Initialize the tool analyzer."""
        self.analyzed_modules = {}
        
    def analyze_module(self, module_path: str) -> Dict[str, Any]:
        """
        Analyze a module to extract tool functions.
        
        Args:
            module_path: Import path for the module (e.g., 'src.tools.core_memory_tools')
            
        Returns:
            Dictionary with analysis results
        """
        # Skip if already analyzed
        if module_path in self.analyzed_modules:
            return self.analyzed_modules[module_path]
            
        try:
            # Import the module
            module = importlib.import_module(module_path)
            
            # Find the register_*_tools function
            register_func = None
            register_func_name = None
            for name, obj in inspect.getmembers(module):
                if name.startswith('register_') and name.endswith('_tools') and callable(obj):
                    register_func = obj
                    register_func_name = name
                    break
            
            if not register_func:
                logger.warn(f"No registration function found in module {module_path}")
                return {}
            
            # Get source code of the module
            try:
                source = inspect.getsource(module)
            except Exception as e:
                logger.error(f"Failed to get source code for module {module_path}: {str(e)}")
                return {}
                
            # Parse the source code with AST
            tree = ast.parse(source)
            
            # Extract tool definitions
            tool_functions = self._find_tool_decorators(tree, register_func_name)
            
            # Get detailed function information
            functions_info = {}
            for func_name in tool_functions:
                # Find the actual function in the module
                for name, obj in inspect.getmembers(module):
                    if name == func_name and callable(obj):
                        functions_info[func_name] = self._extract_function_info(obj)
                        break
            
            # Store the results
            result = {
                "module_path": module_path,
                "module_name": module.__name__,
                "register_function": register_func_name,
                "functions": functions_info
            }
            
            self.analyzed_modules[module_path] = result
            return result
            
        except Exception as e:
            logger.error(f"Error analyzing module {module_path}: {str(e)}")
            return {}
    
    def _find_tool_decorators(self, tree: ast.AST, register_func_name: Optional[str]) -> List[str]:
        """Find function names decorated with @server.tool() in an AST."""
        tool_functions = []
        
        # If no register function name, return empty list
        if not register_func_name:
            return []
        
        # Find the register function first
        register_func_node = None
        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef) and node.name == register_func_name:
                register_func_node = node
                break
                
        if not register_func_node:
            return []
            
        # Look for @server.tool() decorators inside the register function
        for node in ast.walk(register_func_node):
            if isinstance(node, ast.FunctionDef):
                # This is a nested function definition
                tool_name = node.name
                for decorator in node.decorator_list:
                    # Look for @server.tool() decorator
                    if isinstance(decorator, ast.Call):
                        func = decorator.func
                        if isinstance(func, ast.Attribute) and func.attr == 'tool':
                            tool_functions.append(tool_name)
                            break
        
        return tool_functions
    
    def _extract_function_info(self, func: Callable) -> Dict[str, Any]:
        """Extract detailed information about a function."""
        # Get signature
        sig = inspect.signature(func)
        
        # Get docstring
        docstring = inspect.getdoc(func) or ""
        
        # Extract parameter information
        parameters = {}
        for name, param in sig.parameters.items():
            if name in ('self', 'cls'):
                continue
                
            # Get type annotation
            param_type = 'any'
            if param.annotation != inspect.Parameter.empty:
                if hasattr(param.annotation, '__name__'):
                    param_type = param.annotation.__name__
                else:
                    param_type = str(param.annotation)
            
            # Check if required
            has_default = param.default != inspect.Parameter.empty
            
            # Get default value
            default_value = None if not has_default else param.default
            
            # Store parameter info
            parameters[name] = {
                'type': param_type,
                'required': not has_default,
                'default': default_value
            }
        
        # Extract return type
        return_type = 'any'
        if sig.return_annotation != inspect.Signature.empty:
            if hasattr(sig.return_annotation, '__name__'):
                return_type = sig.return_annotation.__name__
            else:
                return_type = str(sig.return_annotation)
        
        # Check if async
        is_async = inspect.iscoroutinefunction(func)
        
        # Parse docstring to extract parameter descriptions
        param_descriptions = self._extract_param_descriptions(docstring)
        for name, desc in param_descriptions.items():
            if name in parameters:
                parameters[name]['description'] = desc
                
        # Extract description from docstring
        description = self._extract_description(docstring)
        
        return {
            'name': func.__name__,
            'description': description,
            'parameters': parameters,
            'return_type': return_type,
            'is_async': is_async,
            'docstring': docstring
        }
    
    def _extract_param_descriptions(self, docstring: str) -> Dict[str, str]:
        """Extract parameter descriptions from a docstring."""
        params = {}
        
        # Look for Args: section in docstring
        args_match = re.search(r'Args:(.*?)(?:\n\s*\n|\n\s*Returns:|\n\s*Raises:|\Z)', 
                              docstring, re.DOTALL)
        if not args_match:
            return params
            
        args_section = args_match.group(1)
        
        # Extract individual parameter descriptions
        param_pattern = re.compile(r'\s*(\w+):\s*(.*?)(?=\n\s*\w+:|$)', re.DOTALL)
        for match in param_pattern.finditer(args_section):
            param_name = match.group(1)
            param_desc = ' '.join(match.group(2).split())  # Normalize whitespace
            params[param_name] = param_desc
            
        return params
    
    def _extract_description(self, docstring: str) -> str:
        """Extract the main description from a docstring."""
        if not docstring:
            return ""
            
        # Take the first paragraph as description
        paragraphs = re.split(r'\n\s*\n', docstring)
        if paragraphs:
            # Remove any leading/trailing whitespace and normalize internal whitespace
            return ' '.join(paragraphs[0].split())
            
        return ""


class MigrationManager:
    """
    Manages the migration of tools to the function registry.
    
    This class provides utilities for registering existing tools with
    the registry and maintaining backward compatibility.
    """
    
    def __init__(self):
        """Initialize the migration manager."""
        self.analyzer = ToolAnalyzer()
        self.registry = get_registry()
        self.migrated_modules = set()
        self.namespace_mappings = {
            'core_memory_tools': 'memory',
            'lesson_memory_tools': 'lesson',
            'project_memory_tools': 'project',
            'config_tools': 'config'
        }
    
    def migrate_module(self, module_path: str, namespace: Optional[str] = None) -> Dict[str, Any]:
        """
        Migrate all tools in a module to the registry.
        
        Args:
            module_path: Import path for the module (e.g., 'src.tools.core_memory_tools')
            namespace: Optional namespace override (defaults to derived from module name)
            
        Returns:
            Dictionary with migration results
        """
        # Skip if already migrated
        if module_path in self.migrated_modules:
            logger.info(f"Module {module_path} already migrated")
            return {"status": "already_migrated", "module_path": module_path}
            
        # Analyze the module
        analysis = self.analyzer.analyze_module(module_path)
        if not analysis:
            logger.error(f"Failed to analyze module {module_path}")
            return {"status": "error", "message": f"Failed to analyze module {module_path}"}
            
        # Determine namespace
        if not namespace:
            # Try to derive from module name
            module_name = analysis.get('module_name', '').split('.')[-1]
            namespace = self.namespace_mappings.get(module_name, module_name)
        
        # Register functions with the registry
        try:
            module = importlib.import_module(module_path)
            
            registered_functions = []
            for func_name, func_info in analysis.get('functions', {}).items():
                # Find the actual function in the module
                for name, obj in inspect.getmembers(module):
                    if name == func_name and callable(obj):
                        # Create proxy function for the registry
                        proxy_func = self._create_proxy_function(obj, func_info)
                        
                        # Register with the registry
                        self.registry.register(namespace, func_name, proxy_func)
                        registered_functions.append(func_name)
                        break
            
            # Mark as migrated
            self.migrated_modules.add(module_path)
            
            return {
                "status": "success",
                "module_path": module_path,
                "namespace": namespace,
                "registered_functions": registered_functions
            }
            
        except Exception as e:
            logger.error(f"Error migrating module {module_path}: {str(e)}")
            return {"status": "error", "message": str(e)}
    
    def _create_proxy_function(self, original_func: Callable, func_info: Dict[str, Any]) -> Callable:
        """
        Create a proxy function that preserves the original function's signature.
        
        Args:
            original_func: The original function to proxy
            func_info: Information about the function
            
        Returns:
            Proxy function
        """
        # For async functions
        if func_info.get('is_async', False):
            async def async_proxy_func(*args, **kwargs):
                try:
                    # Call the original function
                    result = await original_func(*args, **kwargs)
                    return result
                except Exception as e:
                    # Handle errors
                    error_message = f"Error in {func_info.get('name', 'unknown')}: {str(e)}"
                    if isinstance(e, (ValueError, TypeError, KeyError)):
                        # Common errors that might be related to parameter validation
                        return FunctionResult.error(
                            message=error_message,
                            error_code="parameter_error"
                        ).to_json()
                    else:
                        # General execution error
                        return FunctionResult.error(
                            message=error_message,
                            error_code="execution_error"
                        ).to_json()
                    
            # Set metadata on the proxy function
            async_proxy_func.__name__ = original_func.__name__
            async_proxy_func.__doc__ = original_func.__doc__
            async_proxy_func.__module__ = original_func.__module__
            
            return async_proxy_func
        
        # For regular functions
        else:
            def sync_proxy_func(*args, **kwargs):
                try:
                    # Call the original function
                    result = original_func(*args, **kwargs)
                    return result
                except Exception as e:
                    # Handle errors
                    error_message = f"Error in {func_info.get('name', 'unknown')}: {str(e)}"
                    if isinstance(e, (ValueError, TypeError, KeyError)):
                        # Common errors that might be related to parameter validation
                        return FunctionResult.error(
                            message=error_message,
                            error_code="parameter_error"
                        ).to_json()
                    else:
                        # General execution error
                        return FunctionResult.error(
                            message=error_message,
                            error_code="execution_error"
                        ).to_json()
            
            # Set metadata on the proxy function
            sync_proxy_func.__name__ = original_func.__name__
            sync_proxy_func.__doc__ = original_func.__doc__
            sync_proxy_func.__module__ = original_func.__module__
            
            return sync_proxy_func


def migrate_all_tools() -> Dict[str, Any]:
    """
    Migrate all standard tool modules to the registry.
    
    Returns:
        Dictionary with migration results
    """
    manager = MigrationManager()
    
    # Define modules to migrate
    modules = [
        'src.tools.core_memory_tools',
        'src.tools.lesson_memory_tools',
        'src.tools.project_memory_tools',
        'src.tools.config_tools'
    ]
    
    # Migrate each module
    results = {}
    for module_path in modules:
        result = manager.migrate_module(module_path)
        results[module_path] = result
    
    return results 