#!/usr/bin/env python3
"""
IDE Integration Optimizations

This module provides utilities for optimizing the Function Registry Pattern
for integration with IDEs that have limitations on tool counts or documentation.
"""

import json
import os
import re
from typing import Dict, List, Any, Optional, Union, Set

from src.registry.registry_manager import get_registry
from src.registry.function_models import FunctionMetadata
from src.logger import get_logger

# Initialize logger
logger = get_logger()

class IDEMetadataGenerator:
    """
    Generates IDE-friendly metadata for registry tools.
    
    This class creates optimized tool documentation, parameter suggestions,
    and discoverability hints for IDEs with limitations.
    """
    
    def __init__(self):
        """Initialize the IDE metadata generator."""
        self.registry = get_registry()
    
    def generate_tool_documentation(self, namespace: Optional[str] = None) -> Dict[str, Any]:
        """
        Generate IDE-friendly documentation for tools.
        
        Args:
            namespace: Optional namespace to limit documentation
            
        Returns:
            Dictionary with tool documentation
        """
        tools_docs = {}
        
        # Get relevant functions
        if namespace:
            functions = self.registry.get_functions_by_namespace(namespace)
            namespaces = [namespace]
        else:
            functions = self.registry.get_all_functions()
            namespaces = self.registry.get_namespaces()
        
        # Create documentation for each namespace
        namespace_docs = {}
        for ns in namespaces:
            ns_functions = [f for f in functions if f.namespace == ns]
            namespace_docs[ns] = {
                "description": f"Functions in the {ns} namespace",
                "function_count": len(ns_functions),
                "categories": self._categorize_functions(ns_functions)
            }
        
        # Create documentation for each function
        function_docs = {}
        for func in functions:
            function_docs[func.name] = {
                "short_description": func.description,
                "parameters": self._format_parameters(func.parameters),
                "return_type": func.return_type,
                "examples": func.examples,
                "namespace": func.namespace
            }
        
        # Create the main documentation object
        tools_docs = {
            "namespaces": namespace_docs,
            "functions": function_docs,
            "meta": {
                "total_functions": len(functions),
                "total_namespaces": len(namespaces)
            }
        }
        
        return tools_docs
    
    def generate_ide_tool_hints(self) -> Dict[str, Any]:
        """
        Generate hints for IDE tool discoverability.
        
        Returns:
            Dictionary with tool hints
        """
        # Get all functions
        functions = self.registry.get_all_functions()
        
        # Create hints for each function
        function_hints = {}
        for func in functions:
            # Generate parameter examples
            param_examples = {}
            for param_name, param_info in func.parameters.items():
                # Skip client_id and similar parameters
                if param_name.lower() in ('client_id', 'context'):
                    continue
                    
                # Generate example value based on type
                param_type = param_info.get('type', 'any')
                if 'default' in param_info and param_info['default'] is not None:
                    # Use default value as example
                    param_examples[param_name] = param_info['default']
                elif param_type == 'str':
                    param_examples[param_name] = f"example_{param_name}"
                elif param_type == 'int':
                    param_examples[param_name] = 42
                elif param_type == 'float':
                    param_examples[param_name] = 3.14
                elif param_type == 'bool':
                    param_examples[param_name] = True
                elif 'List' in param_type or 'list' in param_type:
                    param_examples[param_name] = ["example"]
                elif 'Dict' in param_type or 'dict' in param_type:
                    param_examples[param_name] = {"key": "value"}
                
            # Create hint object
            function_hints[func.name] = {
                "summary": func.description,
                "parameter_examples": param_examples,
                "required_parameters": [
                    name for name, info in func.parameters.items()
                    if info.get('required', False)
                ],
                "common_use_case": f"Use to {func.description.lower() if func.description else 'perform operation'}"
            }
        
        return function_hints
    
    def generate_category_tools(self) -> Dict[str, Any]:
        """
        Generate category-based meta-tools for IDE integration.
        
        This creates specialized tool definitions for each namespace,
        which can be used to provide IDE-friendly access to functions.
        
        Returns:
            Dictionary with category tool definitions
        """
        # Get all namespaces
        namespaces = self.registry.get_namespaces()
        
        # Create tool definitions for each namespace
        category_tools = {}
        for namespace in namespaces:
            # Get functions in namespace
            functions = self.registry.get_functions_by_namespace(namespace)
            
            # Create parameter definition for the category tool
            # This will include a 'command' parameter and all possible parameters
            # from the functions in this namespace
            parameters = {
                "command": {
                    "type": "str",
                    "description": f"The {namespace} function to execute",
                    "required": True,
                    "enum": [func.short_name for func in functions]
                }
            }
            
            # Add all parameters from all functions in this namespace
            for func in functions:
                for param_name, param_info in func.parameters.items():
                    # Skip client_id and similar parameters
                    if param_name.lower() in ('client_id', 'context'):
                        continue
                        
                    # Add parameter with function name prefix to avoid conflicts
                    param_key = f"{func.short_name}.{param_name}"
                    parameters[param_key] = {
                        "type": param_info.get('type', 'any'),
                        "description": f"Parameter for {func.short_name}: {param_info.get('description', param_name)}",
                        "required": False
                    }
            
            # Create the category tool definition
            category_tools[namespace] = {
                "name": f"{namespace}_tool",
                "description": f"Execute {namespace} functions",
                "parameters": parameters,
                "function_count": len(functions),
                "function_names": [func.short_name for func in functions]
            }
        
        return category_tools
    
    def _categorize_functions(self, functions: List[FunctionMetadata]) -> Dict[str, List[str]]:
        """Categorize functions by common prefixes or functionality."""
        categories = {}
        
        # Group by common prefixes
        for func in functions:
            short_name = func.short_name
            
            # Try to extract category from function name
            category = None
            if '_' in short_name:
                parts = short_name.split('_')
                if len(parts) >= 2:
                    category = parts[0]
            
            # If no category found, use first letter
            if not category:
                category = "other"
                
            # Add to category
            if category not in categories:
                categories[category] = []
            categories[category].append(short_name)
        
        return categories
    
    def _format_parameters(self, parameters: Dict[str, Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
        """Format parameters for IDE-friendly documentation."""
        formatted_params = {}
        
        for name, info in parameters.items():
            # Format the parameter info
            param_info = {
                "type": info.get('type', 'any'),
                "description": info.get('description', f"Parameter {name}"),
                "required": info.get('required', False)
            }
            
            # Add default value if present
            if 'default' in info and info['default'] is not None:
                param_info['default'] = info['default']
                
            formatted_params[name] = param_info
            
        return formatted_params


def generate_ide_optimized_tools() -> Dict[str, Any]:
    """
    Generate IDE-optimized tool definitions.
    
    This function creates specialized tool definitions that work well
    with IDEs that have limitations on tool counts or documentation.
    
    Returns:
        Dictionary with optimized tool definitions
    """
    generator = IDEMetadataGenerator()
    
    # Generate various metadata
    docs = generator.generate_tool_documentation()
    hints = generator.generate_ide_tool_hints()
    category_tools = generator.generate_category_tools()
    
    # Combine into a single result
    return {
        "documentation": docs,
        "hints": hints,
        "category_tools": category_tools
    }


def export_ide_optimized_tools(export_path: str) -> str:
    """
    Export IDE-optimized tool definitions to a JSON file.
    
    Args:
        export_path: Path to export the JSON file
        
    Returns:
        Path to the exported file
    """
    # Generate the optimized tools
    tools = generate_ide_optimized_tools()
    
    # Ensure the directory exists
    os.makedirs(os.path.dirname(os.path.abspath(export_path)), exist_ok=True)
    
    # Write to file
    with open(export_path, 'w') as f:
        json.dump(tools, f, indent=2)
        
    logger.info(f"Exported IDE-optimized tools to {export_path}")
    return export_path 