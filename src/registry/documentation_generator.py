#!/usr/bin/env python3
"""
Documentation Generator

This module provides utilities for auto-generating documentation from the Function Registry,
including examples for common operations and version tracking.
"""

import os
import json
import datetime
import re
from typing import Dict, List, Any, Optional, Set

from src.registry.registry_manager import get_registry
from src.registry.function_models import FunctionMetadata
from src.logger import get_logger

logger = get_logger()

class DocumentationGenerator:
    """
    Generates comprehensive documentation for functions in the registry.
    
    Features:
    - Auto-extracts documentation from function metadata
    - Creates formatted documentation in multiple formats
    - Supports versioning and change tracking
    - Generates examples for common use cases
    """
    
    def __init__(self, version: str = "1.0.0"):
        """
        Initialize the documentation generator.
        
        Args:
            version: The current documentation version
        """
        self.registry = get_registry()
        self.version = version
        self.generation_date = datetime.datetime.now()
        self.doc_history = {}
        
    def generate_documentation(self, 
                             include_examples: bool = True,
                             include_history: bool = True) -> Dict[str, Any]:
        """
        Generate comprehensive documentation for all functions in the registry.
        
        Args:
            include_examples: Whether to include usage examples
            include_history: Whether to include version history
            
        Returns:
            Dictionary containing all documentation
        """
        # Collect all functions
        functions = self.registry.get_all_functions()
        namespaces = self.registry.get_namespaces()
        
        # Initialize documentation structure
        documentation = {
            "meta": {
                "version": self.version,
                "generated_at": self.generation_date.isoformat(),
                "function_count": len(functions),
                "namespace_count": len(namespaces)
            },
            "namespaces": {},
            "functions": {}
        }
        
        # Process each namespace
        for namespace in namespaces:
            namespace_functions = self.registry.get_functions_by_namespace(namespace)
            
            # Create namespace documentation
            documentation["namespaces"][namespace] = {
                "description": self._get_namespace_description(namespace),
                "function_count": len(namespace_functions),
                "functions": [f.name for f in namespace_functions]
            }
            
            # Add categorization if applicable
            categories = self._categorize_functions(namespace_functions)
            if categories:
                documentation["namespaces"][namespace]["categories"] = categories
        
        # Process each function
        for function in functions:
            # Generate function documentation
            func_doc = self._generate_function_documentation(function)
            
            # Add examples if requested
            if include_examples:
                func_doc["examples"] = self._generate_examples(function)
                
            # Add version history if requested
            if include_history:
                func_doc["history"] = self._get_function_history(function.name)
                
            documentation["functions"][function.name] = func_doc
            
        return documentation
    
    def export_markdown(self, output_path: str) -> str:
        """
        Export documentation as Markdown files.
        
        Args:
            output_path: Directory to write the documentation
            
        Returns:
            Path to the index Markdown file
        """
        # Generate documentation
        docs = self.generate_documentation()
        
        # Ensure output directory exists
        os.makedirs(output_path, exist_ok=True)
        
        # Write index file
        index_path = os.path.join(output_path, "index.md")
        with open(index_path, "w") as f:
            f.write(f"# Function Registry Documentation\n\n")
            f.write(f"Version: {docs['meta']['version']}  \n")
            f.write(f"Generated: {docs['meta']['generated_at']}  \n\n")
            
            f.write("## Overview\n\n")
            f.write(f"Total Functions: {docs['meta']['function_count']}  \n")
            f.write(f"Total Namespaces: {docs['meta']['namespace_count']}  \n\n")
            
            f.write("## Namespaces\n\n")
            for ns_name, ns_info in docs["namespaces"].items():
                f.write(f"* [{ns_name}]({ns_name}.md) ({ns_info['function_count']} functions)\n")
        
        # Write namespace files
        for ns_name, ns_info in docs["namespaces"].items():
            ns_path = os.path.join(output_path, f"{ns_name}.md")
            with open(ns_path, "w") as f:
                f.write(f"# Namespace: {ns_name}\n\n")
                
                if ns_info.get("description"):
                    f.write(f"{ns_info['description']}\n\n")
                    
                f.write("## Functions\n\n")
                for func_name in ns_info["functions"]:
                    func_info = docs["functions"][func_name]
                    short_desc = func_info.get("short_description", "No description")
                    f.write(f"* [{func_name.split('.')[-1]}](functions/{func_name.replace('.', '_')}.md) - {short_desc}\n")
                    
                # Write category section if available
                if "categories" in ns_info:
                    f.write("\n## Categories\n\n")
                    for category, funcs in ns_info["categories"].items():
                        f.write(f"### {category}\n\n")
                        for func_name in funcs:
                            short_desc = docs["functions"][func_name].get("short_description", "No description")
                            f.write(f"* [{func_name.split('.')[-1]}](functions/{func_name.replace('.', '_')}.md) - {short_desc}\n")
        
        # Create functions directory
        functions_dir = os.path.join(output_path, "functions")
        os.makedirs(functions_dir, exist_ok=True)
        
        # Write function files
        for func_name, func_info in docs["functions"].items():
            func_path = os.path.join(functions_dir, f"{func_name.replace('.', '_')}.md")
            with open(func_path, "w") as f:
                f.write(f"# Function: {func_name}\n\n")
                
                if func_info.get("short_description"):
                    f.write(f"{func_info['short_description']}\n\n")
                    
                if func_info.get("long_description"):
                    f.write(f"{func_info['long_description']}\n\n")
                    
                # Parameters
                if func_info.get("parameters"):
                    f.write("## Parameters\n\n")
                    f.write("| Name | Type | Required | Description |\n")
                    f.write("|------|------|----------|-------------|\n")
                    for param_name, param_info in func_info["parameters"].items():
                        required = "Yes" if param_info.get("required") else "No"
                        param_type = param_info.get("type", "any")
                        description = param_info.get("description", "")
                        f.write(f"| `{param_name}` | `{param_type}` | {required} | {description} |\n")
                        
                # Return value
                if func_info.get("return_info"):
                    f.write("\n## Return Value\n\n")
                    f.write(f"**Type**: `{func_info['return_info'].get('type', 'any')}`\n\n")
                    if func_info['return_info'].get('description'):
                        f.write(f"{func_info['return_info']['description']}\n\n")
                        
                # Examples
                if func_info.get("examples"):
                    f.write("## Examples\n\n")
                    for i, example in enumerate(func_info["examples"]):
                        f.write(f"### Example {i+1}: {example.get('title', '')}\n\n")
                        if example.get('description'):
                            f.write(f"{example['description']}\n\n")
                        f.write("```python\n")
                        f.write(example.get('code', '# No code provided'))
                        f.write("\n```\n\n")
                        
                # Version history
                if func_info.get("history"):
                    f.write("## Version History\n\n")
                    f.write("| Version | Date | Changes |\n")
                    f.write("|---------|------|----------|\n")
                    for version_entry in func_info["history"]:
                        version = version_entry.get("version", "")
                        date = version_entry.get("date", "")
                        changes = version_entry.get("changes", "")
                        f.write(f"| {version} | {date} | {changes} |\n")
        
        return index_path
    
    def export_json(self, output_path: str) -> str:
        """
        Export documentation as a JSON file.
        
        Args:
            output_path: Path to write the JSON file
            
        Returns:
            Path to the JSON file
        """
        # Generate documentation
        docs = self.generate_documentation()
        
        # Create directory if needed
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        # Write JSON file
        with open(output_path, "w") as f:
            json.dump(docs, f, indent=2)
            
        return output_path
    
    def record_function_change(self, function_name: str, 
                             version: str, 
                             changes: str, 
                             date: Optional[str] = None) -> None:
        """
        Record a change to a function's documentation history.
        
        Args:
            function_name: Name of the function
            version: Version where the change occurred
            changes: Description of the changes
            date: Date of the change (defaults to today)
        """
        if not date:
            date = datetime.datetime.now().strftime("%Y-%m-%d")
            
        if function_name not in self.doc_history:
            self.doc_history[function_name] = []
            
        self.doc_history[function_name].append({
            "version": version,
            "date": date,
            "changes": changes
        })
    
    def _generate_function_documentation(self, function: FunctionMetadata) -> Dict[str, Any]:
        """Generate detailed documentation for a single function."""
        # Extract basic information
        func_doc = {
            "name": function.name,
            "short_description": function.description,
            "long_description": self._extract_long_description(function),
            "parameters": {},
            "return_info": {
                "type": function.return_type,
                "description": self._extract_return_description(function)
            },
            "is_async": function.is_async
        }
        
        # Process parameters
        for param_name, param_info in function.parameters.items():
            func_doc["parameters"][param_name] = {
                "type": param_info.type,
                "required": param_info.required,
                "default": param_info.default,
                "description": param_info.description
            }
            
        return func_doc
    
    def _generate_examples(self, function: FunctionMetadata) -> List[Dict[str, Any]]:
        """Generate usage examples for a function."""
        examples = []
        
        # Basic example that shows all required parameters
        required_params = {}
        for param_name, param_info in function.parameters.items():
            if param_info.required:
                if param_info.type == "str":
                    required_params[param_name] = f'"{param_name}_value"'
                elif param_info.type == "int":
                    required_params[param_name] = "42"
                elif param_info.type == "float":
                    required_params[param_name] = "3.14"
                elif param_info.type == "bool":
                    required_params[param_name] = "True"
                elif param_info.type == "list":
                    required_params[param_name] = "[]"
                elif param_info.type == "dict":
                    required_params[param_name] = "{}"
                else:
                    required_params[param_name] = "None"
        
        # Build the example code
        param_str = ", ".join([f"{k}={v}" for k, v in required_params.items()])
        code = f'await execute_function(\n    "{function.name}",\n    {{{param_str}}}\n)'
        
        examples.append({
            "title": "Basic Usage",
            "description": "Basic example using required parameters",
            "code": code
        })
        
        # If this is a common function, add a specific example
        common_examples = self._get_common_examples()
        if function.name in common_examples:
            examples.append(common_examples[function.name])
            
        return examples
    
    def _get_common_examples(self) -> Dict[str, Dict[str, Any]]:
        """Get examples for common functions."""
        # In a real implementation, these would come from a database or config file
        return {
            "memory.create_entity": {
                "title": "Creating a Person Entity",
                "description": "Example of creating a new person entity",
                "code": 'await execute_function(\n    "memory.create_entity",\n    {\n        "name": "John Doe",\n        "entity_type": "person",\n        "properties": {\n            "age": 30,\n            "occupation": "Developer"\n        }\n    }\n)'
            },
            "memory.create_relation": {
                "title": "Creating a Relationship Between Entities",
                "description": "Example of creating a relationship between two entities",
                "code": 'await execute_function(\n    "memory.create_relation",\n    {\n        "source": "John Doe",\n        "relation_type": "works_at",\n        "target": "ACME Corp",\n        "properties": {\n            "start_date": "2023-01-15",\n            "position": "Senior Developer"\n        }\n    }\n)'
            }
        }
    
    def _get_function_history(self, function_name: str) -> List[Dict[str, Any]]:
        """Get version history for a function."""
        # Use the stored history if available
        if function_name in self.doc_history:
            return self.doc_history[function_name]
        
        # Otherwise, return a single entry for the current version
        return [{
            "version": self.version,
            "date": self.generation_date.strftime("%Y-%m-%d"),
            "changes": "Initial documentation"
        }]
    
    def _get_namespace_description(self, namespace: str) -> str:
        """Get a description for a namespace."""
        # In a real implementation, these would come from a database or config file
        descriptions = {
            "memory": "Functions for working with the knowledge graph memory system",
            "project": "Functions for project management and organization",
            "tool": "Meta-functions for working with the function registry itself",
            "util": "Utility functions for common operations",
            "file": "Functions for file system operations",
            "search": "Functions for searching across various data sources"
        }
        
        return descriptions.get(namespace, f"Functions in the {namespace} category")
    
    def _categorize_functions(self, functions: List[FunctionMetadata]) -> Dict[str, List[str]]:
        """Categorize functions within a namespace."""
        categories = {}
        
        for function in functions:
            # Extract just the function name (after the namespace)
            function_short_name = function.name.split(".")[-1]
            
            # Try to determine category based on naming convention
            category = None
            
            # Check if it starts with a common verb
            for verb in ["create", "get", "update", "delete", "list", "search"]:
                if function_short_name.startswith(verb):
                    category = verb.capitalize()
                    break
            
            # If no verb was found, use "Other"
            if not category:
                category = "Other"
                
            # Add to categories
            if category not in categories:
                categories[category] = []
                
            categories[category].append(function.name)
            
        return categories
    
    def _extract_long_description(self, function: FunctionMetadata) -> str:
        """Extract a long description from function metadata."""
        if not function.docstring:
            return ""
            
        # Extract description part (everything before Args/Parameters/Returns)
        desc_match = re.search(r'^(.*?)(?:\n\s*(?:Args|Parameters|Returns|Raises):|$)', 
                              function.docstring, re.DOTALL | re.MULTILINE)
        if desc_match:
            # Clean up whitespace
            return re.sub(r'\s+', ' ', desc_match.group(1)).strip()
            
        return ""
    
    def _extract_return_description(self, function: FunctionMetadata) -> str:
        """Extract return value description from docstring."""
        if not function.docstring:
            return ""
            
        # Look for Returns section
        returns_match = re.search(r'Returns:(.*?)(?:\n\s*(?:Args|Parameters|Raises):|$)', 
                                 function.docstring, re.DOTALL)
        if returns_match:
            # Clean up whitespace
            return re.sub(r'\s+', ' ', returns_match.group(1)).strip()
            
        return ""


# Convenience functions for using the generator
def generate_documentation(version: str = "1.0.0") -> Dict[str, Any]:
    """
    Generate documentation for all registered functions.
    
    Args:
        version: Documentation version
        
    Returns:
        Dictionary with all documentation
    """
    generator = DocumentationGenerator(version)
    return generator.generate_documentation()


def export_documentation_markdown(output_dir: str, version: str = "1.0.0") -> str:
    """
    Export documentation to Markdown files.
    
    Args:
        output_dir: Directory to write documentation
        version: Documentation version
        
    Returns:
        Path to index file
    """
    generator = DocumentationGenerator(version)
    return generator.export_markdown(output_dir)


def export_documentation_json(output_path: str, version: str = "1.0.0") -> str:
    """
    Export documentation to a JSON file.
    
    Args:
        output_path: Path to write JSON file
        version: Documentation version
        
    Returns:
        Path to JSON file
    """
    generator = DocumentationGenerator(version)
    return generator.export_json(output_path) 