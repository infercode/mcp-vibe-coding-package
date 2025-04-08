#!/usr/bin/env python3
"""
Docstring Parser for Function Registry

This module provides utilities for parsing function docstrings to extract
rich metadata, including detailed nested parameter information.
"""

import re
from typing import Dict, List, Any, Optional, Tuple

class DocstringParser:
    """
    Parser for extracting rich metadata from function docstrings.
    
    This class parses docstrings to extract comprehensive parameter descriptions,
    including nested structures like dictionaries, which is especially important
    for AI agent tool usage.
    """
    
    @staticmethod
    def parse_docstring(docstring: str) -> Dict[str, Any]:
        """
        Parse a function docstring to extract structured information.
        
        Args:
            docstring: The function docstring to parse
            
        Returns:
            Dictionary with parsed information including description, parameters, returns, etc.
        """
        if not docstring:
            return {"description": "", "parameters": {}, "returns": ""}
        
        # Extract main description (everything before Args/Parameters section)
        description = DocstringParser._extract_description(docstring)
        
        # Extract parameter information
        parameters = DocstringParser._extract_parameters(docstring)
        
        # Extract return information
        returns = DocstringParser._extract_returns(docstring)
        
        # Extract examples
        examples = DocstringParser._extract_examples(docstring)
        
        return {
            "description": description,
            "parameters": parameters,
            "returns": returns,
            "examples": examples
        }
    
    @staticmethod
    def _extract_description(docstring: str) -> str:
        """Extract the main description from a docstring."""
        # Split by sections (Args:, Parameters:, Returns:, etc.)
        sections = re.split(r'\n\s*(?:Args|Parameters|Returns|Raises|Examples|Notes|Yields):', docstring)
        
        if not sections:
            return ""
            
        # First section is the description
        description = sections[0].strip()
        
        # If description has multiple paragraphs, join them
        paragraphs = re.split(r'\n\s*\n', description)
        description = '\n\n'.join(p.strip() for p in paragraphs)
        
        return description
    
    @staticmethod
    def _extract_parameters(docstring: str) -> Dict[str, Dict[str, Any]]:
        """
        Extract detailed parameter information from a docstring.
        
        This method handles both standard parameter descriptions and nested
        dictionary structures like:
        
        Args:
            config_data: Dictionary with configuration options
                - name: The name of the configuration
                - values: List of allowed values
                - metadata: Additional settings
                    - visible: Whether the option is visible
                    - priority: Importance level
            timeout: Time to wait in seconds
        """
        parameters = {}
        
        # Find the Args/Parameters section
        args_match = re.search(r'(?:Args|Parameters):(.*?)(?:\n\s*(?:Returns|Raises|Examples|Notes|Yields):|$)', 
                              docstring, re.DOTALL)
        
        if not args_match:
            return parameters
            
        args_section = args_match.group(1).strip()
        if not args_section:
            return parameters
            
        # Split into parameter blocks (each parameter and its description)
        # Pattern matches parameter name followed by description, handling indentation
        param_pattern = re.compile(r'(\s*)(\w+)(?:\s*\(([^)]+)\))?\s*:\s*(.*?)(?=\n\s*\w+\s*:|$)', re.DOTALL)
        
        # First pass: Extract all parameters with their indentation levels
        param_blocks = []
        for match in param_pattern.finditer(args_section):
            indent = len(match.group(1))
            name = match.group(2)
            param_type = match.group(3)  # Might be None
            description = match.group(4).strip()
            
            param_blocks.append({
                "name": name,
                "type": param_type,
                "description": description,
                "indent": indent,
                "nested_params": {}
            })
        
        # Second pass: Build hierarchy of nested parameters
        root_params = {}
        param_stack = []
        
        for param in param_blocks:
            # Process indentation to determine nesting level
            while param_stack and param_stack[-1]["indent"] >= param["indent"]:
                param_stack.pop()
                
            if not param_stack:
                # This is a root parameter
                root_params[param["name"]] = param
            else:
                # This is a nested parameter
                parent = param_stack[-1]
                parent["nested_params"][param["name"]] = param
                
            param_stack.append(param)
        
        # Third pass: Extract nested field information from descriptions
        for param_name, param_info in root_params.items():
            # Check if this parameter's description contains nested fields
            nested_fields = DocstringParser._extract_nested_fields(param_info["description"])
            
            # Store the structured parameter info
            parameters[param_name] = {
                "description": param_info["description"].split('\n')[0] if param_info["description"] else "",
                "type": param_info["type"],
                "nested_fields": nested_fields,
                "nested_params": {name: {
                    "description": info["description"],
                    "type": info["type"],
                    "nested_fields": DocstringParser._extract_nested_fields(info["description"])
                } for name, info in param_info["nested_params"].items()}
            }
        
        return parameters
    
    @staticmethod
    def _extract_nested_fields(description: str) -> Dict[str, Dict[str, Any]]:
        """Extract nested field descriptions from a parameter description."""
        nested_fields = {}
        
        # Look for field descriptions in the format:
        # - field_name: field_description
        # or 
        # * field_name: field_description
        lines = description.split('\n')
        
        current_indent = None
        current_field = None
        
        for line in lines:
            # Skip empty lines
            if not line.strip():
                continue
                
            # Check if this line defines a field
            field_match = re.match(r'(\s*)[-*]\s+(\w+):\s*(.*)', line)
            if field_match:
                indent = len(field_match.group(1))
                field_name = field_match.group(2)
                field_desc = field_match.group(3).strip()
                
                # Determine if this is a top-level field or nested
                if current_indent is None or indent == current_indent:
                    # Top-level field
                    current_indent = indent
                    current_field = field_name
                    
                    # Extract required/optional info
                    is_required = "required" in field_desc.lower() or "Required" in field_desc
                    
                    nested_fields[field_name] = {
                        "description": field_desc,
                        "required": is_required,
                        "nested_fields": {}
                    }
                elif indent > current_indent and current_field:
                    # Nested field
                    parent_field = current_field
                    current_field = field_name
                    
                    if "nested_fields" not in nested_fields[parent_field]:
                        nested_fields[parent_field]["nested_fields"] = {}
                        
                    nested_fields[parent_field]["nested_fields"][field_name] = {
                        "description": field_desc,
                        "required": "required" in field_desc.lower() or "Required" in field_desc
                    }
            elif current_field and line.strip() and re.match(r'(\s+)', line) and not re.match(r'\s*[-*]', line):
                # Continuation of previous field description
                if current_field in nested_fields:
                    nested_fields[current_field]["description"] += " " + line.strip()
        
        return nested_fields
    
    @staticmethod
    def _extract_returns(docstring: str) -> Dict[str, Any]:
        """Extract return value information from a docstring."""
        returns = {"description": "", "type": None}
        
        # Find the Returns section
        returns_match = re.search(r'Returns:(.*?)(?:\n\s*(?:Args|Parameters|Raises|Examples|Notes|Yields):|$)', 
                                 docstring, re.DOTALL)
        
        if not returns_match:
            return returns
            
        returns_section = returns_match.group(1).strip()
        
        # Check if return type is specified
        type_match = re.match(r'(\w+):\s*(.*)', returns_section)
        if type_match:
            returns["type"] = type_match.group(1)
            returns["description"] = type_match.group(2).strip()
        else:
            returns["description"] = returns_section
            
        return returns
    
    @staticmethod
    def _extract_examples(docstring: str) -> List[Dict[str, str]]:
        """Extract usage examples from a docstring."""
        examples = []
        
        # Find the Examples section
        examples_match = re.search(r'Examples:(.*?)(?:\n\s*(?:Args|Parameters|Returns|Raises|Notes|Yields):|$)', 
                                  docstring, re.DOTALL)
        
        if not examples_match:
            return examples
            
        examples_section = examples_match.group(1).strip()
        
        # Look for code blocks
        code_blocks = re.findall(r'```(?:python)?\s*(.*?)```', examples_section, re.DOTALL)
        
        if code_blocks:
            for block in code_blocks:
                examples.append({"code": block.strip()})
        else:
            # If no code blocks found, try to split by lines or paragraphs
            example_blocks = re.split(r'\n\s*\n', examples_section)
            for block in example_blocks:
                if block.strip():
                    examples.append({"code": block.strip()})
        
        return examples


def parse_docstring(docstring: str) -> Dict[str, Any]:
    """
    Parse a function docstring to extract structured information.
    
    This is a convenience function that uses DocstringParser to extract
    rich metadata from function docstrings.
    
    Args:
        docstring: The function docstring to parse
        
    Returns:
        Dictionary with parsed information including description, parameters, returns, etc.
    """
    return DocstringParser.parse_docstring(docstring) 