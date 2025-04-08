#!/usr/bin/env python3
"""
IDE Integration Optimizations Demo

This script demonstrates the functionality of the IDE Integration Optimizations,
which provides IDE-friendly tool definitions, documentation, and discoverability.
"""

import os
import sys
import json
import asyncio
from typing import Dict, List, Any, Optional, Union, Callable

# Add the project root to the path for proper imports
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

# Declare the types to avoid linter errors with fallbacks
FunctionRegistry = Any
FunctionMetadataType = Any
RegistryFunctionResult = Any
IDEMetadataGeneratorType = Any

# Import necessary components with fallbacks to handle potential circular imports
try:
    from src.registry.registry_manager import get_registry
    registry_available = True
except ImportError:
    print("Warning: Could not import registry_manager. Using mock implementation.")
    registry_available = False
    
    # Mock registry
    class MockRegistry:
        def get_functions_by_namespace(self, namespace):
            return []
        
        def get_namespaces(self):
            return []
    
    def get_registry():
        return MockRegistry()

try:
    from src.registry.function_models import FunctionMetadata, FunctionResult
    function_models_available = True
except ImportError:
    print("Warning: Could not import function_models. Using mock implementation.")
    function_models_available = False
    
    # Mock function models
    class MockFunctionMetadata:
        def __init__(self, name="", description="", parameters=None, namespace="", short_name=""):
            self.name = name
            self.description = description
            self.parameters = parameters or {}
            self.namespace = namespace
            self.short_name = short_name
            self.return_type = "any"
            self.examples = []
    
    class MockFunctionResult:
        def __init__(self, success=True, data=None, error=None):
            self.success = success
            self.data = data or {}
            self.error = error or {}
        
        def to_json(self):
            return json.dumps({"success": self.success, "data": self.data, "error": self.error})
    
    FunctionMetadata = MockFunctionMetadata
    FunctionResult = MockFunctionResult

try:
    from src.registry.ide_integration import (
        IDEMetadataGenerator,
        generate_ide_optimized_tools,
        export_ide_optimized_tools
    )
    ide_integration_available = True
except ImportError:
    print("Warning: Could not import ide_integration. Demo will show simulated outputs.")
    ide_integration_available = False
    
    # Mock IDE integration components
    class MockIDEMetadataGenerator:
        def __init__(self):
            self.registry = None
            
        def generate_tool_documentation(self, namespace=None):
            return {
                "meta": {"total_functions": 0, "total_namespaces": 0},
                "namespaces": {},
                "functions": {}
            }
            
        def generate_ide_tool_hints(self):
            return {}
            
        def generate_category_tools(self):
            return {}
    
    IDEMetadataGenerator = MockIDEMetadataGenerator
    
    def generate_ide_optimized_tools():
        return {
            "category_tools": {},
            "documentation": {},
            "hints": {}
        }
        
    def export_ide_optimized_tools(export_path):
        os.makedirs(os.path.dirname(export_path), exist_ok=True)
        with open(export_path, 'w') as f:
            json.dump({}, f)
        return export_path

# Sample output directory for exported files
DEMO_OUTPUT_DIR = "demo_output"

async def demo_tool_documentation():
    """Demonstrate IDE-friendly tool documentation generation."""
    print("\n=== IDE-Friendly Documentation Demo ===\n")
    
    # Create a metadata generator
    generator = IDEMetadataGenerator()
    
    # Generate documentation for all functions
    print("Generating IDE-friendly documentation for all functions...")
    docs = generator.generate_tool_documentation()
    
    # Display summary
    print(f"Generated documentation for {docs['meta']['total_functions']} functions across {docs['meta']['total_namespaces']} namespaces")
    
    # Show documentation for each namespace
    print("\nNamespace documentation:")
    for ns_name, ns_info in docs['namespaces'].items():
        print(f"  Namespace: {ns_name}")
        print(f"  Description: {ns_info['description']}")
        print(f"  Function count: {ns_info['function_count']}")
        
        # Show categories within namespace
        if ns_info.get('categories'):
            print("  Categories:")
            for category, functions in ns_info['categories'].items():
                print(f"    - {category}: {len(functions)} functions")
        print()
    
    # Show sample function documentation
    print("Sample function documentation:")
    if docs['functions']:
        # Get the first function
        func_name = next(iter(docs['functions']))
        func_info = docs['functions'][func_name]
        
        print(f"  Function: {func_name}")
        print(f"  Description: {func_info['short_description']}")
        print("  Parameters:")
        for param_name, param_info in func_info['parameters'].items():
            required = " (required)" if param_info.get('required') else ""
            print(f"    - {param_name}: {param_info['type']}{required}")
            print(f"      Description: {param_info['description']}")
        
        print(f"  Return type: {func_info['return_type']}")
        
        if func_info.get('examples'):
            print("  Examples:")
            for example in func_info['examples']:
                print(f"    - {example}")

async def demo_tool_hints():
    """Demonstrate IDE tool discoverability hints."""
    print("\n=== IDE Tool Hints Demo ===\n")
    
    # Create a metadata generator
    generator = IDEMetadataGenerator()
    
    # Generate tool hints
    print("Generating IDE tool hints...")
    hints = generator.generate_ide_tool_hints()
    
    # Display summary
    print(f"Generated hints for {len(hints)} functions")
    
    # Show sample hints
    print("\nSample function hints:")
    if hints:
        # Get the first hint
        func_name = next(iter(hints))
        hint_info = hints[func_name]
        
        print(f"  Function: {func_name}")
        print(f"  Summary: {hint_info['summary']}")
        print(f"  Common use case: {hint_info['common_use_case']}")
        
        if hint_info.get('required_parameters'):
            print("  Required parameters:")
            for param in hint_info['required_parameters']:
                print(f"    - {param}")
                
        print("  Parameter examples:")
        for param_name, example in hint_info['parameter_examples'].items():
            print(f"    - {param_name}: {example}")

async def demo_category_tools():
    """Demonstrate category-based meta-tools for IDE integration."""
    print("\n=== Category-Based Tools Demo ===\n")
    
    # Create a metadata generator
    generator = IDEMetadataGenerator()
    
    # Generate category tools
    print("Generating category-based tools...")
    category_tools = generator.generate_category_tools()
    
    # Display summary
    print(f"Generated category tools for {len(category_tools)} namespaces")
    
    # Show sample category tool
    print("\nSample category tool definition:")
    if category_tools:
        # Get the first category
        category_name = next(iter(category_tools))
        tool_info = category_tools[category_name]
        
        print(f"  Tool name: {tool_info['name']}")
        print(f"  Description: {tool_info['description']}")
        print(f"  Contains {tool_info['function_count']} functions")
        
        print("\n  Available commands:")
        for func_name in tool_info['function_names']:
            print(f"    - {func_name}")
            
        # Show parameter structure
        print("\n  Parameter structure:")
        print(f"    - command: Enum of {len(tool_info['function_names'])} values")
        
        # Show a sample of other parameters
        param_count = 0
        for param_name, param_info in tool_info['parameters'].items():
            if param_name != "command" and param_count < 3:
                print(f"    - {param_name}: {param_info['type']} - {param_info['description']}")
                param_count += 1
        
        if len(tool_info['parameters']) > 4:
            additional = len(tool_info['parameters']) - 4
            print(f"    ... and {additional} more parameters")

async def demo_optimized_tools():
    """Demonstrate generation of IDE-optimized tool definitions."""
    print("\n=== IDE-Optimized Tools Demo ===\n")
    
    # Generate optimized tools
    print("Generating IDE-optimized tool definitions...")
    
    try:
        tools = generate_ide_optimized_tools()
        
        # Display summary
        if tools:
            print(f"Generated optimized tool definitions:")
            print(f"  Category tools: {len(tools.get('category_tools', {}))}")
            print(f"  Documentation size: {len(json.dumps(tools.get('documentation', {})))} bytes")
            print(f"  Tool hints: {len(tools.get('hints', {}))}")
            
            # Export the tools
            os.makedirs(DEMO_OUTPUT_DIR, exist_ok=True)
            export_path = os.path.join(DEMO_OUTPUT_DIR, "ide_optimized_tools.json")
            
            try:
                result_path = export_ide_optimized_tools(export_path)
                print(f"\nExported IDE-optimized tools to: {result_path}")
            except Exception as e:
                print(f"Export error: {str(e)}")
        else:
            print("Failed to generate IDE-optimized tools")
    except Exception as e:
        print(f"Error generating optimized tools: {str(e)}")

async def demo_ide_integration():
    """Run the complete IDE Integration Optimizations demonstration."""
    print("\n=== IDE Integration Optimizations Demonstration ===\n")
    
    # Ensure output directory exists
    os.makedirs(DEMO_OUTPUT_DIR, exist_ok=True)
    
    # Run each demo in sequence
    await demo_tool_documentation()
    await demo_tool_hints()
    await demo_category_tools()
    await demo_optimized_tools()
    
    print("\n=== IDE Integration Optimizations Demo Complete ===\n")
    
    # Show conclusion
    print("The IDE Integration Optimizations enable:")
    print("1. IDE-friendly tool documentation with categorized functions")
    print("2. Tool discoverability hints with examples and usage patterns")
    print("3. Category-based meta-tools for consolidated function access")
    print("4. Export capabilities for IDE integration")
    print("\nThese optimizations improve the developer experience in IDEs")
    print("with limited tool expansion capabilities, enhancing discoverability")
    print("and usability of registry functions.")

if __name__ == "__main__":
    asyncio.run(demo_ide_integration()) 