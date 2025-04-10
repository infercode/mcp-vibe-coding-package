#!/usr/bin/env python3
"""
IDE Integration Demonstration

This script demonstrates the IDE Integration features of the Function Registry Pattern,
showing how to generate IDE-optimized tool definitions and category tools.
"""

import asyncio
import json
import os
import sys
from typing import Dict, Any

# Add the parent directory to the path so imports work
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from src.registry.ide_integration import IDEMetadataGenerator, generate_ide_optimized_tools, export_ide_optimized_tools
from src.registry.registry_manager import get_registry
import src.registry.example_functions  # Import to register example functions

# Import the migration framework to register real tools
from src.registry.migration_framework import migrate_all_tools


async def demo_tool_documentation():
    """Demonstrate the IDE-friendly tool documentation."""
    print("\n=== IDE Tool Documentation Demonstration ===\n")
    
    generator = IDEMetadataGenerator()
    
    # Generate tool documentation
    docs = generator.generate_tool_documentation()
    
    # Print summary
    print(f"Total namespaces: {docs['meta']['total_namespaces']}")
    print(f"Total functions: {docs['meta']['total_functions']}")
    
    # Print namespaces
    print("\nNamespaces:")
    for ns_name, ns_info in docs['namespaces'].items():
        print(f"  {ns_name}: {ns_info['function_count']} functions")
        
        # Print categories in namespace
        if ns_info.get('categories'):
            print("  Categories:")
            for category, functions in ns_info['categories'].items():
                print(f"    {category}: {len(functions)} functions")
    
    # Print some function documentation
    print("\nSample Function Documentation:")
    for i, (func_name, func_info) in enumerate(list(docs['functions'].items())[:3]):
        print(f"\n  {i+1}. {func_name}:")
        print(f"     Description: {func_info.get('short_description', 'No description')}")
        print(f"     Return type: {func_info.get('return_type', 'unknown')}")
        print(f"     Parameters: {len(func_info.get('parameters', {}))}")


async def demo_tool_hints():
    """Demonstrate the IDE-friendly tool hints."""
    print("\n=== IDE Tool Hints Demonstration ===\n")
    
    generator = IDEMetadataGenerator()
    
    # Generate tool hints
    hints = generator.generate_ide_tool_hints()
    
    # Print number of hints
    print(f"Generated hints for {len(hints)} functions")
    
    # Print some function hints
    print("\nSample Function Hints:")
    for i, (func_name, hint_info) in enumerate(list(hints.items())[:3]):
        print(f"\n  {i+1}. {func_name}:")
        print(f"     Summary: {hint_info.get('summary', 'No summary')}")
        print(f"     Required parameters: {hint_info.get('required_parameters', [])}")
        print(f"     Parameter examples: {json.dumps(hint_info.get('parameter_examples', {}), indent=2)}")
        print(f"     Common use case: {hint_info.get('common_use_case', 'N/A')}")


async def demo_category_tools():
    """Demonstrate the category-based tools for IDE integration."""
    print("\n=== Category Tools Demonstration ===\n")
    
    generator = IDEMetadataGenerator()
    
    # Generate category tools
    category_tools = generator.generate_category_tools()
    
    # Print number of category tools
    print(f"Generated {len(category_tools)} category tools")
    
    # Print some category tool definitions
    print("\nSample Category Tool Definitions:")
    for i, (ns_name, tool_def) in enumerate(list(category_tools.items())[:2]):
        print(f"\n  {i+1}. {tool_def['name']}:")
        print(f"     Description: {tool_def['description']}")
        print(f"     Function count: {tool_def['function_count']}")
        print(f"     Functions: {', '.join(tool_def['function_names'][:5])}{'...' if len(tool_def['function_names']) > 5 else ''}")
        
        # Print command parameter
        if 'command' in tool_def['parameters']:
            command_param = tool_def['parameters']['command']
            print(f"     Command parameter: {command_param['description']}")
            print(f"     Available commands: {len(command_param['enum'])}")


async def demo_optimized_tools_export():
    """Demonstrate exporting IDE-optimized tools."""
    print("\n=== IDE-Optimized Tools Export Demonstration ===\n")
    
    # Generate and export tools
    export_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "ide_optimized_tools.json")
    result_path = export_ide_optimized_tools(export_path)
    
    print(f"Exported IDE-optimized tools to: {result_path}")
    
    # Read the exported file
    with open(result_path, 'r') as f:
        tools = json.load(f)
    
    # Print summary
    print(f"\nExported file contains:")
    print(f"  {len(tools['category_tools'])} category tools")
    print(f"  {tools['documentation']['meta']['total_functions']} functions")
    print(f"  {len(tools['hints'])} function hints")
    
    print("\nFile structure:")
    for section, content in tools.items():
        print(f"  {section}: {type(content).__name__}")


async def demo_ide_integration():
    """Run the complete IDE integration demonstration."""
    print("\n=== IDE Integration Demonstration ===\n")
    
    # First, migrate some real tools to have a meaningful registry
    print("Migrating tools to registry...")
    migrate_all_tools()
    
    # Show each feature in action
    await demo_tool_documentation()
    await demo_tool_hints()
    await demo_category_tools()
    await demo_optimized_tools_export()
    
    print("\n=== IDE Integration Complete ===\n")
    
    # Show conclusion
    print("The IDE Integration features demonstrate how to:")
    print("1. Generate IDE-friendly documentation for tools")
    print("2. Create parameter examples and hints for better discoverability")
    print("3. Define category-based meta-tools for IDE integration")
    print("4. Export optimized tool definitions for use in IDEs")
    print("\nThese features help reduce the number of exposed tools while")
    print("maintaining full functionality and good documentation.")


if __name__ == "__main__":
    # Run the demo
    asyncio.run(demo_ide_integration()) 