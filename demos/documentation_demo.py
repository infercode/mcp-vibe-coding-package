#!/usr/bin/env python3
"""
Documentation Generator Demonstration

This script demonstrates the Documentation Generator feature of the Function Registry Pattern,
showing how to generate comprehensive documentation for registered functions.
"""

import asyncio
import os
import json
import sys
from typing import Dict, Any

# Add the parent directory to the path so imports work
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from src.registry.documentation_generator import (
    DocumentationGenerator, 
    generate_documentation,
    export_documentation_markdown,
    export_documentation_json
)
from src.registry.registry_manager import get_registry
import src.registry.example_functions  # Import to register example functions

# Import the migration framework to register real tools
from src.registry.migration_framework import migrate_all_tools

# Constants
OUTPUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "generated_docs")
JSON_PATH = os.path.join(OUTPUT_DIR, "registry_docs.json")
MARKDOWN_DIR = os.path.join(OUTPUT_DIR, "markdown")
VERSION = "1.0.0"


async def demo_documentation_generation():
    """Demonstrate the documentation generation functionality."""
    print("\n=== Documentation Generation Demonstration ===\n")
    
    # Create generator instance
    generator = DocumentationGenerator(VERSION)
    
    # Generate documentation
    docs = generator.generate_documentation()
    
    # Print summary
    print(f"Generated documentation version {docs['meta']['version']}")
    print(f"Total functions: {docs['meta']['function_count']}")
    print(f"Total namespaces: {docs['meta']['namespace_count']}")
    
    # Print some namespace information
    print("\nNamespaces:")
    for ns_name, ns_info in list(docs["namespaces"].items())[:3]:
        print(f"  {ns_name}: {ns_info['function_count']} functions")
        print(f"  Description: {ns_info.get('description', 'No description')}")
        
        # Show categorization if available
        if "categories" in ns_info:
            categories = list(ns_info["categories"].keys())
            print(f"  Categories: {', '.join(categories)}")
            
    # Print some function information
    print("\nSample Function Documentation:")
    for i, (func_name, func_info) in enumerate(list(docs["functions"].items())[:2]):
        print(f"\n  {i+1}. {func_name}:")
        print(f"     Description: {func_info.get('short_description', 'No description')}")
        print(f"     Return type: {func_info.get('return_info', {}).get('type', 'unknown')}")
        print(f"     Parameters: {len(func_info.get('parameters', {}))}")
        
        # Show examples if available
        if "examples" in func_info:
            print(f"     Examples: {len(func_info['examples'])}")


async def demo_markdown_export():
    """Demonstrate exporting documentation to Markdown."""
    print("\n=== Markdown Export Demonstration ===\n")
    
    # Create directory if it doesn't exist
    os.makedirs(MARKDOWN_DIR, exist_ok=True)
    
    # Export to Markdown
    index_path = export_documentation_markdown(MARKDOWN_DIR, VERSION)
    
    print(f"Exported Markdown documentation to: {index_path}")
    print("\nDirectory structure:")
    
    # List files in the Markdown directory
    for root, dirs, files in os.walk(MARKDOWN_DIR):
        level = root.replace(MARKDOWN_DIR, '').count(os.sep)
        indent = ' ' * 4 * level
        print(f"{indent}{os.path.basename(root)}/")
        sub_indent = ' ' * 4 * (level + 1)
        for file in files[:5]:  # Show only first 5 files to avoid clutter
            print(f"{sub_indent}{file}")
        if len(files) > 5:
            print(f"{sub_indent}... ({len(files) - 5} more files)")


async def demo_json_export():
    """Demonstrate exporting documentation to JSON."""
    print("\n=== JSON Export Demonstration ===\n")
    
    # Create directory if it doesn't exist
    os.makedirs(os.path.dirname(JSON_PATH), exist_ok=True)
    
    # Export to JSON
    json_path = export_documentation_json(JSON_PATH, VERSION)
    
    print(f"Exported JSON documentation to: {json_path}")
    
    # Read the JSON file and show structure
    with open(json_path, 'r') as f:
        data = json.load(f)
        
    print("\nJSON structure:")
    for section, content in data.items():
        if isinstance(content, dict):
            print(f"  {section}: dictionary with {len(content)} items")
        else:
            print(f"  {section}: {type(content).__name__}")


async def demo_versioning():
    """Demonstrate the versioning capabilities."""
    print("\n=== Documentation Versioning Demonstration ===\n")
    
    # Create generator instance
    generator = DocumentationGenerator("1.1.0")
    
    # Record some changes
    generator.record_function_change(
        "memory.create_entity", 
        "1.1.0", 
        "Added support for additional entity properties",
        "2025-04-10"
    )
    
    generator.record_function_change(
        "memory.get_entity", 
        "1.1.0", 
        "Improved error handling for missing entities",
        "2025-04-10"
    )
    
    # Generate documentation with history
    docs = generator.generate_documentation(include_history=True)
    
    # Show version history for some functions
    print("Version history for functions:")
    for func_name, func_info in docs["functions"].items():
        if "history" in func_info and len(func_info["history"]) > 1:
            print(f"\n  {func_name} history:")
            for entry in func_info["history"]:
                print(f"    {entry['version']} ({entry['date']}): {entry['changes']}")


async def demo_documentation():
    """Run the complete documentation demonstration."""
    print("\n=== Documentation Generator Demonstration ===\n")
    
    # First, migrate some real tools to have a meaningful registry
    print("Migrating tools to registry...")
    migrate_all_tools()
    
    # Show each feature in action
    await demo_documentation_generation()
    await demo_markdown_export()
    await demo_json_export()
    await demo_versioning()
    
    print("\n=== Documentation Generator Complete ===\n")
    
    # Show conclusion
    print("The Documentation Generator features demonstrate how to:")
    print("1. Auto-generate comprehensive documentation from function metadata")
    print("2. Create usage examples for functions")
    print("3. Export documentation to Markdown and JSON formats")
    print("4. Track version history for function documentation")
    print("\nThese features help maintain up-to-date documentation")
    print("and improve the discoverability of registry functions.")


if __name__ == "__main__":
    # Run the demo
    asyncio.run(demo_documentation()) 