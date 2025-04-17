#!/usr/bin/env python3
"""
Function Registry Pattern Demo

This script demonstrates the Function Registry Pattern in action,
showing how to register functions, execute them, and list available functions.
"""

import asyncio
import json
import os
import sys
from typing import Dict, Any

# Add the parent directory to the path so imports work
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from src.registry.registry_manager import get_registry
from src.registry.function_models import FunctionResult
import src.registry.example_functions  # Import to register example functions

async def demo_registry():
    """Run a demonstration of the Function Registry Pattern."""
    registry = get_registry()
    
    print("\n=== Function Registry Pattern Demo ===\n")
    
    # List all registered namespaces
    namespaces = registry.get_namespaces()
    print(f"Registered namespaces: {namespaces}")
    
    # List functions by namespace
    for namespace in namespaces:
        functions = registry.get_functions_by_namespace(namespace)
        print(f"\nFunctions in namespace '{namespace}':")
        for func in functions:
            print(f"  - {func.name}: {func.description}")
    
    # Execute some example functions
    print("\n=== Executing Example Functions ===\n")
    
    # Example 1: Get entity
    print("Executing memory.get_entity:")
    result = await registry.execute("memory.get_entity", entity_name="TestEntity")
    print(f"Result: {result.to_json()}\n")
    
    # Example 2: Search entities
    print("Executing memory.search_entities:")
    result = await registry.execute("memory.search_entities", query="test query", limit=3)
    print(f"Result: {result.to_json()}\n")
    
    # Example 3: List projects
    print("Executing project.list_projects:")
    result = await registry.execute("project.list_projects")
    print(f"Result: {result.to_json()}\n")
    
    # Example 4: Format JSON
    print("Executing utils.format_json:")
    result = await registry.execute("utils.format_json", 
                                   data={"key": "value", "nested": {"inner": "data"}})
    print(f"Result: {result.to_json()}\n")
    
    # Example 5: Get detailed function metadata
    print("Metadata for memory.get_entity:")
    metadata = registry.get_function_metadata("memory.get_entity")
    print(json.dumps(metadata.dict(), indent=2))
    
    print("\n=== Demo Complete ===")

if __name__ == "__main__":
    asyncio.run(demo_registry()) 