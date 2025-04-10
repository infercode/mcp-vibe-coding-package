#!/usr/bin/env python3
"""
Tool Migration Framework Demonstration

This script demonstrates the Tool Migration Framework in action,
showing how to analyze, migrate, and use tools through the registry.
"""

import asyncio
import json
import os
import sys
from typing import Dict, Any, List

# Add the parent directory to the path so imports work
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from src.registry.migration_framework import ToolAnalyzer, MigrationManager, migrate_all_tools
from src.registry.registry_manager import get_registry
import src.registry.example_functions  # Import to register example functions


async def demo_tool_analyzer():
    """Demonstrate the tool analyzer."""
    print("\n=== Tool Analyzer Demonstration ===\n")
    
    analyzer = ToolAnalyzer()
    
    # Analyze a core memory tools module
    module_path = 'src.tools.core_memory_tools'
    print(f"Analyzing module: {module_path}")
    
    analysis = analyzer.analyze_module(module_path)
    
    # Print summary
    print(f"\nModule name: {analysis.get('module_name')}")
    print(f"Registration function: {analysis.get('register_function')}")
    print(f"Found {len(analysis.get('functions', {}))} tool functions:")
    
    # Print function summary
    for func_name, func_info in analysis.get('functions', {}).items():
        print(f"\n  {func_name}:")
        print(f"    Description: {func_info.get('description', 'No description')}")
        print(f"    Parameters: {', '.join(func_info.get('parameters', {}).keys())}")
        print(f"    Return type: {func_info.get('return_type', 'unknown')}")
        print(f"    Is async: {func_info.get('is_async', False)}")


async def demo_migration_manager():
    """Demonstrate the migration manager."""
    print("\n=== Migration Manager Demonstration ===\n")
    
    manager = MigrationManager()
    
    # Migrate a specific module
    module_path = 'src.tools.core_memory_tools'
    print(f"Migrating module: {module_path}")
    
    result = manager.migrate_module(module_path)
    
    # Print migration result
    print(f"\nMigration status: {result.get('status')}")
    print(f"Namespace: {result.get('namespace')}")
    print(f"Registered functions: {len(result.get('registered_functions', []))}")
    
    # Print some registered functions
    print("\nRegistered functions:")
    for func in result.get('registered_functions', [])[:5]:  # Show first 5
        print(f"  {func}")
        
    # If more than 5 functions, show count of remaining
    if len(result.get('registered_functions', [])) > 5:
        remaining = len(result.get('registered_functions', [])) - 5
        print(f"  ...and {remaining} more")


async def demo_registry_access():
    """Demonstrate access to migrated tools through the registry."""
    print("\n=== Registry Access Demonstration ===\n")
    
    registry = get_registry()
    
    # List namespaces
    namespaces = registry.get_namespaces()
    print(f"Registry namespaces: {namespaces}")
    
    # If 'memory' namespace exists, show functions in it
    if 'memory' in namespaces:
        memory_functions = registry.get_functions_by_namespace('memory')
        print(f"\nFunctions in 'memory' namespace: {len(memory_functions)}")
        
        # Show first 5 functions
        for i, func in enumerate(memory_functions[:5]):
            print(f"  {i+1}. {func.name}: {func.description}")
            
        # If we have functions, try executing one
        if memory_functions:
            func_name = memory_functions[0].name
            print(f"\nExecuting first function: {func_name}")
            
            # For demonstration, just use empty parameters
            # In a real case, we'd need to provide proper parameters
            try:
                result = await registry.execute(func_name)
                print(f"Result: {result.to_json()}")
            except Exception as e:
                print(f"Error executing function: {str(e)}")


async def demo_migration():
    """Run the complete migration demonstration."""
    print("\n=== Tool Migration Framework Demonstration ===\n")
    
    # Show each component in action
    await demo_tool_analyzer()
    await demo_migration_manager()
    await demo_registry_access()
    
    print("\n=== Migration Complete ===\n")
    
    # Show complete picture after migration
    registry = get_registry()
    namespaces = registry.get_namespaces()
    
    # Print summary statistics
    total_functions = sum(len(registry.get_functions_by_namespace(ns)) for ns in namespaces)
    print(f"Total namespaces: {len(namespaces)}")
    print(f"Total functions: {total_functions}")
    
    # Show breakdown by namespace
    print("\nFunctions by namespace:")
    for ns in namespaces:
        functions = registry.get_functions_by_namespace(ns)
        print(f"  {ns}: {len(functions)} functions")


if __name__ == "__main__":
    # Run the demo
    asyncio.run(demo_migration()) 