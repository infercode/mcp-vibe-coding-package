#!/usr/bin/env python3
"""
Tool Migration Framework Demo

This script demonstrates the functionality of the Tool Migration Framework,
which allows migrating existing tools to the Function Registry Pattern.
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
RegistryFunctionResult = Any
MigrationManagerType = Any

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
        
        async def execute(self, function_name):
            return MockFunctionResult(False, {}, {"message": "Registry not available"})
    
    def get_registry():
        return MockRegistry()

try:
    from src.registry.function_models import FunctionResult
    function_models_available = True
except ImportError:
    print("Warning: Could not import function_models. Using mock implementation.")
    function_models_available = False
    
    # Mock function result
    class MockFunctionResult:
        def __init__(self, success=True, data=None, error=None):
            self.success = success
            self.data = data or {}
            self.error = error or {}
        
        def to_json(self):
            return json.dumps({"success": self.success, "data": self.data, "error": self.error})
    
    FunctionResult = MockFunctionResult

try:
    from src.registry.migration_framework import MigrationManager, migrate_all_tools
    migration_framework_available = True
except ImportError:
    print("Warning: Could not import migration_framework. Demo will show simulated outputs.")
    migration_framework_available = False
    
    # Mock migration manager
    class MockMigrationManager:
        def __init__(self):
            class MockAnalyzer:
                def analyze_module(self, path):
                    return {"functions": {}}
            
            self.analyzer = MockAnalyzer()
        
        def migrate_module(self, module_path, namespace):
            print(f"  [MOCK] Migration would migrate {module_path} to {namespace}")
            return {"migrated_functions": []}
    
    MigrationManager = MockMigrationManager
    
    def migrate_all_tools():
        return {"src.tools.mock_tools": {"migrated_functions": []}}

# Sample tool module paths for demonstration
SAMPLE_TOOL_MODULES = [
    "src.tools.core_memory_tools",
    "src.tools.messaging_tools",
    "src.tools.data_processing_tools"
]

async def demo_module_analysis():
    """Demonstrate analyzing a tool module."""
    print("\n=== Module Analysis Demo ===\n")
    
    manager = MigrationManager()
    
    for module_path in SAMPLE_TOOL_MODULES:
        print(f"Analyzing module: {module_path}")
        
        try:
            result = manager.analyzer.analyze_module(module_path)
            
            if not result:
                print(f"  No tool functions found in {module_path}")
                continue
                
            function_count = len(result.get("functions", {}))
            print(f"  Found {function_count} tool functions:")
            
            for func_name, func_info in result.get("functions", {}).items():
                print(f"    - {func_name}: {func_info.get('description', 'No description')}")
                param_count = len(func_info.get("parameters", {}))
                print(f"      Parameters: {param_count}")
                print(f"      Return type: {func_info.get('return_type', 'unknown')}")
                print(f"      Async: {'Yes' if func_info.get('is_async', False) else 'No'}")
                print()
        except Exception as e:
            print(f"  Error analyzing module: {str(e)}")

async def demo_module_migration():
    """Demonstrate migrating a tool module to the registry."""
    print("\n=== Module Migration Demo ===\n")
    
    manager = MigrationManager()
    registry = get_registry()
    
    for module_path in SAMPLE_TOOL_MODULES:
        print(f"Migrating module: {module_path}")
        
        try:
            # Determine namespace from module path
            parts = module_path.split(".")
            if len(parts) > 2:
                # Use the last part of the module path as namespace
                namespace = parts[-1].replace("_tools", "")
            else:
                namespace = "migrated"
            
            # Migrate the module
            result = manager.migrate_module(module_path, namespace)
            
            if not result:
                print(f"  Migration failed for {module_path}")
                continue
                
            migrated_count = len(result.get("migrated_functions", []))
            print(f"  Successfully migrated {migrated_count} functions to namespace '{namespace}'")
            
            for func_name in result.get("migrated_functions", []):
                print(f"    - {func_name}")
            
            # Show available functions after migration
            if registry:
                functions = registry.get_functions_by_namespace(namespace)
                print(f"\n  Functions available in registry (namespace '{namespace}'):")
                for func in functions:
                    print(f"    - {func.name}: {func.description}")
        except Exception as e:
            print(f"  Error migrating module: {str(e)}")

async def demo_backward_compatibility():
    """Demonstrate backward compatibility after migration."""
    print("\n=== Backward Compatibility Demo ===\n")
    
    # We'll use one module for this demo
    module_path = SAMPLE_TOOL_MODULES[0] if SAMPLE_TOOL_MODULES else "src.tools.core_memory_tools"
    
    # Import the module dynamically
    print(f"Testing backward compatibility for {module_path}")
    
    try:
        import importlib
        module = importlib.import_module(module_path)
        
        # Find a suitable function to test
        test_function = None
        function_name = ""
        
        # Look for a function that might work for our demo
        for name, obj in module.__dict__.items():
            if callable(obj) and name.startswith("get_") and not name.startswith("__"):
                test_function = obj
                function_name = name
                break
        
        if not test_function:
            print("  Could not find a suitable function to test")
            return
            
        print(f"  Testing function: {function_name}")
        
        # Call the original function
        try:
            if asyncio.iscoroutinefunction(test_function):
                result = await test_function()
            else:
                result = test_function()
                
            print(f"  Original function returned: {type(result)}")
            if hasattr(result, "__dict__") and hasattr(result, "to_json"):
                print(f"  Result data: {result.to_json()[:100]}...")
            elif isinstance(result, (dict, list)):
                print(f"  Result data: {json.dumps(result)[:100]}...")
            else:
                print(f"  Result data: {str(result)[:100]}...")
        except Exception as e:
            print(f"  Error calling original function: {str(e)}")
        
        # Now call via registry
        print("\n  Testing via registry:")
        
        # Determine the registry function name
        if module_path == "src.tools.core_memory_tools":
            namespace = "memory"
        elif module_path == "src.tools.messaging_tools":
            namespace = "messaging"
        else:
            namespace = "data"
            
        registry_function_name = f"{namespace}.{function_name}"
        
        # Call the function via registry
        try:
            registry = get_registry()
            result = await registry.execute(registry_function_name)
            
            print(f"  Registry function returned: {type(result)}")
            print(f"  Result data: {result.to_json()[:100]}...")
        except Exception as e:
            print(f"  Error calling via registry: {str(e)}")
    except ImportError:
        print(f"  Could not import module {module_path}")
    except Exception as e:
        print(f"  Error testing backward compatibility: {str(e)}")

async def demo_migrate_all_tools():
    """Demonstrate migrating all standard tools."""
    print("\n=== Migrate All Tools Demo ===\n")
    
    print("Migrating all standard tools...")
    
    try:
        results = migrate_all_tools()
        
        module_count = len(results)
        function_count = sum(len(result.get("migrated_functions", [])) for result in results.values())
        
        print(f"Successfully migrated {function_count} functions from {module_count} modules")
        
        for module_path, result in results.items():
            migrated_count = len(result.get("migrated_functions", []))
            print(f"  - {module_path}: {migrated_count} functions")
            
        # Show namespaces after migration
        registry = get_registry()
        if registry:
            namespaces = registry.get_namespaces()
            print("\nNamespaces available after migration:")
            for namespace in namespaces:
                functions = registry.get_functions_by_namespace(namespace)
                print(f"  - {namespace}: {len(functions)} functions")
    except Exception as e:
        print(f"Error migrating all tools: {str(e)}")

async def demo_migration_framework():
    """Run the complete Tool Migration Framework demonstration."""
    print("\n=== Tool Migration Framework Demonstration ===\n")
    
    # Run each demo in sequence
    await demo_module_analysis()
    await demo_module_migration()
    await demo_backward_compatibility()
    await demo_migrate_all_tools()
    
    print("\n=== Tool Migration Framework Demo Complete ===\n")
    
    # Show conclusion
    print("The Tool Migration Framework enables:")
    print("1. Analysis of existing tool modules to extract function information")
    print("2. Migration of tool functions to the registry with namespace organization")
    print("3. Maintaining backward compatibility with existing tool interfaces")
    print("4. Bulk migration of standard tool modules")
    print("\nThis allows for a gradual transition to the Function Registry Pattern")
    print("while maintaining compatibility with existing code.")

if __name__ == "__main__":
    asyncio.run(demo_migration_framework()) 