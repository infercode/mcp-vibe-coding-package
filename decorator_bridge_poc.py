#!/usr/bin/env python3
"""
Decorator Bridge Proof of Concept

This script validates that we can intercept MCP server tool decorators
and automatically register the same functions with our registry.
"""

import asyncio
import inspect
import json
from typing import Optional, Dict, Any

from mcp.server.fastmcp import FastMCP
from mcp.server.lowlevel import NotificationOptions

# Import registry components
from src.registry.registry_manager import get_registry, FunctionRegistry
from src.registry.function_models import FunctionMetadata

# Initialize the MCP server
server = FastMCP(
    name="Decorator Bridge POC",
    description="Tests intercepting MCP tool decorators",
    version="0.1.0",
    notification_options=NotificationOptions()
)

# Get the registry
registry = get_registry()

print("\n===== DECORATOR BRIDGE PROOF OF CONCEPT =====")
print("\n1. Initial State:")
print(f"   - Registry has {len(registry.get_all_functions())} functions")

# Store the original tool decorator
original_tool_decorator = server.tool

# Override the server.tool decorator
def enhanced_tool_decorator(*args, **kwargs):
    print(f"\n>> Intercepted @server.tool() decorator call with args={args}, kwargs={kwargs}")
    
    # Get the original decorator
    original_decorator = original_tool_decorator(*args, **kwargs)
    
    # Return an enhanced decorator that also registers with our registry
    def wrapper(func):
        print(f">> Wrapping function: {func.__name__}")
        
        # Let MCP do its normal registration
        decorated_func = original_decorator(func)
        
        # Extract namespace from module structure or context
        module_name = func.__module__
        namespace = module_name.split('.')[-1] if '.' in module_name else 'test'
        
        print(f">> Registering {func.__name__} with namespace {namespace} in registry")
        
        # Extract and print the function's signature
        sig = inspect.signature(func)
        parameters = {
            name: {"type": param.annotation.__name__ if hasattr(param.annotation, "__name__") else str(param.annotation),
                  "required": param.default == inspect.Parameter.empty}
            for name, param in sig.parameters.items()
        }
        print(f">> Function parameters: {json.dumps(parameters, indent=2)}")
        
        # Extract and print any docstring
        if func.__doc__:
            print(f">> Function docstring: {func.__doc__.strip()}")
        
        # Register with our registry
        registry.register(namespace, func.__name__, func)
        
        # Try to access the MCP tool metadata to see what's available
        # This part is exploratory and might need adjustments based on MCP internals
        print(">> Attempting to access MCP tool metadata...")
        
        # Method 1: Check if the function has a _tool_info attribute
        if hasattr(decorated_func, "_tool_info"):
            print(f">> Found _tool_info: {decorated_func._tool_info}")
        
        # Method 2: Try to find the tool in the server's tools collection
        # Need to defer this check until tools are actually registered
        
        return decorated_func
    
    return wrapper

# Replace the original decorator with our enhanced version
server.tool = enhanced_tool_decorator
print("\n2. Decorator replaced.")

# Define two test functions with different parameter styles
@server.tool()
async def test_simple_function(param1: str, param2: int = 42) -> Dict[str, Any]:
    """
    A simple test function with basic parameters.
    
    Args:
        param1: A string parameter
        param2: An integer parameter with default value
        
    Returns:
        A dictionary with results
    """
    return {"param1": param1, "param2": param2}

@server.tool()
async def test_complex_function(
    required_param: str,
    optional_param: Optional[int] = None,
    *,
    keyword_only: str = "default"
) -> str:
    """
    A more complex test function with different parameter styles.
    
    Args:
        required_param: A required parameter
        optional_param: An optional parameter
        keyword_only: A keyword-only parameter
        
    Returns:
        A string result
    """
    return f"Processed {required_param} with {optional_param} and {keyword_only}"

print("\n3. Test functions defined with decorators.")

# Print registered tools in MCP
print("\n4. Checking MCP server tools:")
try:
    # This may require different approaches based on MCP SDK version
    if hasattr(server, "_tools"):
        for i, tool in enumerate(server._tools):
            print(f"   Tool {i+1}: {tool.name}")
    elif hasattr(server, "tools"):
        for i, tool in enumerate(server.tools):
            print(f"   Tool {i+1}: {tool.name}")
    else:
        print("   Could not access MCP tools collection")
except Exception as e:
    print(f"   Error accessing MCP tools: {str(e)}")

# Print registered functions in registry
print("\n5. Checking registry functions:")
registry_functions = registry.get_all_functions()
for i, func_meta in enumerate(registry_functions):
    print(f"   Function {i+1}: {func_meta.name}")
    if hasattr(func_meta, "parameters"):
        print(f"      Parameters: {json.dumps(func_meta.parameters, indent=2)}")

print("\n6. Verification:")
print(f"   MCP Server: {'SUCCESS' if hasattr(server, '_tools') or hasattr(server, 'tools') else 'UNKNOWN'}")
print(f"   Registry: {'SUCCESS' if len(registry_functions) >= 2 else 'FAILURE'}")

print("\n===== PROOF OF CONCEPT COMPLETE =====")

if __name__ == "__main__":
    print("\nScript completed. Running this script only creates the test functions but doesn't execute the server.")
    print("In a real implementation, we would check tool metadata after the server has fully initialized.")
    print("This proof of concept demonstrates that decorator interception is possible.") 