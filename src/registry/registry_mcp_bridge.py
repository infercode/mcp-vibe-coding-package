#!/usr/bin/env python3
"""
Registry MCP Bridge

This module provides a bridge between the MCP server's tool registration
and the Function Registry Pattern, ensuring all tools are automatically
registered with the registry with complete metadata.
"""

import inspect
import logging
from typing import Any, Dict, Optional, Callable, List, Type, Union, get_type_hints
import importlib
import importlib.util
import sys
import os
from types import ModuleType

from mcp.server.fastmcp import FastMCP
from mcp.server.lowlevel import Server

from src.registry.registry_manager import get_registry, register_function
from src.registry.function_models import FunctionMetadata, FunctionParameters
from src.logger import get_logger

logger = get_logger()

def enhance_server(server: FastMCP) -> FastMCP:
    """
    Enhance an MCP server with automatic registry integration.
    
    This function replaces the server's tool decorator with an enhanced version
    that automatically registers all tools with the Function Registry Pattern.
    
    Args:
        server: The MCP server instance to enhance
        
    Returns:
        The enhanced server instance
    """
    logger.info("Enhancing MCP server with registry bridge")
    
    # Store a reference to the original tool decorator
    original_tool_decorator = server.tool
    
    # Track all registered tools for debugging
    registered_tools = []
    
    # Register any existing tools that are already in the server's tool collection
    registry = get_registry()
    existing_tools_registered = 0
    
    # Try different attributes where MCP might store tools
    tool_attrs = ["_tools", "tools", "_tool_manager", "_tool_registry"]
    server_tools = None
    
    for attr in tool_attrs:
        if hasattr(server, attr):
            server_tools = getattr(server, attr)
            # If it's a manager or registry object, try to get its tools attribute
            if hasattr(server_tools, "tools"):
                server_tools = server_tools.tools
            break
    
    if server_tools and isinstance(server_tools, dict):
        logger.info(f"Found {len(server_tools)} existing tools in server")
        
        for tool_name, tool_func in server_tools.items():
            try:
                # Extract function info
                module_name = tool_func.__module__ if hasattr(tool_func, "__module__") else "__main__"
                func_name = tool_func.__name__ if hasattr(tool_func, "__name__") else tool_name
                
                # Determine namespace from module
                namespace_parts = module_name.split('.')
                if len(namespace_parts) > 0:
                    namespace = namespace_parts[-1]
                else:
                    namespace = "__main__"
                
                # Register with registry
                full_name = f"{namespace}.{func_name}"
                
                # Check if already registered
                if registry.get_function_metadata(full_name) is None:
                    register_function(namespace, func_name)(tool_func)
                    existing_tools_registered += 1
                    logger.info(f"Registered existing server tool: {full_name}")
                    
                    # Track for debugging
                    registered_tools.append({
                        "name": full_name,
                        "original_module": module_name
                    })
            except Exception as e:
                logger.error(f"Error registering existing server tool {tool_name}: {str(e)}")
    
    if existing_tools_registered > 0:
        logger.info(f"Registered {existing_tools_registered} existing tools from server")

    # Define the enhanced decorator
    def enhanced_tool_decorator(*args, **kwargs):
        """
        Enhanced version of server.tool() that also registers with the registry.
        
        All arguments are passed through to the original decorator.
        """
        # Get the original decorator
        original_decorator = original_tool_decorator(*args, **kwargs)
        
        # Define the wrapper function
        def wrapper(func):
            # First, let MCP do its normal decoration
            decorated_func = original_decorator(func)
            
            # Now register with the registry
            try:
                # Extract namespace from module path
                module_name = func.__module__
                namespace_parts = module_name.split('.')
                
                # Try to determine appropriate namespace
                if len(namespace_parts) >= 2 and namespace_parts[-2] == "registry":
                    # If in registry.X module, use X as namespace
                    namespace = namespace_parts[-1]
                elif len(namespace_parts) >= 2 and namespace_parts[-2] == "tools":
                    # If in tools.X module, use X as namespace
                    namespace = namespace_parts[-1]
                    # Strip _tools suffix if present
                    if namespace.endswith("_tools"):
                        namespace = namespace[:-6]
                    # Convert common prefixes
                    if namespace.startswith("core_"):
                        namespace = "memory"
                else:
                    # Default to the last part of the module path
                    namespace = namespace_parts[-1]
                
                # Extract function name
                func_name = func.__name__
                
                # Log the registration
                logger.info(f"Auto-registering tool {func_name} with namespace {namespace}")
                
                # Get or create registry
                registry = get_registry()
                
                # Apply the register_function decorator
                register_function(namespace, func_name)(func)
                
                # Track for debugging
                registered_tools.append({
                    "name": f"{namespace}.{func_name}",
                    "original_module": func.__module__
                })
                
                logger.debug(f"Successfully registered {func_name} with registry")
            except Exception as e:
                logger.error(f"Error registering {func.__name__} with registry: {str(e)}")
            
            # Return the decorated function (from MCP)
            return decorated_func
        
        return wrapper
    
    # Replace the server's tool decorator with our enhanced version
    server.tool = enhanced_tool_decorator
    
    # Store the registered tools on the server for debugging
    setattr(server, "_registry_bridge_tools", registered_tools)
    
    logger.info("MCP server enhanced with registry bridge")
    return server

def scan_and_register_existing_tools(module_paths=None):
    """
    Scan for and register existing tools that were already decorated with @server.tool().
    
    This function is used to find tools in modules that were decorated before
    the registry bridge was applied. It imports the modules and manually registers
    any functions that appear to be MCP tools.
    
    Args:
        module_paths: Optional list of module paths to scan. If None, uses default paths.
        
    Returns:
        Dict with registration statistics
    """
    if module_paths is None:
        # Default module paths to scan for tools
        module_paths = [
            'src.tools.core_memory_tools',
            'src.tools.project_memory_tools', 
            'src.tools.lesson_memory_tools',
            'src.tools.config_tools',
            'src.registry.registry_tools',
            'src.registry.core_registry_tools',
            'src.registry.consolidated_tools'
        ]
        
    logger.info(f"Scanning for existing tools in {len(module_paths)} modules")
    
    registry = get_registry()
    stats = {
        "modules_scanned": 0,
        "tools_found": 0,
        "tools_registered": 0,
        "errors": 0
    }
    
    for module_path in module_paths:
        try:
            # Import the module
            module = importlib.import_module(module_path)
            stats["modules_scanned"] += 1
            
            # Determine namespace from module path
            namespace_parts = module_path.split('.')
            
            # Try to determine appropriate namespace
            if len(namespace_parts) >= 2 and namespace_parts[-2] == "registry":
                # If in registry.X module, use X as namespace
                namespace = namespace_parts[-1]
            elif len(namespace_parts) >= 2 and namespace_parts[-2] == "tools":
                # If in tools.X module, use X as namespace
                namespace = namespace_parts[-1]
                # Strip _tools suffix if present
                if namespace.endswith("_tools"):
                    namespace = namespace[:-6]
                # Convert common prefixes
                if namespace.startswith("core_"):
                    namespace = "memory"
            else:
                # Default to the last part of the module path
                namespace = namespace_parts[-1]
                
            logger.info(f"Scanning module {module_path} (namespace: {namespace})")
            
            # Find all module members that appear to be decorated tools
            for name, obj in inspect.getmembers(module):
                # Look for functions that might be tools
                if callable(obj) and hasattr(obj, "__module__") and obj.__module__ == module.__name__:
                    # Several ways to detect if it's likely a tool:
                    is_tool = False
                    
                    # 1. Check if it's a coroutine function (most tools are async)
                    if inspect.iscoroutinefunction(obj):
                        is_tool = True
                    
                    # 2. Check for common MCP tool attributes
                    if hasattr(obj, "_mcp_tool") or hasattr(obj, "_is_tool") or hasattr(obj, "tool_metadata"):
                        is_tool = True
                        
                    # 3. Check name patterns typical for tools
                    if name.endswith("_tool") or name.startswith("tool_"):
                        is_tool = True
                    
                    # Check the function's signature for parameters
                    try:
                        sig = inspect.signature(obj)
                        # Most tools have parameters
                        if len(sig.parameters) > 0:
                            is_tool = True
                    except (ValueError, TypeError):
                        # If we can't get the signature, skip this check
                        pass
                        
                    if is_tool:
                        stats["tools_found"] += 1
                        
                        try:
                            # Check if this function is already registered
                            full_name = f"{namespace}.{name}"
                            existing = registry.get_function_metadata(full_name)
                            
                            if existing is None:
                                # Register the function with our registry
                                register_function(namespace, name)(obj)
                                logger.info(f"Registered existing tool: {namespace}.{name}")
                                stats["tools_registered"] += 1
                            else:
                                logger.debug(f"Tool already registered: {namespace}.{name}")
                        except Exception as e:
                            logger.error(f"Error registering existing tool {namespace}.{name}: {str(e)}")
                            stats["errors"] += 1
                        
        except ImportError as e:
            logger.error(f"Could not import module {module_path}: {e}")
            stats["errors"] += 1
        except Exception as e:
            logger.error(f"Error scanning module {module_path}: {e}")
            stats["errors"] += 1
    
    logger.info(f"Completed tool scanning: {stats['tools_registered']} tools registered from {stats['modules_scanned']} modules")
    return stats

def list_registered_tools(server: FastMCP) -> List[Dict[str, str]]:
    """
    List all tools that were registered through the bridge.
    
    Args:
        server: The enhanced server instance
        
    Returns:
        List of registered tool information
    """
    if hasattr(server, "_registry_bridge_tools"):
        return getattr(server, "_registry_bridge_tools")
    return []

def get_registry_stats() -> Dict[str, Any]:
    """
    Get statistics about the registry after bridge integration.
    
    Returns:
        Dictionary with registry statistics
    """
    registry = get_registry()
    functions = registry.get_all_functions()
    namespaces = registry.get_namespaces()
    
    # Count functions per namespace
    namespace_counts = {}
    for ns in namespaces:
        ns_functions = registry.get_functions_by_namespace(ns)
        namespace_counts[ns] = len(ns_functions)
    
    return {
        "total_functions": len(functions),
        "namespaces": list(namespaces),
        "namespace_counts": namespace_counts
    }

def scan_and_register_dir(directory_path, package_prefix="src", skip_init=True):
    """
    Scan a directory for Python modules and register all functions that look like tools.
    
    This is a more aggressive tool finder that will try to find and register
    all callable async functions in a directory.
    
    Args:
        directory_path: Path to the directory to scan
        package_prefix: Package prefix for imports
        skip_init: Whether to skip __init__.py files
        
    Returns:
        Dict with scanning statistics
    """
    if not os.path.exists(directory_path) or not os.path.isdir(directory_path):
        logger.error(f"Directory does not exist: {directory_path}")
        return {"error": "Directory not found", "modules_scanned": 0, "tools_registered": 0}
    
    logger.info(f"Aggressive scanning of directory: {directory_path}")
    
    stats = {
        "modules_scanned": 0,
        "tools_found": 0,
        "tools_registered": 0,
        "errors": 0
    }
    
    registry = get_registry()
    
    # Walk through the directory
    for root, dirs, files in os.walk(directory_path):
        # Skip __pycache__ directories
        if "__pycache__" in root:
            continue
            
        for file in files:
            # Skip non-Python files
            if not file.endswith(".py"):
                continue
                
            # Skip __init__.py if requested
            if skip_init and file == "__init__.py":
                continue
                
            # Get full file path
            file_path = os.path.join(root, file)
            
            try:
                # Calculate relative path and module name
                rel_path = os.path.relpath(file_path, os.path.dirname(directory_path))
                module_name = os.path.splitext(rel_path)[0].replace(os.sep, ".")
                full_module_name = f"{package_prefix}.{module_name}"
                
                # Try to import the module
                spec = importlib.util.spec_from_file_location(full_module_name, file_path)
                if spec and spec.loader:
                    module = importlib.util.module_from_spec(spec)
                    spec.loader.exec_module(module)
                    stats["modules_scanned"] += 1
                    
                    # Calculate reasonable namespace from module path
                    path_parts = module_name.split('.')
                    
                    # Try to get sensible namespace from path
                    if len(path_parts) > 0:
                        namespace = path_parts[-1]
                        # Handle common patterns
                        if namespace.endswith("_tools"):
                            namespace = namespace[:-6]
                        if namespace.startswith("core_"):
                            namespace = "memory"
                    else:
                        namespace = "tools"
                        
                    logger.info(f"Scanning module {full_module_name} (namespace: {namespace})")
                    
                    # Find all async functions in the module
                    for name, obj in inspect.getmembers(module):
                        if callable(obj) and hasattr(obj, "__module__") and obj.__module__ == full_module_name:
                            # Is this likely to be a tool?
                            is_tool = False
                            
                            # Most common indicator: it's an async function
                            if inspect.iscoroutinefunction(obj):
                                is_tool = True
                                
                            # Check for tool name patterns
                            if name.endswith("_tool") or name.startswith("tool_"):
                                is_tool = True
                                
                            # Check for common tool decorator attributes
                            if hasattr(obj, "_mcp_tool") or hasattr(obj, "_is_tool") or hasattr(obj, "tool_metadata"):
                                is_tool = True
                                
                            if is_tool:
                                stats["tools_found"] += 1
                                
                                # Get full name for registry
                                full_name = f"{namespace}.{name}"
                                
                                # Check if already registered
                                if registry.get_function_metadata(full_name) is None:
                                    try:
                                        register_function(namespace, name)(obj)
                                        logger.info(f"Registered tool from directory scan: {full_name}")
                                        stats["tools_registered"] += 1
                                    except Exception as e:
                                        logger.error(f"Error registering tool {full_name}: {str(e)}")
                                        stats["errors"] += 1
                                else:
                                    logger.debug(f"Tool already registered: {full_name}")
            except Exception as e:
                logger.error(f"Error processing file {file_path}: {str(e)}")
                stats["errors"] += 1
                
    logger.info(f"Directory scan complete: {stats['tools_registered']} tools registered from {stats['modules_scanned']} modules")
    return stats

def register_tool_modules(server=None):
    """
    Register all tool modules by calling their registration functions.
    
    This function takes care of calling the various registration functions
    that would normally be called during server startup, providing them
    with mock server and client manager objects if needed.
    
    Args:
        server: Optional server instance to use. If None, creates a mock server.
        
    Returns:
        Statistics about the registration process
    """
    from src.registry import get_registry
    logger.info("Registering tool modules with the registry")
    
    # Statistics for the registration process
    stats = {
        "modules_found": 0,
        "registration_functions_found": 0,
        "registration_functions_called": 0,
        "tools_registered": 0,
        "errors": 0
    }
    
    # Create a mock server if none is provided
    mock_server = server
    if mock_server is None:
        # Create a minimal mock server with a tool method
        class MockServer:
            def __init__(self):
                self.registered_tools = {}
                
            def tool(self, *args, **kwargs):
                """Mock implementation of server.tool() decorator"""
                def decorator(func):
                    # Track the registration
                    name = func.__name__
                    self.registered_tools[name] = func
                    logger.debug(f"Registered tool {name} with mock server")
                    
                    # Get namespace from module path
                    module_name = func.__module__
                    namespace_parts = module_name.split('.')
                    
                    # Try to determine appropriate namespace
                    if len(namespace_parts) >= 2 and namespace_parts[-2] == "registry":
                        # If in registry.X module, use X as namespace
                        namespace = namespace_parts[-1]
                    elif len(namespace_parts) >= 2 and namespace_parts[-2] == "tools":
                        # If in tools.X module, use X as namespace
                        namespace = namespace_parts[-1]
                        # Strip _tools suffix if present
                        if namespace.endswith("_tools"):
                            namespace = namespace[:-6]
                        # Convert common prefixes
                        if namespace.startswith("core_"):
                            namespace = "memory"
                    else:
                        # Default to the last part of the module path
                        namespace = namespace_parts[-1]
                    
                    # Register the function with our registry
                    try:
                        register_function(namespace, name)(func)
                        stats["tools_registered"] += 1
                        logger.info(f"Registered tool {namespace}.{name} with registry")
                    except Exception as e:
                        logger.error(f"Error registering {name}: {str(e)}")
                        stats["errors"] += 1
                    
                    # Return the original function (we're not really decorating it)
                    return func
                
                # Handle both @server.tool and @server.tool()
                if len(args) == 1 and callable(args[0]):
                    # Called as @server.tool
                    return decorator(args[0])
                else:
                    # Called as @server.tool()
                    return decorator
        
        mock_server = MockServer()
        logger.info("Created mock server for tool registration")
    
    # Create a mock client manager getter
    def get_mock_client_manager(client_id=None):
        """Mock implementation of get_client_manager"""
        # This would normally return a GraphMemoryManager for the client
        # but we'll just return a mock object with the methods we need
        class MockClientManager:
            def __init__(self):
                self.client_id = client_id or "default-client"
                
            def get_client_project_memory(self):
                return self
                
            def get_client_lesson_memory(self):
                return self
                
            def create_project_container(self, *args, **kwargs):
                return {"status": "success", "message": "Mock project container created"}
                
            def get_project_container(self, *args, **kwargs):
                return {"status": "success", "message": "Mock project container retrieved"}
                
            def create_lesson_container(self, *args, **kwargs):
                return {"status": "success", "message": "Mock lesson container created"}
                
            def get_lesson_container(self, *args, **kwargs):
                return {"status": "success", "message": "Mock lesson container retrieved"}
        
        return MockClientManager()
    
    # Find all registration functions
    import src.tools as tools_module
    
    # First, try to get the registration functions directly from the tools module
    registration_funcs = []
    for name, func in inspect.getmembers(tools_module, inspect.isfunction):
        if name.startswith("register_") and name != "register_all_tools":
            registration_funcs.append((name, func))
            stats["registration_functions_found"] += 1
    
    # Also try to import individual tool modules
    try:
        tool_modules_dir = os.path.dirname(tools_module.__file__)
        for filename in os.listdir(tool_modules_dir):
            if filename.endswith("_tools.py") and not filename.startswith("__"):
                module_name = f"src.tools.{filename[:-3]}"
                try:
                    module = importlib.import_module(module_name)
                    stats["modules_found"] += 1
                    
                    # Look for register_X_tools functions
                    for name, func in inspect.getmembers(module, inspect.isfunction):
                        if name.startswith("register_") and callable(func):
                            # Check if it's already in our list
                            if not any(rf[0] == name for rf in registration_funcs):
                                registration_funcs.append((name, func))
                                stats["registration_functions_found"] += 1
                except ImportError as e:
                    logger.error(f"Could not import module {module_name}: {e}")
                    stats["errors"] += 1
    except Exception as e:
        logger.error(f"Error scanning tools directory: {e}")
        stats["errors"] += 1
    
    # Call each registration function
    for name, func in registration_funcs:
        try:
            # Call the registration function with our mock server and client manager
            func(mock_server, get_mock_client_manager)
            stats["registration_functions_called"] += 1
            logger.info(f"Called registration function {name}")
        except Exception as e:
            logger.error(f"Error calling registration function {name}: {e}")
            stats["errors"] += 1
    
    # Get final registry stats
    final_registry_stats = get_registry_stats()
    
    logger.info(f"Tool module registration complete. {stats['tools_registered']} tools registered")
    logger.info(f"Registry now contains {final_registry_stats['total_functions']} functions in {len(final_registry_stats['namespace_counts'])} namespaces")
    
    return stats 