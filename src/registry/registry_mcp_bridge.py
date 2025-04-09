#!/usr/bin/env python3
"""
Registry MCP Bridge

This module provides a bridge between the MCP server's tool registration
and the Tool Registry Pattern, ensuring all tools are automatically
registered with the registry with complete metadata.
"""

import inspect
import logging
from typing import Any, Dict, Optional, Callable, List, Type, Union, get_type_hints, Set, Tuple, cast, TypeVar
import importlib
import importlib.util
import sys
import os
from types import ModuleType
import glob

from mcp.server.fastmcp import FastMCP
from mcp.server.lowlevel import Server

from src.registry.registry_manager import get_registry, register_tool, ToolRegistry
from src.registry.function_models import ToolMetadata, ToolParameters
from src.registry.docstring_parser import DocstringParser, parse_docstring
from src.logger import get_logger

logger = get_logger()

# Get the global registry instance
registry = get_registry()

# Type definitions to help linter
T = TypeVar('T', bound=Callable)

class EnhancedServer(Server):
    """Type hint for the Server with known tool attribute."""
    tool: Callable

def extract_tool_metadata(func: Callable) -> Dict[str, Any]:
    """
    Extract rich metadata from a tool, including docstring, parameters, and return type.
    
    Uses the DocstringParser to extract detailed parameter information, including nested
    dictionary structures, descriptions, and validation rules.
    
    Args:
        func: The function to extract metadata from
        
    Returns:
        A dictionary containing complete tool metadata
    """
    # Get function module and name
    module_name = func.__module__
    namespace = module_name.split('.')[-1]
    func_name = func.__name__
    
    # Get function signature
    sig = inspect.signature(func)
    
    # Get docstring and parse it using the DocstringParser
    docstring = inspect.getdoc(func) or ""
    parsed_docstring = DocstringParser.parse_docstring(docstring)
    
    # Extract parameter info with rich docstring data
    params = {}
    for name, param in sig.parameters.items():
        # Basic parameter info from signature
        param_info = {
            'name': name,
            'required': param.default is inspect.Parameter.empty,
            'type': str(param.annotation) if param.annotation is not inspect.Parameter.empty else 'Any',
        }
        
        # Enhance with docstring information if available
        if parsed_docstring.get('parameters') and name in parsed_docstring['parameters']:
            docstring_param = parsed_docstring['parameters'][name]
            param_info['description'] = docstring_param.get('description', '')
            
            # If parameter is a Dict type with nested fields, include those
            if 'fields' in docstring_param:
                param_info['fields'] = docstring_param['fields']
                
            # Add any validation info from docstring
            for validation_key in ['enum', 'default', 'min_value', 'max_value', 
                                  'min_length', 'max_length', 'pattern', 'format']:
                if validation_key in docstring_param:
                    param_info[validation_key] = docstring_param[validation_key]
        
        params[name] = param_info
    
    # Create metadata
    metadata = {
        'namespace': namespace,
        'name': func_name,
        'qualified_name': f"{namespace}.{func_name}",
        'description': parsed_docstring.get('description', docstring),
        'parameters': params,
        'return_type': str(sig.return_annotation) if sig.return_annotation is not inspect.Parameter.empty else 'Any',
        'source_file': inspect.getfile(func),
    }
    
    # Add examples if available
    if 'examples' in parsed_docstring and parsed_docstring['examples']:
        metadata['examples'] = parsed_docstring['examples']
        
    # Add return information if available
    if 'returns' in parsed_docstring and parsed_docstring['returns']:
        metadata['return_description'] = parsed_docstring['returns'].get('description', '')
    
    logger.debug(f"Extracted metadata for {namespace}.{func_name}: {len(params)} parameters")
    return metadata

def register_tool_with_registry(func: Callable, metadata: Dict[str, Any]) -> None:
    """Register a tool with the Tool Registry Pattern."""
    qualified_name = metadata['qualified_name']
    namespace = metadata['namespace']
    func_name = metadata['name']
    
    # Check if the tool is already registered
    if not registry.get_tool_metadata(qualified_name):
        # Register the tool with the registry
        register_tool(namespace, func_name)(func)
        logger.debug(f"Registered tool {qualified_name} with the registry")
    else:
        logger.debug(f"Tool {qualified_name} already registered with the registry")

def enhance_server(server: Union[Server, FastMCP], registry: Optional[ToolRegistry] = None) -> Tuple[Union[Server, FastMCP], ToolRegistry]:
    """Enhance an MCP server with the Tool Registry Pattern.
    
    This replaces the server's tool decorator with an enhanced version that registers
    all tools with the Tool Registry Pattern.
    
    Args:
        server: The MCP server to enhance
        registry: Optional existing registry to use. If None, a new registry is created.
        
    Returns:
        Tuple of (enhanced server, registry)
    """
    if registry is None:
        registry = ToolRegistry()
    
    # Cast server to enhanced type for type checking
    enhanced_server = cast(EnhancedServer, server)
    
    # Store original decorator
    original_tool_decorator = enhanced_server.tool
    
    # Create enhanced decorator
    def enhanced_tool_decorator(*args, **kwargs):
        # Get the original decorator
        original_decorator = original_tool_decorator(*args, **kwargs)
        
        # Return a wrapper that registers the tool with the registry
        def wrapper(func):
            # Register with registry before decorating with original decorator
            metadata = extract_tool_metadata(func)
            register_tool(metadata['namespace'], metadata['name'])(func)
            
            # Apply original decorator
            return original_decorator(func)
        
        return wrapper
    
    # Replace server.tool with enhanced decorator
    enhanced_server.tool = enhanced_tool_decorator
    
    # Scan for and register existing tools
    num_existing = scan_and_register_existing_tools(enhanced_server, registry)
    logger.info(f"Registered {num_existing} existing tools with registry")
    
    # Register tools from modules
    num_imported = register_tools_from_modules(registry)
    if num_imported > 0:
        logger.info(f"Registered {num_imported} tools from modules")
    
    return server, registry

def scan_and_register_existing_tools(server: Union[Server, FastMCP], registry: ToolRegistry, 
                                     module_paths: Optional[List[str]] = None) -> int:
    """
    Scan for and register existing tools that were already decorated with @server.tool()
    before the bridge was applied.
    
    Args:
        server: The MCP server containing the tools
        registry: The tool registry to register tools with
        module_paths: Optional module paths to scan. If None, all modules are scanned.
        
    Returns:
        The number of tools registered
    """
    if module_paths is None:
        module_paths = list(sys.modules.keys())
    
    # Exclude problematic modules to avoid warnings and errors
    excluded_module_prefixes = [
        "openai._extras",        # Requires additional dependencies
        "neo4j.graphql",         # Contains preview features
        "neo4j.notifications",   # Contains preview features
        "neo4j._graphql",        # Contains preview features (internal module)
        "neo4j._async.graphql",  # Contains preview features (async version)
        "neo4j.exceptions",      # May contain preview feature references
    ]
    
    # Specific Neo4j preview features to skip
    neo4j_preview_features = [
        "GqlStatusObject",
        "NotificationClassification",
        "NotificationDisabledClassification"
    ]
    
    # Filter out excluded modules
    filtered_module_paths = []
    for module_path in module_paths:
        if not any(module_path.startswith(prefix) for prefix in excluded_module_prefixes):
            filtered_module_paths.append(module_path)
    
    logger.info(f"Scanning for existing tools in {len(filtered_module_paths)} modules")
    
    tools_registered = 0
    
    for module_name in filtered_module_paths:
        try:
            module = sys.modules.get(module_name)
            if module is None:
                continue
                
            # Check if module has defined tools
            for name in dir(module):
                # Skip Neo4j preview features
                if name in neo4j_preview_features:
                    continue
                    
                try:
                    obj = getattr(module, name)
                    
                    # Check if this is a function decorated with @server.tool()
                    if inspect.isfunction(obj) and hasattr(obj, "_is_tool"):
                        # Check if this function uses our server (via checking _server attribute)
                        if hasattr(obj, "_server") and getattr(obj, "_server") is server:
                            # Extract metadata and register
                            metadata = extract_tool_metadata(obj)
                            namespace = metadata['namespace']
                            func_name = metadata['name']
                            
                            # Check if already registered
                            if not registry.get_tool_metadata(f"{namespace}.{func_name}"):
                                register_tool(namespace, func_name)(obj)
                                tools_registered += 1
                                logger.info(f"Registered existing tool: {obj.__module__}.{obj.__name__}")
                except AttributeError:
                    # Skip attributes that can't be accessed
                    continue
        except Exception as e:
            logger.error(f"Error scanning module {module_name}: {str(e)}")
    
    return tools_registered

def module_exists(module_name: str) -> bool:
    """Check if a module exists without importing it"""
    try:
        importlib.util.find_spec(module_name)
        return True
    except ImportError:
        return False

def find_submodules(package_path: str) -> List[str]:
    """Find all submodules in a package."""
    try:
        # Convert package path to directory path
        package_spec = importlib.util.find_spec(package_path)
        if not package_spec or not package_spec.submodule_search_locations:
            logger.error(f"Cannot find package path for {package_path}")
            return []
            
        package_dir = package_spec.submodule_search_locations[0]
        
        # Find all Python files in the directory
        submodules = []
        for file_path in glob.glob(os.path.join(package_dir, "*.py")):
            file_name = os.path.basename(file_path)
            if file_name == "__init__.py":
                continue
            module_name = file_name[:-3]  # Remove .py extension
            submodules.append(module_name)
        
        return submodules
    except Exception as e:
        logger.error(f"Error finding submodules for {package_path}: {str(e)}")
        return []

def register_tools_from_modules(registry: ToolRegistry, package_paths: Optional[List[str]] = None) -> int:
    """Import and register all tool modules from the specified packages.
    
    Args:
        registry: The tool registry to register tools with
        package_paths: List of package paths to import from. If None, defaults to common tool paths.
        
    Returns:
        The number of tools registered
    """
    if package_paths is None:
        package_paths = [
            "src.tools",
            "src.registry.tools"
        ]
    
    # Modules to exclude from registration due to warnings or dependencies
    excluded_module_prefixes = [
        "openai._extras",  # Requires additional dependencies
        "neo4j.graphql",   # Contains preview features
        "neo4j.notifications",  # Contains preview features
    ]
    
    total_registered = 0
    for package_path in package_paths:
        # Skip excluded packages
        if any(package_path.startswith(prefix) for prefix in excluded_module_prefixes):
            logger.info(f"Skipping excluded package: {package_path}")
            continue
            
        try:
            logger.info(f"Importing tools from {package_path}")
            
            # Find all submodules in this package
            submodules = find_submodules(package_path)
            
            # Import each submodule
            for submodule in submodules:
                full_module_name = f"{package_path}.{submodule}"
                
                # Skip excluded modules
                if any(full_module_name.startswith(prefix) for prefix in excluded_module_prefixes):
                    logger.info(f"Skipping excluded module: {full_module_name}")
                    continue
                    
                try:
                    logger.info(f"Importing module {full_module_name}")
                    importlib.import_module(full_module_name)
                except Exception as e:
                    logger.error(f"Error importing module {full_module_name}: {str(e)}")
        except Exception as e:
            logger.error(f"Error processing package {package_path}: {str(e)}")
    
    return total_registered

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
    """Get statistics about the registry."""
    registry = get_registry()
    
    # Count functions by namespace
    namespace_counts = {}
    for namespace in registry.get_namespaces():
        # Changed from get_functions_by_namespace to get_tools_by_namespace
        tools = registry.get_tools_by_namespace(namespace)
        namespace_counts[namespace] = len(tools)
    
    # Get all functions
    # Changed from get_all_functions to get_all_tools
    all_tools = registry.get_all_tools()
    
    return {
        "total_functions": len(all_tools),
        "namespaces": list(registry.get_namespaces()),
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
    
    # Modules to exclude from registration due to warnings or dependencies
    excluded_module_prefixes = [
        "openai._extras",  # Requires additional dependencies
        "neo4j.graphql",   # Contains preview features
        "neo4j.notifications",  # Contains preview features
    ]
    
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
                
                # Skip excluded modules
                if any(full_module_name.startswith(prefix) for prefix in excluded_module_prefixes):
                    logger.info(f"Skipping excluded module: {full_module_name}")
                    continue
                
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
                                if registry.get_tool_metadata(full_name) is None:
                                    try:
                                        register_tool(namespace, name)(obj)
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
                        register_tool(namespace, name)(func)
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