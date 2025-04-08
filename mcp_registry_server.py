#!/usr/bin/env python3
"""
MCP Registry Server

This server implements the Function Registry Pattern with automatic tool registration:
- All tools are automatically registered with the registry via the registry-MCP bridge
- Only exposes three essential registry tools to clients
- Registry functions provide full access to all functionality internally
"""

import os
import sys
import asyncio
import json
from typing import Dict, List, Any, Optional, AsyncIterator
from contextlib import asynccontextmanager
import importlib.util

# Ensure repo root is in path
repo_root = os.path.dirname(os.path.abspath(__file__))
if repo_root not in sys.path:
    sys.path.insert(0, repo_root)

try:
    from mcp.server.fastmcp import FastMCP
    from mcp.server.lowlevel import NotificationOptions, Server
    from mcp.types import Tool, TextContent

    from src.graph_memory import GraphMemoryManager
    from src.logger import LogLevel, get_logger
    from src.session_manager import SessionManager
    from src.registry.registry_manager import get_registry
    from src.registry.function_models import FunctionResult
    from src.registry.parameter_helper import ParameterHelper
    from src.registry import initialize_registry
    from src.registry.registry_mcp_bridge import enhance_server, get_registry_stats, list_registered_tools, register_tool_modules

    # Initialize logger
    logger = get_logger()
    logger.set_level(LogLevel.DEBUG)
    logger.info("Initializing MCP Registry Server with bridge")

    # Dynamic module discovery and import
    def import_all_modules_in_directory(base_dir, package_prefix):
        """
        Recursively discover and import all Python modules in a directory.
        
        Args:
            base_dir: The base directory to scan
            package_prefix: The package prefix to use for imports
        
        Returns:
            List of imported module names
        """
        imported_modules = []
        
        # Make sure the directory exists
        if not os.path.exists(base_dir) or not os.path.isdir(base_dir):
            logger.error(f"Directory does not exist: {base_dir}")
            return imported_modules
        
        # Get all Python files in the directory
        for root, dirs, files in os.walk(base_dir):
            # Skip __pycache__ directories
            if "__pycache__" in root:
                continue
                
            for file in files:
                if file.endswith(".py") and not file.startswith("__"):
                    # Get the full path to the file
                    file_path = os.path.join(root, file)
                    
                    # Calculate the module name
                    rel_path = os.path.relpath(file_path, os.path.dirname(base_dir))
                    module_name = os.path.splitext(rel_path)[0].replace(os.sep, ".")
                    full_module_name = f"{package_prefix}.{module_name}"
                    
                    try:
                        # Import the module
                        spec = importlib.util.spec_from_file_location(full_module_name, file_path)
                        if spec and spec.loader:
                            module = importlib.util.module_from_spec(spec)
                            spec.loader.exec_module(module)
                            imported_modules.append(full_module_name)
                            logger.info(f"Imported module: {full_module_name}")
                    except Exception as e:
                        logger.error(f"Error importing module {full_module_name}: {e}")
        
        return imported_modules

    # Import all modules from tools and registry directories
    logger.info("Dynamically discovering and importing tool modules...")
    src_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "src")
    tools_dir = os.path.join(src_dir, "tools")
    registry_dir = os.path.join(src_dir, "registry")
    
    imported_tool_modules = import_all_modules_in_directory(tools_dir, "src")
    logger.info(f"Imported {len(imported_tool_modules)} tool modules")
    
    imported_registry_modules = import_all_modules_in_directory(registry_dir, "src")
    logger.info(f"Imported {len(imported_registry_modules)} registry modules")

    # Store of client-specific GraphMemoryManager instances
    client_managers = {}

    # Create session manager with default settings
    session_manager = SessionManager(inactive_timeout=3600, cleanup_interval=300)

    # Lifespan context manager for Neo4j connections
    @asynccontextmanager
    async def server_lifespan(server: Server) -> AsyncIterator[dict]:
        """Manage Neo4j connection lifecycle and session cleanup."""
        # Start the session cleanup task
        logger.info("Initializing client manager store")
        await session_manager.start_cleanup_task()
        
        try:
            yield {"client_managers": client_managers, "session_manager": session_manager}
        finally:
            # Stop the cleanup task
            await session_manager.stop_cleanup_task()
            
            # Clean up at shutdown - close all client Neo4j connections
            logger.info("Shutting down all client Neo4j connections")
            for client_id, manager in list(client_managers.items()):
                logger.info(f"Closing Neo4j connection for client {client_id}")
                manager.close()
                # Remove from the dictionary after closing
                client_managers.pop(client_id, None)

    # Function to get or create a client-specific manager
    # This is the REAL implementation from main.py, not a mock
    def get_client_manager(client_id=None):
        """
        Get the GraphMemoryManager for the current client or create one if it doesn't exist.
        
        Args:
            client_id: Optional client ID to use. If not provided, a default client ID is used.
                     In a real implementation, this would be derived from the SSE connection.
        
        Returns:
            GraphMemoryManager instance for the client
        """
        try:
            # Use provided client ID or default
            effective_client_id = client_id or "default-client"
            
            logger.debug(f"Getting manager for client ID: {effective_client_id}")
            
            # Update the last activity time for this client
            session_manager.update_activity(effective_client_id)
            
            # Create a new manager if this client doesn't have one yet
            if effective_client_id not in client_managers:
                logger.info(f"Creating new GraphMemoryManager for client {effective_client_id}")
                manager = GraphMemoryManager(logger)
                manager.initialize()
                client_managers[effective_client_id] = manager
                
                # Register the client with the session manager
                session_manager.register_client(effective_client_id, manager)
                
            return client_managers[effective_client_id]
        except Exception as e:
            logger.error(f"Error getting client manager: {str(e)}")
            # Fall back to a temporary manager if something goes wrong
            return GraphMemoryManager(logger)

    # Tool-only server for registration - tools will be registered
    # with the registry but NOT with the real server exposed to clients
    class MockServer:
        def __init__(self):
            self.registered_tools = {}
            
        def tool(self, *args, **kwargs):
            """Mock implementation of server.tool() decorator that registers with registry only"""
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
                
                # Register with the registry, NOT with the real server
                from src.registry.registry_manager import register_function
                try:
                    register_function(namespace, name)(func)
                    logger.info(f"Registered tool {namespace}.{name} with registry only")
                except Exception as e:
                    logger.error(f"Error registering {name} with registry: {str(e)}")
                
                # Return the original function (we're not really decorating it)
                return func
            
            # Handle both @server.tool and @server.tool()
            if len(args) == 1 and callable(args[0]):
                # Called as @server.tool
                return decorator(args[0])
            else:
                # Called as @server.tool()
                return decorator

    # Initialize the registry first
    logger.info("Initializing Function Registry...")
    initialize_registry()

    # Register all tools with our registry using real implementations
    logger.info("Registering tool modules...")
    
    # Import the registration functions
    from src.tools import register_all_tools
    
    # Create a server just for tool registration
    mock_server = MockServer()
    
    # Register all tools with the mock server but real client manager
    # This ensures tools are registered with the registry but not exposed to clients
    register_all_tools(mock_server, get_client_manager)
    tools_registered = len(mock_server.registered_tools)
    logger.info(f"Registration complete: {tools_registered} tools registered with registry")

    # Create the MCP server with only essential tools
    server = FastMCP(
        name="MCP Registry Server",
        description="Registry-MCP bridge with automatic function registration",
        version="1.0.0",
        notification_options=NotificationOptions(),
        lifespan=server_lifespan,
        instructions="Use execute_function, list_available_functions, and list_function_categories to access all functionality"
    )

    logger.info("MCP Server created with only essential tools exposed")

    # Enhance the server with our registry bridge for future tool registrations
    registry = get_registry()
    server, registry = enhance_server(server, registry)

    # Define the 3 essential tools
    # Tool 1: execute_function
    @server.tool()  # type: ignore
    async def execute_function(function_name: str, parameters: Optional[str] = None) -> str:
        """
        Execute any registered function by name with provided parameters.
        
        This tool provides a unified interface to all registered functions, allowing
        access to the full range of functionality through a single entry point.
        
        Args:
            function_name: Full name of function (e.g., 'memory.create_entity')
            parameters: JSON string with parameters to pass to the function
            
        Returns:
            JSON string with the function result
        """
        # Get the registry instance
        registry = get_registry()
        
        if not function_name:
            error_result = FunctionResult(
                status="error",
                data=None,
                message="Missing required parameter 'function_name'",
                error_code="MISSING_PARAMETER",
                error_details={"message": "function_name parameter is required"}
            )
            return error_result.to_json()
        
        try:
            # Get function metadata for validation and conversion
            metadata = registry.get_function_metadata(function_name)
            if metadata is None:
                error_result = FunctionResult(
                    status="error",
                    data=None,
                    message=f"Function '{function_name}' not found",
                    error_code="FUNCTION_NOT_FOUND",
                    error_details={
                        "available_namespaces": list(registry.get_namespaces())
                    }
                )
                return error_result.to_json()
            
            # Parse parameters from JSON string if provided
            parsed_params = {}
            if parameters:
                try:
                    if isinstance(parameters, str):
                        parsed_params = json.loads(parameters)
                    elif isinstance(parameters, dict):
                        parsed_params = parameters
                    else:
                        error_result = FunctionResult(
                            status="error",
                            data=None,
                            message=f"Invalid parameters format: {type(parameters)}",
                            error_code="INVALID_PARAMETERS",
                            error_details={"expected": "JSON string or dictionary"}
                        )
                        return error_result.to_json()
                except json.JSONDecodeError as e:
                    error_result = FunctionResult(
                        status="error",
                        data=None,
                        message=f"Invalid JSON in parameters: {str(e)}",
                        error_code="INVALID_JSON",
                        error_details={"json_error": str(e)}
                    )
                    return error_result.to_json()
            
            # Try to validate and convert parameters
            try:
                # Basic parameter validation
                validation_errors = ParameterHelper.validate_parameters(metadata, parsed_params)
                if validation_errors:
                    error_details = {
                        "validation_errors": [str(error) for error in validation_errors]
                    }
                    error_result = FunctionResult(
                        status="error",
                        message=f"Parameter validation failed for function '{function_name}'",
                        data=None,
                        error_code="PARAMETER_VALIDATION_ERROR",
                        error_details=error_details
                    )
                    return error_result.to_json()
                
                # Convert parameters to the right types
                parsed_params = ParameterHelper.convert_parameters(metadata, parsed_params)
            except Exception as e:
                logger.error(f"Error processing parameters: {str(e)}")
                error_result = FunctionResult(
                    status="error",
                    data=None,
                    message=f"Error processing parameters: {str(e)}",
                    error_code="PARAMETER_PROCESSING_ERROR",
                    error_details={"exception": str(e)}
                )
                return error_result.to_json()
            
            # Execute the function with the parsed parameters (not as 'parameters' kwarg)
            result = await registry.execute(function_name, **parsed_params)
            return result.to_json()
        except Exception as e:
            logger.error(f"Error executing function: {str(e)}")
            error_result = FunctionResult(
                status="error",
                data=None,
                message=f"Error executing function: {str(e)}",
                error_code="FUNCTION_EXECUTION_ERROR",
                error_details={"exception": str(e)}
            )
            return error_result.to_json()

    # Tool 2: list_available_functions
    @server.tool()  # type: ignore
    async def list_available_functions(category = None) -> str:
        """
        List all available functions, optionally filtered by category.
        
        This tool allows discovery of all registered functions and their documentation.
        
        Args:
            category: Optional category to filter by
            
        Returns:
            JSON string with function metadata
        """
        # Get the registry instance
        registry = get_registry()
        
        try:
            if category:
                functions = registry.get_functions_by_namespace(category)
            else:
                functions = registry.get_all_functions()
                
            # Convert to dictionary for better serialization
            result = {
                "functions": [f.model_dump() for f in functions],
                "count": len(functions),
                "categories": list(registry.get_namespaces())
            }
            
            return json.dumps(result, indent=2)
        except Exception as e:
            error_result = {
                "error": str(e),
                "functions": []
            }
            return json.dumps(error_result)

    # Tool 3: list_function_categories
    @server.tool()  # type: ignore
    async def list_function_categories() -> str:
        """
        List all available function categories/namespaces.
        
        Returns:
            JSON string with category information
        """
        # Get the registry instance
        registry = get_registry()
        
        try:
            namespaces = registry.get_namespaces()
            
            # Get function count per namespace
            category_counts = {}
            for ns in namespaces:
                functions = registry.get_functions_by_namespace(ns)
                category_counts[ns] = len(functions)
            
            result = {
                "categories": list(namespaces),
                "category_counts": category_counts,
                "total_categories": len(namespaces)
            }
            
            return json.dumps(result, indent=2)
        except Exception as e:
            error_result = {
                "error": str(e),
                "categories": []
            }
            return json.dumps(error_result)

    # Define middleware for tracking client activity
    async def client_tracking_middleware(request, call_next):
        """Middleware to track client sessions and mark disconnections."""
        # Extract session ID from request
        session_id = request.query_params.get("session_id", None)
        
        # Mark client activity
        if session_id:
            logger.debug(f"Client activity: {session_id}")
            session_manager.update_activity(session_id)
        
        # Process the request
        response = await call_next(request)
        
        # Handle disconnection event for SSE requests
        if session_id and request.url.path == "/sse":
            # In SSE, we need to set up background cleanup for when the connection ends
            async def on_disconnect():
                try:
                    # Small delay to ensure cleanup happens after the connection is fully closed
                    await asyncio.sleep(1)
                    logger.info(f"Client disconnected: {session_id}")
                    session_manager.mark_client_inactive(session_id)
                except Exception as e:
                    logger.error(f"Error during disconnect handling for {session_id}: {str(e)}")
            
            response.background = on_disconnect()
        
        return response

    # Helper for main server run
    async def run_server():
        """Run the MCP server with the configured transport."""
        try:
            # Determine transport type from environment variable
            use_sse = os.environ.get("USE_SSE", "false").lower() == "true"
            port = int(os.environ.get("PORT", "8080"))

            # Show registry statistics after all tools have been registered
            stats = get_registry_stats()
            logger.info(f"Registry now contains {stats['total_functions']} functions in {len(stats['namespaces'])} namespaces")
            logger.info(f"Registry breakdown:")
            for ns, count in stats['namespace_counts'].items():
                logger.info(f"  - {ns}: {count} functions")

            if use_sse:
                # Using SSE transport
                logger.info(f"MCP Registry Server running with SSE on http://0.0.0.0:{port}")
                
                # Get the standard SSE app
                app = server.sse_app()  # type: ignore
                
                # Add our middleware for client tracking
                # Import necessary Starlette components
                try:
                    from starlette.middleware.base import BaseHTTPMiddleware
                    from starlette.applications import Starlette
                    
                    # Create a new Starlette app with middleware
                    app_with_middleware = Starlette(routes=app.routes)
                    app_with_middleware.add_middleware(BaseHTTPMiddleware, dispatch=client_tracking_middleware)
                    
                    return app_with_middleware
                except ImportError as e:
                    logger.error(f"Starlette import error: {e}. Make sure 'starlette' is installed.")
                    # Fall back to standard app without middleware
                    logger.error("Running without client tracking middleware")
                    return app
            else:
                # Using stdio transport
                logger.info("MCP Registry Server running on stdio")
                
                # Check if run is a coroutine
                if asyncio.iscoroutinefunction(server.run):
                    # If it's a coroutine function, await it
                    await server.run()  # type: ignore
                else:
                    # If it's not a coroutine function, just call it
                    server.run()  # type: ignore
                return None
        except Exception as e:
            logger.error(f"Failed to start server: {str(e)}")
            sys.exit(1)

    def main():
        """Main entry point with enhanced error handling."""
        try:
            # Set Windows event loop policy if needed
            if sys.platform == 'win32':
                asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
            
            # Determine transport type
            use_sse = os.environ.get("USE_SSE", "false").lower() == "true"
            port = int(os.environ.get("PORT", "8080"))
            
            if use_sse:
                # For SSE, we need to run the server in a different way
                try:
                    import uvicorn
                    
                    # Get the app with middleware
                    app = asyncio.run(run_server())
                    
                    # Run the server if app was returned
                    if app is not None:
                        # Type check to make sure app is an ASGI application
                        from starlette.applications import Starlette
                        if isinstance(app, Starlette):
                            uvicorn.run(app, host="0.0.0.0", port=port)
                        else:
                            logger.error("Invalid app type returned from run_server")
                            sys.exit(1)
                    else:
                        logger.error("No app returned from run_server")
                        sys.exit(1)
                except ImportError:
                    logger.error("uvicorn is required for SSE transport. Please install it with 'pip install uvicorn'.")
                    sys.exit(1)
            else:
                # For stdio, we can use asyncio.run
                asyncio.run(run_server())
        except Exception as e:
            logger.error(f"Failed to run server: {str(e)}")
            sys.exit(1)

except ImportError as e:
    print(f"Error importing required modules: {e}")
    sys.exit(1)

if __name__ == "__main__":
    main() 