#!/usr/bin/env python3
"""
Hybrid MCP Server

This server implements the hybrid approach for the Function Registry Pattern:
- All functions are registered with the internal registry
- Only the essential registry tools are exposed to clients
"""

import os
import sys
import asyncio
import json
from typing import Dict, List, Any, Optional, AsyncIterator, Callable, Union
from contextlib import asynccontextmanager

from mcp.server.fastmcp import FastMCP
from mcp.server.lowlevel import NotificationOptions, Server
from mcp.server.models import InitializationOptions
from mcp.types import Tool, TextContent, PromptMessage

from src.graph_memory import GraphMemoryManager
from src.logger import LogLevel, get_logger
from src.utils import dict_to_json, dump_neo4j_nodes
from src.tools import register_all_tools
from src.session_manager import SessionManager
from src.registry.registry_manager import get_registry
from src.registry.function_models import FunctionResult, FunctionMetadata
from src.registry.parameter_helper import ParameterHelper

# Initialize logger
logger = get_logger()
logger.set_level(LogLevel.DEBUG)
logger.info("Initializing Hybrid Neo4j MCP Graph Memory Server", context={"version": "1.0.0"})

# Store of client-specific GraphMemoryManager instances
client_managers = {}

# Create session manager with default settings
# 1 hour inactive timeout, 5 minutes cleanup interval
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
        
        # Clean up at shutdown - close all client connections
        logger.info("Shutting down all client Neo4j connections")
        for client_id, manager in list(client_managers.items()):
            logger.info(f"Closing Neo4j connection for client {client_id}")
            manager.close()
            # Remove from the dictionary after closing
            client_managers.pop(client_id, None)

# Create the MCP server with proper configuration based on main.py
server = FastMCP(
    name="Neo4j Graph Memory Server",
    description="Advanced memory systems with Neo4j graph database and Function Registry Pattern",
    version="1.0.0",
    notification_options=NotificationOptions(),  # Use default options
    experimental_capabilities={"function_registry": True},
    lifespan=server_lifespan,
    instructions="Use the execute_function tool to access all registry functions"
)

# Define the list of essential tools we want to expose
ESSENTIAL_TOOL_NAMES = [
    "execute_function",
    "list_available_functions",
    "list_function_categories"
]

# Get client-specific manager
async def get_client_manager(client_id: str) -> GraphMemoryManager:
    """
    Get or create a client-specific GraphMemoryManager.
    
    This function ensures each client gets its own isolated Neo4j connection.
    
    Args:
        client_id: The client identifier
        
    Returns:
        GraphMemoryManager instance for the client
    """
    # Update session activity timestamp
    if hasattr(session_manager, "update_activity"):
        session_manager.update_activity(client_id)
    
    # Get or create client manager
    if client_id not in client_managers:
        logger.info(f"Creating new GraphMemoryManager for client {client_id}")
        client_managers[client_id] = GraphMemoryManager(logger)
        
    return client_managers[client_id]

# Define middleware for tracking client activity
async def client_tracking_middleware(request, call_next):
    """Track client activity for session management."""
    # Extract client ID from headers or query params
    session_id = request.query_params.get("session_id", None)
    
    # If client ID found, update activity
    if session_id and hasattr(session_manager, "update_activity"):
        logger.debug(f"Client activity: {session_id}")
        session_manager.update_activity(session_id)
    
    # Continue with the request
    response = await call_next(request)
    return response

# Register all tools with the registry (but don't expose them all)
register_all_tools(server, get_client_manager)

# Get a list of all tools currently registered
registered_tool_count = len(getattr(server, "_tools", []))
logger.info(f"Total number of registered tools: {registered_tool_count}")

# Override the tools listing behavior
# This is the key to the hybrid approach:
# We filter the tools that are exposed to clients
async def custom_list_tools():
    """
    List only the essential tools that should be exposed to clients.
    
    Returns:
        List of essential tools
    """
    # Get all registered tools - safely using getattr to avoid attribute errors
    all_tools = getattr(server, "_tools", [])
    
    # Filter to only include essential tools
    essential_tools = [tool for tool in all_tools if tool.name in ESSENTIAL_TOOL_NAMES]
    
    logger.info(f"Exposing {len(essential_tools)} essential tools out of {registered_tool_count} total tools")
    return essential_tools

# Replace the server's list_tools method with our custom implementation
original_list_tools = server.list_tools
server.list_tools = custom_list_tools

# REMOVE direct modification of _tools attribute which causes linter errors
# Instead, just log that we're using a custom list_tools implementation
logger.info("Replaced server's list_tools method with custom implementation to filter exposed tools")
logger.info(f"Server will now only expose {len(ESSENTIAL_TOOL_NAMES)} essential tools to clients")

# Define essential tool: execute_function
@server.tool()
async def execute_function(function_name: str, **parameters) -> str:
    """
    Execute any registered function by name with provided parameters.
    
    This tool provides a unified interface to all registered functions, allowing
    access to the full range of functionality through a single entry point.
    
    Args:
        function_name: Full name of function (e.g., 'memory.create_entity')
        **parameters: Parameters to pass to the function
        
    Returns:
        JSON string with the function result
    """
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
        
        # Try to validate and convert parameters
        try:
            # Basic parameter validation
            validation_errors = ParameterHelper.validate_parameters(metadata, parameters)
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
            parameters = ParameterHelper.convert_parameters(metadata, parameters)
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
        
        # Execute the function
        result = await registry.execute(function_name, **parameters)
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

# Define essential tool: list_available_functions
@server.tool()
async def list_available_functions(category: Optional[str] = None) -> str:
    """
    List all available functions, optionally filtered by category.
    
    This tool allows discovery of all registered functions and their documentation.
    
    Args:
        category: Optional category to filter by
        
    Returns:
        JSON string with function metadata
    """
    registry = get_registry()
    
    try:
        if category:
            functions = registry.get_functions_by_namespace(category)
        else:
            functions = registry.get_all_functions()
            
        # Convert to dictionary for better serialization
        result = {
            "functions": [f.dict() for f in functions],
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

# Define essential tool: list_function_categories
@server.tool()
async def list_function_categories() -> str:
    """
    List all available function categories/namespaces.
    
    Returns:
        JSON string with category information
    """
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

# Helper for main server run
async def run_server():
    """Run the MCP server with the configured transport."""
    try:
        # Determine transport type from environment variable
        use_sse = os.environ.get("USE_SSE", "false").lower() == "true"
        port = int(os.environ.get("PORT", "8080"))

        if use_sse:
            # Using SSE transport
            logger.info(f"Hybrid Neo4j Graph Memory MCP Server running with SSE on http://0.0.0.0:{port}")
            
            # Get the standard SSE app
            app = server.sse_app()
            
            # Add our middleware for client tracking
            from starlette.middleware.base import BaseHTTPMiddleware
            from starlette.applications import Starlette
            
            # Create a new Starlette app with middleware
            app_with_middleware = Starlette(routes=app.routes)
            app_with_middleware.add_middleware(BaseHTTPMiddleware, dispatch=client_tracking_middleware)
            
            return app_with_middleware
        else:
            # Using stdio transport
            logger.info("Hybrid Neo4j Graph Memory MCP Server running on stdio")
            
            # Check if run is a coroutine
            if asyncio.iscoroutinefunction(server.run):
                # If it's a coroutine function, await it
                await server.run()
            else:
                # If it's not a coroutine function, just call it
                server.run()
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

if __name__ == "__main__":
    main() 