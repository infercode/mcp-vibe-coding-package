#!/usr/bin/env python3
"""
Minimal MCP Server

This server implements a minimal approach for the Function Registry Pattern:
- Only registers and exposes the three essential registry tools
- These tools provide access to all registry functions internally
"""

import os
import sys
import asyncio
import json
from typing import Dict, List, Any, Optional, AsyncIterator, Callable, Union
from contextlib import asynccontextmanager
import importlib
import time
from datetime import datetime

from mcp.server.fastmcp import FastMCP
from mcp.server.lowlevel import NotificationOptions, Server
from mcp.types import Tool, TextContent

from src.graph_memory import GraphMemoryManager
from src.logger import LogLevel, get_logger
from src.session_manager import SessionManager
from src.registry.registry_manager import get_registry, register_function
from src.registry.function_models import FunctionResult
from src.registry.parameter_helper import ParameterHelper
from src.registry import initialize_registry
from src.registry.registry_mcp_bridge import enhance_server, get_registry_stats

# Initialize logger
logger = get_logger()
logger.set_level(LogLevel.DEBUG)
logger.info("Initializing Minimal MCP Server with only essential tools")

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
        
        # Clean up at shutdown - close all client connections
        logger.info("Shutting down all client Neo4j connections")
        for client_id, manager in list(client_managers.items()):
            logger.info(f"Closing Neo4j connection for client {client_id}")
            manager.close()
            # Remove from the dictionary after closing
            client_managers.pop(client_id, None)

# Get client-specific manager
async def get_client_manager(client_id: str) -> GraphMemoryManager:
    """
    Get or create a client-specific GraphMemoryManager.
    
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

# Initialize registry without exposing tools
# This registers all functions with the registry but doesn't create tools for them
registry = initialize_registry()

# Register functions directly to ensure registry has content
from src.registry.registry_manager import register_function

# Try to manually register other functions from modules
try:
    # Import and initialize modules that register functions
    from src.registry.health_diagnostics import register_health_diagnostics_tools
    register_health_diagnostics_tools()
    
    # Import and register feedback functions
    try:
        # Check if the feedback mechanism module exists
        feedback_module = importlib.import_module('src.registry.feedback_mechanism')
        
        # Look for registration function with appropriate name
        if hasattr(feedback_module, 'register_feedback_mechanism_tools'):
            feedback_module.register_feedback_mechanism_tools()
        elif hasattr(feedback_module, 'register_feedback_tools'):
            feedback_module.register_feedback_tools()
        else:
            logger.info("No feedback tool registration function found")
    except (ImportError, AttributeError):
        logger.info("Could not register feedback tools")
    
    # Import and register other essential registry functions
    try:
        # Attempt to import registry module
        registry_module = importlib.import_module('src.registry.registry_tools')
        
        # Check for appropriate registration function
        if hasattr(registry_module, 'register_registry_tools'):
            registry_module.register_registry_tools(None)  # Pass None as server
        else:
            logger.info("No registry tools registration function found")
    except (ImportError, AttributeError):
        logger.info("Could not register core registry functions")
        
except Exception as e:
    logger.error(f"Error manually registering functions: {str(e)}")

logger.info(f"Registry initialized with {len(registry.get_all_functions())} functions in {len(registry.get_namespaces())} namespaces")

# Create the MCP server with only essential tools
server = FastMCP(
    name="Minimal Registry Server",
    description="Exposes only essential registry tools but allows accessing all functions",
    version="1.0.0",
    notification_options=NotificationOptions(),
    lifespan=server_lifespan,
    instructions="Use the execute_function tool to access all registry functions"
)

# Enhance the server with our registry bridge to auto-register all tools
server = enhance_server(server)

# Define essential tool: execute_function
@server.tool()
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

# Define essential tool: list_available_functions
@server.tool()
async def list_available_functions(category = None) -> str:
    """
    List all available functions, optionally filtered by category.
    
    This tool allows discovery of all registered functions and their documentation.
    
    Args:
        category: Optional category to filter by
        
    Returns:
        JSON string with function metadata
    """
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
            logger.info(f"Minimal MCP Server running with SSE on http://0.0.0.0:{port}")
            
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
            logger.info("Minimal MCP Server running on stdio")
            
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