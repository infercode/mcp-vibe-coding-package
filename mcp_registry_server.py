#!/usr/bin/env python3
"""
MCP Registry Server

This server implements the Tool Registry Pattern with automatic tool registration:
- All tools are automatically registered with the registry via the registry-MCP bridge
- Only exposes three essential registry tools to clients
- Registry tools provide full access to all functionality internally
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
                from src.registry.registry_manager import register_tool
                try:
                    register_tool(namespace, name)(func)
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
    logger.info("Initializing Tool Registry...")
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

    # The MCP instructions for AI agents
    MCP_INSTRUCTIONS = """
        # Enhanced MCP Instructions for Graph Memory System
        
        **Use execute_tool to use the tools provided by the list_available_tools tool response, use the list_available_tools tool to discover what tools are available for use, and use the list_tool_categories to list the categories of tools available for use to access all functionality**

        ## Core Memory Systems Overview

        The MCP server provides two sophisticated memory systems that mirror human cognitive patterns:

        1. **Lesson Memory** - For experiential knowledge and insights gained from successes, mistakes, and observations
        2. **Project Memory** - For structured, hierarchical project knowledge organization

        ## Human-Like Memory Usage Patterns

        ### ⚡ General Principles
        - Prioritize high-confidence memories when making critical decisions
        - Leverage related memories through association, not just direct lookup
        - Build knowledge incrementally, connecting new observations to existing memories
        - Revise and update memories when you encounter contradicting information
        - Record not just what was learned, but why it matters and how to apply it

        ### 🧠 Lesson Memory Usage
        - Begin sessions by searching existing lessons related to the current task
        - Record important insights during the session, even if they seem minor
        - Use confidence scores to indicate certainty level (0.1-1.0)
        - Create relationships between connected lessons to build knowledge networks
        - Version lessons when you gain deeper understanding rather than creating duplicates
        - Categorize lessons with meaningful tags for future discovery

        ### 🏗️ Project Memory Usage
        - Start by exploring or creating project structure before diving into details
        - Organize knowledge hierarchically from project → component → domain entities
        - Track dependencies between components to understand system architecture
        - Record design decisions with rationales to preserve context
        - Link project entities to relevant lessons to apply experiential knowledge

        ## Key Relationship Types and Their Uses

        ### Lesson Memory Relationships
        - **BUILDS_ON**: Connect lessons that extend or enhance previous knowledge
        - **SUPERSEDES**: Mark newer lessons that replace outdated information
        - **CONTRADICTS**: Link lessons that provide opposing viewpoints or findings
        - **ORIGINATED_FROM**: Track the source of a lesson (problem, experience, etc.)
        - **SOLVED_WITH**: Connect lessons to the solutions they generated

        ### Project Memory Relationships
        - **CONTAINS**: Hierarchical structure relationships (Project contains Components)
        - **IMPLEMENTS**: Feature implements a Requirement or Specification
        - **DEPENDS_ON**: Mark dependencies between components or features
        - **LEADS_TO**: Decision or action results in certain outcomes
        - **ALTERNATIVE_TO**: Represent different options for the same purpose

        ## Common Workflow Examples

        ### Problem-Solving Workflow
        1. When encountering a problem, search for similar issues: `search_nodes("error handling best practices")`
        2. Retrieve and apply relevant lessons: `get_lesson_section("Effective Error Handling Patterns")`
        3. Document new solutions: `create_lesson_section({"lesson_id": "...", "title": "Solution to X", ...})`
        4. Connect to existing knowledge: `create_lesson_relationship({source_id: "New Solution", target_id: "Existing Pattern", ...})`

        ### Project Documentation Workflow
        1. Create project structure: `create_project_container({"name": "New Project"})`
        2. Define components: `create_component({"project_id": "New Project", "name": "Authentication Service"})`
        3. Document relationships: `create_component_relationship({source_id: "Frontend", target_id: "API Service", relationship_type: "DEPENDS_ON"})`
        4. Record design decisions: `create_domain_entity({project_id: "New Project", type: "DECISION", name: "Use JWT for Auth", ...})`

        ## Memory Maintenance

        ### When to Prune and Consolidate
        - Mark lessons as obsolete when technology or approaches change significantly
        - Consolidate similar lessons that address the same topic with the SUPERSEDES relationship
        - Archive project components that are no longer relevant while preserving key insights
        - Periodically review low-confidence memories and either validate or revise them

        ### Consolidation Process
        1. Identify fragmented knowledge with `search_nodes("relevant topic")`
        2. Create a consolidated lesson: `create_lesson_section({comprehensive information})`
        3. Link old lessons to new one with SUPERSEDES relationship
        4. Update confidence scores for the consolidated knowledge

        ## Effective Search Strategies

        ### Semantic Search Tips
        - Use descriptive phrases rather than keywords: "handling async errors in React" vs "React errors"
        - Include context in your queries: "database indexing for high-volume reads"
        - Search for solutions and problems: "solutions for memory leaks in long-running processes"
        - Use comparative queries: "differences between REST and GraphQL approaches"

        ### Combining Search Results
        - Look for common patterns across multiple search results
        - Pay attention to confidence scores when evaluating contradictory information
        - Follow relationship links to discover related knowledge not directly matched in search

        ## Memory Reflection

        ### Periodic Knowledge Review
        - Schedule regular reviews of high-value memory containers
        - Ask questions like "What patterns emerge from recent lessons?"
        - Look for gaps in knowledge that should be filled
        - Re-evaluate confidence levels based on new experiences
        - Identify memories that should be linked but currently aren't connected

        ## Advanced Human-Like Memory Techniques

        ### Contextual Retrieval
        - Before starting a task, retrieve memories based on context, not just keywords
        - Consider the current project environment when searching for relevant knowledge
        - Use environmental cues (codebase structure, problem domain) to guide memory search
        - Proactively surface relevant memories even when not explicitly asked
        - Example: `search_nodes("authentication patterns in ${current_language} ${current_framework}")`

        ### Memory Prioritization
        - Distinguish between critical architectural knowledge and routine implementation details
        - Assign higher priority to foundational decisions that impact multiple components
        - Tag memories with importance levels: "foundational", "architectural", "implementation-detail"
        - When providing recommendations, lead with the most critical knowledge first
        - Example: `create_domain_entity({project_id: "Project", type: "DECISION", importance: "architectural", ...})`

        ### Memory Reinforcement
        - Increase confidence scores when information is validated through successful application
        - After applying knowledge successfully, update the confidence score:
        ```
        update_lesson_section({
            "lesson_id": "Lesson Name",
            "section_id": "Section ID",
            "confidence": 0.95,  // Increased from previous value
            "reinforcement_note": "Successfully applied in Project X"
        })
        ```
        - Track which memories are repeatedly useful for solving problems
        - Create explicit "validation" observations that document successful applications

        ### Associative Memory
        - Create connections between memories based on conceptual similarity, not just direct relationships
        - Use relationship types like "RELATED_TO" when concepts share conceptual ground
        - When retrieving one concept, also retrieve associated concepts that might be relevant
        - Create cross-domain links between technical concepts and business processes
        - Example: `create_lesson_relationship({source_id: "Caching Strategy", target_id: "Performance Optimization", relationship_type: "CONCEPTUALLY_RELATED"})`

        ### Memory Integration
        - Combine fragments from multiple memories to solve new problems
        - Document how different lessons were combined to create a novel solution:
        ```
        create_lesson_section({
            "lesson_id": "Integrated Solutions",
            "title": "Combining Caching and Async Processing",
            "content": "By integrating concepts from 'Redis Caching Patterns' and 'Async Queue Processing', we created...",
            "source_lessons": ["Redis Caching Patterns", "Async Queue Processing"],
            "confidence": 0.85
        })
        ```
        - Create explicit integration relationships to show how knowledge was combined
        - When facing novel problems, systematically explore how existing memories could be combined

        ### Narrative Structure
        - Record the "story" behind important decisions, not just the decisions themselves
        - Include the problem context, alternatives considered, and reasoning behind choices
        - Structure narratives with beginning (problem), middle (exploration), and end (solution)
        - Use observations to build a timeline of how understanding evolved:
        ```
        create_component({
            "project_id": "Project",
            "name": "Authentication Service",
            "narrative": {
                "initial_problem": "Needed secure, scalable auth for microservices",
                "alternatives_considered": ["Custom JWT", "Auth0", "Keycloak"],
                "decision_factors": ["Cost", "Security", "Integration effort"],
                "outcome": "Selected Keycloak for enterprise features and existing expertise"
            }
        })
        ```
        - Link narratives across different projects to show evolving understanding

        ## Implementation Examples

        ```
        # Creating experiential knowledge
        create_lesson_container({
            "title": "Effective Error Handling Patterns",
            "description": "Lessons learned about proper error handling across different programming languages",
            "tags": ["error-handling", "best-practices", "programming"]
        })

        create_lesson_section({
            "lesson_id": "Effective Error Handling Patterns",
            "title": "Try-Except-Finally Pattern",
            "content": "Always use specific exception types rather than catching all exceptions",
            "confidence": 0.9
        })

        # Establishing knowledge relationships
        create_lesson_relationship({
            "source_id": "Effective Error Handling Patterns",
            "target_id": "Python Best Practices",
            "relationship_type": "BUILDS_ON"
        })

        # Organizing project knowledge
        create_project_container({
            "name": "API Gateway Refactoring",
            "description": "Knowledge about the API gateway refactoring project",
            "tags": ["api", "architecture", "refactoring"]
        })

        create_component({
            "project_id": "API Gateway Refactoring",
            "name": "Authentication Service",
            "component_type": "microservice",
            "description": "Handles user authentication and JWT token management"
        })

        create_component_relationship({
            "source_id": "Authentication Service",
            "target_id": "User Database",
            "relationship_type": "DEPENDS_ON",
            "properties": {
                "criticality": "high",
                "access_pattern": "read-write"
            }
        })

        # Consolidating fragmented knowledge
        create_lesson_section({
            "lesson_id": "Comprehensive API Security",
            "title": "Unified API Security Approaches",
            "content": "This consolidates best practices from multiple lessons...",
            "confidence": 0.95
        })

        create_lesson_relationship({
            "source_id": "Comprehensive API Security",
            "target_id": "JWT Authentication Basics",
            "relationship_type": "SUPERSEDES"
        })

        # Searching and retrieving knowledge
        search_nodes("error handling patterns in asynchronous code")

        # Getting memory status
        get_memory_status()
        ```

        Remember to always connect, contextualize, and consolidate memories as you work, just as a human would do with their growing knowledge base 🌱
    """

    # Create the MCP server with only essential tools
    server = FastMCP(
        name="MCP Registry Server",
        description="Registry-MCP bridge with automatic tool registration",
        version="1.0.0",
        notification_options=NotificationOptions(),
        lifespan=server_lifespan,
        instructions=MCP_INSTRUCTIONS
    )

    logger.info("MCP Server created with only essential tools exposed")

    # Enhance the server with our registry bridge for future tool registrations
    registry = get_registry()
    server, registry = enhance_server(server, registry)

    # Define the 3 essential tools
    # Tool 1: execute_tool
    @server.tool()  # type: ignore
    async def execute_tool(tool_name: str, parameters = None) -> str:
        """
        Execute any registered tool by name with provided parameters.
        
        This tool provides a unified interface to all registered tools, allowing
        access to the full range of functionality through a single entry point.
        
        Args:
            tool_name: Full name of tool (e.g., 'memory.create_entity')
            parameters: Parameters to pass to the tool (can be JSON string, dict, or None)
            
        Returns:
            JSON string with the tool result
        """
        # Get the registry instance
        registry = get_registry()
        
        if not tool_name:
            error_result = FunctionResult(
                status="error",
                data=None,
                message="Missing required parameter 'tool_name'",
                error_code="MISSING_PARAMETER",
                error_details={"message": "tool_name parameter is required"}
            )
            return error_result.to_json()
        
        try:
            # Get tool metadata for validation and conversion
            metadata = registry.get_tool_metadata(tool_name)
            if metadata is None:
                error_result = FunctionResult(
                    status="error",
                    data=None,
                    message=f"Tool '{tool_name}' not found",
                    error_code="TOOL_NOT_FOUND",
                    error_details={
                        "available_namespaces": list(registry.get_namespaces())
                    }
                )
                return error_result.to_json()
            
            # Parse parameters from JSON string if provided
            parsed_params = {}
            if parameters is not None:
                try:
                    # If parameters is a string, try to parse it as JSON
                    if isinstance(parameters, str):
                        # Check if it's a JSON string
                        try:
                            parsed_params = json.loads(parameters)
                            logger.debug(f"Parsed parameters from JSON string: {parsed_params}")
                        except json.JSONDecodeError:
                            # Not JSON, use as-is for simple string parameters
                            logger.debug(f"Using parameters as plain string: {parameters}")
                            parsed_params = {"value": parameters}
                    elif isinstance(parameters, dict):
                        # Already a dict, use directly
                        parsed_params = parameters
                        logger.debug(f"Using parameters as dict: {parsed_params}")
                    else:
                        # Other types (like None) - use empty dict
                        logger.debug(f"Using empty params for parameters type: {type(parameters)}")
                except Exception as e:
                    logger.error(f"Error parsing parameters: {str(e)}")
                    error_result = FunctionResult(
                        status="error",
                        data=None,
                        message=f"Error parsing parameters: {str(e)}",
                        error_code="PARAMETER_PARSING_ERROR",
                        error_details={"exception": str(e)}
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
                        message=f"Parameter validation failed for tool '{tool_name}'",
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
            
            # Execute the tool with the parsed parameters (not as 'parameters' kwarg)
            result = await registry.execute(tool_name, **parsed_params)
            
            # Instead of returning the wrapped result, return just the actual tool output
            if hasattr(result, 'data') and result.data is not None and 'result' in result.data:
                # If there's a 'result' field in the data, return just that
                return result.data['result']
            elif hasattr(result, 'data') and result.data is not None:
                # If there's data but no 'result' field, return the data
                return json.dumps(result.data)
            else:
                # Otherwise, return the full result as JSON
                return result.to_json()
        except Exception as e:
            logger.error(f"Error executing tool: {str(e)}")
            error_result = FunctionResult(
                status="error",
                data=None,
                message=f"Error executing tool: {str(e)}",
                error_code="TOOL_EXECUTION_ERROR",
                error_details={"exception": str(e)}
            )
            return error_result.to_json()

    # Tool 2: list_available_tools
    @server.tool()  # type: ignore
    async def list_available_tools(category = None) -> str:
        """
        List all available tools, optionally filtered by category.
        
        This tool allows discovery of all registered tools and their documentation.
        
        Args:
            category: Optional category to filter by
            
        Returns:
            JSON string with tool metadata
        """
        # Get the registry instance
        registry = get_registry()
        
        try:
            if category:
                tools = registry.get_tools_by_namespace(category)
            else:
                tools = registry.get_all_tools()
                
            # Convert to dictionary for better serialization
            result = {
                "tools": [t.model_dump() for t in tools],
                "count": len(tools),
                "categories": list(registry.get_namespaces())
            }
            
            return json.dumps(result, indent=2)
        except Exception as e:
            error_result = {
                "error": str(e),
                "tools": []
            }
            return json.dumps(error_result)

    # Tool 3: list_tool_categories
    @server.tool()  # type: ignore
    async def list_tool_categories() -> str:
        """
        List all available tool categories/namespaces.
        
        Returns:
            JSON string with category information
        """
        # Get the registry instance
        registry = get_registry()
        
        try:
            namespaces = registry.get_namespaces()
            
            # Get tool count per namespace
            category_counts = {}
            for ns in namespaces:
                tools = registry.get_tools_by_namespace(ns)
                category_counts[ns] = len(tools)
            
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
            logger.info(f"Registry now contains {stats['total_functions']} tools in {len(stats['namespaces'])} namespaces")
            logger.info(f"Registry breakdown:")
            for ns, count in stats['namespace_counts'].items():
                logger.info(f"  - {ns}: {count} tools")

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