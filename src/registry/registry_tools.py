#!/usr/bin/env python3
"""
Registry Tools

This module provides MCP tools for the Tool Registry Pattern. These tools
expose the registry functionality to AI agents through a unified interface.
"""

import json
import asyncio
import logging
import os
from typing import Dict, List, Any, Optional, Union, cast
import time

from src.registry.registry_manager import get_registry, ToolRegistry
from src.registry.function_models import FunctionResult, ToolParameters, ToolMetadata
from src.registry.parameter_helper import ParameterHelper, ValidationError as ParameterValidationError
from src.registry.migration_framework import migrate_all_tools, MigrationManager
from src.registry.documentation_generator import generate_documentation, export_documentation_markdown, export_documentation_json
from src.registry.advanced_parameter_handler import parse_parameters, validate_and_convert, register_context_provider
from src.logger import get_logger

# Define fallback classes if imports not available
class FunctionRecommendationEngine:
    """Simple recommendation engine for tracking tool usage."""
    def track_tool_usage(self, tool_name, params, context, success=True):
        pass
    
    def get_tool_chains(self, query, context=None):
        return []

class ResultCache:
    """Simple result cache implementation."""
    def get(self, tool_name, params):
        return None
    
    def set(self, tool_name, params, result):
        pass
    
    def invalidate(self, tool_name=None):
        return 0
        
    def set_tool_cache_settings(self, tool_name, ttl, cacheable=True):
        pass

# Use the fallback implementations by default
recommendation_engine = FunctionRecommendationEngine()
result_cache = ResultCache()

# Import IDE integration tools
try:
    from src.registry.ide_integration import generate_ide_optimized_tools, export_ide_optimized_tools
except ImportError:
    generate_ide_optimized_tools = None
    export_ide_optimized_tools = None

# Import performance optimization tools
try:
    from src.registry.performance_optimization import (
        get_result_cache,
        get_batch_processor,
        get_parameter_serializer
    )
except ImportError:
    get_result_cache = None
    get_batch_processor = None
    get_parameter_serializer = None

# Import agent guidance tools
try:
    from src.registry.agent_guidance import (
        get_recommendation_engine,
        get_pattern_detector,
        get_complex_helper
    )
except ImportError:
    get_recommendation_engine = None
    get_pattern_detector = None
    get_complex_helper = None

# Initialize logger
logger = logging.getLogger(__name__)

# Helper function for safe conversion of validation errors
def _format_validation_error(error: Any) -> Dict[str, str]:
    """Safely format a validation error object to a dictionary, regardless of its type."""
    try:
        # Handle parameter_helper.ValidationError
        if isinstance(error, ParameterValidationError) and hasattr(error, "to_dict"):
            return error.to_dict()
        
        # Create a basic error dict with safe attribute access
        result = {}
        
        # Always include param_name
        result["param_name"] = getattr(error, "param_name", "unknown")
            
        # Try to get error message
        if hasattr(error, "error_message"):
            result["message"] = str(getattr(error, "error_message"))
        elif hasattr(error, "message"):  # type: ignore
            result["message"] = str(getattr(error, "message"))  # type: ignore
        else:
            result["message"] = str(error)
            
        # Try to get error code/type
        if hasattr(error, "error_code"):
            result["code"] = str(getattr(error, "error_code"))
        elif hasattr(error, "error_type"):  # type: ignore
            result["code"] = str(getattr(error, "error_type"))  # type: ignore
        else:
            result["code"] = "validation_error"
            
        return result
    except Exception:
        # Last resort fallback
        return {
            "param_name": "unknown",
            "message": str(error),
            "code": "validation_error"
        }

def register_registry_tools(server, get_client_manager=None):
    """
    Register tool registry tools with the server.
    
    Args:
        server: The MCP server instance
        get_client_manager: Optional function to get client manager (not used by registry)
    """
    # Access the global registry
    registry = get_registry()
    
    @server.tool()
    async def execute_tool(
        tool_name: str,
        params: Optional[Dict[str, Any]] = None,
        context: Optional[Dict[str, Any]] = None,
        skip_cache: bool = False
    ) -> FunctionResult:
        """
        Execute a registered tool with the given parameters.
        
        Args:
            tool_name: The name of the tool to execute
            params: The parameters to pass to the tool
            context: Additional context for tool execution
            skip_cache: Flag to skip result caching for this call
            
        Returns:
            FunctionResult object containing the execution result
        """
        params_dict = params or {}
        context_dict = context or {}
        
        registry = get_registry()
        
        # Get metadata for the requested tool
        tool_metadata = registry.get_tool_metadata(tool_name)
        if not tool_metadata:
            return FunctionResult(
                status="error",
                message=f"Tool '{tool_name}' not found",
                data=None,
                error_code="TOOL_NOT_FOUND",
                error_details={"tool_name": tool_name, "params": params_dict}
            )
        
        # Check for required parameters
        missing_params = []
        for param_name, param_info in tool_metadata.parameters.items():
            if param_info.get("required", False) and param_name not in params_dict:
                missing_params.append(param_name)
        
        if missing_params:
            return FunctionResult(
                status="error",
                message=f"Missing required parameters: {', '.join(missing_params)}",
                data=None,
                error_code="MISSING_PARAMETERS",
                error_details={"tool_name": tool_name, "params": params_dict, "missing": missing_params}
            )
        
        # Check cache for existing result
        if not skip_cache:
            cached_result = result_cache.get(tool_name, params_dict)
            if cached_result:
                # Track the execution if analytics is enabled
                recommendation_engine.track_tool_usage(tool_name, params_dict, context_dict)
                return cached_result
        
        # Execute the tool
        start_time = time.time()
        try:
            # Use the registry's execute method to run the tool
            result_obj = await registry.execute(tool_name, **params_dict)
            
            # If the result is already a FunctionResult, return it
            if isinstance(result_obj, FunctionResult):
                # Cache the result if caching is enabled
                if not skip_cache:
                    result_cache.set(tool_name, params_dict, result_obj)
                
                # Track the execution if analytics is enabled
                recommendation_engine.track_tool_usage(tool_name, params_dict, context_dict)
                
                return result_obj
            
            # Create a successful result object
            func_result = FunctionResult(
                status="success",
                message=f"Tool '{tool_name}' executed successfully",
                data=result_obj,
                error_code=None,
                error_details=None
            )
            
            # Cache the result if caching is enabled
            if not skip_cache:
                result_cache.set(tool_name, params_dict, func_result)
            
            # Track the execution if analytics is enabled
            recommendation_engine.track_tool_usage(tool_name, params_dict, context_dict)
            
            return func_result
        except Exception as e:
            # Create an error result object
            error_result = FunctionResult(
                status="error",
                message=str(e),
                data=None,
                error_code="EXECUTION_ERROR",
                error_details={"tool_name": tool_name, "params": params_dict}
            )
            
            # Track the failed execution if analytics is enabled
            recommendation_engine.track_tool_usage(tool_name, params_dict, context_dict, success=False)
            
            return error_result
    
    @server.tool()
    async def list_available_tools(category: Optional[str] = None) -> str:
        """
        List all available tools, optionally filtered by category.
        
        This tool allows discovery of all registered tools and their documentation.
        
        Args:
            category: Optional category to filter by
            
        Returns:
            JSON string with tool metadata
        """
        try:
            if category:
                tools = registry.get_tools_by_namespace(category)
            else:
                tools = registry.get_all_tools()
                
            # Convert to dictionary for better serialization
            result = {
                "tools": [f.model_dump() for f in tools],
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
    
    @server.tool()
    async def get_tool_details(tool_name: str) -> str:
        """
        Get detailed information about a specific tool.
        
        Args:
            tool_name: Name of the tool to get details for
            
        Returns:
            JSON string with tool details
        """
        try:
            metadata = registry.get_tool_metadata(tool_name)
            
            if not metadata:
                return json.dumps({
                    "error": f"Tool '{tool_name}' not found",
                    "tool_name": tool_name
                })
                
            return json.dumps(metadata.model_dump(), indent=2)
        except Exception as e:
            error_result = {
                "error": str(e),
                "tool_name": tool_name
            }
            return json.dumps(error_result)
    
    @server.tool()
    async def list_tool_categories() -> str:
        """
        List all available tool categories/namespaces.
        
        Returns:
            JSON string with category information
        """
        try:
            namespaces = registry.get_namespaces()
            
            # Count tools in each namespace
            namespace_counts = {}
            for ns in namespaces:
                tools = registry.get_tools_by_namespace(ns)
                namespace_counts[ns] = len(tools)
                
            result = {
                "categories": namespaces,
                "tool_counts": namespace_counts,
                "total_categories": len(namespaces)
            }
            
            return json.dumps(result, indent=2)
        except Exception as e:
            error_result = {
                "error": str(e),
                "categories": []
            }
            return json.dumps(error_result)
    
    @server.tool()
    async def execute_tool_bundle(bundle_name: str, context: Dict[str, Any]) -> str:
        """
        Execute a pre-defined sequence of tools with shared context.
        
        Bundles allow multiple related tools to be executed in sequence,
        with results from earlier tools available to later ones.
        
        Args:
            bundle_name: Name of the tool bundle to execute
            context: Initial context for the bundle execution
            
        Returns:
            JSON string with results from all tools in the bundle
        """
        # Define bundles - in a real implementation, these would be loaded from a config
        bundles = {
            "create_project": [
                {
                    "tool": "project.create_container", 
                    "params_map": {"name": "project_name"}
                },
                {
                    "tool": "project.create_component", 
                    "params_map": {"project_id": "project_name", "name": "component_name"}
                }
            ],
            "create_entity_with_observation": [
                {
                    "tool": "memory.create_entity",
                    "params_map": {"name": "entity_name", "entity_type": "entity_type"}
                },
                {
                    "tool": "memory.create_observation",
                    "params_map": {
                        "entity_name": "entity_name",
                        "content": "observation_content"
                    }
                }
            ]
        }
        
        if bundle_name not in bundles:
            error_result = FunctionResult.error(
                message=f"Bundle '{bundle_name}' not found",
                error_code="bundle_not_found",
                error_details={"available_bundles": list(bundles.keys())}
            )
            return error_result.to_json()
        
        # Execute the bundle
        results = {}
        bundle_context = context.copy()  # Make a copy to avoid modifying the original
        
        for step in bundles[bundle_name]:
            tool_name = step["tool"]
            params_map = step.get("params_map", {})
            
            # Map parameters from context
            params = {}
            for param_key, context_key in params_map.items():
                if context_key in bundle_context:
                    params[param_key] = bundle_context[context_key]
            
            # Execute the tool
            result = await registry.execute(tool_name, **params)
            results[tool_name] = json.loads(result.to_json())
            
            # Update context with results for subsequent steps
            if result.status == "success" and result.data:
                bundle_context.update(result.data)
        
        # Create the final result
        bundle_result = FunctionResult.success(
            message=f"Successfully executed bundle '{bundle_name}'",
            data={
                "bundle_name": bundle_name,
                "results": results,
                "final_context": bundle_context
            }
        )
        
        return bundle_result.to_json()
    
    @server.tool()
    async def get_parameter_suggestions(tool_name: str) -> str:
        """
        Get parameter suggestions for a tool.
        
        This tool helps AI agents understand how to use tools by providing
        type-specific parameter suggestions.
        
        Args:
            tool_name: Name of the tool to get suggestions for
            
        Returns:
            JSON string with parameter suggestions
        """
        try:
            metadata = registry.get_tool_metadata(tool_name)
            
            if not metadata:
                error_result = {
                    "error": f"Tool '{tool_name}' not found",
                    "tool_name": tool_name
                }
                return json.dumps(error_result, indent=2)
            
            # Generate suggestions
            suggestions = ParameterHelper.generate_parameter_suggestions(metadata)
            
            result = {
                "tool_name": tool_name,
                "parameter_suggestions": suggestions,
                "required_parameters": [
                    name for name, info in metadata.parameters.items()
                    if info.get("required", False)
                ]
            }
            
            return json.dumps(result, indent=2)
        except Exception as e:
            error_result = {
                "error": str(e),
                "tool_name": tool_name
            }
            return json.dumps(error_result, indent=2)
    
    @server.tool()
    async def migrate_tools(module_paths: Optional[List[str]] = None) -> str:
        """
        Migrate existing tools to the tool registry.
        
        This tool allows migrating tool modules to the registry, making them
        available through the unified interface.
        
        Args:
            module_paths: Optional list of module paths to migrate (defaults to all standard tools)
            
        Returns:
            JSON string with migration results
        """
        try:
            if module_paths:
                # Migrate specific modules
                manager = MigrationManager()
                results = {}
                for module_path in module_paths:
                    result = manager.migrate_module(module_path)
                    results[module_path] = result
            else:
                # Migrate all standard tools
                results = migrate_all_tools()
                
            return json.dumps({
                "status": "success",
                "migrated_modules": list(results.keys()),
                "details": results
            }, indent=2)
        except Exception as e:
            error_result = {
                "status": "error",
                "message": f"Error migrating tools: {str(e)}"
            }
            return json.dumps(error_result, indent=2)
    
    @server.tool()
    async def generate_ide_tools(export_path: Optional[str] = None) -> str:
        """
        Generate IDE-optimized tool definitions.
        
        This tool creates a set of IDE-friendly tool definitions, which can be
        used directly in IDEs with limited tool expansion capabilities.
        
        Args:
            export_path: Optional path to export the definitions as JSON
            
        Returns:
            JSON string with generated tools or path to exported file
        """
        try:
            # Generate the optimized tools
            if generate_ide_optimized_tools is not None:
                tools = generate_ide_optimized_tools()
                
                # Export if requested
                if export_path:
                    if export_ide_optimized_tools is not None:
                        result_path = export_ide_optimized_tools(export_path)
                        return json.dumps({
                            "status": "success",
                            "message": f"IDE optimized tools exported to {result_path}",
                            "export_path": result_path
                        })
                    else:
                        return json.dumps({
                            "status": "error",
                            "message": "export_ide_optimized_tools function is not available"
                        })
                
                # Return the tools directly
                return json.dumps({
                    "status": "success",
                    "message": f"Generated {len(tools['category_tools'])} IDE-optimized tools",
                    "tools_count": len(tools['category_tools'])
                })
            else:
                return json.dumps({
                    "status": "error",
                    "message": "IDE integration module not available"
                })
        except Exception as e:
            return json.dumps({
                "status": "error",
                "message": f"Error generating IDE tools: {str(e)}"
            })
    
    @server.tool()
    async def create_category_tool(namespace: str) -> str:
        """
        Create a category-specific tool for IDE integration.
        
        This tool creates a specialized tool definition for a specific namespace,
        which can be used in IDEs with limited tool expansion capabilities.
        
        Args:
            namespace: The namespace to create a tool for
            
        Returns:
            JSON string with the tool definition
        """
        try:
            # Check if namespace exists
            tools = registry.get_tools_by_namespace(namespace)
            if not tools:
                return json.dumps({
                    "status": "error",
                    "message": f"Namespace '{namespace}' not found or has no tools"
                })
            
            # Generate the IDE metadata
            from src.registry.ide_integration import IDEMetadataGenerator
            generator = IDEMetadataGenerator()
            
            # Generate category tools and find the one for our namespace
            category_tools = generator.generate_category_tools()
            
            if namespace not in category_tools:
                return json.dumps({
                    "status": "error",
                    "message": f"Failed to generate category tool for '{namespace}'"
                })
                
            # Return the specific tool
            return json.dumps({
                "status": "success",
                "message": f"Generated category tool for '{namespace}'",
                "tool": category_tools[namespace]
            })
        except Exception as e:
            return json.dumps({
                "status": "error",
                "message": f"Error creating category tool: {str(e)}"
            })

    @server.tool()
    async def generate_tool_documentation(format: str = "json", output_path: Optional[str] = None, version: str = "1.0.0") -> str:
        """
        Generate comprehensive documentation for all registered tools.
        
        This tool creates documentation for all tools in the registry and exports
        it in the specified format.
        
        Args:
            format: Output format ("json" or "markdown")
            output_path: Path to export the documentation
            version: Documentation version
            
        Returns:
            JSON string with the documentation or path to the exported files
        """
        try:
            # Check format
            if format.lower() not in ["json", "markdown"]:
                return json.dumps({
                    "status": "error",
                    "message": f"Invalid format '{format}'. Must be 'json' or 'markdown'."
                })
            
            # Generate the documentation
            if format.lower() == "json":
                if not output_path:
                    # If no path is provided, just generate and return
                    docs = generate_documentation(version)
                    return json.dumps({
                        "status": "success",
                        "message": f"Generated documentation for {docs['meta']['tool_count']} tools",
                        "documentation": docs
                    })
                else:
                    # Export to file
                    json_path = export_documentation_json(output_path, version)
                    return json.dumps({
                        "status": "success",
                        "message": f"Documentation exported to {json_path}",
                        "export_path": json_path
                    })
            else:  # markdown
                if not output_path:
                    return json.dumps({
                        "status": "error",
                        "message": "output_path is required for markdown format"
                    })
                
                # Export to markdown
                index_path = export_documentation_markdown(output_path, version)
                return json.dumps({
                    "status": "success",
                    "message": f"Documentation exported to {index_path}",
                    "export_path": index_path
                })
        except Exception as e:
            return json.dumps({
                "status": "error",
                "message": f"Error generating documentation: {str(e)}"
            })
    
    @server.tool()
    async def get_tool_examples(tool_name: str) -> str:
        """
        Get usage examples for a specific tool.
        
        This tool generates usage examples for a tool, including basic
        examples and specialized examples for common tools.
        
        Args:
            tool_name: Name of the tool to get examples for
            
        Returns:
            JSON string with usage examples
        """
        try:
            # Check if tool exists
            metadata = registry.get_tool_metadata(tool_name)
            if not metadata:
                return json.dumps({
                    "status": "error",
                    "message": f"Tool '{tool_name}' not found"
                })
            
            # Create a documentation generator
            from src.registry.documentation_generator import DocumentationGenerator
            generator = DocumentationGenerator()
            
            # Generate examples for the tool
            examples = generator._generate_examples(metadata)
            
            return json.dumps({
                "status": "success",
                "message": f"Generated {len(examples)} examples for {tool_name}",
                "tool_name": tool_name,
                "examples": examples
            })
        except Exception as e:
            return json.dumps({
                "status": "error",
                "message": f"Error generating examples: {str(e)}"
            })

    @server.tool()
    async def parse_natural_language_parameters(tool_name: str, text: str) -> str:
        """
        Parse parameters for a tool from natural language text.
        
        This tool extracts parameter values from natural language descriptions,
        making it easier to use tools in conversational contexts.
        
        Args:
            tool_name: Name of the tool to parse parameters for
            text: Natural language text containing parameter values
            
        Returns:
            JSON string with extracted parameters
        """
        try:
            # Check if tool exists
            metadata = registry.get_tool_metadata(tool_name)
            if not metadata:
                return json.dumps({
                    "status": "error",
                    "message": f"Tool '{tool_name}' not found"
                })
            
            # Parse parameters from text
            try:
                from src.registry.advanced_parameter_handler import parse_parameters
                parameters = parse_parameters(metadata, text)
                
                return json.dumps({
                    "status": "success",
                    "message": f"Extracted parameters for '{tool_name}'",
                    "tool_name": tool_name,
                    "parameters": parameters,
                    "parameter_count": len(parameters)
                })
            except ImportError:
                return json.dumps({
                    "status": "error",
                    "message": "Advanced parameter handling module not available"
                })
        except Exception as e:
            return json.dumps({
                "status": "error",
                "message": f"Error parsing parameters: {str(e)}"
            })
    
    @server.tool()
    async def register_user_context(namespace: str, context_data: Dict[str, Any]) -> str:
        """
        Register user context data for parameter handling.
        
        This tool allows storing context information that can be used
        to provide defaults for tool parameters.
        
        Args:
            namespace: Context namespace (e.g., 'user', 'project')
            context_data: Dictionary of context data
            
        Returns:
            JSON string with registration result
        """
        try:
            # Register the context provider
            try:
                from src.registry.advanced_parameter_handler import register_context_provider
                
                # Create a provider function that returns the provided data
                provider_func = lambda: context_data.copy()
                
                # Register the provider
                register_context_provider(namespace, provider_func)
                
                return json.dumps({
                    "status": "success",
                    "message": f"Registered context provider for namespace '{namespace}'",
                    "namespace": namespace,
                    "context_keys": list(context_data.keys())
                })
            except ImportError:
                return json.dumps({
                    "status": "error",
                    "message": "Advanced parameter handling module not available"
                })
        except Exception as e:
            return json.dumps({
                "status": "error",
                "message": f"Error registering context: {str(e)}"
            })

    @server.tool()
    async def get_tool_recommendations(query: str, context: Optional[Dict[str, Any]] = None, limit: int = 5) -> str:
        """
        Get tool recommendations based on a natural language query.
        
        This tool helps suggest relevant tools based on what you're trying to do,
        using contextual understanding and usage patterns.
        
        Args:
            query: Natural language description of what you want to do
            context: Optional context information (e.g., current entities, data)
            limit: Maximum number of recommendations to return
            
        Returns:
            JSON string with tool recommendations
        """
        try:
            # Check if recommendation engine is available
            if get_recommendation_engine is None:
                return json.dumps({
                    "status": "error",
                    "message": "Tool recommendation engine is not available"
                })
                
            # Get recommendations
            engine = get_recommendation_engine()
            recommendations = engine.get_recommendations(query, context, limit)
            
            return json.dumps({
                "status": "success",
                "message": f"Found {len(recommendations)} tool recommendations",
                "recommendations": recommendations
            })
        except Exception as e:
            return json.dumps({
                "status": "error",
                "message": f"Error getting tool recommendations: {str(e)}"
            })
    
    @server.tool()
    async def get_tool_chains(query: str, context: Optional[Dict[str, Any]] = None, limit: int = 3) -> List[Dict[str, Any]]:
        """
        Get recommended tool chains for accomplishing a goal.
        
        Args:
            query: The query describing the goal
            context: Additional context for recommendations
            limit: Maximum number of chains to return
            
        Returns:
            List of recommended tool chains, where each chain contains information about the tools
        """
        try:
            # Check if recommendation engine is available
            if get_recommendation_engine is None:
                return []
                
            # Get the global recommendation engine instance
            engine = get_recommendation_engine()
            return engine.get_tool_chains(query, limit)
        except Exception as e:
            logger.error(f"Error getting tool chains: {str(e)}")
            return []
    
    @server.tool()
    async def track_tool_usage(
        tool_name: str,
        params: Dict[str, Any],
        context: Optional[Dict[str, Any]] = None,
        success: bool = True
    ) -> None:
        """
        Track tool usage for analytics and recommendations.
        
        Args:
            tool_name: The name of the tool that was used
            params: The parameters that were passed to the tool
            context: Additional context for the tool usage
            success: Whether the tool execution was successful
        """
        recommendation_engine = FunctionRecommendationEngine()
        recommendation_engine.track_tool_usage(tool_name, params, context, success)
    
    @server.tool()
    async def end_tool_sequence() -> str:
        """
        End the current tool call sequence.
        
        This tool marks the end of a logical sequence of tool calls,
        which helps with pattern detection and recommendations.
        
        Returns:
            JSON string with confirmation
        """
        try:
            # Check if pattern detector is available
            if get_pattern_detector is None:
                return json.dumps({
                    "status": "error",
                    "message": "Pattern detector is not available"
                })
                
            # End sequence
            detector = get_pattern_detector()
            detector.end_sequence()
            
            return json.dumps({
                "status": "success",
                "message": "Tool sequence ended"
            })
        except Exception as e:
            return json.dumps({
                "status": "error",
                "message": f"Error ending tool sequence: {str(e)}"
            })
    
    @server.tool()
    async def get_detected_patterns() -> str:
        """
        Get detected tool usage patterns.
        
        This tool provides insights into common patterns of tool usage,
        which can help optimize your approach.
        
        Returns:
            JSON string with detected patterns
        """
        try:
            # Check if pattern detector is available
            if get_pattern_detector is None:
                return json.dumps({
                    "status": "error",
                    "message": "Pattern detector is not available"
                })
                
            # Get patterns
            detector = get_pattern_detector()
            patterns = detector.get_detected_patterns()
            anti_patterns = detector.get_detected_anti_patterns()
            
            return json.dumps({
                "status": "success",
                "message": f"Found {len(patterns)} usage patterns and {len(anti_patterns)} anti-patterns",
                "patterns": patterns,
                "anti_patterns": anti_patterns
            })
        except Exception as e:
            return json.dumps({
                "status": "error",
                "message": f"Error getting detected patterns: {str(e)}"
            })
    
    @server.tool()
    async def execute_complex_operation(operation_name: str, parameters: Dict[str, Any]) -> str:
        """
        Execute a complex operation with multiple steps.
        
        This tool provides high-level operations that combine multiple
        tools in a coordinated way.
        
        Args:
            operation_name: Name of the complex operation
            parameters: Parameters for the operation
            
        Returns:
            JSON string with operation results
        """
        try:
            # Check if complex helper is available
            if get_complex_helper is None:
                return json.dumps({
                    "status": "error",
                    "message": "Complex operation helper is not available"
                })
                
            # Execute operation
            helper = get_complex_helper()
            result = await helper.execute_complex_operation(operation_name, parameters)
            
            return result.to_json()
        except Exception as e:
            error_result = FunctionResult(
                status="error",
                message=f"Error executing complex operation: {str(e)}",
                data=None,
                error_code="OPERATION_EXECUTION_ERROR",
                error_details={"exception": str(e)}
            )
            return error_result.to_json()
    
    @server.tool()
    async def get_available_complex_operations() -> str:
        """
        Get all available complex operations.
        
        This tool lists all the high-level operations you can perform
        with the execute_complex_operation tool.
        
        Returns:
            JSON string with available operations
        """
        try:
            # Check if complex helper is available
            if get_complex_helper is None:
                return json.dumps({
                    "status": "error",
                    "message": "Complex operation helper is not available"
                })
                
            # Get available operations
            helper = get_complex_helper()
            operations = helper.get_available_operations()
            
            return json.dumps({
                "status": "success",
                "message": f"Found {sum(len(info['operations']) for info in operations.values())} complex operations",
                "operations": operations
            })
        except Exception as e:
            return json.dumps({
                "status": "error",
                "message": f"Error getting available operations: {str(e)}"
            })
    
    @server.tool()
    async def get_complex_operation_help(operation_name: str) -> str:
        """
        Get help information for a complex operation.
        
        This tool provides detailed information about how to use a
        specific complex operation.
        
        Args:
            operation_name: Name of the operation to get help for
            
        Returns:
            JSON string with operation help information
        """
        try:
            # Check if complex helper is available
            if get_complex_helper is None:
                return json.dumps({
                    "status": "error",
                    "message": "Complex operation helper is not available"
                })
                
            # Get operation help
            helper = get_complex_helper()
            help_info = helper.get_operation_help(operation_name)
            
            return json.dumps({
                "status": "success",
                "operation": operation_name,
                "help": help_info
            })
        except Exception as e:
            return json.dumps({
                "status": "error",
                "message": f"Error getting operation help: {str(e)}"
            })

    # Performance Optimization Tools
    @server.tool()
    async def execute_batch(batch: List[Dict[str, Any]], parallel: bool = True, use_cache: bool = True) -> str:
        """
        Execute a batch of tool calls.
        
        Args:
            batch: List of tool call specifications
            parallel: Whether to execute calls in parallel when possible
            use_cache: Whether to use the result cache
            
        Returns:
            JSON string with batch execution results
        """
        # Check if batch processor is available
        if get_batch_processor is None:
            error_result = FunctionResult(
                status="error",
                data=None,
                message="Batch processing is not available",
                error_code="BATCH_PROCESSOR_UNAVAILABLE",
                error_details={"reason": "Performance optimization module is not imported"}
            )
            return error_result.to_json()
        
        try:
            # Get the batch processor
            batch_processor = get_batch_processor()
            if batch_processor is None:
                error_result = FunctionResult(
                    status="error",
                    data=None,
                    message="Batch processor initialization failed",
                    error_code="BATCH_PROCESSOR_INIT_FAILED",
                    error_details={"reason": "Could not initialize batch processor"}
                )
                return error_result.to_json()
            
            # Execute the batch
            results = await batch_processor.execute_batch(batch, parallel, use_cache)
            
            result = FunctionResult(
                status="success",
                data={"results": results},
                message=f"Executed batch of {len(batch)} tool calls",
                error_code="",
                error_details={}
            )
            return result.to_json()
        except Exception as e:
            logger.error(f"Error executing batch: {str(e)}")
            error_result = FunctionResult(
                status="error",
                data=None,
                message=f"Error executing batch: {str(e)}",
                error_code="BATCH_EXECUTION_ERROR",
                error_details={"exception": str(e)}
            )
            return error_result.to_json()

    @server.tool()
    async def configure_result_cache(
        tool_name: str,
        ttl: int,
        cacheable: bool = True
    ) -> None:
        """
        Configure cache settings for a specific tool.
        
        Args:
            tool_name: The name of the tool
            ttl: Time-to-live in seconds for this tool's results
            cacheable: Whether this tool's results should be cached
        """
        result_cache = ResultCache()
        result_cache.set_tool_cache_settings(tool_name, ttl, cacheable)

    @server.tool()
    async def invalidate_cache(tool_name: Optional[str] = None) -> str:
        """
        Invalidate cache entries for a tool or all tools.
        
        Args:
            tool_name: Name of the tool or None to invalidate all
            
        Returns:
            JSON string with number of invalidated entries
        """
        # Check if cache is available
        if get_result_cache is None:
            error_result = FunctionResult(
                status="error",
                data=None,
                message="Result caching is not available",
                error_code="CACHE_UNAVAILABLE",
                error_details={"reason": "Performance optimization module is not imported"}
            )
            return error_result.to_json()
        
        try:
            # Get the cache
            cache = get_result_cache()
            if cache is None:
                error_result = FunctionResult(
                    status="error",
                    data=None,
                    message="Cache initialization failed",
                    error_code="CACHE_INIT_FAILED",
                    error_details={"reason": "Could not initialize cache"}
                )
                return error_result.to_json()
            
            # If tool name provided, check if it exists
            if tool_name is not None:
                registry = get_registry()
                tool_metadata = registry.get_tool_metadata(tool_name)
                if tool_metadata is None:
                    error_result = FunctionResult(
                        status="error",
                        data=None,
                        message=f"Tool '{tool_name}' does not exist",
                        error_code="TOOL_NOT_FOUND",
                        error_details={"tool_name": tool_name}
                    )
                    return error_result.to_json()
            
            # Invalidate cache entries
            count = cache.invalidate(tool_name)
            
            if tool_name is None:
                message = f"Invalidated all cache entries ({count} total)"
            else:
                message = f"Invalidated {count} cache entries for '{tool_name}'"
            
            result = FunctionResult(
                status="success",
                data={"invalidated_count": count},
                message=message,
                error_code="",
                error_details={}
            )
            return result.to_json()
        except Exception as e:
            logger.error(f"Error invalidating cache: {str(e)}")
            error_result = FunctionResult(
                status="error",
                data=None,
                message=f"Error invalidating cache: {str(e)}",
                error_code="CACHE_INVALIDATION_ERROR",
                error_details={"exception": str(e)}
            )
            return error_result.to_json()

    @server.tool()
    async def register_parameter_serializer(
        type_name: str, 
        serialize_code: str, 
        deserialize_code: str
    ) -> str:
        """
        Register a custom serializer for a parameter type.
        
        Args:
            type_name: Name of the parameter type
            serialize_code: Python code for serializer function (must take one argument and return serialized form)
            deserialize_code: Python code for deserializer function (must take one argument and return native type)
            
        Returns:
            JSON string with confirmation of serializer registration
        """
        # Check if parameter serializer is available
        if get_parameter_serializer is None:
            error_result = FunctionResult(
                status="error",
                data=None,
                message="Parameter serialization is not available",
                error_code="SERIALIZER_UNAVAILABLE",
                error_details={"reason": "Performance optimization module is not imported"}
            )
            return error_result.to_json()
        
        try:
            # Compile the serializer code
            serialize_locals = {}
            exec(serialize_code, globals(), serialize_locals)
            if "serializer" not in serialize_locals:
                error_result = FunctionResult(
                    status="error",
                    data=None,
                    message="Serializer code must define a 'serializer' function",
                    error_code="INVALID_SERIALIZER_CODE",
                    error_details={"reason": "No serializer function defined"}
                )
                return error_result.to_json()
            serializer = serialize_locals["serializer"]
            
            # Compile the deserializer code
            deserialize_locals = {}
            exec(deserialize_code, globals(), deserialize_locals)
            if "deserializer" not in deserialize_locals:
                error_result = FunctionResult(
                    status="error",
                    data=None,
                    message="Deserializer code must define a 'deserializer' function",
                    error_code="INVALID_DESERIALIZER_CODE",
                    error_details={"reason": "No deserializer function defined"}
                )
                return error_result.to_json()
            deserializer = deserialize_locals["deserializer"]
            
            # Register the serializer
            parameter_serializer = get_parameter_serializer()
            parameter_serializer.register_serializer(type_name, serializer, deserializer)
            
            result = FunctionResult(
                status="success",
                data={"type_name": type_name},
                message=f"Registered custom serializer for type '{type_name}'",
                error_code="",
                error_details={}
            )
            return result.to_json()
        except Exception as e:
            logger.error(f"Error registering parameter serializer: {str(e)}")
            error_result = FunctionResult(
                status="error",
                data=None,
                message=f"Error registering parameter serializer: {str(e)}",
                error_code="SERIALIZER_REGISTRATION_ERROR",
                error_details={"exception": str(e)}
            )
            return error_result.to_json() 