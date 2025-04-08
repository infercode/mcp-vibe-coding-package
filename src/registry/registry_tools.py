#!/usr/bin/env python3
"""
Registry Tools

This module provides MCP tools for the Function Registry Pattern. These tools
expose the registry functionality to AI agents through a unified interface.
"""

import json
import asyncio
import logging
import os
from typing import Dict, List, Any, Optional, Union, cast

from src.registry.registry_manager import get_registry, FunctionRegistry
from src.registry.function_models import FunctionResult, FunctionParameters, FunctionMetadata
from src.registry.parameter_helper import ParameterHelper, ValidationError as ParameterValidationError
from src.registry.migration_framework import migrate_all_tools, MigrationManager
from src.registry.ide_integration import generate_ide_optimized_tools, export_ide_optimized_tools
from src.registry.documentation_generator import generate_documentation, export_documentation_markdown, export_documentation_json
from src.registry.advanced_parameter_handler import parse_parameters, validate_and_convert, register_context_provider
from src.logger import get_logger

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

# Import advanced parameter handling
try:
    from src.registry.advanced_parameter_handler import (
        parse_parameters,
        validate_and_convert,
        register_context_provider
    )
except ImportError:
    parse_parameters = None
    validate_and_convert = None
    register_context_provider = None

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
    Register function registry tools with the server.
    
    Args:
        server: The MCP server instance
        get_client_manager: Optional function to get client manager (not used by registry)
    """
    # Access the global registry
    registry = get_registry()
    
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
        if registry is None:
            error_result = FunctionResult(
                status="error",
                data=None,
                message="Registry not available",
                error_code="REGISTRY_UNAVAILABLE",
                error_details={"reason": "Registry could not be initialized"}
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
                    error_details={"available_functions": list(registry.get_all_functions())}
                )
                return error_result.to_json()
            
            # Try advanced parameter handling if available
            try:
                if parse_parameters is not None and validate_and_convert is not None:
                    # Parse parameters from various input formats
                    params = parse_parameters(metadata, parameters)
                    
                    # Validate and convert parameters
                    converted_params, validation_errors = validate_and_convert(metadata, params)
                    
                    if validation_errors:
                        # Format errors safely
                        formatted_errors = [_format_validation_error(err) for err in validation_errors]
                        
                        error_result = FunctionResult(
                            status="error",
                            message=f"Parameter validation failed for function '{function_name}'",
                            data=None,
                            error_code="PARAMETER_VALIDATION_ERROR",
                            error_details={"validation_errors": formatted_errors}
                        )
                        return error_result.to_json()
                    
                    # Use the converted parameters
                    parameters = converted_params
                else:
                    # Fall back to basic parameter handling
                    validation_errors = ParameterHelper.validate_parameters(metadata, parameters)
                    if validation_errors:
                        # Format errors safely
                        formatted_errors = [_format_validation_error(err) for err in validation_errors]
                        
                        error_result = FunctionResult(
                            status="error",
                            message=f"Parameter validation failed for function '{function_name}'",
                            data=None,
                            error_code="PARAMETER_VALIDATION_ERROR",
                            error_details={"validation_errors": formatted_errors}
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
            
            # Execute the function with processed parameters
            result = await registry.execute(function_name, **parameters)
            
            # Return JSON result
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
    
    @server.tool()
    async def get_function_details(function_name: str) -> str:
        """
        Get detailed information about a specific function.
        
        Args:
            function_name: Name of the function to get details for
            
        Returns:
            JSON string with function details
        """
        try:
            metadata = registry.get_function_metadata(function_name)
            
            if not metadata:
                return json.dumps({
                    "error": f"Function '{function_name}' not found",
                    "function_name": function_name
                })
                
            return json.dumps(metadata.model_dump(), indent=2)
        except Exception as e:
            error_result = {
                "error": str(e),
                "function_name": function_name
            }
            return json.dumps(error_result)
    
    @server.tool()
    async def list_function_categories() -> str:
        """
        List all available function categories/namespaces.
        
        Returns:
            JSON string with category information
        """
        try:
            namespaces = registry.get_namespaces()
            
            # Count functions in each namespace
            namespace_counts = {}
            for ns in namespaces:
                functions = registry.get_functions_by_namespace(ns)
                namespace_counts[ns] = len(functions)
                
            result = {
                "categories": namespaces,
                "function_counts": namespace_counts,
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
    async def execute_function_bundle(bundle_name: str, context: Dict[str, Any]) -> str:
        """
        Execute a pre-defined sequence of functions with shared context.
        
        Bundles allow multiple related functions to be executed in sequence,
        with results from earlier functions available to later ones.
        
        Args:
            bundle_name: Name of the function bundle to execute
            context: Initial context for the bundle execution
            
        Returns:
            JSON string with results from all functions in the bundle
        """
        # Define bundles - in a real implementation, these would be loaded from a config
        bundles = {
            "create_project": [
                {
                    "function": "project.create_container", 
                    "params_map": {"name": "project_name"}
                },
                {
                    "function": "project.create_component", 
                    "params_map": {"project_id": "project_name", "name": "component_name"}
                }
            ],
            "create_entity_with_observation": [
                {
                    "function": "memory.create_entity",
                    "params_map": {"name": "entity_name", "entity_type": "entity_type"}
                },
                {
                    "function": "memory.create_observation",
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
            function_name = step["function"]
            params_map = step.get("params_map", {})
            
            # Map parameters from context
            params = {}
            for param_key, context_key in params_map.items():
                if context_key in bundle_context:
                    params[param_key] = bundle_context[context_key]
            
            # Execute the function
            result = await registry.execute(function_name, **params)
            results[function_name] = json.loads(result.to_json())
            
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
    async def get_parameter_suggestions(function_name: str) -> str:
        """
        Get parameter suggestions for a function.
        
        This tool helps AI agents understand how to use functions by providing
        type-specific parameter suggestions.
        
        Args:
            function_name: Name of the function to get suggestions for
            
        Returns:
            JSON string with parameter suggestions
        """
        try:
            metadata = registry.get_function_metadata(function_name)
            
            if not metadata:
                error_result = {
                    "error": f"Function '{function_name}' not found",
                    "function_name": function_name
                }
                return json.dumps(error_result, indent=2)
            
            # Generate suggestions
            suggestions = ParameterHelper.generate_parameter_suggestions(metadata)
            
            result = {
                "function_name": function_name,
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
                "function_name": function_name
            }
            return json.dumps(error_result, indent=2)
    
    @server.tool()
    async def migrate_tools(module_paths: Optional[List[str]] = None) -> str:
        """
        Migrate existing tools to the function registry.
        
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
            functions = registry.get_functions_by_namespace(namespace)
            if not functions:
                return json.dumps({
                    "status": "error",
                    "message": f"Namespace '{namespace}' not found or has no functions"
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
    async def generate_function_documentation(format: str = "json", output_path: Optional[str] = None, version: str = "1.0.0") -> str:
        """
        Generate comprehensive documentation for all registered functions.
        
        This tool creates documentation for all functions in the registry and exports
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
                        "message": f"Generated documentation for {docs['meta']['function_count']} functions",
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
    async def get_function_examples(function_name: str) -> str:
        """
        Get usage examples for a specific function.
        
        This tool generates usage examples for a function, including basic
        examples and specialized examples for common functions.
        
        Args:
            function_name: Name of the function to get examples for
            
        Returns:
            JSON string with usage examples
        """
        try:
            # Check if function exists
            metadata = registry.get_function_metadata(function_name)
            if not metadata:
                return json.dumps({
                    "status": "error",
                    "message": f"Function '{function_name}' not found"
                })
            
            # Create a documentation generator
            from src.registry.documentation_generator import DocumentationGenerator
            generator = DocumentationGenerator()
            
            # Generate examples for the function
            examples = generator._generate_examples(metadata)
            
            return json.dumps({
                "status": "success",
                "message": f"Generated {len(examples)} examples for {function_name}",
                "function_name": function_name,
                "examples": examples
            })
        except Exception as e:
            return json.dumps({
                "status": "error",
                "message": f"Error generating examples: {str(e)}"
            })

    @server.tool()
    async def parse_natural_language_parameters(function_name: str, text: str) -> str:
        """
        Parse parameters for a function from natural language text.
        
        This tool extracts parameter values from natural language descriptions,
        making it easier to use functions in conversational contexts.
        
        Args:
            function_name: Name of the function to parse parameters for
            text: Natural language text containing parameter values
            
        Returns:
            JSON string with extracted parameters
        """
        try:
            # Check if function exists
            metadata = registry.get_function_metadata(function_name)
            if not metadata:
                return json.dumps({
                    "status": "error",
                    "message": f"Function '{function_name}' not found"
                })
            
            # Parse parameters from text
            try:
                from src.registry.advanced_parameter_handler import parse_parameters
                parameters = parse_parameters(metadata, text)
                
                return json.dumps({
                    "status": "success",
                    "message": f"Extracted parameters for '{function_name}'",
                    "function_name": function_name,
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
        to provide defaults for function parameters.
        
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
    async def get_function_recommendations(query: str, context: Optional[Dict[str, Any]] = None, limit: int = 5) -> str:
        """
        Get function recommendations based on a natural language query.
        
        This tool helps suggest relevant functions based on what you're trying to do,
        using contextual understanding and usage patterns.
        
        Args:
            query: Natural language description of what you want to do
            context: Optional context information (e.g., current entities, data)
            limit: Maximum number of recommendations to return
            
        Returns:
            JSON string with function recommendations
        """
        try:
            # Check if recommendation engine is available
            if get_recommendation_engine is None:
                return json.dumps({
                    "status": "error",
                    "message": "Function recommendation engine is not available"
                })
                
            # Get recommendations
            engine = get_recommendation_engine()
            recommendations = engine.get_recommendations(query, context, limit)
            
            return json.dumps({
                "status": "success",
                "message": f"Found {len(recommendations)} function recommendations",
                "recommendations": recommendations
            })
        except Exception as e:
            return json.dumps({
                "status": "error",
                "message": f"Error getting function recommendations: {str(e)}"
            })
    
    @server.tool()
    async def get_function_chains(goal: str, limit: int = 3) -> str:
        """
        Get recommended function chains for accomplishing a goal.
        
        This tool suggests sequences of functions that work well together
        to accomplish complex tasks.
        
        Args:
            goal: Description of what you want to accomplish
            limit: Maximum number of chains to return
            
        Returns:
            JSON string with function chains
        """
        try:
            # Check if recommendation engine is available
            if get_recommendation_engine is None:
                return json.dumps({
                    "status": "error",
                    "message": "Function recommendation engine is not available"
                })
                
            # Get function chains
            engine = get_recommendation_engine()
            chains = engine.get_function_chains(goal, limit)
            
            return json.dumps({
                "status": "success",
                "message": f"Found {len(chains)} function chains for your goal",
                "chains": chains
            })
        except Exception as e:
            return json.dumps({
                "status": "error",
                "message": f"Error getting function chains: {str(e)}"
            })
    
    @server.tool()
    async def track_function_usage(function_name: str, parameters: Dict[str, Any], 
                                 result: Dict[str, Any], context: Optional[Dict[str, Any]] = None) -> str:
        """
        Track function usage for pattern analysis.
        
        This tool records function usage to improve recommendations and
        detect patterns.
        
        Args:
            function_name: Name of the function used
            parameters: Parameters passed to the function
            result: Result of the function call (dict format of FunctionResult)
            context: Optional context information
            
        Returns:
            JSON string with confirmation
        """
        try:
            # Check if tracking is available
            if get_recommendation_engine is None or get_pattern_detector is None:
                return json.dumps({
                    "status": "error",
                    "message": "Function usage tracking is not available"
                })
                
            # Convert result dict to FunctionResult if needed
            if isinstance(result, dict):
                func_result = FunctionResult(**result)
            else:
                func_result = result
                
            # Track in recommendation engine
            engine = get_recommendation_engine()
            engine.track_function_usage(function_name, parameters, func_result, context)
            
            # Add to pattern detector
            detector = get_pattern_detector()
            detector.add_function_call(function_name, parameters, func_result)
            
            return json.dumps({
                "status": "success",
                "message": f"Tracked usage of function '{function_name}'"
            })
        except Exception as e:
            return json.dumps({
                "status": "error",
                "message": f"Error tracking function usage: {str(e)}"
            })
    
    @server.tool()
    async def end_function_sequence() -> str:
        """
        End the current function call sequence.
        
        This tool marks the end of a logical sequence of function calls,
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
                "message": "Function sequence ended"
            })
        except Exception as e:
            return json.dumps({
                "status": "error",
                "message": f"Error ending function sequence: {str(e)}"
            })
    
    @server.tool()
    async def get_detected_patterns() -> str:
        """
        Get detected function usage patterns.
        
        This tool provides insights into common patterns of function usage,
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
        functions in a coordinated way.
        
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
        Execute a batch of function calls.
        
        Args:
            batch: List of function call specifications
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
                message=f"Executed batch of {len(batch)} function calls",
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
    async def configure_result_cache(function_name: str, ttl: int = 300, cacheable: bool = True) -> str:
        """
        Configure caching settings for a specific function.
        
        Args:
            function_name: Name of the function
            ttl: Time-to-live in seconds for cache entries
            cacheable: Whether this function's results should be cached
            
        Returns:
            JSON string with confirmation of cache settings
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
            
            # Check if function exists
            registry = get_registry()
            function_metadata = registry.get_function_metadata(function_name)
            if function_metadata is None:
                error_result = FunctionResult(
                    status="error",
                    data=None,
                    message=f"Function '{function_name}' does not exist",
                    error_code="FUNCTION_NOT_FOUND",
                    error_details={"function_name": function_name}
                )
                return error_result.to_json()
            
            # Set cache settings
            cache.set_function_cache_settings(function_name, ttl, cacheable)
            
            result = FunctionResult(
                status="success",
                data={
                    "function_name": function_name,
                    "ttl": ttl,
                    "cacheable": cacheable
                },
                message=f"Cache settings configured for '{function_name}'",
                error_code="",
                error_details={}
            )
            return result.to_json()
        except Exception as e:
            logger.error(f"Error configuring cache settings: {str(e)}")
            error_result = FunctionResult(
                status="error",
                data=None,
                message=f"Error configuring cache settings: {str(e)}",
                error_code="CACHE_CONFIG_ERROR",
                error_details={"exception": str(e)}
            )
            return error_result.to_json()

    @server.tool()
    async def invalidate_cache(function_name: Optional[str] = None) -> str:
        """
        Invalidate cache entries for a function or all functions.
        
        Args:
            function_name: Name of the function or None to invalidate all
            
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
            
            # If function name provided, check if it exists
            if function_name is not None:
                registry = get_registry()
                function_metadata = registry.get_function_metadata(function_name)
                if function_metadata is None:
                    error_result = FunctionResult(
                        status="error",
                        data=None,
                        message=f"Function '{function_name}' does not exist",
                        error_code="FUNCTION_NOT_FOUND",
                        error_details={"function_name": function_name}
                    )
                    return error_result.to_json()
            
            # Invalidate cache entries
            count = cache.invalidate(function_name)
            
            if function_name is None:
                message = f"Invalidated all cache entries ({count} total)"
            else:
                message = f"Invalidated {count} cache entries for '{function_name}'"
            
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