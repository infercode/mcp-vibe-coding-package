#!/usr/bin/env python3
"""
Function Registry Pattern Implementation

This module provides a registry pattern for consolidating multiple function tools
into a few category-based meta-tools to address IDE integration limitations.
"""

from src.registry.registry_manager import FunctionRegistry, register_function, get_registry
from src.registry.function_models import FunctionMetadata, FunctionResult, FunctionParameters
from src.registry.registry_tools import register_registry_tools
from src.registry.parameter_helper import ParameterHelper, ValidationError

# Import all modules that register functions with the registry
from src.registry import registry_tools
from src.registry import function_models
from src.registry import parameter_helper
from src.registry import advanced_parameter_handler
from src.registry import documentation_generator
from src.registry import ide_integration
from src.registry import migration_framework
from src.registry import performance_optimization
from src.registry import agent_guidance
from src.registry import health_diagnostics
from src.registry import feedback_mechanism

__all__ = [
    "FunctionRegistry",
    "register_function",
    "get_registry",
    "FunctionMetadata",
    "FunctionResult",
    "FunctionParameters",
    "ParameterHelper",
    "register_registry_tools",
    "ValidationError",
    "initialize_registry"
]

def initialize_registry():
    """
    Initialize all registry functions without exposing them as tools.
    
    This function ensures that all function modules register their functions
    with the registry, making them available via the execute_function tool
    even when not all tools are directly exposed to the MCP server.
    
    Returns:
        FunctionRegistry: The initialized registry instance
    """
    # Get the registry singleton
    registry = get_registry()
    
    # The imports at the top of this file should trigger registration
    # of functions in each module on import, so we don't need to call
    # explicit registration functions.
    
    # Import additional modules if needed
    try:
        from src.registry import function_bundles
    except ImportError:
        pass
    
    # Log the number of registered functions
    function_count = len(registry.get_all_functions())
    namespace_count = len(registry.get_namespaces())
    print(f"Registry initialized with {function_count} functions in {namespace_count} namespaces")
    
    return registry 