#!/usr/bin/env python3
"""
Function Registry Pattern Implementation

This module provides a registry pattern for consolidating multiple function tools
into a few category-based meta-tools to address IDE integration limitations.
"""

from src.registry.registry_manager import FunctionRegistry, register_function
from src.registry.function_models import FunctionMetadata, FunctionResult, FunctionParameters
from src.registry.registry_tools import register_registry_tools
from src.registry.parameter_helper import ParameterHelper, ValidationError

__all__ = [
    "FunctionRegistry",
    "register_function",
    "FunctionMetadata",
    "FunctionResult",
    "FunctionParameters",
    "register_registry_tools",
    "ParameterHelper",
    "ValidationError"
] 