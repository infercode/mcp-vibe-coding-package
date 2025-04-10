#!/usr/bin/env python3
"""
Advanced Parameter Handler Demonstration

This script demonstrates the Advanced Parameter Handler component of the Function Registry Pattern,
showing flexible parameter parsing, context-aware defaults, middleware, and custom type conversion.
"""

import asyncio
import json
import os
import sys
from datetime import datetime, date
from typing import Dict, List, Any, Optional

# Add the parent directory to the path so imports work
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from src.registry.advanced_parameter_handler import (
    AdvancedParameterHandler,
    parse_parameters,
    validate_and_convert,
    register_middleware,
    register_context_provider,
    register_type_converter
)
from src.registry.function_models import FunctionMetadata, ParameterInfo
from src.registry.registry_manager import get_registry
import src.registry.example_functions  # Import to register example functions

# Import the migration framework to register real tools
from src.registry.migration_framework import migrate_all_tools


async def demo_flexible_parsing():
    """Demonstrate flexible parameter parsing from various input formats."""
    print("\n=== Flexible Parameter Parsing Demonstration ===\n")
    
    # Create sample function metadata
    func_metadata = FunctionMetadata(
        name="test.sample_function",
        description="A sample function for testing",
        parameters={
            "name": ParameterInfo(type="str", required=True, description="Person's name"),
            "age": ParameterInfo(type="int", required=False, description="Person's age"),
            "is_active": ParameterInfo(type="bool", required=False, description="Active status")
        },
        return_type="dict",
        is_async=True,
        docstring="A sample function for testing parameter parsing."
    )
    
    # Example 1: Dictionary input
    dict_input = {"name": "John Doe", "age": 30, "is_active": True}
    dict_result = parse_parameters(func_metadata, dict_input)
    
    print("Dictionary Input:")
    print(f"  Input: {dict_input}")
    print(f"  Result: {dict_result}")
    
    # Example 2: JSON string input
    json_input = '{"name": "Jane Smith", "age": 25, "is_active": false}'
    json_result = parse_parameters(func_metadata, json_input)
    
    print("\nJSON String Input:")
    print(f"  Input: {json_input}")
    print(f"  Result: {json_result}")
    
    # Example 3: Positional arguments as list
    list_input = ["Robert Johnson", 45, True]
    list_result = parse_parameters(func_metadata, list_input)
    
    print("\nList Input (Positional Args):")
    print(f"  Input: {list_input}")
    print(f"  Result: {list_result}")
    
    # Example 4: Natural language input
    nl_input = "Create a person with name John Doe, age is 35, is_active as true"
    nl_result = parse_parameters(func_metadata, nl_input)
    
    print("\nNatural Language Input:")
    print(f"  Input: {nl_input}")
    print(f"  Result: {nl_result}")


async def demo_context_aware_defaults():
    """Demonstrate context-aware parameter defaults."""
    print("\n=== Context-Aware Defaults Demonstration ===\n")
    
    # Create a function metadata with context paths
    func_metadata = FunctionMetadata(
        name="user.update_preferences",
        description="Update user preferences",
        parameters={
            "user_id": ParameterInfo(
                type="str", 
                required=True, 
                description="User ID",
                context_path="user.id"
            ),
            "theme": ParameterInfo(
                type="str", 
                required=False, 
                description="UI theme",
                context_path="user.preferences.theme"
            ),
            "timezone": ParameterInfo(
                type="str",
                required=False,
                description="User timezone",
                context_path="user.timezone"
            )
        },
        return_type="dict",
        is_async=True,
        docstring="Update user preferences with context-aware defaults."
    )
    
    # Create context with user data
    context = {
        "user": {
            "id": "user123",
            "preferences": {
                "theme": "dark",
                "language": "en"
            },
            "timezone": "America/New_York"
        }
    }
    
    # Example 1: Empty parameters (all from context)
    empty_input = {}
    context_result1 = parse_parameters(func_metadata, empty_input, context)
    
    print("All Parameters from Context:")
    print(f"  Input: {empty_input}")
    print(f"  Context: {context}")
    print(f"  Result: {context_result1}")
    
    # Example 2: Override some context values
    override_input = {"theme": "light"}
    context_result2 = parse_parameters(func_metadata, override_input, context)
    
    print("\nOverride Some Context Values:")
    print(f"  Input: {override_input}")
    print(f"  Result: {context_result2}")
    
    # Register a custom context provider
    register_context_provider("session", lambda: {
        "last_login": "2025-04-07T10:00:00Z",
        "session_id": "sess_12345"
    })
    
    # Create function using the session provider
    session_func = FunctionMetadata(
        name="session.extend",
        description="Extend session",
        parameters={
            "session_id": ParameterInfo(
                type="str",
                required=True,
                description="Session ID",
                context_path="session.session_id"
            ),
            "duration": ParameterInfo(
                type="int",
                required=True,
                description="Extension duration in minutes"
            )
        },
        return_type="dict",
        is_async=True,
        docstring="Extend a session duration."
    )
    
    # Example 3: Using registered context provider
    provider_input = {"duration": 30}
    provider_result = parse_parameters(session_func, provider_input)
    
    print("\nUsing Registered Context Provider:")
    print(f"  Input: {provider_input}")
    print(f"  Result: {provider_result}")


async def demo_parameter_middleware():
    """Demonstrate parameter middleware pipeline."""
    print("\n=== Parameter Middleware Demonstration ===\n")
    
    # Create function metadata with aliases
    func_metadata = FunctionMetadata(
        name="document.create",
        description="Create a document",
        parameters={
            "title": ParameterInfo(
                type="str",
                required=True,
                description="Document title",
                aliases=["name", "heading"]
            ),
            "content": ParameterInfo(
                type="str",
                required=True,
                description="Document content",
                aliases=["text", "body"]
            ),
            "tags": ParameterInfo(
                type="list",
                required=False,
                description="Document tags",
                aliases=["categories", "labels"]
            )
        },
        return_type="dict",
        is_async=True,
        docstring="Create a new document."
    )
    
    # Example 1: Using parameter aliases
    alias_input = {"name": "Meeting Notes", "body": "Discussion points...", "labels": ["work", "meeting"]}
    alias_result = parse_parameters(func_metadata, alias_input)
    
    print("Parameter Aliases:")
    print(f"  Input: {alias_input}")
    print(f"  Result: {alias_result}")
    
    # Register a custom middleware
    def add_timestamp_middleware(func_metadata, params, context):
        """Add timestamp to parameters if applicable."""
        result = params.copy()
        
        # Add timestamp if function accepts it but it's not provided
        if "timestamp" in func_metadata.parameters and "timestamp" not in result:
            result["timestamp"] = context.get("system", {}).get("timestamp")
            
        return result
    
    register_middleware(add_timestamp_middleware, priority=5)
    
    # Create function that accepts timestamp
    timestamp_func = FunctionMetadata(
        name="event.log",
        description="Log an event",
        parameters={
            "event_type": ParameterInfo(type="str", required=True, description="Event type"),
            "details": ParameterInfo(type="str", required=True, description="Event details"),
            "timestamp": ParameterInfo(type="str", required=False, description="Event timestamp")
        },
        return_type="dict",
        is_async=True,
        docstring="Log an event with timestamp."
    )
    
    # Example 2: Using custom middleware
    event_input = {"event_type": "user_login", "details": "User logged in from mobile app"}
    event_result = parse_parameters(timestamp_func, event_input)
    
    print("\nCustom Middleware (Add Timestamp):")
    print(f"  Input: {event_input}")
    print(f"  Result: {event_result}")


async def demo_type_conversion():
    """Demonstrate advanced type conversion."""
    print("\n=== Advanced Type Conversion Demonstration ===\n")
    
    # Create function metadata with custom types
    func_metadata = FunctionMetadata(
        name="appointment.schedule",
        description="Schedule an appointment",
        parameters={
            "title": ParameterInfo(type="str", required=True, description="Appointment title"),
            "date": ParameterInfo(type="date", required=True, description="Appointment date"),
            "start_time": ParameterInfo(type="datetime", required=True, description="Start time"),
            "duration": ParameterInfo(type="duration", required=True, description="Duration")
        },
        return_type="dict",
        is_async=True,
        docstring="Schedule a new appointment."
    )
    
    # Example 1: Converting various date formats
    date_inputs = [
        {"title": "Meeting 1", "date": "2025-05-01", "start_time": "2025-05-01 10:00:00", "duration": "1h 30m"},
        {"title": "Meeting 2", "date": "05/15/2025", "start_time": "2025-05-15T14:30:00", "duration": "45m"},
        {"title": "Meeting 3", "date": "June 1, 2025", "start_time": "06/01/2025 09:00:00", "duration": "2h"}
    ]
    
    handler = AdvancedParameterHandler()
    
    print("Date and Time Conversion:")
    for i, input_data in enumerate(date_inputs):
        print(f"\n  Example {i+1}:")
        print(f"    Input: {input_data}")
        
        # Parse and convert
        parsed = handler.parse_parameters(func_metadata, input_data)
        converted, errors = handler.validate_and_convert(func_metadata, parsed)
        
        if errors:
            print(f"    Errors: {errors}")
        else:
            # Format for display
            result = {
                "title": converted["title"],
                "date": str(converted["date"]),
                "start_time": str(converted["start_time"]),
                "duration": f"{converted['duration']} seconds"
            }
            print(f"    Converted: {result}")
    
    # Register a custom type converter for coordinates
    def convert_coordinates(value):
        """Convert string coordinates to (lat, lng) tuple."""
        if isinstance(value, tuple) and len(value) == 2:
            return value
            
        if isinstance(value, str):
            # Try "lat,lng" format
            try:
                parts = value.split(',')
                if len(parts) == 2:
                    return (float(parts[0]), float(parts[1]))
            except ValueError:
                pass
                
            # Try "lat lng" format
            try:
                parts = value.split()
                if len(parts) == 2:
                    return (float(parts[0]), float(parts[1]))
            except ValueError:
                pass
        
        raise ValueError(f"Cannot convert to coordinates: {value}")
    
    register_type_converter("coordinates", convert_coordinates)
    
    # Create function with coordinates
    location_func = FunctionMetadata(
        name="map.add_marker",
        description="Add a marker to the map",
        parameters={
            "label": ParameterInfo(type="str", required=True, description="Marker label"),
            "position": ParameterInfo(type="coordinates", required=True, description="Coordinates (lat, lng)")
        },
        return_type="dict",
        is_async=True,
        docstring="Add a marker to the map at specified coordinates."
    )
    
    # Example 2: Custom type converter
    coord_inputs = [
        {"label": "Office", "position": "40.7128, -74.006"},
        {"label": "Home", "position": "37.7749 -122.4194"},
        {"label": "Airport", "position": (33.9416, -118.4085)}
    ]
    
    print("\nCustom Type Converter (Coordinates):")
    for i, input_data in enumerate(coord_inputs):
        print(f"\n  Example {i+1}:")
        print(f"    Input: {input_data}")
        
        # Parse and convert
        parsed = handler.parse_parameters(location_func, input_data)
        converted, errors = handler.validate_and_convert(location_func, parsed)
        
        if errors:
            print(f"    Errors: {errors}")
        else:
            print(f"    Converted: {converted}")


async def demo_advanced_parameters():
    """Run the complete Advanced Parameter Handler demonstration."""
    print("\n=== Advanced Parameter Handler Demonstration ===\n")
    
    # First, migrate some real tools to have a meaningful registry
    print("Migrating tools to registry...")
    migrate_all_tools()
    
    # Show each feature in action
    await demo_flexible_parsing()
    await demo_context_aware_defaults()
    await demo_parameter_middleware()
    await demo_type_conversion()
    
    print("\n=== Advanced Parameter Handler Complete ===\n")
    
    # Show conclusion
    print("The Advanced Parameter Handler features demonstrate how to:")
    print("1. Parse parameters from various input formats, including natural language")
    print("2. Apply context-aware defaults from user and system state")
    print("3. Use middleware pipeline for parameter processing")
    print("4. Perform advanced type conversion with custom types")
    print("\nThese features make the Function Registry more flexible and user-friendly")
    print("for both AI agents and human users.")


if __name__ == "__main__":
    # Run the demo
    asyncio.run(demo_advanced_parameters()) 