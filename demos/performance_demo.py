#!/usr/bin/env python3
"""
Performance Optimization Demo

This script demonstrates the performance optimization features of the Function Registry:
1. Result caching for repeated function calls
2. Batch operations for executing multiple functions
3. Parameter serialization for efficient data handling
"""

import asyncio
import time
import json
from datetime import datetime, date
from typing import Dict, Any, List

from src.registry.function_models import FunctionInfo, FunctionResult
from src.registry.registry_manager import get_registry, register_function
from src.registry.registry_tools import migrate_tools
from src.registry.performance_optimization import (
    get_result_cache, 
    get_batch_processor,
    get_parameter_serializer,
    cacheable
)

# Demo functions for testing performance optimizations

@cacheable(ttl=60)
async def demo_expensive_calculation(a: int, b: int) -> FunctionResult:
    """
    Simulates an expensive calculation that benefits from caching.
    
    Args:
        a: First number
        b: Second number
        
    Returns:
        Result of the calculation
    """
    # Simulate an expensive operation
    await asyncio.sleep(1)
    
    result = a * b
    
    return FunctionResult(
        status="success",
        data={"result": result},
        message=f"Calculated {a} * {b} = {result}"
    )

async def demo_date_operation(input_date: date, days_to_add: int = 0) -> FunctionResult:
    """
    Performs an operation on a date, used to demonstrate parameter serialization.
    
    Args:
        input_date: The input date
        days_to_add: Number of days to add
        
    Returns:
        Result with the new date
    """
    # Add days to the date
    result_date = input_date
    if days_to_add:
        from datetime import timedelta
        result_date = input_date + timedelta(days=days_to_add)
    
    return FunctionResult(
        status="success",
        data={
            "original_date": input_date.isoformat(),
            "result_date": result_date.isoformat(),
            "days_added": days_to_add
        },
        message=f"Date operation completed: {input_date} + {days_to_add} days = {result_date}"
    )

async def demo_string_processor(text: str, operation: str = "uppercase") -> FunctionResult:
    """
    Processes a string, used for batch operation demo.
    
    Args:
        text: Input text
        operation: Operation to perform (uppercase, lowercase, reverse)
        
    Returns:
        Processed text
    """
    result = text
    
    if operation == "uppercase":
        result = text.upper()
    elif operation == "lowercase":
        result = text.lower()
    elif operation == "reverse":
        result = text[::-1]
        
    return FunctionResult(
        status="success",
        data={"result": result},
        message=f"String processed with operation '{operation}'"
    )

async def register_demo_functions():
    """Register the demo functions with the registry."""
    registry = get_registry()
    
    # Register the expensive calculation function
    calc_info = FunctionInfo(
        name="demo.expensive_calculation",
        description="Simulates an expensive calculation that benefits from caching",
        parameters={
            "a": {"type": "integer", "description": "First number", "required": True},
            "b": {"type": "integer", "description": "Second number", "required": True}
        },
        function=demo_expensive_calculation
    )
    await registry.register(calc_info)
    
    # Register the date operation function
    date_info = FunctionInfo(
        name="demo.date_operation",
        description="Performs an operation on a date",
        parameters={
            "input_date": {"type": "date", "description": "The input date", "required": True},
            "days_to_add": {"type": "integer", "description": "Number of days to add", "required": False}
        },
        function=demo_date_operation
    )
    await registry.register(date_info)
    
    # Register the string processor function
    string_info = FunctionInfo(
        name="demo.string_processor",
        description="Processes a string",
        parameters={
            "text": {"type": "string", "description": "Input text", "required": True},
            "operation": {
                "type": "string", 
                "description": "Operation to perform", 
                "required": False,
                "enum": ["uppercase", "lowercase", "reverse"]
            }
        },
        function=demo_string_processor
    )
    await registry.register(string_info)
    
    print("Demo functions registered.")

async def demo_result_caching():
    """Demonstrate the result caching feature."""
    print("\n=== Result Caching Demo ===")
    registry = get_registry()
    
    # First call - should take full time
    print("First call to expensive_calculation...")
    start_time = time.time()
    result1 = await registry.execute("demo.expensive_calculation", a=5, b=7)
    elapsed1 = time.time() - start_time
    print(f"Result: {result1.data['result']}")
    print(f"Time taken: {elapsed1:.2f} seconds")
    
    # Second call with same parameters - should use cache
    print("\nSecond call with same parameters...")
    start_time = time.time()
    result2 = await registry.execute("demo.expensive_calculation", a=5, b=7)
    elapsed2 = time.time() - start_time
    print(f"Result: {result2.data['result']}")
    print(f"Time taken: {elapsed2:.2f} seconds")
    
    # Call with different parameters - should not use cache
    print("\nCall with different parameters...")
    start_time = time.time()
    result3 = await registry.execute("demo.expensive_calculation", a=10, b=20)
    elapsed3 = time.time() - start_time
    print(f"Result: {result3.data['result']}")
    print(f"Time taken: {elapsed3:.2f} seconds")
    
    # Show cache statistics
    cache = get_result_cache()
    print(f"\nCurrent cache size: {len(cache.cache)} entries")
    
    # Invalidate the cache for the function
    invalidated = cache.invalidate("demo.expensive_calculation")
    print(f"Invalidated {invalidated} cache entries for demo.expensive_calculation")
    
    # Call again after invalidation - should take full time
    print("\nCall after cache invalidation...")
    start_time = time.time()
    result4 = await registry.execute("demo.expensive_calculation", a=5, b=7)
    elapsed4 = time.time() - start_time
    print(f"Result: {result4.data['result']}")
    print(f"Time taken: {elapsed4:.2f} seconds")

async def demo_batch_operations():
    """Demonstrate the batch operations feature."""
    print("\n=== Batch Operations Demo ===")
    batch_processor = get_batch_processor()
    
    # Define a batch of operations
    batch = [
        {
            "id": "calc1",
            "function": "demo.expensive_calculation",
            "params": {"a": 3, "b": 4}
        },
        {
            "id": "calc2",
            "function": "demo.expensive_calculation",
            "params": {"a": 5, "b": 6}
        },
        {
            "id": "string1",
            "function": "demo.string_processor",
            "params": {"text": "Hello World", "operation": "uppercase"}
        }
    ]
    
    # Execute the batch in parallel
    print("Executing batch in parallel...")
    start_time = time.time()
    results = await batch_processor.execute_batch(batch, parallel=True)
    elapsed = time.time() - start_time
    
    print(f"Batch execution completed in {elapsed:.2f} seconds")
    for i, result in enumerate(results):
        print(f"Result {i+1} (ID: {result.get('call_id', 'unknown')}): {result.get('status')} - {result.get('data')}")
    
    # Demonstrate dependency between batch operations
    print("\nExecuting batch with dependencies...")
    
    dependent_batch = [
        {
            "id": "string_op",
            "function": "demo.string_processor",
            "params": {"text": "Function Registry", "operation": "uppercase"}
        },
        {
            "id": "reverse_op",
            "function": "demo.string_processor",
            "params": {},
            "param_refs": {"text": "string_op.data.result", "operation": "reverse"},
            "depends_on": ["string_op"]
        }
    ]
    
    results = await batch_processor.execute_batch(dependent_batch, parallel=False)
    
    for i, result in enumerate(results):
        print(f"Result {i+1} (ID: {result.get('call_id', 'unknown')}): {result.get('status')} - {result.get('data')}")

async def demo_parameter_serialization():
    """Demonstrate the parameter serialization feature."""
    print("\n=== Parameter Serialization Demo ===")
    serializer = get_parameter_serializer()
    registry = get_registry()
    
    # Get function metadata
    function_metadata = await registry.get_function_metadata("demo.date_operation")
    
    # Get parameter types from metadata
    param_types = serializer.get_parameter_types(function_metadata)
    print(f"Parameter types for demo.date_operation: {param_types}")
    
    # Create some parameters with a date
    today = date.today()
    params = {
        "input_date": today,
        "days_to_add": 7
    }
    
    print(f"\nOriginal parameters: {params}")
    
    # Serialize the parameters
    serialized = serializer.serialize_parameters(params, param_types)
    print(f"Serialized parameters: {serialized}")
    
    # Deserialize the parameters
    deserialized = serializer.deserialize_parameters(serialized, param_types)
    print(f"Deserialized parameters: {deserialized}")
    
    # Execute the function with the deserialized parameters
    result = await registry.execute("demo.date_operation", **deserialized)
    print(f"Function result: {result.message}")
    
    # Demo custom serializer for a complex type
    print("\nRegistering custom serializer for 'coordinates' type...")
    
    # Define custom serializers for a coordinates type
    def serialize_coordinates(coords):
        if isinstance(coords, dict) and "lat" in coords and "lng" in coords:
            return f"{coords['lat']},{coords['lng']}"
        return str(coords)
    
    def deserialize_coordinates(coords_str):
        try:
            lat, lng = map(float, coords_str.split(','))
            return {"lat": lat, "lng": lng}
        except:
            return coords_str
    
    # Register the custom serializer
    serializer.register_serializer(
        "coordinates", 
        serialize_coordinates, 
        deserialize_coordinates
    )
    
    # Test the custom serializer
    coords = {"lat": 37.7749, "lng": -122.4194}
    custom_types = {"location": "coordinates"}
    
    serialized = serializer.serialize_parameters({"location": coords}, custom_types)
    print(f"Serialized coordinates: {serialized}")
    
    deserialized = serializer.deserialize_parameters(serialized, custom_types)
    print(f"Deserialized coordinates: {deserialized}")

async def run_demo():
    """Run the complete performance optimization demo."""
    print("===== Function Registry Performance Optimization Demo =====")
    
    # Register the demo functions
    await register_demo_functions()
    
    # Run the individual demos
    await demo_result_caching()
    await demo_batch_operations()
    await demo_parameter_serialization()
    
    print("\n===== Demo Complete =====")

if __name__ == "__main__":
    asyncio.run(run_demo()) 