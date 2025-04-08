#!/usr/bin/env python3
"""
Function Bundles with Performance Optimization Demo

This script demonstrates:
1. Creating and executing function bundles
2. Using performance optimizations with bundles
3. Conditional bundle execution
4. Parameter transformations
"""

import asyncio
import time
import json
from datetime import datetime, date
from typing import Dict, Any, List

from src.registry.function_models import FunctionResult
from src.registry.registry_manager import get_registry
try:
    from src.registry.registry_tools import migrate_tools, execute_batch
except ImportError:
    # Create placeholder functions if imports fail
    async def migrate_tools(paths):
        print(f"migrate_tools not available, would import: {paths}")
        return False

    async def execute_batch(batch, parallel=True, use_cache=True):
        print(f"execute_batch not available")
        return FunctionResult(
            status="error", 
            message="execute_batch not available",
            data=None,
            error_code="IMPORT_ERROR",
            error_details={"reason": "Module not available"}
        )

try:
    from src.registry.performance_optimization import get_batch_processor, get_result_cache
except ImportError:
    # Create placeholder functions if imports fail
    def get_batch_processor():
        print("get_batch_processor not available")
        return None

    def get_result_cache():
        print("get_result_cache not available")
        return None

# Import BundleManager with try/except to handle missing implementation
try:
    from src.registry.function_bundles import BundleManager, get_bundle_manager
except ImportError:
    BundleManager = None
    get_bundle_manager = None

# Example function bundle definitions
FUNCTION_BUNDLES = {
    "data_processing": {
        "name": "data_processing",
        "description": "Process data through a series of operations",
        "steps": [
            {
                "id": "fetch_data",
                "function": "demo.string_processor",
                "parameters": {
                    "text": "{input_text}",
                    "operation": "uppercase"
                }
            },
            {
                "id": "transform_data",
                "function": "demo.string_processor",
                "parameters": {
                    "operation": "reverse"
                },
                "parameter_mappings": {
                    "text": "fetch_data.data.result"
                },
                "conditions": {
                    "depends_on": ["fetch_data"],
                    "success_required": ["fetch_data"]
                }
            },
            {
                "id": "final_transform",
                "function": "demo.string_processor",
                "parameters": {
                    "operation": "lowercase"
                },
                "parameter_mappings": {
                    "text": "transform_data.data.result"
                },
                "conditions": {
                    "depends_on": ["transform_data"],
                    "success_required": ["transform_data"]
                }
            }
        ]
    },
    
    "calculation_chain": {
        "name": "calculation_chain",
        "description": "Chain of calculations with numeric results",
        "steps": [
            {
                "id": "calc_step1",
                "function": "demo.expensive_calculation",
                "parameters": {
                    "a": "{a}",
                    "b": "{b}"
                }
            },
            {
                "id": "calc_step2",
                "function": "demo.expensive_calculation",
                "parameters": {
                    "b": "{c}"
                },
                "parameter_mappings": {
                    "a": "calc_step1.data.result"
                },
                "conditions": {
                    "depends_on": ["calc_step1"]
                }
            }
        ]
    },
    
    "date_processing": {
        "name": "date_processing",
        "description": "Process dates through multiple operations",
        "steps": [
            {
                "id": "initial_date",
                "function": "demo.date_operation",
                "parameters": {
                    "input_date": "{start_date}",
                    "days_to_add": "{days1}"
                }
            },
            {
                "id": "second_date",
                "function": "demo.date_operation",
                "parameters": {
                    "days_to_add": "{days2}"
                },
                "parameter_mappings": {
                    "input_date": "initial_date.data.result_date",
                    "input_date_transform": "date_from_iso"
                },
                "conditions": {
                    "depends_on": ["initial_date"]
                }
            }
        ]
    },
    
    "conditional_bundle": {
        "name": "conditional_bundle",
        "description": "Bundle with conditional execution paths",
        "steps": [
            {
                "id": "check_condition",
                "function": "demo.string_processor",
                "parameters": {
                    "text": "{condition_text}",
                    "operation": "uppercase"
                }
            },
            {
                "id": "path_a",
                "function": "demo.string_processor",
                "parameters": {
                    "text": "Path A was taken",
                    "operation": "uppercase"
                },
                "conditions": {
                    "custom": "check_condition.data.result == 'YES'"
                }
            },
            {
                "id": "path_b",
                "function": "demo.string_processor",
                "parameters": {
                    "text": "Path B was taken",
                    "operation": "uppercase"
                },
                "conditions": {
                    "custom": "check_condition.data.result != 'YES'"
                }
            },
            {
                "id": "final_step",
                "function": "demo.string_processor",
                "parameters": {
                    "operation": "lowercase"
                },
                "parameter_mappings": {
                    "text": {
                        "source": ["path_a.data.result", "path_b.data.result"],
                        "transform": "first_available"
                    }
                },
                "conditions": {
                    "depends_on": ["check_condition"]
                }
            }
        ]
    }
}

# Custom parameter transformers
def date_from_iso(value):
    """Convert ISO date string to date object."""
    if isinstance(value, str):
        return datetime.fromisoformat(value).date()
    return value

async def register_demo_functions_and_bundles():
    """Register demo functions and bundles."""
    # First register the same demo functions from performance_demo.py
    # We can import them or re-register them here
    success = await migrate_tools(["src.registry.performance_demo"])
    
    # Check if bundle manager is available
    if get_bundle_manager is None:
        print("Bundle Manager is not available - cannot register bundles.")
        return False
    
    # Get bundle manager
    bundle_manager = get_bundle_manager()
    if bundle_manager is None:
        print("Bundle Manager initialization failed - cannot register bundles.")
        return False
    
    # Register custom parameter transformers
    bundle_manager.register_transformer("date_from_iso", date_from_iso)
    
    # Register the bundles
    for bundle_name, bundle_def in FUNCTION_BUNDLES.items():
        try:
            bundle_manager.register_bundle(bundle_name, bundle_def)
            print(f"Registered bundle: {bundle_name}")
        except Exception as e:
            print(f"Failed to register bundle: {bundle_name} - {str(e)}")
    
    return True

async def demo_bundle_execution():
    """Demonstrate bundle execution with caching."""
    print("\n=== Bundle Execution Demo ===")
    
    # Check if bundle manager is available
    if get_bundle_manager is None:
        print("Bundle Manager is not available - cannot execute demo.")
        return
    
    bundle_manager = get_bundle_manager()
    if bundle_manager is None:
        print("Bundle Manager initialization failed - cannot execute demo.")
        return
    
    # Execute the data processing bundle
    print("\nExecuting data_processing bundle...")
    start_time = time.time()
    try:
        result = await bundle_manager.execute_bundle(
            "data_processing",
            {"input_text": "Function Registry Pattern"}
        )
        elapsed = time.time() - start_time
        
        print(f"Time taken: {elapsed:.2f} seconds")
        print("Results:")
        
        # Check result type and extract data
        if hasattr(result, "data") and isinstance(result.data, dict):
            # Handle FunctionResult type
            for step_id, step_data in result.data.items():
                if step_id != "_step_results":  # Skip internal results
                    print(f"  Step '{step_id}': {step_data}")
        elif isinstance(result, dict):
            # Handle dictionary result
            for step_id, step_result in result.items():
                if step_id != "_step_results":  # Skip internal results
                    status = step_result.get("status", "unknown")
                    data = step_result.get("data", {})
                    print(f"  Step '{step_id}': {status} - {data}")
    except Exception as e:
        print(f"Error executing bundle: {str(e)}")
    
    # Execute the calculation chain bundle (should use cache for expensive calculations)
    print("\nExecuting calculation_chain bundle...")
    start_time = time.time()
    try:
        result = await bundle_manager.execute_bundle(
            "calculation_chain",
            {"a": 5, "b": 7, "c": 3}
        )
        elapsed1 = time.time() - start_time
        
        print(f"First execution time: {elapsed1:.2f} seconds")
        print("Results:")
        
        # Check result type and extract data
        if hasattr(result, "data") and isinstance(result.data, dict):
            # Handle FunctionResult type
            for step_id, step_data in result.data.items():
                if step_id != "_step_results":  # Skip internal results
                    print(f"  Step '{step_id}': {step_data}")
        elif isinstance(result, dict):
            # Handle dictionary result
            for step_id, step_result in result.items():
                if step_id != "_step_results":  # Skip internal results
                    status = step_result.get("status", "unknown")
                    data = step_result.get("data", {})
                    print(f"  Step '{step_id}': {status} - {data}")
        
        # Execute again with same parameters to demonstrate caching
        print("\nExecuting calculation_chain bundle again (should use cache)...")
        start_time = time.time()
        result = await bundle_manager.execute_bundle(
            "calculation_chain",
            {"a": 5, "b": 7, "c": 3}
        )
        elapsed2 = time.time() - start_time
        
        print(f"Second execution time: {elapsed2:.2f} seconds")
        print(f"Cache speedup: {elapsed1/max(0.0001, elapsed2):.1f}x faster")
    except Exception as e:
        print(f"Error executing bundle: {str(e)}")

async def demo_conditional_bundle():
    """Demonstrate conditional bundle execution."""
    print("\n=== Conditional Bundle Execution Demo ===")
    
    # Check if bundle manager is available
    if get_bundle_manager is None:
        print("Bundle Manager is not available - cannot execute demo.")
        return
    
    bundle_manager = get_bundle_manager()
    if bundle_manager is None:
        print("Bundle Manager initialization failed - cannot execute demo.")
        return
    
    # Execute with condition yielding "YES" path
    print("\nExecuting conditional_bundle with YES condition...")
    try:
        result1 = await bundle_manager.execute_bundle(
            "conditional_bundle",
            {"condition_text": "yes"}
        )
        
        print("Results:")
        # Check result type and extract data
        if hasattr(result1, "data") and isinstance(result1.data, dict):
            # Handle FunctionResult type
            for step_id, step_data in result1.data.items():
                if step_id != "_step_results":  # Skip internal results
                    print(f"  Step '{step_id}': {step_data}")
        elif isinstance(result1, dict):
            # Handle dictionary result
            for step_id, step_result in result1.items():
                if step_id != "_step_results" and isinstance(step_result, dict):
                    if step_result.get('status') == 'success':
                        print(f"  Step '{step_id}': {step_result.get('data')}")
                    else:
                        print(f"  Step '{step_id}': {step_result.get('status')} (skipped or failed)")
    except Exception as e:
        print(f"Error executing bundle: {str(e)}")
    
    # Execute with condition yielding "NO" path
    print("\nExecuting conditional_bundle with NO condition...")
    try:
        result2 = await bundle_manager.execute_bundle(
            "conditional_bundle",
            {"condition_text": "no"}
        )
        
        print("Results:")
        # Check result type and extract data
        if hasattr(result2, "data") and isinstance(result2.data, dict):
            # Handle FunctionResult type
            for step_id, step_data in result2.data.items():
                if step_id != "_step_results":  # Skip internal results
                    print(f"  Step '{step_id}': {step_data}")
        elif isinstance(result2, dict):
            # Handle dictionary result
            for step_id, step_result in result2.items():
                if step_id != "_step_results" and isinstance(step_result, dict):
                    if step_result.get('status') == 'success':
                        print(f"  Step '{step_id}': {step_result.get('data')}")
                    else:
                        print(f"  Step '{step_id}': {step_result.get('status')} (skipped or failed)")
    except Exception as e:
        print(f"Error executing bundle: {str(e)}")

async def demo_batch_vs_bundle():
    """Compare performance of batch operations vs bundles."""
    print("\n=== Batch vs Bundle Performance Comparison ===")
    
    # Check required components
    if get_batch_processor is None or get_bundle_manager is None:
        print("Required components not available for this demo.")
        return
    
    batch_processor = get_batch_processor()
    if batch_processor is None:
        print("Batch processor initialization failed - cannot execute demo.")
        return
        
    bundle_manager = get_bundle_manager()
    if bundle_manager is None:
        print("Bundle Manager initialization failed - cannot execute demo.")
        return
    
    # Define the same operation as both a batch and a bundle
    batch = [
        {
            "id": "step1",
            "function": "demo.expensive_calculation",
            "params": {"a": 10, "b": 20}
        },
        {
            "id": "step2",
            "function": "demo.expensive_calculation",
            "params": {"a": 30, "b": 40}
        },
        {
            "id": "step3",
            "function": "demo.expensive_calculation",
            "params": {"a": 50, "b": 60}
        }
    ]
    
    # Define an equivalent bundle
    bundle_def = {
        "name": "perf_test_bundle",
        "description": "Bundle for performance testing",
        "steps": [
            {
                "id": "step1",
                "function": "demo.expensive_calculation",
                "parameters": {
                    "a": 10,
                    "b": 20
                }
            },
            {
                "id": "step2",
                "function": "demo.expensive_calculation",
                "parameters": {
                    "a": 30,
                    "b": 40
                }
            },
            {
                "id": "step3",
                "function": "demo.expensive_calculation",
                "parameters": {
                    "a": 50,
                    "b": 60
                }
            }
        ]
    }
    
    # Register the test bundle
    try:
        bundle_manager.register_bundle("perf_test_bundle", bundle_def)
    except Exception as e:
        print(f"Error registering test bundle: {str(e)}")
        return
    
    # Clear cache before tests
    try:
        if get_result_cache:
            cache = get_result_cache()
            if cache:
                cache.clear()
    except Exception as e:
        print(f"Error clearing cache: {str(e)}")
    
    # Execute as batch
    print("\nExecuting as batch operation...")
    start_time = time.time()
    try:
        batch_result = await batch_processor.execute_batch(batch, parallel=True)
        batch_time = time.time() - start_time
        
        print(f"Batch execution time: {batch_time:.2f} seconds")
    except Exception as e:
        print(f"Error executing batch: {str(e)}")
        batch_time = 0
    
    # Execute as bundle
    print("\nExecuting as bundle...")
    start_time = time.time()
    try:
        bundle_result = await bundle_manager.execute_bundle("perf_test_bundle", {})
        bundle_time = time.time() - start_time
        
        print(f"Bundle execution time: {bundle_time:.2f} seconds")
    except Exception as e:
        print(f"Error executing bundle: {str(e)}")
        bundle_time = 0
    
    # Compare results
    if batch_time > 0 and bundle_time > 0:
        print(f"\nComparison: {'Batch faster' if batch_time < bundle_time else 'Bundle faster'} by a factor of {max(batch_time, bundle_time)/max(0.0001, min(batch_time, bundle_time)):.2f}x")
    
    # Execute again with caching
    print("\nExecuting both again with caching...")
    
    start_time = time.time()
    try:
        batch_result2 = await batch_processor.execute_batch(batch, parallel=True)
        batch_time2 = time.time() - start_time
        
        print(f"Batch cached execution time: {batch_time2:.2f} seconds {f'(speedup: {batch_time/max(0.0001, batch_time2):.1f}x)' if batch_time > 0 else ''}")
    except Exception as e:
        print(f"Error executing batch: {str(e)}")
    
    start_time = time.time()
    try:
        bundle_result2 = await bundle_manager.execute_bundle("perf_test_bundle", {})
        bundle_time2 = time.time() - start_time
        
        print(f"Bundle cached execution time: {bundle_time2:.2f} seconds {f'(speedup: {bundle_time/max(0.0001, bundle_time2):.1f}x)' if bundle_time > 0 else ''}")
    except Exception as e:
        print(f"Error executing bundle: {str(e)}")

async def run_demo():
    """Run the complete function bundles demo."""
    print("===== Function Bundles with Performance Optimization Demo =====")
    
    # Register demo functions and bundles
    registered = await register_demo_functions_and_bundles()
    if not registered:
        print("Cannot continue demo without registered functions and bundles.")
        return
    
    # Run the individual demos
    await demo_bundle_execution()
    await demo_conditional_bundle()
    await demo_batch_vs_bundle()
    
    print("\n===== Demo Complete =====")

if __name__ == "__main__":
    asyncio.run(run_demo()) 