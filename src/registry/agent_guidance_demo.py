#!/usr/bin/env python3
"""
Agent Guidance System Demo

This script demonstrates the functionality of the Agent Guidance System,
including function recommendations, pattern detection, and complex operations.
"""

import os
import json
import asyncio
import datetime
import time
from typing import Dict, List, Any, Optional

from src.registry.registry_manager import get_registry
from src.registry.function_models import FunctionResult
from src.registry.migration_framework import migrate_all_tools
from src.registry.agent_guidance import (
    get_recommendation_engine,
    get_pattern_detector,
    get_complex_helper
)

# Sample entities for demonstrations
SAMPLE_ENTITIES = [
    {"name": "John Doe", "entity_type": "person"},
    {"name": "Acme Corporation", "entity_type": "organization"},
    {"name": "San Francisco", "entity_type": "location"}
]

# Sample observations for demonstrations
SAMPLE_OBSERVATIONS = [
    {"entity_name": "John Doe", "content": "John is 35 years old"},
    {"entity_name": "John Doe", "content": "John works as a software engineer"},
    {"entity_name": "Acme Corporation", "content": "Acme Corporation was founded in 2010"},
    {"entity_name": "San Francisco", "content": "San Francisco is in California, USA"}
]

# Sample relations for demonstrations
SAMPLE_RELATIONS = [
    {"source": "John Doe", "relation": "works_at", "target": "Acme Corporation"},
    {"source": "John Doe", "relation": "lives_in", "target": "San Francisco"},
    {"source": "Acme Corporation", "relation": "located_in", "target": "San Francisco"}
]

async def demo_function_recommendations():
    """Demonstrate function recommendations based on natural language queries."""
    print("\n=== Function Recommendation Demo ===\n")
    
    engine = get_recommendation_engine()
    
    # Prepare some sample queries
    queries = [
        "create a new entity",
        "search for information about a person",
        "update entity data",
        "find connections between entities",
        "create an observation about an entity"
    ]
    
    # Sample context for context-aware recommendations
    context = {
        "entity_name": "John Doe",
        "entity_type": "person",
        "content": "John enjoys hiking on weekends"
    }
    
    # Show recommendations for each query
    for query in queries:
        print(f"\nQuery: \"{query}\"")
        recommendations = engine.get_recommendations(query, context, limit=3)
        
        if not recommendations:
            print("  No recommendations found")
            continue
            
        print(f"  Top {len(recommendations)} recommendations:")
        for i, rec in enumerate(recommendations):
            print(f"  {i+1}. {rec['function']} (relevance: {rec['relevance']:.2f})")
            print(f"     Description: {rec['description']}")
            if rec['pattern']:
                print(f"     Part of pattern: {rec['pattern']}")
            if rec['next_steps']:
                print(f"     Common next steps: {', '.join(rec['next_steps'])}")
    
    # Show function chains for goals
    print("\n=== Function Chain Recommendations ===\n")
    
    goals = [
        "create and document an entity",
        "process and analyze data",
        "search and update entities"
    ]
    
    for goal in goals:
        print(f"\nGoal: \"{goal}\"")
        chains = engine.get_function_chains(goal, limit=2)
        
        if not chains:
            print("  No chains found")
            continue
            
        print(f"  {len(chains)} chains found:")
        for i, chain in enumerate(chains):
            print(f"  Chain {i+1}: {chain['name']} (relevance: {chain['relevance']:.2f})")
            print(f"    Description: {chain['description']}")
            print("    Functions:")
            for func in chain['functions']:
                print(f"      - {func['name']}: {func['description']}")

async def demo_pattern_detection():
    """Demonstrate pattern detection from function usage."""
    print("\n=== Pattern Detection Demo ===\n")
    
    detector = get_pattern_detector()
    registry = get_registry()
    
    # Simulate some function calls for pattern detection
    print("Simulating function call sequences...\n")
    
    # Sequence 1: Entity creation pattern
    print("Sequence 1: Entity creation")
    await simulate_sequence([
        {"function": "memory.create_entity", "params": {"name": "Jane Smith", "entity_type": "person"}, "success": True},
        {"function": "memory.create_observation", "params": {"entity_name": "Jane Smith", "content": "Jane is 42"}, "success": True},
        {"function": "memory.create_observation", "params": {"entity_name": "Jane Smith", "content": "Jane is a doctor"}, "success": True}
    ])
    
    # Sequence 2: Search and update pattern
    print("\nSequence 2: Search and update")
    await simulate_sequence([
        {"function": "memory.semantic_search", "params": {"query": "software engineer"}, "success": True},
        {"function": "memory.get_entity", "params": {"name": "John Doe"}, "success": True},
        {"function": "memory.update_entity", "params": {"name": "John Doe", "properties": {"title": "Senior Engineer"}}, "success": True}
    ])
    
    # Sequence 3: Failed calls (anti-pattern)
    print("\nSequence 3: Anti-pattern - repeated failures")
    await simulate_sequence([
        {"function": "memory.create_relation", "params": {"source": "Alice", "relation": "knows"}, "success": False},
        {"function": "memory.create_relation", "params": {"source": "Alice", "relation": "knows"}, "success": False},
        {"function": "memory.create_relation", "params": {"source": "Alice", "relation": "knows"}, "success": False}
    ])
    
    # Sequence 4: Repeat of entity creation pattern
    print("\nSequence 4: Entity creation (repeat)")
    await simulate_sequence([
        {"function": "memory.create_entity", "params": {"name": "Bob Johnson", "entity_type": "person"}, "success": True},
        {"function": "memory.create_observation", "params": {"entity_name": "Bob Johnson", "content": "Bob is 28"}, "success": True},
        {"function": "memory.create_observation", "params": {"entity_name": "Bob Johnson", "content": "Bob is an artist"}, "success": True}
    ])
    
    # Get detected patterns
    print("\nDetected patterns:")
    patterns = detector.get_detected_patterns()
    
    if not patterns:
        print("  No patterns detected yet")
    else:
        for i, pattern in enumerate(patterns):
            print(f"  Pattern {i+1}: {pattern['description']}")
            print(f"    Functions: {' → '.join(pattern['functions'])}")
            print(f"    Frequency: {pattern['frequency']} occurrences")
            print(f"    Effectiveness: {pattern['effectiveness'] * 100:.1f}%")
    
    # Get anti-patterns
    print("\nDetected anti-patterns:")
    anti_patterns = detector.get_detected_anti_patterns()
    
    if not anti_patterns:
        print("  No anti-patterns detected")
    else:
        for i, pattern in enumerate(anti_patterns):
            print(f"  Anti-pattern {i+1}: {pattern['name']}")
            print(f"    Details: {pattern['details']['message']}")
            print(f"    Suggestion: {pattern['suggestion']}")

async def demo_complex_operations():
    """Demonstrate complex operations that combine multiple functions."""
    print("\n=== Complex Operations Demo ===\n")
    
    helper = get_complex_helper()
    
    # Show available operations
    print("Available complex operations:")
    operations = helper.get_available_operations()
    
    for category, info in operations.items():
        print(f"\n  Category: {category}")
        print(f"  Description: {info['description']}")
        print(f"  Operations:")
        
        for op in info['operations']:
            help_info = helper.get_operation_help(op)
            print(f"    - {op}: {help_info.get('description', 'No description')}")
    
    # Execute some complex operations
    print("\nExecuting complex operations:\n")
    
    # 1. Create entity with observations
    print("1. Create entity with observations:")
    entity_params = {
        "entity_name": "Alice Brown",
        "entity_type": "person",
        "observations": [
            "Alice is 32 years old",
            "Alice works as a teacher",
            "Alice lives in New York"
        ]
    }
    
    result = await helper.execute_complex_operation("create_entity_with_observations", entity_params)
    print(f"  Status: {result.status}")
    print(f"  Message: {result.message}")
    if result.data:
        print(f"  Entity created with {len(result.data.get('observations', []))} observations")
    
    # 2. Extract-transform-load operation
    print("\n2. Extract-transform-load operation:")
    etl_params = {
        "source": "database.customers",
        "transformations": [
            {"type": "rename", "from": "customer_id", "to": "id"},
            {"type": "filter", "field": "status", "value": "active"}
        ],
        "destination": "analytics.active_customers"
    }
    
    result = await helper.execute_complex_operation("extract_transform_load", etl_params)
    print(f"  Status: {result.status}")
    print(f"  Message: {result.message}")
    if result.data:
        print(f"  Records processed: {result.data.get('records_processed', 0)}")

async def simulate_sequence(calls: List[Dict[str, Any]]):
    """Simulate a sequence of function calls for pattern detection."""
    detector = get_pattern_detector()
    engine = get_recommendation_engine()
    
    for call in calls:
        function_name = call["function"]
        params = call.get("params", {})
        success = call.get("success", True)
        
        # Create a sample result
        result = FunctionResult(
            status="success" if success else "error",
            message=f"{'Successfully executed' if success else 'Error executing'} {function_name}",
            data=params if success else None,
            error_code="" if success else "EXECUTION_ERROR",
            error_details={} if success else {"reason": "Simulated error"}
        )
        
        # Track the function call
        engine.track_function_usage(function_name, params, result)
        detector.add_function_call(function_name, params, result)
        
        # Print information about the call
        status_str = "✓" if success else "✗"
        param_str = ", ".join(f'{k}="{v}"' for k, v in params.items())
        print(f"  {status_str} {function_name}({param_str})")
    
    # End the sequence
    detector.end_sequence()

async def demo_agent_guidance():
    """Run the complete Agent Guidance System demonstration."""
    print("\n=== Agent Guidance System Demonstration ===\n")
    
    # First, migrate some real tools to have a meaningful registry
    print("Migrating tools to registry...")
    migrate_all_tools()
    
    # Show each feature in action
    await demo_function_recommendations()
    await demo_pattern_detection()
    await demo_complex_operations()
    
    print("\n=== Agent Guidance System Demo Complete ===\n")
    
    # Show conclusion
    print("The Agent Guidance System features demonstrate how to:")
    print("1. Get function recommendations based on natural language queries")
    print("2. Detect patterns and anti-patterns in function usage")
    print("3. Execute complex operations that combine multiple functions")
    print("\nThese features help AI agents navigate the Function Registry more effectively,")
    print("learn from past interactions, and accomplish complex tasks efficiently.")

if __name__ == "__main__":
    asyncio.run(demo_agent_guidance()) 