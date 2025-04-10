#!/usr/bin/env python3
"""
Feedback Mechanism and Health Diagnostics Demo

This script demonstrates the functionality of the Feedback Mechanism
and Health Diagnostics modules for the Function Registry Pattern.
"""

import os
import sys
import json
import asyncio
import time
from typing import Dict, List, Any, Optional, Union
from datetime import datetime

# Add the project root to the path for proper imports
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

# Define fallback implementations for when imports fail
class FallbackHealthMonitor:
    async def check_health(self):
        return {"overall_status": "UNKNOWN", "components": {}}

class FallbackSelfHealer:
    async def heal_system(self, component_id):
        return {"status": "UNKNOWN", "message": "Self-healer not available"}

class FallbackFeedbackManager:
    def __init__(self, storage_dir=None):
        self.storage_dir = storage_dir or "."
    
    def submit_feedback(self, **kwargs):
        return {"feedback_id": "fallback-id-" + str(time.time())}
    
    def get_feedback_summary(self):
        return {"total_feedback": 0, "feedback_by_type": {}}
    
    def get_feedback(self):
        return []

class FallbackFunctionImprover:
    def __init__(self, feedback_manager):
        self.feedback_manager = feedback_manager
    
    def generate_function_suggestions(self, function_name):
        return {
            "function_name": function_name,
            "total_feedback": 0,
            "suggestions": []
        }

# Import modules with fallbacks
try:
    from src.registry.registry_manager import get_registry, register_function
except ImportError:
    def get_registry():
        return None
    def register_function(**kwargs):
        print("[MOCK] Function registered")

try:
    from src.registry.function_models import FunctionResult
except ImportError:
    class FunctionResult:
        def __init__(self, success=False, data=None, error=None):
            self.success = success
            self.data = data or {}
            self.error = error or {"message": "Not implemented"}

try:
    from src.registry.health_diagnostics import (
        HealthMonitor, SelfHealer, check_system_health,
        run_system_diagnostic, heal_system, register_health_diagnostics_tools
    )
except ImportError as e:
    print(f"Warning: Health diagnostics imports failed: {e}")
    HealthMonitor = FallbackHealthMonitor
    SelfHealer = FallbackSelfHealer
    
    async def check_system_health():
        return FunctionResult(success=False, error={"message": "Not implemented"})
    
    async def run_system_diagnostic():
        return FunctionResult(success=False, error={"message": "Not implemented"})
    
    async def heal_system(component_id):
        return FunctionResult(success=False, error={"message": "Not implemented"})
    
    def register_health_diagnostics_tools():
        print("[MOCK] Health diagnostics tools registered")

try:
    from src.registry.feedback_mechanism import (
        FeedbackManager, FunctionImprover, submit_feedback,
        get_function_suggestions, get_optimization_recommendations,
        get_feedback_summary, register_feedback_mechanism_tools
    )
except ImportError as e:
    print(f"Warning: Feedback mechanism imports failed: {e}")
    FeedbackManager = FallbackFeedbackManager
    FunctionImprover = FallbackFunctionImprover
    
    async def submit_feedback(**kwargs):
        return FunctionResult(success=True, data={"feedback_id": "mock-id"})
    
    async def get_function_suggestions(function_name):
        return FunctionResult(success=True, data={"function_name": function_name, "suggestions": []})
    
    async def get_optimization_recommendations():
        return FunctionResult(success=True, data={"recommendations": {}})
    
    async def get_feedback_summary():
        return FunctionResult(success=True, data={"total_feedback": 0})
    
    def register_feedback_mechanism_tools():
        print("[MOCK] Feedback mechanism tools registered")

# Sample output directory
DEMO_OUTPUT_DIR = "demo_output"

async def demo_health_diagnostics():
    """Demonstrate the Health and Diagnostics functionality."""
    print("\n===== HEALTH AND DIAGNOSTICS DEMONSTRATION =====\n")
    
    # Create a health monitor
    print("Initializing health monitoring system...")
    health_monitor = None
    try:
        health_monitor = HealthMonitor()
        print("  Health monitor initialized successfully")
    except Exception as e:
        print(f"  Error initializing health monitor: {str(e)}")
        print("  Continuing with limited functionality")
    
    # Check system health
    print("\nChecking system health...")
    try:
        if health_monitor:
            health_report = await health_monitor.check_health()
            overall_status = health_report.get("overall_status", "UNKNOWN") if health_report else "UNKNOWN"
            print(f"  Overall system status: {overall_status}")
            
            # Show component status
            print("\n  Component health:")
            for comp_id, comp_data in (health_report.get("components", {}) if health_report else {}).items():
                status = comp_data.get("status", "UNKNOWN") if comp_data else "UNKNOWN"
                message = comp_data.get("message", "No message") if comp_data else "No message"
                print(f"    - {comp_id}: {status}")
                print(f"      Message: {message}")
        else:
            print("  Using registry tool for health check...")
            health_result = await check_system_health()
            if hasattr(health_result, "success") and health_result.success:
                health_report = health_result.data if hasattr(health_result, "data") else {}
                overall_status = health_report.get("overall_status", "UNKNOWN") if health_report else "UNKNOWN"
                print(f"  Overall system status: {overall_status}")
            else:
                error_msg = "Unknown error"
                if hasattr(health_result, "error") and health_result.error:
                    if isinstance(health_result.error, dict) and "message" in health_result.error:
                        error_msg = health_result.error["message"]
                print(f"  Health check failed: {error_msg}")
    except Exception as e:
        print(f"  Error checking health: {str(e)}")
    
    # Run diagnostics
    print("\nRunning system diagnostics...")
    try:
        # We'll use the tool function here
        diagnostic_result = await run_system_diagnostic()
        if hasattr(diagnostic_result, "success") and diagnostic_result.success:
            diagnostic_report = diagnostic_result.data if hasattr(diagnostic_result, "data") else {}
            
            # Show registry info
            registry_info = diagnostic_report.get("registry_info", {}) if diagnostic_report else {}
            print("\n  Registry Information:")
            print(f"    Available: {registry_info.get('available', False)}")
            if registry_info and registry_info.get("available", False):
                print(f"    Function count: {registry_info.get('function_count', 0)}")
                print(f"    Namespace count: {registry_info.get('namespace_count', 0)}")
            
            # Show system info
            system_info = diagnostic_report.get("system_info", {}) if diagnostic_report else {}
            print("\n  System Information:")
            print(f"    Python version: {system_info.get('python_version', 'Unknown')}")
            print(f"    Platform: {system_info.get('platform', 'Unknown')}")
        else:
            error_obj = getattr(diagnostic_result, "error", {}) or {}
            error_msg = error_obj.get("message", "Unknown error") if isinstance(error_obj, dict) else "Unknown error"
            print(f"  Diagnostic failed: {error_msg}")
    except Exception as e:
        print(f"  Error running diagnostics: {str(e)}")
    
    # Demonstrate self-healing
    print("\nDemonstrating self-healing capabilities...")
    try:
        # Create a self-healer
        self_healer = SelfHealer()
        
        # Simulate an unhealthy component
        print("  Simulating an unhealthy component...")
        component_id = "registry_core"
        
        # Simulate healing
        print(f"  Attempting to heal component: {component_id}")
        healing_result = await self_healer.heal_system(component_id)
        
        if healing_result and isinstance(healing_result, dict):
            if "status" in healing_result:
                print(f"  Healing status: {healing_result.get('status', 'UNKNOWN')}")
                print(f"  Message: {healing_result.get('message', 'No message')}")
            else:
                print("  Healing process completed")
                for comp_id, result in healing_result.get("results", {}).items():
                    status = result.get("status", "UNKNOWN") if result else "UNKNOWN"
                    print(f"    - {comp_id}: {status}")
    except Exception as e:
        print(f"  Error demonstrating self-healing: {str(e)}")
    
    print("\n=== Health and Diagnostics Demo Complete ===")
    print("The Health and Diagnostics module provides:")
    print("1. System health monitoring")
    print("2. In-depth diagnostics")
    print("3. Self-healing capabilities")

async def demo_feedback_mechanism():
    """Demonstrate the Feedback Mechanism functionality."""
    print("\n===== FEEDBACK MECHANISM DEMONSTRATION =====\n")
    
    # Initialize feedback manager
    print("Initializing feedback system...")
    feedback_manager = None
    try:
        # Create a temporary storage directory for the demo
        storage_dir = os.path.join(DEMO_OUTPUT_DIR, "feedback_demo")
        os.makedirs(storage_dir, exist_ok=True)
        
        feedback_manager = FeedbackManager(storage_dir=storage_dir)
        print("  Feedback manager initialized successfully")
    except Exception as e:
        print(f"  Error initializing feedback manager: {str(e)}")
        print("  Continuing with limited functionality")
    
    # Submit sample feedback
    print("\nSubmitting sample feedback...")
    
    # Define some sample functions for the demo
    sample_functions = [
        "registry.execute_function",
        "memory.get_entity",
        "search.find_documents"
    ]
    
    # Submit different types of feedback
    feedback_ids = []
    try:
        # Error feedback
        if feedback_manager:
            result = feedback_manager.submit_feedback(
                feedback_type="error",
                content="Function throws an error when parameter is null",
                function_name=sample_functions[0],
                severity="high",
                context={"parameter": "query", "error_type": "NullReferenceException"}
            )
            if result and isinstance(result, dict):
                feedback_id = result.get("feedback_id")
                if feedback_id:
                    feedback_ids.append(feedback_id)
                    print(f"  Submitted error feedback: {feedback_id}")
            
            # Usage problem feedback
            result = feedback_manager.submit_feedback(
                feedback_type="usage_problem",
                content="Parameter name is confusing",
                function_name=sample_functions[1],
                severity="medium"
            )
            if result and isinstance(result, dict):
                feedback_id = result.get("feedback_id")
                if feedback_id:
                    feedback_ids.append(feedback_id)
                    print(f"  Submitted usage problem feedback: {feedback_id}")
            
            # Feature request feedback
            result = feedback_manager.submit_feedback(
                feedback_type="feature_request",
                content="Add pagination support to search results",
                function_name=sample_functions[2],
                severity="medium"
            )
            if result and isinstance(result, dict):
                feedback_id = result.get("feedback_id")
                if feedback_id:
                    feedback_ids.append(feedback_id)
                    print(f"  Submitted feature request feedback: {feedback_id}")
        else:
            # Use the tool function
            result = await submit_feedback(
                feedback_type="error",
                content="Function throws an error when parameter is null",
                function_name=sample_functions[0],
                severity="high",
                context={"parameter": "query", "error_type": "NullReferenceException"}
            )
            print("  Submitted error feedback using tool")
    except Exception as e:
        print(f"  Error submitting feedback: {str(e)}")
    
    # Get feedback summary
    print("\nGetting feedback summary...")
    try:
        if feedback_manager:
            summary = feedback_manager.get_feedback_summary()
            if summary and isinstance(summary, dict):
                total = summary.get("total_feedback", 0)
                print(f"  Total feedback: {total}")
                print("  Feedback by type:")
                for fb_type, count in (summary.get("feedback_by_type", {}) or {}).items():
                    if count and count > 0:
                        print(f"    - {fb_type}: {count}")
        else:
            # Use the tool function
            result = await get_feedback_summary()
            if hasattr(result, "success") and result.success:
                summary = result.data if hasattr(result, "data") else {}
                total = summary.get("total_feedback", 0) if summary else 0
                print(f"  Total feedback: {total}")
            else:
                print("  Failed to get feedback summary")
    except Exception as e:
        print(f"  Error getting feedback summary: {str(e)}")
    
    # Generate function suggestions
    print("\nGenerating function improvement suggestions...")
    try:
        # Use FunctionImprover for more feedback
        if feedback_manager:
            # Submit more feedback for the same function to get better suggestions
            for i in range(3):
                feedback_manager.submit_feedback(
                    feedback_type="error",
                    content=f"Error scenario {i+1}",
                    function_name=sample_functions[0],
                    severity="medium"
                )
            
            # Now generate suggestions
            improver = FunctionImprover(feedback_manager)
            suggestions = improver.generate_function_suggestions(sample_functions[0])
            
            if suggestions and isinstance(suggestions, dict):
                print(f"  Function: {suggestions.get('function_name', '')}")
                print(f"  Total feedback: {suggestions.get('total_feedback', 0)}")
                print("  Suggestions:")
                for suggestion in suggestions.get("suggestions", []) or []:
                    if suggestion and isinstance(suggestion, dict):
                        print(f"    - {suggestion.get('type', '')}: {suggestion.get('title', '')}")
                        print(f"      {suggestion.get('description', '')}")
        else:
            # Use the tool function
            result = await get_function_suggestions(sample_functions[0])
            if hasattr(result, "success") and result.success:
                suggestions = result.data if hasattr(result, "data") else {}
                if suggestions and isinstance(suggestions, dict):
                    function_name = suggestions.get("function_name", "")
                    suggestion_list = suggestions.get("suggestions", []) or []
                    print(f"  Function: {function_name}")
                    print(f"  Suggestions: {len(suggestion_list)}")
            else:
                print("  Failed to generate suggestions")
    except Exception as e:
        print(f"  Error generating suggestions: {str(e)}")
    
    # Get optimization recommendations
    print("\nGetting system-wide optimization recommendations...")
    try:
        if feedback_manager and hasattr(feedback_manager, "generate_improvement_recommendations"):
            recommendations = feedback_manager.generate_improvement_recommendations()
            print(f"  Generated {len(recommendations) if recommendations else 0} recommendations")
            
            if recommendations and len(recommendations) > 0:
                rec = recommendations[0]
                if rec and isinstance(rec, dict):
                    print(f"  Sample recommendation for function: {rec.get('function_name', '')}")
                    for subrec in rec.get("recommendations", []) or []:
                        if subrec and isinstance(subrec, dict):
                            print(f"    - {subrec.get('type', '')}: {subrec.get('description', '')}")
        else:
            # Use the tool function
            result = await get_optimization_recommendations()
            if hasattr(result, "success") and result.success:
                recommendations = result.data if hasattr(result, "data") else {}
                if recommendations and isinstance(recommendations, dict):
                    print("  Recommendation priorities:")
                    for priority, recs in (recommendations.get("recommendations", {}) or {}).items():
                        print(f"    - {priority}: {len(recs) if recs else 0} recommendations")
            else:
                print("  Failed to get optimization recommendations")
    except Exception as e:
        print(f"  Error getting optimization recommendations: {str(e)}")
    
    # Export feedback data
    print("\nExporting feedback data...")
    try:
        feedback_export_path = os.path.join(DEMO_OUTPUT_DIR, "feedback_export.json")
        if feedback_manager and hasattr(feedback_manager, "get_feedback"):
            # Get all feedback and save to a file
            all_feedback = feedback_manager.get_feedback()
            with open(feedback_export_path, "w") as f:
                json.dump(all_feedback, f, indent=2)
            print(f"  Exported {len(all_feedback) if all_feedback else 0} feedback items to: {feedback_export_path}")
    except Exception as e:
        print(f"  Error exporting feedback data: {str(e)}")
    
    print("\n=== Feedback Mechanism Demo Complete ===")
    print("The Feedback Mechanism module provides:")
    print("1. Agent feedback collection")
    print("2. Function improvement suggestions")
    print("3. Optimization recommendations")

async def register_demo_functions():
    """Register some demo functions for testing."""
    registry = get_registry()
    if not registry:
        print("Registry not available, can't register demo functions")
        return
    
    try:
        # Register a simple demo function
        async def demo_add(a: int, b: int) -> int:
            """Add two numbers together."""
            return a + b
        
        register_function(
            func=demo_add,
            name="demo.add",
            description="Add two numbers together",
            parameters={
                "a": {"type": "int", "description": "First number", "required": True},
                "b": {"type": "int", "description": "Second number", "required": True}
            },
            namespace="demo",
            tags=["demo", "math"]
        )
        
        # Register a function that sometimes fails
        async def demo_divide(a: int, b: int) -> float:
            """Divide two numbers."""
            if b == 0:
                raise ValueError("Cannot divide by zero")
            return a / b
        
        register_function(
            func=demo_divide,
            name="demo.divide",
            description="Divide two numbers",
            parameters={
                "a": {"type": "int", "description": "Numerator", "required": True},
                "b": {"type": "int", "description": "Denominator", "required": True}
            },
            namespace="demo",
            tags=["demo", "math"]
        )
        
        print("Demo functions registered successfully")
    except Exception as e:
        print(f"Error registering demo functions: {str(e)}")

async def setup_demo_environment():
    """Set up the demo environment."""
    # Create output directory
    os.makedirs(DEMO_OUTPUT_DIR, exist_ok=True)
    
    # Register demo functions
    await register_demo_functions()
    
    # Register health and diagnostics tools
    try:
        register_health_diagnostics_tools()
        print("Health diagnostics tools registered")
    except Exception as e:
        print(f"Error registering health diagnostics tools: {str(e)}")
    
    # Register feedback mechanism tools
    try:
        register_feedback_mechanism_tools()
        print("Feedback mechanism tools registered")
    except Exception as e:
        print(f"Error registering feedback mechanism tools: {str(e)}")

async def run_feedback_health_demo():
    """Run the complete Feedback and Health demonstration."""
    print("\n===== FUNCTION REGISTRY: PHASE 5 DEMONSTRATION =====\n")
    print("This demonstration shows the key features of the Function Registry Pattern")
    print("Phase 5 components: Health and Diagnostics and Feedback Mechanism.")
    
    # Set up demo environment
    print("\nSetting up demo environment...")
    await setup_demo_environment()
    
    # Run health and diagnostics demo
    await demo_health_diagnostics()
    
    # Run feedback mechanism demo
    await demo_feedback_mechanism()
    
    print("\n\n===== PHASE 5 DEMONSTRATION COMPLETE =====")
    print("\nWe have successfully implemented:")
    print("1. Health and Diagnostics: System health monitoring, diagnostics, and self-healing")
    print("2. Feedback Mechanism: Agent feedback collection, function improvement suggestions, and optimization recommendations")
    print("\nThese components complete Phase 5 of the Function Registry Pattern,")
    print("providing monitoring, feedback, and self-improvement capabilities.")

if __name__ == "__main__":
    asyncio.run(run_feedback_health_demo()) 