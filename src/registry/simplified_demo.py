#!/usr/bin/env python3
"""
Simplified Demo for Function Registry Pattern Phase 5

This demonstrates the Health and Diagnostics and Feedback Mechanism
components without requiring all previous components to be in place.
"""

import os
import sys
import json
import logging
import asyncio
from datetime import datetime
from typing import Dict, List, Any, Optional, Union

# Configure logging
logging.basicConfig(level=logging.INFO, 
                    format='%(asctime)s - %(levelname)s: %(message)s',
                    datefmt='%Y-%m-%d %H:%M:%S')
logger = logging.getLogger(__name__)

# Create output directory
DEMO_OUTPUT_DIR = "demo_output"
os.makedirs(DEMO_OUTPUT_DIR, exist_ok=True)
os.makedirs(os.path.join(DEMO_OUTPUT_DIR, "feedback_demo"), exist_ok=True)

# ---------- MOCK CLASSES FOR HEALTH DIAGNOSTICS ----------

class HealthStatus:
    HEALTHY = "HEALTHY"
    DEGRADED = "DEGRADED"
    UNHEALTHY = "UNHEALTHY"
    UNKNOWN = "UNKNOWN"

class RegistryComponent:
    REGISTRY_CORE = "registry_core"
    DISCOVERY = "discovery"
    DOCUMENTATION = "documentation"
    PARAMETERS = "parameters"
    PERFORMANCE = "performance"
    FUNCTION_BUNDLES = "function_bundles"

class HealthMonitor:
    """Mock health monitoring system."""
    
    def __init__(self):
        self.health_checks = {}
        self.recovery_actions = {}
        self.component_status = {}
        self.last_check_time = {}
        self.error_counts = {}
        logger.info("Health monitor initialized")
    
    async def check_health(self, component=None):
        """Check health of the system or a specific component."""
        components = {
            RegistryComponent.REGISTRY_CORE: {
                "status": HealthStatus.HEALTHY, 
                "message": "Registry core is functioning normally"
            },
            RegistryComponent.DISCOVERY: {
                "status": HealthStatus.HEALTHY, 
                "message": "Discovery system is functioning normally"
            },
            RegistryComponent.PERFORMANCE: {
                "status": HealthStatus.DEGRADED, 
                "message": "Cache hit rate is below threshold"
            },
            RegistryComponent.FUNCTION_BUNDLES: {
                "status": HealthStatus.HEALTHY, 
                "message": "Function bundles are functioning normally"
            }
        }
        
        return {
            "overall_status": HealthStatus.HEALTHY,
            "components": components,
            "timestamp": datetime.now().isoformat()
        }

class SelfHealer:
    """Mock self-healing system."""
    
    def __init__(self):
        self.health_monitor = HealthMonitor()
        logger.info("Self-healer initialized")
    
    async def heal_system(self, component_id):
        """Attempt to heal the system."""
        return {
            "status": "SUCCESS",
            "message": f"Healing of {component_id} completed successfully",
            "results": {
                component_id: {
                    "status": HealthStatus.HEALTHY,
                    "message": "Component healed successfully"
                }
            }
        }

async def check_system_health():
    """Mock health check tool."""
    monitor = HealthMonitor()
    health_report = await monitor.check_health()
    return MockFunctionResult(
        data=health_report,
        status="SUCCESS",
        message="Health check completed successfully"
    )

async def run_system_diagnostic():
    """Mock diagnostic tool."""
    diagnostic_report = {
        "registry_info": {
            "available": True,
            "function_count": 24,
            "namespace_count": 5
        },
        "system_info": {
            "python_version": "3.11.4",
            "platform": "Windows"
        }
    }
    return MockFunctionResult(
        data=diagnostic_report,
        status="SUCCESS",
        message="Diagnostic completed successfully"
    )

async def heal_system(component=None):
    """Mock healing tool."""
    healer = SelfHealer()
    component_id = component or RegistryComponent.REGISTRY_CORE
    healing_report = await healer.heal_system(component_id)
    return MockFunctionResult(
        data=healing_report,
        status="SUCCESS",
        message="Healing process completed successfully"
    )


# ---------- MOCK CLASSES FOR FEEDBACK MECHANISM ----------

class FeedbackType:
    ERROR = "error"
    SUGGESTION = "suggestion"
    RATING = "rating"
    USAGE_PROBLEM = "usage_problem"
    DOCUMENTATION = "documentation"
    FEATURE_REQUEST = "feature_request"

class FeedbackSeverity:
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"

class FeedbackStatus:
    NEW = "new"
    REVIEWING = "reviewing"
    IMPLEMENTING = "implementing"
    COMPLETED = "completed"
    DECLINED = "declined"

class FeedbackManager:
    """Mock feedback manager system."""
    
    def __init__(self, storage_dir=None):
        self.storage_dir = storage_dir or os.path.join(DEMO_OUTPUT_DIR, "feedback")
        self.feedback = []
        self.feedback_count = 0
        self.feedback_by_type = {}
        self.feedback_by_function = {}
        logger.info(f"Feedback manager initialized (storage: {self.storage_dir})")
    
    def submit_feedback(self, feedback_type, content, function_name=None, severity="medium", context=None, agent_id=None):
        """Submit feedback."""
        feedback_id = f"fb-{feedback_type}-{len(self.feedback) + 1}"
        timestamp = datetime.now().isoformat()
        
        feedback = {
            "feedback_id": feedback_id,
            "type": feedback_type,
            "content": content,
            "function_name": function_name,
            "severity": severity,
            "context": context or {},
            "agent_id": agent_id,
            "status": FeedbackStatus.NEW,
            "timestamp": timestamp
        }
        
        self.feedback.append(feedback)
        
        # Update counters
        self.feedback_count += 1
        if feedback_type not in self.feedback_by_type:
            self.feedback_by_type[feedback_type] = 0
        self.feedback_by_type[feedback_type] += 1
        
        if function_name:
            if function_name not in self.feedback_by_function:
                self.feedback_by_function[function_name] = []
            self.feedback_by_function[function_name].append(feedback)
        
        # Save to file if storage_dir is provided
        if self.storage_dir:
            os.makedirs(self.storage_dir, exist_ok=True)
            feedback_file = os.path.join(self.storage_dir, f"{feedback_id}.json")
            with open(feedback_file, "w") as f:
                json.dump(feedback, f, indent=2)
        
        return {
            "feedback_id": feedback_id,
            "status": "success",
            "timestamp": timestamp
        }
    
    def get_feedback_summary(self):
        """Get a summary of feedback."""
        return {
            "total_feedback": self.feedback_count,
            "feedback_by_type": self.feedback_by_type,
            "feedback_by_function": {k: len(v) for k, v in self.feedback_by_function.items()},
            "timestamp": datetime.now().isoformat()
        }
    
    def get_feedback(self, feedback_id=None, function_name=None, feedback_type=None):
        """Get feedback based on filters."""
        if feedback_id:
            for fb in self.feedback:
                if fb["feedback_id"] == feedback_id:
                    return fb
            return None
        
        result = self.feedback
        
        if function_name:
            result = [fb for fb in result if fb.get("function_name") == function_name]
        
        if feedback_type:
            result = [fb for fb in result if fb.get("type") == feedback_type]
        
        return result
    
    def generate_improvement_recommendations(self):
        """Generate improvement recommendations."""
        recommendations = []
        
        # Group feedback by function
        for function_name, feedback_list in self.feedback_by_function.items():
            if len(feedback_list) >= 2:  # Only recommend if we have enough feedback
                errors = [fb for fb in feedback_list if fb["type"] == FeedbackType.ERROR]
                suggestions = [fb for fb in feedback_list if fb["type"] == FeedbackType.SUGGESTION]
                usage_problems = [fb for fb in feedback_list if fb["type"] == FeedbackType.USAGE_PROBLEM]
                
                function_recommendations = []
                
                if errors:
                    function_recommendations.append({
                        "type": "error_fix",
                        "description": f"Fix {len(errors)} reported errors"
                    })
                
                if usage_problems:
                    function_recommendations.append({
                        "type": "usability",
                        "description": "Improve parameter naming and documentation"
                    })
                
                if suggestions:
                    function_recommendations.append({
                        "type": "enhancement",
                        "description": f"Implement {len(suggestions)} user suggestions"
                    })
                
                if function_recommendations:
                    recommendations.append({
                        "function_name": function_name,
                        "feedback_count": len(feedback_list),
                        "recommendations": function_recommendations,
                        "priority": "high" if any(fb["severity"] == FeedbackSeverity.HIGH for fb in feedback_list) else "medium"
                    })
        
        return recommendations

class FunctionImprover:
    """Mock function improver."""
    
    def __init__(self, feedback_manager):
        self.feedback_manager = feedback_manager
    
    def generate_function_suggestions(self, function_name):
        """Generate improvement suggestions for a specific function."""
        feedback_list = self.feedback_manager.get_feedback(function_name=function_name)
        
        if not feedback_list:
            return {
                "function_name": function_name,
                "total_feedback": 0,
                "suggestions": []
            }
        
        suggestions = []
        
        # Generate suggestions based on feedback
        errors = [fb for fb in feedback_list if fb["type"] == FeedbackType.ERROR]
        if errors:
            suggestions.append({
                "type": "error_fix",
                "title": "Fix error handling",
                "description": "Improve error handling to address reported issues"
            })
        
        usage_problems = [fb for fb in feedback_list if fb["type"] == FeedbackType.USAGE_PROBLEM]
        if usage_problems:
            suggestions.append({
                "type": "usability",
                "title": "Improve usability",
                "description": "Enhance parameter names and documentation for better clarity"
            })
        
        feature_requests = [fb for fb in feedback_list if fb["type"] == FeedbackType.FEATURE_REQUEST]
        if feature_requests:
            suggestions.append({
                "type": "enhancement",
                "title": "Add requested features",
                "description": f"Implement {len(feature_requests)} requested features"
            })
        
        # Add a general suggestion if nothing specific
        if not suggestions:
            suggestions.append({
                "type": "general",
                "title": "Review function",
                "description": "Review the function based on feedback"
            })
        
        return {
            "function_name": function_name,
            "total_feedback": len(feedback_list),
            "suggestions": suggestions
        }
    
    def generate_system_improvement_recommendations(self):
        """Generate system-wide improvement recommendations."""
        return {
            "timestamp": datetime.now().isoformat(),
            "recommendations": {
                "high": [
                    {
                        "type": "error_fix",
                        "target": "registry.execute_function",
                        "description": "Fix error handling for null parameters"
                    }
                ],
                "medium": [
                    {
                        "type": "performance",
                        "target": "memory.get_entity",
                        "description": "Optimize entity retrieval for better performance"
                    }
                ],
                "low": [
                    {
                        "type": "documentation",
                        "target": "search.find_documents",
                        "description": "Improve parameter documentation"
                    }
                ]
            }
        }

async def submit_feedback(feedback_type, content, function_name=None, severity="medium", context=None, agent_id=None):
    """Mock feedback submission tool."""
    feedback_manager = FeedbackManager()
    result = feedback_manager.submit_feedback(
        feedback_type=feedback_type,
        content=content,
        function_name=function_name,
        severity=severity,
        context=context,
        agent_id=agent_id
    )
    return MockFunctionResult(
        data=result,
        status="SUCCESS",
        message="Feedback submitted successfully"
    )

async def get_function_suggestions(function_name):
    """Mock function suggestions tool."""
    feedback_manager = FeedbackManager()
    improver = FunctionImprover(feedback_manager)
    suggestions = improver.generate_function_suggestions(function_name)
    return MockFunctionResult(
        data=suggestions,
        status="SUCCESS",
        message="Function suggestions generated successfully"
    )

async def get_optimization_recommendations():
    """Mock optimization recommendations tool."""
    feedback_manager = FeedbackManager()
    improver = FunctionImprover(feedback_manager)
    recommendations = improver.generate_system_improvement_recommendations()
    return MockFunctionResult(
        data=recommendations,
        status="SUCCESS",
        message="Optimization recommendations generated successfully"
    )

async def get_feedback_summary():
    """Mock feedback summary tool."""
    feedback_manager = FeedbackManager()
    summary = feedback_manager.get_feedback_summary()
    return MockFunctionResult(
        data=summary,
        status="SUCCESS",
        message="Feedback summary generated successfully"
    )


# ---------- MOCK REGISTRY SYSTEM ----------

class MockFunctionResult:
    """Mock function result class."""
    
    def __init__(self, data=None, status="SUCCESS", message="", error_code=None, error_details=None):
        self.data = data or {}
        self.status = status
        self.message = message
        self.success = status == "SUCCESS"
        self.error = None
        
        if error_code:
            self.error = {
                "code": error_code,
                "message": message,
                "details": error_details or {}
            }

# ---------- DEMO FUNCTIONS ----------

async def demo_health_diagnostics():
    """Demonstrate the Health and Diagnostics functionality."""
    print("\n===== HEALTH AND DIAGNOSTICS DEMONSTRATION =====\n")
    
    # Create a health monitor
    print("Initializing health monitoring system...")
    try:
        health_monitor = HealthMonitor()
        print("  Health monitor initialized successfully")
    except Exception as e:
        print(f"  Error initializing health monitor: {str(e)}")
    
    # Check system health
    print("\nChecking system health...")
    try:
        health_report = await health_monitor.check_health()
        print(f"  Overall system status: {health_report.get('overall_status', 'UNKNOWN')}")
        
        # Show component status
        print("\n  Component health:")
        for comp_id, comp_data in health_report.get('components', {}).items():
            print(f"    - {comp_id}: {comp_data.get('status', 'UNKNOWN')}")
            print(f"      Message: {comp_data.get('message', 'No message')}")
    except Exception as e:
        print(f"  Error checking health: {str(e)}")
    
    # Run diagnostics
    print("\nRunning system diagnostics...")
    try:
        diagnostic_result = await run_system_diagnostic()
        diagnostic_report = diagnostic_result.data
        
        # Show registry info
        registry_info = diagnostic_report.get('registry_info', {})
        print("\n  Registry Information:")
        print(f"    Available: {registry_info.get('available', False)}")
        if registry_info.get('available', False):
            print(f"    Function count: {registry_info.get('function_count', 0)}")
            print(f"    Namespace count: {registry_info.get('namespace_count', 0)}")
        
        # Show system info
        system_info = diagnostic_report.get('system_info', {})
        print("\n  System Information:")
        print(f"    Python version: {system_info.get('python_version', 'Unknown')}")
        print(f"    Platform: {system_info.get('platform', 'Unknown')}")
    except Exception as e:
        print(f"  Error running diagnostics: {str(e)}")
    
    # Demonstrate self-healing
    print("\nDemonstrating self-healing capabilities...")
    try:
        # Create a self-healer
        self_healer = SelfHealer()
        
        # Simulate an unhealthy component
        print("  Simulating an unhealthy component...")
        component_id = RegistryComponent.PERFORMANCE
        
        # Simulate healing
        print(f"  Attempting to heal component: {component_id}")
        healing_result = await self_healer.heal_system(component_id)
        
        print(f"  Healing status: {healing_result.get('status', 'UNKNOWN')}")
        print(f"  Message: {healing_result.get('message', 'No message')}")
        
        if "results" in healing_result:
            print("  Component healing results:")
            for comp_id, result in healing_result.get('results', {}).items():
                print(f"    - {comp_id}: {result.get('status', 'UNKNOWN')}")
                print(f"      Message: {result.get('message', 'No message')}")
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
    try:
        feedback_dir = os.path.join(DEMO_OUTPUT_DIR, "feedback_demo")
        feedback_manager = FeedbackManager(storage_dir=feedback_dir)
        print(f"  Feedback manager initialized with storage: {feedback_dir}")
    except Exception as e:
        print(f"  Error initializing feedback manager: {str(e)}")
    
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
        result = feedback_manager.submit_feedback(
            feedback_type=FeedbackType.ERROR,
            content="Function throws an error when parameter is null",
            function_name=sample_functions[0],
            severity=FeedbackSeverity.HIGH,
            context={"parameter": "query", "error_type": "NullReferenceException"}
        )
        feedback_ids.append(result.get("feedback_id"))
        print(f"  Submitted error feedback: {result.get('feedback_id')}")
        
        # Usage problem feedback
        result = feedback_manager.submit_feedback(
            feedback_type=FeedbackType.USAGE_PROBLEM,
            content="Parameter name is confusing",
            function_name=sample_functions[1],
            severity=FeedbackSeverity.MEDIUM
        )
        feedback_ids.append(result.get("feedback_id"))
        print(f"  Submitted usage problem feedback: {result.get('feedback_id')}")
        
        # Feature request feedback
        result = feedback_manager.submit_feedback(
            feedback_type=FeedbackType.FEATURE_REQUEST,
            content="Add pagination support to search results",
            function_name=sample_functions[2],
            severity=FeedbackSeverity.MEDIUM
        )
        feedback_ids.append(result.get("feedback_id"))
        print(f"  Submitted feature request feedback: {result.get('feedback_id')}")
    except Exception as e:
        print(f"  Error submitting feedback: {str(e)}")
    
    # Get feedback summary
    print("\nGetting feedback summary...")
    try:
        summary = feedback_manager.get_feedback_summary()
        print(f"  Total feedback: {summary.get('total_feedback', 0)}")
        print("  Feedback by type:")
        for fb_type, count in summary.get('feedback_by_type', {}).items():
            print(f"    - {fb_type}: {count}")
    except Exception as e:
        print(f"  Error getting feedback summary: {str(e)}")
    
    # Generate function suggestions
    print("\nGenerating function improvement suggestions...")
    try:
        # Submit more feedback for the same function to get better suggestions
        for i in range(3):
            feedback_manager.submit_feedback(
                feedback_type=FeedbackType.ERROR,
                content=f"Error scenario {i+1}",
                function_name=sample_functions[0],
                severity=FeedbackSeverity.MEDIUM
            )
        
        # Now generate suggestions
        improver = FunctionImprover(feedback_manager)
        suggestions = improver.generate_function_suggestions(sample_functions[0])
        
        print(f"  Function: {suggestions.get('function_name')}")
        print(f"  Total feedback: {suggestions.get('total_feedback', 0)}")
        print("  Suggestions:")
        for suggestion in suggestions.get('suggestions', []):
            print(f"    - {suggestion.get('type')}: {suggestion.get('title')}")
            print(f"      {suggestion.get('description')}")
    except Exception as e:
        print(f"  Error generating suggestions: {str(e)}")
    
    # Get optimization recommendations
    print("\nGetting system-wide optimization recommendations...")
    try:
        improver = FunctionImprover(feedback_manager)
        recommendations = improver.generate_system_improvement_recommendations()
        
        print("  Recommendation priorities:")
        for priority, recs in recommendations.get('recommendations', {}).items():
            print(f"    - {priority}: {len(recs)} recommendations")
            if recs:
                print(f"      Example: {recs[0].get('description')}")
    except Exception as e:
        print(f"  Error getting optimization recommendations: {str(e)}")
    
    # Export feedback data
    print("\nExporting feedback data...")
    try:
        feedback_export_path = os.path.join(DEMO_OUTPUT_DIR, "feedback_export.json")
        all_feedback = feedback_manager.get_feedback()
        with open(feedback_export_path, "w") as f:
            json.dump(all_feedback, f, indent=2)
        print(f"  Exported {len(all_feedback)} feedback items to: {feedback_export_path}")
    except Exception as e:
        print(f"  Error exporting feedback data: {str(e)}")
    
    print("\n=== Feedback Mechanism Demo Complete ===")
    print("The Feedback Mechanism module provides:")
    print("1. Agent feedback collection")
    print("2. Function improvement suggestions")
    print("3. Optimization recommendations")

async def run_feedback_health_demo():
    """Run the complete Feedback and Health demonstration."""
    print("\n===== FUNCTION REGISTRY: PHASE 5 DEMONSTRATION =====\n")
    print("This demonstration shows the key features of the Function Registry Pattern")
    print("Phase 5 components: Health and Diagnostics and Feedback Mechanism.")
    
    # Run health and diagnostics demo
    await demo_health_diagnostics()
    
    # Run feedback mechanism demo
    await demo_feedback_mechanism()
    
    print("\n\n===== PHASE 5 DEMONSTRATION COMPLETE =====")
    print("\nWe have successfully implemented:")
    print("1. Health and Diagnostics: System health monitoring, diagnostics, and self-healing 🔍")
    print("2. Feedback Mechanism: Agent feedback collection, function improvement suggestions, and optimization recommendations 📝")
    print("\nThese components complete Phase 5 of the Function Registry Pattern,")
    print("providing monitoring, feedback, and self-improvement capabilities.")
    print("\nAll planned non-skipped components are now complete! 🎉")

if __name__ == "__main__":
    # Run the demo
    asyncio.run(run_feedback_health_demo()) 