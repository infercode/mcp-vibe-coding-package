#!/usr/bin/env python3
"""
Feedback Mechanism Module

This module provides capabilities for collecting feedback from agents,
generating function improvement suggestions, and creating automated
optimization recommendations for the Function Registry Pattern.
"""

import json
import time
import asyncio
import traceback
from typing import Dict, List, Any, Optional, Union, Callable, Set, Tuple
from datetime import datetime, timedelta
from enum import Enum
import os

from src.registry.registry_manager import get_registry
from src.registry.function_models import FunctionResult, FunctionMetadata
from src.logger import get_logger

# Initialize logger
logger = get_logger()

# Feedback types
class FeedbackType(str, Enum):
    """Types of feedback that can be submitted."""
    GENERAL = "general"
    ERROR = "error"
    SUGGESTION = "suggestion"
    RATING = "rating"
    USAGE_PROBLEM = "usage_problem"
    DOCUMENTATION = "documentation"
    FEATURE_REQUEST = "feature_request"

# Feedback severity levels
class FeedbackSeverity(str, Enum):
    """Severity levels for feedback."""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"

# Feedback status
class FeedbackStatus(str, Enum):
    """Status of the feedback."""
    NEW = "new"
    ACKNOWLEDGED = "acknowledged"
    IN_PROGRESS = "in_progress"
    RESOLVED = "resolved"
    WONT_FIX = "wont_fix"
    DUPLICATE = "duplicate"

class FeedbackManager:
    """
    Manages feedback collection and processing.
    
    This class provides utilities for collecting, storing, and analyzing
    feedback from agents about function usage.
    """
    
    def __init__(self, storage_dir: Optional[str] = None):
        """
        Initialize the feedback manager.
        
        Args:
            storage_dir: Optional directory for storing feedback data
        """
        self.registry = get_registry()
        self.storage_dir = storage_dir or os.path.join(os.path.dirname(__file__), "data", "feedback")
        self.feedback_store = {}
        self.feedback_counters = {fb_type.value: 0 for fb_type in FeedbackType}
        self.function_feedback = {}
        self.suggestion_store = {}
        self.optimization_recommendations = {}
        
        # Create storage directory if it doesn't exist
        os.makedirs(self.storage_dir, exist_ok=True)
        
        # Load existing feedback if available
        self._load_feedback()
    
    def _load_feedback(self):
        """Load feedback data from storage."""
        try:
            feedback_file = os.path.join(self.storage_dir, "feedback.json")
            if os.path.exists(feedback_file):
                with open(feedback_file, "r") as f:
                    data = json.load(f)
                    self.feedback_store = data.get("feedback", {})
                    self.feedback_counters = data.get("counters", {})
                    self.function_feedback = data.get("function_feedback", {})
                    self.suggestion_store = data.get("suggestions", {})
                    self.optimization_recommendations = data.get("recommendations", {})
                    
                logger.info(f"Loaded {len(self.feedback_store)} feedback items from storage")
        except Exception as e:
            logger.error(f"Error loading feedback data: {str(e)}")
    
    def _save_feedback(self):
        """Save feedback data to storage."""
        try:
            feedback_file = os.path.join(self.storage_dir, "feedback.json")
            data = {
                "feedback": self.feedback_store,
                "counters": self.feedback_counters,
                "function_feedback": self.function_feedback,
                "suggestions": self.suggestion_store,
                "recommendations": self.optimization_recommendations,
                "last_updated": datetime.now().isoformat()
            }
            
            with open(feedback_file, "w") as f:
                json.dump(data, f, indent=2)
                
            logger.info(f"Saved {len(self.feedback_store)} feedback items to storage")
        except Exception as e:
            logger.error(f"Error saving feedback data: {str(e)}")
    
    def submit_feedback(self, 
                       feedback_type: Union[FeedbackType, str],
                       content: str,
                       function_name: Optional[str] = None,
                       severity: Union[FeedbackSeverity, str] = FeedbackSeverity.MEDIUM,
                       context: Optional[Dict[str, Any]] = None,
                       agent_id: Optional[str] = None) -> Dict[str, Any]:
        """
        Submit feedback about a function or the registry system.
        
        Args:
            feedback_type: Type of feedback
            content: Feedback content
            function_name: Optional name of the function the feedback is about
            severity: Severity level of the feedback
            context: Optional context information
            agent_id: Optional identifier for the agent submitting feedback
            
        Returns:
            Dictionary with feedback submission results
        """
        # Convert enum values to strings if needed
        fb_type = feedback_type.value if isinstance(feedback_type, FeedbackType) else feedback_type
        sev = severity.value if isinstance(severity, FeedbackSeverity) else severity
        
        # Generate a unique ID for the feedback
        feedback_id = f"fb_{int(time.time())}_{self.feedback_counters.get(fb_type, 0) + 1}"
        
        # Create the feedback entry
        feedback_entry = {
            "id": feedback_id,
            "type": fb_type,
            "content": content,
            "function_name": function_name,
            "severity": sev,
            "context": context or {},
            "agent_id": agent_id,
            "timestamp": datetime.now().isoformat(),
            "status": FeedbackStatus.NEW.value
        }
        
        # Store the feedback
        self.feedback_store[feedback_id] = feedback_entry
        
        # Update counters
        self.feedback_counters[fb_type] = self.feedback_counters.get(fb_type, 0) + 1
        
        # Update function-specific feedback
        if function_name:
            if function_name not in self.function_feedback:
                self.function_feedback[function_name] = []
            self.function_feedback[function_name].append(feedback_id)
        
        # Save to storage
        self._save_feedback()
        
        # Process the feedback right away
        self._process_feedback(feedback_entry)
        
        return {
            "success": True,
            "feedback_id": feedback_id,
            "message": "Feedback submitted successfully"
        }
    
    def _process_feedback(self, feedback: Dict[str, Any]):
        """
        Process incoming feedback for potential actions.
        
        Args:
            feedback: The feedback entry to process
        """
        fb_type = feedback["type"]
        
        # Process based on feedback type
        if fb_type == FeedbackType.ERROR.value:
            self._process_error_feedback(feedback)
        elif fb_type == FeedbackType.SUGGESTION.value:
            self._process_suggestion_feedback(feedback)
        elif fb_type == FeedbackType.USAGE_PROBLEM.value:
            self._process_usage_problem_feedback(feedback)
        elif fb_type == FeedbackType.FEATURE_REQUEST.value:
            self._process_feature_request_feedback(feedback)
    
    def _process_error_feedback(self, feedback: Dict[str, Any]):
        """Process error feedback for potential improvements."""
        function_name = feedback.get("function_name")
        if not function_name:
            return
            
        # Create recommendation for error handling improvement
        recommendation_id = f"rec_error_{function_name}_{int(time.time())}"
        
        error_feedback_count = 0
        for fb_id in self.function_feedback.get(function_name, []):
            fb = self.feedback_store.get(fb_id, {})
            if fb.get("type") == FeedbackType.ERROR.value:
                error_feedback_count += 1
        
        if error_feedback_count >= 3:  # Only create recommendation after multiple errors
            self.optimization_recommendations[recommendation_id] = {
                "id": recommendation_id,
                "type": "error_handling",
                "function_name": function_name,
                "description": f"Improve error handling for function '{function_name}'",
                "evidence": [feedback["id"]],
                "severity": FeedbackSeverity.HIGH.value if error_feedback_count >= 5 else FeedbackSeverity.MEDIUM.value,
                "created": datetime.now().isoformat(),
                "status": "open"
            }
    
    def _process_suggestion_feedback(self, feedback: Dict[str, Any]):
        """Process suggestion feedback."""
        function_name = feedback.get("function_name")
        content = feedback.get("content", "")
        
        # Store the suggestion
        suggestion_id = f"sug_{int(time.time())}"
        self.suggestion_store[suggestion_id] = {
            "id": suggestion_id,
            "function_name": function_name,
            "content": content,
            "source_feedback": feedback["id"],
            "created": datetime.now().isoformat(),
            "status": "pending_review"
        }
    
    def _process_usage_problem_feedback(self, feedback: Dict[str, Any]):
        """Process usage problem feedback."""
        function_name = feedback.get("function_name")
        if not function_name:
            return
            
        # Create recommendation for documentation or parameter improvement
        has_similar_rec = False
        for rec_id, rec in self.optimization_recommendations.items():
            if rec.get("function_name") == function_name and rec.get("type") == "usability":
                # Update existing recommendation
                rec["evidence"].append(feedback["id"])
                rec["severity"] = FeedbackSeverity.HIGH.value if len(rec["evidence"]) >= 3 else FeedbackSeverity.MEDIUM.value
                has_similar_rec = True
                break
        
        if not has_similar_rec:
            recommendation_id = f"rec_usability_{function_name}_{int(time.time())}"
            self.optimization_recommendations[recommendation_id] = {
                "id": recommendation_id,
                "type": "usability",
                "function_name": function_name,
                "description": f"Improve usability of function '{function_name}'",
                "evidence": [feedback["id"]],
                "severity": FeedbackSeverity.MEDIUM.value,
                "created": datetime.now().isoformat(),
                "status": "open"
            }
    
    def _process_feature_request_feedback(self, feedback: Dict[str, Any]):
        """Process feature request feedback."""
        # Store as a suggestion
        suggestion_id = f"feat_{int(time.time())}"
        self.suggestion_store[suggestion_id] = {
            "id": suggestion_id,
            "function_name": feedback.get("function_name"),
            "content": feedback.get("content", ""),
            "type": "feature_request",
            "source_feedback": feedback["id"],
            "created": datetime.now().isoformat(),
            "status": "pending_review"
        }
    
    def get_feedback(self, feedback_id: Optional[str] = None) -> Union[Dict[str, Any], List[Dict[str, Any]]]:
        """
        Get specific feedback or all feedback.
        
        Args:
            feedback_id: Optional ID of specific feedback to retrieve
            
        Returns:
            Dictionary or list with feedback data
        """
        if feedback_id:
            return self.feedback_store.get(feedback_id, {})
            
        # Return all feedback, sorted by timestamp
        all_feedback = list(self.feedback_store.values())
        all_feedback.sort(key=lambda x: x.get("timestamp", ""), reverse=True)
        return all_feedback
    
    def get_function_feedback(self, function_name: str) -> List[Dict[str, Any]]:
        """
        Get all feedback for a specific function.
        
        Args:
            function_name: Name of the function
            
        Returns:
            List of feedback entries for the function
        """
        feedback_ids = self.function_feedback.get(function_name, [])
        function_feedback = []
        
        for fb_id in feedback_ids:
            if fb_id in self.feedback_store:
                function_feedback.append(self.feedback_store[fb_id])
        
        # Sort by timestamp
        function_feedback.sort(key=lambda x: x.get("timestamp", ""), reverse=True)
        return function_feedback
    
    def get_feedback_summary(self) -> Dict[str, Any]:
        """
        Get a summary of all feedback.
        
        Returns:
            Dictionary with feedback summary statistics
        """
        # Count feedback by type
        type_counts = {fb_type.value: 0 for fb_type in FeedbackType}
        for feedback in self.feedback_store.values():
            fb_type = feedback.get("type")
            if fb_type in type_counts:
                type_counts[fb_type] += 1
        
        # Count feedback by severity
        severity_counts = {sev.value: 0 for sev in FeedbackSeverity}
        for feedback in self.feedback_store.values():
            sev = feedback.get("severity")
            if sev in severity_counts:
                severity_counts[sev] += 1
        
        # Count feedback by status
        status_counts = {status.value: 0 for status in FeedbackStatus}
        for feedback in self.feedback_store.values():
            status = feedback.get("status")
            if status in status_counts:
                status_counts[status] += 1
        
        # Get functions with most feedback
        function_feedback_counts = {}
        for func_name, fb_ids in self.function_feedback.items():
            function_feedback_counts[func_name] = len(fb_ids)
        
        top_functions = sorted(
            function_feedback_counts.items(),
            key=lambda x: x[1],
            reverse=True
        )[:5]  # Top 5 functions
        
        return {
            "total_feedback": len(self.feedback_store),
            "feedback_by_type": type_counts,
            "feedback_by_severity": severity_counts,
            "feedback_by_status": status_counts,
            "top_functions": top_functions,
            "total_suggestions": len(self.suggestion_store),
            "total_recommendations": len(self.optimization_recommendations)
        }
    
    def update_feedback_status(self, feedback_id: str, 
                              status: Union[FeedbackStatus, str]) -> Dict[str, Any]:
        """
        Update the status of a feedback entry.
        
        Args:
            feedback_id: ID of the feedback to update
            status: New status for the feedback
            
        Returns:
            Dictionary with update results
        """
        if feedback_id not in self.feedback_store:
            return {
                "success": False,
                "message": f"Feedback ID {feedback_id} not found"
            }
        
        # Convert enum to string if needed
        status_value = status.value if isinstance(status, FeedbackStatus) else status
        
        # Update the status
        self.feedback_store[feedback_id]["status"] = status_value
        self.feedback_store[feedback_id]["updated"] = datetime.now().isoformat()
        
        # Save to storage
        self._save_feedback()
        
        return {
            "success": True,
            "message": f"Feedback status updated to {status_value}"
        }
    
    def get_suggestions(self, function_name: Optional[str] = None) -> List[Dict[str, Any]]:
        """
        Get improvement suggestions, optionally filtered by function.
        
        Args:
            function_name: Optional function name to filter suggestions
            
        Returns:
            List of suggestions
        """
        suggestions = list(self.suggestion_store.values())
        
        if function_name:
            suggestions = [s for s in suggestions if s.get("function_name") == function_name]
        
        # Sort by created timestamp
        suggestions.sort(key=lambda x: x.get("created", ""), reverse=True)
        return suggestions
    
    def get_optimization_recommendations(self) -> List[Dict[str, Any]]:
        """
        Get automated optimization recommendations.
        
        Returns:
            List of optimization recommendations
        """
        recommendations = list(self.optimization_recommendations.values())
        
        # Sort by severity and then created timestamp
        recommendations.sort(key=lambda x: (
            0 if x.get("severity") == FeedbackSeverity.CRITICAL.value else
            1 if x.get("severity") == FeedbackSeverity.HIGH.value else
            2 if x.get("severity") == FeedbackSeverity.MEDIUM.value else 3,
            x.get("created", "")
        ), reverse=True)
        
        return recommendations
    
    def analyze_function_usage_patterns(self, function_name: str) -> Dict[str, Any]:
        """
        Analyze usage patterns for a specific function based on feedback.
        
        Args:
            function_name: Name of the function to analyze
            
        Returns:
            Dictionary with analysis results
        """
        if function_name not in self.function_feedback:
            return {
                "function_name": function_name,
                "total_feedback": 0,
                "message": "No feedback available for this function"
            }
        
        feedback_ids = self.function_feedback.get(function_name, [])
        function_feedback = []
        
        for fb_id in feedback_ids:
            if fb_id in self.feedback_store:
                function_feedback.append(self.feedback_store[fb_id])
        
        # Analyze feedback by type
        feedback_by_type = {}
        for feedback in function_feedback:
            fb_type = feedback.get("type")
            if fb_type not in feedback_by_type:
                feedback_by_type[fb_type] = []
            feedback_by_type[fb_type].append(feedback)
        
        # Extract common issues from error and usage problem feedback
        common_issues = []
        if FeedbackType.ERROR.value in feedback_by_type:
            error_feedback = feedback_by_type[FeedbackType.ERROR.value]
            # Here we would analyze error patterns
            # For demo purposes, just count them
            common_issues.append({
                "type": "error",
                "count": len(error_feedback),
                "description": f"{len(error_feedback)} error reports for this function"
            })
        
        if FeedbackType.USAGE_PROBLEM.value in feedback_by_type:
            usage_problems = feedback_by_type[FeedbackType.USAGE_PROBLEM.value]
            # Here we would analyze usage problem patterns
            common_issues.append({
                "type": "usage_problem",
                "count": len(usage_problems),
                "description": f"{len(usage_problems)} usage problems reported for this function"
            })
        
        # Get improvement suggestions
        function_suggestions = self.get_suggestions(function_name)
        
        return {
            "function_name": function_name,
            "total_feedback": len(function_feedback),
            "feedback_by_type": {k: len(v) for k, v in feedback_by_type.items()},
            "common_issues": common_issues,
            "suggestions": function_suggestions[:5],  # Top 5 suggestions
            "analyzed_at": datetime.now().isoformat()
        }
    
    def generate_improvement_recommendations(self) -> List[Dict[str, Any]]:
        """
        Generate improvement recommendations based on feedback analysis.
        
        Returns:
            List of improvement recommendations
        """
        recommendations = []
        
        # Analyze functions with the most feedback
        function_feedback_counts = {}
        for func_name, fb_ids in self.function_feedback.items():
            function_feedback_counts[func_name] = len(fb_ids)
        
        top_functions = sorted(
            function_feedback_counts.items(),
            key=lambda x: x[1],
            reverse=True
        )[:10]  # Top 10 functions
        
        for func_name, count in top_functions:
            # Skip if count is too low
            if count < 3:
                continue
                
            # Analyze the function
            analysis = self.analyze_function_usage_patterns(func_name)
            
            # Generate recommendations based on analysis
            function_recommendations = []
            
            # Check for high error counts
            feedback_by_type = analysis.get("feedback_by_type", {})
            if feedback_by_type.get(FeedbackType.ERROR.value, 0) >= 3:
                function_recommendations.append({
                    "type": "error_handling",
                    "severity": FeedbackSeverity.HIGH.value,
                    "description": f"Improve error handling for function '{func_name}'"
                })
            
            # Check for usage problems
            if feedback_by_type.get(FeedbackType.USAGE_PROBLEM.value, 0) >= 2:
                function_recommendations.append({
                    "type": "usability",
                    "severity": FeedbackSeverity.MEDIUM.value,
                    "description": f"Improve usability of function '{func_name}'"
                })
            
            # Check for documentation issues
            if feedback_by_type.get(FeedbackType.DOCUMENTATION.value, 0) >= 1:
                function_recommendations.append({
                    "type": "documentation",
                    "severity": FeedbackSeverity.MEDIUM.value,
                    "description": f"Improve documentation for function '{func_name}'"
                })
            
            # Add the recommendations
            if function_recommendations:
                recommendations.append({
                    "function_name": func_name,
                    "feedback_count": count,
                    "recommendations": function_recommendations,
                    "generated_at": datetime.now().isoformat()
                })
        
        return recommendations


class FunctionImprover:
    """
    Generates improvement suggestions for functions based on feedback.
    
    This class analyzes feedback and creates specific suggestions for
    improving function implementation, documentation, and usability.
    """
    
    def __init__(self, feedback_manager: Optional[FeedbackManager] = None):
        """
        Initialize the function improver.
        
        Args:
            feedback_manager: Optional existing feedback manager to use
        """
        self.registry = get_registry()
        self.feedback_manager = feedback_manager or FeedbackManager()
    
    def generate_function_suggestions(self, function_name: str) -> Dict[str, Any]:
        """
        Generate improvement suggestions for a specific function.
        
        Args:
            function_name: Name of the function to generate suggestions for
            
        Returns:
            Dictionary with improvement suggestions
        """
        # Get function metadata
        function_metadata = None
        if self.registry:
            function_metadata = self.registry.get_function_metadata(function_name)
            
        if not function_metadata:
            return {
                "function_name": function_name,
                "error": "Function not found in registry"
            }
        
        # Get feedback for the function
        function_feedback = self.feedback_manager.get_function_feedback(function_name)
        
        if not function_feedback:
            return {
                "function_name": function_name,
                "message": "No feedback available for this function",
                "suggestions": []
            }
        
        # Analyze feedback for issues
        error_feedback = [fb for fb in function_feedback if fb.get("type") == FeedbackType.ERROR.value]
        usage_feedback = [fb for fb in function_feedback if fb.get("type") == FeedbackType.USAGE_PROBLEM.value]
        doc_feedback = [fb for fb in function_feedback if fb.get("type") == FeedbackType.DOCUMENTATION.value]
        
        # Generate suggestions
        suggestions = []
        
        # Parameter improvement suggestions
        if usage_feedback:
            parameter_suggestions = self._generate_parameter_suggestions(function_metadata, usage_feedback)
            suggestions.extend(parameter_suggestions)
        
        # Error handling suggestions
        if error_feedback:
            error_suggestions = self._generate_error_handling_suggestions(function_metadata, error_feedback)
            suggestions.extend(error_suggestions)
        
        # Documentation suggestions
        if doc_feedback:
            doc_suggestions = self._generate_documentation_suggestions(function_metadata, doc_feedback)
            suggestions.extend(doc_suggestions)
        
        return {
            "function_name": function_name,
            "total_feedback": len(function_feedback),
            "suggestions": suggestions,
            "generated_at": datetime.now().isoformat()
        }
    
    def _generate_parameter_suggestions(self, 
                                       function_metadata: FunctionMetadata,
                                       usage_feedback: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Generate parameter improvement suggestions."""
        suggestions = []
        
        # For demo purposes, generate simple suggestions
        # In a real implementation, we would analyze parameter usage patterns
        if len(usage_feedback) >= 2:
            suggestions.append({
                "type": "parameter_improvement",
                "title": "Add better parameter validation",
                "description": "Improve parameter validation with more descriptive error messages",
                "implementation_hint": "Add explicit type checking and validation for all parameters"
            })
            
            suggestions.append({
                "type": "parameter_improvement",
                "title": "Add parameter examples",
                "description": "Provide usage examples for parameters in the function documentation",
                "implementation_hint": "Update the function docstring with example parameter values"
            })
        
        return suggestions
    
    def _generate_error_handling_suggestions(self,
                                           function_metadata: FunctionMetadata,
                                           error_feedback: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Generate error handling improvement suggestions."""
        suggestions = []
        
        # For demo purposes, generate simple suggestions
        if len(error_feedback) >= 3:
            suggestions.append({
                "type": "error_handling",
                "title": "Improve error reporting",
                "description": "Enhance error reporting with more specific error types",
                "implementation_hint": "Add custom error types for different failure scenarios"
            })
            
            suggestions.append({
                "type": "error_handling",
                "title": "Add input validation",
                "description": "Validate input parameters more thoroughly to prevent errors",
                "implementation_hint": "Add comprehensive parameter validation before processing"
            })
        
        return suggestions
    
    def _generate_documentation_suggestions(self,
                                          function_metadata: FunctionMetadata,
                                          doc_feedback: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Generate documentation improvement suggestions."""
        suggestions = []
        
        # For demo purposes, generate simple suggestions
        if len(doc_feedback) >= 1:
            suggestions.append({
                "type": "documentation",
                "title": "Improve function description",
                "description": "Enhance the function description with more details",
                "implementation_hint": "Expand the docstring with clearer explanation of function purpose"
            })
            
            suggestions.append({
                "type": "documentation",
                "title": "Add usage examples",
                "description": "Include usage examples in the documentation",
                "implementation_hint": "Add code examples showing common usage patterns"
            })
        
        return suggestions
    
    def generate_system_improvement_recommendations(self) -> Dict[str, Any]:
        """
        Generate system-wide improvement recommendations.
        
        Returns:
            Dictionary with system improvement recommendations
        """
        # Get feedback summary
        feedback_summary = self.feedback_manager.get_feedback_summary()
        
        # Generate recommendations
        recommendations = {
            "high_priority": [],
            "medium_priority": [],
            "low_priority": []
        }
        
        # Check for functions with high error rates
        top_functions = feedback_summary.get("top_functions", [])
        for func_name, count in top_functions:
            function_feedback = self.feedback_manager.get_function_feedback(func_name)
            error_count = len([fb for fb in function_feedback if fb.get("type") == FeedbackType.ERROR.value])
            
            if error_count >= 5:
                recommendations["high_priority"].append({
                    "type": "function_improvement",
                    "function_name": func_name,
                    "description": f"Fix critical error issues in '{func_name}' (reported {error_count} times)",
                    "action": "review_function"
                })
            elif error_count >= 3:
                recommendations["medium_priority"].append({
                    "type": "function_improvement",
                    "function_name": func_name,
                    "description": f"Address error issues in '{func_name}' (reported {error_count} times)",
                    "action": "review_function"
                })
        
        # Check for documentation issues
        doc_feedback_count = feedback_summary.get("feedback_by_type", {}).get(FeedbackType.DOCUMENTATION.value, 0)
        if doc_feedback_count >= 5:
            recommendations["medium_priority"].append({
                "type": "system_improvement",
                "description": "Improve system-wide documentation (multiple issues reported)",
                "action": "review_documentation"
            })
        
        # Check for feature requests
        feature_requests = feedback_summary.get("feedback_by_type", {}).get(FeedbackType.FEATURE_REQUEST.value, 0)
        if feature_requests >= 3:
            recommendations["low_priority"].append({
                "type": "feature_addition",
                "description": f"Review and prioritize feature requests ({feature_requests} pending)",
                "action": "review_feature_requests"
            })
        
        return {
            "recommendations": recommendations,
            "summary": feedback_summary,
            "generated_at": datetime.now().isoformat()
        }


# Tool functions for feedback mechanism

async def submit_feedback(feedback_type: str,
                     content: str,
                     function_name: Optional[str] = None,
                     severity: str = "medium",
                     context: Optional[Dict[str, Any]] = None,
                     agent_id: Optional[str] = None) -> FunctionResult:
    """
    Submit feedback about a function or the registry system.
    
    Args:
        feedback_type: Type of feedback (error, suggestion, etc.)
        content: Content of the feedback
        function_name: Name of the function the feedback is about (optional)
        severity: Severity of the feedback (low, medium, high, critical)
        context: Additional context information (optional)
        agent_id: ID of the agent submitting the feedback (optional)
        
    Returns:
        FunctionResult with the feedback ID and status
    """
    try:
        feedback_manager = FeedbackManager()
        
        result = feedback_manager.submit_feedback(
            feedback_type=feedback_type,
            content=content,
            function_name=function_name,
            severity=severity,
            context=context,
            agent_id=agent_id
        )
        
        return FunctionResult(
            data=result,
            status="SUCCESS",
            message="Feedback submitted successfully",
            error_code=None,
            error_details=None
        )
    except Exception as e:
        logger.error(f"Error submitting feedback: {str(e)}")
        return FunctionResult(
            data=None,
            status="ERROR",
            message=f"Error submitting feedback: {str(e)}",
            error_code="FEEDBACK_SUBMISSION_ERROR",
            error_details={"details": traceback.format_exc()}
        )

async def get_function_suggestions(function_name: str) -> FunctionResult:
    """
    Get improvement suggestions for a specific function.
    
    Args:
        function_name: Name of the function to get suggestions for
        
    Returns:
        FunctionResult with the suggestions
    """
    try:
        feedback_manager = FeedbackManager()
        improver = FunctionImprover(feedback_manager)
        
        suggestions = improver.generate_function_suggestions(function_name)
        
        return FunctionResult(
            data=suggestions,
            status="SUCCESS",
            message="Function suggestions generated successfully",
            error_code=None,
            error_details=None
        )
    except Exception as e:
        logger.error(f"Error generating function suggestions: {str(e)}")
        return FunctionResult(
            data=None,
            status="ERROR",
            message=f"Error generating function suggestions: {str(e)}",
            error_code="SUGGESTION_GENERATION_ERROR",
            error_details={"details": traceback.format_exc()}
        )

async def get_optimization_recommendations() -> FunctionResult:
    """
    Get system-wide optimization recommendations.
    
    Returns:
        FunctionResult with the recommendations
    """
    try:
        feedback_manager = FeedbackManager()
        improver = FunctionImprover(feedback_manager)
        
        recommendations = improver.generate_system_improvement_recommendations()
        
        return FunctionResult(
            data=recommendations,
            status="SUCCESS",
            message="Optimization recommendations generated successfully",
            error_code=None,
            error_details=None
        )
    except Exception as e:
        logger.error(f"Error generating optimization recommendations: {str(e)}")
        return FunctionResult(
            data=None,
            status="ERROR",
            message=f"Error generating optimization recommendations: {str(e)}",
            error_code="RECOMMENDATION_GENERATION_ERROR",
            error_details={"details": traceback.format_exc()}
        )

async def get_feedback_summary() -> FunctionResult:
    """
    Get a summary of all feedback.
    
    Returns:
        FunctionResult with the summary information
    """
    try:
        feedback_manager = FeedbackManager()
        summary = feedback_manager.get_feedback_summary()
        
        return FunctionResult(
            data=summary,
            status="SUCCESS",
            message="Feedback summary generated successfully",
            error_code=None,
            error_details=None
        )
    except Exception as e:
        logger.error(f"Error getting feedback summary: {str(e)}")
        return FunctionResult(
            data=None,
            status="ERROR",
            message=f"Error getting feedback summary: {str(e)}",
            error_code="FEEDBACK_SUMMARY_ERROR",
            error_details={"details": traceback.format_exc()}
        )

def register_feedback_mechanism_tools():
    """Register feedback mechanism tools with the function registry."""
    try:
        # Attempt to get registry
        try:
            from src.registry.registry_manager import get_registry
            registry = get_registry()
        except ImportError:
            logger.error("Could not import registry_manager")
            registry = None
            
        if not registry:
            logger.info("Registry not available, feedback mechanism tools not registered")
            return
            
        # Try different registration methods
        registered = False
        
        # Method 1: Try registry_tools if available
        try:
            # Try to import registry_tools using importlib for safer import
            import importlib
            try:
                registry_tools_module = importlib.import_module('src.registry.registry_tools')
                # Check if it has register_function attribute
                if hasattr(registry_tools_module, 'register_function'):
                    register_fn = getattr(registry_tools_module, 'register_function')
                    
                    # Register submit_feedback
                    register_fn(
                        func=submit_feedback,
                        name="submit_feedback",
                        description="Submit feedback about a function or the registry system",
                        parameters={
                            "feedback_type": {"type": "str", "description": "Type of feedback", "required": True},
                            "content": {"type": "str", "description": "Feedback content", "required": True},
                            "function_name": {"type": "str", "description": "Function name", "required": False},
                            "severity": {"type": "str", "description": "Severity level", "required": False},
                            "context": {"type": "object", "description": "Additional context", "required": False},
                            "agent_id": {"type": "str", "description": "Agent ID", "required": False}
                        },
                        namespace="feedback"
                    )
                    
                    # Register other functions
                    register_fn(
                        func=get_function_suggestions,
                        name="get_function_suggestions",
                        description="Get function improvement suggestions",
                        parameters={"function_name": {"type": "str", "description": "Function name", "required": True}},
                        namespace="feedback"
                    )
                    
                    register_fn(
                        func=get_optimization_recommendations,
                        name="get_optimization_recommendations",
                        description="Get optimization recommendations",
                        parameters={},
                        namespace="feedback"
                    )
                    
                    register_fn(
                        func=get_feedback_summary,
                        name="get_feedback_summary",
                        description="Get feedback summary",
                        parameters={},
                        namespace="feedback"
                    )
                    
                    registered = True
                    logger.info("Feedback mechanism tools registered via registry_tools")
            except (ImportError, AttributeError):
                pass
        except Exception as e:
            logger.info(f"Could not register using registry_tools: {str(e)}")
        
        # Method 2: Try registry_manager if available
        if not registered:
            try:
                # Import using importlib
                import importlib
                try:
                    registry_manager_module = importlib.import_module('src.registry.registry_manager')
                    
                    # Check if register_function exists
                    if hasattr(registry_manager_module, 'register_function'):
                        reg_func = getattr(registry_manager_module, 'register_function')
                        
                        # Register all tools using full parameter lists to avoid namespace duplication
                        reg_func(
                            func=submit_feedback,
                            name="submit_feedback",
                            description="Submit feedback about a function or the registry system",
                            parameters={
                                "feedback_type": {"type": "str", "description": "Type of feedback", "required": True},
                                "content": {"type": "str", "description": "Feedback content", "required": True},
                                "function_name": {"type": "str", "description": "Function name", "required": False},
                                "severity": {"type": "str", "description": "Severity level", "required": False},
                                "context": {"type": "object", "description": "Additional context", "required": False},
                                "agent_id": {"type": "str", "description": "Agent ID", "required": False}
                            },
                            namespace="feedback"
                        )
                        
                        reg_func(
                            func=get_function_suggestions,
                            name="get_function_suggestions",
                            description="Get function improvement suggestions",
                            parameters={"function_name": {"type": "str", "description": "Function name", "required": True}},
                            namespace="feedback"
                        )
                        
                        reg_func(
                            func=get_optimization_recommendations,
                            name="get_optimization_recommendations",
                            description="Get optimization recommendations",
                            parameters={},
                            namespace="feedback"
                        )
                        
                        reg_func(
                            func=get_feedback_summary,
                            name="get_feedback_summary",
                            description="Get feedback summary",
                            parameters={},
                            namespace="feedback"
                        )
                        
                        registered = True
                        logger.info("Feedback mechanism tools registered via registry_manager")
                except (ImportError, AttributeError):
                    pass
            except Exception as e:
                logger.info(f"Could not register using registry_manager: {str(e)}")
        
        # Method 3: Try basic registry methods as a last resort
        if not registered:
            # Just check if registry has any register method we can try
            register_methods = [attr for attr in dir(registry) if "register" in attr.lower()]
            
            if register_methods:
                try:
                    # Choose the first available register method
                    register_method = getattr(registry, register_methods[0])
                    
                    # Try with minimal arguments
                    register_method(submit_feedback, "submit_feedback")
                    register_method(get_function_suggestions, "get_function_suggestions")
                    register_method(get_optimization_recommendations, "get_optimization_recommendations")
                    register_method(get_feedback_summary, "get_feedback_summary")
                    
                    registered = True
                    logger.info(f"Feedback mechanism tools registered via {register_methods[0]}")
                except Exception as e:
                    logger.error(f"Error registering with {register_methods[0]}: {str(e)}")
        
        if registered:
            logger.info("Feedback mechanism tools registered successfully")
        else:
            logger.error("Failed to register feedback mechanism tools - no registration method available")
            
    except Exception as e:
        logger.error(f"Error registering feedback mechanism tools: {str(e)}") 