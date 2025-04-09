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

from src.registry.registry_manager import get_registry, register_function, register_tool
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
                       tool_name: Optional[str] = None,
                       severity: Union[FeedbackSeverity, str] = FeedbackSeverity.MEDIUM,
                       context: Optional[Dict[str, Any]] = None,
                       agent_id: Optional[str] = None) -> Dict[str, Any]:
        """
        Submit feedback about the function registry system.
        
        Args:
            feedback_type: Type of feedback (general, error, suggestion, etc.)
            content: The feedback content
            tool_name: Optional name of the specific tool the feedback is about
            severity: Severity level of the feedback
            context: Optional context information (e.g., parameters used)
            agent_id: Optional identifier for the agent submitting feedback
            
        Returns:
            Dictionary with the submitted feedback information including ID
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
            "tool_name": tool_name,
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
        if tool_name:
            if tool_name not in self.function_feedback:
                self.function_feedback[tool_name] = []
            self.function_feedback[tool_name].append(feedback_id)
        
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
        tool_name = feedback.get("tool_name")
        if not tool_name:
            return
            
        # Create recommendation for error handling improvement
        recommendation_id = f"rec_error_{tool_name}_{int(time.time())}"
        
        error_feedback_count = 0
        for fb_id in self.function_feedback.get(tool_name, []):
            fb = self.feedback_store.get(fb_id, {})
            if fb.get("type") == FeedbackType.ERROR.value:
                error_feedback_count += 1
        
        if error_feedback_count >= 3:  # Only create recommendation after multiple errors
            self.optimization_recommendations[recommendation_id] = {
                "id": recommendation_id,
                "type": "error_handling",
                "tool_name": tool_name,
                "description": f"Improve error handling for tool '{tool_name}'",
                "evidence": [feedback["id"]],
                "severity": FeedbackSeverity.HIGH.value if error_feedback_count >= 5 else FeedbackSeverity.MEDIUM.value,
                "created": datetime.now().isoformat(),
                "status": "open"
            }
    
    def _process_suggestion_feedback(self, feedback: Dict[str, Any]):
        """Process suggestion feedback."""
        tool_name = feedback.get("tool_name")
        content = feedback.get("content", "")
        
        # Store the suggestion
        suggestion_id = f"sug_{int(time.time())}"
        self.suggestion_store[suggestion_id] = {
            "id": suggestion_id,
            "tool_name": tool_name,
            "content": content,
            "source_feedback": feedback["id"],
            "created": datetime.now().isoformat(),
            "status": "pending_review"
        }
    
    def _process_usage_problem_feedback(self, feedback: Dict[str, Any]):
        """Process usage problem feedback."""
        tool_name = feedback.get("tool_name")
        if not tool_name:
            return
            
        # Create recommendation for documentation or parameter improvement
        has_similar_rec = False
        for rec_id, rec in self.optimization_recommendations.items():
            if rec.get("tool_name") == tool_name and rec.get("type") == "usability":
                # Update existing recommendation
                rec["evidence"].append(feedback["id"])
                rec["severity"] = FeedbackSeverity.HIGH.value if len(rec["evidence"]) >= 3 else FeedbackSeverity.MEDIUM.value
                has_similar_rec = True
                break
        
        if not has_similar_rec:
            recommendation_id = f"rec_usability_{tool_name}_{int(time.time())}"
            self.optimization_recommendations[recommendation_id] = {
                "id": recommendation_id,
                "type": "usability",
                "tool_name": tool_name,
                "description": f"Improve usability of tool '{tool_name}'",
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
            "tool_name": feedback.get("tool_name"),
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
    
    def get_tool_feedback(self, tool_name: str) -> List[Dict[str, Any]]:
        """
        Get all feedback for a specific tool.
        
        Args:
            tool_name: Name of the tool to get feedback for
            
        Returns:
            List of feedback items for the specified tool
        """
        feedback_ids = self.function_feedback.get(tool_name, [])
        tool_feedback = []
        
        for fb_id in feedback_ids:
            if fb_id in self.feedback_store:
                tool_feedback.append(self.feedback_store[fb_id])
        
        # Sort by timestamp
        tool_feedback.sort(key=lambda x: x.get("timestamp", ""), reverse=True)
        return tool_feedback
    
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
        
        # Get tools with most feedback
        tool_feedback_counts = {}
        for func_name, fb_ids in self.function_feedback.items():
            tool_feedback_counts[func_name] = len(fb_ids)
        
        top_tools = sorted(
            tool_feedback_counts.items(),
            key=lambda x: x[1],
            reverse=True
        )[:5]  # Top 5 tools
        
        return {
            "total_feedback": len(self.feedback_store),
            "feedback_by_type": type_counts,
            "feedback_by_severity": severity_counts,
            "feedback_by_status": status_counts,
            "top_tools": top_tools,
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
    
    def get_suggestions(self, tool_name: Optional[str] = None) -> List[Dict[str, Any]]:
        """
        Get improvement suggestions, optionally filtered by tool.
        
        Args:
            tool_name: Optional tool name to filter suggestions
            
        Returns:
            List of suggestions
        """
        suggestions = list(self.suggestion_store.values())
        
        if tool_name:
            suggestions = [s for s in suggestions if s.get("tool_name") == tool_name]
        
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
    
    def analyze_tool_usage_patterns(self, tool_name: str) -> Dict[str, Any]:
        """
        Analyze usage patterns for a specific tool.
        
        Args:
            tool_name: Name of the tool to analyze
            
        Returns:
            Dictionary with usage pattern information
        """
        if tool_name not in self.function_feedback:
            return {
                "tool_name": tool_name,
                "total_feedback": 0,
                "message": "No feedback available for this tool"
            }
        
        feedback_ids = self.function_feedback.get(tool_name, [])
        tool_feedback = []
        
        for fb_id in feedback_ids:
            if fb_id in self.feedback_store:
                tool_feedback.append(self.feedback_store[fb_id])
        
        # Analyze feedback by type
        feedback_by_type = {}
        for feedback in tool_feedback:
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
                "description": f"{len(error_feedback)} error reports for this tool"
            })
        
        if FeedbackType.USAGE_PROBLEM.value in feedback_by_type:
            usage_problems = feedback_by_type[FeedbackType.USAGE_PROBLEM.value]
            # Here we would analyze usage problem patterns
            common_issues.append({
                "type": "usage_problem",
                "count": len(usage_problems),
                "description": f"{len(usage_problems)} usage problems reported for this tool"
            })
        
        # Get improvement suggestions
        tool_suggestions = self.get_suggestions(tool_name)
        
        return {
            "tool_name": tool_name,
            "total_feedback": len(tool_feedback),
            "feedback_by_type": {k: len(v) for k, v in feedback_by_type.items()},
            "common_issues": common_issues,
            "suggestions": tool_suggestions[:5],  # Top 5 suggestions
            "analyzed_at": datetime.now().isoformat()
        }
    
    def generate_improvement_recommendations(self) -> List[Dict[str, Any]]:
        """
        Generate improvement recommendations based on feedback analysis.
        
        Returns:
            List of improvement recommendations
        """
        recommendations = []
        
        # Analyze tools with the most feedback
        tool_feedback_counts = {}
        for func_name, fb_ids in self.function_feedback.items():
            tool_feedback_counts[func_name] = len(fb_ids)
        
        top_tools = sorted(
            tool_feedback_counts.items(),
            key=lambda x: x[1],
            reverse=True
        )[:10]  # Top 10 tools
        
        for func_name, count in top_tools:
            # Skip if count is too low
            if count < 3:
                continue
                
            # Analyze the tool
            analysis = self.analyze_tool_usage_patterns(func_name)
            
            # Generate recommendations based on analysis
            tool_recommendations = []
            
            # Check for high error counts
            feedback_by_type = analysis.get("feedback_by_type", {})
            if feedback_by_type.get(FeedbackType.ERROR.value, 0) >= 3:
                tool_recommendations.append({
                    "type": "error_handling",
                    "severity": FeedbackSeverity.HIGH.value,
                    "description": f"Improve error handling for tool '{func_name}'"
                })
            
            # Check for usage problems
            if feedback_by_type.get(FeedbackType.USAGE_PROBLEM.value, 0) >= 2:
                tool_recommendations.append({
                    "type": "usability",
                    "severity": FeedbackSeverity.MEDIUM.value,
                    "description": f"Improve usability of tool '{func_name}'"
                })
            
            # Check for documentation issues
            if feedback_by_type.get(FeedbackType.DOCUMENTATION.value, 0) >= 1:
                tool_recommendations.append({
                    "type": "documentation",
                    "severity": FeedbackSeverity.MEDIUM.value,
                    "description": f"Improve documentation for tool '{func_name}'"
                })
            
            # Add the recommendations
            if tool_recommendations:
                recommendations.append({
                    "tool_name": func_name,
                    "feedback_count": count,
                    "recommendations": tool_recommendations,
                    "generated_at": datetime.now().isoformat()
                })
        
        return recommendations


class ToolImprover:
    """
    Tool improvement suggestion generator.
    
    Uses feedback and usage data to generate improvement suggestions for tools
    and the overall system.
    """
    
    def __init__(self, feedback_manager: Optional[FeedbackManager] = None):
        """Initialize the tool improver."""
        self.feedback_manager = feedback_manager or FeedbackManager()
        self.registry = get_registry()
    
    def generate_tool_suggestions(self, tool_name: str) -> Dict[str, Any]:
        """
        Generate improvement suggestions for a specific tool.
        
        Args:
            tool_name: Name of the tool to generate suggestions for
            
        Returns:
            Dictionary with various improvement suggestions
        """
        # Get the tool metadata
        tool_metadata = self.registry.get_tool_metadata(tool_name)
        if not tool_metadata:
            return {
                "error": f"Tool '{tool_name}' not found",
                "suggestions": []
            }
            
        # Get all feedback for this tool
        tool_feedback = self.feedback_manager.get_tool_feedback(tool_name)
        
        if not tool_feedback:
            return {
                "tool_name": tool_name,
                "message": "No feedback available for this tool",
                "suggestions": []
            }
        
        # Analyze feedback for issues
        error_feedback = [fb for fb in tool_feedback if fb.get("type") == FeedbackType.ERROR.value]
        usage_feedback = [fb for fb in tool_feedback if fb.get("type") == FeedbackType.USAGE_PROBLEM.value]
        doc_feedback = [fb for fb in tool_feedback if fb.get("type") == FeedbackType.DOCUMENTATION.value]
        
        # Generate suggestions
        suggestions = []
        
        # Parameter improvement suggestions
        if usage_feedback:
            parameter_suggestions = self._generate_parameter_suggestions(tool_metadata, usage_feedback)
            suggestions.extend(parameter_suggestions)
        
        # Error handling suggestions
        if error_feedback:
            error_suggestions = self._generate_error_handling_suggestions(tool_metadata, error_feedback)
            suggestions.extend(error_suggestions)
        
        # Documentation suggestions
        if doc_feedback:
            doc_suggestions = self._generate_documentation_suggestions(tool_metadata, doc_feedback)
            suggestions.extend(doc_suggestions)
        
        return {
            "tool_name": tool_name,
            "total_feedback": len(tool_feedback),
            "suggestions": suggestions,
            "generated_at": datetime.now().isoformat()
        }
    
    def _generate_parameter_suggestions(self, 
                                       tool_metadata: FunctionMetadata,
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
                "description": "Provide usage examples for parameters in the tool documentation",
                "implementation_hint": "Update the tool docstring with example parameter values"
            })
        
        return suggestions
    
    def _generate_error_handling_suggestions(self,
                                           tool_metadata: FunctionMetadata,
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
                                          tool_metadata: FunctionMetadata,
                                          doc_feedback: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Generate documentation improvement suggestions."""
        suggestions = []
        
        # For demo purposes, generate simple suggestions
        if len(doc_feedback) >= 1:
            suggestions.append({
                "type": "documentation",
                "title": "Improve tool description",
                "description": "Enhance the tool description with more details",
                "implementation_hint": "Expand the docstring with clearer explanation of tool purpose"
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
        
        # Check for tools with high error rates
        top_tools = feedback_summary.get("top_tools", [])
        for func_name, count in top_tools:
            tool_feedback = self.feedback_manager.get_tool_feedback(func_name)
            error_count = len([fb for fb in tool_feedback if fb.get("type") == FeedbackType.ERROR.value])
            
            if error_count >= 5:
                recommendations["high_priority"].append({
                    "type": "tool_improvement",
                    "tool_name": func_name,
                    "description": f"Fix critical error issues in '{func_name}' (reported {error_count} times)",
                    "action": "review_tool"
                })
            elif error_count >= 3:
                recommendations["medium_priority"].append({
                    "type": "tool_improvement",
                    "tool_name": func_name,
                    "description": f"Address error issues in '{func_name}' (reported {error_count} times)",
                    "action": "review_tool"
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


# Main feedback functions with direct registration decorators

@register_tool("feedback", "submit_feedback")
async def submit_feedback(feedback_type: str,
                     content: str,
                     tool_name: Optional[str] = None,
                     severity: str = "medium",
                     context: Optional[Dict[str, Any]] = None,
                     agent_id: Optional[str] = None) -> FunctionResult:
    """
    Submit feedback about a tool or the registry system.
    
    Args:
        feedback_type: Type of feedback (general, error, suggestion, rating, etc.)
        content: Parameter content
        tool_name: Parameter tool_name
        severity: Parameter severity
        context: Parameter context
        agent_id: Parameter agent_id
        
    Returns:
        FunctionResult with the submitted feedback
    """
    try:
        feedback_manager = FeedbackManager()
        
        result = feedback_manager.submit_feedback(
            feedback_type=feedback_type,
            content=content,
            tool_name=tool_name,
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

@register_tool("feedback", "get_tool_suggestions")
async def get_tool_suggestions(tool_name: str) -> FunctionResult:
    """
    Get tool improvement suggestions.
    
    Args:
        tool_name: Name of the tool to get suggestions for
        
    Returns:
        Function result with tool suggestions
    """
    try:
        feedback_manager = FeedbackManager()
        improver = ToolImprover(feedback_manager)
        
        suggestions = improver.generate_tool_suggestions(tool_name)
        
        return FunctionResult(
            data=suggestions,
            status="SUCCESS",
            message="Tool suggestions generated successfully",
            error_code=None,
            error_details=None
        )
    except Exception as e:
        logger.error(f"Error generating tool suggestions: {str(e)}")
        return FunctionResult(
            data=None,
            status="ERROR",
            message=f"Error generating tool suggestions: {str(e)}",
            error_code="SUGGESTION_GENERATION_ERROR",
            error_details={"details": traceback.format_exc()}
        )

@register_tool("feedback", "get_optimization_recommendations")
async def get_optimization_recommendations() -> FunctionResult:
    """
    Get optimization recommendations for the Tool Registry system.
    
    Returns:
        Function result with optimization recommendations
    """
    try:
        feedback_manager = FeedbackManager()
        improver = ToolImprover(feedback_manager)
        
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

@register_tool("feedback", "get_feedback_summary")
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
    # The functions are now directly registered via decorators
    # This function is kept for backward compatibility
    logger.info("Feedback mechanism tools already registered via decorators")
    return

# The rest of the old register_feedback_mechanism_tools function has been removed
# as it's no longer needed due to direct registration with decorators 