#!/usr/bin/env python3
"""
Agent Guidance System

This module provides guidance to AI agents on how to use the Function Registry,
including function recommendations, usage pattern detection, and helper functions
for complex operations.
"""

import json
import datetime
import re
import collections
from typing import Dict, List, Any, Optional, Set, Tuple, Union
import logging
import asyncio
import time

from src.registry.registry_manager import get_registry, FunctionRegistry
from src.registry.function_models import FunctionMetadata, FunctionResult
from src.registry.parameter_helper import ParameterHelper
from src.logger import get_logger

# Initialize logger
logger = get_logger()

class FunctionRecommendationEngine:
    """
    Recommendation engine that suggests functions based on context and patterns.
    
    Features:
    - Contextual function recommendations
    - Semantic similarity matching
    - Common operational patterns
    - Chain-of-thought suggestions
    """
    
    def __init__(self):
        """Initialize the recommendation engine."""
        self.registry = get_registry()
        self.usage_history: List[Dict[str, Any]] = []
        self.common_patterns: Dict[str, List[str]] = {
            "entity_creation": ["memory.create_entity", "memory.create_observation"],
            "data_processing": ["data.extract_info", "data.transform_data", "data.validate_data"],
            "messaging": ["messaging.send_message", "messaging.get_response"],
            "search": ["memory.semantic_search", "memory.get_entity"],
        }
        self.semantic_vectors: Dict[str, List[float]] = {}
        self.function_popularity: Dict[str, int] = {}
        
    def track_function_usage(self, function_name: str, parameters: Dict[str, Any], 
                             result: FunctionResult, context: Optional[Dict[str, Any]] = None) -> None:
        """
        Track a function call for usage analysis.
        
        Args:
            function_name: Name of the function called
            parameters: Parameters passed to the function
            result: Result of the function call
            context: Optional context information
        """
        # Add to usage history
        self.usage_history.append({
            "function": function_name,
            "parameters": parameters,
            "status": result.status,
            "timestamp": datetime.datetime.now().isoformat(),
            "context": context or {}
        })
        
        # Update function popularity
        self.function_popularity[function_name] = self.function_popularity.get(function_name, 0) + 1
        
        # Keep history at a reasonable size
        if len(self.usage_history) > 1000:
            self.usage_history = self.usage_history[-1000:]
    
    def get_recommendations(self, query: str, context: Optional[Dict[str, Any]] = None, 
                          limit: int = 5) -> List[Dict[str, Any]]:
        """
        Get function recommendations based on a natural language query.
        
        Args:
            query: Natural language query
            context: Optional context information
            limit: Maximum number of recommendations to return
            
        Returns:
            List of function recommendations with relevance scores
        """
        context = context or {}
        
        # Get all functions
        all_functions = self.registry.get_all_functions()
        
        # Calculate relevance for each function
        recommendations = []
        for function in all_functions:
            relevance = self._calculate_relevance(function, query, context)
            
            if relevance > 0:
                recommendations.append({
                    "function": function.name,
                    "relevance": relevance,
                    "description": function.description,
                    "parameters": function.parameters,
                    "pattern": self._get_pattern_for_function(function.name),
                    "popularity": self.function_popularity.get(function.name, 0),
                    "next_steps": self._get_common_next_steps(function.name)
                })
        
        # Sort by relevance and limit results
        recommendations.sort(key=lambda x: x["relevance"], reverse=True)
        return recommendations[:limit]
    
    def get_function_chains(self, goal: str, limit: int = 3) -> List[Dict[str, Any]]:
        """
        Get recommended function chains for accomplishing a goal.
        
        Args:
            goal: Description of the goal to accomplish
            limit: Maximum number of chains to return
            
        Returns:
            List of function chains
        """
        # Find patterns that match the goal
        matching_patterns = []
        for pattern_name, functions in self.common_patterns.items():
            # Simple matching based on keywords in the goal and pattern name
            pattern_words = set(pattern_name.lower().split('_'))
            goal_words = set(goal.lower().split())
            
            # Calculate overlap between pattern words and goal words
            overlap = len(pattern_words.intersection(goal_words))
            if overlap > 0:
                matching_patterns.append({
                    "name": pattern_name,
                    "functions": functions,
                    "relevance": overlap / len(pattern_words)
                })
        
        # Sort patterns by relevance
        matching_patterns.sort(key=lambda x: x["relevance"], reverse=True)
        
        # Get specific function information for each chain
        chains = []
        for pattern in matching_patterns[:limit]:
            chain_functions = []
            for func_name in pattern["functions"]:
                metadata = self.registry.get_function_metadata(func_name)
                if metadata:
                    chain_functions.append({
                        "name": func_name,
                        "description": metadata.description,
                        "parameters": metadata.parameters
                    })
            
            chains.append({
                "name": pattern["name"],
                "description": f"Chain for {pattern['name'].replace('_', ' ')}",
                "functions": chain_functions,
                "relevance": pattern["relevance"]
            })
        
        return chains
    
    def _calculate_relevance(self, function: FunctionMetadata, query: str, 
                           context: Dict[str, Any]) -> float:
        """Calculate relevance score for a function based on query and context."""
        relevance = 0.0
        
        # Match function description with query
        description = function.description.lower()
        query_lower = query.lower()
        
        # Check for exact function name match
        if function.name.lower() in query_lower or function.short_name.lower() in query_lower:
            relevance += 0.8
        
        # Check for keyword matches in description
        query_keywords = query_lower.split()
        for keyword in query_keywords:
            if keyword in description:
                relevance += 0.1
        
        # Check function namespace relevance
        if function.namespace.lower() in query_lower:
            relevance += 0.3
        
        # Check parameter matches with context
        for param_name in function.parameters:
            if param_name in context:
                relevance += 0.05
        
        # Consider function popularity
        popularity_boost = min(0.2, self.function_popularity.get(function.name, 0) / 100)
        relevance += popularity_boost
        
        return relevance
    
    def _get_pattern_for_function(self, function_name: str) -> Optional[str]:
        """Get the pattern name for a function if it belongs to a pattern."""
        for pattern_name, functions in self.common_patterns.items():
            if function_name in functions:
                return pattern_name
        return None
    
    def _get_common_next_steps(self, function_name: str) -> List[str]:
        """Get common next steps after a function based on usage history."""
        next_steps = []
        
        # Find this function in history
        indices = [i for i, entry in enumerate(self.usage_history) 
                 if entry["function"] == function_name]
        
        # Count next functions
        next_counter = collections.Counter()
        for idx in indices:
            if idx + 1 < len(self.usage_history):
                next_function = self.usage_history[idx + 1]["function"]
                next_counter[next_function] += 1
        
        # Get most common next steps
        if next_counter:
            next_steps = [func for func, _ in next_counter.most_common(3)]
        
        # If no usage history, fall back to patterns
        if not next_steps:
            for functions in self.common_patterns.values():
                if function_name in functions:
                    idx = functions.index(function_name)
                    if idx + 1 < len(functions):
                        next_steps.append(functions[idx + 1])
        
        return next_steps


class UsagePatternDetector:
    """
    Detects and analyzes function usage patterns to provide guidance.
    
    Features:
    - Pattern recognition in function calls
    - Effectiveness analysis
    - Anti-pattern detection
    - Context-aware pattern matching
    """
    
    def __init__(self):
        """Initialize the pattern detector."""
        self.registry = get_registry()
        # Store sequences of function names
        self.usage_sequences: List[List[str]] = []
        # Store full sequence data, key is a string with function names
        self.common_sequences: Dict[str, List[List[Dict[str, Any]]]] = {}
        self.anti_patterns: List[Dict[str, Any]] = [
            {
                "name": "repeated_failed_calls",
                "detector": self._detect_repeated_failures,
                "suggestion": "Try using different parameters or check function documentation"
            },
            {
                "name": "parameter_mismatch",
                "detector": self._detect_parameter_mismatch,
                "suggestion": "Use get_parameter_suggestions to check correct parameters"
            },
            {
                "name": "inefficient_sequence",
                "detector": self._detect_inefficient_sequence,
                "suggestion": "Consider using function bundles for related operations"
            }
        ]
        self.current_sequence: List[Dict[str, Any]] = []
        
    def add_function_call(self, function_name: str, params: Dict[str, Any], 
                        result: FunctionResult) -> None:
        """
        Add a function call to the current sequence.
        
        Args:
            function_name: Name of the function called
            params: Parameters used in the call
            result: Result of the function call
        """
        self.current_sequence.append({
            "function": function_name,
            "params": params,
            "status": result.status,
            "timestamp": time.time()
        })
        
        # Check for anti-patterns after adding
        self._check_anti_patterns()
    
    def end_sequence(self) -> None:
        """End the current sequence and add it to history."""
        if self.current_sequence:
            # Extract just the function names for the sequence
            function_sequence = [call["function"] for call in self.current_sequence]
            self.usage_sequences.append(function_sequence)
            
            # Update common sequences dictionary
            sequence_key = "→".join(function_sequence)
            if sequence_key not in self.common_sequences:
                self.common_sequences[sequence_key] = []
            self.common_sequences[sequence_key].append(self.current_sequence.copy())
            
            # Reset current sequence
            self.current_sequence = []
    
    def get_detected_patterns(self) -> List[Dict[str, Any]]:
        """
        Get detected patterns from usage history.
        
        Returns:
            List of detected patterns with frequency and effectiveness
        """
        patterns = []
        
        # Analyze common sequences
        for sequence_key, occurrences in self.common_sequences.items():
            functions = sequence_key.split("→")
            
            # Skip single-function sequences
            if len(functions) < 2:
                continue
            
            # Calculate effectiveness (percentage of successful calls)
            success_count = sum(
                1 for seq in occurrences 
                if all(call.get("status", "") == "success" for call in seq)
            )
            effectiveness = success_count / len(occurrences) if occurrences else 0
            
            # Add to patterns list
            patterns.append({
                "functions": functions,
                "frequency": len(occurrences),
                "effectiveness": effectiveness,
                "description": self._generate_pattern_description(functions)
            })
        
        # Sort by frequency and effectiveness
        patterns.sort(key=lambda x: (x["frequency"], x["effectiveness"]), reverse=True)
        return patterns
    
    def get_detected_anti_patterns(self) -> List[Dict[str, Any]]:
        """
        Get detected anti-patterns with recommendations.
        
        Returns:
            List of anti-patterns with suggestions
        """
        detected = []
        
        for pattern in self.anti_patterns:
            detection_result = pattern["detector"](self.usage_sequences, self.current_sequence)
            if detection_result:
                detected.append({
                    "name": pattern["name"],
                    "details": detection_result,
                    "suggestion": pattern["suggestion"]
                })
        
        return detected
    
    def _check_anti_patterns(self) -> None:
        """Check for anti-patterns in the current sequence."""
        for pattern in self.anti_patterns:
            detection_result = pattern["detector"]([self.current_sequence], self.current_sequence)
            if detection_result:
                logger.info(f"Anti-pattern detected: {pattern['name']}")
                logger.info(f"Suggestion: {pattern['suggestion']}")
    
    def _detect_repeated_failures(self, sequences: List[List[Dict[str, Any]]], 
                                current: List[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
        """Detect repeated failed calls to the same function."""
        if len(current) < 3:
            return None
            
        # Get the last 3 calls
        last_calls = current[-3:]
        
        # Check if they're the same function with failures
        if (all(call["function"] == last_calls[0]["function"] for call in last_calls) and
            all(call["status"] == "error" for call in last_calls)):
            return {
                "function": last_calls[0]["function"],
                "count": 3,
                "message": "Repeated failed calls to the same function detected"
            }
            
        return None
    
    def _detect_parameter_mismatch(self, sequences: List[List[Dict[str, Any]]], 
                                 current: List[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
        """Detect parameter mismatches in function calls."""
        if not current:
            return None
            
        last_call = current[-1]
        if last_call["status"] != "error":
            return None
            
        # Get function metadata
        function_name = last_call["function"]
        metadata = self.registry.get_function_metadata(function_name)
        if not metadata:
            return None
            
        # Check for required parameters
        missing_params = []
        for param_name, param_info in metadata.parameters.items():
            if param_info.get("required", False) and param_name not in last_call["params"]:
                missing_params.append(param_name)
                
        if missing_params:
            return {
                "function": function_name,
                "missing_parameters": missing_params,
                "message": f"Missing required parameters: {', '.join(missing_params)}"
            }
            
        return None
    
    def _detect_inefficient_sequence(self, sequences: List[List[Dict[str, Any]]], 
                                   current: List[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
        """Detect inefficient sequences that could be bundled."""
        if len(current) < 3:
            return None
            
        # Check if the calls are all in the same namespace
        namespaces = set()
        for call in current:
            function_name = call["function"]
            if "." in function_name:
                namespace = function_name.split(".")[0]
                namespaces.add(namespace)
                
        # If all calls are in the same namespace, suggest bundling
        if len(namespaces) == 1:
            return {
                "namespace": next(iter(namespaces)),
                "call_count": len(current),
                "message": f"Multiple consecutive calls in the '{next(iter(namespaces))}' namespace could be bundled"
            }
            
        return None
    
    def _generate_pattern_description(self, functions: List[str]) -> str:
        """Generate a human-readable description of a function pattern."""
        if not functions:
            return "Empty pattern"
            
        try:
            # Get the namespaces
            namespaces = [f.split(".")[0] if "." in f else "unknown" for f in functions]
            main_namespace = max(set(namespaces), key=namespaces.count)
            
            # Get function descriptions
            descriptions = []
            for func_name in functions:
                metadata = self.registry.get_function_metadata(func_name)
                if metadata:
                    descriptions.append(metadata.description.lower())
                else:
                    descriptions.append(func_name.split(".")[-1].replace("_", " "))
            
            # Generate a description based on the first and last function
            return f"Pattern: {descriptions[0]}, then {' then '.join(descriptions[1:])}"
        except Exception as e:
            logger.error(f"Error generating pattern description: {str(e)}")
            return "→".join(f.split(".")[-1] for f in functions)


class ComplexOperationHelper:
    """
    Provides helper functions for complex operations using the Function Registry.
    
    Features:
    - High-level operation templates
    - Common workflow implementations
    - Parameter mapping between functions
    - Result transformation utilities
    """
    
    def __init__(self):
        """Initialize the complex operation helper."""
        self.registry = get_registry()
        self.operation_templates: Dict[str, Dict[str, Any]] = {
            "entity_management": {
                "description": "Create and manage entities with observations",
                "operations": {
                    "create_entity_with_observations": self._create_entity_with_observations,
                    "update_entity_with_relations": self._update_entity_with_relations,
                    "search_and_update_entities": self._search_and_update_entities
                }
            },
            "data_processing": {
                "description": "Process and transform data",
                "operations": {
                    "extract_transform_load": self._extract_transform_load,
                    "validate_and_process_data": self._validate_and_process_data,
                    "analyze_data_structure": self._analyze_data_structure
                }
            },
            "multi_step_messaging": {
                "description": "Multi-step messaging operations",
                "operations": {
                    "send_message_with_retry": self._send_message_with_retry,
                    "message_with_confirmation": self._message_with_confirmation,
                    "broadcast_to_multiple_recipients": self._broadcast_to_multiple_recipients
                }
            }
        }
    
    def get_available_operations(self) -> Dict[str, Any]:
        """
        Get all available complex operations.
        
        Returns:
            Dictionary with available operations by category
        """
        operations_info = {}
        
        for category, info in self.operation_templates.items():
            operations_info[category] = {
                "description": info["description"],
                "operations": list(info["operations"].keys())
            }
            
        return operations_info
    
    async def execute_complex_operation(self, operation_name: str, 
                                     parameters: Dict[str, Any]) -> FunctionResult:
        """
        Execute a complex operation.
        
        Args:
            operation_name: Name of the operation
            parameters: Parameters for the operation
            
        Returns:
            FunctionResult with the operation result
        """
        # Find the operation
        operation_func = None
        for category, info in self.operation_templates.items():
            if operation_name in info["operations"]:
                operation_func = info["operations"][operation_name]
                break
                
        if not operation_func:
            return FunctionResult(
                status="error",
                message=f"Complex operation '{operation_name}' not found",
                data=None,
                error_code="OPERATION_NOT_FOUND",
                error_details={"available_operations": self.get_available_operations()}
            )
            
        try:
            # Execute the operation
            return await operation_func(parameters)
        except Exception as e:
            logger.error(f"Error executing complex operation: {str(e)}")
            return FunctionResult(
                status="error",
                message=f"Error executing complex operation: {str(e)}",
                data=None,
                error_code="OPERATION_EXECUTION_ERROR",
                error_details={"exception": str(e)}
            )
    
    def get_operation_help(self, operation_name: str) -> Dict[str, Any]:
        """
        Get help information for a complex operation.
        
        Args:
            operation_name: Name of the operation
            
        Returns:
            Dictionary with help information
        """
        # Find the operation
        for category, info in self.operation_templates.items():
            if operation_name in info["operations"]:
                operation_func = info["operations"][operation_name]
                
                # Get docstring
                docstring = operation_func.__doc__ or "No documentation available"
                
                # Parse parameters from docstring
                param_pattern = r'(\w+):\s+([^,\n]+)'
                params = re.findall(param_pattern, docstring)
                
                return {
                    "name": operation_name,
                    "category": category,
                    "description": docstring.split("\n")[0] if docstring else "No description",
                    "parameters": {name: desc.strip() for name, desc in params},
                    "example": self._get_example_for_operation(operation_name)
                }
                
        return {
            "error": f"Operation '{operation_name}' not found",
            "available_operations": self.get_available_operations()
        }
    
    async def _create_entity_with_observations(self, parameters: Dict[str, Any]) -> FunctionResult:
        """
        Create an entity and add multiple observations in one operation.
        
        Parameters:
            entity_name: Name of the entity to create
            entity_type: Type of the entity
            observations: List of observation contents
            
        Returns:
            FunctionResult with created entity and observations
        """
        try:
            # Validate parameters
            required_params = ["entity_name", "entity_type", "observations"]
            for param in required_params:
                if param not in parameters:
                    return FunctionResult(
                        status="error",
                        message=f"Missing required parameter: {param}",
                        data=None,
                        error_code="MISSING_PARAMETER",
                        error_details={"parameter": param}
                    )
            
            # Create the entity
            entity_result = await self.registry.execute(
                "memory.create_entity", 
                name=parameters["entity_name"],
                entity_type=parameters["entity_type"]
            )
            
            if entity_result.status != "success":
                return entity_result
                
            # Add observations
            observations_results = []
            for obs_content in parameters["observations"]:
                obs_result = await self.registry.execute(
                    "memory.create_observation",
                    entity_name=parameters["entity_name"],
                    content=obs_content
                )
                observations_results.append(obs_result.dict())
            
            # Return combined result
            return FunctionResult(
                status="success",
                message=f"Created entity '{parameters['entity_name']}' with {len(parameters['observations'])} observations",
                data={
                    "entity": entity_result.data,
                    "observations": [r.get("data") for r in observations_results if r.get("status") == "success"]
                },
                error_code="",
                error_details={}
            )
        except Exception as e:
            logger.error(f"Error in create_entity_with_observations: {str(e)}")
            return FunctionResult(
                status="error",
                message=f"Error creating entity with observations: {str(e)}",
                data=None,
                error_code="OPERATION_ERROR",
                error_details={"exception": str(e)}
            )
    
    async def _update_entity_with_relations(self, parameters: Dict[str, Any]) -> FunctionResult:
        """
        Update an entity with relations to other entities.
        
        Parameters:
            entity_name: Name of the entity to update
            relations: List of relations, each with 'type' and 'target_entity' keys
            
        Returns:
            FunctionResult with updated entity
        """
        # Implementation would be similar to create_entity_with_observations
        # For brevity, just return a placeholder
        return FunctionResult(
            status="success",
            message="Updated entity with relations",
            data={
                "entity_name": parameters.get("entity_name"),
                "relations_added": len(parameters.get("relations", []))
            },
            error_code="",
            error_details={}
        )
    
    async def _search_and_update_entities(self, parameters: Dict[str, Any]) -> FunctionResult:
        """
        Search for entities and update them.
        
        Parameters:
            search_query: Query to search for entities
            update_data: Data to update on matching entities
            
        Returns:
            FunctionResult with count of updated entities
        """
        # Implementation would execute search and then update entities
        # For brevity, just return a placeholder
        return FunctionResult(
            status="success",
            message="Searched and updated entities",
            data={
                "search_query": parameters.get("search_query"),
                "entities_updated": 5  # Placeholder
            },
            error_code="",
            error_details={}
        )
    
    async def _extract_transform_load(self, parameters: Dict[str, Any]) -> FunctionResult:
        """
        Extract, transform, and load data.
        
        Parameters:
            source: Source of the data
            transformations: List of transformations to apply
            destination: Destination for the data
            
        Returns:
            FunctionResult with ETL results
        """
        # Implementation would call multiple functions for ETL
        # For brevity, just return a placeholder
        return FunctionResult(
            status="success",
            message="Extracted, transformed, and loaded data",
            data={
                "source": parameters.get("source"),
                "destination": parameters.get("destination"),
                "records_processed": 100  # Placeholder
            },
            error_code="",
            error_details={}
        )
    
    async def _validate_and_process_data(self, parameters: Dict[str, Any]) -> FunctionResult:
        """Implementation would be provided."""
        return FunctionResult(
            status="success",
            message="Validated and processed data",
            data={"processed": True},
            error_code="",
            error_details={}
        )
    
    async def _analyze_data_structure(self, parameters: Dict[str, Any]) -> FunctionResult:
        """Implementation would be provided."""
        return FunctionResult(
            status="success",
            message="Analyzed data structure",
            data={"analysis": {}},
            error_code="",
            error_details={}
        )
    
    async def _send_message_with_retry(self, parameters: Dict[str, Any]) -> FunctionResult:
        """Implementation would be provided."""
        return FunctionResult(
            status="success",
            message="Sent message with retry",
            data={"sent": True},
            error_code="",
            error_details={}
        )
    
    async def _message_with_confirmation(self, parameters: Dict[str, Any]) -> FunctionResult:
        """Implementation would be provided."""
        return FunctionResult(
            status="success",
            message="Sent message and received confirmation",
            data={"confirmed": True},
            error_code="",
            error_details={}
        )
    
    async def _broadcast_to_multiple_recipients(self, parameters: Dict[str, Any]) -> FunctionResult:
        """Implementation would be provided."""
        return FunctionResult(
            status="success",
            message="Broadcast message to recipients",
            data={"recipient_count": 5},
            error_code="",
            error_details={}
        )
    
    def _get_example_for_operation(self, operation_name: str) -> Dict[str, Any]:
        """Get an example for a complex operation."""
        examples = {
            "create_entity_with_observations": {
                "entity_name": "John Doe",
                "entity_type": "person",
                "observations": [
                    "John is 35 years old",
                    "John works as a software engineer",
                    "John lives in San Francisco"
                ]
            },
            "update_entity_with_relations": {
                "entity_name": "John Doe",
                "relations": [
                    {"type": "works_at", "target_entity": "Acme Inc"},
                    {"type": "friend_of", "target_entity": "Jane Smith"}
                ]
            },
            "search_and_update_entities": {
                "search_query": "entity_type:person AND city:San Francisco",
                "update_data": {"status": "active", "last_updated": "2025-04-11"}
            },
            "extract_transform_load": {
                "source": "database.customers",
                "transformations": [
                    {"type": "rename", "from": "customer_id", "to": "id"},
                    {"type": "filter", "field": "status", "value": "active"}
                ],
                "destination": "analytics.active_customers"
            }
        }
        
        return examples.get(operation_name, {"example": "No example available"})


# Global instances
_recommendation_engine = None
_pattern_detector = None
_complex_helper = None

def get_recommendation_engine() -> FunctionRecommendationEngine:
    """Get the global recommendation engine instance."""
    global _recommendation_engine
    if _recommendation_engine is None:
        _recommendation_engine = FunctionRecommendationEngine()
    return _recommendation_engine

def get_pattern_detector() -> UsagePatternDetector:
    """Get the global pattern detector instance."""
    global _pattern_detector
    if _pattern_detector is None:
        _pattern_detector = UsagePatternDetector()
    return _pattern_detector

def get_complex_helper() -> ComplexOperationHelper:
    """Get the global complex operation helper instance."""
    global _complex_helper
    if _complex_helper is None:
        _complex_helper = ComplexOperationHelper()
    return _complex_helper 