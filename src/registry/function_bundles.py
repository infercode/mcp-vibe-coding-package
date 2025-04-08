#!/usr/bin/env python3
"""
Function Bundles

This module provides functionality for defining and executing bundles of related functions.
Bundles allow combining multiple functions into a single logical operation.
"""

import json
import asyncio
import copy
import os
from typing import Dict, List, Any, Optional, Union, Callable, Set

from src.registry.registry_manager import get_registry
from src.registry.function_models import FunctionResult
from src.registry.advanced_parameter_handler import parse_parameters
from src.logger import get_logger

logger = get_logger()

class BundleManager:
    """
    Manages function bundles and provides bundle execution capabilities.
    
    Features:
    - Bundle definition and validation
    - Bundle execution with context sharing
    - Result transformation between steps
    - Conditional execution based on results
    """
    
    def __init__(self):
        """Initialize the bundle manager."""
        self.registry = get_registry()
        self.bundles = {}
        self.transformers = {}
        
        # Register some default bundles
        self._register_default_bundles()
        
        # Register default transformers
        self.register_transformer("extract_entity_id", self._extract_entity_id)
        self.register_transformer("extract_entities", self._extract_entities)
        self.register_transformer("json_path", self._json_path_extract)
        self.register_transformer("combine", self._combine_values)
        
    def register_bundle(self, bundle_name: str, bundle_definition: Dict[str, Any]) -> None:
        """
        Register a function bundle.
        
        Args:
            bundle_name: Name of the bundle
            bundle_definition: Bundle specification
        """
        # Validate the bundle definition
        self._validate_bundle_definition(bundle_definition)
        
        # Add bundle to the registry
        self.bundles[bundle_name] = bundle_definition
        logger.info(f"Registered bundle '{bundle_name}' with {len(bundle_definition['steps'])} steps")
        
    def register_transformer(self, name: str, transformer_func: Callable) -> None:
        """
        Register a result transformer function.
        
        Transformers convert results from one step to parameters for the next step.
        
        Args:
            name: Name of the transformer
            transformer_func: Function that transforms results
        """
        self.transformers[name] = transformer_func
        logger.info(f"Registered transformer '{name}'")
        
    def get_bundle(self, bundle_name: str) -> Optional[Dict[str, Any]]:
        """
        Get a bundle definition by name.
        
        Args:
            bundle_name: Name of the bundle
            
        Returns:
            Bundle definition or None if not found
        """
        return self.bundles.get(bundle_name)
    
    def list_bundles(self) -> List[Dict[str, Any]]:
        """
        List all registered bundles.
        
        Returns:
            List of bundle information
        """
        return [
            {
                "name": name,
                "description": bundle.get("description", ""),
                "steps": len(bundle.get("steps", [])),
                "required_parameters": bundle.get("required_parameters", [])
            }
            for name, bundle in self.bundles.items()
        ]
        
    async def execute_bundle(self, 
                           bundle_name: str, 
                           parameters: Dict[str, Any],
                           include_step_results: bool = True) -> FunctionResult:
        """
        Execute a function bundle.
        
        Args:
            bundle_name: Name of the bundle to execute
            parameters: Parameters for the bundle
            include_step_results: Whether to include individual step results
            
        Returns:
            FunctionResult with bundle execution results
        """
        # Check if bundle exists
        bundle = self.get_bundle(bundle_name)
        if not bundle:
            return FunctionResult.error(
                message=f"Bundle '{bundle_name}' not found",
                error_code="bundle_not_found",
                error_details={"available_bundles": list(self.bundles.keys())}
            )
        
        # Check required parameters
        required_params = bundle.get("required_parameters", [])
        missing_params = [param for param in required_params if param not in parameters]
        
        if missing_params:
            return FunctionResult.error(
                message=f"Missing required parameters: {', '.join(missing_params)}",
                error_code="missing_parameters",
                error_details={"missing_parameters": missing_params}
            )
        
        # Initialize context with input parameters
        context = copy.deepcopy(parameters)
        step_results = {}
        
        # Execute each step
        for step_index, step in enumerate(bundle["steps"]):
            step_name = step.get("name", f"step_{step_index + 1}")
            
            # Check if this step should be executed based on condition
            if not self._evaluate_condition(step.get("condition"), context, step_results):
                logger.info(f"Skipping step '{step_name}' due to condition")
                step_results[step_name] = {
                    "status": "skipped",
                    "reason": "Condition not met"
                }
                continue
            
            # Get function and parameters for this step
            function_name = step["function"]
            params_mapping = step.get("params_mapping", {})
            transform = step.get("transform", {})
            
            # Map parameters from context to function parameters
            try:
                step_params = self._map_parameters(params_mapping, transform, context, step_results)
            except Exception as e:
                logger.error(f"Error mapping parameters for step '{step_name}': {str(e)}")
                step_results[step_name] = {
                    "status": "error",
                    "message": f"Parameter mapping error: {str(e)}"
                }
                
                # Check if we should continue on error
                if not step.get("continue_on_error", False):
                    return FunctionResult.error(
                        message=f"Error in bundle step '{step_name}': {str(e)}",
                        error_code="step_parameter_error",
                        error_details={
                            "step": step_name,
                            "step_index": step_index,
                            "error": str(e),
                            "partial_results": step_results if include_step_results else None
                        }
                    )
                continue
            
            # Execute the function
            try:
                result = await self.registry.execute(function_name, **step_params)
                step_results[step_name] = json.loads(result.to_json())
                
                # Update context with results if successful
                if result.status == "success" and result.data:
                    # Store the result data in the context
                    result_key = step.get("result_key", step_name)
                    context[result_key] = result.data
                    
                    # Also store individual fields if specified
                    if "result_mapping" in step and isinstance(step["result_mapping"], dict):
                        for context_key, result_path in step["result_mapping"].items():
                            context[context_key] = self._extract_value(result.data, result_path)
            except Exception as e:
                logger.error(f"Error executing step '{step_name}': {str(e)}")
                step_results[step_name] = {
                    "status": "error",
                    "message": f"Execution error: {str(e)}"
                }
                
                # Check if we should continue on error
                if not step.get("continue_on_error", False):
                    return FunctionResult.error(
                        message=f"Error in bundle step '{step_name}': {str(e)}",
                        error_code="step_execution_error",
                        error_details={
                            "step": step_name,
                            "step_index": step_index,
                            "error": str(e),
                            "partial_results": step_results if include_step_results else None
                        }
                    )
        
        # Create final result
        output_mapping = bundle.get("output_mapping", {})
        bundle_output = {}
        
        # Map output values if specified
        if output_mapping:
            for output_key, mapping_info in output_mapping.items():
                if isinstance(mapping_info, str):
                    # Simple mapping from context
                    bundle_output[output_key] = context.get(mapping_info)
                elif isinstance(mapping_info, dict) and "transform" in mapping_info:
                    # Transform the value
                    try:
                        bundle_output[output_key] = self._apply_transform(
                            mapping_info["transform"],
                            context,
                            step_results
                        )
                    except Exception as e:
                        logger.error(f"Error transforming output '{output_key}': {str(e)}")
        else:
            # Use the entire context as output
            bundle_output = context
        
        # Add step results if requested
        if include_step_results:
            bundle_output["_step_results"] = step_results
        
        return FunctionResult.success(
            message=f"Successfully executed bundle '{bundle_name}'",
            data=bundle_output
        )
    
    def save_bundles(self, file_path: str) -> None:
        """
        Save all bundle definitions to a JSON file.
        
        Args:
            file_path: Path to save the bundle definitions
        """
        with open(file_path, 'w') as f:
            json.dump(self.bundles, f, indent=2)
    
    def load_bundles(self, file_path: str) -> int:
        """
        Load bundle definitions from a JSON file.
        
        Args:
            file_path: Path to the bundle definitions file
            
        Returns:
            Number of bundles loaded
        """
        if not os.path.exists(file_path):
            logger.info(f"Bundle file not found: {file_path}")
            return 0
            
        with open(file_path, 'r') as f:
            bundles = json.load(f)
            
        # Register each bundle
        count = 0
        for name, definition in bundles.items():
            try:
                self.register_bundle(name, definition)
                count += 1
            except Exception as e:
                logger.error(f"Error registering bundle '{name}': {str(e)}")
                
        return count
    
    def _map_parameters(self, 
                       params_mapping: Dict[str, Any], 
                       transform: Dict[str, Any],
                       context: Dict[str, Any],
                       step_results: Dict[str, Any]) -> Dict[str, Any]:
        """Map parameters from context to function parameters."""
        params = {}
        
        # Apply direct mappings
        for param_name, mapping_info in params_mapping.items():
            if isinstance(mapping_info, str):
                # Simple mapping from context
                params[param_name] = context.get(mapping_info)
            elif isinstance(mapping_info, dict) and "default" in mapping_info:
                # Mapping with default
                context_key = mapping_info.get("from", param_name)
                params[param_name] = context.get(context_key, mapping_info["default"])
        
        # Apply transforms
        for param_name, transform_info in transform.items():
            params[param_name] = self._apply_transform(transform_info, context, step_results)
            
        return params
    
    def _apply_transform(self, 
                        transform_info: Dict[str, Any],
                        context: Dict[str, Any],
                        step_results: Dict[str, Any]) -> Any:
        """Apply a transform to generate a parameter value."""
        if "transformer" in transform_info:
            # Use a registered transformer
            transformer_name = transform_info["transformer"]
            if transformer_name not in self.transformers:
                raise ValueError(f"Transformer '{transformer_name}' not found")
                
            # Get inputs for the transformer
            inputs = {}
            for input_name, input_source in transform_info.get("inputs", {}).items():
                inputs[input_name] = self._extract_value(context, input_source)
                
            # Apply the transformer
            return self.transformers[transformer_name](**inputs)
        elif "template" in transform_info:
            # Use a string template
            template = transform_info["template"]
            
            # Replace placeholders with values from context
            for key, value in context.items():
                if isinstance(value, (str, int, float, bool)):
                    template = template.replace(f"{{{key}}}", str(value))
                    
            return template
        elif "concat" in transform_info:
            # Concatenate multiple values
            values = []
            for source in transform_info["concat"]:
                values.append(str(self._extract_value(context, source)))
                
            # Join with separator if specified
            separator = transform_info.get("separator", "")
            return separator.join(values)
        
        raise ValueError(f"Invalid transform: {transform_info}")
    
    def _extract_value(self, data: Dict[str, Any], path: str) -> Any:
        """Extract a value from a nested dictionary using a dot-notation path."""
        if not path:
            return data
            
        parts = path.split('.')
        value = data
        
        for part in parts:
            if isinstance(value, dict) and part in value:
                value = value[part]
            else:
                return None
                
        return value
    
    def _evaluate_condition(self, 
                          condition: Optional[Dict[str, Any]], 
                          context: Dict[str, Any],
                          step_results: Dict[str, Any]) -> bool:
        """Evaluate if a step condition is met."""
        if not condition:
            return True  # No condition means always execute
            
        # Check for 'exists' condition (parameter must exist and not be None)
        if "exists" in condition:
            param_name = condition["exists"]
            return param_name in context and context[param_name] is not None
            
        # Check for 'equals' condition
        if "equals" in condition:
            for key, value in condition["equals"].items():
                if key not in context or context[key] != value:
                    return False
            return True
            
        # Check for 'not_equals' condition
        if "not_equals" in condition:
            for key, value in condition["not_equals"].items():
                if key in context and context[key] == value:
                    return False
            return True
            
        # Check for 'step_success' condition (a previous step must have succeeded)
        if "step_success" in condition:
            step_name = condition["step_success"]
            return (step_name in step_results and
                   isinstance(step_results[step_name], dict) and
                   step_results[step_name].get("status") == "success")
            
        # Check for 'step_error' condition (a previous step must have failed)
        if "step_error" in condition:
            step_name = condition["step_error"]
            return (step_name in step_results and
                   isinstance(step_results[step_name], dict) and
                   step_results[step_name].get("status") == "error")
                   
        # Check for custom expression
        if "expression" in condition:
            try:
                # Create a safe evaluation context
                eval_context = {"context": context, "results": step_results}
                return bool(eval(condition["expression"], {"__builtins__": {}}, eval_context))
            except Exception as e:
                logger.error(f"Error evaluating condition expression: {str(e)}")
                return False
                
        return True  # Default to executing if the condition is not recognized
    
    def _validate_bundle_definition(self, definition: Dict[str, Any]) -> None:
        """Validate that a bundle definition has the required structure."""
        if not isinstance(definition, dict):
            raise ValueError("Bundle definition must be a dictionary")
            
        if "steps" not in definition or not isinstance(definition["steps"], list):
            raise ValueError("Bundle definition must contain a 'steps' list")
            
        # Validate each step
        for i, step in enumerate(definition["steps"]):
            if not isinstance(step, dict):
                raise ValueError(f"Step {i} must be a dictionary")
                
            if "function" not in step:
                raise ValueError(f"Step {i} must contain a 'function' key")
                
            # Check if the function exists
            if not self.registry.get_function_metadata(step["function"]):
                logger.info(f"Function '{step['function']}' in step {i} not found in registry")
    
    def _register_default_bundles(self) -> None:
        """Register default function bundles."""
        # Memory operations bundle
        self.register_bundle("create_entity_with_observation", {
            "description": "Create an entity and add an observation to it",
            "required_parameters": ["entity_name", "entity_type", "observation_content"],
            "steps": [
                {
                    "name": "create_entity",
                    "function": "memory.create_entity",
                    "params_mapping": {
                        "name": "entity_name",
                        "entity_type": "entity_type",
                        "properties": {"default": {}}
                    }
                },
                {
                    "name": "add_observation",
                    "function": "memory.create_observation",
                    "params_mapping": {
                        "entity_name": "entity_name",
                        "content": "observation_content"
                    },
                    "condition": {
                        "step_success": "create_entity"
                    }
                }
            ],
            "output_mapping": {
                "entity": "create_entity.entity",
                "observation": "add_observation.observation"
            }
        })
        
        # Project setup bundle
        self.register_bundle("create_project", {
            "description": "Create a project with initial components",
            "required_parameters": ["project_name", "component_names"],
            "steps": [
                {
                    "name": "create_project",
                    "function": "project.create_container",
                    "params_mapping": {
                        "name": "project_name"
                    }
                },
                {
                    "name": "create_components",
                    "function": "project.create_component",
                    "params_mapping": {
                        "project_id": "project_name",
                        "name": "current_component"
                    },
                    "condition": {
                        "step_success": "create_project"
                    },
                    "iterate_over": "component_names",
                    "iterate_as": "current_component"
                }
            ]
        })
        
        # Search and summarize bundle
        self.register_bundle("search_and_summarize", {
            "description": "Search for entities and generate a summary",
            "required_parameters": ["search_query", "search_types"],
            "steps": [
                {
                    "name": "search_entities",
                    "function": "memory.search_entities",
                    "params_mapping": {
                        "query": "search_query",
                        "entity_types": "search_types",
                        "limit": {"default": 10}
                    }
                },
                {
                    "name": "fetch_details",
                    "function": "memory.get_entity_details",
                    "params_mapping": {
                        "entity_name": "current_entity"
                    },
                    "condition": {
                        "step_success": "search_entities"
                    },
                    "iterate_over": "search_entities.entities",
                    "iterate_as": "current_entity",
                    "collect_as": "entity_details"
                },
                {
                    "name": "generate_summary",
                    "function": "util.generate_summary",
                    "params_mapping": {
                        "items": "entity_details",
                        "format": {"default": "text"}
                    },
                    "condition": {
                        "exists": "entity_details"
                    }
                }
            ],
            "output_mapping": {
                "search_results": "search_entities.entities",
                "detail_count": {"transform": {"transformer": "count", "inputs": {"items": "entity_details"}}},
                "summary": "generate_summary.summary"
            }
        })
    
    # Default transformers
    
    def _extract_entity_id(self, entity_data: Dict[str, Any]) -> str:
        """Extract entity ID from entity data."""
        return entity_data.get("id", "")
    
    def _extract_entities(self, search_results: Dict[str, Any]) -> List[str]:
        """Extract entity names from search results."""
        if "entities" in search_results and isinstance(search_results["entities"], list):
            return [entity.get("name", "") for entity in search_results["entities"]]
        return []
    
    def _json_path_extract(self, data: Dict[str, Any], path: str) -> Any:
        """Extract value using a JSON path."""
        return self._extract_value(data, path)
    
    def _combine_values(self, values: List[Any], separator: str = " ") -> str:
        """Combine multiple values into a string."""
        return separator.join(str(v) for v in values)


# Create a singleton instance
_manager = None

def get_bundle_manager() -> BundleManager:
    """Get the global bundle manager instance."""
    global _manager
    if _manager is None:
        _manager = BundleManager()
    return _manager


# Convenience functions
def register_bundle(bundle_name: str, bundle_definition: Dict[str, Any]) -> None:
    """
    Register a function bundle.
    
    Args:
        bundle_name: Name of the bundle
        bundle_definition: Bundle specification
    """
    get_bundle_manager().register_bundle(bundle_name, bundle_definition)


def execute_bundle(bundle_name: str, 
                 parameters: Dict[str, Any],
                 include_step_results: bool = True) -> asyncio.Task:
    """
    Execute a function bundle.
    
    Args:
        bundle_name: Name of the bundle to execute
        parameters: Parameters for the bundle
        include_step_results: Whether to include individual step results
        
    Returns:
        Task with bundle execution results
    """
    return asyncio.create_task(get_bundle_manager().execute_bundle(bundle_name, parameters, include_step_results))


def list_bundles() -> List[Dict[str, Any]]:
    """
    List all registered bundles.
    
    Returns:
        List of bundle information
    """
    return get_bundle_manager().list_bundles()


def save_bundles(file_path: str) -> None:
    """
    Save all bundle definitions to a JSON file.
    
    Args:
        file_path: Path to save the bundle definitions
    """
    get_bundle_manager().save_bundles(file_path)


def load_bundles(file_path: str) -> int:
    """
    Load bundle definitions from a JSON file.
    
    Args:
        file_path: Path to the bundle definitions file
        
    Returns:
        Number of bundles loaded
    """
    return get_bundle_manager().load_bundles(file_path) 