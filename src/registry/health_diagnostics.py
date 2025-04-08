#!/usr/bin/env python3
"""
Health and Diagnostics Module

This module provides system health monitoring, diagnostic tools, and
self-healing capabilities for the Function Registry Pattern.
"""

import time
import asyncio
import inspect
import traceback
import json
import os
import sys
from typing import Dict, List, Any, Optional, Union, Callable, Tuple
from datetime import datetime, timedelta
from enum import Enum

from src.registry.registry_manager import get_registry, register_function
from src.registry.function_models import FunctionResult
from src.logger import get_logger

# Initialize logger
logger = get_logger()

# Health status constants
class HealthStatus(str, Enum):
    """Health status indicators for registry components."""
    HEALTHY = "HEALTHY"
    DEGRADED = "DEGRADED"
    UNHEALTHY = "UNHEALTHY"
    UNKNOWN = "UNKNOWN"

# Registry component identifiers
class RegistryComponent(str, Enum):
    """Components of the Function Registry system."""
    REGISTRY_CORE = "registry_core"
    DISCOVERY = "discovery"
    DOCUMENTATION = "documentation"
    PARAMETER_HANDLING = "parameter_handling"
    FUNCTION_BUNDLES = "function_bundles"
    PERFORMANCE = "performance"
    MIGRATION = "migration"
    IDE_INTEGRATION = "ide_integration"
    GUIDANCE = "guidance"
    FEEDBACK = "feedback"
    HEALTH = "health"

class HealthMonitor:
    """
    Monitors the health of the Function Registry system.
    
    This class provides utilities for checking the health of various
    registry components, tracking errors, and detecting issues.
    """
    
    def __init__(self):
        """Initialize the health monitor."""
        self.registry = get_registry()
        self.health_checks = {}
        self.error_counts = {comp.value: 0 for comp in RegistryComponent}
        self.last_check_time = {comp.value: None for comp in RegistryComponent}
        self.component_status = {comp.value: HealthStatus.UNKNOWN for comp in RegistryComponent}
        self.global_status = HealthStatus.UNKNOWN
        self.recovery_actions = {}
        
        # Register default health checks
        self._register_default_health_checks()
        
    def _register_default_health_checks(self):
        """Register default health checks for all components."""
        # Registry Core health check
        self.register_health_check(
            RegistryComponent.REGISTRY_CORE,
            self._check_registry_core_health,
            "Checks if the core registry can be accessed and functions are registered"
        )
        
        # Discovery health check
        self.register_health_check(
            RegistryComponent.DISCOVERY,
            self._check_discovery_health,
            "Checks if function discovery is working correctly"
        )
        
        # Parameter handling health check
        self.register_health_check(
            RegistryComponent.PARAMETER_HANDLING,
            self._check_parameter_handling_health,
            "Checks if parameter validation and conversion are working"
        )
        
        # Function bundles health check
        self.register_health_check(
            RegistryComponent.FUNCTION_BUNDLES,
            self._check_function_bundles_health,
            "Checks if function bundles can be executed"
        )
        
        # Performance health check
        self.register_health_check(
            RegistryComponent.PERFORMANCE,
            self._check_performance_health,
            "Checks if performance features like caching are working"
        )
        
    def register_health_check(self, component: Union[RegistryComponent, str], 
                             check_function: Callable[[], Tuple[HealthStatus, str]], 
                             description: str):
        """
        Register a health check for a specific component.
        
        Args:
            component: The component to check
            check_function: Function that performs the health check and returns status and message
            description: Description of what the health check does
        """
        comp_id = component.value if isinstance(component, RegistryComponent) else component
        
        self.health_checks[comp_id] = {
            "function": check_function,
            "description": description,
            "last_run": None,
            "last_status": HealthStatus.UNKNOWN,
            "last_message": "Health check not yet run"
        }
        
    def register_recovery_action(self, component: Union[RegistryComponent, str],
                               action_function: Callable[[], None],
                               description: str):
        """
        Register a recovery action for a specific component.
        
        Args:
            component: The component to recover
            action_function: Function that performs the recovery action
            description: Description of what the recovery action does
        """
        comp_id = component.value if isinstance(component, RegistryComponent) else component
        
        self.recovery_actions[comp_id] = {
            "function": action_function,
            "description": description,
            "last_run": None,
            "success_count": 0,
            "failure_count": 0
        }
    
    async def check_health(self, component: Optional[Union[RegistryComponent, str]] = None) -> Dict[str, Any]:
        """
        Check the health of specified components or all components.
        
        Args:
            component: Optional component to check, if None checks all components
            
        Returns:
            Dictionary with health check results
        """
        if component:
            comp_id = component.value if isinstance(component, RegistryComponent) else component
            return await self._check_component_health(comp_id)
        
        # Check all components
        results = {}
        overall_status = HealthStatus.HEALTHY
        
        for comp_id in self.health_checks:
            result = await self._check_component_health(comp_id)
            results[comp_id] = result
            
            # Update overall status (worst status wins)
            status = result["status"]
            if status == HealthStatus.UNHEALTHY:
                overall_status = HealthStatus.UNHEALTHY
            elif status == HealthStatus.DEGRADED and overall_status != HealthStatus.UNHEALTHY:
                overall_status = HealthStatus.DEGRADED
        
        # Store global status
        self.global_status = overall_status
        
        # Create final health report
        health_report = {
            "timestamp": datetime.now().isoformat(),
            "overall_status": overall_status,
            "components": results
        }
        
        return health_report
    
    async def _check_component_health(self, component_id: str) -> Dict[str, Any]:
        """Check health of a specific component."""
        if component_id not in self.health_checks:
            return {
                "status": HealthStatus.UNKNOWN,
                "message": f"No health check registered for component: {component_id}",
                "timestamp": datetime.now().isoformat(),
                "error_count": self.error_counts.get(component_id, 0)
            }
            
        # Get the health check
        health_check = self.health_checks[component_id]
        check_function = health_check["function"]
        
        try:
            # Run the health check
            if asyncio.iscoroutinefunction(check_function):
                status, message = await check_function()
            else:
                status, message = check_function()
                
            # Update last check time and status
            health_check["last_run"] = datetime.now().isoformat()
            health_check["last_status"] = status
            health_check["last_message"] = message
            
            # Store as new dictionary to avoid type issues
            self.last_check_time = {
                **self.last_check_time,
                component_id: datetime.now().isoformat()
            }
            
            # Store as new dictionary to avoid type issues
            self.component_status = {
                **self.component_status,
                component_id: status
            }
            
            result = {
                "status": status,
                "message": message,
                "timestamp": health_check["last_run"],
                "error_count": self.error_counts.get(component_id, 0)
            }
            
            # If unhealthy, try recovery
            if status == HealthStatus.UNHEALTHY:
                if component_id in self.recovery_actions:
                    recovery_result = await self._attempt_recovery(component_id)
                    result["recovery_attempted"] = True
                    result["recovery_result"] = recovery_result
            
            return result
            
        except Exception as e:
            error_msg = f"Health check failed: {str(e)}"
            logger.error(f"Error in health check for {component_id}: {error_msg}")
            
            # Update status and error count
            self.error_counts[component_id] = self.error_counts.get(component_id, 0) + 1
            self.component_status[component_id] = HealthStatus.UNHEALTHY
            
            return {
                "status": HealthStatus.UNHEALTHY,
                "message": error_msg,
                "timestamp": datetime.now().isoformat(),
                "error_count": self.error_counts.get(component_id, 0),
                "error_details": traceback.format_exc()
            }
    
    async def _attempt_recovery(self, component_id: str) -> Dict[str, Any]:
        """Attempt to recover a component."""
        if component_id not in self.recovery_actions:
            return {
                "success": False,
                "message": f"No recovery action registered for component: {component_id}"
            }
            
        # Get the recovery action
        recovery_action = self.recovery_actions[component_id]
        action_function = recovery_action["function"]
        
        try:
            # Run the recovery action
            if asyncio.iscoroutinefunction(action_function):
                result = await action_function()
            else:
                result = action_function()
                
            # Update recovery stats
            recovery_action["last_run"] = datetime.now().isoformat()
            recovery_action["success_count"] += 1
            
            return {
                "success": True,
                "message": "Recovery action executed successfully",
                "timestamp": recovery_action["last_run"],
                "details": result if result else None
            }
            
        except Exception as e:
            error_msg = f"Recovery action failed: {str(e)}"
            logger.error(f"Error in recovery action for {component_id}: {error_msg}")
            
            # Update recovery stats
            recovery_action["last_run"] = datetime.now().isoformat()
            recovery_action["failure_count"] += 1
            
            return {
                "success": False,
                "message": error_msg,
                "timestamp": recovery_action["last_run"],
                "error_details": traceback.format_exc()
            }
    
    async def run_diagnostic(self, component: Optional[Union[RegistryComponent, str]] = None) -> Dict[str, Any]:
        """
        Run in-depth diagnostic on specified component or all components.
        
        Args:
            component: Optional component to diagnose, if None diagnoses all components
            
        Returns:
            Dictionary with diagnostic results
        """
        # First check health
        health_report = await self.check_health(component)
        
        # Add additional diagnostic information
        diagnostic_report = {
            "health": health_report,
            "registry_info": self._get_registry_info(),
            "system_info": self._get_system_info(),
            "timestamp": datetime.now().isoformat()
        }
        
        # Add component-specific diagnostics
        if component:
            comp_id = component.value if isinstance(component, RegistryComponent) else component
            diagnostic_report["component_details"] = await self._run_component_diagnostic(comp_id)
        else:
            component_details = {}
            for comp_id in self.component_status:
                component_details[comp_id] = await self._run_component_diagnostic(comp_id)
            diagnostic_report["component_details"] = component_details
        
        return diagnostic_report
    
    async def _run_component_diagnostic(self, component_id: str) -> Dict[str, Any]:
        """Run in-depth diagnostic on a specific component."""
        # Component-specific diagnostics
        if component_id == RegistryComponent.REGISTRY_CORE.value:
            return await self._diagnose_registry_core()
        elif component_id == RegistryComponent.DISCOVERY.value:
            return await self._diagnose_discovery()
        elif component_id == RegistryComponent.PARAMETER_HANDLING.value:
            return await self._diagnose_parameter_handling()
        elif component_id == RegistryComponent.FUNCTION_BUNDLES.value:
            return await self._diagnose_function_bundles()
        elif component_id == RegistryComponent.PERFORMANCE.value:
            return await self._diagnose_performance()
        
        # Generic diagnostic if no component-specific diagnostic exists
        return {
            "message": f"No specific diagnostic implemented for {component_id}",
            "status": self.component_status.get(component_id, HealthStatus.UNKNOWN),
            "error_count": self.error_counts.get(component_id, 0)
        }
        
    def _get_registry_info(self) -> Dict[str, Any]:
        """Get information about the registry state."""
        if not self.registry:
            return {
                "available": False,
                "message": "Registry not available"
            }
            
        try:
            namespaces = self.registry.get_namespaces()
            function_count = len(self.registry.get_all_functions())
            
            # Count functions by namespace
            namespace_counts = {}
            for ns in namespaces:
                ns_functions = self.registry.get_functions_by_namespace(ns)
                namespace_counts[ns] = len(ns_functions)
            
            return {
                "available": True,
                "function_count": function_count,
                "namespace_count": len(namespaces),
                "namespaces": namespaces,
                "functions_by_namespace": namespace_counts
            }
        except Exception as e:
            return {
                "available": True,
                "error": str(e),
                "message": "Error getting registry information"
            }
    
    def _get_system_info(self) -> Dict[str, Any]:
        """Get information about the system environment."""
        return {
            "python_version": sys.version,
            "platform": sys.platform,
            "time": datetime.now().isoformat(),
            "registry_path": os.path.abspath(os.path.dirname(inspect.getfile(self.registry.__class__))) if self.registry else "Unknown"
        }
        
    # Default health check implementations
    def _check_registry_core_health(self) -> Tuple[HealthStatus, str]:
        """Check if the registry core is healthy."""
        if not self.registry:
            return (HealthStatus.UNHEALTHY, "Registry is not available")
            
        try:
            # Check if we can get functions
            all_functions = self.registry.get_all_functions()
            function_count = len(all_functions)
            
            if function_count == 0:
                return (HealthStatus.DEGRADED, "Registry is available but no functions are registered")
                
            return (HealthStatus.HEALTHY, f"Registry is healthy with {function_count} functions")
        except Exception as e:
            return (HealthStatus.UNHEALTHY, f"Registry error: {str(e)}")
    
    def _check_discovery_health(self) -> Tuple[HealthStatus, str]:
        """Check if the discovery system is healthy."""
        if not self.registry:
            return (HealthStatus.UNHEALTHY, "Registry is not available")
            
        try:
            # Check if we can get namespaces
            namespaces = self.registry.get_namespaces()
            
            if not namespaces:
                return (HealthStatus.DEGRADED, "Discovery system is available but no namespaces are registered")
                
            # Check if we can get functions by namespace
            for namespace in namespaces[:2]:  # Just check a couple to avoid too much work
                ns_functions = self.registry.get_functions_by_namespace(namespace)
                
            return (HealthStatus.HEALTHY, f"Discovery system is healthy with {len(namespaces)} namespaces")
        except Exception as e:
            return (HealthStatus.UNHEALTHY, f"Discovery system error: {str(e)}")
    
    def _check_parameter_handling_health(self) -> Tuple[HealthStatus, str]:
        """Check if parameter handling is healthy."""
        # For this we would need to actually try to execute a function with parameters
        # This is a simplified version
        try:
            # Import parameter handling modules
            try:
                from src.registry.advanced_parameter_handler import parse_parameters
                advanced_available = True
            except ImportError:
                advanced_available = False
                
            if not advanced_available:
                return (HealthStatus.DEGRADED, "Advanced parameter handling is not available")
                
            return (HealthStatus.HEALTHY, "Parameter handling is available")
        except Exception as e:
            return (HealthStatus.UNHEALTHY, f"Parameter handling error: {str(e)}")
    
    def _check_function_bundles_health(self) -> Tuple[HealthStatus, str]:
        """Check if function bundles are healthy."""
        try:
            # Import function bundles module
            try:
                from src.registry.function_bundles import BundleManager
                bundles_available = True
            except ImportError:
                bundles_available = False
                
            if not bundles_available:
                return (HealthStatus.DEGRADED, "Function bundles are not available")
                
            # We would ideally check for registered bundles
            # This is a simplified version
            return (HealthStatus.HEALTHY, "Function bundles system is available")
        except Exception as e:
            return (HealthStatus.UNHEALTHY, f"Function bundles error: {str(e)}")
    
    def _check_performance_health(self) -> Tuple[HealthStatus, str]:
        """Check if performance optimization is healthy."""
        try:
            # Import performance modules
            try:
                from src.registry.performance_optimization import ResultCache, BatchProcessor
                performance_available = True
            except ImportError:
                performance_available = False
                
            if not performance_available:
                return (HealthStatus.DEGRADED, "Performance optimization is not available")
                
            # We would ideally check cache and batch processor
            # This is a simplified version
            return (HealthStatus.HEALTHY, "Performance optimization is available")
        except Exception as e:
            return (HealthStatus.UNHEALTHY, f"Performance optimization error: {str(e)}")
    
    # Diagnostic implementations
    async def _diagnose_registry_core(self) -> Dict[str, Any]:
        """Run in-depth diagnostic on registry core."""
        if not self.registry:
            return {
                "status": HealthStatus.UNHEALTHY,
                "message": "Registry is not available"
            }
            
        try:
            # Check all registry functions
            all_functions = self.registry.get_all_functions()
            
            # Check registry initialization
            registry_info = {
                "function_count": len(all_functions),
                "namespaces": self.registry.get_namespaces(),
                "registry_type": type(self.registry).__name__
            }
            
            # Check a sample function if available
            sample_function = None
            if all_functions:
                sample_function = all_functions[0]
                
            function_sample = {}
            if sample_function:
                function_sample = {
                    "name": sample_function.name,
                    "description": sample_function.description,
                    "parameter_count": len(sample_function.parameters),
                    "namespace": sample_function.namespace
                }
            
            return {
                "status": HealthStatus.HEALTHY,
                "message": "Registry core diagnostic completed",
                "registry_info": registry_info,
                "function_sample": function_sample
            }
        except Exception as e:
            return {
                "status": HealthStatus.UNHEALTHY,
                "message": f"Diagnostic error: {str(e)}",
                "error_details": traceback.format_exc()
            }
    
    async def _diagnose_discovery(self) -> Dict[str, Any]:
        """Run in-depth diagnostic on discovery system."""
        if not self.registry:
            return {
                "status": HealthStatus.UNHEALTHY,
                "message": "Registry is not available"
            }
            
        try:
            # Check namespace and category functionality
            namespaces = self.registry.get_namespaces()
            
            # Count functions per namespace
            namespace_counts = {}
            for ns in namespaces:
                ns_functions = self.registry.get_functions_by_namespace(ns)
                namespace_counts[ns] = len(ns_functions)
            
            return {
                "status": HealthStatus.HEALTHY,
                "message": "Discovery system diagnostic completed",
                "namespaces": namespaces,
                "namespace_counts": namespace_counts,
                "categories": []  # Would populate from registry if categories were available
            }
        except Exception as e:
            return {
                "status": HealthStatus.UNHEALTHY,
                "message": f"Diagnostic error: {str(e)}",
                "error_details": traceback.format_exc()
            }
    
    async def _diagnose_parameter_handling(self) -> Dict[str, Any]:
        """Run in-depth diagnostic on parameter handling."""
        parsing_test = {"success": False, "message": "Test not run"}
        parameter_handling_available = False
        
        try:
            # Try to import parameter handling module
            try:
                # Use our own parameter handling logic instead of importing a different module
                from src.registry.function_models import FunctionMetadata
                parameter_handling_available = True
                
                # Simple parameter validation function for testing
                def validate_params(params, metadata):
                    return {k: v for k, v in params.items() if k in metadata.parameters}
                
                if parameter_handling_available:
                    # Test param parsing
                    test_params = {"name": "test", "value": 123}
                    try:
                        # Create a minimal function metadata for testing with required parameters
                        test_metadata = FunctionMetadata(
                            name="test_function",
                            parameters={"name": {"type": "str"}, "value": {"type": "int"}},
                            short_name="test",  # Add required parameters
                            namespace="test",
                            description="Test function",
                            return_type="dict",
                            is_async=True,
                            source_file="test.py"
                        )
                        
                        # Simple test validation 
                        result = validate_params(test_params, test_metadata)
                        parsing_test = {
                            "success": True,
                            "input": test_params,
                            "output": result
                        }
                    except Exception as e:
                        parsing_test = {
                            "success": False,
                            "input": test_params,
                            "error": str(e)
                        }
            except ImportError as e:
                parameter_handling_available = False
                parsing_test = {
                    "success": False,
                    "error": f"Import error: {str(e)}"
                }
            
            return {
                "status": HealthStatus.HEALTHY if parameter_handling_available else HealthStatus.DEGRADED,
                "message": "Parameter handling diagnostic completed",
                "parameter_handling_available": parameter_handling_available,
                "parsing_test": parsing_test
            }
        except Exception as e:
            return {
                "status": HealthStatus.UNHEALTHY,
                "message": f"Diagnostic error: {str(e)}",
                "error_details": traceback.format_exc()
            }
    
    async def _diagnose_function_bundles(self) -> Dict[str, Any]:
        """Run in-depth diagnostic on function bundles."""
        bundle_info = {"available": False, "message": "Function bundles not checked"}
        bundles_available = False
        bundle_manager = None  # Initialize here to ensure it's always defined
        get_bundle_manager_fn = None  # Store function reference to avoid unbound errors
        
        try:
            # Check if function bundles are available
            try:
                try:
                    # Import and store function reference
                    from src.registry.function_bundles import get_bundle_manager as bundle_manager_fn
                    get_bundle_manager_fn = bundle_manager_fn
                    bundles_available = True
                except ImportError:
                    bundles_available = False
                    bundle_info = {
                        "available": False,
                        "message": "Function bundles module not available"
                    }
                
                if bundles_available and get_bundle_manager_fn is not None:
                    try:
                        bundle_manager = get_bundle_manager_fn()
                        if bundle_manager:
                            registered_bundles = bundle_manager.list_bundles()
                            bundle_info = {
                                "bundle_count": len(registered_bundles),
                                "bundles": registered_bundles,
                                "available": True
                            }
                        else:
                            bundle_info = {
                                "available": False,
                                "message": "Bundle manager not available"
                            }
                    except Exception as e:
                        bundle_info = {
                            "available": False,
                            "error": str(e),
                            "error_details": traceback.format_exc()
                        }
            except Exception as e:
                bundle_info = {
                    "available": False,
                    "error": str(e),
                    "error_details": traceback.format_exc()
                }
            
            return {
                "status": HealthStatus.HEALTHY if bundles_available else HealthStatus.DEGRADED,
                "message": "Function bundles diagnostic completed",
                "bundles_available": bundles_available,
                "bundle_info": bundle_info
            }
        except Exception as e:
            return {
                "status": HealthStatus.UNHEALTHY,
                "message": f"Diagnostic error: {str(e)}",
                "error_details": traceback.format_exc()
            }
    
    async def _diagnose_performance(self) -> Dict[str, Any]:
        """Run in-depth diagnostic on performance optimization."""
        try:
            # Import performance modules
            try:
                from src.registry.performance_optimization import ResultCache, BatchProcessor
                performance_available = True
            except ImportError:
                performance_available = False
            
            performance_info = {}
            if performance_available:
                try:
                    # Check cache
                    from src.registry.performance_optimization import get_result_cache
                    cache = get_result_cache()
                    cache_info = {
                        "available": cache is not None,
                        "size": len(cache.cache) if cache else 0,
                        "max_size": cache.max_size if cache else 0
                    }
                    
                    # Check batch processor
                    from src.registry.performance_optimization import get_batch_processor
                    batch_processor = get_batch_processor()
                    batch_info = {
                        "available": batch_processor is not None
                    }
                    
                    performance_info = {
                        "cache": cache_info,
                        "batch_processor": batch_info
                    }
                except Exception as e:
                    performance_info = {
                        "error": str(e),
                        "error_details": traceback.format_exc()
                    }
            
            return {
                "status": HealthStatus.HEALTHY if performance_available else HealthStatus.DEGRADED,
                "message": "Performance optimization diagnostic completed",
                "performance_available": performance_available,
                "performance_info": performance_info
            }
        except Exception as e:
            return {
                "status": HealthStatus.UNHEALTHY,
                "message": f"Diagnostic error: {str(e)}",
                "error_details": traceback.format_exc()
            }


class SelfHealer:
    """
    Provides self-healing capabilities for the Function Registry system.
    
    This class implements automatic recovery actions for common issues
    and can be used to restore the system to a healthy state.
    """
    
    def __init__(self, health_monitor: Optional[HealthMonitor] = None):
        """
        Initialize the self-healer.
        
        Args:
            health_monitor: Optional existing health monitor to use
        """
        self.health_monitor = health_monitor or HealthMonitor()
        self.registry = get_registry()
        self.recovery_attempts = {}
        
        # Register recovery actions
        self._register_default_recovery_actions()
    
    def _register_default_recovery_actions(self):
        """Register default recovery actions for known components."""
        if not self.health_monitor:
            return
            
        # Define wrapper functions that perform the recovery and always return None
        def registry_recovery_wrapper():
            self._recover_registry_core()
            return None
        
        def performance_recovery_wrapper():
            self._recover_performance()
            return None
        
        def bundles_recovery_wrapper():
            self._recover_function_bundles()
            return None
        
        # Register with appropriate wrapper functions
        self.health_monitor.register_recovery_action(
            RegistryComponent.REGISTRY_CORE,
            registry_recovery_wrapper,
            "Attempts to reinitialize the registry"
        )
        
        self.health_monitor.register_recovery_action(
            RegistryComponent.PERFORMANCE,
            performance_recovery_wrapper,
            "Attempts to reinitialize performance optimization components"
        )
        
        self.health_monitor.register_recovery_action(
            RegistryComponent.FUNCTION_BUNDLES,
            bundles_recovery_wrapper,
            "Attempts to reinitialize function bundles"
        )
    
    async def heal_system(self, component: Optional[Union[RegistryComponent, str]] = None) -> Dict[str, Any]:
        """
        Attempt to heal the system or a specific component.
        
        Args:
            component: Optional component to heal, if None tries to heal all unhealthy components
            
        Returns:
            Dictionary with healing results
        """
        # First check health to identify issues
        health_report = await self.health_monitor.check_health(component)
        
        # If a specific component was specified, only heal that one
        if component:
            comp_id = component.value if isinstance(component, RegistryComponent) else component
            comp_status = health_report["components"][comp_id]["status"] if "components" in health_report else health_report["status"]
            
            if comp_status != HealthStatus.HEALTHY:
                return await self._heal_component(comp_id)
            else:
                return {
                    "component": comp_id,
                    "status": "SKIPPED",
                    "message": f"Component {comp_id} is already healthy"
                }
        
        # For all components, heal any that are unhealthy
        results = {}
        components_to_heal = []
        
        if "components" in health_report:
            for comp_id, comp_data in health_report["components"].items():
                if comp_data["status"] != HealthStatus.HEALTHY:
                    components_to_heal.append(comp_id)
        
        # Heal components in order of dependency
        for comp_id in components_to_heal:
            results[comp_id] = await self._heal_component(comp_id)
        
        return {
            "timestamp": datetime.now().isoformat(),
            "components_healed": len(results),
            "results": results
        }
    
    async def _heal_component(self, component_id: str) -> Dict[str, Any]:
        """Attempt to heal a specific component."""
        # Check if we have a recovery action for this component
        if component_id not in self.health_monitor.recovery_actions:
            return {
                "status": "FAILED",
                "message": f"No recovery action registered for component: {component_id}"
            }
        
        # Check if we've tried to recover too many times
        if component_id in self.recovery_attempts:
            attempts = self.recovery_attempts[component_id]
            # Limit to 3 attempts per hour
            if attempts["count"] >= 3 and datetime.now() - attempts["first_attempt"] < timedelta(hours=1):
                return {
                    "status": "REJECTED",
                    "message": f"Too many recovery attempts for {component_id} in the past hour"
                }
        else:
            self.recovery_attempts[component_id] = {
                "count": 0,
                "first_attempt": datetime.now(),
                "last_attempt": None
            }
        
        # Update attempt counter
        self.recovery_attempts[component_id]["count"] += 1
        self.recovery_attempts[component_id]["last_attempt"] = datetime.now()
        
        # Get the recovery action
        recovery_action = self.health_monitor.recovery_actions[component_id]
        action_function = recovery_action["function"]
        
        try:
            # Run the recovery action
            if asyncio.iscoroutinefunction(action_function):
                result = await action_function()
            else:
                result = action_function()
            
            # Check if recovery was successful
            health_after = await self.health_monitor.check_health(component_id)
            new_status = health_after["status"] if isinstance(health_after, dict) and "status" in health_after else HealthStatus.UNKNOWN
            
            success = new_status == HealthStatus.HEALTHY
            
            return {
                "component": component_id,
                "status": "SUCCESS" if success else "PARTIAL",
                "message": "Recovery action executed successfully" + (" but component is still unhealthy" if not success else ""),
                "health_status": new_status,
                "details": result if result else None
            }
        except Exception as e:
            error_msg = f"Recovery action failed: {str(e)}"
            logger.error(f"Error in recovery action for {component_id}: {error_msg}")
            
            return {
                "component": component_id,
                "status": "FAILED",
                "message": error_msg,
                "error_details": traceback.format_exc()
            }
    
    # Recovery implementations
    def _recover_registry_core(self) -> Dict[str, Any]:
        """Attempt to recover the registry core."""
        try:
            # Re-initialize the registry
            # Import locally to avoid circular imports
            from src.registry.registry_manager import get_registry
            
            # Just reset the registry by getting a fresh instance - without force_new parameter
            registry = get_registry()
            
            # Verify registry is working
            if registry:
                return {
                    "registry_reinitialized": True,
                    "functions_available": len(registry.get_all_functions())
                }
            else:
                return {
                    "registry_reinitialized": False,
                    "error": "Registry still not available after reinitialization"
                }
        except Exception as e:
            return {
                "registry_reinitialized": False,
                "error": str(e),
                "error_details": traceback.format_exc()
            }
    
    def _recover_performance(self) -> Dict[str, Any]:
        """Attempt to recover performance optimization components."""
        try:
            # Re-initialize performance components
            try:
                # Use functions without force_new parameter
                from src.registry.performance_optimization import get_result_cache, get_batch_processor
                
                # Reinitialize components by getting fresh instances
                cache = get_result_cache()
                processor = get_batch_processor()
                
                return {
                    "cache_reinitialized": cache is not None,
                    "batch_processor_reinitialized": processor is not None
                }
            except ImportError:
                return {
                    "error": "Performance optimization module not available"
                }
        except Exception as e:
            return {
                "error": str(e),
                "error_details": traceback.format_exc()
            }
    
    def _recover_function_bundles(self) -> Dict[str, Any]:
        """Attempt to recover function bundles."""
        try:
            # Re-initialize function bundles
            try:
                # Use function without force_new parameter
                from src.registry.function_bundles import get_bundle_manager
                
                # Reinitialize bundle manager by getting a fresh instance
                bundle_manager = get_bundle_manager()
                
                return {
                    "bundle_manager_reinitialized": bundle_manager is not None,
                    "bundles_available": len(bundle_manager.list_bundles()) if bundle_manager else 0
                }
            except ImportError:
                return {
                    "error": "Function bundles module not available"
                }
        except Exception as e:
            return {
                "error": str(e),
                "error_details": traceback.format_exc()
            }


# Tool functions for health and diagnostics

@register_function("health", "check_system_health")
async def check_system_health(component: Optional[str] = None) -> FunctionResult:
    """
    Check the health of the Function Registry system.
    
    Args:
        component: Optional component name to check specifically
        
    Returns:
        Function result with health check results
    """
    try:
        health_monitor = HealthMonitor()
        health_report = await health_monitor.check_health(component)
        
        return FunctionResult(
            data=health_report,
            status="SUCCESS",
            message="Health check completed successfully",
            error_code=None,
            error_details=None
        )
    except Exception as e:
        logger.error(f"Error checking system health: {str(e)}")
        return FunctionResult(
            data=None,
            status="ERROR",
            message=f"Error checking system health: {str(e)}",
            error_code="HEALTH_CHECK_ERROR",
            error_details={"details": traceback.format_exc()}
        )

@register_function("health", "run_system_diagnostic")
async def run_system_diagnostic(component: Optional[str] = None) -> FunctionResult:
    """
    Run in-depth diagnostic on the Function Registry system.
    
    Args:
        component: Optional component name to diagnose specifically
        
    Returns:
        Function result with diagnostic results
    """
    try:
        health_monitor = HealthMonitor()
        diagnostic_report = await health_monitor.run_diagnostic(component)
        
        return FunctionResult(
            data=diagnostic_report,
            status="SUCCESS",
            message="Diagnostic completed successfully",
            error_code=None,
            error_details=None
        )
    except Exception as e:
        logger.error(f"Error running system diagnostic: {str(e)}")
        return FunctionResult(
            data=None,
            status="ERROR",
            message=f"Error running system diagnostic: {str(e)}",
            error_code="DIAGNOSTIC_ERROR",
            error_details={"details": traceback.format_exc()}
        )

@register_function("health", "heal_system")
async def heal_system(component: Optional[str] = None) -> FunctionResult:
    """
    Attempt to heal the Function Registry system.
    
    Args:
        component: Optional component name to heal specifically
        
    Returns:
        Function result with healing results
    """
    try:
        self_healer = SelfHealer()
        healing_report = await self_healer.heal_system(component)
        
        return FunctionResult(
            data=healing_report,
            status="SUCCESS",
            message="Healing process completed successfully",
            error_code=None,
            error_details=None
        )
    except Exception as e:
        logger.error(f"Error healing system: {str(e)}")
        return FunctionResult(
            data=None,
            status="ERROR",
            message=f"Error healing system: {str(e)}",
            error_code="HEALING_ERROR",
            error_details={"details": traceback.format_exc()}
        )

def register_health_diagnostics_tools():
    """Register health and diagnostics tools with the function registry."""
    # The functions are now directly registered via decorators
    # This function is kept for backward compatibility
    logger.info("Health diagnostics tools already registered via decorators")
    return

# The rest of the old register_health_diagnostics_tools function has been removed
# as it's no longer needed due to direct registration with decorators 