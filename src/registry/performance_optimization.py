#!/usr/bin/env python3
"""
Performance Optimization

This module provides performance optimizations for the Function Registry,
including result caching, batch operations, and parameter serialization.
"""

import json
import time
import asyncio
import hashlib
import inspect
from typing import Dict, List, Any, Optional, Union, Callable, Tuple, Set
from functools import wraps
from datetime import datetime, timedelta

from src.registry.registry_manager import get_registry
from src.registry.function_models import FunctionResult
from src.logger import get_logger

logger = get_logger()

class ResultCache:
    """
    Caches function results to avoid repeated execution of identical calls.
    
    Features:
    - Time-based cache expiration
    - Size-limited cache with LRU eviction
    - Cache key generation from function name and parameters
    - Configurable per-function cache settings
    """
    
    def __init__(self, max_size: int = 1000, default_ttl: int = 300):
        """
        Initialize the result cache.
        
        Args:
            max_size: Maximum number of cache entries
            default_ttl: Default time-to-live in seconds for cache entries
        """
        self.cache = {}  # Maps cache keys to entries
        self.access_times = {}  # Maps cache keys to last access time
        self.max_size = max_size
        self.default_ttl = default_ttl
        self.function_settings = {}  # Maps function names to cache settings
        
    def set_function_cache_settings(self, function_name: str, ttl: int, cacheable: bool = True) -> None:
        """
        Set cache settings for a specific function.
        
        Args:
            function_name: Name of the function
            ttl: Time-to-live in seconds for this function's results
            cacheable: Whether this function's results should be cached
        """
        self.function_settings[function_name] = {
            "ttl": ttl,
            "cacheable": cacheable
        }
        
    def set_tool_cache_settings(self, function_name: str, ttl: int, cacheable: bool = True) -> None:
        """
        Set cache settings for a specific tool.
        
        Args:
            function_name: Name of the tool
            ttl: Time-to-live in seconds for this tool's results
            cacheable: Whether this tool's results should be cached
        """
        return self.set_function_cache_settings(function_name, ttl, cacheable)
        
    def get_cache_key(self, function_name: str, params: Dict[str, Any]) -> str:
        """
        Generate a cache key for a function call.
        
        Args:
            function_name: Name of the function
            params: Function parameters
            
        Returns:
            Cache key string
        """
        # Serialize parameters to a consistent string representation
        param_str = json.dumps(params, sort_keys=True)
        
        # Create a hash from the function name and parameters
        key = hashlib.md5(f"{function_name}:{param_str}".encode()).hexdigest()
        return key
        
    def get(self, function_name: str, params: Dict[str, Any]) -> Optional[FunctionResult]:
        """
        Get a cached result for a function call if available.
        
        Args:
            function_name: Name of the function
            params: Function parameters
            
        Returns:
            Cached function result or None if not found
        """
        # Check if function is cacheable
        settings = self.function_settings.get(function_name, {"cacheable": True, "ttl": self.default_ttl})
        if not settings["cacheable"]:
            return None
            
        # Generate cache key
        key = self.get_cache_key(function_name, params)
        
        # Check if key exists in cache
        if key not in self.cache:
            return None
            
        # Get cache entry
        entry = self.cache[key]
        
        # Check if entry has expired
        ttl = settings["ttl"]
        if time.time() - entry["timestamp"] > ttl:
            # Remove expired entry
            del self.cache[key]
            if key in self.access_times:
                del self.access_times[key]
            return None
            
        # Update access time
        self.access_times[key] = time.time()
        
        # Return cached result
        return entry["result"]
        
    def set(self, function_name: str, params: Dict[str, Any], result: FunctionResult) -> None:
        """
        Cache a function result.
        
        Args:
            function_name: Name of the function
            params: Function parameters
            result: Function result to cache
        """
        # Check if function is cacheable
        settings = self.function_settings.get(function_name, {"cacheable": True, "ttl": self.default_ttl})
        if not settings["cacheable"]:
            return
            
        # Don't cache error results
        if result.status != "success":
            return
            
        # Generate cache key
        key = self.get_cache_key(function_name, params)
        
        # Check if cache is full and needs eviction
        if len(self.cache) >= self.max_size:
            self._evict_lru_entry()
            
        # Add to cache
        self.cache[key] = {
            "result": result,
            "timestamp": time.time()
        }
        self.access_times[key] = time.time()
        
    def invalidate(self, function_name: Optional[str] = None) -> int:
        """
        Invalidate cache entries for a function or all functions.
        
        Args:
            function_name: Name of the function or None to invalidate all
            
        Returns:
            Number of invalidated entries
        """
        if function_name is None:
            # Invalidate all entries
            count = len(self.cache)
            self.cache = {}
            self.access_times = {}
            return count
            
        # Invalidate entries for a specific function
        keys_to_remove = []
        
        # Find all keys for the function
        for key in list(self.cache.keys()):
            # The actual function name extraction from the key would be complex
            # For simplicity, we can re-generate keys for all cache entries
            # and check if they match the current function
            if key.startswith(function_name):
                keys_to_remove.append(key)
                
        # Remove the keys
        for key in keys_to_remove:
            del self.cache[key]
            if key in self.access_times:
                del self.access_times[key]
                
        return len(keys_to_remove)
        
    def clear(self) -> None:
        """Clear the entire cache."""
        self.cache = {}
        self.access_times = {}
        
    def _evict_lru_entry(self) -> None:
        """Evict the least recently used cache entry."""
        if not self.access_times:
            return
            
        # Find the least recently used key
        lru_key = min(self.access_times.items(), key=lambda item: item[1])[0]
        
        # Remove it from cache
        if lru_key in self.cache:
            del self.cache[lru_key]
        del self.access_times[lru_key]


class BatchProcessor:
    """
    Processes multiple function calls in a single batch operation.
    
    Features:
    - Batch execution of multiple function calls
    - Parallel execution for independent calls
    - Sequential execution for dependent calls
    - Result mapping to match original call order
    """
    
    def __init__(self):
        """Initialize the batch processor."""
        self.registry = get_registry()
        
    async def execute_batch(self, 
                          batch: List[Dict[str, Any]], 
                          parallel: bool = True,
                          use_cache: bool = True) -> List[Dict[str, Any]]:
        """
        Execute a batch of function calls.
        
        Args:
            batch: List of function call specifications
            parallel: Whether to execute calls in parallel when possible
            use_cache: Whether to use the result cache
            
        Returns:
            List of function results in the same order as the batch
        """
        if not batch:
            return []
            
        # Check if we have dependencies between calls
        has_dependencies = any("depends_on" in call for call in batch)
        
        # If we have dependencies or parallel execution is disabled, use sequential execution
        if has_dependencies or not parallel:
            return await self._execute_sequential(batch, use_cache)
        else:
            return await self._execute_parallel(batch, use_cache)
    
    async def _execute_sequential(self, 
                                batch: List[Dict[str, Any]], 
                                use_cache: bool) -> List[Dict[str, Any]]:
        """Execute batch calls sequentially with dependency handling."""
        results = {}
        result_list = []
        
        # First, assign each call an ID if not provided
        for i, call in enumerate(batch):
            if "id" not in call:
                call["id"] = f"call_{i}"
                
        # Build dependency graph
        dependencies = {call["id"]: call.get("depends_on", []) for call in batch}
        
        # Find calls with no dependencies
        ready = [call["id"] for call in batch if not call.get("depends_on")]
        remaining = {call["id"] for call in batch} - set(ready)
        
        # Process calls in dependency order
        while ready or remaining:
            if not ready:
                # If we're stuck with circular dependencies, execute one of the remaining calls
                ready = [next(iter(remaining))]
                
            # Get the next call to execute
            current_id = ready.pop(0)
            remaining.discard(current_id)
            
            # Find the call specification
            call_spec = next(call for call in batch if call["id"] == current_id)
            
            # Execute the call
            result = await self._execute_single_call(call_spec, results, use_cache)
            
            # Store the result
            results[current_id] = result
            
            # Find calls that now have all dependencies satisfied
            for call_id in list(remaining):
                deps = dependencies[call_id]
                if all(dep in results for dep in deps):
                    ready.append(call_id)
                    remaining.discard(call_id)
        
        # Build the result list in the original order
        for call in batch:
            result_list.append(results[call["id"]])
            
        return result_list
    
    async def _execute_parallel(self, 
                              batch: List[Dict[str, Any]], 
                              use_cache: bool) -> List[Dict[str, Any]]:
        """Execute batch calls in parallel."""
        tasks = []
        
        # Create a task for each call
        for call in batch:
            task = asyncio.create_task(self._execute_single_call(call, {}, use_cache))
            tasks.append(task)
            
        # Wait for all tasks to complete
        results = await asyncio.gather(*tasks)
        return results
    
    async def _execute_single_call(self, 
                                 call: Dict[str, Any], 
                                 previous_results: Dict[str, Any], 
                                 use_cache: bool) -> Dict[str, Any]:
        """Execute a single function call in a batch."""
        function_name = call.get("function")
        params = call.get("params", {})
        
        if not function_name:
            return {
                "status": "error",
                "error": "Missing function name",
                "data": None
            }
            
        # Apply parameter references if needed
        if "param_refs" in call and previous_results:
            params = self._resolve_param_references(params, call["param_refs"], previous_results)
            
        # Execute the function
        try:
            # Check if this call should use the cache
            cache = get_result_cache() if use_cache else None
            
            # Try to get cached result
            result = None
            if cache:
                result = cache.get(function_name, params)
                
            if result is None:
                # Execute the function
                result = await self.registry.execute(function_name, **params)
                
                # Cache the result if applicable
                if cache and result.status == "success":
                    cache.set(function_name, params, result)
                    
            # Convert result to dict
            result_dict = json.loads(result.to_json())
            
            # Add call ID if provided
            if "id" in call:
                result_dict["call_id"] = call["id"]
                
            return result_dict
        except Exception as e:
            error_result = {
                "status": "error",
                "error": str(e),
                "data": None
            }
            
            # Add call ID if provided
            if "id" in call:
                error_result["call_id"] = call["id"]
                
            return error_result
    
    def _resolve_param_references(self, 
                                params: Dict[str, Any], 
                                param_refs: Dict[str, str], 
                                previous_results: Dict[str, Any]) -> Dict[str, Any]:
        """Resolve parameter references to previous results."""
        resolved_params = params.copy()
        
        for param_name, ref_path in param_refs.items():
            # Parse the reference path (format: "call_id.result_path")
            parts = ref_path.split('.', 1)
            if len(parts) != 2:
                continue
                
            call_id, result_path = parts
            
            # Check if the referenced call exists
            if call_id not in previous_results:
                continue
                
            # Get the referenced result
            ref_result = previous_results[call_id]
            
            # Extract the value from the result
            value = self._extract_path_value(ref_result, result_path)
            
            # Set the parameter value
            resolved_params[param_name] = value
            
        return resolved_params
    
    def _extract_path_value(self, data: Dict[str, Any], path: str) -> Any:
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


class ParameterSerializer:
    """
    Optimizes parameter serialization and deserialization.
    
    Features:
    - Efficient parameter serialization for storage and transmission
    - Schema-based serialization for consistent formats
    - Custom serializers for specific parameter types
    """
    
    def __init__(self):
        """Initialize the parameter serializer."""
        self.serializers = {}
        self.deserializers = {}
        
        # Register default serializers
        self.register_serializer("date", self._serialize_date, self._deserialize_date)
        self.register_serializer("datetime", self._serialize_datetime, self._deserialize_datetime)
        self.register_serializer("binary", self._serialize_binary, self._deserialize_binary)
        self.register_serializer("set", self._serialize_set, self._deserialize_set)
        
    def register_serializer(self, 
                          type_name: str, 
                          serializer: Callable[[Any], Any], 
                          deserializer: Callable[[Any], Any]) -> None:
        """
        Register a custom serializer for a parameter type.
        
        Args:
            type_name: Name of the parameter type
            serializer: Function to convert from native type to serializable form
            deserializer: Function to convert from serialized form to native type
        """
        self.serializers[type_name] = serializer
        self.deserializers[type_name] = deserializer
        
    def serialize_parameters(self, 
                           parameters: Dict[str, Any], 
                           parameter_types: Dict[str, str]) -> Dict[str, Any]:
        """
        Serialize parameters according to their types.
        
        Args:
            parameters: Parameters to serialize
            parameter_types: Parameter type mappings
            
        Returns:
            Serialized parameters dictionary
        """
        serialized = {}
        
        for name, value in parameters.items():
            if value is None:
                serialized[name] = None
                continue
                
            # Get the parameter type
            param_type = parameter_types.get(name, "unknown")
            
            # Check if we have a serializer for this type
            if param_type in self.serializers:
                serialized[name] = self.serializers[param_type](value)
            else:
                # Use default serialization
                serialized[name] = value
                
        return serialized
    
    def deserialize_parameters(self, 
                             serialized: Dict[str, Any], 
                             parameter_types: Dict[str, str]) -> Dict[str, Any]:
        """
        Deserialize parameters according to their types.
        
        Args:
            serialized: Serialized parameters
            parameter_types: Parameter type mappings
            
        Returns:
            Deserialized parameters dictionary
        """
        deserialized = {}
        
        for name, value in serialized.items():
            if value is None:
                deserialized[name] = None
                continue
                
            # Get the parameter type
            param_type = parameter_types.get(name, "unknown")
            
            # Check if we have a deserializer for this type
            if param_type in self.deserializers:
                deserialized[name] = self.deserializers[param_type](value)
            else:
                # Use as-is
                deserialized[name] = value
                
        return deserialized
    
    def get_parameter_types(self, function_metadata: Dict[str, Any]) -> Dict[str, str]:
        """
        Extract parameter type information from function metadata.
        
        Args:
            function_metadata: Function metadata from registry
            
        Returns:
            Mapping of parameter names to type names
        """
        types = {}
        
        if "parameters" in function_metadata:
            for name, param_info in function_metadata["parameters"].items():
                # Extract type information
                types[name] = param_info.get("type", "unknown")
                
        return types
    
    # Default serializers and deserializers
    
    def _serialize_date(self, value: Any) -> str:
        """Serialize a date object to ISO format string."""
        from datetime import date
        if isinstance(value, date):
            return value.isoformat()
        return str(value)
    
    def _deserialize_date(self, value: str) -> Any:
        """Deserialize an ISO format string to a date object."""
        from datetime import datetime
        try:
            return datetime.fromisoformat(value).date()
        except:
            return value
    
    def _serialize_datetime(self, value: Any) -> str:
        """Serialize a datetime object to ISO format string."""
        from datetime import datetime
        if isinstance(value, datetime):
            return value.isoformat()
        return str(value)
    
    def _deserialize_datetime(self, value: str) -> Any:
        """Deserialize an ISO format string to a datetime object."""
        from datetime import datetime
        try:
            return datetime.fromisoformat(value)
        except:
            return value
    
    def _serialize_binary(self, value: Any) -> str:
        """Serialize binary data to base64 string."""
        import base64
        if isinstance(value, bytes):
            return base64.b64encode(value).decode('ascii')
        return str(value)
    
    def _deserialize_binary(self, value: str) -> Any:
        """Deserialize base64 string to binary data."""
        import base64
        try:
            return base64.b64decode(value)
        except:
            return value
    
    def _serialize_set(self, value: Any) -> List[Any]:
        """Serialize a set to a list."""
        if isinstance(value, set):
            return list(value)
        return value
    
    def _deserialize_set(self, value: List[Any]) -> Any:
        """Deserialize a list to a set."""
        if isinstance(value, list):
            return set(value)
        return value


# Singleton instances
_cache = None
_batch_processor = None
_serializer = None

def get_result_cache() -> ResultCache:
    """Get the global result cache instance."""
    global _cache
    if _cache is None:
        _cache = ResultCache()
    return _cache

def get_batch_processor() -> BatchProcessor:
    """Get the global batch processor instance."""
    global _batch_processor
    if _batch_processor is None:
        _batch_processor = BatchProcessor()
    return _batch_processor

def get_parameter_serializer() -> ParameterSerializer:
    """Get the global parameter serializer instance."""
    global _serializer
    if _serializer is None:
        _serializer = ParameterSerializer()
    return _serializer


# Decorator for enabling result caching on functions
def cacheable(ttl: int = 300):
    """
    Decorator to mark a function as cacheable with a specific TTL.
    
    Args:
        ttl: Time-to-live in seconds
    """
    def decorator(func):
        # Get the full function name
        if hasattr(func, "__qualname__"):
            func_name = func.__qualname__
        else:
            func_name = func.__name__
            
        # Register with cache
        cache = get_result_cache()
        cache.set_function_cache_settings(func_name, ttl, cacheable=True)
        
        return func
    return decorator 