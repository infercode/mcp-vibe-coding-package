"""
DEPRECATED: Legacy Graph Memory Manager

This module is deprecated and will be removed in a future version.
Please use the modular implementation from src.graph_memory instead.

Example usage:
```python
from src.graph_memory import GraphMemoryManager

# Create the memory manager
memory_manager = GraphMemoryManager()

# Initialize
memory_manager.initialize()

# Use the memory manager...
```

For documentation on the new API, please refer to the docs/refactoring_plan.md file.
"""

import warnings
import functools
import inspect
from typing import Callable, Any, Dict, List, Optional, Union

# Import the new implementation
from src.graph_memory import GraphMemoryManager as NewGraphMemoryManager

# Display deprecation warning
warnings.warn(
    "The legacy_graph_manager module is deprecated and will be removed in a future version. "
    "Please use src.graph_memory.GraphMemoryManager instead.",
    DeprecationWarning,
    stacklevel=2
)

def deprecated(func: Callable) -> Callable:
    """
    Decorator to mark functions as deprecated.
    """
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        warnings.warn(
            f"Function {func.__name__} is deprecated and will be removed in a future version. "
            f"Please use the equivalent method from src.graph_memory.GraphMemoryManager.",
            DeprecationWarning,
            stacklevel=2
        )
        return func(*args, **kwargs)
    return wrapper

class GraphMemoryManager(NewGraphMemoryManager):
    """
    DEPRECATED: Legacy Graph Memory Manager for Neo4j
    
    Please use src.graph_memory.GraphMemoryManager instead.
    
    This class is maintained for backward compatibility and delegates all operations
    to the new modular implementation.
    """
    
    def __init__(self, *args, **kwargs):
        """
        Initialize the Legacy Graph Memory Manager for Neo4j.
        
        This constructor is deprecated. Please use src.graph_memory.GraphMemoryManager.
        """
        warnings.warn(
            "The GraphMemoryManager class from legacy_graph_manager is deprecated. "
            "Please use src.graph_memory.GraphMemoryManager instead.",
            DeprecationWarning,
            stacklevel=2
        )
        super().__init__(*args, **kwargs)
    
    # Mark all public methods as deprecated
    # We do this by patching the class after definition
    
# Dynamically add deprecation warnings to all public methods
for name, method in inspect.getmembers(GraphMemoryManager, inspect.isfunction):
    if not name.startswith('_'):
        setattr(GraphMemoryManager, name, deprecated(method))
        
# Provide the class as a direct export
__all__ = ['GraphMemoryManager']