"""
Utility functions for the MCP Graph Memory system.
"""

import json
import uuid
from typing import Any, Dict

def dict_to_json(data: Dict[str, Any]) -> str:
    """
    Convert a dictionary to a JSON string.
    
    Args:
        data: The dictionary to convert
        
    Returns:
        A JSON string representation of the dictionary
    """
    return json.dumps(data, default=str)

def generate_id(prefix: str = "") -> str:
    """
    Generate a unique ID with an optional prefix.
    
    Args:
        prefix: Optional prefix for the ID (e.g., "les" for lesson, "obs" for observation)
    
    Returns:
        A unique ID string with optional prefix
    """
    unique_id = str(uuid.uuid4())
    if prefix:
        return f"{prefix}_{unique_id}"
    return unique_id 