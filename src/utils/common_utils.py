#!/usr/bin/env python3
"""
Common Utility Functions

This module provides general utility functions used throughout the codebase.
"""

import uuid
from typing import Any, Dict, Optional
import json

def extract_error(error: Exception) -> str:
    """
    Extract the error message from an exception.
    
    Args:
        error: The exception to extract information from
        
    Returns:
        A string containing the error message
    """
    return f"{type(error).__name__}: {str(error)}"


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