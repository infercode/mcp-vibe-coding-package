#!/usr/bin/env python3
"""
JSON Utility Functions

This module provides utility functions for working with JSON data.
"""

import json
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