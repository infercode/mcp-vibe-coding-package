import json
import random
import string
import uuid
import os
from typing import Any, Dict, Optional


def extract_error(error: Exception) -> str:
    """
    Extract the error message from an exception.
    
    Args:
        error: The exception to extract information from
        
    Returns:
        A string containing the error message
    """
    return str(error)


def generate_id() -> str:
    """
    Generate a unique ID.
    
    Returns:
        A unique ID string
    """
    return str(uuid.uuid4())


def dict_to_json(data: Dict[str, Any]) -> str:
    """
    Convert a dictionary to a JSON string.
    
    Args:
        data: The dictionary to convert
        
    Returns:
        A JSON string representation of the dictionary
    """
    return json.dumps(data) 