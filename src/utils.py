import json
import random
import string
from typing import Any, Dict


def extract_error(error: Exception) -> Dict[str, str]:
    """
    Extract error information from an exception.
    
    Args:
        error: The exception to extract information from
        
    Returns:
        A dictionary containing error message
    """
    if isinstance(error, Exception):
        return {
            "message": str(error)
        }
    return {
        "message": "Unknown error"
    }


def generate_id() -> str:
    """
    Generate a unique random ID.
    
    Returns:
        A unique ID string
    """
    chars = string.ascii_lowercase + string.digits
    return ''.join(random.choice(chars) for _ in range(12))


def dict_to_json(data: Dict[str, Any]) -> str:
    """
    Convert a dictionary to a JSON string.
    
    Args:
        data: The dictionary to convert
        
    Returns:
        A JSON string representation of the dictionary
    """
    return json.dumps(data, indent=2) 