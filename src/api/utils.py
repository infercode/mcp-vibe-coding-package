import json
from typing import Dict, Any

def parse_response(response: str) -> Dict[str, Any]:
    """Parse JSON string response to dictionary."""
    try:
        return json.loads(response)
    except json.JSONDecodeError:
        return {"result": response} 