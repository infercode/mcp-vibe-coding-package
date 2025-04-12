import json
from typing import Any, Dict, Union

def parse_response(response: Union[str, Dict, Any]) -> Dict[str, Any]:
    """
    Parse response from GraphMemoryManager into a proper Dict.
    
    Args:
        response: Response from GraphMemoryManager (string, dict, or other)
        
    Returns:
        Dict representation of the response
    """
    if isinstance(response, str):
        try:
            return json.loads(response)
        except json.JSONDecodeError:
            return {"result": response}
    elif isinstance(response, dict):
        return response
    else:
        return {"result": response} 