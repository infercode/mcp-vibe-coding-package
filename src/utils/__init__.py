"""
Utility functions for the MCP Graph Memory system.
"""

# Import and re-export utility functions to maintain backwards compatibility
from src.utils.common_utils import (
    extract_error,
    generate_id
)

from src.utils.json_utils import (
    dict_to_json
)

from src.utils.neo4j_query_utils import (
    dump_neo4j_nodes,
    is_json_serializable,
    sanitize_query_parameters,
    validate_query,
    safe_execute_validated_query,
    safe_execute_query,
    create_node_query,
    create_match_node_query,
    create_relationship_query,
    build_match_query
)

# For backward compatibility and convenience, explicitly define exports
__all__ = [
    # Common utils
    'extract_error',
    'generate_id',
    
    # JSON utils
    'dict_to_json',
    
    # Neo4j utils
    'dump_neo4j_nodes',
    'is_json_serializable',
    'sanitize_query_parameters',
    'validate_query',
    'safe_execute_validated_query',
    'safe_execute_query',
    'create_node_query',
    'create_match_node_query',
    'create_relationship_query',
    'build_match_query',
] 