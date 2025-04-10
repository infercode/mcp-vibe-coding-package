from fastapi import Depends, Request
from typing import Optional
from src.graph_memory import GraphMemoryManager
from src.logger import get_logger, LogLevel
from .config import Settings, get_settings

# Initialize logger
logger = get_logger()
logger.set_level(LogLevel.DEBUG)

# Helper dependency to extract client ID from request
def get_client_id(request: Request, settings: Settings = Depends(get_settings)) -> str:
    """Extract appropriate client ID from request path parameters and settings."""
    effective_client_id = settings.client_id
    
    if settings.use_project_as_client:
        # Try to get project_id from path parameters
        project_id = request.path_params.get('project_id')
        
        # For project-related endpoints that don't have project_id in path
        if not project_id:
            # Look for lesson_id and other parameters that might be useful
            lesson_id = request.path_params.get('lesson_id')
            if lesson_id:
                effective_client_id = f"lesson-{lesson_id}"
        else:
            effective_client_id = f"project-{project_id}"
    
    return effective_client_id

def get_memory_manager(client_id: str = Depends(get_client_id), settings: Settings = Depends(get_settings)):
    """
    Dependency for getting GraphMemoryManager instance with settings.
    
    Uses client_id from the request path parameters when available.
    """
    manager = GraphMemoryManager(
        logger=logger,
        neo4j_uri=settings.neo4j_uri,
        neo4j_username=settings.neo4j_user,
        neo4j_password=settings.neo4j_password,
        database=settings.neo4j_database,
        embedding_api_key=settings.embedding_api_key,
        embedding_model=settings.embedding_model
    )
    
    # Initialize with client ID
    manager.initialize(client_id=client_id)
    
    try:
        yield manager
    finally:
        manager.close() 