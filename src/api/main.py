from typing import List, Optional, Dict, Any, Union
from fastapi import FastAPI, HTTPException, Depends
from pydantic import BaseModel, Field
from src.graph_memory import GraphMemoryManager
from src.logger import get_logger, LogLevel
from src.api.config import Settings, get_settings
from .utils import parse_response
from .routers import core, relations, observations, search, projects, lessons

# Initialize settings
settings = get_settings()

# Initialize FastAPI app
app = FastAPI(
    title=settings.api_title,
    description=settings.api_description,
    version=settings.api_version
)

# Initialize logger
logger = get_logger()
logger.set_level(LogLevel.DEBUG)

# Models for request/response
class Entity(BaseModel):
    name: str = Field(..., description="The name of the entity")
    entity_type: str = Field(..., description="The type of the entity")
    observations: List[str] = Field(default=[], description="List of observations about the entity")

class Relation(BaseModel):
    from_entity: str = Field(..., description="Source entity name")
    to_entity: str = Field(..., description="Target entity name")
    relation_type: str = Field(..., description="Type of relation")
    properties: Optional[Dict[str, Any]] = Field(default=None, description="Additional properties")

class Observation(BaseModel):
    entity: str = Field(..., description="Entity name")
    content: str = Field(..., description="Observation content")
    observation_type: Optional[str] = None

class SearchQuery(BaseModel):
    query: str = Field(..., description="Search query string")
    limit: int = Field(default=10, description="Maximum number of results")
    semantic: bool = Field(default=True, description="Whether to use semantic search")

# Dependency for getting GraphMemoryManager instance with settings
def get_memory_manager(settings: Settings = Depends(get_settings)):
    manager = GraphMemoryManager(
        logger=logger,
        neo4j_uri=settings.neo4j_uri,
        neo4j_username=settings.neo4j_user,
        neo4j_password=settings.neo4j_password,
        database=settings.neo4j_database,
        embedding_api_key=settings.embedding_api_key,
        embedding_model=settings.embedding_model
    )
    
    # Initialize with default client ID
    initialized = manager.initialize(client_id=settings.client_id)
    
    if not initialized:
        logger.error(f"Failed to initialize GraphMemoryManager for default client")
    
    try:
        yield manager
    finally:
        manager.close()

# Include routers
app.include_router(core.router)
app.include_router(relations.router)
app.include_router(observations.router)
app.include_router(search.router)
app.include_router(projects.router)
app.include_router(lessons.router)

# Root endpoint
@app.get("/")
async def root():
    """Root endpoint with API information."""
    return {
        "title": settings.api_title,
        "version": settings.api_version,
        "description": settings.api_description
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        app,
        host=settings.host,
        port=settings.port
    ) 