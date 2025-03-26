from typing import Dict, List, TypedDict


class Entity(TypedDict):
    """Entity representation in the knowledge graph."""
    name: str
    entityType: str
    observations: List[str]


class Relation(TypedDict):
    """Relation between entities in the knowledge graph."""
    from_: str  # Renamed for Python compatibility, will be mapped to "from" when needed
    to: str
    relationType: str


class Observation(TypedDict):
    """Observation associated with an entity."""
    entityName: str
    contents: List[str]


class KnowledgeGraph(TypedDict):
    """Knowledge graph representation."""
    entities: List[Entity]
    relations: List[Relation]


class ObservationDeletion(TypedDict):
    """Represents an observation to be deleted."""
    entityName: str
    contents: List[str]


class SearchResult(TypedDict):
    """Search result representation."""
    entity: Entity
    score: float


class SearchResponse(TypedDict):
    """Response from a search operation."""
    results: List[SearchResult]
    query: str 