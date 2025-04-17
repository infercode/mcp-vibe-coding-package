"""
Unit tests for the types module.
"""

import pytest
from typing import Dict, List, Any, cast

from src.types import (
    Entity, Relation, Observation, KnowledgeGraph, 
    ObservationDeletion, SearchResult, SearchResponse
)


class TestTypes:
    """Test suite for types module validation."""
    
    def test_entity_schema(self):
        """Test Entity TypedDict schema validation."""
        # Valid entity
        entity: Entity = {
            "name": "TestEntity",
            "entityType": "TestType",
            "observations": ["Observation 1", "Observation 2"]
        }
        
        # Verify basic properties
        assert entity["name"] == "TestEntity"
        assert entity["entityType"] == "TestType"
        assert len(entity["observations"]) == 2
        
        # Test dictionary conversion
        entity_dict: Dict[str, Any] = dict(entity)
        assert entity_dict["name"] == "TestEntity"
        assert entity_dict["entityType"] == "TestType"
        assert entity_dict["observations"] == ["Observation 1", "Observation 2"]
    
    def test_relation_schema(self):
        """Test Relation TypedDict schema validation."""
        # Valid relation
        relation: Relation = {
            "from_": "SourceEntity",
            "to": "TargetEntity",
            "relationType": "DEPENDS_ON"
        }
        
        # Verify basic properties
        assert relation["from_"] == "SourceEntity"
        assert relation["to"] == "TargetEntity"
        assert relation["relationType"] == "DEPENDS_ON"
        
        # Test dictionary conversion
        relation_dict: Dict[str, Any] = dict(relation)
        assert relation_dict["from_"] == "SourceEntity"
        assert relation_dict["to"] == "TargetEntity"
        assert relation_dict["relationType"] == "DEPENDS_ON"
    
    def test_observation_schema(self):
        """Test Observation TypedDict schema validation."""
        # Valid observation
        observation: Observation = {
            "entityName": "TestEntity",
            "contents": ["Observation content 1", "Observation content 2"]
        }
        
        # Verify basic properties
        assert observation["entityName"] == "TestEntity"
        assert len(observation["contents"]) == 2
        
        # Test dictionary conversion
        observation_dict: Dict[str, Any] = dict(observation)
        assert observation_dict["entityName"] == "TestEntity"
        assert observation_dict["contents"] == ["Observation content 1", "Observation content 2"]
    
    def test_knowledge_graph_schema(self):
        """Test KnowledgeGraph TypedDict schema validation."""
        # Create entities and relations
        entities: List[Entity] = [
            {
                "name": "Entity1",
                "entityType": "Type1",
                "observations": ["Observation 1"]
            },
            {
                "name": "Entity2",
                "entityType": "Type2",
                "observations": ["Observation 2"]
            }
        ]
        
        relations: List[Relation] = [
            {
                "from_": "Entity1",
                "to": "Entity2",
                "relationType": "CONNECTS_TO"
            }
        ]
        
        # Valid knowledge graph
        knowledge_graph: KnowledgeGraph = {
            "entities": entities,
            "relations": relations
        }
        
        # Verify basic properties
        assert len(knowledge_graph["entities"]) == 2
        assert len(knowledge_graph["relations"]) == 1
        
        # Test dictionary conversion
        graph_dict: Dict[str, Any] = dict(knowledge_graph)
        assert len(cast(List[Entity], graph_dict["entities"])) == 2
        assert len(cast(List[Relation], graph_dict["relations"])) == 1
        assert cast(List[Entity], graph_dict["entities"])[0]["name"] == "Entity1"
        assert cast(List[Relation], graph_dict["relations"])[0]["from_"] == "Entity1"
    
    def test_observation_deletion_schema(self):
        """Test ObservationDeletion TypedDict schema validation."""
        # Valid observation deletion
        deletion: ObservationDeletion = {
            "entityName": "TestEntity",
            "contents": ["Observation to delete"]
        }
        
        # Verify basic properties
        assert deletion["entityName"] == "TestEntity"
        assert len(deletion["contents"]) == 1
        
        # Test dictionary conversion
        deletion_dict: Dict[str, Any] = dict(deletion)
        assert deletion_dict["entityName"] == "TestEntity"
        assert deletion_dict["contents"] == ["Observation to delete"]
    
    def test_search_result_schema(self):
        """Test SearchResult TypedDict schema validation."""
        # Valid entity
        entity: Entity = {
            "name": "SearchedEntity",
            "entityType": "TestType",
            "observations": ["Observation 1"]
        }
        
        # Valid search result
        search_result: SearchResult = {
            "entity": entity,
            "score": 0.95
        }
        
        # Verify basic properties
        assert search_result["entity"]["name"] == "SearchedEntity"
        assert search_result["score"] == 0.95
        
        # Test dictionary conversion
        result_dict: Dict[str, Any] = dict(search_result)
        assert cast(Dict[str, Any], result_dict["entity"])["name"] == "SearchedEntity"
        assert result_dict["score"] == 0.95
    
    def test_search_response_schema(self):
        """Test SearchResponse TypedDict schema validation."""
        # Create search results
        entity1: Entity = {
            "name": "Result1",
            "entityType": "Type1",
            "observations": ["Observation 1"]
        }
        
        entity2: Entity = {
            "name": "Result2",
            "entityType": "Type2",
            "observations": ["Observation 2"]
        }
        
        search_results: List[SearchResult] = [
            {
                "entity": entity1,
                "score": 0.95
            },
            {
                "entity": entity2,
                "score": 0.85
            }
        ]
        
        # Valid search response
        search_response: SearchResponse = {
            "results": search_results,
            "query": "test query"
        }
        
        # Verify basic properties
        assert len(search_response["results"]) == 2
        assert search_response["query"] == "test query"
        
        # Test dictionary conversion
        response_dict: Dict[str, Any] = dict(search_response)
        assert len(cast(List[SearchResult], response_dict["results"])) == 2
        assert response_dict["query"] == "test query"
        assert cast(List[SearchResult], response_dict["results"])[0]["entity"]["name"] == "Result1"
        assert cast(List[SearchResult], response_dict["results"])[1]["entity"]["name"] == "Result2" 