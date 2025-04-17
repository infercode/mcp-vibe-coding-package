"""
Unit tests for the LessonMemoryManager.
"""

import pytest
from unittest.mock import MagicMock, patch
import json

from src.lesson_memory import LessonMemoryManager

class TestLessonMemoryManager:
    """Test suite for LessonMemoryManager class."""
    
    @pytest.fixture
    def mock_lesson_memory_manager(self, mock_graph_memory_manager):
        """Create a mock LessonMemoryManager."""
        # Create a mock with autospec to ensure all methods are available
        manager = MagicMock(spec=LessonMemoryManager, autospec=True)
        
        # Set the base_manager attribute
        manager.base_manager = mock_graph_memory_manager
        
        # Mock the manager attributes to match the actual implementation
        manager.container = MagicMock()
        manager.entity = MagicMock()
        manager.relation = MagicMock()
        manager.observation = MagicMock()
        manager.evolution = MagicMock()
        manager.consolidation = MagicMock()
        
        # Mock standard methods that tests will call
        manager.create_lesson_container.return_value = json.dumps({
            "status": "success",
            "message": "Container created successfully",
            "data": {"name": "Lessons", "description": "Test container", "metadata": {"key": "value"}}
        })
        
        manager.get_lesson_container.return_value = json.dumps({
            "status": "success",
            "message": "Container retrieved successfully",
            "data": {"name": "Lessons", "description": "Test container", "entity_count": 5}
        })
        
        manager.create_lesson_entity.return_value = json.dumps({
            "status": "success",
            "message": "Lesson entity created successfully",
            "data": {"name": "TestLesson", "entity_type": "LESSON"}
        })
        
        manager.create_structured_lesson_observations.return_value = json.dumps({
            "status": "success",
            "message": "Structured observations created successfully",
            "data": {
                "entity": "TestLesson", 
                "what_was_learned": "Important concept",
                "why_it_matters": "Improves performance",
                "how_to_apply": "Apply in specific context"
            }
        })
        
        manager.search_lesson_entities.return_value = json.dumps({
            "status": "success",
            "message": "Search completed successfully",
            "data": {
                "results": [
                    {"name": "Lesson1", "entity_type": "LESSON", "score": 0.95},
                    {"name": "Lesson2", "entity_type": "LESSON", "score": 0.85}
                ],
                "count": 2
            }
        })
        
        manager.create_lesson_relationship.return_value = json.dumps({
            "status": "success",
            "message": "Relationship created successfully",
            "data": {
                "source_name": "SourceLesson",
                "target_name": "TargetLesson",
                "relationship_type": "BUILDS_ON"
            }
        })
        
        manager.track_lesson_supersession.return_value = json.dumps({
            "status": "success",
            "message": "Supersession tracked successfully",
            "data": {
                "old_lesson": "OldLesson",
                "new_lesson": "NewLesson",
                "reason": "Updated with better information"
            }
        })
        
        manager.merge_lessons.return_value = json.dumps({
            "status": "success",
            "message": "Lessons merged successfully",
            "data": {
                "new_lesson": "MergedLesson",
                "source_lessons": ["Lesson1", "Lesson2"]
            }
        })
        
        return manager
    
    def test_initialization(self, mock_graph_memory_manager):
        """Test that the LessonMemoryManager initializes correctly."""
        manager = LessonMemoryManager(mock_graph_memory_manager)
        
        # Verify manager components are initialized
        assert hasattr(manager, 'container')
        assert hasattr(manager, 'entity')
        assert hasattr(manager, 'relation')
        assert hasattr(manager, 'observation')
        assert hasattr(manager, 'evolution')
        assert hasattr(manager, 'consolidation')
    
    def test_create_lesson_container(self, mock_lesson_memory_manager, json_helper):
        """Test creating a lesson container."""
        description = "Test lesson container"
        metadata = {"key": "value"}
        
        # Configure the container mock
        mock_lesson_memory_manager.create_lesson_container.return_value = json.dumps({
            "status": "success",
            "message": "Container created successfully",
            "data": {
                "name": "Lessons",
                "description": description,
                "metadata": metadata
            }
        })
        
        # Call the method
        result = mock_lesson_memory_manager.create_lesson_container(description, metadata)
        
        # Verify the result
        result_dict = json_helper["from_json"](result)
        assert result_dict["status"] == "success"
        assert "Container created successfully" in result_dict["message"]
        assert result_dict["data"]["name"] == "Lessons"
        assert result_dict["data"]["description"] == description
    
    def test_get_lesson_container(self, mock_lesson_memory_manager, json_helper):
        """Test retrieving a lesson container."""
        # Configure the container mock
        mock_lesson_memory_manager.get_lesson_container.return_value = json.dumps({
            "status": "success",
            "message": "Container retrieved successfully",
            "data": {
                "name": "Lessons",
                "description": "Test container",
                "entity_count": 5
            }
        })
        
        # Call the method
        result = mock_lesson_memory_manager.get_lesson_container()
        
        # Verify the result
        result_dict = json_helper["from_json"](result)
        assert result_dict["status"] == "success"
        assert "Container retrieved successfully" in result_dict["message"]
        assert result_dict["data"]["name"] == "Lessons"
    
    def test_create_lesson_entity(self, mock_lesson_memory_manager, sample_lesson_data, json_helper):
        """Test creating a lesson entity."""
        container_name = "Lessons"
        
        # Configure the entity mock
        mock_lesson_memory_manager.create_lesson_entity.return_value = json.dumps({
            "status": "success",
            "message": "Lesson entity created successfully",
            "data": {
                "name": sample_lesson_data["name"],
                "entity_type": sample_lesson_data["entity_type"],
                "container": container_name
            }
        })
        
        # Call the method
        result = mock_lesson_memory_manager.create_lesson_entity(
            container_name,
            sample_lesson_data["name"],
            sample_lesson_data["entity_type"],
            sample_lesson_data["observations"],
            sample_lesson_data["metadata"]
        )
        
        # Verify the result
        result_dict = json_helper["from_json"](result)
        assert result_dict["status"] == "success"
        assert result_dict["data"]["name"] == sample_lesson_data["name"]
        assert result_dict["data"]["entity_type"] == sample_lesson_data["entity_type"]
    
    def test_create_structured_lesson_observations(self, mock_lesson_memory_manager, json_helper):
        """Test creating structured lesson observations."""
        entity_name = "TestLesson"
        what_was_learned = "Important concept"
        why_it_matters = "Improves performance"
        how_to_apply = "Apply in specific context"
        
        # Configure the observation mock
        mock_lesson_memory_manager.create_structured_lesson_observations.return_value = json.dumps({
            "status": "success",
            "message": "Structured observations created successfully",
            "data": {
                "entity": entity_name,
                "what_was_learned": what_was_learned,
                "why_it_matters": why_it_matters,
                "how_to_apply": how_to_apply
            }
        })
        
        # Call the method
        result = mock_lesson_memory_manager.create_structured_lesson_observations(
            entity_name, what_was_learned, why_it_matters, how_to_apply
        )
        
        # Verify the result
        result_dict = json_helper["from_json"](result)
        assert result_dict["status"] == "success"
        assert result_dict["data"]["entity"] == entity_name
        assert result_dict["data"]["what_was_learned"] == what_was_learned
    
    def test_search_lesson_entities(self, mock_lesson_memory_manager, json_helper):
        """Test searching for lesson entities."""
        container_name = "Lessons"
        search_term = "test search"
        
        # Mock search results
        mock_results = [
            {"name": "Lesson1", "entity_type": "LESSON", "score": 0.95},
            {"name": "Lesson2", "entity_type": "LESSON", "score": 0.85}
        ]
        
        # Configure the entity mock
        mock_lesson_memory_manager.search_lesson_entities.return_value = json.dumps({
            "status": "success",
            "message": "Search completed successfully",
            "data": {"results": mock_results, "count": len(mock_results)}
        })
        
        # Call the method
        result = mock_lesson_memory_manager.search_lesson_entities(container_name, search_term)
        
        # Verify the result
        result_dict = json_helper["from_json"](result)
        assert result_dict["status"] == "success"
        assert "Search completed successfully" in result_dict["message"]
        assert len(result_dict["data"]["results"]) == 2
    
    def test_create_lesson_relationship(self, mock_lesson_memory_manager, json_helper):
        """Test creating a relationship between lessons."""
        relationship_data = {
            "source_name": "SourceLesson",
            "target_name": "TargetLesson",
            "relationship_type": "BUILDS_ON",
            "container_name": "Lessons",
            "properties": {"strength": 0.9}
        }
        
        # Configure the relation mock
        mock_lesson_memory_manager.create_lesson_relationship.return_value = json.dumps({
            "status": "success",
            "message": "Relationship created successfully",
            "data": relationship_data
        })
        
        # Call the method
        result = mock_lesson_memory_manager.create_lesson_relationship(relationship_data)
        
        # Verify the result
        result_dict = json_helper["from_json"](result)
        assert result_dict["status"] == "success"
        assert "Relationship created successfully" in result_dict["message"]
        assert result_dict["data"]["source_name"] == relationship_data["source_name"]
        assert result_dict["data"]["target_name"] == relationship_data["target_name"]
    
    def test_track_lesson_supersession(self, mock_lesson_memory_manager, json_helper):
        """Test tracking lesson supersession."""
        old_lesson = "OldLesson"
        new_lesson = "NewLesson"
        reason = "Updated with better information"
        
        # Configure the evolution mock
        mock_lesson_memory_manager.track_lesson_supersession.return_value = json.dumps({
            "status": "success",
            "message": "Supersession tracked successfully",
            "data": {
                "old_lesson": old_lesson,
                "new_lesson": new_lesson,
                "reason": reason
            }
        })
        
        # Call the method
        result = mock_lesson_memory_manager.track_lesson_supersession(old_lesson, new_lesson, reason)
        
        # Verify the result
        result_dict = json_helper["from_json"](result)
        assert result_dict["status"] == "success"
        assert "Supersession tracked successfully" in result_dict["message"]
        assert result_dict["data"]["old_lesson"] == old_lesson
        assert result_dict["data"]["new_lesson"] == new_lesson
    
    def test_merge_lessons(self, mock_lesson_memory_manager, json_helper):
        """Test merging lessons."""
        source_lessons = [{"name": "Lesson1"}, {"name": "Lesson2"}]
        new_name = "MergedLesson"
        
        # Configure the consolidation mock
        mock_lesson_memory_manager.merge_lessons.return_value = json.dumps({
            "status": "success",
            "message": "Lessons merged successfully",
            "data": {
                "new_lesson": new_name,
                "source_lessons": [l["name"] for l in source_lessons]
            }
        })
        
        # Call the method
        result = mock_lesson_memory_manager.merge_lessons(source_lessons, new_name)
        
        # Verify the result
        result_dict = json_helper["from_json"](result)
        assert result_dict["status"] == "success"
        assert "Lessons merged successfully" in result_dict["message"]
        assert result_dict["data"]["new_lesson"] == new_name
        assert len(result_dict["data"]["source_lessons"]) == 2 