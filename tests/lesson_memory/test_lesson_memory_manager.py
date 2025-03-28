import pytest
from unittest.mock import MagicMock, patch, call
import json

from src.lesson_memory import LessonMemoryManager


def test_init(mock_base_manager, mock_entity_manager, mock_relation_manager, mock_observation_manager, mock_logger):
    """Test initialization of LessonMemoryManager."""
    # Create manager
    manager = LessonMemoryManager(
        base_manager=mock_base_manager,
        entity_manager=mock_entity_manager,
        relation_manager=mock_relation_manager,
        observation_manager=mock_observation_manager,
        logger=mock_logger
    )
    
    # Verify manager attributes
    assert manager.base_manager == mock_base_manager
    assert manager.entity_manager == mock_entity_manager
    assert manager.relation_manager == mock_relation_manager
    assert manager.observation_manager == mock_observation_manager
    assert manager.logger == mock_logger


def test_create_lesson_container(mock_base_manager, mock_entity_manager, mock_relation_manager, mock_observation_manager, mock_logger):
    """Test creating a lesson container."""
    # Create manager
    manager = LessonMemoryManager(
        base_manager=mock_base_manager,
        entity_manager=mock_entity_manager,
        relation_manager=mock_relation_manager,
        observation_manager=mock_observation_manager,
        logger=mock_logger
    )
    
    # Mock entity creation
    mock_entity_manager.create_entity.return_value = json.dumps({
        "id": "lesson-123",
        "name": "Test Lesson"
    })
    
    # Create lesson container
    lesson_data = {
        "title": "Test Lesson",
        "description": "This is a test lesson",
        "difficulty": "intermediate",
        "tags": ["test", "python"],
        "estimated_time": "30 minutes",
        "visibility": "public"
    }
    
    result = manager.create_lesson_container(lesson_data)
    
    # Verify result
    result_obj = json.loads(result)
    assert result_obj["status"] == "success"
    assert "id" in result_obj
    
    # Verify entity_manager was called correctly
    mock_entity_manager.create_entity.assert_called_once()
    # Check entity data passed to create_entity
    entity_data = mock_entity_manager.create_entity.call_args[0][0]
    assert entity_data["name"] == "Test Lesson"
    assert entity_data["type"] == "LessonContainer"
    assert "metadata" in entity_data
    assert entity_data["metadata"]["difficulty"] == "intermediate"


def test_create_lesson_container_missing_title(mock_component_managers, mock_logger):
    """Test creating a lesson container with missing title."""
    # Create manager with component managers
    manager = LessonMemoryManager(**mock_component_managers, logger=mock_logger)
    
    # Create lesson container with missing title
    lesson_data = {
        "description": "This is a test lesson",
    }
    
    result = manager.create_lesson_container(lesson_data)
    
    # Verify result indicates error
    result_obj = json.loads(result)
    assert result_obj["status"] == "error"
    assert "title is required" in result_obj["message"].lower()
    
    # Verify entity_manager was not called
    mock_component_managers["entity_manager"].create_entity.assert_not_called()


def test_get_lesson_container_by_id(mock_component_managers, mock_logger):
    """Test retrieving a lesson container by ID."""
    # Create manager
    manager = LessonMemoryManager(**mock_component_managers, logger=mock_logger)
    
    # Mock entity retrieval
    mock_component_managers["entity_manager"].get_entity.return_value = json.dumps({
        "id": "lesson-123",
        "name": "Test Lesson",
        "type": "LessonContainer",
        "metadata": {
            "description": "This is a test lesson",
            "difficulty": "intermediate"
        }
    })
    
    # Get lesson container
    result = manager.get_lesson_container("lesson-123")
    
    # Verify result
    result_obj = json.loads(result)
    assert result_obj["id"] == "lesson-123"
    assert result_obj["name"] == "Test Lesson"
    assert result_obj["metadata"]["difficulty"] == "intermediate"
    
    # Verify entity_manager was called correctly
    mock_component_managers["entity_manager"].get_entity.assert_called_once_with("lesson-123")


def test_get_lesson_container_not_found(mock_component_managers, mock_logger):
    """Test retrieving a non-existent lesson container."""
    # Create manager
    manager = LessonMemoryManager(**mock_component_managers, logger=mock_logger)
    
    # Mock entity retrieval - not found
    mock_component_managers["entity_manager"].get_entity.return_value = None
    
    # Get lesson container
    result = manager.get_lesson_container("nonexistent-lesson")
    
    # Verify result
    result_obj = json.loads(result)
    assert result_obj["status"] == "error"
    assert "not found" in result_obj["message"].lower()
    
    # Verify entity_manager was called correctly
    mock_component_managers["entity_manager"].get_entity.assert_called_once_with("nonexistent-lesson")


def test_update_lesson_container(mock_component_managers, mock_logger):
    """Test updating a lesson container."""
    # Create manager
    manager = LessonMemoryManager(**mock_component_managers, logger=mock_logger)
    
    # Mock entity retrieval and update
    mock_component_managers["entity_manager"].get_entity.return_value = json.dumps({
        "id": "lesson-123",
        "name": "Test Lesson",
        "type": "LessonContainer",
        "metadata": {
            "description": "This is a test lesson",
            "difficulty": "beginner"
        }
    })
    
    mock_component_managers["entity_manager"].update_entity.return_value = json.dumps({
        "status": "success",
        "id": "lesson-123"
    })
    
    # Update lesson container
    lesson_data = {
        "id": "lesson-123",
        "title": "Updated Lesson Title",
        "description": "Updated description",
        "difficulty": "advanced"
    }
    
    result = manager.update_lesson_container(lesson_data)
    
    # Verify result
    result_obj = json.loads(result)
    assert result_obj["status"] == "success"
    
    # Verify entity_manager calls
    mock_component_managers["entity_manager"].get_entity.assert_called_once_with("lesson-123")
    mock_component_managers["entity_manager"].update_entity.assert_called_once()
    
    # Check updated data
    update_data = mock_component_managers["entity_manager"].update_entity.call_args[0][0]
    assert update_data["name"] == "Updated Lesson Title"
    assert update_data["metadata"]["description"] == "Updated description"
    assert update_data["metadata"]["difficulty"] == "advanced"


def test_update_lesson_container_not_found(mock_component_managers, mock_logger):
    """Test updating a non-existent lesson container."""
    # Create manager
    manager = LessonMemoryManager(**mock_component_managers, logger=mock_logger)
    
    # Mock entity retrieval - not found
    mock_component_managers["entity_manager"].get_entity.return_value = None
    
    # Update lesson container
    lesson_data = {
        "id": "nonexistent-lesson",
        "title": "Updated Lesson Title"
    }
    
    result = manager.update_lesson_container(lesson_data)
    
    # Verify result
    result_obj = json.loads(result)
    assert result_obj["status"] == "error"
    assert "not found" in result_obj["message"].lower()
    
    # Verify entity_manager calls
    mock_component_managers["entity_manager"].get_entity.assert_called_once_with("nonexistent-lesson")
    mock_component_managers["entity_manager"].update_entity.assert_not_called()


def test_delete_lesson_container(mock_component_managers, mock_logger):
    """Test deleting a lesson container."""
    # Create manager
    manager = LessonMemoryManager(**mock_component_managers, logger=mock_logger)
    
    # Mock entity deletion
    mock_component_managers["entity_manager"].delete_entity.return_value = json.dumps({
        "status": "success"
    })
    
    # Delete lesson container
    result = manager.delete_lesson_container("lesson-123")
    
    # Verify result
    result_obj = json.loads(result)
    assert result_obj["status"] == "success"
    
    # Verify entity_manager was called correctly
    mock_component_managers["entity_manager"].delete_entity.assert_called_once_with("lesson-123")


def test_list_lesson_containers(mock_component_managers, mock_logger):
    """Test listing lesson containers."""
    # Create manager
    manager = LessonMemoryManager(**mock_component_managers, logger=mock_logger)
    
    # Mock entity search
    mock_entities = [
        {
            "id": "lesson-1",
            "name": "Lesson 1",
            "type": "LessonContainer",
            "metadata": {
                "description": "First lesson",
                "difficulty": "beginner",
                "tags": ["python", "beginner"]
            }
        },
        {
            "id": "lesson-2",
            "name": "Lesson 2",
            "type": "LessonContainer",
            "metadata": {
                "description": "Second lesson",
                "difficulty": "intermediate",
                "tags": ["python", "algorithms"]
            }
        }
    ]
    mock_component_managers["entity_manager"].get_entities_by_type.return_value = json.dumps(mock_entities)
    
    # List lesson containers
    result = manager.list_lesson_containers()
    
    # Verify result
    result_list = json.loads(result)
    assert len(result_list) == 2
    assert result_list[0]["name"] == "Lesson 1"
    assert result_list[1]["name"] == "Lesson 2"
    
    # Verify entity_manager was called correctly
    mock_component_managers["entity_manager"].get_entities_by_type.assert_called_once_with("LessonContainer")


def test_list_lesson_containers_with_filters(mock_component_managers, mock_logger):
    """Test listing lesson containers with filters."""
    # Create manager
    manager = LessonMemoryManager(**mock_component_managers, logger=mock_logger)
    
    # Mock entity search
    mock_entities = [
        {
            "id": "lesson-1",
            "name": "Lesson 1",
            "type": "LessonContainer",
            "metadata": {
                "description": "First lesson",
                "difficulty": "beginner",
                "tags": ["python", "beginner"],
                "visibility": "public"
            }
        }
    ]
    mock_component_managers["entity_manager"].get_entities_by_type.return_value = json.dumps(mock_entities)
    
    # List lesson containers with filters
    result = manager.list_lesson_containers(
        tags=["python"],
        difficulty="beginner",
        visibility="public"
    )
    
    # Verify result
    result_list = json.loads(result)
    assert len(result_list) == 1
    assert result_list[0]["name"] == "Lesson 1"
    
    # Verify entity_manager was called correctly
    mock_component_managers["entity_manager"].get_entities_by_type.assert_called_once_with("LessonContainer")


def test_create_lesson_section(mock_component_managers, mock_logger):
    """Test creating a lesson section."""
    # Create manager
    manager = LessonMemoryManager(**mock_component_managers, logger=mock_logger)
    
    # Mock entity creation and relation creation
    mock_component_managers["entity_manager"].create_entity.return_value = json.dumps({
        "id": "section-123",
        "name": "Test Section"
    })
    
    mock_component_managers["relation_manager"].create_relationship.return_value = json.dumps({
        "status": "success"
    })
    
    # Mock lesson container retrieval
    mock_component_managers["entity_manager"].get_entity.return_value = json.dumps({
        "id": "lesson-123",
        "name": "Test Lesson",
        "type": "LessonContainer"
    })
    
    # Create lesson section
    section_data = {
        "lesson_id": "lesson-123",
        "title": "Test Section",
        "content": "This is the section content",
        "order": 1,
        "section_type": "explanation"
    }
    
    result = manager.create_lesson_section(section_data)
    
    # Verify result
    result_obj = json.loads(result)
    assert result_obj["status"] == "success"
    assert "section_id" in result_obj
    
    # Verify entity_manager and relation_manager calls
    mock_component_managers["entity_manager"].get_entity.assert_called_once_with("lesson-123")
    mock_component_managers["entity_manager"].create_entity.assert_called_once()
    mock_component_managers["relation_manager"].create_relationship.assert_called_once()
    
    # Check entity data
    entity_data = mock_component_managers["entity_manager"].create_entity.call_args[0][0]
    assert entity_data["name"] == "Test Section"
    assert entity_data["type"] == "LessonSection"
    assert entity_data["metadata"]["content"] == "This is the section content"
    assert entity_data["metadata"]["order"] == 1
    assert entity_data["metadata"]["section_type"] == "explanation"


def test_create_lesson_section_lesson_not_found(mock_component_managers, mock_logger):
    """Test creating a lesson section for non-existent lesson."""
    # Create manager
    manager = LessonMemoryManager(**mock_component_managers, logger=mock_logger)
    
    # Mock lesson container retrieval - not found
    mock_component_managers["entity_manager"].get_entity.return_value = None
    
    # Create lesson section
    section_data = {
        "lesson_id": "nonexistent-lesson",
        "title": "Test Section",
        "content": "This is the section content"
    }
    
    result = manager.create_lesson_section(section_data)
    
    # Verify result
    result_obj = json.loads(result)
    assert result_obj["status"] == "error"
    assert "lesson not found" in result_obj["message"].lower()
    
    # Verify entity_manager calls
    mock_component_managers["entity_manager"].get_entity.assert_called_once_with("nonexistent-lesson")
    mock_component_managers["entity_manager"].create_entity.assert_not_called()
    mock_component_managers["relation_manager"].create_relationship.assert_not_called()


def test_update_lesson_section(mock_component_managers, mock_logger):
    """Test updating a lesson section."""
    # Create manager
    manager = LessonMemoryManager(**mock_component_managers, logger=mock_logger)
    
    # Mock section retrieval and update
    mock_component_managers["entity_manager"].get_entity.return_value = json.dumps({
        "id": "section-123",
        "name": "Original Section Title",
        "type": "LessonSection",
        "metadata": {
            "content": "Original content",
            "order": 1,
            "section_type": "introduction"
        }
    })
    
    mock_component_managers["entity_manager"].update_entity.return_value = json.dumps({
        "status": "success",
        "id": "section-123"
    })
    
    # Update lesson section
    section_data = {
        "id": "section-123",
        "title": "Updated Section Title",
        "content": "Updated content",
        "order": 2,
        "section_type": "explanation"
    }
    
    result = manager.update_lesson_section(section_data)
    
    # Verify result
    result_obj = json.loads(result)
    assert result_obj["status"] == "success"
    
    # Verify entity_manager calls
    mock_component_managers["entity_manager"].get_entity.assert_called_once_with("section-123")
    mock_component_managers["entity_manager"].update_entity.assert_called_once()
    
    # Check updated data
    update_data = mock_component_managers["entity_manager"].update_entity.call_args[0][0]
    assert update_data["name"] == "Updated Section Title"
    assert update_data["metadata"]["content"] == "Updated content"
    assert update_data["metadata"]["order"] == 2
    assert update_data["metadata"]["section_type"] == "explanation"


def test_create_lesson_relationship(mock_component_managers, mock_logger):
    """Test creating a relationship between lessons."""
    # Create manager
    manager = LessonMemoryManager(**mock_component_managers, logger=mock_logger)
    
    # Mock source and target lesson retrieval
    mock_component_managers["entity_manager"].get_entity.side_effect = [
        json.dumps({
            "id": "lesson-1",
            "name": "Source Lesson",
            "type": "LessonContainer"
        }),
        json.dumps({
            "id": "lesson-2",
            "name": "Target Lesson",
            "type": "LessonContainer"
        })
    ]
    
    # Mock relation creation
    mock_component_managers["relation_manager"].create_relationship.return_value = json.dumps({
        "status": "success"
    })
    
    # Create lesson relationship
    relationship_data = {
        "source_id": "lesson-1",
        "target_id": "lesson-2",
        "relationship_type": "PREREQUISITE",
        "metadata": {
            "strength": "strong"
        }
    }
    
    result = manager.create_lesson_relationship(relationship_data)
    
    # Verify result
    result_obj = json.loads(result)
    assert result_obj["status"] == "success"
    
    # Verify entity_manager and relation_manager calls
    assert mock_component_managers["entity_manager"].get_entity.call_count == 2
    mock_component_managers["relation_manager"].create_relationship.assert_called_once()
    
    # Check relationship data
    rel_data = mock_component_managers["relation_manager"].create_relationship.call_args[0][0]
    assert rel_data["from_entity"] == "lesson-1"
    assert rel_data["to_entity"] == "lesson-2"
    assert rel_data["relation_type"] == "PREREQUISITE"
    assert rel_data["properties"]["strength"] == "strong"


def test_find_related_lessons(mock_component_managers, mock_logger):
    """Test finding lessons related to a specific lesson."""
    # Create manager
    manager = LessonMemoryManager(**mock_component_managers, logger=mock_logger)
    
    # Mock lesson retrieval
    mock_component_managers["entity_manager"].get_entity.return_value = json.dumps({
        "id": "lesson-1",
        "name": "Test Lesson",
        "type": "LessonContainer",
        "metadata": {
            "description": "This is a test lesson about Python",
            "tags": ["python", "programming"]
        }
    })
    
    # Mock entity search
    mock_entities = [
        {
            "id": "lesson-2",
            "name": "Related Lesson 1",
            "type": "LessonContainer",
            "metadata": {
                "description": "Another lesson about Python",
                "tags": ["python", "advanced"]
            },
            "score": 0.85
        },
        {
            "id": "lesson-3",
            "name": "Related Lesson 2",
            "type": "LessonContainer",
            "metadata": {
                "description": "Introduction to programming",
                "tags": ["programming", "beginner"]
            },
            "score": 0.75
        }
    ]
    mock_component_managers["entity_manager"].search_entities.return_value = json.dumps(mock_entities)
    
    # Find related lessons
    result = manager.find_related_lessons("lesson-1", 0.7)
    
    # Verify result
    result_list = json.loads(result)
    assert len(result_list) == 2
    assert result_list[0]["name"] == "Related Lesson 1"
    assert result_list[1]["name"] == "Related Lesson 2"
    assert result_list[0]["similarity"] == 0.85
    
    # Verify entity_manager calls
    mock_component_managers["entity_manager"].get_entity.assert_called_once_with("lesson-1")
    mock_component_managers["entity_manager"].search_entities.assert_called_once()


def test_generate_learning_path(mock_component_managers, mock_logger):
    """Test generating a learning path for a topic."""
    # Create manager
    manager = LessonMemoryManager(**mock_component_managers, logger=mock_logger)
    
    # Mock entity search
    mock_entities = [
        {
            "id": "lesson-1",
            "name": "Introduction to Python",
            "type": "LessonContainer",
            "metadata": {
                "description": "Basic Python concepts",
                "difficulty": "beginner",
                "tags": ["python", "beginner"]
            }
        },
        {
            "id": "lesson-2",
            "name": "Intermediate Python",
            "type": "LessonContainer",
            "metadata": {
                "description": "Intermediate Python concepts",
                "difficulty": "intermediate",
                "tags": ["python", "intermediate"]
            }
        },
        {
            "id": "lesson-3",
            "name": "Advanced Python",
            "type": "LessonContainer",
            "metadata": {
                "description": "Advanced Python concepts",
                "difficulty": "advanced",
                "tags": ["python", "advanced"]
            }
        }
    ]
    mock_component_managers["entity_manager"].search_entities.return_value = json.dumps(mock_entities)
    
    # Mock relationship search for prerequisites
    mock_relationships = [
        {
            "from": {"id": "lesson-1", "name": "Introduction to Python"},
            "to": {"id": "lesson-2", "name": "Intermediate Python"},
            "type": "PREREQUISITE"
        },
        {
            "from": {"id": "lesson-2", "name": "Intermediate Python"},
            "to": {"id": "lesson-3", "name": "Advanced Python"},
            "type": "PREREQUISITE"
        }
    ]
    mock_component_managers["relation_manager"].get_relationships.return_value = json.dumps(mock_relationships)
    
    # Generate learning path
    result = manager.generate_learning_path("python", "beginner", 5)
    
    # Verify result
    result_obj = json.loads(result)
    assert result_obj["status"] == "success"
    assert "learning_path" in result_obj
    assert len(result_obj["learning_path"]) == 3
    assert result_obj["learning_path"][0]["name"] == "Introduction to Python"
    assert result_obj["learning_path"][1]["name"] == "Intermediate Python"
    assert result_obj["learning_path"][2]["name"] == "Advanced Python"
    
    # Verify entity_manager and relation_manager calls
    mock_component_managers["entity_manager"].search_entities.assert_called_once()
    assert mock_component_managers["relation_manager"].get_relationships.call_count > 0


def test_create_lesson(mock_component_managers, mock_logger):
    """Test creating a complete lesson with sections."""
    # Create manager
    manager = LessonMemoryManager(**mock_component_managers, logger=mock_logger)
    
    # Mock container creation
    mock_component_managers["entity_manager"].create_entity.side_effect = [
        json.dumps({"id": "lesson-123", "name": "Test Lesson"}),  # Container
        json.dumps({"id": "section-1", "name": "Introduction"}),  # Section 1
        json.dumps({"id": "section-2", "name": "Main Content"})   # Section 2
    ]
    
    # Mock relation creation
    mock_component_managers["relation_manager"].create_relationship.return_value = json.dumps({
        "status": "success"
    })
    
    # Create a lesson
    result = manager.create_lesson(
        title="Test Lesson",
        problem_statement="This is a test problem",
        sections=[
            {"title": "Introduction", "content": "This is the introduction", "section_type": "introduction"},
            {"title": "Main Content", "content": "This is the main content", "section_type": "explanation"}
        ],
        difficulty="beginner",
        tags=["test", "lesson"]
    )
    
    # Verify result
    result_obj = json.loads(result)
    assert result_obj["status"] == "success"
    assert result_obj["lesson_id"] == "lesson-123"
    
    # Verify entity_manager calls for container and sections
    assert mock_component_managers["entity_manager"].create_entity.call_count == 3
    assert mock_component_managers["relation_manager"].create_relationship.call_count == 2


def test_get_lesson_with_sections(mock_component_managers, mock_logger):
    """Test retrieving a lesson with its sections."""
    # Create manager
    manager = LessonMemoryManager(**mock_component_managers, logger=mock_logger)
    
    # Mock lesson container retrieval
    mock_component_managers["entity_manager"].get_entity.return_value = json.dumps({
        "id": "lesson-123",
        "name": "Test Lesson",
        "type": "LessonContainer",
        "metadata": {
            "description": "This is a test lesson",
            "difficulty": "intermediate"
        }
    })
    
    # Mock section relationships retrieval
    mock_sections = [
        {
            "r": {"type": "HAS_SECTION", "properties": {}},
            "section": {
                "id": "section-1", 
                "properties": {
                    "name": "Introduction",
                    "type": "LessonSection",
                    "metadata": {
                        "content": "This is the introduction",
                        "section_type": "introduction",
                        "order": 1
                    }
                }
            }
        },
        {
            "r": {"type": "HAS_SECTION", "properties": {}},
            "section": {
                "id": "section-2", 
                "properties": {
                    "name": "Main Content",
                    "type": "LessonSection",
                    "metadata": {
                        "content": "This is the main content",
                        "section_type": "explanation",
                        "order": 2
                    }
                }
            }
        }
    ]
    mock_component_managers["relation_manager"].get_relationships.return_value = json.dumps(mock_sections)
    
    # Get lesson with sections
    result = manager.get_lesson("lesson-123")
    
    # Verify result
    result_obj = json.loads(result)
    assert result_obj["id"] == "lesson-123"
    assert result_obj["name"] == "Test Lesson"
    assert "sections" in result_obj
    assert len(result_obj["sections"]) == 2
    assert result_obj["sections"][0]["name"] == "Introduction"
    assert result_obj["sections"][1]["name"] == "Main Content"
    
    # Verify entity_manager and relation_manager calls
    mock_component_managers["entity_manager"].get_entity.assert_called_once_with("lesson-123")
    mock_component_managers["relation_manager"].get_relationships.assert_called_once()


def test_get_lessons(mock_component_managers, mock_logger):
    """Test retrieving all lessons."""
    # Create manager
    manager = LessonMemoryManager(**mock_component_managers, logger=mock_logger)
    
    # Mock lessons retrieval
    mock_lessons = [
        {
            "id": "lesson-1",
            "name": "Lesson 1",
            "type": "LessonContainer",
            "metadata": {"difficulty": "beginner"}
        },
        {
            "id": "lesson-2",
            "name": "Lesson 2",
            "type": "LessonContainer",
            "metadata": {"difficulty": "advanced"}
        }
    ]
    mock_component_managers["entity_manager"].get_entities_by_type.return_value = json.dumps(mock_lessons)
    
    # Get all lessons
    result = manager.get_lessons()
    
    # Verify result
    result_list = json.loads(result)
    assert len(result_list) == 2
    assert result_list[0]["name"] == "Lesson 1"
    assert result_list[1]["name"] == "Lesson 2"
    
    # Verify entity_manager call
    mock_component_managers["entity_manager"].get_entities_by_type.assert_called_once_with("LessonContainer") 