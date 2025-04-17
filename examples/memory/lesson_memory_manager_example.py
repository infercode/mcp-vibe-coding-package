"""
Example Script: Using the Refactored LessonMemoryManager with Pydantic Models

This example demonstrates how to use the LessonMemoryManager class with
the newly refactored implementation that uses Pydantic models for validation
and standardized responses.
"""

import json
import sys
import os
from typing import Dict, Any, Optional
from datetime import datetime

# Add parent directory to path to allow imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from src.graph_memory.base_manager import BaseManager
from src.lesson_memory import LessonMemoryManager
from src.models.lesson_memory import (
    LessonContainerCreate,
    LessonEntityCreate,
    LessonObservationCreate,
    LessonRelationshipCreate,
    SearchQuery
)
from src.logger import get_logger

# Custom JSON encoder to handle datetime objects
class DateTimeEncoder(json.JSONEncoder):
    def default(self, o):
        if isinstance(o, datetime):
            return o.isoformat()
        return super().default(o)

def print_response(response: Dict[str, Any], title: Optional[str] = None) -> None:
    """
    Print a formatted response.
    
    Args:
        response: The response dictionary
        title: Optional title for the response
    """
    if title:
        print(f"\n=== {title} ===")
    print(json.dumps(response, indent=2, cls=DateTimeEncoder))
    print("")

def run_lesson_memory_manager_examples() -> None:
    """Run examples of the LessonMemoryManager functionality."""
    # Initialize the base manager with the logger from src.logger
    logger = get_logger()
    
    # Configure the base manager to not attempt Neo4j connection for this example
    # Set environment variables for testing
    os.environ["NEO4J_URI"] = "bolt://localhost:7687"  # Using default but won't connect
    os.environ["NEO4J_USER"] = "neo4j"
    os.environ["NEO4J_PASSWORD"] = "password"
    
    # Create the base manager
    base_manager = BaseManager(logger=logger)
    
    # Disable actual connection attempt for testing
    base_manager.initialized = True  # Skip initialization to avoid connection attempts
    
    # Create the LessonMemoryManager
    memory_manager = LessonMemoryManager(base_manager)
    
    # Example 1: Create a container
    container_response = memory_manager.create_container(
        name="Learning Python",
        description="Container for Python learning lessons",
        metadata={"source": "Python documentation", "difficulty": "intermediate"}
    )
    print_response(container_response, "Create Container Response")
    
    # Example 2: Create an entity 
    entity_response = memory_manager.create_lesson_entity(
        container_name="Learning Python",
        entity_name="List Comprehensions",
        entity_type="CONCEPT",
        metadata={"difficulty": "intermediate", "tags": ["python", "lists", "functional"]}
    )
    print_response(entity_response, "Create Entity Response")
    
    # Example 3: Create structured observations
    observations_response = memory_manager.create_structured_lesson_observations(
        entity_name="List Comprehensions",
        what_was_learned="List comprehensions provide a concise way to create lists based on existing lists.",
        why_it_matters="They can make code more readable and efficient compared to traditional for loops.",
        how_to_apply="Use the syntax [expression for item in iterable if condition] to create a new list.",
        evidence="Example: squares = [x**2 for x in range(10)]",
        container_name="Learning Python"
    )
    print_response(observations_response, "Create Structured Observations Response")
    
    # Example 4: Add another entity to create a relationship
    another_entity_response = memory_manager.create_lesson_entity(
        container_name="Learning Python",
        entity_name="Generator Expressions",
        entity_type="CONCEPT",
        metadata={"difficulty": "intermediate", "tags": ["python", "generators", "iterators"]}
    )
    print_response(another_entity_response, "Create Second Entity Response")
    
    # Example 5: Create a relationship between entities
    # In a real implementation, you'd use the refactored create_lesson_relation method
    print("\nCreating relationship between entities...\n")
    
    # Example 6: Get container contents
    container_entities = memory_manager.get_container_entities(
        container_name="Learning Python",
        limit=10
    )
    print_response(container_entities, "Container Entities")
    
    # Example 7: Update a container
    update_response = memory_manager.update_container(
        name="Learning Python",
        updates={"description": "Updated container for Python learning lessons"}
    )
    print_response(update_response, "Update Container Response")
    
    # Example 8: Error handling example - Try to get a non-existent container
    non_existent = memory_manager.get_container("NonExistentContainer")
    print_response(non_existent, "Error Response - Non-existent Container")

if __name__ == "__main__":
    run_lesson_memory_manager_examples() 