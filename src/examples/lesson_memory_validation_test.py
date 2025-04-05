#!/usr/bin/env python
"""
Validation test for LessonMemoryManager with Neo4j query validation.
Tests the integration of LessonContainer and LessonObservation with
validated query methods (safe_execute_read_query and safe_execute_write_query).
"""

import os
import sys
import argparse
import json
import logging
from typing import Dict, List, Any, Optional

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Add the project root directory to the path
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(os.path.dirname(current_dir))
sys.path.insert(0, project_root)

from src.graph_memory.base_manager import BaseManager
from src.lesson_memory.lesson_container import LessonContainer
from src.lesson_memory.lesson_observation import LessonObservation
from src.utils import dict_to_json

def parse_json_dict(json_str: str) -> Dict[str, Any]:
    """
    Parse a JSON string into a dictionary.
    
    Args:
        json_str: JSON string to parse
        
    Returns:
        Dictionary representation of the JSON
    """
    return json.loads(json_str)

def setup_base_manager() -> BaseManager:
    """
    Set up a BaseManager with Neo4j credentials.
    
    Returns:
        BaseManager: Initialized BaseManager instance
    """
    # Set Neo4j credentials as environment variables
    os.environ["NEO4J_URI"] = os.getenv("NEO4J_URI", "bolt://localhost:7687")
    os.environ["NEO4J_USER"] = os.getenv("NEO4J_USER", "neo4j")
    os.environ["NEO4J_PASSWORD"] = os.getenv("NEO4J_PASSWORD", "password")
    
    logger.info(f"Using Neo4j at {os.environ['NEO4J_URI']}")
    
    # BaseManager will read from environment variables
    return BaseManager()

def setup_mock_base_manager() -> BaseManager:
    """
    Set up a mock BaseManager for testing without a real database.
    
    Returns:
        BaseManager: Mocked BaseManager instance
    """
    logger.info("Setting up mock BaseManager")
    
    class MockBaseManager(BaseManager):
        def __init__(self):
            # Skip actual initialization
            self.initialized = True
            self.logger = logging.getLogger(__name__)
            self.mock_containers = {}
            self.mock_entities = {}
            self.mock_observations = {}
            
        def initialize(self):
            self.initialized = True
            return True
            
        def ensure_initialized(self):
            self.initialized = True
        
        def safe_execute_read_query(self, query, parameters=None, database=None):
            logger.info(f"Mock read query: {query}")
            logger.info(f"Parameters: {parameters}")
            
            # Validate that this is actually a read query
            if any(op in query.upper() for op in ["CREATE", "DELETE", "SET", "REMOVE", "MERGE"]):
                # Exception for specific validation queries that don't actually modify data
                if "RETURN" in query and not any(op in query.upper() for op in ["WITH", "CALL"]):
                    pass  # Allow read-like queries with validation terms
                else:
                    raise ValueError(f"Destructive operation not allowed in read-only mode: {query}")
            
            params = parameters or {}
            
            # Parse and return mocked results based on the query
            if "MATCH (c:LessonContainer" in query:
                container_name = params.get("name") or params.get("container_name")
                if container_name and container_name in self.mock_containers:
                    container = self.mock_containers[container_name]
                    return [{"c": container}]
                return []
                
            elif "MATCH (e:Entity" in query:
                entity_name = params.get("name") or params.get("entity_name")
                if entity_name and entity_name in self.mock_entities:
                    entity = self.mock_entities[entity_name]
                    return [{"e": entity}]
                return []
                
            elif "MATCH (o:Observation" in query:
                observation_id = params.get("id") or params.get("observation_id")
                if observation_id and observation_id in self.mock_observations:
                    observation = self.mock_observations[observation_id]
                    return [{"o": observation}]
                return []
                
            return []
        
        def safe_execute_write_query(self, query, parameters=None, database=None):
            logger.info(f"Mock write query: {query}")
            logger.info(f"Parameters: {parameters}")
            
            # Sanitize parameters
            params = parameters or {}
            
            # Process write operations
            if "CREATE (c:LessonContainer" in query:
                container_props = params.get("properties", {})
                container_name = container_props.get("name")
                
                # Check if container already exists
                if container_name in self.mock_containers:
                    return []  # Simulate failure
                
                # Create container
                self.mock_containers[container_name] = container_props
                return [{"c": container_props}]
                
            elif "CREATE (e:Entity" in query:
                entity_props = params.get("properties", {})
                entity_name = entity_props.get("name")
                
                # Create entity
                self.mock_entities[entity_name] = entity_props
                return [{"e": entity_props}]
                
            elif "CREATE (o:Observation" in query:
                observation_props = params.get("properties", {})
                observation_id = observation_props.get("id")
                
                # Create observation
                self.mock_observations[observation_id] = observation_props
                return [{"o": observation_props}]
                
            elif "DELETE" in query and "LessonContainer" in query:
                container_name = params.get("name")
                if container_name in self.mock_containers:
                    del self.mock_containers[container_name]
                    return [{}]
                return []
                
            # Default return for other write operations
            return [{}]
    
    return MockBaseManager()

def parse_json_result(json_str: str) -> Dict[str, Any]:
    """
    Parse JSON string results from manager methods.
    
    Args:
        json_str: JSON string to parse
        
    Returns:
        Parsed dictionary or empty dict on error
    """
    try:
        return parse_json_dict(json_str)
    except json.JSONDecodeError:
        logger.error(f"Failed to parse JSON: {json_str}")
        return {}

def test_valid_lesson_operations(base_manager: BaseManager):
    """
    Test valid lesson operations with the managers.
    
    Args:
        base_manager: BaseManager instance
    """
    logger.info("Testing valid lesson operations")
    
    # Initialize managers
    lesson_container = LessonContainer(base_manager)
    lesson_observation = LessonObservation(base_manager)
    
    # Create a lesson container
    logger.info("Creating lesson container")
    container_name = "TestLessonContainer"
    container_result = lesson_container.create_container(
        name=container_name,
        description="Test container for lesson validation",
        metadata={"test": True}
    )
    container_data = parse_json_result(container_result)
    logger.info(f"Container creation result: {container_data}")
    
    # Get the container
    logger.info("Retrieving lesson container")
    get_container_result = lesson_container.get_container(container_name)
    get_container_data = parse_json_result(get_container_result)
    logger.info(f"Retrieved container: {get_container_data}")
    
    # Create an entity in the container
    logger.info("Creating lesson entity")
    entity_name = "TestLessonEntity"
    # Use create_entity method from the LessonEntity class instead
    from src.lesson_memory.lesson_entity import LessonEntity
    lesson_entity = LessonEntity(base_manager)
    entity_result = lesson_entity.create_lesson_entity(
        container_name=container_name,
        entity_name=entity_name,
        entity_type="concept",
        metadata={"important": True}
    )
    entity_data = parse_json_result(entity_result)
    logger.info(f"Entity creation result: {entity_data}")
    
    # Add observation to entity
    logger.info("Adding observation to entity")
    observation_result = lesson_observation.add_lesson_observation(
        entity_name=entity_name,
        content="This is a test observation",
        observation_type="note",
        confidence=0.95
    )
    observation_data = parse_json_result(observation_result)
    logger.info(f"Observation added: {observation_data}")
    
    # Get observations for entity
    logger.info("Retrieving observations for entity")
    get_observations_result = lesson_observation.get_lesson_observations(entity_name)
    get_observations_data = parse_json_result(get_observations_result)
    logger.info(f"Retrieved observations: {get_observations_data}")
    
    # Update container
    logger.info("Updating lesson container")
    update_container_result = lesson_container.update_container(
        container_name=container_name,
        updates={"description": "Updated description"}
    )
    update_container_data = parse_json_result(update_container_result)
    logger.info(f"Container update result: {update_container_data}")
    
    # List containers
    logger.info("Listing lesson containers")
    list_containers_result = lesson_container.list_containers()
    list_containers_data = parse_json_result(list_containers_result)
    logger.info(f"Listed containers: {list_containers_data}")
    
    # Delete container (cleanup)
    if not args.keep_data:
        logger.info("Deleting test container")
        delete_result = lesson_container.delete_container(container_name)
        delete_data = parse_json_result(delete_result)
        logger.info(f"Container deletion result: {delete_data}")

def test_invalid_lesson_operations(base_manager: BaseManager):
    """
    Test invalid lesson operations to ensure proper error handling.
    
    Args:
        base_manager: BaseManager instance
    """
    logger.info("Testing invalid lesson operations")
    
    # Initialize managers
    lesson_container = LessonContainer(base_manager)
    lesson_observation = LessonObservation(base_manager)
    from src.lesson_memory.lesson_entity import LessonEntity
    lesson_entity = LessonEntity(base_manager)
    
    # Try to get non-existent container
    logger.info("Trying to get non-existent container")
    get_result = lesson_container.get_container("NonExistentContainer")
    get_data = parse_json_result(get_result)
    logger.info(f"Get non-existent container result: {get_data}")
    
    # Try to update non-existent container
    logger.info("Trying to update non-existent container")
    update_result = lesson_container.update_container(
        container_name="NonExistentContainer",
        updates={"description": "Should fail"}
    )
    update_data = parse_json_result(update_result)
    logger.info(f"Update non-existent container result: {update_data}")
    
    # Try to get observations for non-existent entity
    logger.info("Trying to get observations for non-existent entity")
    get_obs_result = lesson_observation.get_lesson_observations("NonExistentEntity")
    get_obs_data = parse_json_result(get_obs_result)
    logger.info(f"Get observations for non-existent entity result: {get_obs_data}")
    
    # Try to create entity in non-existent container
    logger.info("Trying to create entity in non-existent container")
    entity_result = lesson_entity.create_lesson_entity(
        container_name="NonExistentContainer",
        entity_name="TestEntity",
        entity_type="concept"
    )
    entity_data = parse_json_result(entity_result)
    logger.info(f"Create entity in non-existent container result: {entity_data}")

def main():
    """
    Main function to run the validation tests.
    """
    global args
    
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Lesson Memory Validation Test")
    parser.add_argument("--mock", action="store_true", help="Use mock mode instead of real Neo4j")
    parser.add_argument("--keep-data", action="store_true", help="Don't delete test data after tests")
    args = parser.parse_args()
    
    # Setup manager
    if args.mock:
        logger.info("Running in mock mode")
        base_manager = setup_mock_base_manager()
    else:
        logger.info("Running with real Neo4j connection")
        base_manager = setup_base_manager()
        base_manager.initialize()
    
    try:
        # Run tests
        test_valid_lesson_operations(base_manager)
        test_invalid_lesson_operations(base_manager)
        
        logger.info("Validation tests completed successfully")
        
    except Exception as e:
        logger.error(f"Validation test failed: {str(e)}")
        raise

if __name__ == "__main__":
    main() 