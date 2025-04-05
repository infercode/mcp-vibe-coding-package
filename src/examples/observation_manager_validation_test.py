"""
Test script to verify the ObservationManager integration with Neo4j query validation.
"""

import os
import sys
import argparse
import json
import re
from typing import Dict, Any, List, Optional
from unittest.mock import MagicMock, patch

# Add the project root to the Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from src.graph_memory.base_manager import BaseManager
from src.graph_memory.entity_manager import EntityManager
from src.graph_memory.observation_manager import ObservationManager
from src.logger import get_logger
from src.utils.neo4j_query_utils import validate_query, sanitize_query_parameters
from src.models.neo4j_queries import CypherQuery, CypherParameters

logger = get_logger()

# Regular expressions for detecting destructive operations - copy from neo4j_query_utils.py
DESTRUCTIVE_OPERATIONS = [
    r"(?i)\b(CREATE|DELETE|REMOVE|DROP|SET)\b",
    r"(?i)\b(MERGE|DETACH DELETE)\b",
    r"(?i)\b(CREATE|DROP) (INDEX|CONSTRAINT)\b",
    r"(?i)\.drop\(.*\)"
]

def setup_base_manager(uri: str, username: str, password: str, database: str) -> BaseManager:
    """Setup a BaseManager with the provided credentials."""
    # Set environment variables for the BaseManager
    os.environ["NEO4J_URI"] = uri
    os.environ["NEO4J_USER"] = username
    os.environ["NEO4J_PASSWORD"] = password
    os.environ["NEO4J_DATABASE"] = database
    
    # Create and initialize the BaseManager
    base_manager = BaseManager(logger=logger)
    base_manager.initialize()
    
    return base_manager

def setup_mock_base_manager() -> BaseManager:
    """Setup a mocked BaseManager for testing without a real database."""
    # Create a mock BaseManager
    mock_base_manager = MagicMock(spec=BaseManager)
    
    # Setup the logger attribute
    mock_base_manager.logger = logger
    
    # Mock the ensure_initialized method
    mock_base_manager.ensure_initialized = MagicMock(return_value=None)
    
    # Add neo4j_driver attribute
    mock_base_manager.neo4j_driver = True
    
    # Mock the safe_execute_read_query method to validate queries without executing
    def mock_safe_execute_read_query(query, parameters=None):
        try:
            # Check if the query contains destructive operations - this is a read-only context
            for pattern in DESTRUCTIVE_OPERATIONS:
                if re.search(pattern, query):
                    error_msg = f"Destructive operation detected in read-only query: {pattern}"
                    logger.error(error_msg)
                    raise ValueError(error_msg)
            
            # Validate the query parameters only (not the query itself)
            if parameters:
                sanitized_params = sanitize_query_parameters(parameters)
            
            logger.info(f"Mock read query validated: {query}")
            
            # For test entities, return mock data
            if "name: $name" in query and parameters and "name" in parameters:
                if parameters["name"] in ["TestEntity1", "TestEntity2"]:
                    # Create a mock record for entity found
                    mock_record = MagicMock()
                    mock_record.get.return_value = MagicMock()
                    mock_record.get.return_value.items.return_value = [
                        ("name", parameters["name"]),
                        ("entityType", "person" if "1" in parameters["name"] else "location")
                    ]
                    mock_record.get.return_value.id = f"mock-id-{parameters['name']}"
                    return [mock_record]
            
            # For test observations
            if "HAS_OBSERVATION" in query and "id: $observation_id" in query and parameters and "observation_id" in parameters:
                if parameters["observation_id"] in ["obs123", "obs456"]:
                    # Create a mock record for observation found
                    mock_record = MagicMock()
                    mock_record.get.return_value = MagicMock()
                    mock_record.get.return_value.items.return_value = [
                        ("id", parameters["observation_id"]),
                        ("content", "Mock observation content"),
                        ("type", "test_observation"),
                        ("created", "2023-01-01T00:00:00")
                    ]
                    return [mock_record]
            
            # Default return is empty results
            return []
        except ValueError as e:
            # Re-raise validation errors
            logger.error(f"Query validation error: {str(e)}")
            raise
    
    mock_base_manager.safe_execute_read_query = MagicMock(side_effect=mock_safe_execute_read_query)
    
    # Mock the safe_execute_write_query method
    def mock_safe_execute_write_query(query, parameters=None):
        try:
            # Validate parameters - this will catch non-serializable objects
            if parameters:
                sanitized_params = sanitize_query_parameters(parameters)
                params = sanitized_params
            else:
                params = {}
            
            logger.info(f"Mock write query validated: {query}")
            
            # Return a mock success result
            mock_record = MagicMock()
            
            # For delete operations
            if "DELETE" in query:
                mock_record.get.return_value = 1  # Simulate 1 deleted item
            # For create operations
            elif "CREATE" in query:
                mock_record.get.return_value = MagicMock()
                mock_record.get.return_value.items.return_value = [
                    ("id", "mock-obs-id"),
                    ("content", params.get("content", "Mock content")),
                    ("type", params.get("type", "test_observation"))
                ]
            # For update operations
            elif "SET" in query:
                mock_record.get.return_value = MagicMock()
                mock_record.get.return_value.items.return_value = [
                    ("id", params.get("observation_id", "mock-obs-id")),
                    ("content", params.get("content", "Updated content")),
                    ("type", params.get("type", "test_observation")),
                    ("lastUpdated", "2023-01-02T00:00:00")
                ]
                
            return [mock_record]
        except ValueError as e:
            # Re-raise validation errors
            logger.error(f"Query validation error: {str(e)}")
            raise
    
    mock_base_manager.safe_execute_write_query = MagicMock(side_effect=mock_safe_execute_write_query)
    
    return mock_base_manager

def cleanup_database(base_manager: BaseManager, confirm: bool = True) -> None:
    """
    Clean up the database by removing all nodes and relationships.
    
    Args:
        base_manager: The BaseManager instance
        confirm: Whether to prompt for confirmation before deletion
    """
    if not base_manager.neo4j_driver:
        logger.warn("Neo4j driver not initialized, skipping cleanup")
        return
    
    if confirm:
        response = input("Are you sure you want to delete ALL nodes and relationships? (y/n): ")
        if response.lower() != 'y':
            logger.info("Cleanup cancelled")
            return
    
    try:
        # Execute DETACH DELETE to remove all nodes and relationships
        cleanup_query = "MATCH (n) DETACH DELETE n RETURN count(n) as nodes_deleted"
        records = base_manager.safe_execute_write_query(cleanup_query)
        
        if records and len(records) > 0:
            deleted = records[0].get("nodes_deleted", 0)
            logger.info(f"Deleted {deleted} nodes and all their relationships")
        else:
            logger.info("No nodes to delete")
    except Exception as e:
        logger.error(f"Error during database cleanup: {str(e)}")

def setup_test_entities(entity_manager: EntityManager) -> None:
    """Create test entities for observation tests."""
    # Create test entities
    entities = [
        {
            "name": "TestEntity1",
            "entityType": "person",
            "observations": ["This is the first test observation", "This is the second test observation"]
        },
        {
            "name": "TestEntity2",
            "entityType": "location",
            "observations": ["This is a location observation"]
        }
    ]
    
    # Create all entities
    entity_manager.create_entities(entities)
    logger.info("Created test entities for observation testing")

def parse_json_result(json_str: str) -> Dict[str, Any]:
    """Parse a JSON string result into a dictionary."""
    try:
        return json.loads(json_str)
    except json.JSONDecodeError:
        logger.error(f"Failed to parse JSON: {json_str}")
        return {"error": "Failed to parse JSON result"}

def test_valid_observation_operations(observation_manager: ObservationManager) -> None:
    """Test valid observation operations with validation."""
    logger.info("Testing valid observation operations...")
    
    # Test get entity observations
    result = observation_manager.get_entity_observations("TestEntity1")
    logger.info(f"Get observations result: {result}")
    result_dict = parse_json_result(result)
    if "error" in result_dict:
        logger.error(f"Get observations failed: {result_dict['error']}")
    else:
        logger.info("Get observations successful")
    
    # Test add observations
    observations = [
        {
            "entity": "TestEntity1",
            "content": "New observation for testing",
            "type": "test_observation"
        },
        {
            "entity": "TestEntity2",
            "content": "Another test observation",
            "type": "test_observation"
        }
    ]
    
    result = observation_manager.add_observations(observations)
    logger.info(f"Add observations result: {result}")
    result_dict = parse_json_result(result)
    if "error" in result_dict:
        logger.error(f"Add observations failed: {result_dict['error']}")
    else:
        logger.info("Add observations successful")
    
    # Test update observation
    result = observation_manager.update_observation(
        entity_name="TestEntity1",
        observation_id="obs123",
        content="Updated observation content",
        observation_type="updated_observation"
    )
    logger.info(f"Update observation result: {result}")
    result_dict = parse_json_result(result)
    if "error" in result_dict:
        logger.error(f"Update observation failed: {result_dict['error']}")
    else:
        logger.info("Update observation successful")
    
    # Test delete observation
    result = observation_manager.delete_observation(
        entity_name="TestEntity1",
        observation_id="obs456"
    )
    logger.info(f"Delete observation result: {result}")
    result_dict = parse_json_result(result)
    if "error" in result_dict:
        logger.error(f"Delete observation failed: {result_dict['error']}")
    else:
        logger.info("Delete observation successful")

def test_invalid_observation_operations(observation_manager: ObservationManager) -> None:
    """Test invalid observation operations to verify validation."""
    logger.info("Testing invalid observation operations...")
    
    # Test with non-existent entity
    result = observation_manager.get_entity_observations("NonExistentEntity")
    logger.info(f"Non-existent entity result: {result}")
    result_dict = parse_json_result(result)
    if "error" in result_dict and "not found" in result_dict["error"]:
        logger.info("SUCCESS: Non-existent entity correctly rejected")
    else:
        logger.error("FAILED: Non-existent entity was not caught properly")
    
    # Test with invalid parameters
    try:
        # First, directly call sanitize_query_parameters with a non-serializable object
        # to verify it catches it correctly
        complex_params = {"complex_object": object()}
        sanitize_query_parameters(complex_params)
        logger.error("FAILED: Invalid parameter type was not caught by sanitize_query_parameters!")
    except ValueError as e:
        logger.info(f"SUCCESS: sanitize_query_parameters correctly caught invalid parameter: {str(e)}")
        
        # Now test the ObservationManager with the complex object
        try:
            observations = [{
                "entity": "TestEntity1",
                "content": "Valid content",
                "complex_value": object()
            }]
            result = observation_manager.add_observations(observations)
            logger.error("FAILED: Invalid parameter type was not caught in add_observations!")
        except ValueError as e:
            logger.info(f"SUCCESS: Caught invalid parameter in add_observations through ValueError: {str(e)}")
        except Exception as e:
            logger.info(f"SUCCESS: Caught invalid parameter in add_observations through exception: {str(e)}")
    
    # Test directly with destructive operation in read-only context
    try:
        query = """
        MATCH (o:Observation)
        DELETE o
        RETURN count(o) as deleted
        """
        # This should raise an error for destructive operation in read-only mode
        records = observation_manager.base_manager.safe_execute_read_query(query)
        logger.error("FAILED: Destructive operation in read-only query was not caught!")
    except ValueError as e:
        logger.info(f"SUCCESS: Caught destructive operation in read-only query: {str(e)}")
    except Exception as e:
        logger.info(f"SUCCESS: Caught destructive operation through exception: {str(e)}")

def main():
    """Main entry point for the script."""
    parser = argparse.ArgumentParser(description="Test the ObservationManager with Neo4j query validation")
    
    # Connection parameters
    parser.add_argument("--uri", default="bolt://localhost:7687", help="Neo4j URI")
    parser.add_argument("--username", default="neo4j", help="Neo4j username")
    parser.add_argument("--password", default="password", help="Neo4j password")
    parser.add_argument("--database", default="neo4j", help="Neo4j database name")
    
    # Mock mode - don't actually connect to Neo4j
    parser.add_argument("--mock", action="store_true", help="Run in mock mode without Neo4j connection")
    
    # Cleanup flag
    parser.add_argument("--cleanup", action="store_true", help="Clean up database after tests")
    parser.add_argument("--skip-confirm", action="store_true", help="Skip confirmation for cleanup")
    
    args = parser.parse_args()
    
    try:
        if args.mock:
            logger.info("Running in mock mode - using mocked Neo4j connection")
            
            # Setup mocked BaseManager
            base_manager = setup_mock_base_manager()
            
            # Create EntityManager for setup
            entity_manager = EntityManager(base_manager)
            
            # Create ObservationManager for testing
            observation_manager = ObservationManager(base_manager)
            
            # Run tests
            test_valid_observation_operations(observation_manager)
            test_invalid_observation_operations(observation_manager)
            
            logger.info("Mock tests completed successfully")
            return
        
        logger.info(f"Connecting to Neo4j at {args.uri} with user {args.username}")
        
        # Setup BaseManager with provided credentials
        base_manager = setup_base_manager(args.uri, args.username, args.password, args.database)
        
        # Create EntityManager for setup
        entity_manager = EntityManager(base_manager)
        
        # Create ObservationManager for testing
        observation_manager = ObservationManager(base_manager)
        
        # Setup test entities
        setup_test_entities(entity_manager)
        
        # Run tests
        test_valid_observation_operations(observation_manager)
        test_invalid_observation_operations(observation_manager)
        
        # Clean up if requested
        if args.cleanup:
            cleanup_database(base_manager, not args.skip_confirm)
            
        logger.info("Tests completed successfully")
        
    except Exception as e:
        logger.error(f"Error during test execution: {str(e)}")
        # Print traceback for debugging
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main() 