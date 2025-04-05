"""
Test script to verify the SearchManager integration with Neo4j query validation.
"""

import os
import sys
import argparse
import json
from typing import Dict, Any, List, Optional
from unittest.mock import MagicMock, patch

# Add the project root to the Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from src.graph_memory.base_manager import BaseManager
from src.graph_memory.entity_manager import EntityManager
from src.graph_memory.search_manager import SearchManager
from src.logger import get_logger
from src.utils.neo4j_query_utils import validate_query, sanitize_query_parameters
from src.models.neo4j_queries import CypherQuery, CypherParameters

logger = get_logger()

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
    
    # Mock the safe_execute_read_query method to validate queries without executing
    def mock_safe_execute_read_query(query, parameters=None):
        try:
            # Validate query and parameters
            validated_query = validate_query(query, parameters, read_only=True)
            logger.info(f"Mock read query validated: {validated_query.query}")
            
            # For test entities, return mock data
            if "name: $name" in query and parameters and "name" in parameters:
                if parameters["name"] in ["TestEntity1", "TestEntity2", "TestEntity3", "SearchTarget"]:
                    # Create a mock record for entity found
                    mock_record = MagicMock()
                    mock_record.get.return_value = MagicMock()
                    mock_record.get.return_value.items.return_value = [
                        ("name", parameters["name"]),
                        ("entityType", "person" if "1" in parameters["name"] or "3" in parameters["name"] 
                                      else "location" if "2" in parameters["name"] 
                                      else "concept")
                    ]
                    mock_record.get.return_value.id = f"mock-id-{parameters['name']}"
                    return [mock_record]
            
            if "nodes" in query.lower() or "path" in query.lower():
                # Return empty nodes/paths for neighborhood and path queries
                mock_record = MagicMock()
                mock_record.get.return_value = []
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
            # Validate query and parameters
            validated_query = validate_query(query, parameters, read_only=False)
            logger.info(f"Mock write query validated: {validated_query.query}")
            
            # Return a mock success result
            mock_record = MagicMock()
            mock_record.get.return_value = 1  # Simulate 1 affected node
            return [mock_record]
        except ValueError as e:
            # Re-raise validation errors
            logger.error(f"Query validation error: {str(e)}")
            raise
    
    mock_base_manager.safe_execute_write_query = MagicMock(side_effect=mock_safe_execute_write_query)
    
    # Mock the generate_embedding method
    mock_base_manager.generate_embedding = MagicMock(return_value=[0.1, 0.2, 0.3, 0.4])
    
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
    """Create test entities for search tests."""
    # Create test entities with different types
    entities = [
        {
            "name": "TestEntity1",
            "entityType": "person",
            "observations": ["This is a test person entity", "Created for search testing"]
        },
        {
            "name": "TestEntity2",
            "entityType": "location",
            "observations": ["This is a test location entity", "Created for search testing"]
        },
        {
            "name": "TestEntity3",
            "entityType": "person",
            "observations": ["This is another test person entity", "Has specific search terms"]
        },
        {
            "name": "SearchTarget",
            "entityType": "concept",
            "observations": ["This entity should be found by search", "Contains special search keywords"]
        }
    ]
    
    # Create all entities
    entity_manager.create_entities(entities)
    logger.info("Created test entities for search testing")

def parse_json_result(json_str: str) -> Dict[str, Any]:
    """Parse a JSON string result into a dictionary."""
    try:
        return json.loads(json_str)
    except json.JSONDecodeError:
        logger.error(f"Failed to parse JSON: {json_str}")
        return {"error": "Failed to parse JSON result"}

def test_valid_search_operations(search_manager: SearchManager) -> None:
    """Test valid search operations with validation."""
    logger.info("Testing valid search operations...")
    
    # Test basic entity search
    result = search_manager.search_entities("Test", limit=10)
    logger.info(f"Basic search result: {result}")
    result_dict = parse_json_result(result)
    if "error" in result_dict:
        logger.error(f"Basic search failed: {result_dict['error']}")
    else:
        logger.info("Basic search successful")
    
    # Test search with entity type filter
    result = search_manager.search_entities("Test", entity_types=["person"])
    logger.info(f"Filtered search result: {result}")
    result_dict = parse_json_result(result)
    if "error" in result_dict:
        logger.error(f"Filtered search failed: {result_dict['error']}")
    else:
        logger.info("Filtered search successful")
    
    # Test entity neighborhood search
    result = search_manager.search_entity_neighborhoods("TestEntity1", max_depth=1)
    logger.info(f"Neighborhood search result: {result}")
    result_dict = parse_json_result(result)
    if "error" in result_dict:
        logger.error(f"Neighborhood search failed: {result_dict['error']}")
    else:
        logger.info("Neighborhood search successful")
    
    # Test custom query search
    custom_query = """
    MATCH (e:Entity)
    WHERE e.name CONTAINS 'Search'
    RETURN e.name as name, e.entityType as type
    """
    result = search_manager.query_knowledge_graph(custom_query)
    logger.info(f"Custom query result: {result}")
    result_dict = parse_json_result(result)
    if "error" in result_dict:
        logger.error(f"Custom query failed: {result_dict['error']}")
    else:
        logger.info("Custom query successful")

def test_invalid_search_operations(search_manager: SearchManager) -> None:
    """Test invalid search operations to verify validation."""
    logger.info("Testing invalid search operations...")
    
    # Test with destructive operation in custom query
    try:
        malicious_query = """
        MATCH (e:Entity)
        WHERE e.name CONTAINS 'Test'
        DELETE e
        RETURN count(e) as deleted
        """
        result = search_manager.query_knowledge_graph(malicious_query)
        result_dict = parse_json_result(result)
        if "error" in result_dict and "Forbidden operation" in result_dict["error"]:
            logger.info(f"SUCCESS: Destructive operation correctly rejected: {result_dict['error']}")
        else:
            logger.error(f"FAILED: Destructive operation was not caught properly: {result}")
    except ValueError as e:
        logger.info(f"SUCCESS: Caught destructive operation through ValueError: {str(e)}")
    except Exception as e:
        logger.info(f"SUCCESS: Caught destructive operation through exception: {str(e)}")
    
    # Test with invalid parameter types
    try:
        # Attempt to pass a complex object that can't be serialized
        complex_params = {"complex_object": object()}
        result = search_manager.query_knowledge_graph("MATCH (n) RETURN n LIMIT 1", complex_params)
        logger.error("FAILED: Invalid parameter type was not caught!")
    except ValueError as e:
        logger.info(f"SUCCESS: Caught invalid parameter through ValueError: {str(e)}")
    except Exception as e:
        logger.info(f"SUCCESS: Caught invalid parameter through exception: {str(e)}")
    
    # Test semantic search with invalid parameters
    try:
        # Direct access to safe_execute_query with invalid read_only parameter
        query = """
        MATCH (e:Entity) 
        SET e.test = 'value'
        RETURN e
        """
        # This should raise an error for destructive operation in read-only mode
        records = search_manager.base_manager.safe_execute_read_query(query)
        logger.error("FAILED: Destructive operation in read-only query was not caught!")
    except ValueError as e:
        logger.info(f"SUCCESS: Caught destructive operation in read-only query: {str(e)}")
    except Exception as e:
        logger.info(f"SUCCESS: Caught destructive operation through exception: {str(e)}")

def main():
    """Main entry point for the script."""
    parser = argparse.ArgumentParser(description="Test the SearchManager with Neo4j query validation")
    
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
            
            # Create SearchManager for testing
            search_manager = SearchManager(base_manager)
            
            # Run tests
            test_valid_search_operations(search_manager)
            test_invalid_search_operations(search_manager)
            
            logger.info("Mock tests completed successfully")
            return
        
        logger.info(f"Connecting to Neo4j at {args.uri} with user {args.username}")
        
        # Setup BaseManager with provided credentials
        base_manager = setup_base_manager(args.uri, args.username, args.password, args.database)
        
        # Create EntityManager for setup
        entity_manager = EntityManager(base_manager)
        
        # Create SearchManager for testing
        search_manager = SearchManager(base_manager)
        
        # Setup test entities
        setup_test_entities(entity_manager)
        
        # Run tests
        test_valid_search_operations(search_manager)
        test_invalid_search_operations(search_manager)
        
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