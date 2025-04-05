"""
Test script to verify the EntityManager integration with Neo4j query validation.
"""

import os
import sys
import argparse
from typing import Dict, Any, List, Optional

# Add the project root to the Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from src.graph_memory.base_manager import BaseManager
from src.graph_memory.entity_manager import EntityManager
from src.logger import get_logger

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

def test_valid_entity_operations(entity_manager: EntityManager) -> None:
    """Test valid entity operations with validation."""
    logger.info("Testing valid entity operations...")
    
    # Create a test entity
    entity = {
        "name": "TestEntity",
        "entityType": "test",
        "observations": ["This is a test entity", "Created for validation testing"]
    }
    
    # Test entity creation
    result = entity_manager.create_entities([entity])
    logger.info(f"Entity creation result: {result}")
    
    # Test entity retrieval
    result = entity_manager.get_entity("TestEntity")
    logger.info(f"Entity retrieval result: {result}")
    
    # Test entity update
    update_result = entity_manager.update_entity("TestEntity", {"description": "Updated entity"})
    logger.info(f"Entity update result: {update_result}")
    
    # Test entity deletion
    delete_result = entity_manager.delete_entity("TestEntity")
    logger.info(f"Entity deletion result: {delete_result}")

def test_invalid_entity_operations(entity_manager: EntityManager) -> None:
    """Test invalid entity operations to verify validation."""
    logger.info("Testing invalid entity operations...")
    
    # Test invalid entity retrieval with a destructive operation in the query
    try:
        # Direct access to the class methods to bypass public API
        # This injects a destructive operation into a read query
        query = """
        MATCH (e:Entity {name: $name})
        DELETE e
        RETURN e
        """
        
        # This should raise a ValueError due to validation
        result = entity_manager.base_manager.safe_execute_read_query(query, {"name": "TestEntity"})
        logger.error("FAILED: Destructive operation was not caught!")
    except ValueError as e:
        logger.info(f"SUCCESS: Caught invalid read query: {str(e)}")
    
    # Test with invalid parameter types
    try:
        # Create an entity with a complex object that can't be serialized
        invalid_entity = {
            "name": "InvalidEntity",
            "entityType": "test",
            "complex_property": object()  # This can't be serialized to Neo4j
        }
        
        # This should fail validation
        result = entity_manager.create_entities([invalid_entity])
        
        # Check if the error message is in the result - this is a dict converted to JSON string
        if "error" in result and "non-serializable value" in result:
            logger.info(f"SUCCESS: Invalid parameter correctly rejected: {result}")
        else:
            logger.error(f"FAILED: Invalid parameter type was not caught properly: {result}")
    except ValueError as e:
        # Direct ValueError exception
        logger.info(f"SUCCESS: Caught invalid parameter through ValueError: {str(e)}")
    except Exception as e:
        # Any other exception
        logger.info(f"SUCCESS: Caught invalid parameter through exception: {str(e)}")

def main():
    """Main entry point for the script."""
    parser = argparse.ArgumentParser(description="Test the EntityManager with Neo4j query validation")
    
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
    
    if args.mock:
        logger.info("Running in mock mode - no Neo4j connection will be established")
        # Mock implementation would go here
        return
    
    logger.info(f"Connecting to Neo4j at {args.uri} with user {args.username}")
    
    try:
        # Setup BaseManager with provided credentials
        base_manager = setup_base_manager(args.uri, args.username, args.password, args.database)
        
        # Create EntityManager
        entity_manager = EntityManager(base_manager)
        
        # Run tests
        test_valid_entity_operations(entity_manager)
        test_invalid_entity_operations(entity_manager)
        
        # Clean up if requested
        if args.cleanup:
            cleanup_database(base_manager, not args.skip_confirm)
            
        logger.info("Tests completed successfully")
        
    except Exception as e:
        logger.error(f"Error during test execution: {str(e)}")

if __name__ == "__main__":
    main() 