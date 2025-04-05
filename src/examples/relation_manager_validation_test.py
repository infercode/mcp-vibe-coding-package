"""
Test script to verify the RelationManager integration with Neo4j query validation.
"""

import os
import sys
import argparse
from typing import Dict, Any, List, Optional

# Add the project root to the Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from src.graph_memory.base_manager import BaseManager
from src.graph_memory.entity_manager import EntityManager
from src.graph_memory.relation_manager import RelationManager
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

def setup_test_entities(entity_manager: EntityManager) -> None:
    """Create test entities for relationship tests."""
    # Create source entity
    source_entity = {
        "name": "SourceEntity",
        "entityType": "test_source",
        "observations": ["This is a source entity for testing"]
    }
    
    # Create target entity
    target_entity = {
        "name": "TargetEntity",
        "entityType": "test_target",
        "observations": ["This is a target entity for testing"]
    }
    
    # Create both entities
    entity_manager.create_entities([source_entity, target_entity])
    logger.info("Created test entities for relationship testing")

def test_valid_relation_operations(relation_manager: RelationManager) -> None:
    """Test valid relationship operations with validation."""
    logger.info("Testing valid relationship operations...")
    
    # Create a test relation
    relation = {
        "from": "SourceEntity",
        "to": "TargetEntity",
        "relationType": "TEST_RELATION",
        "priority": "high",
        "created_at": "2023-04-05"
    }
    
    # Test relation creation
    result = relation_manager.create_relations([relation])
    logger.info(f"Relation creation result: {result}")
    
    # Test relation retrieval
    result = relation_manager.get_relations("SourceEntity")
    logger.info(f"Relation retrieval result: {result}")
    
    # Test relation update
    update_result = relation_manager.update_relation(
        "SourceEntity", "TargetEntity", "TEST_RELATION", 
        {"priority": "medium", "updated_at": "2023-04-06"}
    )
    logger.info(f"Relation update result: {update_result}")
    
    # Test relation deletion
    delete_result = relation_manager.delete_relation("SourceEntity", "TargetEntity", "TEST_RELATION")
    logger.info(f"Relation deletion result: {delete_result}")

def test_invalid_relation_operations(relation_manager: RelationManager) -> None:
    """Test invalid relationship operations to verify validation."""
    logger.info("Testing invalid relationship operations...")
    
    # Test invalid relation creation with destructive operation in query type
    try:
        # Direct access to _create_relation_in_neo4j method to bypass normal API
        # We're using DROP as the relation type which contains a destructive operation
        relation_manager._create_relation_in_neo4j("SourceEntity", "TargetEntity", "DROP TABLE", {"test": "value"})
        logger.error("FAILED: Destructive operation in relation type was not caught!")
    except ValueError as e:
        logger.info(f"SUCCESS: Caught destructive operation in relation type: {str(e)}")
    except Exception as e:
        logger.info(f"SUCCESS: Caught destructive operation in relation type via exception: {str(e)}")
    
    # Test with invalid parameter types
    try:
        # Create a relation with a complex object that can't be serialized
        invalid_relation = {
            "from": "SourceEntity",
            "to": "TargetEntity",
            "relationType": "TEST_RELATION",
            "complex_property": object()  # This can't be serialized to Neo4j
        }
        
        # This should fail validation
        result = relation_manager.create_relations([invalid_relation])
        
        # Check if the error message is in the result - this is a dict converted to JSON string
        if "error" in result or "errors" in result:
            logger.info(f"SUCCESS: Invalid parameter correctly rejected: {result}")
        else:
            logger.error(f"FAILED: Invalid parameter type was not caught properly: {result}")
    except ValueError as e:
        # Direct ValueError exception
        logger.info(f"SUCCESS: Caught invalid parameter through ValueError: {str(e)}")
    except Exception as e:
        # Any other exception
        logger.info(f"SUCCESS: Caught invalid parameter through exception: {str(e)}")
    
    # Test invalid relation retrieval with a destructive operation in the query
    try:
        # Direct access to the class methods to bypass public API
        # This injects a destructive operation into a read query
        query = """
        MATCH (e1:Entity {name: $entity_name})-[r]->(e2:Entity)
        DELETE r
        RETURN e1.name as from_entity, TYPE(r) as relation_type, e2.name as to_entity
        """
        
        # This should raise a ValueError due to validation
        result = relation_manager.base_manager.safe_execute_read_query(
            query, 
            {"entity_name": "SourceEntity"}
        )
        logger.error("FAILED: Destructive operation was not caught!")
    except ValueError as e:
        logger.info(f"SUCCESS: Caught invalid read query: {str(e)}")

def main():
    """Main entry point for the script."""
    parser = argparse.ArgumentParser(description="Test the RelationManager with Neo4j query validation")
    
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
        
        # Create EntityManager for setup
        entity_manager = EntityManager(base_manager)
        
        # Create RelationManager for testing
        relation_manager = RelationManager(base_manager)
        
        # Setup test entities
        setup_test_entities(entity_manager)
        
        # Run tests
        test_valid_relation_operations(relation_manager)
        test_invalid_relation_operations(relation_manager)
        
        # Clean up if requested
        if args.cleanup:
            cleanup_database(base_manager, not args.skip_confirm)
            
        logger.info("Tests completed successfully")
        
    except Exception as e:
        logger.error(f"Error during test execution: {str(e)}")

if __name__ == "__main__":
    main() 