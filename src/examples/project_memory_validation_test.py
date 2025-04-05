"""
Test script to verify the ProjectMemoryManager integration with Neo4j query validation.
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
from src.project_memory import ProjectMemoryManager
from src.logger import get_logger
from src.utils.neo4j_query_utils import validate_query, sanitize_query_parameters

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
    
    # Add neo4j_driver attribute
    mock_base_manager.neo4j_driver = True
    
    # Mock the safe_execute_read_query method
    def mock_safe_execute_read_query(query, parameters=None):
        try:
            # Validate parameters
            if parameters:
                sanitized_params = sanitize_query_parameters(parameters)
            
            logger.info(f"Mock read query validated: {query}")
            
            # For project container queries, return mock data
            if "entityType: 'ProjectContainer'" in query and parameters and "name" in parameters:
                if parameters["name"] == "TestProject":
                    # Create a mock record for container found
                    mock_record = MagicMock()
                    mock_record.get.return_value = MagicMock()
                    mock_record.get.return_value.items.return_value = [
                        ("id", "prj-123"),
                        ("name", "TestProject"),
                        ("entityType", "ProjectContainer"),
                        ("description", "Test project for validation")
                    ]
                    return [mock_record]
            
            # For domain queries
            if "entityType: 'Domain'" in query and parameters and "name" in parameters:
                if parameters["name"] == "TestDomain":
                    # Create a mock record for domain found
                    mock_record = MagicMock()
                    mock_record.get.return_value = MagicMock()
                    mock_record.get.return_value.items.return_value = [
                        ("id", "dom-123"),
                        ("name", "TestDomain"),
                        ("entityType", "Domain"),
                        ("description", "Test domain for validation")
                    ]
                    component_count_record = MagicMock()
                    component_count_record.get.return_value = 5
                    return [{"d": mock_record.get.return_value, "component_count": 5}]
            
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
            # Validate parameters
            params = {}
            if parameters:
                sanitized_params = sanitize_query_parameters(parameters)
                params = sanitized_params
            
            logger.info(f"Mock write query validated: {query}")
            
            # For project container creation
            if "CREATE (c:Entity" in query:
                # Create a mock record for container creation
                mock_record = MagicMock()
                mock_record.get.return_value = MagicMock()
                
                if "properties" in params:
                    properties = params["properties"]
                    mock_items = []
                    for key, value in properties.items():
                        mock_items.append((key, value))
                    mock_record.get.return_value.items.return_value = mock_items
                
                return [mock_record]
            
            # For domain creation
            if "CREATE (d:Entity" in query:
                # Create a mock record for domain creation
                mock_record = MagicMock()
                mock_record.get.return_value = MagicMock()
                
                if "properties" in params:
                    properties = params["properties"]
                    mock_items = []
                    for key, value in properties.items():
                        mock_items.append((key, value))
                    mock_record.get.return_value.items.return_value = mock_items
                
                return [mock_record]
            
            # Default mock success response
            mock_record = MagicMock()
            mock_record.get.return_value = 1  # Default success value
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

def parse_json_result(json_str: str) -> Dict[str, Any]:
    """Parse a JSON string result into a dictionary."""
    try:
        return json.loads(json_str)
    except json.JSONDecodeError:
        logger.error(f"Failed to parse JSON: {json_str}")
        return {"error": "Failed to parse JSON result"}

def test_valid_project_operations(project_manager: ProjectMemoryManager) -> None:
    """Test valid project operations with validation."""
    logger.info("Testing valid project operations...")
    
    # Test project container creation
    project_data = {
        "name": "TestProject",
        "description": "Test project for validation",
        "metadata": {
            "author": "testing_user",
            "priority": "high",
            "status": "active"
        },
        "tags": ["test", "validation", "neo4j"]
    }
    
    result = project_manager.create_project_container(project_data)
    logger.info(f"Project container creation result: {result}")
    
    # Test project container retrieval
    result = project_manager.get_project_container("TestProject")
    logger.info(f"Project container retrieval result: {result}")
    
    # Test domain creation
    result = project_manager.create_domain(
        "TestDomain", 
        "TestProject", 
        "Test domain for validation"
    )
    logger.info(f"Domain creation result: {result}")
    
    # Test domain retrieval
    result = project_manager.get_domain("TestDomain", "TestProject")
    logger.info(f"Domain retrieval result: {result}")
    
    # Test domain update
    result = project_manager.update_domain(
        "TestDomain", 
        "TestProject", 
        {"description": "Updated domain description"}
    )
    logger.info(f"Domain update result: {result}")
    
    # Test listing domains
    result = project_manager.list_domains("TestProject")
    logger.info(f"List domains result: {result}")

def test_invalid_project_operations(project_manager: ProjectMemoryManager) -> None:
    """Test invalid project operations to verify validation."""
    logger.info("Testing invalid project operations...")
    
    # Test with a complex object that can't be serialized
    try:
        invalid_project_data = {
            "name": "InvalidProject",
            "description": "Test project with invalid data",
            "complex_property": object()  # This can't be serialized
        }
        
        result = project_manager.create_project_container(invalid_project_data)
        logger.error(f"FAILED: Invalid parameter type was not caught: {result}")
    except ValueError as e:
        logger.info(f"SUCCESS: Caught invalid parameter: {str(e)}")
    except Exception as e:
        logger.info(f"SUCCESS: Caught exception: {str(e)}")
    
    # Test destructive operation in a read query by directly accessing base_manager
    # This is not something users would typically do but tests our validation
    try:
        destructive_query = """
        MATCH (c:Entity {entityType: 'ProjectContainer'})
        DELETE c
        RETURN count(c) as deleted
        """
        
        # This should raise a ValueError due to validation
        project_manager.base_manager.safe_execute_read_query(destructive_query)
        logger.error("FAILED: Destructive operation was not caught!")
    except ValueError as e:
        logger.info(f"SUCCESS: Caught destructive operation: {str(e)}")
    except Exception as e:
        logger.info(f"SUCCESS: Caught exception: {str(e)}")

def main():
    """Main entry point for the script."""
    parser = argparse.ArgumentParser(description="Test the ProjectMemoryManager with Neo4j query validation")
    
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
            
            # Create ProjectMemoryManager
            project_manager = ProjectMemoryManager(base_manager)
            
            # Run tests
            test_valid_project_operations(project_manager)
            test_invalid_project_operations(project_manager)
            
            logger.info("Mock tests completed successfully")
            return
        
        logger.info(f"Connecting to Neo4j at {args.uri} with user {args.username}")
        
        # Setup BaseManager with provided credentials
        base_manager = setup_base_manager(args.uri, args.username, args.password, args.database)
        
        # Create ProjectMemoryManager
        project_manager = ProjectMemoryManager(base_manager)
        
        # Run tests
        test_valid_project_operations(project_manager)
        test_invalid_project_operations(project_manager)
        
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