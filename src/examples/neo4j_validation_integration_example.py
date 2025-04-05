#!/usr/bin/env python3
"""
Neo4j Query Validation Integration Example

This script demonstrates how to integrate the Neo4j query validation 
utilities with the EntityManager for type-safe and validated operations.
"""

import sys
import os
import json
import argparse
from typing import Dict, Any, List, Optional, cast
from unittest.mock import MagicMock, patch

# Add the project root to the Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

# Import required modules
from src.logger import get_logger
from src.graph_memory.base_manager import BaseManager
from src.graph_memory.entity_manager import EntityManager

# Initialize logger
logger = get_logger()

class ValidatedEntityManager(EntityManager):
    """Extended EntityManager with Neo4j query validation."""
    
    def get_entity_safe(self, entity_name: str) -> Dict[str, Any]:
        """
        Get an entity with validated query execution.
        
        Args:
            entity_name: Name of the entity to retrieve
            
        Returns:
            Entity data dictionary or error response
        """
        try:
            self.base_manager.ensure_initialized()
            
            # Use safe_execute_read_query for validation
            query = """
            MATCH (e:Entity {name: $name})
            RETURN e
            """
            
            # This will validate the query is read-only and sanitize parameters
            records = self.base_manager.safe_execute_read_query(
                query=query,
                parameters={"name": entity_name}
            )
            
            if not records or len(records) == 0:
                return {"error": f"Entity '{entity_name}' not found"}
            
            # Extract entity data
            entity_node = records[0].get("e", {})
            if not entity_node:
                return {"error": "Entity data not found in result"}
                
            # Build entity response
            entity = dict(entity_node)
            entity["id"] = getattr(entity_node, "id", None)
            
            return {"entity": entity}
            
        except ValueError as e:
            # Handle validation errors
            logger.error(f"Validation error getting entity: {e}")
            return {"error": f"Query validation failed: {str(e)}"}
        except Exception as e:
            # Handle other errors
            logger.error(f"Error getting entity: {e}")
            return {"error": f"Failed to get entity: {str(e)}"}
    
    def create_entity_safe(self, entity_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Create an entity with validated query execution.
        
        Args:
            entity_data: Entity data to create
            
        Returns:
            Created entity data dictionary or error response
        """
        try:
            self.base_manager.ensure_initialized()
            
            # Validate entity data
            if not entity_data or "name" not in entity_data:
                return {"error": "Entity name is required"}
            
            entity_name = cast(str, entity_data.get("name"))
            
            # Check if entity already exists
            existing = self.get_entity_safe(entity_name)
            if "entity" in existing:
                return {"error": f"Entity '{entity_name}' already exists"}
            
            # Build properties for creation
            properties = {}
            for key, value in entity_data.items():
                properties[key] = value
            
            # Use safe_execute_write_query for validation
            create_query = """
            CREATE (e:Entity $props)
            RETURN e
            """
            
            # This will validate the query and sanitize parameters
            records = self.base_manager.safe_execute_write_query(
                query=create_query,
                parameters={"props": properties}
            )
            
            if not records or len(records) == 0:
                return {"error": "Failed to create entity, no result returned"}
            
            # Extract created entity
            entity_node = records[0].get("e", {})
            if not entity_node:
                return {"error": "Created entity data not found in result"}
                
            # Build entity response
            created_entity = dict(entity_node)
            created_entity["id"] = getattr(entity_node, "id", None)
            
            return {"entity": created_entity, "created": True}
            
        except ValueError as e:
            # Handle validation errors
            logger.error(f"Validation error creating entity: {e}")
            return {"error": f"Query validation failed: {str(e)}"}
        except Exception as e:
            # Handle other errors
            logger.error(f"Error creating entity: {e}")
            return {"error": f"Failed to create entity: {str(e)}"}

def print_json(data: Any, title: Optional[str] = None) -> None:
    """Print data as formatted JSON."""
    if title:
        print(f"\n=== {title} ===")
    print(json.dumps(data, indent=2, default=str))
    print()

def run_with_mock():
    """Run the example with mocked Neo4j connections."""
    print("=== Neo4j Query Validation Integration Example (Mock Mode) ===\n")
    
    # Create mocked base manager with patched methods
    base_manager = BaseManager(logger=logger)
    
    # Mock initialization to avoid connecting to a real Neo4j database
    base_manager.initialize = MagicMock(return_value=True)
    base_manager.initialized = True
    
    # Mock safe_execute_query to return test data
    def mock_safe_execute_query(query, parameters=None, database=None):
        """Mock implementation that returns test data without Neo4j."""
        # Default to empty parameters if None
        params = parameters or {}
        
        if "MATCH (e:Entity {name: $name})" in query:
            entity_name = params.get("name", "")
            if entity_name == "ValidationTestEntity":
                # Return a mock entity when requested
                return [{"e": {
                    "name": "ValidationTestEntity",
                    "type": "Test",
                    "description": "Entity created using validation",
                    "created_at": "2023-04-05T12:00:00Z",
                    "id": "test-id-123"
                }}], {"mock": "summary"}
            else:
                # No entity found
                return [], {"mock": "summary"}
        elif "CREATE (e:Entity $props)" in query:
            # Return the created entity
            props = params.get("props", {})
            return [{"e": {
                **props,
                "id": "new-id-456"
            }}], {"mock": "summary"}
        else:
            # Default empty response
            return [], {"mock": "summary"}
    
    # Patch the safe_execute_query method
    base_manager.safe_execute_query = mock_safe_execute_query
    
    print("Base manager initialized successfully (in mock mode)")
    return base_manager

def run_with_real_neo4j(uri, username, password, database):
    """Run the example with real Neo4j connection."""
    print(f"=== Neo4j Query Validation Integration Example (Real Neo4j Mode) ===\n")
    print(f"Connecting to Neo4j at {uri}")
    
    # Set environment variables for Neo4j connection
    os.environ["NEO4J_URI"] = uri
    os.environ["NEO4J_USER"] = username
    os.environ["NEO4J_PASSWORD"] = password
    if database:
        os.environ["NEO4J_DATABASE"] = database
    
    # Create and initialize base manager
    base_manager = BaseManager(logger=logger)
    if not base_manager.initialize():
        print("Failed to initialize base manager with Neo4j connection")
        print("Falling back to mock mode")
        return run_with_mock()
    
    print("Base manager initialized successfully with Neo4j connection")
    return base_manager

def cleanup_database(base_manager, confirm=True):
    """
    Clean up the database by removing all nodes and relationships.
    
    Args:
        base_manager: The BaseManager instance to use
        confirm: Whether to ask for confirmation before cleaning up
        
    Returns:
        True if cleanup was successful, False otherwise
    """
    if not base_manager or not base_manager.initialized:
        print("Cannot clean up: Base manager not initialized")
        return False
        
    if confirm:
        user_input = input("\nWARNING: This will delete ALL data in the database. Type 'yes' to confirm: ")
        if user_input.lower() != "yes":
            print("Database cleanup cancelled")
            return False
    
    try:
        # Execute delete query
        print("\n=== Cleaning up database ===")
        cleanup_query = "MATCH (n) DETACH DELETE n"
        
        # Use direct safe_execute_query to bypass read-only validation
        result, summary = base_manager.safe_execute_query(
            cleanup_query,
            parameters={}
        )
        
        # Get counts from summary
        if summary and "counters" in summary:
            nodes_deleted = summary["counters"].get("nodes_deleted", 0)
            relationships_deleted = summary["counters"].get("relationships_deleted", 0)
            print(f"Deleted {nodes_deleted} nodes and {relationships_deleted} relationships")
        else:
            print("Database cleanup completed")
            
        return True
    except Exception as e:
        print(f"Error cleaning up database: {e}")
        return False

def main():
    """Run the Neo4j validation integration example."""
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Neo4j Query Validation Example")
    parser.add_argument("--mock", action="store_true", help="Use mock mode instead of real Neo4j")
    parser.add_argument("--uri", default="bolt://localhost:7687", help="Neo4j URI")
    parser.add_argument("--username", default="neo4j", help="Neo4j username")
    parser.add_argument("--password", default="password", help="Neo4j password")
    parser.add_argument("--database", default="neo4j", help="Neo4j database name")
    parser.add_argument("--cleanup", action="store_true", help="Clean up database after running examples")
    parser.add_argument("--skip-confirm", action="store_true", help="Skip confirmation for database cleanup")
    
    args = parser.parse_args()
    
    # Initialize base manager based on mode
    if args.mock:
        base_manager = run_with_mock()
    else:
        base_manager = run_with_real_neo4j(args.uri, args.username, args.password, args.database)
    
    # Create validated entity manager
    entity_manager = ValidatedEntityManager(base_manager)
    
    # Example 1: Create an entity with validated query
    entity_data = {
        "name": "ValidationTestEntity",
        "type": "Test",
        "description": "Entity created using validation",
        "created_at": "2023-04-05T12:00:00Z" 
    }
    
    create_result = entity_manager.create_entity_safe(entity_data)
    print_json(create_result, "Create Entity Result")
    
    # Example 2: Get the entity with validated query
    get_result = entity_manager.get_entity_safe("ValidationTestEntity")
    print_json(get_result, "Get Entity Result")
    
    # Example 3: Try to execute a destructive query (should fail validation)
    try:
        query = """
        MATCH (e:Entity {name: $name})
        DELETE e
        RETURN e
        """
        
        base_manager.safe_execute_read_query(
            query=query,
            parameters={"name": "ValidationTestEntity"}
        )
        print("❌ Destructive query was allowed (should not happen)")
    except ValueError as e:
        print(f"✅ Validation caught destructive query: {e}")
    
    # Example 4: Try to create an entity with invalid parameters
    try:
        invalid_data = {
            "name": "InvalidEntity",
            "type": "Test",
            "invalid_object": object()  # Not serializable
        }
        
        entity_manager.create_entity_safe(invalid_data)
        print("❌ Invalid parameters were allowed (should not happen)")
    except Exception as e:
        print(f"✅ Validation caught invalid parameters: {e}")
    
    print("\n=== End of Demo ===")
    
    # Clean up database if requested and not in mock mode
    if not args.mock and args.cleanup:
        cleanup_database(base_manager, not args.skip_confirm)

if __name__ == "__main__":
    main() 