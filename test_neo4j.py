#!/usr/bin/env python3
import os
from neo4j import GraphDatabase
import json

# Connection settings from .env
NEO4J_URI = "bolt://localhost:7687"
NEO4J_USER = "neo4j"
NEO4J_PASSWORD = "P@ssW0rd2025!"

# Test entity data
TEST_ENTITY = {
    "name": "test_user",
    "entityType": "Person",
    "observations": ["Test observation 1", "Test observation 2"]
}

def create_entity_direct(driver, entity):
    """Create an entity directly using Neo4j driver."""
    print(f"Creating entity with name: {entity['name']}, type: {entity['entityType']}")
    
    # First create the entity node
    entity_query = """
    MERGE (e:Entity {name: $name})
    SET e.entityType = $entity_type
    RETURN e
    """
    
    result = driver.execute_query(
        entity_query,
        name=entity["name"],
        entity_type=entity["entityType"],
        database_="neo4j"
    )
    
    print(f"Entity creation result summary: {result[1].counters}")
    
    # Then add observations
    for observation in entity["observations"]:
        obs_query = """
        MATCH (e:Entity {name: $name})
        MERGE (o:Observation {content: $content})
        MERGE (e)-[:HAS_OBSERVATION]->(o)
        RETURN o
        """
        
        obs_result = driver.execute_query(
            obs_query,
            name=entity["name"],
            content=observation,
            database_="neo4j"
        )
        
        print(f"Observation creation result summary: {obs_result[1].counters}")
    
    return "Entity created with observations"

def verify_entity_exists(driver, entity_name):
    """Verify that an entity exists in Neo4j."""
    query = """
    MATCH (e:Entity {name: $name})
    OPTIONAL MATCH (e)-[:HAS_OBSERVATION]->(o:Observation)
    RETURN e.name as name, e.entityType as type, collect(o.content) as observations
    """
    
    result = driver.execute_query(
        query,
        name=entity_name,
        database_="neo4j"
    )
    
    if result[0]:
        record = result[0][0]
        print(f"Found entity: {record['name']}, type: {record['type']}")
        print(f"Observations: {record['observations']}")
        return record
    else:
        print(f"Entity {entity_name} not found!")
        return None

def list_all_entities(driver):
    """List all entities in the database."""
    query = """
    MATCH (e:Entity)
    RETURN e.name as name, e.entityType as type
    """
    
    result = driver.execute_query(
        query,
        database_="neo4j"
    )
    
    entities = []
    if result[0]:
        for record in result[0]:
            entities.append({
                "name": record["name"],
                "type": record["type"]
            })
            print(f"Entity: {record['name']}, type: {record['type']}")
    
    print(f"Found {len(entities)} entities")
    return entities

def main():
    # Connect to Neo4j
    print(f"Connecting to Neo4j at {NEO4J_URI}")
    
    try:
        driver = GraphDatabase.driver(
            NEO4J_URI, 
            auth=(NEO4J_USER, NEO4J_PASSWORD),
            max_connection_lifetime=30 * 60,
            max_connection_pool_size=50,
            connection_acquisition_timeout=60,
            keep_alive=True
        )
        
        # Test connection
        driver.verify_connectivity()
        print("Successfully connected to Neo4j!")
        
        # List existing entities
        print("\n--- Existing Entities ---")
        list_all_entities(driver)
        
        # Create test entity
        print("\n--- Creating Test Entity ---")
        create_entity_direct(driver, TEST_ENTITY)
        
        # Verify entity was created
        print("\n--- Verifying Entity Creation ---")
        verify_entity_exists(driver, TEST_ENTITY["name"])
        
        # List entities again to confirm
        print("\n--- Updated Entity List ---")
        list_all_entities(driver)
        
        # Close the driver
        driver.close()
        print("\nConnection closed")
        
    except Exception as e:
        print(f"Error: {str(e)}")

if __name__ == "__main__":
    main() 