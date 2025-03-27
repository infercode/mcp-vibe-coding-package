#!/usr/bin/env python3
from neo4j import GraphDatabase

# Connection settings
NEO4J_URI = "bolt://localhost:7687"
NEO4J_USER = "neo4j"
NEO4J_PASSWORD = "P@ssW0rd2025!"
NEO4J_DATABASE = "neo4j"

def clean_database():
    """Clean all nodes and relationships from Neo4j database."""
    print(f"Connecting to Neo4j at {NEO4J_URI}...")
    
    try:
        driver = GraphDatabase.driver(
            NEO4J_URI,
            auth=(NEO4J_USER, NEO4J_PASSWORD)
        )
        
        # Test connection
        driver.verify_connectivity()
        print("Successfully connected to Neo4j!")
        
        # Get node count before cleaning
        count_query = """
        MATCH (n)
        RETURN count(n) as count
        """
        
        result = driver.execute_query(
            count_query,
            database_=NEO4J_DATABASE
        )
        
        node_count = result[0][0]["count"] if result[0] else 0
        print(f"Found {node_count} nodes before cleaning.")
        
        # Delete all relationships first
        rel_query = """
        MATCH ()-[r]-()
        DELETE r
        """
        
        rel_result = driver.execute_query(
            rel_query,
            database_=NEO4J_DATABASE
        )
        
        print(f"Deleted {rel_result[1].counters.relationships_deleted} relationships.")
        
        # Delete all nodes
        node_query = """
        MATCH (n)
        DELETE n
        """
        
        node_result = driver.execute_query(
            node_query,
            database_=NEO4J_DATABASE
        )
        
        print(f"Deleted {node_result[1].counters.nodes_deleted} nodes.")
        
        # Verify database is empty
        verify_result = driver.execute_query(
            count_query,
            database_=NEO4J_DATABASE
        )
        
        final_count = verify_result[0][0]["count"] if verify_result[0] else 0
        print(f"Database now has {final_count} nodes.")
        
        # Close the driver
        driver.close()
        print("Connection closed")
        
    except Exception as e:
        print(f"Error: {str(e)}")

if __name__ == "__main__":
    clean_database() 