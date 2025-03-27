#!/usr/bin/env python3
"""
Simple example client for the Neo4j MCP Graph Memory Server
"""

import asyncio
import json
from typing import Dict, Any

from mcp import Client
from mcp.client.stdio import StdioClientTransport


async def main():
    # Create a client transport
    transport = StdioClientTransport(
        command="python",
        args=["../neo4j_mcp_server.py"],
        env={
            "NEO4J_URI": "bolt://localhost:7687",
            "NEO4J_USER": "neo4j",
            "NEO4J_PASSWORD": "password",
            "NEO4J_DATABASE": "neo4j",
            "EMBEDDER_PROVIDER": "openai",
            "OPENAI_API_KEY": "YOUR_API_KEY"  # Replace with your actual API key
        }
    )

    # Create and connect the client
    client = Client(name="example-client", version="1.0.0")
    await client.connect(transport)

    # Configure the project
    await configure_project(client, "example-project")

    # Clean up previous test data
    await delete_all(client)

    # Create entities
    await create_entities(client)

    # Create relations
    await create_relations(client)

    # Search the graph
    await search_graph(client, "Who uses Python?")

    # Get all memories
    await get_all_memories(client)

    # Disconnect the client
    await client.disconnect()


async def configure_project(client: Client, project_name: str) -> None:
    """Configure the project name."""
    print(f"\n-- Configuring project: {project_name} --")
    
    config = {
        "provider": "openai",  # Can be changed to use a different provider
        "project_name": project_name,
    }
    
    result = await client.call_tool("configure_embedding", config)
    print(json.dumps(json.loads(result), indent=2))


async def create_entities(client: Client) -> None:
    """Create sample entities in the graph."""
    print("\n-- Creating Entities --")
    
    entities = [
        {
            "name": "John",
            "entityType": "Person",
            "observations": ["is 30 years old", "lives in New York", "is a developer"]
        },
        {
            "name": "Python",
            "entityType": "ProgrammingLanguage",
            "observations": ["is popular for data science", "is easy to learn"]
        },
        {
            "name": "VSCode",
            "entityType": "Software",
            "observations": ["is a code editor", "supports many extensions"]
        }
    ]
    
    result = await client.call_tool("create_entities", {"entities": entities})
    print(json.dumps(json.loads(result), indent=2))


async def create_relations(client: Client) -> None:
    """Create sample relations between entities."""
    print("\n-- Creating Relations --")
    
    relations = [
        {
            "from": "John",
            "to": "Python",
            "relationType": "USES"
        },
        {
            "from": "John",
            "to": "VSCode",
            "relationType": "USES"
        }
    ]
    
    result = await client.call_tool("create_relations", {"relations": relations})
    print(json.dumps(json.loads(result), indent=2))


async def search_graph(client: Client, query: str) -> None:
    """Search for entities in the graph."""
    print(f"\n-- Searching: '{query}' --")
    
    result = await client.call_tool("search_nodes", {"query": query})
    print(json.dumps(json.loads(result), indent=2))


async def get_all_memories(client: Client) -> None:
    """Get all memories from the graph."""
    print("\n-- Getting All Memories --")
    
    result = await client.call_tool("get_all_memories", {})
    print(json.dumps(json.loads(result), indent=2))


async def delete_all(client: Client) -> None:
    """Delete all data in the graph."""
    print("\n-- Deleting All Data --")
    
    result = await client.call_tool("delete_all_memories", {})
    print(json.dumps(json.loads(result), indent=2))


if __name__ == "__main__":
    asyncio.run(main()) 