#!/usr/bin/env python3
"""
Example of using Neo4jVector with LiteLLM embeddings.
This script demonstrates how to:
1. Create a LiteLLM embeddings wrapper for LangChain
2. Connect to Neo4j and create a vector store
3. Add documents to the vector store
4. Perform similarity search
"""

import os
import sys
from dotenv import load_dotenv
from typing import List, Dict, Any

# Add the src directory to the path so we can import our modules
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

# Import the LiteLLM embeddings wrapper
from src.litellm_langchain import LiteLLMEmbeddings

# Import Neo4jVector from LangChain
from langchain_neo4j import Neo4jVector
from langchain.docstore.document import Document

# Load environment variables from .env file
load_dotenv()

# Neo4j connection parameters
NEO4J_URI = os.getenv("NEO4J_URI", "bolt://localhost:7687")
NEO4J_USERNAME = os.getenv("NEO4J_USERNAME", "neo4j")
NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD", "P@ssW0rd2025!")
NEO4J_DATABASE = os.getenv("NEO4J_DATABASE", "neo4j")

# Embedding model parameters
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "openai/text-embedding-3-small")
EMBEDDING_API_KEY = os.getenv("OPENAI_API_KEY", "")

def main():
    """Main function to demonstrate Neo4jVector with LiteLLM embeddings."""
    print(f"Connecting to Neo4j at {NEO4J_URI}...")
    
    # Initialize the LiteLLM embeddings wrapper
    embeddings = LiteLLMEmbeddings(
        model=EMBEDDING_MODEL,
        api_key=EMBEDDING_API_KEY,
        dimensions=1536  # Optional: specify dimensions for the embedding model
    )
    
    # Create or connect to an existing vector store
    vector_store = Neo4jVector(
        embedding=embeddings,
        url=NEO4J_URI,
        username=NEO4J_USERNAME,
        password=NEO4J_PASSWORD,
        database=NEO4J_DATABASE,
        index_name="document_embeddings",
        node_label="Document",
        text_node_property="content",
        pre_delete_collection=True  # Set to False in production
    )
    
    print("Connected to Neo4j and initialized vector store.")
    
    # Example documents
    documents = [
        Document(page_content="Neo4j is a graph database management system developed by Neo4j, Inc.", 
                 metadata={"source": "wiki", "category": "database"}),
        Document(page_content="Graph databases are designed to handle highly connected data efficiently.", 
                 metadata={"source": "textbook", "category": "database"}),
        Document(page_content="Vector embeddings represent text as numerical vectors for semantic search.", 
                 metadata={"source": "article", "category": "machine_learning"}),
        Document(page_content="LiteLLM provides a unified API for various LLM providers.", 
                 metadata={"source": "documentation", "category": "library"}),
        Document(page_content="LangChain is a framework for building applications with LLMs through composability.", 
                 metadata={"source": "github", "category": "framework"})
    ]
    
    # Add documents to the vector store
    print(f"Adding {len(documents)} documents to the vector store...")
    vector_store.add_documents(documents)
    print("Documents added successfully.")
    
    # Perform similarity search
    query = "How do graph databases work with vector embeddings?"
    k = 3  # Number of results to return
    
    print(f"\nPerforming similarity search for: '{query}'")
    results = vector_store.similarity_search(query, k=k)
    
    print(f"\nTop {k} results:")
    for i, doc in enumerate(results):
        print(f"\n{i+1}. Content: {doc.page_content}")
        print(f"   Metadata: {doc.metadata}")
    
    # Clean up (optional)
    # Uncomment the following line to delete the vector store after the example
    # vector_store.delete()
    
if __name__ == "__main__":
    main() 