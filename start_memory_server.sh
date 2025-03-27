#!/bin/bash

# Script to start the Neo4j MCP Graph Memory Server

# Default to stdio mode
USE_SSE=${USE_SSE:-false}
PORT=${PORT:-8080}
EMBEDDER_PROVIDER=${EMBEDDER_PROVIDER:-none}

# Load environment variables from .env if it exists
if [ -f .env ]; then
    echo "Loading environment variables from .env file"
    export $(cat .env | grep -v '#' | xargs)
fi

# Check embedding dependencies for the configured provider
if [ "$EMBEDDER_PROVIDER" != "none" ] && [ "$EMBEDDER_PROVIDER" != "" ]; then
    echo "Using embedder provider: $EMBEDDER_PROVIDER"
    
    case "$EMBEDDER_PROVIDER" in
        "openai")
            if [ -z "$OPENAI_API_KEY" ]; then
                echo "Warning: OPENAI_API_KEY is not set. Embeddings will not work correctly."
            fi
            ;;
        "azure" | "azure_openai")
            if [ -z "$AZURE_API_KEY" ] || [ -z "$AZURE_DEPLOYMENT" ] || [ -z "$AZURE_ENDPOINT" ]; then
                echo "Warning: One or more Azure OpenAI configuration variables are missing. Embeddings may not work correctly."
            fi
            ;;
        "huggingface")
            echo "Installing sentence-transformers may be needed if not already installed."
            echo "Run: ./install_embedding_deps.sh"
            echo "Or: pip install sentence-transformers>=2.2.2"
            ;;
        "vertexai")
            if [ -z "$GOOGLE_APPLICATION_CREDENTIALS" ]; then
                echo "Warning: GOOGLE_APPLICATION_CREDENTIALS is not set. Embeddings will not work correctly."
            fi
            echo "Installing vertexai dependencies may be needed if not already installed."
            echo "Run: ./install_embedding_deps.sh"
            echo "Or: pip install google-cloud-aiplatform>=1.26.0 vertexai>=0.0.1"
            ;;
        "gemini")
            if [ -z "$GOOGLE_API_KEY" ]; then
                echo "Warning: GOOGLE_API_KEY is not set. Embeddings will not work correctly."
            fi
            echo "Installing google-generativeai may be needed if not already installed."
            echo "Run: ./install_embedding_deps.sh"
            echo "Or: pip install google-generativeai>=0.3.0"
            ;;
        *)
            echo "Unknown embedder provider: $EMBEDDER_PROVIDER"
            echo "Supported providers: openai, azure, huggingface, vertexai, gemini, none"
            ;;
    esac
else
    echo "Embeddings are disabled. Using basic mode."
fi

# Start server with appropriate options
if [ "$USE_SSE" = "true" ]; then
    echo "Starting Neo4j MCP Graph Memory Server in SSE mode on port $PORT"
    python neo4j_mcp_server.py
else
    echo "Starting Neo4j MCP Graph Memory Server in stdio mode"
    python neo4j_mcp_server.py
fi 