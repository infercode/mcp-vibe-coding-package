#!/bin/bash

# Script to start the Mem0 MCP Graph Memory Server

# Default to stdio mode
USE_SSE=${USE_SSE:-false}
PORT=${PORT:-8080}

# Load environment variables from .env if it exists
if [ -f .env ]; then
    echo "Loading environment variables from .env file"
    export $(cat .env | grep -v '#' | xargs)
fi

# Check if OpenAI API key is set
if [ -z "$OPENAI_API_KEY" ]; then
    echo "Warning: OPENAI_API_KEY is not set. You will need to set it for embeddings to work correctly."
fi

# Start server with appropriate options
if [ "$USE_SSE" = "true" ]; then
    echo "Starting Mem0 MCP Graph Memory Server in SSE mode on port $PORT"
    python mem0_mcp_server.py
else
    echo "Starting Mem0 MCP Graph Memory Server in stdio mode"
    python mem0_mcp_server.py
fi 