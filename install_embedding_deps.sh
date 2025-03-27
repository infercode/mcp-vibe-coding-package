#!/bin/bash

# Script to install necessary embedding dependencies

# Default is not to install any embedding dependencies
EMBEDDING_PROVIDER=${EMBEDDING_PROVIDER:-none}

echo "Installing Neo4j Memory MCP with embedding provider: $EMBEDDING_PROVIDER"

# Base installation
poetry install

# Install specific dependencies based on provider
case "$EMBEDDING_PROVIDER" in
  "openai")
    echo "No additional dependencies needed for OpenAI"
    ;;
  "azure" | "azure_openai")
    echo "No additional dependencies needed for Azure OpenAI"
    ;;
  "huggingface")
    echo "Installing HuggingFace dependencies"
    poetry install --extras "huggingface"
    ;;
  "vertexai")
    echo "Installing VertexAI dependencies"
    poetry install --extras "vertexai"
    ;;
  "gemini")
    echo "Installing Gemini dependencies"
    poetry install --extras "gemini"
    ;;
  "all")
    echo "Installing all embedding dependencies"
    poetry install --extras "all-embeddings"
    ;;
  "none")
    echo "No embedding dependencies needed"
    ;;
  *)
    echo "Unknown embedding provider: $EMBEDDING_PROVIDER"
    echo "Supported providers: openai, azure, huggingface, vertexai, gemini, all, none"
    ;;
esac

echo "Installation complete!" 