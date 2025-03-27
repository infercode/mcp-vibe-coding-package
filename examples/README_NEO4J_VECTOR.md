# Neo4j Vector Store with LiteLLM Integration

This example demonstrates how to use Neo4j's vector capabilities with LiteLLM for flexible embedding generation across multiple providers.

## Setup

1. Install the required dependencies using Poetry:

```bash
# From the project root
poetry add litellm langchain langchain-neo4j python-dotenv
```

2. Create a `.env` file in the root directory with your Neo4j credentials and OpenAI API key:

```
NEO4J_URI=bolt://localhost:7687
NEO4J_USERNAME=neo4j
NEO4J_PASSWORD=your-password
NEO4J_DATABASE=neo4j

# LiteLLM supports many embedding providers, including:
EMBEDDING_MODEL=openai/text-embedding-3-small  # Format: provider/model
OPENAI_API_KEY=sk-your-openai-key

# For other providers:
# EMBEDDING_MODEL=mistral/mistral-embed
# MISTRAL_API_KEY=your-mistral-key

# EMBEDDING_MODEL=cohere/embed-english-v3.0
# COHERE_API_KEY=your-cohere-key
```

3. Make sure Neo4j is running and accessible with the credentials you've provided.

## Usage

Run the example script using Poetry:

```bash
# From the project root
poetry run python examples/neo4j_vector_litellm.py
```

## How It Works

The integration consists of three main components:

1. **LiteLLMEmbeddings**: A LangChain-compatible wrapper for LiteLLM that allows us to use any embedding provider supported by LiteLLM.

2. **Neo4jVector**: LangChain's vector store implementation for Neo4j that handles storing and retrieving embeddings.

3. **Example Application**: Shows how to use these components together for semantic search.

### Benefits

- **Unified Provider Interface**: Use any embedding model from OpenAI, Cohere, Mistral, etc. without changing your code.
- **Vector Storage in Neo4j**: Leverage Neo4j's native vector search capabilities.
- **Metadata Filtering**: Filter search results based on document metadata.
- **Combined with Graph Features**: Utilize both vector similarity and graph relationships in your queries.

## Customizing the Embeddings

You can easily switch embedding providers by changing the `EMBEDDING_MODEL` and corresponding API key environment variables. LiteLLM supports many providers, including:

- OpenAI: `openai/text-embedding-3-small`
- Mistral: `mistral/mistral-embed`
- Cohere: `cohere/embed-english-v3.0`
- Google: `gemini/text-embedding-004`
- Azure OpenAI: `azure/<your-deployment-name>`

## Advanced Neo4j Vector Usage

Neo4jVector supports advanced features like:

- Hybrid search (combining vector similarity with keyword search)
- Custom Cypher queries for retrieval
- Filtering by metadata
- MMR (Maximum Marginal Relevance) search for diversity

Check the [LangChain Neo4j documentation](https://python.langchain.com/api_reference/neo4j/vectorstores/langchain_neo4j.vectorstores.neo4j_vector.Neo4jVector.html) for more details. 