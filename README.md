# MCP Graph Memory Server with Mem0ai

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python](https://img.shields.io/badge/Python-3.9+-blue)](https://www.python.org/)
[![Neo4j](https://img.shields.io/badge/Neo4j-5.x-brightgreen)](https://neo4j.com/)
[![mem0ai](https://img.shields.io/badge/mem0ai-0.1.77+-purple)](https://github.com/mem0ai/mem0)
[![MCP](https://img.shields.io/badge/MCP-1.5.0+-orange)](https://github.com/modelcontextprotocol/python-sdk)

## Introduction

MCP Graph Memory Server is a knowledge graph memory server based on the mem0ai library with Neo4j graph database backend. It's designed to store and retrieve information during the interaction between AI assistants and users, providing powerful memory capabilities for AI agents.

This project implements the [Model Context Protocol (MCP)](https://github.com/modelcontextprotocol/python-sdk) and uses [mem0ai](https://github.com/mem0ai/mem0) with its graph capabilities to provide a robust knowledge graph memory solution.

## Features

- 🚀 High-performance graph database storage based on Neo4j and mem0ai
- 🔍 Powerful fuzzy search and exact matching capabilities
- 🔄 Complete CRUD operations for entities, relationships, and observations
- 🌐 Fully compatible with the MCP protocol
- 📊 Supports complex graph queries and traversals
- 🐳 Docker support for easy deployment
- 📝 Rich metadata support for better context organization
- 📈 Hybrid memory architecture combining mem0ai and direct Neo4j operations
- 🔧 Multiple embedding model support (OpenAI, Hugging Face, Ollama, Azure, LM Studio)
- 🔀 Cross-project memory search for knowledge sharing

## Installation

### Prerequisites

- Python 3.9+
- Poetry (dependency management)
- Neo4j database (local or remote)
- OpenAI API Key (for embeddings) or alternative embedding provider

### Installation Steps

1. Clone the repository and install dependencies with Poetry:

```bash
git clone https://github.com/your-username/mcp-graph-memory-server.git
cd mcp-graph-memory-server
poetry install
```

This will install all required dependencies, including:
- `mem0ai[graph]>=0.1.77`
- `neo4j>=5.0.0`
- `fastmcp`

2. Set up a Neo4j database (locally or using Neo4j AuraDB cloud service)

3. Configure the environment variables (see below)

### Environment Variable Configuration

The server is configured using the following environment variables:

```bash
# Neo4j Connection Settings
NEO4J_URI=bolt://localhost:7687  # Neo4j connection URI
NEO4J_USER=neo4j                 # Neo4j username
NEO4J_PASSWORD=password          # Neo4j password
NEO4J_DATABASE=neo4j             # Neo4j database name
```

#### Embedding Configuration

By default, the server uses OpenAI embeddings. You can configure various embedding providers using the `EMBEDDER_PROVIDER` environment variable:

##### OpenAI (Default)

```bash
EMBEDDER_PROVIDER=openai
OPENAI_API_KEY=your_openai_api_key     # OpenAI API key
OPENAI_API_BASE=                       # Optional: Custom OpenAI API base URL
EMBEDDING_MODEL=text-embedding-3-small # OpenAI embedding model
EMBEDDING_DIMS=1536                    # Dimensions of the embedding model
```

##### Alternative Embedding Providers

The server supports multiple embedding providers that can be configured through environment variables:

**Hugging Face**

```bash
EMBEDDER_PROVIDER=huggingface
HUGGINGFACE_MODEL=sentence-transformers/all-mpnet-base-v2
HUGGINGFACE_MODEL_KWARGS={"device":"cpu"}  # Optional: Custom model parameters
EMBEDDING_DIMS=768
```

**Ollama**

```bash
EMBEDDER_PROVIDER=ollama
OLLAMA_MODEL=llama2
OLLAMA_BASE_URL=http://localhost:11434  # Optional: Custom Ollama API URL
EMBEDDING_DIMS=4096
```

**Azure OpenAI**

```bash
EMBEDDER_PROVIDER=azure_openai
AZURE_MODEL=text-embedding-3-small
EMBEDDING_AZURE_OPENAI_API_KEY=your_azure_api_key
EMBEDDING_AZURE_ENDPOINT=https://your-resource.openai.azure.com
EMBEDDING_AZURE_DEPLOYMENT=your-deployment-name
EMBEDDING_AZURE_API_VERSION=2023-05-15
EMBEDDING_AZURE_DEFAULT_HEADERS={"CustomHeader":"your-custom-header"}
EMBEDDING_DIMS=1536
```

**LM Studio**

```bash
EMBEDDER_PROVIDER=lmstudio
LMSTUDIO_BASE_URL=http://localhost:1234
EMBEDDING_DIMS=4096
```

**VertexAI (Google Cloud)**

```bash
EMBEDDER_PROVIDER=vertexai
VERTEXAI_MODEL=text-embedding-004
GOOGLE_APPLICATION_CREDENTIALS=/path/to/your/credentials.json
VERTEXAI_MEMORY_ADD_EMBEDDING_TYPE=RETRIEVAL_DOCUMENT
VERTEXAI_MEMORY_UPDATE_EMBEDDING_TYPE=RETRIEVAL_DOCUMENT
VERTEXAI_MEMORY_SEARCH_EMBEDDING_TYPE=RETRIEVAL_QUERY
EMBEDDING_DIMS=256
```

**Gemini**

```bash
EMBEDDER_PROVIDER=gemini
GEMINI_MODEL=models/text-embedding-004
GOOGLE_API_KEY=your_google_api_key
EMBEDDING_DIMS=768
```

The embedding types can be one of:
- SEMANTIC_SIMILARITY
- CLASSIFICATION 
- CLUSTERING
- RETRIEVAL_DOCUMENT, RETRIEVAL_QUERY
- QUESTION_ANSWERING, FACT_VERIFICATION
- CODE_RETRIEVAL_QUERY

#### Server Configuration

```bash
# Server Mode Settings
USE_SSE=false                    # Enable SSE mode (true/false)
PORT=8080                        # Port for SSE server (when USE_SSE=true)
```

## Project-Based Memory

The server organizes memories by project. The AI agent specifies the project name when configuring the embedding provider:

```json
{
  "provider": "openai",
  "model": "text-embedding-3-small",
  "project_name": "my-awesome-project"
}
```

This project-based approach enables:

- **Compartmentalized knowledge**: Each project has its own memory space
- **Cross-project searching**: Query information from other projects when needed
- **Contextual relevance**: Information stays organized and relevant to specific projects

### Cross-Project Memory Sharing

The server enables AI agents to access and learn from memory across different projects. This is particularly useful for:

- **Reusing solutions**: Find how similar problems were solved in other projects
- **Sharing dependencies**: Discover what packages are used in related projects
- **Learning from mistakes**: Review lessons learned and avoid repeating errors
- **Identifying patterns**: Recognize common approaches across multiple projects

## Server Modes

### Standard Mode (stdio)

By default, the server operates in stdio mode, which allows it to communicate through standard input/output. This is the default mode used when integrating with CLI tools or when running the server directly.

### SSE Mode

The server also supports Server-Sent Events (SSE) mode, which allows it to communicate over HTTP using the SSE protocol. This mode is useful when you need to integrate the memory server with web applications or services that require HTTP communication.

To enable SSE mode, set the `USE_SSE` environment variable to `true`:

```bash
# Enable SSE mode
USE_SSE=true
PORT=8080  # Optional, defaults to 8080
```

When running in SSE mode, the server will be accessible at:
```
http://localhost:8080/sse
```

## Architecture

The MCP Graph Memory Server uses a hybrid memory architecture:

### Mem0ai Graph Integration

- Primary storage mechanism for entities, relations, and observations
- Natural language formatting with rich metadata
- Automatic entity and relation extraction
- Vector-based semantic search capabilities
- Support for multiple embedding providers (OpenAI, Hugging Face, Ollama, etc.)

### Direct Neo4j Integration

- Used for operations not directly supported by mem0
- Enables selective deletion of entities, relations, and observations
- Allows for complex graph queries and traversals
- Provides direct access to the graph database for advanced operations

This hybrid approach offers both the simplicity of mem0ai and the power of direct Neo4j operations, giving you the best of both worlds.

## Usage

### Starting the Server

To start the server in stdio mode:

```bash
poetry run python mem0_mcp_server.py
```

To start the server in SSE mode:

```bash
USE_SSE=true PORT=8080 poetry run python mem0_mcp_server.py
```

### Using with Claude

Add the following to Claude's custom instructions to make the best use of this memory server:

```
Follow these steps for each interaction:

1. User Identification:
   - You should assume that you are interacting with default_user.
   - If you have not identified default_user, proactively try to do so.

2. Memory Retrieval:
   - Always begin your chat by saying only "Remembering..." and search relevant information from your knowledge graph.
   - Create a search query from user words, and search things from "memory". If nothing matches, try to break down words in the query at first ("A B" to "A" and "B" for example).
   - Always refer to your knowledge graph as your "memory".

3. Memory:
   - While conversing with the user, be attentive to any new information that falls into these categories:
     a) Basic Identity (age, gender, location, job title, education level, etc.)
     b) Behaviors (interests, habits, etc.)
     c) Preferences (communication style, preferred language, etc.)
     d) Goals (goals, targets, aspirations, etc.)
     e) Relationships (personal and professional relationships up to 3 degrees of separation).

4. Memory Update:
   - If any new information was gathered during the interaction, update your memory as follows:
     a) Create entities for recurring organizations, people, and significant events.
     b) Connect them to the current entities using relations.
     c) Store facts about them as observations.
```

## API Examples

### Creating Entities

```python
from mcp import Client
from mcp.client.stdio import StdioClientTransport

# Create a client transport
transport = StdioClientTransport(
    command="poetry",
    args=["run", "python", "mem0_mcp_server.py"],
    env={
        "NEO4J_URI": "bolt://localhost:7687",
        "NEO4J_USER": "neo4j",
        "NEO4J_PASSWORD": "password",
        "NEO4J_DATABASE": "neo4j"
    }
)

# Create and connect the client
client = Client(name="example-client", version="1.0.0")
await client.connect(transport)

# Create entities
result = await client.call_tool("create_entities", {
    "entities": [
        {
            "name": "User",
            "entityType": "Person",
            "observations": ["Likes programming", "Uses Python"]
        }
    ]
})

print(result)
```

### Creating Relations

```python
# Create a relation between entities
result = await client.call_tool("create_relations", {
    "relations": [
        {
            "from": "User",
            "to": "Python",
            "relationType": "USES"
        }
    ]
})

print(result)
```

### Searching the Graph

```python
# Search for entities in the knowledge graph
result = await client.call_tool("search_nodes", {
    "query": "Who uses Python?"
})

print(result)
```

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgements

- This project was inspired by the [MCP Neo4j Knowledge Graph Memory Server](https://github.com/JovanHsu/mcp-neo4j-memory-server) TypeScript implementation
- [Model Context Protocol](https://github.com/modelcontextprotocol/python-sdk)
- [mem0ai](https://github.com/mem0ai/mem0)
- [Neo4j Python Driver](https://neo4j.com/docs/api/python-driver/current/)

## MCP Tools

The server provides the following MCP tools for interacting with the knowledge graph:

### Entity Operations

- **create_entities**: Create multiple entities in the knowledge graph
  ```json
  {
    "entities": [
      {
        "name": "John",
        "entityType": "Person",
        "observations": ["is 30 years old", "lives in New York", "works as an engineer"]
      }
    ]
  }
  ```
  *Under the hood*: Each entity is stored in both mem0's memory system and Neo4j with appropriate relationships.

- **delete_entity**: Delete a specific entity from the knowledge graph
  ```json
  {
    "entity": "John"
  }
  ```
  *Under the hood*: Removes the entity and all its relationships from Neo4j directly.

- **get_all_memories**: Retrieve all memories for a specific project
  ```json
  {
    "user_id": "project-name"  // Optional, defaults to current project
  }
  ```
  *Under the hood*: Uses mem0's `get_all()` function to retrieve all stored memories for the specified project.

### Relation Operations

- **create_relations**: Create relationships between entities
  ```json
  {
    "relations": [
      {
        "from": "John",
        "to": "Jane",
        "relationType": "knows"
      }
    ]
  }
  ```
  *Under the hood*: Creates relationships in both mem0 and Neo4j for comprehensive querying.

- **delete_relation**: Delete a specific relationship
  ```json
  {
    "from_entity": "John",
    "to_entity": "Jane",
    "relationType": "knows"
  }
  ```
  *Under the hood*: Removes the relationship from Neo4j directly.

### Observation Operations

- **add_observations**: Add new observations to entities
  ```json
  {
    "observations": [
      {
        "entity": "John",
        "content": "enjoys playing piano"
      }
    ]
  }
  ```
  *Under the hood*: Adds observations as memories in mem0 and as linked nodes in Neo4j.

- **delete_observation**: Delete a specific observation
  ```json
  {
    "entity": "John",
    "content": "enjoys playing piano"
  }
  ```
  *Under the hood*: Removes the observation from Neo4j directly.

### Search Operations

- **search_nodes**: Search for nodes matching a query, with cross-project capability
  ```json
  {
    "query": "What packages are used for API testing?",
    "limit": 10,  // Optional, defaults to 10
    "user_id": "other-project"  // Optional, search in a specific project
  }
  ```
  *Under the hood*: Uses mem0's semantic search with the specified project ID. When `user_id` is provided, it searches in that project's memory space instead of the current one.

### Memory Management

- **delete_all_memories**: Delete all memories for a project
  ```json
  {
    "user_id": "project-name"  // Optional, defaults to current project
  }
  ```
  *Under the hood*: Uses mem0's `delete_all()` function to clear all memories for the specified project.

### Configuration

- **configure_embedding**: Configure the embedding provider and set the project name
  ```json
  {
    "provider": "openai",
    "model": "text-embedding-3-small",
    "api_key": "your-api-key",
    "dimensions": 1536,
    "project_name": "my-awesome-project",  // Project name for memory operations
    "client_id": "client1",  // Optional, unique identifier for the client
    "config": {  // Optional, provider-specific configurations
      "api_base": "https://api.openai.com/v1"
    }
  }
  ```
  *Under the hood*: Configures the embedding provider, sets the project name for memory operations, and returns the full configuration for client-side storage.

- **get_embedding_config**: Get the current embedding configuration
  ```json
  {}  // No parameters required
  ```
  *Under the hood*: Returns the current mem0 configuration.

- **set_project_name**: Set the project name without reconfiguring embeddings
  ```json
  {
    "project_name": "my-new-project"
  }
  ```
  *Under the hood*: Updates the default user ID for memory operations without changing embedding settings. This is useful when you want to compartmentalize memory by project without needing to configure custom embeddings.

## Client-Side Configuration

The MCP server is designed to be stateless regarding configurations. When using the `configure_embedding` tool:

1. The client sends configuration preferences
2. The server validates and applies the configuration temporarily
3. The complete configuration is returned to the client with instructions for saving it
4. The AI agent should save the configuration locally as instructed

This approach provides several benefits:
- Clients maintain control over their configurations
- Different clients can use different configurations
- No configuration conflicts between clients
- Easy restoration of configurations after server restarts

### Response Format with AI Agent Instructions

When configuration tools are called, they return specific instructions for the AI agent along with the configuration data:

#### Configure Embedding Response

```json
{
  "status": "success",
  "message": "Successfully configured embedding provider: openai",
  "provider": "openai",
  "project_name": "my-awesome-project",
  "config": {
    "graph_store": { ... },
    "embedder": { ... }
  },
  "instructions_for_agent": "IMPORTANT: Save this configuration to 'mcp_memory_config_my-awesome-project.json' in the root directory of your project. This file will be used for future memory operations with the MCP server.",
  "file_name": "mcp_memory_config_my-awesome-project.json"
}
```

#### Set Project Name Response

```json
{
  "status": "success",
  "message": "Project name set to: my-awesome-project",
  "project_name": "my-awesome-project",
  "config": {
    "project_name": "my-awesome-project",
    "timestamp": "2023-05-15T14:32:10.123456"
  },
  "instructions_for_agent": "IMPORTANT: Save this project configuration to 'mcp_project_config_my-awesome-project.json' in the root directory of your project. This file will help maintain project context between sessions.",
  "file_name": "mcp_project_config_my-awesome-project.json"
}
```

### Implementing the Configuration Storage

When the AI agent receives these responses, it should:

1. Extract the configuration data or project name
2. Create a file in the project root with the provided `file_name`
3. Store the configuration in JSON format (both for embedding configurations and project settings)
4. Reference these files in subsequent sessions to maintain context

## Using Cross-Project Memory

To effectively use cross-project memory:

1. **Organize projects clearly**: Use distinctive project names that reflect their purpose.

2. **Search across projects**: Use the `user_id` parameter in `search_nodes` to query other projects:
   ```python
   # Search in a different project
   result = await client.call_tool("search_nodes", {
       "query": "What testing framework is used?",
       "user_id": "authentication-service"
   })
   ```

3. **Compare approaches**: Search multiple projects to compare how different teams solved similar problems:
   ```python
   # Search in current project
   current_result = await client.call_tool("search_nodes", {
       "query": "How is authentication implemented?"
   })
   
   # Search in another project
   other_result = await client.call_tool("search_nodes", {
       "query": "How is authentication implemented?",
       "user_id": "payment-service"
   })
   ```

4. **Learn from history**: Review lessons learned or bugs fixed in previous projects:
   ```python
   result = await client.call_tool("search_nodes", {
       "query": "Common JWT implementation mistakes",
       "user_id": "security-review"
   })
   ```

This cross-project memory capability creates a connected knowledge ecosystem where AI agents can leverage learnings across an entire organization's development efforts.
