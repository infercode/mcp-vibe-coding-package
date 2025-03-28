# MCP Graph Memory Project Understanding

## Core Purpose
This is an MCP (Model Context Protocol) server that provides AI agents with persistent, structured memory using a Neo4j graph database backend. It implements two sophisticated memory systems that mimic human-like knowledge organization:

1. **Lesson Memory System**: Records experiential knowledge with confidence levels, versioning, and temporal tracking
2. **Project Memory System**: Organizes hierarchical project knowledge mirroring human mental models

## Technical Architecture

### Server Implementation
- Implements the Model Context Protocol (MCP) using the FastMCP SDK
- Offers dual interfaces:
  - **Server-Sent Events (SSE)** for web clients with non-disruptive reinitialization
  - **Standard I/O (stdio)** for terminal/CLI interaction
- Built as an MCP server exposing tools, resources, and prompts to AI agents

### Database Layer
- Uses **Neo4j** as the graph database with entity-relationship model
- Implements vector search through **Neo4jVector** integration
- Handles connection pooling, retry mechanisms, and failure recovery
- Supports vector embedding with multiple indexing strategies

### Embedding System
- Multi-provider embedding support via **LiteLLM**:
  - OpenAI, Azure OpenAI, HuggingFace, Vertex AI, Gemini, Mistral, Ollama, LMStudio
- Configurable embedding dimensions and parameters
- Fallback to text-based search when embeddings are disabled
- Custom LangChain integration via the `LiteLLMEmbeddings` class

## Memory Systems

### Lesson Memory
- **Container Structure**: Global container for cross-project lessons
- **Entity Structure**: Lessons with confidence scores, versioning, impact assessment
- **Observation Types**: Structured as "What Was Learned", "Why It Matters", "How To Apply", etc.
- **Relationship Types**: 9+ relationship types including ORIGINATED_FROM, SOLVED_WITH, BUILDS_ON
- **Advanced Features**:
  - Lesson versioning with SUPERSEDES relationships
  - Confidence-based knowledge retrieval
  - Temporal intelligence for knowledge evolution tracking
  - Memory consolidation for combining related lessons

### Project Memory
- **Hierarchical Structure**: Project → Domain → Component/Feature/Decision
- **Entity Types**: 8+ specialized types with confidence and temporal properties
- **Relationship Types**: 10+ relationship types with metadata (CONTAINS, IMPLEMENTS, DEPENDS_ON, etc.)
- **Advanced Features**:
  - Entity evolution tracking through version history
  - Cross-project knowledge transfer
  - Automated structure discovery from code/documentation
  - Project-Lesson integration for applied knowledge

## Technical Implementation Details

### GraphMemoryManager Class
- Monolithic implementation (~3,654 lines) handling all functionality
- Manages Neo4j connection, embeddings, and all memory operations
- Implements Neo4j query safety mechanisms for LiteralString type requirements
- Handles parameter type conversion for Neo4j compatibility

### MCP Tool Implementations
- Creates entities, relationships, and observations via MCP tools
- Exposes search capabilities to AI agents
- Provides specialized tools for lesson and project memory operations
- Handles serialization between Neo4j and MCP protocol

### Data Flow
1. AI agent calls MCP tools to create/query memory entities
2. Server processes requests via GraphMemoryManager
3. Data is stored in Neo4j with optional vector embeddings
4. Results are returned following MCP protocol specs

## Development Status
- Implementation is complete for core functionality
- Both memory systems have documented implementations
- Project memory system folder exists but appears empty (likely pending migration from the monolithic file)
- Refactoring needed for maintenance and extensibility

This project represents a sophisticated approach to providing AI agents with persistent, structured, confidence-scored memory systems that mirror human knowledge organization patterns. 