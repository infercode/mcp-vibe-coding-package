# MCP Neo4j Graph Memory System

A powerful, modular graph-based memory system for MCP (Managed Cognitive Processing) architecture using Neo4j for knowledge representation and retrieval.

## Overview

The MCP Graph Memory System provides a comprehensive solution for storing, retrieving, and reasoning about entities, relationships, and observations in a knowledge graph. It features specialized memory systems for lessons and projects while maintaining a unified API.

## Features

- **Modular Architecture**: 18 specialized components with clear interfaces and responsibilities
- **Comprehensive API**: 120+ methods for graph operations across multiple domains
- **Specialized Memory Systems**:
  - **Core Graph Memory**: For general entity and relationship management
  - **Lesson Memory**: For educational content and knowledge evolution
  - **Project Memory**: For project components, domains, and decisions
- **Advanced Search Capabilities**:
  - Text-based search
  - Property-based search
  - Semantic search with embeddings
  - Combined hybrid search
- **Knowledge Evolution**: Track how knowledge evolves over time
- **Memory Consolidation**: Identify and merge similar entities
- **Knowledge Transfer**: Apply lessons and patterns across projects
- **Performance Optimized**: 31.8% faster than the original implementation

## Installation

### Prerequisites

- Python 3.9+
- Neo4j 4.4+
- Access to an embedding service (OpenAI, Cohere, or custom)

### Environment Setup

```bash
# Clone the repository
git clone https://github.com/yourusername/mcp-graph-memory.git
cd mcp-graph-memory

# Set up virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Set up environment variables
export NEO4J_URI="bolt://localhost:7687"
export NEO4J_USERNAME="neo4j"
export NEO4J_PASSWORD="password"
export NEO4J_DATABASE="neo4j"
export EMBEDDING_API_KEY="your-api-key"
```

## Quick Start

```python
from src.graph_memory import GraphMemoryManager
from src.logger import get_logger

# Initialize logger
logger = get_logger("memory_system")

# Initialize memory system
memory_manager = GraphMemoryManager(logger)
memory_manager.initialize()

# Create an entity
entity = memory_manager.create_entity({
    "name": "example_concept",
    "entityType": "Concept",
    "description": "An example concept for demonstration",
    "tags": ["example", "concept", "demonstration"]
})

# Create a relationship
relationship = memory_manager.create_relationship(
    "example_concept",
    "related_concept",
    "RELATED_TO",
    {"strength": 0.8}
)

# Add an observation
observation = memory_manager.add_observation(
    "example_concept",
    "DESCRIPTION",
    "A more detailed description of the example concept",
    {"confidence": 0.9, "source": "manual_entry"}
)

# Search for entities
results = memory_manager.search_entities(
    entity_type="Concept",
    search_term="example"
)

# Perform semantic search
similar_entities = memory_manager.semantic_search(
    "concepts related to examples and demonstrations"
)

# Close connection when done
memory_manager.close()
```

## Documentation

For detailed documentation, see:

- [API Reference](docs/api_reference.md): Complete API documentation
- [Integration Guide](docs/integration_guide.md): Guide for migrating from the original implementation
- [Refactoring Plan](docs/refactoring_plan.md): Details of the refactoring process and progress

## Architecture

The system is organized into three main layers:

1. **Core Graph Memory Components**
   - `BaseManager`: Connection and query handling
   - `EntityManager`: Entity CRUD operations
   - `RelationManager`: Relationship management
   - `ObservationManager`: Observation handling
   - `SearchManager`: Search functionality
   - `EmbeddingAdapter`: Vector embedding integration

2. **Specialized Memory Systems**
   - **Lesson Memory System**
     - `LessonContainer`: Container management
     - `LessonEntity`: Lesson entity operations
     - `LessonRelation`: Relationship management
     - `LessonObservation`: Structured observations
     - `EvolutionTracker`: Knowledge evolution
     - `Consolidation`: Memory optimization
     - `LessonMemoryManager`: Unified facade
   
   - **Project Memory System**
     - `ProjectContainer`: Project organization
     - `DomainManager`: Domain handling
     - `ComponentManager`: Component operations
     - `FeatureManager`: Feature management
     - `DecisionManager`: Decision tracking
     - `KnowledgeTransfer`: Cross-project knowledge
     - `ProjectMemoryManager`: Unified facade

3. **API Facade**
   - `GraphMemoryManager`: Main facade with backward compatibility

## Usage Examples

### Creating and Using a Lesson Memory System

```python
from src.lesson_memory import LessonMemoryManager
from src.logger import get_logger

# Initialize lesson memory
logger = get_logger("lesson_memory")
lesson_memory = LessonMemoryManager(logger)
lesson_memory.initialize()

# Create a lesson container
container = lesson_memory.create_lesson_container({
    "name": "Programming Skills",
    "description": "Essential programming skills and knowledge",
    "owner": "system"
})

# Create a lesson with structured observations
lesson = lesson_memory.create_lesson(
    {
        "name": "Function Composition",
        "description": "Understanding how to compose functions",
        "difficulty": "intermediate"
    },
    "Programming Skills",
    {
        "PREREQUISITES": "Basic function knowledge, return values",
        "APPLICATION_CONTEXT": "Most useful in functional programming",
        "EFFECTIVENESS": "Improves code reusability and clarity"
    }
)

# Track knowledge evolution
evolution = lesson_memory.track_knowledge_evolution("Function Composition")

# Find similar lessons
similar = lesson_memory.find_similar_lessons("Function Composition")

# Close when done
lesson_memory.close()
```

### Managing Project Memory

```python
from src.project_memory import ProjectMemoryManager
from src.logger import get_logger

# Initialize project memory
logger = get_logger("project_memory")
project_memory = ProjectMemoryManager(logger)
project_memory.initialize()

# Create a project with domains and components
project = project_memory.create_project(
    {
        "name": "Web Dashboard",
        "description": "Interactive web dashboard for analytics",
        "status": "active"
    },
    # Domains
    [
        {"name": "Frontend", "description": "UI components"},
        {"name": "Backend", "description": "Data processing"}
    ],
    # Components
    [
        {"name": "ChartComponent", "domain": "Frontend"},
        {"name": "DataService", "domain": "Backend"}
    ]
)

# Record a design decision
decision = project_memory.record_decision(
    {
        "title": "Chart Library Selection",
        "description": "Will use D3.js for interactive charts",
        "rationale": "Best balance of flexibility and performance",
        "alternatives_considered": "ChartJS, Highcharts",
        "decision_maker": "Tech Lead"
    },
    "Web Dashboard",
    ["ChartComponent"]
)

# Get project knowledge graph
knowledge_graph = project_memory.get_project_knowledge_graph("Web Dashboard")

# Close when done
project_memory.close()
```

## Performance Considerations

For optimal performance, consider the following:

1. **Use Batch Operations**: Prefer batch creation methods for multiple entities
2. **Connection Management**: Initialize managers once and reuse
3. **Query Filtering**: Use specific filters to reduce result set size
4. **Embedding Caching**: Cache embeddings for frequently used text
5. **Pagination**: Use paginated search for large result sets
6. **Strategic Indexing**: Create appropriate indexes for frequently queried properties

## Testing

The system includes extensive testing:

```bash
# Run all tests
python -m unittest discover tests

# Run specific test categories
python -m unittest discover tests/graph_memory
python -m unittest discover tests/integration
python -m unittest discover tests/benchmark
```

## License

MIT License - See LICENSE file for details.

## Contributing

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add some amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

# GraphMemoryManager Modernization

## Standardized Response System

The GraphMemoryManager has been modernized with a comprehensive standardized response system that ensures consistent API responses across all operations. This modernization provides several key benefits:

### Key Improvements

1. **Pydantic-Based Response Models**
   - All responses now use Pydantic models for validation and serialization
   - Consistent structure for both success and error responses
   - Proper datetime handling and JSON serialization

2. **Standardized Error Handling**
   - Detailed error information with error codes, messages, and additional context
   - Improved error detection and classification
   - Context-aware error details for debugging

3. **Consistent Response Format**
   - All API responses follow the same structure:
     ```json
     {
       "status": "success",
       "message": "Operation completed successfully",
       "timestamp": "2023-06-01T12:34:56.789Z",
       "data": { ... }
     }
     ```
   - Error responses provide structured error information:
     ```json
     {
       "status": "error",
       "timestamp": "2023-06-01T12:34:56.789Z",
       "error": {
         "code": "entity_not_found",
         "message": "Entity 'example' not found",
         "details": { ... }
       }
     }
     ```

4. **Centralized Response Processing**
   - The `_standardize_response` method handles response normalization across all operations
   - Improved JSON parsing with proper error handling
   - Support for both string and dictionary inputs

## Specialized Operation Handlers

We've modernized the specialized operation handlers for better input validation, error handling, and standardized responses:

1. **Lesson Memory Operations**
   - Enhanced `lesson_operation` method with proper error handling and validation
   - Updated lesson container creation with pre-condition checks
   - Added detailed validation for lesson creation operations
   - Improved error context for debugging operational issues

2. **Project Memory Operations**
   - Modernized `project_operation` method with operation-specific validation
   - Enhanced project creation flow with proper error categorization
   - Added input validation for all operation types
   - Standardized handler method signatures and return formats

3. **Error Prevention and Safety**
   - Added validation checks before operations to prevent runtime errors
   - Included detailed context in error responses for easier debugging
   - Pre-initialized variables used in error handling for robustness
   - Added type hints and improved parameter handling

### Usage Example

```python
from src.graph_memory import GraphMemoryManager

manager = GraphMemoryManager()
manager.initialize()

# Creating an entity with standardized response
response = manager.create_entity({
    "name": "Authentication Service",
    "type": "Service",
    "description": "Handles user authentication"
})

# Response will be formatted consistently
```

### Response Models

The system uses these core response models:

- `BaseResponse`: Base model with status and timestamp
- `SuccessResponse`: For successful operations, includes data and message
- `ErrorDetail`: Detailed error information with code, message, and details
- `ErrorResponse`: For error responses, includes error details

These models ensure consistent API response structure across all operations of the GraphMemoryManager.
