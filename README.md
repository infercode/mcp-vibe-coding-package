*** Currently under development ***

# MCP Neo4j Graph Memory System

A powerful, modular graph-based memory system for MCP (Managed Cognitive Processing) architecture using Neo4j for knowledge representation and retrieval.

## Overview

The MCP Graph Memory System provides a comprehensive solution for storing, retrieving, and reasoning about entities, relationships, and observations in a knowledge graph. It features specialized memory systems for lessons and projects while maintaining a unified API.

## Features

- **Modular Architecture**: Specialized components with clear interfaces and responsibilities
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
- **Standardized Response System**: Consistent Pydantic-based API responses

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
relationship = memory_manager.create_relationship({
    "from_entity": "example_concept",
    "to_entity": "related_concept",
    "relation_type": "RELATED_TO",
    "properties": {"strength": 0.8}
})

# Add an observation
observation = memory_manager.add_observation({
    "entity_name": "example_concept",
    "observation_type": "DESCRIPTION",
    "content": "A more detailed description of the example concept",
    "properties": {"confidence": 0.9, "source": "manual_entry"}
})

# Search for entities
results = memory_manager.search_nodes(
    "example concept",
    entity_types=["Concept"],
    semantic=True,
    limit=10
)

# Close connection when done
memory_manager.close()
```

## Documentation

For detailed documentation, see:

- [API Reference](docs/api/api_reference.md): Complete API documentation
- [Integration Guide](docs/integration_guide.md): Guide for integrating with your application
- [Refactoring Plan](docs/refactoring_plan.md): Architecture and implementation details

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
     - `DependencyManager`: Dependency tracking
     - `VersionManager`: Version management
     - `ProjectMemoryManager`: Unified facade

3. **API Facade**
   - `GraphMemoryManager`: Main facade with backward compatibility

## Usage Examples

### Creating and Using a Lesson Memory System

```python
from src.graph_memory import GraphMemoryManager
from src.logger import get_logger

# Initialize memory system
logger = get_logger("memory_system")
memory_manager = GraphMemoryManager(logger)
memory_manager.initialize()

# Create a lesson container with a context manager
with memory_manager.lesson_context() as lesson_ctx:
    # Create a lesson with observations
    lesson = lesson_ctx.create(
        name="Function Composition",
        lesson_type="Programming",
        observations={
            "what_was_learned": "How to compose functions for cleaner code",
            "why_it_matters": "Improves code reusability and clarity",
            "how_to_apply": "Use the output of one function as input to another"
        },
        metadata={
            "confidence": 0.9, 
            "language": "Python"
        }
    )
    
    # Add more structured observations
    lesson_ctx.observe(
        entity_name="Function Composition",
        what_was_learned="Function composition can reduce intermediate variables",
        how_to_apply="Use the pipe operator in languages that support it"
    )
    
    # Create relationships
    lesson_ctx.relate(
        source_name="Function Composition",
        target_name="Functional Programming",
        relationship_type="PART_OF",
        properties={"strength": 0.8}
    )
    
    # Track knowledge evolution
    lesson_ctx.track(
        lesson_name="Function Composition",
        context_entity="Code Refactoring Project"
    )

# Close memory system when done
memory_manager.close()
```

### Managing Project Memory

```python
from src.graph_memory import GraphMemoryManager
from src.logger import get_logger

# Initialize memory system
logger = get_logger("memory_system")
memory_manager = GraphMemoryManager(logger)
memory_manager.initialize()

# Create a project with domains and components using a context manager
with memory_manager.project_context(project_name="Web Dashboard") as project_ctx:
    # Create the project
    project = project_ctx.create_project(
        name="Web Dashboard",
        description="Interactive web dashboard for analytics",
        status="active"
    )
    
    # Create domains
    project_ctx.create_domain(
        name="Frontend",
        description="UI components and user interaction"
    )
    
    project_ctx.create_domain(
        name="Backend",
        description="Data processing and API services"
    )
    
    # Create components
    project_ctx.create_component(
        name="ChartComponent",
        component_type="UI",
        domain_name="Frontend",
        description="Interactive data visualization component"
    )
    
    project_ctx.create_component(
        name="DataService",
        component_type="API",
        domain_name="Backend",
        description="REST API for chart data retrieval"
    )
    
    # Create relationships between components
    project_ctx.relate(
        source_name="ChartComponent",
        target_name="DataService",
        relation_type="DEPENDS_ON",
        properties={"data_format": "JSON"}
    )
    
    # Add observations
    project_ctx.add_observation(
        entity_name="ChartComponent",
        content="Initial implementation should support bar and line charts",
        observation_type="REQUIREMENT"
    )
    
    # Get project structure
    structure = project_ctx.get_structure()

# Close memory system when done
memory_manager.close()
```

## Standardized Response System

The system uses a comprehensive standardized response system that ensures consistent API responses:

1. **Pydantic-Based Response Models**
   - All responses use Pydantic models for validation and serialization
   - Consistent structure for both success and error responses

2. **Standardized Response Format**
   - Success responses follow this structure:
     ```json
     {
       "status": "success",
       "message": "Operation completed successfully",
       "timestamp": "2023-06-01T12:34:56.789Z",
       "data": { ... }
     }
     ```
   - Error responses provide structured information:
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

## Performance Considerations

For optimal performance:

1. **Use Context Managers**: Prefer context managers for batch operations
2. **Connection Management**: Initialize managers once and reuse
3. **Query Filtering**: Use specific filters to reduce result set size
4. **Embedding Caching**: Cache embeddings for frequently used text
5. **Pagination**: Use paginated search for large result sets

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
