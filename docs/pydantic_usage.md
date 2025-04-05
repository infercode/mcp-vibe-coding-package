# Using Pydantic in MCP Graph Memory

This document provides guidance on how Pydantic is used in the MCP Graph Memory project for data validation, serialization, and configuration management.

## Overview

[Pydantic](https://docs.pydantic.dev/) is integrated into several key areas of the MCP Graph Memory project:

1. **Data Validation** - For validating input data in MCP tools
2. **Response Formatting** - For standardizing API responses
3. **Configuration Management** - For handling application settings
4. **Schema Documentation** - For documenting data structures

## Core Model Organization

Pydantic models are organized in the `src/models` directory:

```
src/models/
├── __init__.py
├── project_memory.py  # Project memory models
├── responses.py       # Response utilities
└── settings.py        # Configuration models
```

## Data Validation Models

The `project_memory.py` file contains models for validating data used in the project memory system:

```python
# Example of creating a project container
from src.models.project_memory import ProjectContainerCreate

# Valid data will be accepted
container = ProjectContainerCreate(
    name="My Project",
    description="Project description",
    tags=["tag1", "tag2"]
)

# Invalid data will raise ValidationError
try:
    # Missing required 'name' field
    invalid_container = ProjectContainerCreate(description="Invalid")
except ValidationError as e:
    print(e.errors())
```

### Key Validation Features

- **Type Validation** - Ensures data matches expected types
- **Required Fields** - Fields without defaults are required
- **Field Constraints** - Such as min/max values, regex patterns
- **Custom Validators** - Using `@field_validator` and `@model_validator`

## Response Formatting

The `responses.py` module provides utilities for creating standardized responses:

```python
from src.models.responses import create_success_response, create_error_response

# Create a success response
success = create_success_response(
    message="Operation successful",
    data={"id": "123", "name": "Example"}
)

# Create an error response
error = create_error_response(
    message="Operation failed",
    code="validation_error",
    details={"field": "name", "error": "Required field missing"}
)

# Convert to JSON
json_response = model_to_json(success)
```

## Configuration Management

The `settings.py` module leverages Pydantic's settings management capabilities:

```python
from src.models.settings import get_settings

# Get application settings (from environment variables and config files)
settings = get_settings()

# Access configuration values
neo4j_uri = settings.neo4j.uri
vector_enabled = settings.vector.enabled
```

### Configuration Sources (in priority order)

1. Environment Variables - Override all other sources
2. Configuration Files - Loaded from specified path
3. Default Values - Defined in model classes

## Best Practices for Using Pydantic in This Project

1. **Use Models for Validation** - Always validate incoming data with Pydantic models before processing
2. **Standard Responses** - Use the response utilities for consistent API responses
3. **Settings Management** - Access configuration through the settings module instead of direct environment variables
4. **Field Documentation** - Use the `description` parameter in `Field()` to document the purpose of each field
5. **Model Documentation** - Add docstrings to model classes to explain their purpose

## Adding New Models

To add new models for a different aspect of the system:

1. Create a new file in the `src/models` directory
2. Define your models using Pydantic `BaseModel`
3. Add validation logic with field validators where needed
4. Import and use the models in your MCP tools

Example:

```python
from pydantic import BaseModel, Field, field_validator

class MyNewModel(BaseModel):
    """Documentation for this model."""
    field1: str = Field(..., description="Required field")
    field2: int = Field(0, ge=0, description="Optional field with default and validation")
    
    @field_validator('field1')
    def validate_field1(cls, v):
        if len(v) < 3:
            raise ValueError("field1 must be at least 3 characters long")
        return v
```

## Generating JSON Schema

Pydantic models can generate JSON Schema for API documentation:

```python
from src.models.project_memory import ProjectContainerCreate

# Generate JSON Schema
schema = ProjectContainerCreate.model_json_schema()
```

## Error Handling

When using Pydantic models in MCP tools, follow this pattern:

```python
try:
    validated_data = MyModel(**input_data)
    # Process validated data
except ValidationError as e:
    # Handle validation errors
    error_response = create_error_response(
        message="Validation error",
        code="validation_error",
        details=e.errors()
    )
    return model_to_json(error_response)
```

## References

- [Pydantic Documentation](https://docs.pydantic.dev/)
- [Pydantic Settings Management](https://docs.pydantic.dev/latest/concepts/pydantic_settings/)
- [Pydantic Validators](https://docs.pydantic.dev/latest/concepts/validators/) 