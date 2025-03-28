---
description: 
globs: docs/lesson-memory-implementation.md, docs/project-memory-implementation.md
alwaysApply: false
---

# Memory Implementation Progress Tracking

## Context

- This rule applies when working on the memory system implementation
- User is developing a Neo4j Graph Memory architecture for lessons and projects
- Progress updates need to be systematically tracked in documentation
- Only the "Implementation progress" section should be modified unless explicitly requested

## Critical Rules

- ONLY update the `# Implementation progress` section of the documents
- ALWAYS use the time MCP tool (`mcp_time_mcp_current_time`) to get the current date before adding it to the document
- Use the ISO format (YYYY-MM-DD) for dates retrieved from the time MCP
- Each progress entry should include: date, feature implemented, and brief description
- Use bullet points for clarity with nested sub-bullets for implementation details
- Document which implementation features from the specification are completed
- Link implementation progress to specific methods or functionality when available
- Never modify any part of the document above the Implementation progress section without explicit permission
- Lesson implementation updates go in lesson-memory-implementation.md
- Project implementation updates go in project-memory-implementation.md
- Include confidence scores where applicable (0.0-1.0)
- Add an appropriate emoji at the end of each major implementation milestone 🏆

## Examples

<example>
# Implementation progress

## 2025-03-27
- Implemented basic lesson container structure 🏗️
  - Added `create_lesson_container()` method
  - Set up core properties and relationships
  
## 2025-03-29
- Added temporal intelligence tracking ⏱️
  - Implemented `get_knowledge_evolution()` method
  - Added timestamp tracking on lesson relationships
</example>

<example type="invalid">
# Implementation progress

Updates:
- Added some new features
- Changed how lessons work
- Fixed a bunch of stuff

[No dates, no structure, vague descriptions, didn't use time MCP to get current date]

# Core Concepts
[Modifying sections above Implementation progress without permission]
</example> 