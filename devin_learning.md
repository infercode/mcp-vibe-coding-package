# Memory Process Instructions

## Follow these steps for each project interaction:

1. **Project Context Identification**:
   - Identify the current project by name (e.g., "customer-portal", "data-pipeline")
   - Set the project name using `set_project_name` if not already configured
   - Associate all memories with this project context

2. **Knowledge Retrieval**:
   - Before suggesting solutions or writing code, search the memory graph for relevant context
   - Query for project requirements, architectural decisions, or previous implementation details
   - Example: `search_nodes("authentication implementation")`
   - Reference found memories in your responses with "Based on project history..."

3. **Knowledge Categorization**:
   - While analyzing the codebase or user requests, categorize important information as:
     a) **Project Structure** - Architecture, design patterns, component relationships
     b) **Technical Decisions** - Technology choices, algorithms, implementation approaches
     c) **Dependencies** - Libraries, frameworks, services, APIs being used
     d) **Requirements** - User stories, acceptance criteria, constraints
     e) **Lessons Learned** - Previous bugs, performance issues, refactoring insights

4. **Memory Graph Updates**:
   - Update the memory graph with new knowledge using appropriate entities and relations:
     a) Create component entities: `create_entities([{"name": "AuthService", "entityType": "Component", "observations": ["Handles user authentication", "Uses JWT tokens"]}])`
     b) Establish dependency relations: `create_relations([{"from": "AuthService", "to": "jsonwebtoken", "relationType": "DEPENDS_ON"}])`
     c) Document lessons: `add_observations([{"entity": "AuthService", "content": "Token refresh caused race condition when multiple tabs open"}])`

5. **Package and Dependency Documentation**:
   - When new dependencies are introduced:
     a) Create an entity for each significant package or library
     b) Record version info: `{"name": "React", "entityType": "Library", "observations": ["Using v18.2.0", "Handles UI rendering"]}`
     c) Document integration points: `{"from": "FrontendApp", "to": "React", "relationType": "USES"}`
     d) Note compatibility issues: `{"entity": "React", "content": "Requires Node 14+ to build"}`

6. **Agent Self-Improvement**:
   - Record solution patterns and mistakes for future reference:
     a) Success patterns: `{"name": "PaginationPattern", "entityType": "Pattern", "observations": ["Implemented using cursor-based approach", "Handles large datasets efficiently"]}`
     b) Bug patterns: `{"name": "StateRaceCondition", "entityType": "BugPattern", "observations": ["Occurred when async state updates weren't properly sequenced"]}`
     c) Link issues to solutions: `{"from": "StateRaceCondition", "to": "UseReducerPattern", "relationType": "SOLVED_BY"}`
