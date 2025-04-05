import pytest
from unittest.mock import MagicMock, patch, call
import json

from src.graph_memory import SearchManager


def test_init(mock_base_manager):
    """Test initialization of SearchManager."""
    # Create search manager with mocked dependencies
    manager = SearchManager(mock_base_manager)
    
    # Verify manager attributes are set correctly
    assert manager.base_manager == mock_base_manager
    assert manager.logger == mock_base_manager.logger


def test_search_entities_text_based(mock_base_manager):
    """Test text-based search for entities."""
    # Create manager
    manager = SearchManager(mock_base_manager)
    
    # Mock query execution
    mock_records = [
        MagicMock(
            get=lambda x: MagicMock(
                id="entity-1", 
                items=lambda: [("name", "Test Entity 1"), ("content", "This is test entity one")]
            ) if x == "e" else None
        ),
        MagicMock(
            get=lambda x: MagicMock(
                id="entity-2", 
                items=lambda: [("name", "Test Entity 2"), ("content", "This is test entity two")]
            ) if x == "e" else None
        )
    ]
    
    mock_base_manager.safe_execute_read_query = MagicMock(return_value=mock_records)
    
    # Execute search
    result = manager.search_entities("test query", limit=10)
    
    # Verify result
    result_dict = json.loads(result)
    assert len(result_dict["entities"]) == 2
    assert result_dict["entities"][0]["id"] == "entity-1"
    assert result_dict["entities"][1]["id"] == "entity-2"
    
    # Verify base_manager.safe_execute_read_query was called with correct parameters
    mock_base_manager.safe_execute_read_query.assert_called_once()
    # Check that the query contains the search term
    query, params = mock_base_manager.safe_execute_read_query.call_args[0]
    assert "MATCH" in query
    assert "RETURN" in query
    assert "LIMIT" in query
    assert params["search_term"] == "test query"


def test_search_entities_with_entity_types(mock_base_manager):
    """Test search with entity type filters."""
    # Create manager
    manager = SearchManager(mock_base_manager)
    
    # Mock query execution
    mock_records = [
        MagicMock(
            get=lambda x: MagicMock(
                id="entity-3", 
                items=lambda: [("name", "Filtered Entity"), ("type", "Person")]
            ) if x == "e" else None
        )
    ]
    
    mock_base_manager.safe_execute_read_query = MagicMock(return_value=mock_records)
    
    # Execute search with entity type filters
    entity_types = ["Person", "Organization"]
    result = manager.search_entities("test query", entity_types=entity_types, limit=5)
    
    # Verify result
    result_dict = json.loads(result)
    assert len(result_dict["entities"]) == 1
    assert result_dict["entities"][0]["id"] == "entity-3"
    assert result_dict["entities"][0]["type"] == "Person"
    
    # Verify base_manager.safe_execute_read_query was called with correct parameters
    mock_base_manager.safe_execute_read_query.assert_called_once()
    # Check that the query contains the entity types filter
    query, params = mock_base_manager.safe_execute_read_query.call_args[0]
    assert "MATCH" in query
    assert "WHERE" in query
    assert "LIMIT" in query
    assert params["entity_types"] == entity_types


def test_semantic_search_entities(mock_base_manager):
    """Test semantic search for entities."""
    # Create manager
    manager = SearchManager(mock_base_manager)
    
    # Set embedding_index_name attribute
    mock_base_manager.embedding_index_name = "entity_embedding"
    
    # Mock embedding generation and ensure_initialized
    mock_base_manager.ensure_initialized = MagicMock()
    mock_base_manager.generate_embedding = MagicMock(return_value=[0.1, 0.2, 0.3, 0.4])
    
    # Mock query execution for vector search
    class MockNodeRecord:
        def __init__(self, entity_id, entity_name, content, score):
            self.entity_id = entity_id
            self.entity_name = entity_name
            self.content = content
            self.score = score
            
        def get(self, key):
            if key == "e":
                mock_entity = MagicMock()
                mock_entity.id = self.entity_id
                mock_entity.items = lambda: [
                    ("name", self.entity_name), 
                    ("content", self.content)
                ]
                return mock_entity
            elif key == "score":
                return self.score
            return None
    
    mock_records = [
        MockNodeRecord("entity-1", "Test Entity 1", "This is test entity one", 0.95),
        MockNodeRecord("entity-2", "Test Entity 2", "This is test entity two", 0.85)
    ]
    
    # Setup mock response
    mock_base_manager.safe_execute_read_query = MagicMock(return_value=mock_records)
    
    # Execute search
    result = manager.search_entities("semantic query", semantic=True, limit=10)
    
    # Parse the result
    result_dict = json.loads(result)
    
    # Verify result has entities
    assert "entities" in result_dict
    assert len(result_dict["entities"]) == 2
    assert result_dict["entities"][0]["id"] == "entity-1"
    assert result_dict["entities"][1]["id"] == "entity-2"
    assert result_dict["entities"][0]["similarity_score"] == 0.95
    
    # Verify base_manager.generate_embedding was called
    mock_base_manager.generate_embedding.assert_called_once_with("semantic query")
    
    # Verify safe_execute_read_query was called with correct parameters
    mock_base_manager.safe_execute_read_query.assert_called_once()
    query, params = mock_base_manager.safe_execute_read_query.call_args[0]
    assert "CALL db.index.vector.queryNodes" in query
    assert params["index_name"] == "entity_embedding"
    assert params["k"] == 10
    assert params["embedding"] == [0.1, 0.2, 0.3, 0.4]


def test_search_error_handling(mock_base_manager):
    """Test error handling during search operations."""
    # Create manager
    manager = SearchManager(mock_base_manager)
    
    # Mock embedding generation that raises an exception
    mock_base_manager.ensure_initialized.side_effect = Exception("Database connection error")
    
    # Execute search
    result = manager.search_entities("query with error")
    
    # Verify result contains error
    result_dict = json.loads(result)
    assert "error" in result_dict
    assert "Database connection error" in result_dict["error"]
    
    # Verify logger.error was called
    mock_base_manager.logger.error.assert_called()


def test_search_entities_empty_query(mock_base_manager):
    """Test search with empty query."""
    # Create manager
    manager = SearchManager(mock_base_manager)
    
    # Mock query execution
    mock_records = []
    mock_base_manager.safe_execute_read_query = MagicMock(return_value=mock_records)
    
    # Execute search with empty query
    result = manager.search_entities("")
    
    # Verify result is empty list
    result_dict = json.loads(result)
    assert len(result_dict["entities"]) == 0
    
    # Verify base_manager.safe_execute_read_query was called
    mock_base_manager.safe_execute_read_query.assert_called_once()


def test_query_knowledge_graph(mock_base_manager):
    """Test executing custom Cypher queries."""
    # Create manager
    manager = SearchManager(mock_base_manager)
    
    # Mock query execution
    mock_record = MagicMock()
    mock_record.keys.return_value = ["person", "count"]
    mock_record.get = lambda key: (
        MagicMock(
            items=lambda: [("name", "John"), ("age", 30)]
        ) if key == "person" else 5 if key == "count" else None
    )
    
    mock_records = [mock_record]
    mock_base_manager.safe_execute_read_query = MagicMock(return_value=mock_records)
    
    # Execute custom query
    custom_query = "MATCH (p:Person) RETURN p, count(p.friends) as count"
    result = manager.query_knowledge_graph(custom_query, {"age": 30})
    
    # Verify result
    result_dict = json.loads(result)
    assert "results" in result_dict
    assert len(result_dict["results"]) == 1
    assert result_dict["results"][0]["person"]["name"] == "John"
    assert result_dict["results"][0]["count"] == 5
    
    # Verify base_manager was called with the custom query
    mock_base_manager.safe_execute_read_query.assert_called_once_with(custom_query, {"age": 30})


def test_forbidden_query_operations(mock_base_manager):
    """Test rejection of forbidden query operations."""
    # Create manager
    manager = SearchManager(mock_base_manager)
    
    # Try dangerous queries
    dangerous_queries = [
        "CREATE (n:Node {name: 'Test'})",
        "MATCH (n) DELETE n",
        "MATCH (n) SET n.prop = 'value'",
        "MERGE (n:Node {name: 'Test'})",
        "DROP INDEX ON :Node(name)",
        "CALL db.index.vector.createNodeIndex"
    ]
    
    for query in dangerous_queries:
        result = manager.query_knowledge_graph(query)
        result_dict = json.loads(result)
        
        # Verify query was rejected
        assert "error" in result_dict
        assert "Forbidden operation" in result_dict["error"]
        
    # Verify safe_execute_query was not called
    mock_base_manager.safe_execute_query.assert_not_called()


def test_search_entity_neighborhoods(mock_base_manager):
    """Test retrieving entity neighborhoods."""
    # Create manager
    manager = SearchManager(mock_base_manager)
    
    # Mock the basics
    mock_base_manager.ensure_initialized = MagicMock()
    
    # Mock an entity record
    entity_record = MagicMock()
    entity = MagicMock()
    entity.id = "entity-123"
    entity.items = MagicMock(return_value=[("name", "Central Entity"), ("type", "Person")])
    entity_record.get = MagicMock(return_value=entity)
    
    # Mock nodes and relationships
    nodes = [MagicMock()]
    nodes[0].id = "entity-123"
    nodes[0].items = MagicMock(return_value=[("name", "Central Entity"), ("type", "Person")])
    
    node_record = MagicMock()
    node_record.get = MagicMock(return_value=nodes)
    
    relationships = []
    rel_record = MagicMock()
    rel_record.get = MagicMock(return_value=relationships)
    
    # Setup responses for the queries
    mock_base_manager.safe_execute_read_query = MagicMock(side_effect=[
        [entity_record],  # Entity check
        [node_record],    # Node query
        [rel_record]      # Relationship query
    ])
    
    # Execute neighborhood search
    result = manager.search_entity_neighborhoods("Central Entity", max_depth=2, max_nodes=50)
    
    # Check result
    result_dict = json.loads(result)
    assert "graph" in result_dict, f"Expected 'graph', got {result_dict}"
    assert "nodes" in result_dict["graph"]
    assert isinstance(result_dict["graph"]["nodes"], list)
    assert "relationships" in result_dict["graph"]
    assert isinstance(result_dict["graph"]["relationships"], list)
    assert result_dict["graph"]["center_entity"] == "Central Entity"
    
    # Verify the entity check query
    assert mock_base_manager.safe_execute_read_query.call_count >= 2


def test_find_paths_between_entities(mock_base_manager):
    """Test finding paths between entities."""
    # Create manager
    manager = SearchManager(mock_base_manager)
    
    # Mock query execution for path finding
    mock_path_node1 = MagicMock()
    mock_path_node1.items.return_value = [("name", "Start Entity"), ("type", "Person")]
    mock_path_node1.id = "entity-start"
    
    mock_path_rel = MagicMock()
    mock_path_rel.items.return_value = [("type", "KNOWS"), ("since", 2020)]
    mock_path_rel.id = "rel-path"
    
    mock_path_node2 = MagicMock()
    mock_path_node2.items.return_value = [("name", "End Entity"), ("type", "Person")]
    mock_path_node2.id = "entity-end"
    
    mock_path = MagicMock()
    mock_path.nodes = [mock_path_node1, mock_path_node2]
    mock_path.relationships = [mock_path_rel]
    
    # Setup the mock record for the path
    mock_path_record = MagicMock()
    mock_path_record.get = lambda *args: mock_path if args[0] == "path" else None
    
    # Setup the mock records for the entity checks
    mock_from_entity = MagicMock()
    mock_from_entity.items.return_value = [("name", "Start Entity"), ("type", "Person")]
    mock_from_entity.id = "entity-start"
    
    mock_to_entity = MagicMock()
    mock_to_entity.items.return_value = [("name", "End Entity"), ("type", "Person")]
    mock_to_entity.id = "entity-end"
    
    mock_entity_record = MagicMock()
    mock_entity_record.get = lambda *args: mock_from_entity if args[0] == "from" else mock_to_entity if args[0] == "to" else None
    
    # Setup mock to return appropriate values for each call
    mock_base_manager.safe_execute_read_query = MagicMock(side_effect=[
        [mock_entity_record],  # First call to verify entities exist
        [mock_path_record]     # Second call to find paths
    ])
    
    # Execute path finding
    result = manager.find_paths_between_entities("Start Entity", "End Entity", max_depth=3)
    
    # Verify result
    result_dict = json.loads(result)
    assert "paths" in result_dict
    assert len(result_dict["paths"]) == 1
    assert "nodes" in result_dict["paths"][0]
    assert "relationships" in result_dict["paths"][0]
    assert len(result_dict["paths"][0]["nodes"]) == 2
    assert len(result_dict["paths"][0]["relationships"]) == 1
    
    # Verify base_manager.safe_execute_read_query was called twice
    assert mock_base_manager.safe_execute_read_query.call_count == 2 