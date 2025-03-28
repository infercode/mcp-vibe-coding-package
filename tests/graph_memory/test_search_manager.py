import pytest
from unittest.mock import MagicMock, patch, call
import json

from src.graph_memory import SearchManager


def test_init(mock_base_manager, mock_embedding_adapter, mock_logger):
    """Test initialization of SearchManager."""
    # Create search manager with mocked dependencies
    manager = SearchManager(mock_base_manager, mock_embedding_adapter, mock_logger)
    
    # Verify manager attributes are set correctly
    assert manager.base_manager == mock_base_manager
    assert manager.embedding_adapter == mock_embedding_adapter
    assert manager.logger == mock_logger


def test_search_nodes_text_based(mock_base_manager, mock_embedding_adapter, mock_logger):
    """Test text-based search for nodes."""
    # Create manager
    manager = SearchManager(mock_base_manager, mock_embedding_adapter, mock_logger)
    
    # Mock embedding availability check
    mock_embedding_adapter.is_available.return_value = False
    
    # Mock query execution
    mock_search_results = [
        {"node": {"id": "entity-1", "name": "Test Entity 1", "content": "This is test entity one"}},
        {"node": {"id": "entity-2", "name": "Test Entity 2", "content": "This is test entity two"}}
    ]
    mock_base_manager.execute_query.return_value = {"results": mock_search_results}
    
    # Execute search
    result = manager.search_nodes("test query", limit=10)
    
    # Verify result
    result_list = json.loads(result)
    assert len(result_list) == 2
    assert result_list[0]["id"] == "entity-1"
    assert result_list[1]["id"] == "entity-2"
    
    # Verify base_manager.execute_query was called with correct parameters
    mock_base_manager.execute_query.assert_called_once()
    # Check that the query contains the search term
    query = mock_base_manager.execute_query.call_args[0][0]
    assert "MATCH" in query
    assert "RETURN" in query
    assert "LIMIT 10" in query
    
    # Verify embedding_adapter.is_available was called
    mock_embedding_adapter.is_available.assert_called_once()
    # Verify embedding_adapter.get_embedding was NOT called (text-based search)
    mock_embedding_adapter.get_embedding.assert_not_called()


def test_search_nodes_vector_based(mock_base_manager, mock_embedding_adapter, mock_logger):
    """Test vector-based search for nodes."""
    # Create manager
    manager = SearchManager(mock_base_manager, mock_embedding_adapter, mock_logger)
    
    # Mock embedding availability check
    mock_embedding_adapter.is_available.return_value = True
    
    # Mock embedding generation
    mock_embedding_adapter.get_embedding.return_value = [0.1, 0.2, 0.3, 0.4]
    
    # Mock query execution for vector search
    mock_search_results = [
        {
            "node": {"id": "entity-1", "name": "Test Entity 1", "content": "This is test entity one"},
            "score": 0.95
        },
        {
            "node": {"id": "entity-2", "name": "Test Entity 2", "content": "This is test entity two"},
            "score": 0.85
        }
    ]
    mock_base_manager.execute_query.return_value = {"results": mock_search_results}
    
    # Execute search
    result = manager.search_nodes("semantic query", limit=10)
    
    # Verify result
    result_list = json.loads(result)
    assert len(result_list) == 2
    assert result_list[0]["id"] == "entity-1"
    assert result_list[1]["id"] == "entity-2"
    assert result_list[0]["score"] == 0.95
    
    # Verify embedding_adapter.is_available was called
    mock_embedding_adapter.is_available.assert_called_once()
    # Verify embedding_adapter.get_embedding was called with correct parameters
    mock_embedding_adapter.get_embedding.assert_called_once_with("semantic query")
    
    # Verify base_manager.execute_query was called with correct parameters
    mock_base_manager.execute_query.assert_called_once()
    # Check that the query contains vector search specifics
    query = mock_base_manager.execute_query.call_args[0][0]
    assert "MATCH" in query
    assert "RETURN" in query
    assert "LIMIT 10" in query
    assert "vector" in query.lower() or "embedding" in query.lower()


def test_search_nodes_with_filters(mock_base_manager, mock_embedding_adapter, mock_logger):
    """Test search with additional filters."""
    # Create manager
    manager = SearchManager(mock_base_manager, mock_embedding_adapter, mock_logger)
    
    # Mock embedding availability check
    mock_embedding_adapter.is_available.return_value = False
    
    # Mock query execution with filters
    mock_search_results = [
        {"node": {"id": "entity-3", "name": "Filtered Entity", "type": "Person"}}
    ]
    mock_base_manager.execute_query.return_value = {"results": mock_search_results}
    
    # Execute search with filters
    filters = {"type": "Person", "metadata.location": "New York"}
    result = manager.search_nodes("test query", filters=filters, limit=5)
    
    # Verify result
    result_list = json.loads(result)
    assert len(result_list) == 1
    assert result_list[0]["id"] == "entity-3"
    assert result_list[0]["type"] == "Person"
    
    # Verify base_manager.execute_query was called with correct parameters
    mock_base_manager.execute_query.assert_called_once()
    # Check that the query contains the filters
    query = mock_base_manager.execute_query.call_args[0][0]
    assert "MATCH" in query
    assert "Person" in query
    assert "New York" in query
    assert "LIMIT 5" in query


def test_search_error_handling(mock_base_manager, mock_embedding_adapter, mock_logger):
    """Test error handling during search operations."""
    # Create manager
    manager = SearchManager(mock_base_manager, mock_embedding_adapter, mock_logger)
    
    # Mock embedding availability check
    mock_embedding_adapter.is_available.return_value = True
    
    # Mock embedding generation that raises an exception
    mock_embedding_adapter.get_embedding.side_effect = Exception("Embedding service unavailable")
    
    # Execute search - should fall back to text-based search
    result = manager.search_nodes("query with embedding error")
    
    # Verify embedding_adapter.get_embedding was called
    mock_embedding_adapter.get_embedding.assert_called_once()
    
    # Verify logger.error was called
    mock_logger.error.assert_called()
    
    # Verify base_manager.execute_query was still called (fallback to text search)
    mock_base_manager.execute_query.assert_called_once()


def test_search_nodes_empty_query(mock_base_manager, mock_embedding_adapter, mock_logger):
    """Test search with empty query."""
    # Create manager
    manager = SearchManager(mock_base_manager, mock_embedding_adapter, mock_logger)
    
    # Execute search with empty query
    result = manager.search_nodes("")
    
    # Verify result is empty
    result_list = json.loads(result)
    assert len(result_list) == 0
    
    # Verify base_manager.execute_query was NOT called
    mock_base_manager.execute_query.assert_not_called()
    
    # Verify embedding_adapter methods were NOT called
    mock_embedding_adapter.is_available.assert_not_called()
    mock_embedding_adapter.get_embedding.assert_not_called()


def test_search_nodes_db_error(mock_base_manager, mock_embedding_adapter, mock_logger):
    """Test handling of database errors during search."""
    # Create manager
    manager = SearchManager(mock_base_manager, mock_embedding_adapter, mock_logger)
    
    # Mock embedding availability check
    mock_embedding_adapter.is_available.return_value = False
    
    # Mock query execution that raises an exception
    mock_base_manager.execute_query.side_effect = Exception("Database connection error")
    
    # Execute search
    result = manager.search_nodes("test query")
    
    # Verify result is empty
    result_list = json.loads(result)
    assert len(result_list) == 0
    
    # Verify base_manager.execute_query was called
    mock_base_manager.execute_query.assert_called_once()
    
    # Verify logger.error was called
    mock_logger.error.assert_called()


def test_search_nodes_with_custom_project(mock_base_manager, mock_embedding_adapter, mock_logger):
    """Test search with custom project specified."""
    # Create manager
    manager = SearchManager(mock_base_manager, mock_embedding_adapter, mock_logger)
    
    # Mock embedding availability check
    mock_embedding_adapter.is_available.return_value = False
    
    # Mock query execution
    mock_search_results = [
        {"node": {"id": "entity-4", "name": "Project Entity", "project": "custom-project"}}
    ]
    mock_base_manager.execute_query.return_value = {"results": mock_search_results}
    
    # Execute search with project specified
    result = manager.search_nodes("project query", project_name="custom-project")
    
    # Verify result
    result_list = json.loads(result)
    assert len(result_list) == 1
    assert result_list[0]["id"] == "entity-4"
    assert result_list[0]["project"] == "custom-project"
    
    # Verify base_manager.execute_query was called with correct parameters
    mock_base_manager.execute_query.assert_called_once()
    # Check that the query contains the project filter
    query = mock_base_manager.execute_query.call_args[0][0]
    assert "MATCH" in query
    assert "custom-project" in query


def test_search_with_sort_options(mock_base_manager, mock_embedding_adapter, mock_logger):
    """Test search with custom sorting options."""
    # Create manager
    manager = SearchManager(mock_base_manager, mock_embedding_adapter, mock_logger)
    
    # Mock embedding availability check
    mock_embedding_adapter.is_available.return_value = False
    
    # Mock query execution
    mock_search_results = [
        {"node": {"id": "entity-5", "name": "First Entity", "created_at": "2023-01-02"}},
        {"node": {"id": "entity-6", "name": "Second Entity", "created_at": "2023-01-01"}}
    ]
    mock_base_manager.execute_query.return_value = {"results": mock_search_results}
    
    # Execute search with sort options
    sort_options = {"field": "created_at", "order": "DESC"}
    result = manager.search_nodes("sorted query", sort=sort_options)
    
    # Verify result
    result_list = json.loads(result)
    assert len(result_list) == 2
    assert result_list[0]["id"] == "entity-5"  # Should be first due to DESC sort
    
    # Verify base_manager.execute_query was called with correct parameters
    mock_base_manager.execute_query.assert_called_once()
    # Check that the query contains the sort options
    query = mock_base_manager.execute_query.call_args[0][0]
    assert "ORDER BY" in query
    assert "DESC" in query


def test_search_with_pagination(mock_base_manager, mock_embedding_adapter, mock_logger):
    """Test search with pagination options."""
    # Create manager
    manager = SearchManager(mock_base_manager, mock_embedding_adapter, mock_logger)
    
    # Mock embedding availability check
    mock_embedding_adapter.is_available.return_value = False
    
    # Mock query execution
    mock_search_results = [
        {"node": {"id": f"entity-{i}", "name": f"Entity {i}"}} for i in range(5, 10)
    ]
    mock_base_manager.execute_query.return_value = {"results": mock_search_results}
    
    # Execute search with pagination (skip first 5 results)
    result = manager.search_nodes("paginated query", skip=5, limit=5)
    
    # Verify result
    result_list = json.loads(result)
    assert len(result_list) == 5
    assert result_list[0]["id"] == "entity-5"
    assert result_list[4]["id"] == "entity-9"
    
    # Verify base_manager.execute_query was called with correct parameters
    mock_base_manager.execute_query.assert_called_once()
    # Check that the query contains the pagination
    query = mock_base_manager.execute_query.call_args[0][0]
    assert "SKIP 5" in query
    assert "LIMIT 5" in query


def test_semantic_search_with_threshold(mock_base_manager, mock_embedding_adapter, mock_logger):
    """Test semantic search with similarity threshold."""
    # Create manager
    manager = SearchManager(mock_base_manager, mock_embedding_adapter, mock_logger)
    
    # Mock embedding availability check
    mock_embedding_adapter.is_available.return_value = True
    
    # Mock embedding generation
    mock_embedding_adapter.get_embedding.return_value = [0.1, 0.2, 0.3, 0.4]
    
    # Mock query execution for vector search with scores
    mock_search_results = [
        {
            "node": {"id": "entity-1", "name": "High Similarity", "content": "Very relevant"},
            "score": 0.95
        },
        {
            "node": {"id": "entity-2", "name": "Medium Similarity", "content": "Somewhat relevant"},
            "score": 0.75
        },
        {
            "node": {"id": "entity-3", "name": "Low Similarity", "content": "Not very relevant"},
            "score": 0.55
        }
    ]
    mock_base_manager.execute_query.return_value = {"results": mock_search_results}
    
    # Execute search with similarity threshold of 0.7
    result = manager.search_nodes("semantic query", similarity_threshold=0.7)
    
    # Verify result - should only include results above threshold
    result_list = json.loads(result)
    assert len(result_list) == 2  # Only the first two results are above threshold
    assert result_list[0]["id"] == "entity-1"
    assert result_list[1]["id"] == "entity-2"
    assert "entity-3" not in [node["id"] for node in result_list]
    
    # Verify embedding_adapter methods were called
    mock_embedding_adapter.is_available.assert_called_once()
    mock_embedding_adapter.get_embedding.assert_called_once()
    
    # Verify base_manager.execute_query was called
    mock_base_manager.execute_query.assert_called_once()


def test_search_specific_node_types(mock_base_manager, mock_embedding_adapter, mock_logger):
    """Test searching for specific node types."""
    # Create manager
    manager = SearchManager(mock_base_manager, mock_embedding_adapter, mock_logger)
    
    # Mock embedding availability check
    mock_embedding_adapter.is_available.return_value = False
    
    # Mock query execution
    mock_search_results = [
        {"node": {"id": "entity-7", "name": "Document Entity", "type": "Document"}}
    ]
    mock_base_manager.execute_query.return_value = {"results": mock_search_results}
    
    # Execute search with node type filter
    result = manager.search_nodes("document query", node_types=["Document", "File"])
    
    # Verify result
    result_list = json.loads(result)
    assert len(result_list) == 1
    assert result_list[0]["id"] == "entity-7"
    assert result_list[0]["type"] == "Document"
    
    # Verify base_manager.execute_query was called with correct parameters
    mock_base_manager.execute_query.assert_called_once()
    # Check that the query contains the node type filter
    query = mock_base_manager.execute_query.call_args[0][0]
    assert "Document" in query
    assert "File" in query


def test_full_text_search(mock_base_manager, mock_embedding_adapter, mock_logger):
    """Test full-text search functionality."""
    # Create manager
    manager = SearchManager(mock_base_manager, mock_embedding_adapter, mock_logger)
    
    # Mock embedding availability check - doesn't matter for full-text search
    mock_embedding_adapter.is_available.return_value = True
    
    # Mock query execution for full-text search
    mock_search_results = [
        {"node": {"id": "entity-8", "name": "Full Text Result", "content": "This contains the exact search phrase"}}
    ]
    mock_base_manager.execute_query.return_value = {"results": mock_search_results}
    
    # Execute full-text search
    result = manager.full_text_search("exact search phrase")
    
    # Verify result
    result_list = json.loads(result)
    assert len(result_list) == 1
    assert result_list[0]["id"] == "entity-8"
    assert "exact search phrase" in result_list[0]["content"]
    
    # Verify base_manager.execute_query was called with correct parameters
    mock_base_manager.execute_query.assert_called_once()
    # Check that the query is set up for full-text search
    query = mock_base_manager.execute_query.call_args[0][0]
    assert "exact search phrase" in query
    
    # Full-text search should NOT use embeddings
    mock_embedding_adapter.get_embedding.assert_not_called()


def test_get_similar_nodes(mock_base_manager, mock_embedding_adapter, mock_logger):
    """Test retrieval of similar nodes to a specific node."""
    # Create manager
    manager = SearchManager(mock_base_manager, mock_embedding_adapter, mock_logger)
    
    # Mock embedding availability check
    mock_embedding_adapter.is_available.return_value = True
    
    # Mock query to get source node embedding
    source_node = {
        "id": "source-node", 
        "name": "Source Node", 
        "embedding": [0.1, 0.2, 0.3, 0.4]
    }
    mock_base_manager.execute_query.side_effect = [
        {"results": [{"node": source_node}]},  # First call to get source node
        {"results": [  # Second call to get similar nodes
            {
                "node": {"id": "similar-1", "name": "Similar Node 1"},
                "score": 0.92
            },
            {
                "node": {"id": "similar-2", "name": "Similar Node 2"},
                "score": 0.85
            }
        ]}
    ]
    
    # Execute get similar nodes
    result = manager.get_similar_nodes("source-node", limit=5)
    
    # Verify result
    result_list = json.loads(result)
    assert len(result_list) == 2
    assert result_list[0]["id"] == "similar-1"
    assert result_list[1]["id"] == "similar-2"
    assert result_list[0]["score"] == 0.92
    
    # Verify base_manager.execute_query was called twice
    assert mock_base_manager.execute_query.call_count == 2
    
    # First call should get the source node
    first_query = mock_base_manager.execute_query.call_args_list[0][0][0]
    assert "source-node" in first_query
    
    # Second call should search for similar nodes
    second_query = mock_base_manager.execute_query.call_args_list[1][0][0]
    assert "LIMIT 5" in second_query


def test_get_similar_nodes_no_embeddings(mock_base_manager, mock_embedding_adapter, mock_logger):
    """Test similar nodes when embeddings are not available."""
    # Create manager
    manager = SearchManager(mock_base_manager, mock_embedding_adapter, mock_logger)
    
    # Mock embedding availability check
    mock_embedding_adapter.is_available.return_value = False
    
    # Execute get similar nodes
    result = manager.get_similar_nodes("source-node")
    
    # Verify result is empty due to no embedding support
    result_list = json.loads(result)
    assert len(result_list) == 0
    
    # Verify logger.warning was called
    mock_logger.warning.assert_called()
    
    # Should not attempt to execute queries when embeddings unavailable
    mock_base_manager.execute_query.assert_not_called()


def test_get_similar_nodes_source_not_found(mock_base_manager, mock_embedding_adapter, mock_logger):
    """Test similar nodes when source node is not found."""
    # Create manager
    manager = SearchManager(mock_base_manager, mock_embedding_adapter, mock_logger)
    
    # Mock embedding availability check
    mock_embedding_adapter.is_available.return_value = True
    
    # Mock query to get source node - returns empty
    mock_base_manager.execute_query.return_value = {"results": []}
    
    # Execute get similar nodes
    result = manager.get_similar_nodes("nonexistent-node")
    
    # Verify result is empty
    result_list = json.loads(result)
    assert len(result_list) == 0
    
    # Verify base_manager.execute_query was called once (to get source node)
    mock_base_manager.execute_query.assert_called_once()
    
    # Verify logger.error was called
    mock_logger.error.assert_called()


def test_count_search_results(mock_base_manager, mock_embedding_adapter, mock_logger):
    """Test counting search results."""
    # Create manager
    manager = SearchManager(mock_base_manager, mock_embedding_adapter, mock_logger)
    
    # Mock embedding availability check
    mock_embedding_adapter.is_available.return_value = False
    
    # Mock query execution
    mock_base_manager.execute_query.return_value = {"results": [{"count": 42}]}
    
    # Execute count search
    result = manager.count_search_results("count query", filters={"type": "Document"})
    
    # Verify result
    result_obj = json.loads(result)
    assert result_obj["count"] == 42
    
    # Verify base_manager.execute_query was called with correct parameters
    mock_base_manager.execute_query.assert_called_once()
    # Check that the query contains COUNT and filters
    query = mock_base_manager.execute_query.call_args[0][0]
    assert "COUNT" in query
    assert "Document" in query 