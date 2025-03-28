import pytest
import os
from unittest.mock import patch, MagicMock

from src.graph_memory.base_manager import BaseManager


def test_initialization(mock_logger):
    """Test that BaseManager initializes with correct defaults."""
    base_manager = BaseManager(mock_logger)
    
    assert base_manager.logger == mock_logger
    assert base_manager.initialized is False
    assert base_manager.neo4j_driver is None
    assert base_manager.default_project_name == "default-project"
    assert base_manager.neo4j_uri == "bolt://localhost:7687"
    assert base_manager.neo4j_user == "neo4j"
    assert base_manager.neo4j_password == "password"
    assert base_manager.neo4j_database == "neo4j"


@patch('neo4j.GraphDatabase.driver')
def test_initialize_success(mock_driver, mock_logger):
    """Test successful Neo4j connection initialization."""
    # Setup mock driver
    mock_driver_instance = MagicMock()
    mock_driver.return_value = mock_driver_instance
    
    # Configure environment variables
    with patch.dict('os.environ', {
        'NEO4J_URI': 'bolt://localhost:7687',
        'NEO4J_USER': 'neo4j',
        'NEO4J_PASSWORD': 'password',
        'NEO4J_DATABASE': 'neo4j'
    }):
        # Create and initialize the base manager
        base_manager = BaseManager(mock_logger)
        
        # Mock test_neo4j_connection_with_retry to avoid actual connection tests
        base_manager._test_neo4j_connection_with_retry = MagicMock(return_value=True)
        
        # Mock _setup_vector_index to avoid actual index creation
        base_manager._setup_vector_index = MagicMock(return_value=True)
        
        base_manager.initialize()
        
        # Assertions
        assert base_manager.initialized is True
        assert base_manager.neo4j_driver is not None
        mock_driver.assert_called_once_with(
            'bolt://localhost:7687',
            auth=('neo4j', 'password'),
            max_connection_lifetime=30 * 60,
            max_connection_pool_size=50,
            connection_acquisition_timeout=60,
            keep_alive=True
        )


@patch('neo4j.GraphDatabase.driver')
def test_initialize_error(mock_driver, mock_logger):
    """Test that connection errors are handled correctly."""
    # Setup mock driver to raise an exception
    mock_driver.side_effect = Exception("Connection failed")
    
    # Create and try to initialize the base manager
    base_manager = BaseManager(mock_logger)
    
    # Using try/except because the BaseManager raises exceptions on initialization failure
    try:
        base_manager.initialize()
    except Exception:
        # This is expected
        pass
    
    # Assertions
    assert base_manager.initialized is False
    assert base_manager.neo4j_driver is None
    mock_logger.error.assert_called()


@patch('neo4j.GraphDatabase.driver')
def test_safe_execute_query(mock_driver, mock_logger):
    """Test safe_execute_query method with mock session and transaction."""
    # Setup mock driver and session
    mock_driver_instance = MagicMock()
    mock_session = MagicMock()
    mock_driver.return_value = mock_driver_instance
    mock_driver_instance.session.return_value = mock_session
    
    # Mock execute_query to return test data
    expected_results = [{"id": 1, "name": "test"}]
    expected_summary = {"counters": {"nodes_created": 1}}
    
    # Mock the session and its execute_query method
    mock_session.execute_query.return_value = (expected_results, expected_summary)
    
    # Create and initialize base manager
    base_manager = BaseManager(mock_logger)
    base_manager._test_neo4j_connection_with_retry = MagicMock(return_value=True)
    base_manager._setup_vector_index = MagicMock(return_value=True)
    base_manager.initialize()
    
    # Execute a test query
    query = "MATCH (n) RETURN n LIMIT 1"
    params = {"param": "value"}
    results, summary = base_manager.safe_execute_query(query, params)
    
    # Assertions
    assert results == expected_results
    assert summary == expected_summary
    mock_session.execute_query.assert_called_once()


@patch('neo4j.GraphDatabase.driver')
def test_safe_execute_query_error(mock_driver, mock_logger):
    """Test that query execution errors are handled correctly."""
    # Setup mock driver and session
    mock_driver_instance = MagicMock()
    mock_session = MagicMock()
    mock_driver.return_value = mock_driver_instance
    mock_driver_instance.session.return_value = mock_session
    
    # Setup mock to raise an exception
    mock_session.execute_query.side_effect = Exception("Query execution failed")
    
    # Create and initialize base manager
    base_manager = BaseManager(mock_logger)
    base_manager._test_neo4j_connection_with_retry = MagicMock(return_value=True)
    base_manager._setup_vector_index = MagicMock(return_value=True)
    base_manager.initialize()
    
    # Execute a test query that will fail
    query = "INVALID QUERY"
    results, summary = base_manager.safe_execute_query(query)
    
    # Assertions
    assert results == []
    assert summary == {}
    mock_logger.error.assert_called()


@patch('neo4j.GraphDatabase.driver')
def test_setup_vector_index(mock_driver, mock_logger):
    """Test vector index setup."""
    # Setup mock driver and session
    mock_driver_instance = MagicMock()
    mock_session = MagicMock()
    mock_driver.return_value = mock_driver_instance
    mock_driver_instance.session.return_value = mock_session
    
    # Mock successful query execution
    mock_session.execute_query.return_value = ([], {"counters": {"indexes_added": 1}})
    
    # Create base manager with embedding enabled
    with patch.dict('os.environ', {'EMBEDDER_PROVIDER': 'openai'}):
        base_manager = BaseManager(mock_logger)
        base_manager._test_neo4j_connection_with_retry = MagicMock(return_value=True)
        
        # Mock the private _setup_vector_index to test it directly
        original_method = base_manager._setup_vector_index
        base_manager._setup_vector_index = MagicMock(wraps=original_method)
        
        # Initialize to trigger vector index setup
        base_manager.initialize()
        
        # Assertions
        base_manager._setup_vector_index.assert_called_once()
        assert mock_session.execute_query.call_count > 0


@patch('neo4j.GraphDatabase.driver')
def test_close_connection(mock_driver, mock_logger):
    """Test closing the Neo4j connection."""
    # Setup mock driver
    mock_driver_instance = MagicMock()
    mock_driver.return_value = mock_driver_instance
    
    # Create, initialize, and close the connection
    base_manager = BaseManager(mock_logger)
    base_manager._test_neo4j_connection_with_retry = MagicMock(return_value=True)
    base_manager._setup_vector_index = MagicMock(return_value=True)
    base_manager.initialize()
    base_manager.close()
    
    # Assertions
    mock_driver_instance.close.assert_called_once()


@pytest.mark.parametrize("env_vars,expected", [
    # Test with all environment variables set
    (
        {'NEO4J_URI': 'bolt://test:7687', 'NEO4J_USER': 'test_user', 
         'NEO4J_PASSWORD': 'test_pass', 'NEO4J_DATABASE': 'test_db'},
        ('bolt://test:7687', 'test_user', 'test_pass', 'test_db')
    ),
    # Test with defaults for some variables
    (
        {'NEO4J_URI': 'bolt://custom:7687'},
        ('bolt://custom:7687', 'neo4j', 'password', 'neo4j')
    ),
    # Test with empty values
    (
        {'NEO4J_URI': '', 'NEO4J_USER': '', 'NEO4J_PASSWORD': '', 'NEO4J_DATABASE': ''},
        ('bolt://localhost:7687', 'neo4j', 'password', 'neo4j')
    ),
])
def test_environment_configuration(mock_logger, env_vars, expected):
    """Test that environment variables are correctly used for configuration."""
    # Apply environment variables
    with patch.dict('os.environ', env_vars):
        # Create base manager to read environment
        base_manager = BaseManager(mock_logger)
        
        # Assertions
        assert base_manager.neo4j_uri == expected[0]
        assert base_manager.neo4j_user == expected[1]
        assert base_manager.neo4j_password == expected[2]
        assert base_manager.neo4j_database == expected[3] 