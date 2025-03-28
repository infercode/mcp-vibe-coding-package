import pytest
import os
from unittest.mock import patch, MagicMock

from src.graph_memory.base_manager import BaseManager


def test_initialization(mock_logger):
    """Test that BaseManager initializes with correct defaults."""
    base_manager = BaseManager(mock_logger)
    
    assert base_manager.logger == mock_logger
    assert base_manager.driver is None
    assert base_manager.max_retries == 3
    assert base_manager.retry_delay == 1.0


@patch('neo4j.GraphDatabase.driver')
def test_initialize_connection_success(mock_driver, mock_logger):
    """Test successful Neo4j connection initialization."""
    # Setup mock driver
    mock_driver_instance = MagicMock()
    mock_driver.return_value = mock_driver_instance
    
    # Configure environment variables
    with patch.dict('os.environ', {
        'NEO4J_URI': 'bolt://localhost:7687',
        'NEO4J_USERNAME': 'neo4j',
        'NEO4J_PASSWORD': 'password',
        'NEO4J_DATABASE': 'neo4j'
    }):
        # Create and initialize the base manager
        base_manager = BaseManager(mock_logger)
        success = base_manager.initialize()
        
        # Assertions
        assert success is True
        mock_driver.assert_called_once_with(
            'bolt://localhost:7687',
            auth=('neo4j', 'password')
        )
        assert base_manager.database == 'neo4j'
        assert base_manager.driver is not None


@patch('neo4j.GraphDatabase.driver')
def test_initialize_connection_error(mock_driver, mock_logger):
    """Test that connection errors are handled correctly."""
    # Setup mock driver to raise an exception
    mock_driver.side_effect = Exception("Connection failed")
    
    # Create and initialize the base manager
    base_manager = BaseManager(mock_logger)
    success = base_manager.initialize()
    
    # Assertions
    assert success is False
    assert base_manager.driver is None
    mock_logger.error.assert_called()


@patch('neo4j.GraphDatabase.driver')
def test_safe_execute_query_success(mock_driver, mock_logger):
    """Test successful query execution."""
    # Setup mock driver and session
    mock_driver_instance = MagicMock()
    mock_session = MagicMock()
    mock_transaction = MagicMock()
    mock_driver.return_value = mock_driver_instance
    mock_driver_instance.session.return_value = mock_session
    mock_session.__enter__.return_value = mock_session
    mock_session.begin_transaction.return_value = mock_transaction
    mock_transaction.__enter__.return_value = mock_transaction
    
    # Setup mock result
    mock_result = MagicMock()
    mock_result.data.return_value = [{"result": "value"}]
    mock_transaction.run.return_value = mock_result
    
    # Create and initialize base manager
    base_manager = BaseManager(mock_logger)
    base_manager.initialize()
    
    # Execute a test query
    query = "MATCH (n) RETURN n LIMIT 1"
    params = {"param": "value"}
    result = base_manager.execute_query(query, params)
    
    # Assertions
    assert result == [{"result": "value"}]
    mock_session.begin_transaction.assert_called_once()
    mock_transaction.run.assert_called_once_with(query, params)


@patch('neo4j.GraphDatabase.driver')
def test_safe_execute_query_error(mock_driver, mock_logger):
    """Test that query execution errors are handled correctly."""
    # Setup mock driver and session
    mock_driver_instance = MagicMock()
    mock_session = MagicMock()
    mock_driver.return_value = mock_driver_instance
    mock_driver_instance.session.return_value = mock_session
    mock_session.__enter__.return_value = mock_session
    
    # Set up mock transaction to raise an exception
    mock_session.begin_transaction.side_effect = Exception("Query execution failed")
    
    # Create and initialize base manager
    base_manager = BaseManager(mock_logger)
    base_manager.initialize()
    
    # Execute a test query that will fail
    query = "INVALID QUERY"
    result = base_manager.execute_query(query)
    
    # Assertions
    assert result is None
    mock_logger.error.assert_called()


@patch('neo4j.GraphDatabase.driver')
def test_vector_index_setup(mock_driver, mock_logger):
    """Test that vector index is set up correctly."""
    # Setup mock driver and session
    mock_driver_instance = MagicMock()
    mock_session = MagicMock()
    mock_transaction = MagicMock()
    mock_driver.return_value = mock_driver_instance
    mock_driver_instance.session.return_value = mock_session
    mock_session.__enter__.return_value = mock_session
    mock_session.begin_transaction.return_value = mock_transaction
    mock_transaction.__enter__.return_value = mock_transaction
    
    # Create and initialize base manager
    base_manager = BaseManager(mock_logger)
    base_manager.initialize()
    
    # Setup the vector index
    result = base_manager.setup_vector_index()
    
    # Assertions
    assert result is True
    assert mock_transaction.run.call_count >= 1


@patch('neo4j.GraphDatabase.driver')
def test_close_connection(mock_driver, mock_logger):
    """Test that the connection is closed correctly."""
    # Setup mock driver
    mock_driver_instance = MagicMock()
    mock_driver.return_value = mock_driver_instance
    
    # Create, initialize, and close the connection
    base_manager = BaseManager(mock_logger)
    base_manager.initialize()
    base_manager.close()
    
    # Assertions
    mock_driver_instance.close.assert_called_once()
    assert base_manager.driver is None


@patch('neo4j.GraphDatabase.driver')
def test_connection_retry_logic(mock_driver, mock_logger):
    """Test that connection retry logic works correctly."""
    # Setup mock driver to fail on first attempt but succeed on second
    mock_driver_instance = MagicMock()
    mock_driver.side_effect = [Exception("First attempt failed"), mock_driver_instance]
    
    # Create and initialize base manager with retry set to 2
    base_manager = BaseManager(mock_logger)
    base_manager.max_retries = 2
    success = base_manager.initialize()
    
    # Assertions
    assert success is True
    assert mock_driver.call_count == 2
    mock_logger.warning.assert_called()


@pytest.mark.parametrize("env_vars,expected", [
    # Test with all environment variables set
    (
        {'NEO4J_URI': 'bolt://test:7687', 'NEO4J_USERNAME': 'test_user', 
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
        {'NEO4J_URI': '', 'NEO4J_USERNAME': '', 'NEO4J_PASSWORD': '', 'NEO4J_DATABASE': ''},
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