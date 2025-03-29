import pytest
from unittest.mock import MagicMock, patch, call, mock_open
import json
import os
import sys
from typing import Any, Dict, List, Optional

from src.graph_memory.embedding_adapter import EmbeddingAdapter
from src.embedding_manager import LiteLLMEmbeddingManager

@pytest.fixture
def mock_logger():
    """Fixture for mock logger"""
    logger = MagicMock()
    return logger

@pytest.fixture
def embedding_config():
    """Fixture for embedding configuration"""
    return {
        "provider": "openai",
        "model": "text-embedding-ada-002",
        "api_key": "sk-test123456789"
    }

@pytest.fixture
def mock_embedding_manager():
    """Fixture for mock LiteLLMEmbeddingManager"""
    manager = MagicMock(spec=LiteLLMEmbeddingManager)
    manager.provider = "openai"
    manager.model = "text-embedding-ada-002"
    manager.dimensions = 1536
    manager.embedding_enabled = True
    manager.generate_embedding.return_value = [0.1, 0.2, 0.3, 0.4]
    manager.configure.return_value = {"status": "success", "message": "Configured successfully"}
    manager.get_config.return_value = {
        "provider": "openai",
        "model": "text-embedding-ada-002",
        "dimensions": 1536,
        "embedding_enabled": True
    }
    return manager

@pytest.fixture
def mock_openai():
    """Fixture for mock OpenAI client"""
    mock_client = MagicMock()
    mock_client.embeddings.create.return_value = MagicMock(
        data=[MagicMock(embedding=[0.1, 0.2, 0.3, 0.4])]
    )
    return mock_client

def test_init(mock_logger):
    """Test initialization of EmbeddingAdapter."""
    adapter = EmbeddingAdapter(logger=mock_logger)
    
    assert adapter.logger == mock_logger
    assert adapter.embedding_manager is None
    assert adapter._initialized is False
    assert adapter.model_name == "unknown"
    assert adapter.embedding_dim == 1536  # Default OpenAI dimension

@patch('src.graph_memory.embedding_adapter.LiteLLMEmbeddingManager')
def test_init_embedding_manager_success(mock_manager_class, mock_logger):
    """Test successful initialization of embedding manager."""
    # Setup mock manager
    mock_manager = mock_manager_class.return_value
    mock_manager.configure.return_value = {"status": "success", "message": "Configured successfully"}
    # Set attributes that will be accessed
    mock_manager.model = "text-embedding-3-small"
    mock_manager.dimensions = 1536
    
    # Create adapter
    adapter = EmbeddingAdapter(logger=mock_logger)
    
    # Initialize embedding manager
    result = adapter.init_embedding_manager(
        api_key="test_key",
        model_name="text-embedding-3-small"
    )
    
    # Verify result
    assert result is True
    assert adapter._initialized is True
    assert adapter.model_name == "text-embedding-3-small"
    assert adapter.embedding_dim == 1536

@patch('src.graph_memory.embedding_adapter.LiteLLMEmbeddingManager')
def test_init_embedding_manager_error(mock_manager_class, mock_logger):
    """Test error handling in embedding manager initialization."""
    # Setup mock manager to fail
    mock_manager = mock_manager_class.return_value
    mock_manager.configure.return_value = {"status": "error", "message": "API key error"}
    
    # Create adapter
    adapter = EmbeddingAdapter(logger=mock_logger)
    
    # Initialize embedding manager - should fail due to error status
    result = adapter.init_embedding_manager(
        api_key="invalid_key",
        model_name="text-embedding-3-small"
    )
    
    # Verify result is False
    assert result is False
    mock_logger.error.assert_called()

@patch('src.graph_memory.embedding_adapter.LiteLLMEmbeddingManager')
def test_init_embedding_manager_exception(mock_manager_class, mock_logger):
    """Test exception handling in embedding manager initialization."""
    # Setup mock manager to raise exception
    mock_manager = mock_manager_class.return_value
    mock_manager.configure.side_effect = Exception("API key error")
    
    # Create adapter
    adapter = EmbeddingAdapter(logger=mock_logger)
    
    # Initialize embedding manager - should fail due to exception
    result = adapter.init_embedding_manager(
        api_key="test_key",
        model_name="text-embedding-3-small"
    )
    
    # Verify result
    assert result is False
    mock_logger.error.assert_called()

def test_ensure_initialized(mock_logger):
    """Test ensure_initialized method."""
    with patch.object(EmbeddingAdapter, 'init_embedding_manager', return_value=True) as mock_init:
        adapter = EmbeddingAdapter(logger=mock_logger)
        
        # Not initialized yet
        assert adapter._initialized is False
        
        # Call ensure_initialized
        result = adapter.ensure_initialized()
        
        # Verify result
        assert result is True
        mock_init.assert_called_once()

def test_ensure_initialized_already_initialized(mock_logger, mock_embedding_manager):
    """Test ensure_initialized when already initialized."""
    adapter = EmbeddingAdapter(embedding_manager=mock_embedding_manager, logger=mock_logger)
    
    # Already initialized
    assert adapter._initialized is True
    
    # Call ensure_initialized
    with patch.object(EmbeddingAdapter, 'init_embedding_manager') as mock_init:
        result = adapter.ensure_initialized()
        
        # Verify result
        assert result is True
        mock_init.assert_not_called()

@patch('src.graph_memory.embedding_adapter.LiteLLMEmbeddingManager')
def test_get_embedding_success(mock_manager_class, mock_logger):
    """Test successful embedding generation."""
    # Setup mock manager
    mock_manager = mock_manager_class.return_value
    mock_manager.configure.return_value = {"status": "success", "message": "Configured successfully"}
    mock_manager.generate_embedding.return_value = [0.1, 0.2, 0.3, 0.4]
    mock_manager.model = "text-embedding-3-small"
    mock_manager.dimensions = 1536
    
    # Create adapter and patch _clean_text to return the input text unchanged
    adapter = EmbeddingAdapter(logger=mock_logger)
    with patch.object(adapter, '_clean_text', side_effect=lambda x: x):
        # Initialize embedding manager
        adapter.init_embedding_manager(api_key="test_key")
        
        # Get embedding
        result = adapter.get_embedding("test text")
        
        # Verify result
        assert result == [0.1, 0.2, 0.3, 0.4]
        mock_manager.generate_embedding.assert_called_once()

@patch('src.graph_memory.embedding_adapter.LiteLLMEmbeddingManager')
def test_get_embedding_not_initialized(mock_manager_class, mock_logger):
    """Test get_embedding when not initialized."""
    # Create adapter (not initialized)
    adapter = EmbeddingAdapter(logger=mock_logger)
    
    # Patch ensure_initialized to return False
    with patch.object(adapter, 'ensure_initialized', return_value=False):
        # Get embedding
        result = adapter.get_embedding("test text")
        
        # Verify result is None
        assert result is None
        mock_logger.error.assert_called()

@patch('src.graph_memory.embedding_adapter.LiteLLMEmbeddingManager')
def test_get_embedding_manager_error(mock_manager_class, mock_logger):
    """Test error handling in get_embedding."""
    # Setup mock manager
    mock_manager = mock_manager_class.return_value
    mock_manager.configure.return_value = {"status": "success", "message": "Configured successfully"}
    mock_manager.generate_embedding.return_value = None
    mock_manager.model = "text-embedding-3-small"
    mock_manager.dimensions = 1536
    
    # Create adapter and patch _clean_text
    adapter = EmbeddingAdapter(logger=mock_logger)
    with patch.object(adapter, '_clean_text', side_effect=lambda x: x):
        # Initialize embedding manager
        adapter.init_embedding_manager(api_key="test_key")
        
        # Get embedding - should return None since generate_embedding returns None
        result = adapter.get_embedding("test text")
        
        # Verify result is None and warning was logged
        assert result is None
        mock_logger.warning.assert_called()

@patch('src.graph_memory.embedding_adapter.LiteLLMEmbeddingManager')
def test_get_embedding_exception(mock_manager_class, mock_logger):
    """Test exception handling in get_embedding."""
    # Setup mock manager
    mock_manager = mock_manager_class.return_value
    mock_manager.configure.return_value = {"status": "success", "message": "Configured successfully"}
    mock_manager.generate_embedding.side_effect = Exception("API error")
    mock_manager.model = "text-embedding-3-small"
    mock_manager.dimensions = 1536
    
    # Create adapter and patch _clean_text
    adapter = EmbeddingAdapter(logger=mock_logger)
    with patch.object(adapter, '_clean_text', side_effect=lambda x: x):
        # Initialize embedding manager
        adapter.init_embedding_manager(api_key="test_key")
        
        # Get embedding - should catch the exception
        result = adapter.get_embedding("test text")
        
        # Verify result is None and error was logged
        assert result is None
        mock_logger.error.assert_called()

@patch('src.graph_memory.embedding_adapter.LiteLLMEmbeddingManager')
def test_get_embeddings_batch(mock_manager_class, mock_logger):
    """Test batch embedding generation."""
    # Setup mock manager
    mock_manager = mock_manager_class.return_value
    mock_manager.configure.return_value = {"status": "success", "message": "Configured successfully"}
    mock_manager.model = "text-embedding-3-small"
    mock_manager.dimensions = 1536
    
    # Create adapter
    adapter = EmbeddingAdapter(logger=mock_logger)
    
    # Mock the get_embedding method directly
    with patch.object(adapter, 'get_embedding') as mock_get_embedding:
        mock_get_embedding.side_effect = [
            [0.1, 0.2, 0.3, 0.4],
            [0.5, 0.6, 0.7, 0.8]
        ]
        
        # Initialize embedding manager
        adapter.init_embedding_manager(api_key="test_key")
        
        # Get batch embeddings
        texts = ["First text", "Second text"]
        results = adapter.get_embeddings_batch(texts)
        
        # Verify results
        assert len(results) == 2
        assert results[0] == [0.1, 0.2, 0.3, 0.4]
        assert results[1] == [0.5, 0.6, 0.7, 0.8]
        assert mock_get_embedding.call_count == 2

@patch('src.graph_memory.embedding_adapter.LiteLLMEmbeddingManager')
def test_get_embeddings_batch_not_initialized(mock_manager_class, mock_logger):
    """Test batch embedding when not initialized."""
    # Create adapter (not initialized)
    adapter = EmbeddingAdapter(logger=mock_logger)
    
    # Patch ensure_initialized to return False
    with patch.object(adapter, 'ensure_initialized', return_value=False):
        # Get batch embeddings
        texts = ["First text", "Second text"]
        results = adapter.get_embeddings_batch(texts)
        
        # Verify results are None
        assert results == [None, None]
        mock_logger.error.assert_called()

def test_similarity_score(mock_logger):
    """Test similarity score calculation."""
    adapter = EmbeddingAdapter(logger=mock_logger)
    
    # Calculate similarity between two embeddings
    emb1 = [1.0, 0.0, 0.0]
    emb2 = [0.0, 1.0, 0.0]
    result = adapter.similarity_score(emb1, emb2)
    
    # Verify result is 0.0 (orthogonal vectors)
    assert result == 0.0
    
    # Calculate similarity between identical embeddings
    emb3 = [0.5, 0.5, 0.5]
    emb4 = [0.5, 0.5, 0.5]
    result = adapter.similarity_score(emb3, emb4)
    
    # Verify result is 1.0 (identical vectors)
    assert result == 1.0
    
    # Calculate similarity with edge cases
    assert adapter.similarity_score([], []) == 0.0
    assert adapter.similarity_score([0.0], []) == 0.0  # Use empty list instead of None
    assert adapter.similarity_score([0, 0, 0], [1, 1, 1]) == 0.0

def test_find_most_similar(mock_logger):
    """Test finding most similar embeddings."""
    adapter = EmbeddingAdapter(logger=mock_logger)
    
    # Mock the similarity_score method to return expected values
    with patch.object(adapter, 'similarity_score') as mock_score:
        # Set up return values for different calls
        # First call with query and candidates[0] returns 1.0
        # Second call with query and candidates[1] returns 0.0
        # Third call with query and candidates[2] returns 0.5
        mock_score.side_effect = [1.0, 0.0, 0.5]
        
        # Define query and candidates
        query = [1.0, 0.0, 0.0]
        candidates = [
            [1.0, 0.0, 0.0],  # Identical (score = 1.0)
            [0.0, 1.0, 0.0],  # Orthogonal (score = 0.0)
            [0.5, 0.5, 0.0],  # In between (score = 0.5)
        ]
        
        # Find most similar (top 2)
        results = adapter.find_most_similar(query, candidates, top_k=2)
        
        # Verify results
        assert len(results) == 2
        assert results[0]["index"] == 0
        assert results[0]["score"] == 1.0
        assert results[1]["index"] == 2
        assert results[1]["score"] == 0.5

def test_find_most_similar_empty(mock_logger):
    """Test finding most similar with empty inputs."""
    adapter = EmbeddingAdapter(logger=mock_logger)
    
    # Test with empty query
    assert adapter.find_most_similar([], [[1, 2, 3]]) == []
    
    # Test with empty candidates
    assert adapter.find_most_similar([1, 2, 3], []) == []
    
    # Test with exception
    with patch.object(adapter, 'similarity_score', side_effect=Exception("Error")):
        assert adapter.find_most_similar([1, 2, 3], [[1, 2, 3]]) == []
        assert mock_logger.error.called

def test_clean_text(mock_logger):
    """Test text cleaning for embeddings."""
    adapter = EmbeddingAdapter(logger=mock_logger)
    
    # Test basic cleaning
    result = adapter._clean_text("  Test with extra  spaces  ")
    assert "Test with extra spaces" in result
    
    # Test with empty string instead of None
    assert adapter._clean_text("") == ""
    
    # Test with empty string
    assert adapter._clean_text("") == "" 