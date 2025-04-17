"""
Unit tests for the LiteLLMEmbeddingManager functionality.
"""

import pytest
from unittest.mock import MagicMock, patch
import numpy as np
import os
from dotenv import load_dotenv

# Import logger
from src.logger import get_logger, LogLevel

# Import the embedding manager class
from src.embedding_manager import LiteLLMEmbeddingManager

# Load environment variables
load_dotenv()

class TestLiteLLMEmbeddingManager:
    """Test suite for LiteLLMEmbeddingManager class."""
    
    @pytest.fixture
    def logger(self):
        """Get a configured logger instance."""
        logger = get_logger()
        logger.set_level(LogLevel.DEBUG)
        return logger
    
    @pytest.fixture
    def mock_litellm(self):
        """Create a mock for the litellm module."""
        with patch('src.embedding_manager.embedding') as mock_embedding:
            # Configure the mock embedding response
            mock_response = {
                "data": [
                    {"embedding": [0.1, 0.2, 0.3, 0.4, 0.5] * 20}
                ]
            }
            
            # Set up the embedding function
            mock_embedding.return_value = mock_response
            
            yield mock_embedding
    
    @pytest.fixture
    def embedding_manager(self, logger, mock_litellm):
        """Create a LiteLLMEmbeddingManager instance with a mocked litellm."""
        manager = LiteLLMEmbeddingManager(logger)
        manager.configure({
            "provider": "openai",
            "model": "text-embedding-3-small",
            "dimensions": 100,
            "api_key": "fake-api-key"
        })
        return manager
    
    def test_initialization(self, logger):
        """Test initialization of the embedding manager."""
        manager = LiteLLMEmbeddingManager(logger)
        
        # Verify initial state
        assert manager.provider == "none"
        assert manager.model is None
        assert manager.dimensions is None
        assert manager.embedding_enabled is False
        assert manager.api_key is None
        assert manager.api_base is None
        assert manager.additional_params == {}
    
    def test_configure(self, logger):
        """Test configuration with different parameters."""
        manager = LiteLLMEmbeddingManager(logger)
        
        # Test with minimal configuration (disabled)
        result = manager.configure({"provider": "none"})
        assert result["status"] == "success"
        assert manager.embedding_enabled is False
        
        # Test with OpenAI configuration
        result = manager.configure({
            "provider": "openai",
            "model": "text-embedding-3-small",
            "dimensions": 100,
            "api_key": "fake-api-key"
        })
        assert result["status"] == "success"
        assert manager.provider == "openai"
        assert manager.model == "text-embedding-3-small"
        assert manager.dimensions == 100
        assert manager.api_key == "fake-api-key"
        assert manager.embedding_enabled is True
        
        # Test with Azure configuration
        result = manager.configure({
            "provider": "azure",
            "model": "text-embedding-ada-002",
            "api_key": "fake-azure-key",
            "api_base": "https://example.azure.openai.com",
            "additional_params": {
                "api_version": "2023-05-15",
                "deployment": "my-embedding-deployment"
            }
        })
        assert result["status"] == "success"
        assert manager.provider == "azure"
        assert manager.model == "azure/text-embedding-ada-002" # Azure models get prefixed
        assert manager.api_key == "fake-azure-key"
        assert manager.api_base == "https://example.azure.openai.com"
        assert "deployment" in manager.additional_params
        assert manager.additional_params["deployment"] == "my-embedding-deployment"
    
    def test_generate_embedding(self, embedding_manager, mock_litellm):
        """Test generating an embedding."""
        text = "This is a test text for embedding."
        
        # Generate embedding
        embedding = embedding_manager.generate_embedding(text)
        
        # Verify embedding
        assert isinstance(embedding, list)
        assert len(embedding) == 100  # As configured in the manager
        assert all(isinstance(x, float) for x in embedding)
        
        # Verify litellm was called correctly
        mock_litellm.assert_called_once()
        args, kwargs = mock_litellm.call_args
        assert kwargs["model"] == "text-embedding-3-small"
        assert kwargs["input"] == [text]
        assert "dimensions" in kwargs
        assert kwargs["dimensions"] == 100
    
    def test_generate_embedding_disabled(self, logger):
        """Test that embedding generation fails when disabled."""
        manager = LiteLLMEmbeddingManager(logger)
        # Configure with disabled provider
        manager.configure({"provider": "none"})
        
        # Try to generate embedding
        result = manager.generate_embedding("test text")
        assert result is None
    
    def test_get_config(self, embedding_manager):
        """Test getting the current configuration."""
        config = embedding_manager.get_config()
        
        # Check that the configuration has the expected keys
        assert "provider" in config
        assert "model" in config
        assert "dimensions" in config
        assert "embedding_enabled" in config
        
        # Check values
        assert config["provider"] == "openai"
        assert config["model"] == "text-embedding-3-small" 
        assert config["dimensions"] == 100
        assert config["embedding_enabled"] is True
        
        # Ensure sensitive information is not included
        assert "api_key" not in config
    
    @patch('src.embedding_manager.embedding')
    def test_error_handling(self, mock_embedding, logger):
        """Test error handling in embedding generation."""
        # Configure the mock to raise an exception
        mock_embedding.side_effect = Exception("Test error")
        
        # Create manager and configure it
        manager = LiteLLMEmbeddingManager(logger)
        manager.configure({
            "provider": "openai",
            "model": "text-embedding-3-small",
            "api_key": "fake-api-key"
        })
        
        # Try to generate embedding
        result = manager.generate_embedding("test text")
        assert result is None
    
    # Helper functions for vector operations
    def cosine_similarity(self, vec1, vec2):
        """Calculate cosine similarity between two vectors."""
        dot_product = sum(a * b for a, b in zip(vec1, vec2))
        magnitude1 = sum(a ** 2 for a in vec1) ** 0.5
        magnitude2 = sum(b ** 2 for b in vec2) ** 0.5
        
        if magnitude1 * magnitude2 == 0:
            return 0.0
            
        return dot_product / (magnitude1 * magnitude2)
    
    def euclidean_distance(self, vec1, vec2):
        """Calculate Euclidean distance between two vectors."""
        return sum((a - b) ** 2 for a, b in zip(vec1, vec2)) ** 0.5
    
    def test_vector_operations(self, embedding_manager, mock_litellm):
        """Test vector operations on embeddings."""
        # Setup two different embeddings
        embedding1 = [1.0, 0.0, 0.0] + [0.0] * 97
        embedding2 = [0.0, 1.0, 0.0] + [0.0] * 97
        
        # Calculate cosine similarity
        similarity = self.cosine_similarity(embedding1, embedding2)
        assert similarity == 0.0  # Orthogonal vectors
        
        # Calculate with identical vectors
        similarity = self.cosine_similarity(embedding1, embedding1)
        assert similarity == 1.0  # Identical vectors
        
        # Calculate Euclidean distance
        distance = self.euclidean_distance(embedding1, embedding2)
        assert distance == pytest.approx(np.sqrt(2))
        
        # Test with identical embeddings
        distance = self.euclidean_distance(embedding1, embedding1)
        assert distance == 0.0 