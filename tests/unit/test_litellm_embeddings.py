"""
Unit tests for the LiteLLMEmbeddings class.
"""

import pytest
from unittest.mock import patch, MagicMock
import os
import asyncio

from src.litellm_langchain import LiteLLMEmbeddings


class TestLiteLLMEmbeddings:
    """Test suite for LiteLLMEmbeddings class."""
    
    @pytest.fixture
    def mock_litellm(self):
        """Create a mock for the litellm module."""
        with patch('src.litellm_langchain.litellm') as mock_litellm:
            # Configure the mock embedding response
            mock_response = {
                "data": [
                    {"embedding": [0.1, 0.2, 0.3, 0.4, 0.5] * 20},
                    {"embedding": [0.2, 0.3, 0.4, 0.5, 0.6] * 20},
                    {"embedding": [0.3, 0.4, 0.5, 0.6, 0.7] * 20}
                ]
            }
            
            # Set up the embedding function
            mock_litellm.embedding.return_value = mock_response
            
            yield mock_litellm
    
    @pytest.fixture
    def embeddings(self, mock_litellm):
        """Create a LiteLLMEmbeddings instance with a stubbed litellm module."""
        return LiteLLMEmbeddings(
            model="openai/text-embedding-3-small",
            dimensions=100,
            api_key="fake-api-key"
        )
    
    def test_initialization_minimal(self):
        """Test initialization with minimal parameters."""
        embeddings = LiteLLMEmbeddings(model="openai/text-embedding-3-small")
        assert embeddings.model == "openai/text-embedding-3-small"
        assert embeddings.dimensions is None
        assert embeddings.api_key is None
        assert embeddings.kwargs == {}
    
    def test_initialization_full(self):
        """Test initialization with all parameters."""
        embeddings = LiteLLMEmbeddings(
            model="openai/text-embedding-3-small",
            dimensions=100,
            api_key="fake-api-key",
            extra_param="value"
        )
        assert embeddings.model == "openai/text-embedding-3-small"
        assert embeddings.dimensions == 100
        assert embeddings.api_key == "fake-api-key"
        assert embeddings.kwargs.get("extra_param") == "value"
    
    def test_env_var_setting(self):
        """Test that API keys are set as environment variables."""
        # Preserve original environment
        original_env = os.environ.copy()
        
        try:
            # Clear relevant environment variables
            if "OPENAI_API_KEY" in os.environ:
                del os.environ["OPENAI_API_KEY"]
                
            # Initialize with OpenAI
            _ = LiteLLMEmbeddings(
                model="openai/text-embedding-3-small",
                api_key="test-openai-key"
            )
            assert os.environ.get("OPENAI_API_KEY") == "test-openai-key"
            
            # Initialize with Azure
            _ = LiteLLMEmbeddings(
                model="azure/text-embedding-ada-002",
                api_key="test-azure-key"
            )
            assert os.environ.get("AZURE_API_KEY") == "test-azure-key"
            
            # Initialize with Cohere
            _ = LiteLLMEmbeddings(
                model="cohere/embed-english-v3.0",
                api_key="test-cohere-key"
            )
            assert os.environ.get("COHERE_API_KEY") == "test-cohere-key"
            
        finally:
            # Restore original environment
            os.environ.clear()
            os.environ.update(original_env)
    
    def test_embed_documents(self, embeddings, mock_litellm):
        """Test embedding multiple documents."""
        texts = [
            "This is the first test text.",
            "This is the second test text.",
            "This is the third test text."
        ]
        
        # Get embeddings
        result = embeddings.embed_documents(texts)
        
        # Verify embeddings
        assert isinstance(result, list)
        assert len(result) == 3
        assert all(len(embedding) == 100 for embedding in result)
        
        # Verify litellm was called correctly
        mock_litellm.embedding.assert_called_once()
        args, kwargs = mock_litellm.embedding.call_args
        assert kwargs["model"] == "openai/text-embedding-3-small"
        assert kwargs["input"] == texts
        assert "dimensions" in kwargs
        assert kwargs["dimensions"] == 100
    
    def test_embed_query(self, embeddings, mock_litellm):
        """Test embedding a single query."""
        query = "This is a test query."
        
        # Configure the mock for a single query result
        mock_litellm.embedding.return_value = {
            "data": [{"embedding": [0.1, 0.2, 0.3, 0.4, 0.5] * 20}]
        }
        
        # Get embedding
        result = embeddings.embed_query(query)
        
        # Verify embedding
        assert isinstance(result, list)
        assert len(result) == 100
        assert all(isinstance(x, float) for x in result)
        
        # Verify litellm was called correctly
        args, kwargs = mock_litellm.embedding.call_args
        assert kwargs["model"] == "openai/text-embedding-3-small"
        assert kwargs["input"] == [query]  # Input should be a list with one item
    
    @patch('src.litellm_langchain.litellm.embedding')
    def test_error_handling(self, mock_embedding):
        """Test error handling in embedding methods."""
        # Configure the mock to raise an exception
        mock_embedding.side_effect = Exception("Test error")
        
        # Create embeddings instance
        embeddings = LiteLLMEmbeddings(model="openai/text-embedding-3-small")
        
        # Test embed_documents error handling
        with pytest.raises(ValueError) as excinfo:
            embeddings.embed_documents(["Test text"])
        assert "Error generating embeddings with LiteLLM" in str(excinfo.value)
        
        # Test embed_query error handling
        with pytest.raises(ValueError) as excinfo:
            embeddings.embed_query("Test query")
        assert "Error generating embeddings with LiteLLM" in str(excinfo.value)
    
    @patch('src.litellm_langchain.litellm')
    @patch('src.litellm_langchain.asyncio')
    def test_async_response_handling(self, mock_asyncio, mock_litellm):
        """Test handling of async responses from litellm."""
        # Create a mock response
        mock_response = {
            "data": [{"embedding": [0.1, 0.2, 0.3, 0.4, 0.5] * 20}]
        }
        
        # Set up coroutine detection to return True
        mock_asyncio.iscoroutine.return_value = True
        
        # Create a mock event loop
        mock_loop = MagicMock()
        mock_loop.run_until_complete.return_value = mock_response
        mock_asyncio.get_event_loop.return_value = mock_loop
        
        # Create a mock coroutine object
        mock_coroutine = MagicMock()
        
        # Set up the embedding mock to return our "coroutine"
        mock_litellm.embedding.return_value = mock_coroutine
        
        # Create embeddings instance
        embeddings = LiteLLMEmbeddings(model="openai/text-embedding-3-small")
        
        # Get embedding
        result = embeddings.embed_query("Test async query")
        
        # Verify the mocked event loop was used to run the coroutine
        mock_asyncio.iscoroutine.assert_called_once_with(mock_coroutine)
        mock_asyncio.get_event_loop.assert_called_once()
        mock_loop.run_until_complete.assert_called_once_with(mock_coroutine)
        
        # Verify result
        assert isinstance(result, list)
        assert len(result) == 100
    
    def test_provider_specific_settings_azure(self):
        """Test Azure-specific settings."""
        # Preserve original environment
        original_env = os.environ.copy()
        
        try:
            # Clear relevant environment variables
            for key in ["AZURE_API_KEY", "AZURE_API_VERSION"]:
                if key in os.environ:
                    del os.environ[key]
            
            # Initialize with Azure settings including vertex_credentials_json
            embeddings = LiteLLMEmbeddings(
                model="azure/text-embedding-ada-002",
                api_key="test-azure-key",
                api_version="2023-05-15",
                deployment="my-embedding-deployment"
            )
            
            # Verify API key is set
            assert os.environ.get("AZURE_API_KEY") == "test-azure-key"
            
            # Verify parameters are in kwargs
            assert embeddings.kwargs.get("api_version") == "2023-05-15"
            assert embeddings.kwargs.get("deployment") == "my-embedding-deployment"
            
        finally:
            # Restore original environment
            os.environ.clear()
            os.environ.update(original_env)
    
    def test_vertex_ai_credentials(self):
        """Test handling of VertexAI credentials."""
        # Preserve original environment
        original_env = os.environ.copy()
        
        try:
            # Clear relevant environment variables
            if "GOOGLE_APPLICATION_CREDENTIALS" in os.environ:
                del os.environ["GOOGLE_APPLICATION_CREDENTIALS"]
            
            # Initialize with VertexAI settings including vertex_credentials_json
            _ = LiteLLMEmbeddings(
                model="vertexai/text-embedding-gecko",
                api_key="test-vertex-key",
                vertex_credentials_json="/path/to/credentials.json"
            )
            
            # Verify credential path is set
            assert os.environ.get("GOOGLE_APPLICATION_CREDENTIALS") == "/path/to/credentials.json"
            
        finally:
            # Restore original environment
            os.environ.clear()
            os.environ.update(original_env) 