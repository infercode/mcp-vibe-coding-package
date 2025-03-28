import pytest
from unittest.mock import MagicMock, patch, call, mock_open
import json
import os

from src.graph_memory.embedding_adapter import EmbeddingAdapter


@pytest.fixture
def embedding_config():
    """Fixture for embedding configuration"""
    return {
        "provider": "openai",
        "model": "text-embedding-ada-002",
        "api_key": "sk-test123456789"
    }


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
    assert adapter.openai_client is None
    assert adapter.model_name is None
    assert adapter.embedding_provider is None
    assert adapter.embedding_dimension == 1536  # Default OpenAI dimension


def test_load_config_no_file(mock_logger):
    """Test loading configuration when file doesn't exist."""
    adapter = EmbeddingAdapter(logger=mock_logger)
    
    # Mock os.path.exists to return False (file doesn't exist)
    with patch("os.path.exists", return_value=False):
        adapter.load_config()
    
    # Verify that configuration is not loaded
    assert adapter.openai_client is None
    assert adapter.model_name is None
    assert adapter.embedding_provider is None
    
    # Verify warning was logged
    mock_logger.warning.assert_called()


def test_load_config_with_file(mock_logger, embedding_config):
    """Test loading configuration from file."""
    adapter = EmbeddingAdapter(logger=mock_logger)
    
    # Mock os.path.exists to return True
    with patch("os.path.exists", return_value=True):
        # Mock open to return config file
        with patch("builtins.open", mock_open(read_data=json.dumps(embedding_config))):
            adapter.load_config()
    
    # Verify configuration is loaded
    assert adapter.embedding_provider == embedding_config["provider"]
    assert adapter.model_name == embedding_config["model"]
    
    # Verify logger was called
    mock_logger.info.assert_called()


def test_load_config_openai(mock_logger, embedding_config, mock_openai):
    """Test loading OpenAI configuration."""
    adapter = EmbeddingAdapter(logger=mock_logger)
    
    # Mock OpenAI client creation
    with patch("openai.OpenAI", return_value=mock_openai):
        # Mock os.path.exists to return True
        with patch("os.path.exists", return_value=True):
            # Mock open to return config file
            with patch("builtins.open", mock_open(read_data=json.dumps(embedding_config))):
                adapter.load_config()
    
    # Verify OpenAI client is created
    assert adapter.openai_client is not None
    assert adapter.embedding_provider == "openai"
    assert adapter.model_name == embedding_config["model"]


def test_load_config_error(mock_logger, embedding_config):
    """Test error handling during config loading."""
    adapter = EmbeddingAdapter(logger=mock_logger)
    
    # Corrupt the config by removing required field
    bad_config = embedding_config.copy()
    del bad_config["provider"]
    
    # Mock os.path.exists to return True
    with patch("os.path.exists", return_value=True):
        # Mock open to return corrupt config file
        with patch("builtins.open", mock_open(read_data=json.dumps(bad_config))):
            adapter.load_config()
    
    # Verify error was logged
    mock_logger.error.assert_called()
    
    # Configuration should not be applied
    assert adapter.embedding_provider is None


def test_is_available_false(mock_logger):
    """Test is_available returns False when not configured."""
    adapter = EmbeddingAdapter(logger=mock_logger)
    
    # Check availability
    assert adapter.is_available() is False


def test_is_available_true(mock_logger, embedding_config, mock_openai):
    """Test is_available returns True when properly configured."""
    adapter = EmbeddingAdapter(logger=mock_logger)
    
    # Configure the adapter
    with patch("openai.OpenAI", return_value=mock_openai):
        # Directly set configuration
        adapter.embedding_provider = embedding_config["provider"]
        adapter.model_name = embedding_config["model"]
        adapter.openai_client = mock_openai
    
    # Check availability
    assert adapter.is_available() is True


def test_get_embedding_openai(mock_logger, mock_openai):
    """Test getting embedding using OpenAI."""
    adapter = EmbeddingAdapter(logger=mock_logger)
    
    # Configure the adapter with OpenAI
    adapter.embedding_provider = "openai"
    adapter.model_name = "text-embedding-ada-002"
    adapter.openai_client = mock_openai
    
    # Get embedding
    text = "This is a test text"
    embedding = adapter.get_embedding(text)
    
    # Verify embedding is returned
    assert embedding is not None
    assert len(embedding) == 4  # From our mock
    assert isinstance(embedding, list)
    
    # Verify OpenAI client was called with correct parameters
    mock_openai.embeddings.create.assert_called_once()
    call_args = mock_openai.embeddings.create.call_args[1]
    assert call_args["model"] == adapter.model_name
    assert call_args["input"] == text


def test_get_embedding_unsupported_provider(mock_logger):
    """Test error when using unsupported provider."""
    adapter = EmbeddingAdapter(logger=mock_logger)
    
    # Configure the adapter with unsupported provider
    adapter.embedding_provider = "unsupported_provider"
    adapter.model_name = "some_model"
    
    # Get embedding
    text = "This is a test text"
    
    # Should raise exception for unsupported provider
    with pytest.raises(ValueError):
        adapter.get_embedding(text)
    
    # Verify error was logged
    mock_logger.error.assert_called()


def test_get_embedding_not_configured(mock_logger):
    """Test error when getting embedding without configuration."""
    adapter = EmbeddingAdapter(logger=mock_logger)
    
    # Adapter not configured
    
    # Get embedding
    text = "This is a test text"
    
    # Should raise exception for not being configured
    with pytest.raises(ValueError):
        adapter.get_embedding(text)
    
    # Verify error was logged
    mock_logger.error.assert_called()


def test_get_embedding_api_error(mock_logger):
    """Test handling API errors during embedding retrieval."""
    adapter = EmbeddingAdapter(logger=mock_logger)
    
    # Configure the adapter
    adapter.embedding_provider = "openai"
    adapter.model_name = "text-embedding-ada-002"
    
    # Create mock client that raises an exception
    mock_client = MagicMock()
    mock_client.embeddings.create.side_effect = Exception("API Error")
    adapter.openai_client = mock_client
    
    # Get embedding
    text = "This is a test text"
    
    # Should raise exception for API error
    with pytest.raises(Exception):
        adapter.get_embedding(text)
    
    # Verify error was logged
    mock_logger.error.assert_called()


def test_set_config(mock_logger, embedding_config):
    """Test setting configuration directly."""
    adapter = EmbeddingAdapter(logger=mock_logger)
    
    # Mock OpenAI client creation
    with patch("openai.OpenAI") as mock_openai_class:
        # Set configuration
        adapter.set_config(embedding_config)
    
    # Verify configuration was set
    assert adapter.embedding_provider == embedding_config["provider"]
    assert adapter.model_name == embedding_config["model"]
    
    # Verify OpenAI client was created
    mock_openai_class.assert_called_once()


def test_set_config_file_creation(mock_logger, embedding_config):
    """Test that configuration file is created when setting config."""
    adapter = EmbeddingAdapter(logger=mock_logger)
    
    # Mock directory creation and file writing
    with patch("os.makedirs") as mock_makedirs:
        with patch("builtins.open", mock_open()) as mock_file:
            # Mock OpenAI client creation
            with patch("openai.OpenAI"):
                # Set configuration
                adapter.set_config(embedding_config)
    
    # Verify directory was created
    mock_makedirs.assert_called_once()
    
    # Verify file was written
    mock_file.assert_called_once()
    
    # Verify content written to file
    handle = mock_file()
    handle.write.assert_called_once()
    # Check that config was serialized to JSON
    write_arg = handle.write.call_args[0][0]
    assert json.loads(write_arg) == embedding_config


def test_embedding_dimension(mock_logger, embedding_config):
    """Test that embedding dimension is correctly set."""
    adapter = EmbeddingAdapter(logger=mock_logger)
    
    # Set configuration
    with patch("openai.OpenAI"):
        adapter.set_config(embedding_config)
    
    # Default dimension for OpenAI
    assert adapter.embedding_dimension == 1536


def test_batch_get_embeddings(mock_logger, mock_openai):
    """Test getting embeddings for multiple texts."""
    adapter = EmbeddingAdapter(logger=mock_logger)
    
    # Configure the adapter with OpenAI
    adapter.embedding_provider = "openai"
    adapter.model_name = "text-embedding-ada-002"
    adapter.openai_client = mock_openai
    
    # Mock batch embeddings
    mock_openai.embeddings.create.return_value = MagicMock(
        data=[
            MagicMock(embedding=[0.1, 0.2, 0.3, 0.4]),
            MagicMock(embedding=[0.5, 0.6, 0.7, 0.8])
        ]
    )
    
    # Get embeddings for multiple texts
    texts = ["First text", "Second text"]
    embeddings = adapter.batch_get_embeddings(texts)
    
    # Verify embeddings are returned
    assert embeddings is not None
    assert len(embeddings) == 2
    assert isinstance(embeddings, list)
    assert all(isinstance(emb, list) for emb in embeddings)
    
    # Verify OpenAI client was called with correct parameters
    mock_openai.embeddings.create.assert_called_once()
    call_args = mock_openai.embeddings.create.call_args[1]
    assert call_args["model"] == adapter.model_name
    assert call_args["input"] == texts


def test_save_config(mock_logger, embedding_config):
    """Test saving configuration to file."""
    adapter = EmbeddingAdapter(logger=mock_logger)
    
    # Set configuration without saving to file
    adapter.embedding_provider = embedding_config["provider"]
    adapter.model_name = embedding_config["model"]
    
    # Now save configuration
    with patch("os.makedirs") as mock_makedirs:
        with patch("builtins.open", mock_open()) as mock_file:
            adapter.save_config(embedding_config)
    
    # Verify directory was created
    mock_makedirs.assert_called_once()
    
    # Verify file was written
    mock_file.assert_called_once()
    
    # Verify content written to file
    handle = mock_file()
    handle.write.assert_called_once()
    # Check that config was serialized to JSON
    write_arg = handle.write.call_args[0][0]
    assert json.loads(write_arg) == embedding_config


def test_init_embedding_manager_success():
    """Test successful initialization of embedding manager."""
    # Mock logger
    mock_logger = MagicMock()
    
    # Create adapter
    adapter = EmbeddingAdapter(logger=mock_logger)
    
    # Test with OpenAI provider
    with patch('src.graph_memory.embedding_adapter.OpenAIEmbeddings') as mock_openai:
        result = adapter.init_embedding_manager(
            api_key="test_key",
            model_name="text-embedding-3-small",
            provider="openai"
        )
        
        # Verify result
        assert result is True
        assert adapter.embedding_enabled is True
        assert adapter.embedding_provider == "openai"
        assert adapter.embedding_model == "text-embedding-3-small"
        assert adapter.embedding_api_key == "test_key"
        
        # Verify OpenAI was initialized correctly
        mock_openai.assert_called_once_with(
            api_key="test_key",
            model="text-embedding-3-small"
        )


def test_init_embedding_manager_error():
    """Test error handling in embedding manager initialization."""
    # Mock logger
    mock_logger = MagicMock()
    
    # Create adapter
    adapter = EmbeddingAdapter(logger=mock_logger)
    
    # Test with OpenAI provider that raises an exception
    with patch('src.graph_memory.embedding_adapter.OpenAIEmbeddings', 
              side_effect=Exception("API key error")) as mock_openai:
        result = adapter.init_embedding_manager(
            api_key="invalid_key",
            model_name="text-embedding-3-small",
            provider="openai"
        )
        
        # Verify result
        assert result is False
        assert adapter.embedding_enabled is False
        assert mock_logger.error.called


def test_init_embedding_manager_invalid_provider():
    """Test initialization with an invalid provider."""
    # Mock logger
    mock_logger = MagicMock()
    
    # Create adapter
    adapter = EmbeddingAdapter(logger=mock_logger)
    
    # Test with invalid provider
    result = adapter.init_embedding_manager(
        api_key="test_key",
        model_name="test-model",
        provider="nonexistent_provider"
    )
    
    # Verify result
    assert result is False
    assert adapter.embedding_enabled is False
    assert mock_logger.error.called


def test_get_embedding_success():
    """Test successful embedding generation."""
    # Mock logger
    mock_logger = MagicMock()
    
    # Create adapter
    adapter = EmbeddingAdapter(logger=mock_logger)
    
    # Mock embedding generator
    mock_embedding_generator = MagicMock()
    mock_embedding_generator.embed_query.return_value = [0.1, 0.2, 0.3, 0.4]
    adapter.embedding_generator = mock_embedding_generator
    adapter.embedding_enabled = True
    
    # Get embedding
    result = adapter.get_embedding("test text")
    
    # Verify result
    assert result == [0.1, 0.2, 0.3, 0.4]
    mock_embedding_generator.embed_query.assert_called_once_with("test text")


def test_get_embedding_not_enabled():
    """Test get_embedding when embedding is not enabled."""
    # Mock logger
    mock_logger = MagicMock()
    
    # Create adapter with embedding disabled
    adapter = EmbeddingAdapter(logger=mock_logger)
    adapter.embedding_enabled = False
    
    # Get embedding
    result = adapter.get_embedding("test text")
    
    # Verify result is None
    assert result is None
    assert mock_logger.warning.called


def test_get_embedding_error():
    """Test error handling in get_embedding."""
    # Mock logger
    mock_logger = MagicMock()
    
    # Create adapter
    adapter = EmbeddingAdapter(logger=mock_logger)
    
    # Mock embedding generator that raises an exception
    mock_embedding_generator = MagicMock()
    mock_embedding_generator.embed_query.side_effect = Exception("API error")
    adapter.embedding_generator = mock_embedding_generator
    adapter.embedding_enabled = True
    
    # Get embedding
    result = adapter.get_embedding("test text")
    
    # Verify result is None and error was logged
    assert result is None
    assert mock_logger.error.called


def test_configure_embedding_success():
    """Test successful embedding configuration."""
    # Mock logger
    mock_logger = MagicMock()
    
    # Create adapter
    adapter = EmbeddingAdapter(logger=mock_logger)
    
    # Mock the init_embedding_manager method
    with patch.object(EmbeddingAdapter, 'init_embedding_manager', return_value=True) as mock_init:
        # Configure embedding
        config = {
            "provider": "openai",
            "model": "text-embedding-3-small",
            "api_key": "test_key"
        }
        result = adapter.configure_embedding(config)
        
        # Verify result
        assert result["status"] == "success"
        mock_init.assert_called_once_with(
            api_key="test_key",
            model_name="text-embedding-3-small",
            provider="openai"
        )


def test_configure_embedding_error():
    """Test error handling in embedding configuration."""
    # Mock logger
    mock_logger = MagicMock()
    
    # Create adapter
    adapter = EmbeddingAdapter(logger=mock_logger)
    
    # Mock the init_embedding_manager method to fail
    with patch.object(EmbeddingAdapter, 'init_embedding_manager', return_value=False) as mock_init:
        # Configure embedding
        config = {
            "provider": "openai",
            "model": "text-embedding-3-small",
            "api_key": "invalid_key"
        }
        result = adapter.configure_embedding(config)
        
        # Verify result
        assert result["status"] == "error"
        assert "Failed to initialize" in result["message"]


def test_configure_embedding_missing_params():
    """Test embedding configuration with missing parameters."""
    # Mock logger
    mock_logger = MagicMock()
    
    # Create adapter
    adapter = EmbeddingAdapter(logger=mock_logger)
    
    # Test with missing required parameters
    config = {
        "provider": "openai",
        # Missing model and api_key
    }
    result = adapter.configure_embedding(config)
    
    # Verify result
    assert result["status"] == "error"
    assert "Missing required parameters" in result["message"]


def test_get_model_dimensions():
    """Test getting model dimensions."""
    # Mock logger
    mock_logger = MagicMock()
    
    # Create adapter
    adapter = EmbeddingAdapter(logger=mock_logger)
    
    # Test different models
    # These values should match the actual dimensions used in the implementation
    assert adapter.get_model_dimensions("text-embedding-3-small") == 1536
    assert adapter.get_model_dimensions("text-embedding-3-large") == 3072
    assert adapter.get_model_dimensions("text-embedding-ada-002") == 1536
    assert adapter.get_model_dimensions("unknown-model") == 1536  # Default value


@pytest.mark.parametrize("text,expected_chunks", [
    ("Short text", 1),                        # Single chunk for short text
    ("A" * 8000, 2),                          # Two chunks for medium text
    ("A" * 16000, 3)                          # Three chunks for long text
])
def test_split_text_for_embedding(text, expected_chunks):
    """Test text splitting for embedding with various text lengths."""
    # Mock logger
    mock_logger = MagicMock()
    
    # Create adapter
    adapter = EmbeddingAdapter(logger=mock_logger)
    
    # Split text and count chunks
    chunks = adapter.split_text_for_embedding(text)
    
    # Verify number of chunks
    assert len(chunks) == expected_chunks


def test_embed_documents():
    """Test embedding multiple documents."""
    # Mock logger
    mock_logger = MagicMock()
    
    # Create adapter
    adapter = EmbeddingAdapter(logger=mock_logger)
    
    # Mock embedding generator
    mock_embedding_generator = MagicMock()
    mock_embedding_generator.embed_documents.return_value = [
        [0.1, 0.2, 0.3],
        [0.4, 0.5, 0.6]
    ]
    adapter.embedding_generator = mock_embedding_generator
    adapter.embedding_enabled = True
    
    # Embed documents
    documents = ["document 1", "document 2"]
    result = adapter.embed_documents(documents)
    
    # Verify result
    assert len(result) == 2
    assert result[0] == [0.1, 0.2, 0.3]
    assert result[1] == [0.4, 0.5, 0.6]
    mock_embedding_generator.embed_documents.assert_called_once_with(documents)


def test_embed_documents_not_enabled():
    """Test embed_documents when embedding is not enabled."""
    # Mock logger
    mock_logger = MagicMock()
    
    # Create adapter with embedding disabled
    adapter = EmbeddingAdapter(logger=mock_logger)
    adapter.embedding_enabled = False
    
    # Embed documents
    documents = ["document 1", "document 2"]
    result = adapter.embed_documents(documents)
    
    # Verify result is None
    assert result is None
    assert mock_logger.warning.called


def test_embed_documents_error():
    """Test error handling in embed_documents."""
    # Mock logger
    mock_logger = MagicMock()
    
    # Create adapter
    adapter = EmbeddingAdapter(logger=mock_logger)
    
    # Mock embedding generator that raises an exception
    mock_embedding_generator = MagicMock()
    mock_embedding_generator.embed_documents.side_effect = Exception("API error")
    adapter.embedding_generator = mock_embedding_generator
    adapter.embedding_enabled = True
    
    # Embed documents
    documents = ["document 1", "document 2"]
    result = adapter.embed_documents(documents)
    
    # Verify result is None and error was logged
    assert result is None
    assert mock_logger.error.called


def test_save_load_embeddings():
    """Test saving and loading embeddings."""
    # Mock logger
    mock_logger = MagicMock()
    
    # Create adapter
    adapter = EmbeddingAdapter(logger=mock_logger)
    
    # Create temporary file
    import tempfile
    temp_dir = tempfile.mkdtemp()
    temp_file = os.path.join(temp_dir, "embeddings.json")
    
    # Test data
    embeddings = {
        "text1": [0.1, 0.2, 0.3],
        "text2": [0.4, 0.5, 0.6]
    }
    
    try:
        # Test save_embeddings
        with patch.object(adapter, '_ensure_directory_exists') as mock_ensure_dir:
            result = adapter.save_embeddings(embeddings, temp_file)
            assert result is True
            mock_ensure_dir.assert_called_once()
        
        # Verify file exists and contains correct data
        assert os.path.exists(temp_file)
        with open(temp_file, 'r') as f:
            saved_data = json.load(f)
            assert saved_data == embeddings
        
        # Test load_embeddings
        loaded_embeddings = adapter.load_embeddings(temp_file)
        assert loaded_embeddings == embeddings
    
    finally:
        # Clean up
        if os.path.exists(temp_file):
            os.remove(temp_file)
        os.rmdir(temp_dir)


def test_load_embeddings_file_not_found():
    """Test loading embeddings when file doesn't exist."""
    # Mock logger
    mock_logger = MagicMock()
    
    # Create adapter
    adapter = EmbeddingAdapter(logger=mock_logger)
    
    # Try to load from nonexistent file
    result = adapter.load_embeddings("nonexistent_file.json")
    
    # Verify result is None and error was logged
    assert result is None
    assert mock_logger.error.called


def test_ensure_directory_exists():
    """Test directory creation for embedding files."""
    # Mock logger
    mock_logger = MagicMock()
    
    # Create adapter
    adapter = EmbeddingAdapter(logger=mock_logger)
    
    # Test with temporary directory
    import tempfile
    temp_dir = tempfile.mkdtemp()
    test_dir = os.path.join(temp_dir, "test_dir")
    test_file = os.path.join(test_dir, "test_file.json")
    
    try:
        # Ensure directory exists
        adapter._ensure_directory_exists(test_file)
        
        # Verify directory was created
        assert os.path.exists(test_dir)
    
    finally:
        # Clean up
        if os.path.exists(test_dir):
            os.rmdir(test_dir)
        os.rmdir(temp_dir)


@pytest.mark.parametrize("provider,expected_class", [
    ("openai", "OpenAIEmbeddings"),
    ("cohere", None),  # Assuming Cohere is not implemented
    ("invalid", None)  # Invalid provider
])
def test_create_embedding_generator(provider, expected_class):
    """Test creation of embedding generator for different providers."""
    # Mock logger
    mock_logger = MagicMock()
    
    # Create adapter
    adapter = EmbeddingAdapter(logger=mock_logger)
    
    # Mock OpenAIEmbeddings
    with patch('src.graph_memory.embedding_adapter.OpenAIEmbeddings') as mock_openai:
        # Create embedding generator
        generator = adapter._create_embedding_generator(
            provider=provider,
            api_key="test_key",
            model_name="test-model"
        )
        
        # Verify result
        if expected_class == "OpenAIEmbeddings":
            assert generator is not None
            mock_openai.assert_called_once()
        else:
            assert generator is None
            if provider != "invalid":
                assert mock_logger.warning.called


def test_initialize_from_environment():
    """Test initialization from environment variables."""
    # Mock environment variables
    env_vars = {
        "OPENAI_API_KEY": "test_env_key",
        "EMBEDDING_MODEL": "text-embedding-3-small",
        "EMBEDDING_PROVIDER": "openai"
    }
    
    # Mock logger
    mock_logger = MagicMock()
    
    # Create adapter
    adapter = EmbeddingAdapter(logger=mock_logger)
    
    # Mock os.environ and init_embedding_manager
    with patch.dict('os.environ', env_vars), \
         patch.object(adapter, 'init_embedding_manager', return_value=True) as mock_init:
        
        # Initialize from environment
        result = adapter.initialize_from_environment()
        
        # Verify result
        assert result is True
        mock_init.assert_called_once_with(
            api_key="test_env_key",
            model_name="text-embedding-3-small",
            provider="openai"
        )


def test_initialize_from_environment_missing_vars():
    """Test initialization when environment variables are missing."""
    # Mock environment with missing variables
    env_vars = {}
    
    # Mock logger
    mock_logger = MagicMock()
    
    # Create adapter
    adapter = EmbeddingAdapter(logger=mock_logger)
    
    # Mock os.environ
    with patch.dict('os.environ', env_vars, clear=True):
        # Initialize from environment
        result = adapter.initialize_from_environment()
        
        # Verify result
        assert result is False
        assert mock_logger.warning.called


def test_get_embedding_with_caching():
    """Test that get_embedding uses caching when available."""
    # Mock logger
    mock_logger = MagicMock()
    
    # Create adapter
    adapter = EmbeddingAdapter(logger=mock_logger)
    
    # Mock embedding generator
    mock_embedding_generator = MagicMock()
    mock_embedding_generator.embed_query.return_value = [0.1, 0.2, 0.3]
    adapter.embedding_generator = mock_embedding_generator
    adapter.embedding_enabled = True
    
    # Create cache
    adapter.embeddings_cache = {}
    
    # First call - should compute embedding
    result1 = adapter.get_embedding("test text")
    
    # Second call - should use cache
    result2 = adapter.get_embedding("test text")
    
    # Verify results are the same
    assert result1 == result2
    
    # Verify embed_query was called only once
    mock_embedding_generator.embed_query.assert_called_once_with("test text") 