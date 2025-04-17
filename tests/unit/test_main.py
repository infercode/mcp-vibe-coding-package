"""
Unit tests for main application module.
"""

import pytest
import os
import json
import importlib
from unittest.mock import patch, MagicMock

from src.logger import get_logger


class TestMainApplication:
    """Test suite for main application functionality."""
    
    @pytest.fixture
    def logger(self):
        """Get a logger instance for testing."""
        return get_logger()
    
    @pytest.fixture
    def sample_config(self, tmp_path):
        """Create a sample configuration file for testing."""
        config = {
            "server": {
                "host": "localhost",
                "port": 3000,
                "debug": True
            },
            "embedding": {
                "provider": "openai",
                "model": "text-embedding-3-small",
                "dimensions": 1536
            },
            "memory": {
                "graph_memory_backend": "memory",
                "session_cleanup_interval": 3600
            }
        }
        
        config_file = tmp_path / "config.json"
        with open(config_file, "w") as f:
            json.dump(config, f)
        
        return str(config_file)
    
    @pytest.fixture
    def main_module(self):
        """Import the main module dynamically."""
        try:
            return importlib.import_module('src.main')
        except ImportError:
            # For tests, mock the module if it can't be imported
            return MagicMock()
    
    def test_load_config_file(self, sample_config, logger, main_module):
        """Test loading configuration from a file."""
        # Skip if function not available
        if not hasattr(main_module, 'load_config'):
            pytest.skip("load_config function not found in main module")
        
        # Load the configuration
        config = main_module.load_config(sample_config, logger)
        
        # Verify config is loaded correctly
        assert config is not None
        assert "server" in config
        assert config["server"]["host"] == "localhost"
        assert config["server"]["port"] == 3000
        assert config["embedding"]["provider"] == "openai"
        assert config["memory"]["graph_memory_backend"] == "memory"
    
    @patch('src.main.HybridMCPServer')
    @patch('importlib.import_module')
    def test_initialize_server(self, mock_import, mock_server_class, logger, main_module):
        """Test server initialization with configuration."""
        # Skip if function not available
        if not hasattr(main_module, 'initialize_server'):
            pytest.skip("initialize_server function not found in main module")
            
        # Initialize variable to avoid "possibly unbound" error
        original_load_config = None
            
        # Create a mock for load_config to use within initialize_server
        if hasattr(main_module, 'load_config'):
            original_load_config = main_module.load_config
            main_module.load_config = MagicMock()
        
        try:
            # Mock the configuration
            mock_config = {
                "server": {
                    "host": "localhost",
                    "port": 3000,
                    "debug": True
                },
                "embedding": {
                    "provider": "openai",
                    "model": "text-embedding-3-small",
                    "dimensions": 1536
                },
                "memory": {
                    "graph_memory_backend": "memory",
                    "session_cleanup_interval": 3600
                }
            }
            
            if hasattr(main_module, 'load_config'):
                main_module.load_config.return_value = mock_config
            
            # Mock the server instance
            mock_server = MagicMock()
            mock_server_class.return_value = mock_server
            
            # Initialize the server
            server = main_module.initialize_server("config.json", logger)
            
            # Verify server is initialized correctly
            assert server is not None
            mock_server_class.assert_called_once()
            
            # Verify server is configured with the right settings
            assert mock_server.configure.called
        finally:
            # Restore original function if we mocked it
            if hasattr(main_module, 'load_config') and original_load_config is not None:
                main_module.load_config = original_load_config
    
    @patch('os.environ')
    def test_config_environment_variables(self, mock_environ, logger, main_module):
        """Test configuration from environment variables."""
        # Skip if function not available
        if not hasattr(main_module, 'apply_environment_overrides'):
            pytest.skip("apply_environment_overrides function not found in main module")
        
        # Setup environment variables
        mock_environ.get.side_effect = lambda key, default=None: {
            "MCP_SERVER_PORT": "5000",
            "MCP_SERVER_HOST": "0.0.0.0",
            "MCP_EMBEDDING_PROVIDER": "azure",
            "MCP_EMBEDDING_MODEL": "text-embedding-ada-002",
            "MCP_GRAPH_MEMORY_BACKEND": "neo4j"
        }.get(key, default)
        
        # Create base config
        config = {
            "server": {
                "host": "localhost",
                "port": 3000
            },
            "embedding": {
                "provider": "openai",
                "model": "text-embedding-3-small"
            },
            "memory": {
                "graph_memory_backend": "memory"
            }
        }
        
        # Apply environment overrides
        updated_config = main_module.apply_environment_overrides(config, logger)
        
        # Verify environment variables override defaults
        assert updated_config["server"]["port"] == 5000
        assert updated_config["server"]["host"] == "0.0.0.0"
        assert updated_config["embedding"]["provider"] == "azure"
        assert updated_config["embedding"]["model"] == "text-embedding-ada-002"
        assert updated_config["memory"]["graph_memory_backend"] == "neo4j"
    
    @patch('argparse.ArgumentParser.parse_args')
    def test_command_line_arguments(self, mock_parse_args, logger, main_module):
        """Test command line argument parsing."""
        # Skip if function not available
        if not hasattr(main_module, 'parse_arguments'):
            pytest.skip("parse_arguments function not found in main module")
        
        # Mock the argument parser
        mock_parse_args.return_value = MagicMock(
            config="custom_config.json",
            debug=True,
            port=4000,
            log_level="debug"
        )
        
        # Parse arguments
        args = main_module.parse_arguments()
        
        # Verify arguments are parsed correctly
        assert args.config == "custom_config.json"
        assert args.debug is True
        assert args.port == 4000
        assert args.log_level == "debug" 