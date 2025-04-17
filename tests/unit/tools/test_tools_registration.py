"""
Unit tests for tools registration functionality.
"""

import pytest
from unittest.mock import MagicMock, patch

from src.tools import register_all_tools, register_consolidated_tools, register_essential_tools

class TestToolsRegistration:
    """Test suite for tools registration functionality."""
    
    def test_register_all_tools(self, mock_server):
        """Test registering all tools."""
        # Create a mock client_manager getter function
        mock_client_manager_getter = MagicMock()
        
        # Call the function
        register_all_tools(mock_server, mock_client_manager_getter)
        
        # Verify that server.tool was called multiple times
        assert mock_server.tool.call_count > 0
    
    def test_register_consolidated_tools(self, mock_server):
        """Test registering consolidated tools."""
        # Create a mock client_manager getter function
        mock_client_manager_getter = MagicMock()
        
        # Call the function
        register_consolidated_tools(mock_server, mock_client_manager_getter)
        
        # Verify that server.tool was called
        assert mock_server.tool.call_count > 0
    
    def test_register_essential_tools(self, mock_server):
        """Test registering essential tools."""
        # Create a mock client_manager getter function
        mock_client_manager_getter = MagicMock()
        
        # Call the function
        register_essential_tools(mock_server, mock_client_manager_getter)
        
        # Verify that server.tool was called only for essential tools
        # For this test, we're just checking that it's called less frequently than register_all_tools
        essential_count = mock_server.tool.call_count
        
        # Reset the mock and call register_all_tools
        mock_server.tool.reset_mock()
        register_all_tools(mock_server, mock_client_manager_getter)
        all_count = mock_server.tool.call_count
        
        # Essential tools should be a subset of all tools
        assert essential_count < all_count
    
    @patch('src.tools.register_core_tools')
    @patch('src.tools.register_registry_tools')
    def test_core_memory_tools_registration(self, mock_register_registry, mock_register_core, mock_server):
        """Test registration of core memory tools."""
        # Create a mock client_manager getter function
        mock_client_manager_getter = MagicMock()
        
        # Call register_all_tools
        register_all_tools(mock_server, mock_client_manager_getter)
        
        # Verify that register_core_tools was called
        mock_register_core.assert_called_once_with(mock_server, mock_client_manager_getter)
    
    @patch('src.tools.register_lesson_tools')
    @patch('src.tools.register_registry_tools')
    def test_lesson_memory_tools_registration(self, mock_register_registry, mock_register_lesson, mock_server):
        """Test registration of lesson memory tools."""
        # Create a mock client_manager getter function
        mock_client_manager_getter = MagicMock()
        
        # Call register_all_tools
        register_all_tools(mock_server, mock_client_manager_getter)
        
        # Verify that register_lesson_tools was called
        mock_register_lesson.assert_called_once_with(mock_server, mock_client_manager_getter)
    
    @patch('src.tools.register_project_tools')
    @patch('src.tools.register_registry_tools')
    def test_project_memory_tools_registration(self, mock_register_registry, mock_register_project, mock_server):
        """Test registration of project memory tools."""
        # Create a mock client_manager getter function
        mock_client_manager_getter = MagicMock()
        
        # Call register_all_tools
        register_all_tools(mock_server, mock_client_manager_getter)
        
        # Verify that register_project_tools was called
        mock_register_project.assert_called_once_with(mock_server, mock_client_manager_getter)
    
    @patch('src.tools.register_config_tools')
    @patch('src.tools.register_registry_tools')
    def test_config_tools_registration(self, mock_register_registry, mock_register_config, mock_server):
        """Test registration of configuration tools."""
        # Create a mock client_manager getter function
        mock_client_manager_getter = MagicMock()
        
        # Call register_all_tools
        register_all_tools(mock_server, mock_client_manager_getter)
        
        # Verify that register_config_tools was called
        mock_register_config.assert_called_once_with(mock_server, mock_client_manager_getter) 