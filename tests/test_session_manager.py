"""
Unit tests for the SessionManager.
"""

import pytest
import asyncio
import time
from datetime import datetime, timedelta
from unittest.mock import MagicMock

from src.session_manager import SessionManager

class TestSessionManager:
    """Tests for the SessionManager class."""
    
    @pytest.fixture
    def session_manager(self):
        """Create a SessionManager for testing."""
        manager = SessionManager(inactive_timeout=60, cleanup_interval=30)
        yield manager
        # Clean up
        asyncio.run(manager.stop_cleanup_task())
    
    def test_initialization(self):
        """Test that the SessionManager initializes correctly."""
        manager = SessionManager(inactive_timeout=300, cleanup_interval=60)
        
        # Verify manager properties
        assert hasattr(manager, 'clients')  # Updated from _sessions to clients
        assert manager.inactive_timeout == 300
        assert manager.cleanup_interval == 60
        # Check cleanup task
        assert manager.cleanup_task is None
    
    def test_update_activity(self, session_manager):
        """Test updating session activity."""
        client_id = "test_client"
        
        # Need to register a client first
        mock_manager = MagicMock()
        session_manager.register_client(client_id, mock_manager)
        
        # Update activity for the client
        session_manager.update_activity(client_id)
        
        # Verify session exists
        assert client_id in session_manager.clients
        assert "last_activity" in session_manager.clients[client_id]
    
    def test_register_client(self, session_manager):
        """Test checking if a client is registered properly."""
        client_id = "test_client"
        mock_manager = MagicMock()
        
        # Register client
        session_manager.register_client(client_id, mock_manager)
        
        # Verify client exists
        assert client_id in session_manager.clients
        assert session_manager.clients[client_id]["active"] == True
        assert "last_activity" in session_manager.clients[client_id]
        assert session_manager.clients[client_id]["manager"] == mock_manager
    
    def test_mark_client_inactive(self, session_manager):
        """Test marking a client as inactive."""
        client_id = "test_client"
        mock_manager = MagicMock()
        
        # Register client
        session_manager.register_client(client_id, mock_manager)
        assert session_manager.clients[client_id]["active"] == True
        
        # Mark client as inactive
        session_manager.mark_client_inactive(client_id)
        
        # Verify client is marked inactive
        assert session_manager.clients[client_id]["active"] == False
    
    def test_cleanup_client(self, session_manager):
        """Test cleaning up a client."""
        client_id = "test_client"
        mock_manager = MagicMock()
        
        # Register client
        session_manager.register_client(client_id, mock_manager)
        assert client_id in session_manager.clients
        
        # Clean up client
        session_manager.cleanup_client(client_id)
        
        # Verify client is removed
        assert client_id not in session_manager.clients
        # Verify manager's close method was called
        mock_manager.close.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_cleanup_inactive_clients(self):
        """Test that cleanup removes inactive clients."""
        # Create a session manager with a very short timeout
        manager = SessionManager(inactive_timeout=1, cleanup_interval=1)
        
        # Add some clients
        active_client_id = "active_client"
        inactive_client_id = "inactive_client"
        active_mock = MagicMock()
        inactive_mock = MagicMock()
        
        manager.register_client(active_client_id, active_mock)
        manager.register_client(inactive_client_id, inactive_mock)
        
        # Mark inactive client as inactive
        manager.mark_client_inactive(inactive_client_id)
        
        # Run cleanup method
        manager._cleanup_inactive_clients()
        
        # Verify inactive client is removed but active client remains
        assert active_client_id in manager.clients
        assert inactive_client_id not in manager.clients
        
        # Clean up
        await manager.stop_cleanup_task()
    
    @pytest.mark.asyncio
    async def test_start_cleanup_task(self):
        """Test starting the cleanup task."""
        # Create a session manager
        manager = SessionManager(inactive_timeout=1, cleanup_interval=1)
        
        # Start cleanup task
        await manager.start_cleanup_task()
        
        # Verify task is running
        assert manager.cleanup_task is not None
        assert not manager.cleanup_task.done()
        assert manager.running == True
        
        # Clean up
        await manager.stop_cleanup_task()
    
    @pytest.mark.asyncio
    async def test_stop_cleanup_task(self):
        """Test stopping the cleanup task."""
        # Create a session manager
        manager = SessionManager(inactive_timeout=10, cleanup_interval=1)
        
        # Start cleanup task
        await manager.start_cleanup_task()
        
        # Verify task is running
        assert manager.cleanup_task is not None
        assert manager.running == True
        
        # Stop cleanup task
        await manager.stop_cleanup_task()
        
        # Verify task is stopped
        assert manager.running == False 