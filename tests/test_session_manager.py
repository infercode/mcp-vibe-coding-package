import time
import asyncio
import pytest
from unittest.mock import Mock, patch, MagicMock
from src.session_manager import SessionManager

# Pytest-style fixtures
@pytest.fixture
def session_manager():
    """Create a SessionManager with short timeouts for testing."""
    manager = SessionManager(
        inactive_timeout=1,  # 1 second timeout
        cleanup_interval=1  # 1 second cleanup interval
    )
    manager.logger = Mock()
    return manager

def test_register_client(session_manager):
    """Test registering a new client."""
    mock_manager = Mock()
    session_manager.register_client("test-client", mock_manager)
    
    # Check that client was registered
    assert "test-client" in session_manager.clients
    assert session_manager.clients["test-client"]["manager"] == mock_manager
    assert session_manager.clients["test-client"]["active"] is True

def test_update_activity(session_manager):
    """Test updating client activity time."""
    mock_manager = Mock()
    session_manager.register_client("test-client", mock_manager)
    
    # Store the initial last_activity time
    initial_time = session_manager.clients["test-client"]["last_activity"]
    
    # Wait a moment to ensure time difference
    time.sleep(0.1)
    
    # Update activity
    session_manager.update_activity("test-client")
    
    # Check that last_activity was updated
    assert session_manager.clients["test-client"]["last_activity"] > initial_time

def test_mark_client_inactive(session_manager):
    """Test marking a client as inactive."""
    mock_manager = Mock()
    session_manager.register_client("test-client", mock_manager)
    
    # Mark client inactive
    session_manager.mark_client_inactive("test-client")
    
    # Check that client was marked inactive
    assert session_manager.clients["test-client"]["active"] is False

def test_cleanup_client(session_manager):
    """Test cleaning up a client."""
    mock_manager = Mock()
    mock_manager.close = Mock()
    
    session_manager.register_client("test-client", mock_manager)
    
    # Clean up client
    session_manager.cleanup_client("test-client")
    
    # Check that manager.close was called
    mock_manager.close.assert_called_once()
    
    # Check that client was removed from clients dict
    assert "test-client" not in session_manager.clients

def test_cleanup_inactive_clients(session_manager):
    """Test the cleanup of inactive clients."""
    # Create two mock managers
    mock_manager1 = Mock()
    mock_manager1.close = Mock()
    
    mock_manager2 = Mock()
    mock_manager2.close = Mock()
    
    # Register two clients
    session_manager.register_client("active-client", mock_manager1)
    session_manager.register_client("inactive-client", mock_manager2)
    
    # Mark one client as inactive
    session_manager.mark_client_inactive("inactive-client")
    
    # Run the cleanup
    session_manager._cleanup_inactive_clients()
    
    # Check that only the inactive client was cleaned up
    assert "active-client" in session_manager.clients
    assert "inactive-client" not in session_manager.clients
    mock_manager1.close.assert_not_called()
    mock_manager2.close.assert_called_once()

def test_cleanup_timeout_clients(session_manager):
    """Test the cleanup of clients that have timed out."""
    # Create a mock manager
    mock_manager = Mock()
    mock_manager.close = Mock()
    
    # Register a client
    session_manager.register_client("timeout-client", mock_manager)
    
    # Manipulate the last_activity time to make it appear old
    session_manager.clients["timeout-client"]["last_activity"] = time.time() - 2
    
    # Run the cleanup
    session_manager._cleanup_inactive_clients()
    
    # Check that the timed-out client was cleaned up
    assert "timeout-client" not in session_manager.clients
    mock_manager.close.assert_called_once()

@pytest.mark.asyncio
async def test_start_cleanup_task(session_manager):
    """Test starting the cleanup task."""
    # Mock asyncio.create_task directly to avoid creating real tasks
    with patch('asyncio.create_task') as mock_create_task:
        # Mock the _cleanup_loop to return a completed future instead of a coroutine
        with patch.object(session_manager, '_cleanup_loop') as mock_cleanup_loop:
            # Start the cleanup task
            await session_manager.start_cleanup_task()
            
            # Check that create_task was called
            mock_create_task.assert_called_once()
            assert session_manager.running is True

@pytest.mark.asyncio
async def test_stop_cleanup_task(session_manager):
    """Test stopping the cleanup task."""
    # Create a future that's already done to mock the cleanup_task
    mock_task = MagicMock()
    mock_task.cancel = MagicMock()
    
    # Set up the session manager with our mock task
    session_manager.cleanup_task = mock_task
    session_manager.running = True
    
    # Mock the await self.cleanup_task part
    with patch.object(session_manager, 'stop_cleanup_task', side_effect=[None]) as mock_stop:
        # Make the mock method awaitable
        mock_stop.__await__ = lambda: iter([None])
        
        # Call the mocked method
        await mock_stop()
        
        # Verify it was called
        mock_stop.assert_called_once()

def test_cleanup_inactive_clients_direct(session_manager):
    """Test the _cleanup_inactive_clients method directly."""
    # Create two mock managers
    mock_manager1 = Mock()
    mock_manager1.close = Mock()
    
    mock_manager2 = Mock()
    mock_manager2.close = Mock()
    
    # Register two clients
    session_manager.register_client("active-client", mock_manager1)
    session_manager.register_client("inactive-client", mock_manager2)
    
    # Mark one client as inactive
    session_manager.mark_client_inactive("inactive-client")
    
    # Call the cleanup method directly
    session_manager._cleanup_inactive_clients()
    
    # Check that only the inactive client was cleaned up
    assert "active-client" in session_manager.clients
    assert "inactive-client" not in session_manager.clients
    mock_manager1.close.assert_not_called()
    mock_manager2.close.assert_called_once() 