import time
import asyncio
import unittest
from unittest.mock import Mock, patch, MagicMock
from src.session_manager import SessionManager

class TestSessionManager(unittest.TestCase):
    """Test cases for the SessionManager class."""
    
    def setUp(self):
        # Create a SessionManager with short timeouts for testing
        self.session_manager = SessionManager(
            inactive_timeout=1,  # 1 second timeout
            cleanup_interval=1  # 1 second cleanup interval
        )
        self.mock_logger = Mock()
        self.session_manager.logger = self.mock_logger
    
    def test_register_client(self):
        """Test registering a new client."""
        mock_manager = Mock()
        self.session_manager.register_client("test-client", mock_manager)
        
        # Check that client was registered
        self.assertIn("test-client", self.session_manager.clients)
        self.assertEqual(self.session_manager.clients["test-client"]["manager"], mock_manager)
        self.assertTrue(self.session_manager.clients["test-client"]["active"])
    
    def test_update_activity(self):
        """Test updating client activity time."""
        mock_manager = Mock()
        self.session_manager.register_client("test-client", mock_manager)
        
        # Store the initial last_activity time
        initial_time = self.session_manager.clients["test-client"]["last_activity"]
        
        # Wait a moment to ensure time difference
        time.sleep(0.1)
        
        # Update activity
        self.session_manager.update_activity("test-client")
        
        # Check that last_activity was updated
        self.assertGreater(
            self.session_manager.clients["test-client"]["last_activity"],
            initial_time
        )
    
    def test_mark_client_inactive(self):
        """Test marking a client as inactive."""
        mock_manager = Mock()
        self.session_manager.register_client("test-client", mock_manager)
        
        # Mark client inactive
        self.session_manager.mark_client_inactive("test-client")
        
        # Check that client was marked inactive
        self.assertFalse(self.session_manager.clients["test-client"]["active"])
    
    def test_cleanup_client(self):
        """Test cleaning up a client."""
        mock_manager = Mock()
        mock_manager.close = Mock()
        
        self.session_manager.register_client("test-client", mock_manager)
        
        # Clean up client
        self.session_manager.cleanup_client("test-client")
        
        # Check that manager.close was called
        mock_manager.close.assert_called_once()
        
        # Check that client was removed from clients dict
        self.assertNotIn("test-client", self.session_manager.clients)
    
    def test_cleanup_inactive_clients(self):
        """Test the cleanup of inactive clients."""
        # Create two mock managers
        mock_manager1 = Mock()
        mock_manager1.close = Mock()
        
        mock_manager2 = Mock()
        mock_manager2.close = Mock()
        
        # Register two clients
        self.session_manager.register_client("active-client", mock_manager1)
        self.session_manager.register_client("inactive-client", mock_manager2)
        
        # Mark one client as inactive
        self.session_manager.mark_client_inactive("inactive-client")
        
        # Run the cleanup
        self.session_manager._cleanup_inactive_clients()
        
        # Check that only the inactive client was cleaned up
        self.assertIn("active-client", self.session_manager.clients)
        self.assertNotIn("inactive-client", self.session_manager.clients)
        mock_manager1.close.assert_not_called()
        mock_manager2.close.assert_called_once()
    
    def test_cleanup_timeout_clients(self):
        """Test the cleanup of clients that have timed out."""
        # Create a mock manager
        mock_manager = Mock()
        mock_manager.close = Mock()
        
        # Register a client
        self.session_manager.register_client("timeout-client", mock_manager)
        
        # Manipulate the last_activity time to make it appear old
        self.session_manager.clients["timeout-client"]["last_activity"] = time.time() - 2
        
        # Run the cleanup
        self.session_manager._cleanup_inactive_clients()
        
        # Check that the timed-out client was cleaned up
        self.assertNotIn("timeout-client", self.session_manager.clients)
        mock_manager.close.assert_called_once()
    
    @patch('asyncio.create_task')
    async def test_start_cleanup_task(self, mock_create_task):
        """Test starting the cleanup task."""
        # Start the cleanup task
        await self.session_manager.start_cleanup_task()
        
        # Check that create_task was called
        mock_create_task.assert_called_once()
        self.assertTrue(self.session_manager.running)
    
    async def test_stop_cleanup_task(self):
        """Test stopping the cleanup task."""
        # Create a mock task
        mock_task = MagicMock()
        self.session_manager.cleanup_task = mock_task
        self.session_manager.running = True
        
        # Stop the cleanup task
        await self.session_manager.stop_cleanup_task()
        
        # Check that task was cancelled
        mock_task.cancel.assert_called_once()
        self.assertFalse(self.session_manager.running)
    
    @patch('asyncio.sleep')
    async def test_cleanup_loop(self, mock_sleep):
        """Test the cleanup loop."""
        # Mock the _cleanup_inactive_clients method
        self.session_manager._cleanup_inactive_clients = Mock()
        
        # Set running to True, then to False after first loop iteration
        self.session_manager.running = True
        mock_sleep.side_effect = lambda _: setattr(self.session_manager, 'running', False)
        
        # Run the cleanup loop
        await self.session_manager._cleanup_loop()
        
        # Check that _cleanup_inactive_clients was called
        self.session_manager._cleanup_inactive_clients.assert_called_once()
        mock_sleep.assert_called_once_with(1)

if __name__ == '__main__':
    unittest.main() 