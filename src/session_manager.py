import time
import asyncio
from typing import Dict, Any, Optional, Callable
from src.logger import get_logger

class SessionManager:
    """
    Manages client sessions and handles cleanup of inactive/disconnected clients.
    
    This class tracks active client sessions, their last activity time, and
    handles cleanup of resources when clients disconnect or become inactive.
    """
    
    def __init__(self, 
                 inactive_timeout: int = 3600,  # 1 hour by default
                 cleanup_interval: int = 300):  # 5 minutes by default
        """
        Initialize the session manager.
        
        Args:
            inactive_timeout: Seconds of inactivity before a session is considered stale
            cleanup_interval: Interval in seconds between cleanup runs
        """
        self.logger = get_logger()
        self.clients: Dict[str, Dict[str, Any]] = {}
        self.inactive_timeout = inactive_timeout
        self.cleanup_interval = cleanup_interval
        self.cleanup_task = None
        self.running = False
    
    def register_client(self, client_id: str, manager: Any) -> None:
        """
        Register a new client with its manager instance.
        
        Args:
            client_id: The unique identifier for the client
            manager: The manager instance (e.g., GraphMemoryManager) for the client
        """
        self.clients[client_id] = {
            "manager": manager,
            "last_activity": time.time(),
            "active": True
        }
        self.logger.info(f"Registered client: {client_id}")
    
    def update_activity(self, client_id: str) -> None:
        """
        Update the last activity time for a client.
        
        Args:
            client_id: The client identifier to update
        """
        if client_id in self.clients:
            self.clients[client_id]["last_activity"] = time.time()
            
    def mark_client_inactive(self, client_id: str) -> None:
        """
        Mark a client as inactive (disconnected).
        
        Args:
            client_id: The client identifier to mark as inactive
        """
        if client_id in self.clients:
            self.clients[client_id]["active"] = False
            self.logger.info(f"Marked client as inactive: {client_id}")
    
    def cleanup_client(self, client_id: str) -> None:
        """
        Clean up resources for a client and remove it from tracking.
        
        Args:
            client_id: The client identifier to clean up
        """
        if client_id in self.clients:
            try:
                # Get the manager to close connection
                manager = self.clients[client_id]["manager"]
                if hasattr(manager, "close"):
                    manager.close()
                    self.logger.info(f"Closed manager connection for client: {client_id}")
                    
                # Remove the client from tracking
                del self.clients[client_id]
                self.logger.info(f"Cleaned up client: {client_id}")
            except Exception as e:
                self.logger.error(f"Error cleaning up client {client_id}: {str(e)}")
    
    async def start_cleanup_task(self) -> None:
        """Start the periodic cleanup task."""
        self.running = True
        self.cleanup_task = asyncio.create_task(self._cleanup_loop())
        self.logger.info("Started session cleanup task")
    
    async def stop_cleanup_task(self) -> None:
        """Stop the periodic cleanup task."""
        self.running = False
        if self.cleanup_task:
            self.cleanup_task.cancel()
            try:
                await self.cleanup_task
            except asyncio.CancelledError:
                pass
        self.logger.info("Stopped session cleanup task")
    
    async def _cleanup_loop(self) -> None:
        """Periodically check for and clean up inactive clients."""
        while self.running:
            try:
                self._cleanup_inactive_clients()
                await asyncio.sleep(self.cleanup_interval)
            except asyncio.CancelledError:
                break
            except Exception as e:
                self.logger.error(f"Error in cleanup loop: {str(e)}")
                await asyncio.sleep(self.cleanup_interval)
    
    def _cleanup_inactive_clients(self) -> None:
        """Identify and clean up inactive clients."""
        current_time = time.time()
        clients_to_cleanup = []
        
        # First identify clients that need cleanup
        for client_id, info in self.clients.items():
            # Clean up if marked inactive or if inactive for too long
            if not info["active"] or (current_time - info["last_activity"]) > self.inactive_timeout:
                clients_to_cleanup.append(client_id)
        
        # Then clean them up
        for client_id in clients_to_cleanup:
            self.cleanup_client(client_id) 