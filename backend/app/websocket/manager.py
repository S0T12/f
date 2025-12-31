"""
WebSocket Connection Manager
============================
Manage WebSocket connections for real-time updates.
"""

import logging
from typing import Dict, List, Set, Optional
from datetime import datetime
import asyncio
import json

from fastapi import WebSocket
from starlette.websockets import WebSocketState

logger = logging.getLogger(__name__)


class ConnectionManager:
    """Manage WebSocket connections and subscriptions."""
    
    def __init__(self):
        self.active_connections: Dict[str, WebSocket] = {}
        self.subscriptions: Dict[str, Set[str]] = {}  # channel -> set of connection_ids
        self.user_connections: Dict[int, Set[str]] = {}  # user_id -> set of connection_ids
        self._lock = asyncio.Lock()
    
    async def connect(
        self,
        websocket: WebSocket,
        connection_id: str,
        user_id: Optional[int] = None,
    ) -> None:
        """Accept a new WebSocket connection."""
        await websocket.accept()
        
        async with self._lock:
            self.active_connections[connection_id] = websocket
            
            if user_id:
                if user_id not in self.user_connections:
                    self.user_connections[user_id] = set()
                self.user_connections[user_id].add(connection_id)
        
        logger.info(f"WebSocket connected: {connection_id}")
    
    async def disconnect(self, connection_id: str) -> None:
        """Remove a WebSocket connection."""
        async with self._lock:
            # Remove from active connections
            if connection_id in self.active_connections:
                del self.active_connections[connection_id]
            
            # Remove from all subscriptions
            for channel in list(self.subscriptions.keys()):
                if connection_id in self.subscriptions[channel]:
                    self.subscriptions[channel].discard(connection_id)
                    if not self.subscriptions[channel]:
                        del self.subscriptions[channel]
            
            # Remove from user connections
            for user_id in list(self.user_connections.keys()):
                if connection_id in self.user_connections[user_id]:
                    self.user_connections[user_id].discard(connection_id)
                    if not self.user_connections[user_id]:
                        del self.user_connections[user_id]
        
        logger.info(f"WebSocket disconnected: {connection_id}")
    
    async def subscribe(self, connection_id: str, channel: str) -> None:
        """Subscribe a connection to a channel."""
        async with self._lock:
            if channel not in self.subscriptions:
                self.subscriptions[channel] = set()
            self.subscriptions[channel].add(connection_id)
        
        logger.debug(f"Connection {connection_id} subscribed to {channel}")
    
    async def unsubscribe(self, connection_id: str, channel: str) -> None:
        """Unsubscribe a connection from a channel."""
        async with self._lock:
            if channel in self.subscriptions:
                self.subscriptions[channel].discard(connection_id)
    
    async def send_personal_message(
        self,
        message: dict,
        connection_id: str,
    ) -> bool:
        """Send message to specific connection."""
        if connection_id not in self.active_connections:
            return False
        
        websocket = self.active_connections[connection_id]
        
        try:
            if websocket.client_state == WebSocketState.CONNECTED:
                await websocket.send_json(message)
                return True
        except Exception as e:
            logger.error(f"Error sending to {connection_id}: {e}")
            await self.disconnect(connection_id)
        
        return False
    
    async def send_to_user(self, message: dict, user_id: int) -> int:
        """Send message to all connections of a user."""
        if user_id not in self.user_connections:
            return 0
        
        sent = 0
        for connection_id in list(self.user_connections[user_id]):
            if await self.send_personal_message(message, connection_id):
                sent += 1
        
        return sent
    
    async def broadcast(self, message: dict) -> int:
        """Broadcast message to all connections."""
        sent = 0
        for connection_id in list(self.active_connections.keys()):
            if await self.send_personal_message(message, connection_id):
                sent += 1
        
        return sent
    
    async def broadcast_to_channel(self, channel: str, message: dict) -> int:
        """Broadcast message to all subscribers of a channel."""
        if channel not in self.subscriptions:
            return 0
        
        sent = 0
        for connection_id in list(self.subscriptions[channel]):
            if await self.send_personal_message(message, connection_id):
                sent += 1
        
        return sent
    
    def get_connection_count(self) -> int:
        """Get total number of active connections."""
        return len(self.active_connections)
    
    def get_channel_subscriber_count(self, channel: str) -> int:
        """Get number of subscribers to a channel."""
        return len(self.subscriptions.get(channel, set()))


# Global connection manager
manager = ConnectionManager()
