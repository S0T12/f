"""
Redis PubSub Listener
=====================
Listen to Redis channels and broadcast to WebSocket clients.
"""

import logging
import asyncio
import json
from typing import Optional

from app.websocket.manager import manager
from app.services.redis_service import redis_service

logger = logging.getLogger(__name__)


class RedisPubSubListener:
    """Listen to Redis PubSub and forward to WebSocket clients."""
    
    def __init__(self):
        self.pubsub = None
        self.running = False
        self.channels = [
            "ticks:xauusd",
            "signals:xauusd",
            "predictions:xauusd",
            "alerts",
            "trades:*",
        ]
    
    async def start(self):
        """Start listening to Redis PubSub."""
        if self.running:
            return
        
        self.running = True
        self.pubsub = redis_service.redis.pubsub()
        
        # Subscribe to channels
        for channel in self.channels:
            if "*" in channel:
                self.pubsub.psubscribe(channel)
            else:
                self.pubsub.subscribe(channel)
        
        logger.info(f"Started Redis PubSub listener for channels: {self.channels}")
        
        # Start message loop
        asyncio.create_task(self._message_loop())
    
    async def stop(self):
        """Stop listening."""
        self.running = False
        if self.pubsub:
            self.pubsub.unsubscribe()
            self.pubsub.punsubscribe()
            self.pubsub.close()
    
    async def _message_loop(self):
        """Process incoming Redis messages."""
        while self.running:
            try:
                message = self.pubsub.get_message(ignore_subscribe_messages=True)
                
                if message and message["type"] in ("message", "pmessage"):
                    await self._handle_message(message)
                
                await asyncio.sleep(0.01)  # Small delay to prevent CPU spinning
            
            except Exception as e:
                logger.error(f"Redis PubSub error: {e}")
                await asyncio.sleep(1)
    
    async def _handle_message(self, message: dict):
        """Handle incoming Redis message."""
        channel = message.get("channel", b"").decode()
        pattern = message.get("pattern", b"").decode() if message.get("pattern") else None
        
        try:
            data = json.loads(message["data"])
        except (json.JSONDecodeError, TypeError):
            data = message["data"]
            if isinstance(data, bytes):
                data = data.decode()
        
        # Map Redis channels to WebSocket channels
        if channel == "ticks:xauusd":
            ws_message = {"type": "tick", "data": data}
            await manager.broadcast_to_channel("prices", ws_message)
        
        elif channel == "signals:xauusd":
            ws_message = {"type": "signal", "data": data}
            await manager.broadcast_to_channel("signals", ws_message)
        
        elif channel == "predictions:xauusd":
            ws_message = {"type": "prediction", "data": data}
            await manager.broadcast_to_channel("predictions", ws_message)
        
        elif channel == "alerts":
            ws_message = {"type": "alert", "data": data}
            await manager.broadcast(ws_message)
        
        elif pattern == "trades:*":
            # Extract user_id from channel (trades:123)
            try:
                user_id = int(channel.split(":")[1])
                ws_message = {"type": "trade_update", "data": data}
                await manager.send_to_user(ws_message, user_id)
            except (ValueError, IndexError):
                pass


# Global listener instance
pubsub_listener = RedisPubSubListener()
