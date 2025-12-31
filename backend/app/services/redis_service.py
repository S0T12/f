"""
Redis Service
=============
Redis connection management and caching utilities.
"""

import json
from typing import Any, Optional
import redis.asyncio as redis

from app.config import settings


class RedisManager:
    """Redis connection manager."""
    
    def __init__(self):
        self.client: Optional[redis.Redis] = None
    
    async def connect(self) -> None:
        """Connect to Redis."""
        self.client = redis.from_url(
            settings.REDIS_URL,
            encoding="utf-8",
            decode_responses=True,
        )
    
    async def disconnect(self) -> None:
        """Disconnect from Redis."""
        if self.client:
            await self.client.close()
    
    async def get(self, key: str) -> Optional[str]:
        """Get value from cache."""
        if self.client:
            return await self.client.get(key)
        return None
    
    async def set(
        self,
        key: str,
        value: Any,
        ttl: int = None,
    ) -> None:
        """Set value in cache."""
        if self.client:
            ttl = ttl or settings.REDIS_CACHE_TTL
            if isinstance(value, (dict, list)):
                value = json.dumps(value)
            await self.client.setex(key, ttl, value)
    
    async def delete(self, key: str) -> None:
        """Delete key from cache."""
        if self.client:
            await self.client.delete(key)
    
    async def get_json(self, key: str) -> Optional[Any]:
        """Get JSON value from cache."""
        value = await self.get(key)
        if value:
            return json.loads(value)
        return None
    
    async def publish(self, channel: str, message: Any) -> None:
        """Publish message to channel."""
        if self.client:
            if isinstance(message, (dict, list)):
                message = json.dumps(message)
            await self.client.publish(channel, message)
    
    async def subscribe(self, channel: str):
        """Subscribe to channel."""
        if self.client:
            pubsub = self.client.pubsub()
            await pubsub.subscribe(channel)
            return pubsub


# Global Redis manager instance
redis_manager = RedisManager()
