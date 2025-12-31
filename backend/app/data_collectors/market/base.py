"""
Base Market Data Collector
==========================
Abstract base class for market data collectors.
"""

from abc import ABC, abstractmethod
from datetime import datetime
from typing import List, Optional, Dict, Any
from dataclasses import dataclass
from decimal import Decimal
import asyncio
import logging

logger = logging.getLogger(__name__)


@dataclass
class Tick:
    """Tick data structure."""
    symbol: str
    timestamp: datetime
    bid: Decimal
    ask: Decimal
    bid_volume: Decimal = Decimal("0")
    ask_volume: Decimal = Decimal("0")
    
    @property
    def mid(self) -> Decimal:
        return (self.bid + self.ask) / 2
    
    @property
    def spread(self) -> Decimal:
        return self.ask - self.bid


@dataclass
class OHLCV:
    """OHLCV candlestick data structure."""
    symbol: str
    timeframe: str
    timestamp: datetime
    open: Decimal
    high: Decimal
    low: Decimal
    close: Decimal
    volume: Decimal = Decimal("0")
    tick_volume: int = 0


class BaseMarketDataCollector(ABC):
    """Abstract base class for market data collectors."""
    
    def __init__(self, symbol: str = "XAUUSD"):
        self.symbol = symbol
        self.is_connected = False
        self._callbacks: List[callable] = []
    
    @abstractmethod
    async def connect(self) -> bool:
        """Connect to data source."""
        pass
    
    @abstractmethod
    async def disconnect(self) -> None:
        """Disconnect from data source."""
        pass
    
    @abstractmethod
    async def get_current_price(self) -> Optional[Tick]:
        """Get current price tick."""
        pass
    
    @abstractmethod
    async def get_historical_data(
        self,
        timeframe: str,
        start: datetime,
        end: Optional[datetime] = None,
        limit: int = 1000,
    ) -> List[OHLCV]:
        """Get historical OHLCV data."""
        pass
    
    @abstractmethod
    async def subscribe_ticks(self, callback: callable) -> None:
        """Subscribe to real-time tick updates."""
        pass
    
    def add_callback(self, callback: callable) -> None:
        """Add callback for tick updates."""
        self._callbacks.append(callback)
    
    def remove_callback(self, callback: callable) -> None:
        """Remove callback."""
        if callback in self._callbacks:
            self._callbacks.remove(callback)
    
    async def notify_callbacks(self, tick: Tick) -> None:
        """Notify all callbacks with new tick."""
        for callback in self._callbacks:
            try:
                if asyncio.iscoroutinefunction(callback):
                    await callback(tick)
                else:
                    callback(tick)
            except Exception as e:
                logger.error(f"Callback error: {e}")
