"""
OANDA Data Collector
====================
Market data collector using OANDA's REST and Streaming APIs.
"""

import asyncio
import logging
from datetime import datetime, timedelta
from decimal import Decimal
from typing import List, Optional, Dict, Any
import httpx

from app.config import settings
from app.data_collectors.market.base import BaseMarketDataCollector, Tick, OHLCV

logger = logging.getLogger(__name__)


class OandaCollector(BaseMarketDataCollector):
    """OANDA market data collector."""
    
    TIMEFRAME_MAP = {
        "1m": "M1", "5m": "M5", "15m": "M15", "30m": "M30",
        "1h": "H1", "4h": "H4", "1d": "D", "1w": "W",
    }
    
    def __init__(self, symbol: str = "XAU_USD"):
        super().__init__(symbol)
        self.api_key = settings.OANDA_API_KEY
        self.account_id = settings.OANDA_ACCOUNT_ID
        self.environment = settings.OANDA_ENVIRONMENT
        
        if self.environment == "live":
            self.base_url = "https://api-fxtrade.oanda.com"
            self.stream_url = "https://stream-fxtrade.oanda.com"
        else:
            self.base_url = "https://api-fxpractice.oanda.com"
            self.stream_url = "https://stream-fxpractice.oanda.com"
        
        self._client: Optional[httpx.AsyncClient] = None
        self._stream_task: Optional[asyncio.Task] = None
    
    @property
    def headers(self) -> Dict[str, str]:
        return {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
    
    async def connect(self) -> bool:
        """Connect to OANDA API."""
        if not self.api_key or not self.account_id:
            logger.warning("OANDA credentials not configured")
            return False
        
        try:
            self._client = httpx.AsyncClient(
                base_url=self.base_url,
                headers=self.headers,
                timeout=30.0,
            )
            
            # Test connection
            response = await self._client.get(f"/v3/accounts/{self.account_id}")
            if response.status_code == 200:
                self.is_connected = True
                logger.info("Connected to OANDA API")
                return True
            else:
                logger.error(f"OANDA connection failed: {response.text}")
                return False
                
        except Exception as e:
            logger.error(f"OANDA connection error: {e}")
            return False
    
    async def disconnect(self) -> None:
        """Disconnect from OANDA."""
        if self._stream_task:
            self._stream_task.cancel()
            self._stream_task = None
        
        if self._client:
            await self._client.aclose()
            self._client = None
        
        self.is_connected = False
        logger.info("Disconnected from OANDA")
    
    async def get_current_price(self) -> Optional[Tick]:
        """Get current price from OANDA."""
        if not self._client:
            return None
        
        try:
            response = await self._client.get(
                f"/v3/accounts/{self.account_id}/pricing",
                params={"instruments": self.symbol}
            )
            
            if response.status_code == 200:
                data = response.json()
                if data.get("prices"):
                    price = data["prices"][0]
                    return Tick(
                        symbol=self.symbol,
                        timestamp=datetime.fromisoformat(price["time"].replace("Z", "+00:00")),
                        bid=Decimal(price["bids"][0]["price"]),
                        ask=Decimal(price["asks"][0]["price"]),
                    )
        except Exception as e:
            logger.error(f"Error getting price: {e}")
        
        return None
    
    async def get_historical_data(
        self,
        timeframe: str,
        start: datetime,
        end: Optional[datetime] = None,
        limit: int = 1000,
    ) -> List[OHLCV]:
        """Get historical candle data from OANDA."""
        if not self._client:
            return []
        
        oanda_tf = self.TIMEFRAME_MAP.get(timeframe, "H1")
        end = end or datetime.utcnow()
        
        try:
            response = await self._client.get(
                f"/v3/instruments/{self.symbol}/candles",
                params={
                    "granularity": oanda_tf,
                    "from": start.isoformat() + "Z",
                    "to": end.isoformat() + "Z",
                    "count": limit,
                }
            )
            
            if response.status_code == 200:
                data = response.json()
                candles = []
                
                for c in data.get("candles", []):
                    if c.get("complete"):
                        mid = c["mid"]
                        candles.append(OHLCV(
                            symbol=self.symbol.replace("_", ""),
                            timeframe=timeframe,
                            timestamp=datetime.fromisoformat(c["time"].replace("Z", "+00:00")),
                            open=Decimal(mid["o"]),
                            high=Decimal(mid["h"]),
                            low=Decimal(mid["l"]),
                            close=Decimal(mid["c"]),
                            volume=Decimal(str(c.get("volume", 0))),
                        ))
                
                return candles
                
        except Exception as e:
            logger.error(f"Error getting historical data: {e}")
        
        return []
    
    async def subscribe_ticks(self, callback: callable) -> None:
        """Subscribe to real-time tick stream."""
        self.add_callback(callback)
        
        if not self._stream_task:
            self._stream_task = asyncio.create_task(self._stream_prices())
    
    async def _stream_prices(self) -> None:
        """Stream prices from OANDA."""
        url = f"{self.stream_url}/v3/accounts/{self.account_id}/pricing/stream"
        params = {"instruments": self.symbol}
        
        while self.is_connected:
            try:
                async with httpx.AsyncClient() as client:
                    async with client.stream(
                        "GET", url,
                        params=params,
                        headers=self.headers,
                        timeout=None,
                    ) as response:
                        async for line in response.aiter_lines():
                            if line:
                                await self._process_stream_line(line)
                                
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Stream error: {e}")
                await asyncio.sleep(5)
    
    async def _process_stream_line(self, line: str) -> None:
        """Process a line from the price stream."""
        import json
        
        try:
            data = json.loads(line)
            
            if data.get("type") == "PRICE":
                tick = Tick(
                    symbol=data["instrument"].replace("_", ""),
                    timestamp=datetime.fromisoformat(data["time"].replace("Z", "+00:00")),
                    bid=Decimal(data["bids"][0]["price"]),
                    ask=Decimal(data["asks"][0]["price"]),
                )
                await self.notify_callbacks(tick)
                
        except Exception as e:
            logger.debug(f"Error processing stream: {e}")
