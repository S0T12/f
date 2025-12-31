"""
Market Data Routes
==================
Real-time and historical market data endpoints.
"""

from datetime import datetime, timedelta
from typing import List, Optional
from fastapi import APIRouter, Depends, Query, WebSocket, WebSocketDisconnect
from pydantic import BaseModel
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, desc

from app.dependencies import get_db, get_current_user
from app.database.models import OHLCVData, User


router = APIRouter()


class PriceResponse(BaseModel):
    symbol: str
    bid: float
    ask: float
    spread: float
    timestamp: datetime


class OHLCVResponse(BaseModel):
    timestamp: datetime
    open: float
    high: float
    low: float
    close: float
    volume: float

    class Config:
        from_attributes = True


class IndicatorRequest(BaseModel):
    indicator: str
    period: int = 14
    timeframe: str = "1h"


@router.get("/price/current", response_model=PriceResponse)
async def get_current_price(current_user: User = Depends(get_current_user)):
    """Get current XAU/USD price."""
    # TODO: Integrate with live data feed
    return PriceResponse(
        symbol="XAUUSD",
        bid=2650.50,
        ask=2650.80,
        spread=0.30,
        timestamp=datetime.utcnow(),
    )


@router.get("/price/history", response_model=List[OHLCVResponse])
async def get_price_history(
    timeframe: str = Query("1h", regex="^(1m|5m|15m|30m|1h|4h|1d|1w)$"),
    limit: int = Query(100, ge=1, le=1000),
    start_date: Optional[datetime] = None,
    end_date: Optional[datetime] = None,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    """Get historical OHLCV data."""
    query = select(OHLCVData).where(
        OHLCVData.symbol == "XAUUSD",
        OHLCVData.timeframe == timeframe,
    ).order_by(desc(OHLCVData.timestamp)).limit(limit)
    
    if start_date:
        query = query.where(OHLCVData.timestamp >= start_date)
    if end_date:
        query = query.where(OHLCVData.timestamp <= end_date)
    
    result = await db.execute(query)
    return result.scalars().all()


@router.get("/indicators")
async def get_indicators(
    indicator: str = Query(..., description="Indicator name (rsi, macd, sma, etc.)"),
    period: int = Query(14, ge=1, le=500),
    timeframe: str = Query("1h"),
    current_user: User = Depends(get_current_user),
):
    """Get calculated indicator values."""
    # TODO: Calculate indicators from historical data
    return {
        "indicator": indicator,
        "period": period,
        "timeframe": timeframe,
        "values": [],
    }


@router.websocket("/price/stream")
async def price_stream(websocket: WebSocket):
    """WebSocket endpoint for real-time price streaming."""
    await websocket.accept()
    
    try:
        while True:
            # TODO: Stream real prices from data collector
            import asyncio
            await asyncio.sleep(1)
            
            await websocket.send_json({
                "symbol": "XAUUSD",
                "bid": 2650.50,
                "ask": 2650.80,
                "timestamp": datetime.utcnow().isoformat(),
            })
    except WebSocketDisconnect:
        pass
