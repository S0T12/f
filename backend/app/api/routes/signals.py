"""
Signals Routes
==============
AI-generated trading signals endpoints.
"""

from datetime import datetime
from typing import List, Optional
from fastapi import APIRouter, Depends, Query, WebSocket, WebSocketDisconnect
from pydantic import BaseModel
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, desc

from app.dependencies import get_db, get_current_user
from app.database.models import Signal, User, SignalType


router = APIRouter()


class SignalResponse(BaseModel):
    id: int
    symbol: str
    timeframe: str
    signal_type: str
    entry_price: float
    stop_loss: float
    take_profit_1: float
    take_profit_2: Optional[float]
    confidence: float
    technical_score: float
    fundamental_score: float
    sentiment_score: float
    created_at: datetime

    class Config:
        from_attributes = True


class SignalPerformance(BaseModel):
    total_signals: int
    winning_signals: int
    losing_signals: int
    win_rate: float
    avg_profit_pips: float
    avg_confidence: float


@router.get("/current", response_model=List[SignalResponse])
async def get_current_signals(
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    """Get current active trading signals."""
    query = select(Signal).where(
        Signal.is_executed == False,
        Signal.expires_at > datetime.utcnow(),
    ).order_by(desc(Signal.confidence)).limit(10)
    
    result = await db.execute(query)
    return result.scalars().all()


@router.get("/history", response_model=List[SignalResponse])
async def get_signal_history(
    limit: int = Query(50, ge=1, le=500),
    signal_type: Optional[str] = None,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    """Get historical signals."""
    query = select(Signal).order_by(desc(Signal.created_at)).limit(limit)
    
    if signal_type:
        query = query.where(Signal.signal_type == signal_type)
    
    result = await db.execute(query)
    return result.scalars().all()


@router.get("/performance", response_model=SignalPerformance)
async def get_signal_performance(
    days: int = Query(30, ge=1, le=365),
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    """Get signal performance metrics."""
    # TODO: Calculate from actual signal outcomes
    return SignalPerformance(
        total_signals=150,
        winning_signals=123,
        losing_signals=27,
        win_rate=0.82,
        avg_profit_pips=25.5,
        avg_confidence=0.78,
    )


@router.websocket("/stream")
async def signal_stream(websocket: WebSocket):
    """WebSocket endpoint for real-time signal updates."""
    await websocket.accept()
    
    try:
        while True:
            import asyncio
            await asyncio.sleep(5)
            # TODO: Send real signals when generated
    except WebSocketDisconnect:
        pass
