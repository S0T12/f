"""
Trades Routes
=============
Trade execution and management endpoints.
"""

from datetime import datetime
from typing import List, Optional
from decimal import Decimal
from fastapi import APIRouter, Depends, HTTPException, Query
from pydantic import BaseModel
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, desc

from app.dependencies import get_db, get_current_user
from app.database.models import Trade, TradingAccount, User, TradeDirection, TradeStatus


router = APIRouter()


class TradeRequest(BaseModel):
    direction: str  # buy or sell
    volume: float
    entry_price: Optional[float] = None  # None for market order
    stop_loss: Optional[float] = None
    take_profit: Optional[float] = None
    strategy: Optional[str] = None


class TradeResponse(BaseModel):
    id: int
    symbol: str
    direction: str
    status: str
    entry_price: float
    volume: float
    stop_loss: Optional[float]
    take_profit: Optional[float]
    profit_loss: float
    opened_at: datetime
    closed_at: Optional[datetime]

    class Config:
        from_attributes = True


class ModifyTradeRequest(BaseModel):
    stop_loss: Optional[float] = None
    take_profit: Optional[float] = None


@router.get("/open", response_model=List[TradeResponse])
async def get_open_trades(
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    """Get all open trades."""
    query = select(Trade).join(TradingAccount).where(
        TradingAccount.user_id == current_user.id,
        Trade.status == TradeStatus.OPEN,
    ).order_by(desc(Trade.opened_at))
    
    result = await db.execute(query)
    return result.scalars().all()


@router.get("/history", response_model=List[TradeResponse])
async def get_trade_history(
    limit: int = Query(50, ge=1, le=500),
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    """Get trade history."""
    query = select(Trade).join(TradingAccount).where(
        TradingAccount.user_id == current_user.id,
        Trade.status == TradeStatus.CLOSED,
    ).order_by(desc(Trade.closed_at)).limit(limit)
    
    result = await db.execute(query)
    return result.scalars().all()


@router.post("/execute", response_model=TradeResponse)
async def execute_trade(
    request: TradeRequest,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    """Execute a new trade."""
    # Get user's trading account
    account_query = select(TradingAccount).where(
        TradingAccount.user_id == current_user.id,
        TradingAccount.is_active == True,
    )
    result = await db.execute(account_query)
    account = result.scalar_one_or_none()
    
    if not account:
        raise HTTPException(status_code=400, detail="No active trading account")
    
    # TODO: Implement risk management checks
    # TODO: Execute via broker API
    
    trade = Trade(
        account_id=account.id,
        symbol="XAUUSD",
        direction=TradeDirection(request.direction),
        status=TradeStatus.OPEN,
        entry_price=Decimal(str(request.entry_price or 2650.50)),
        volume=Decimal(str(request.volume)),
        stop_loss=Decimal(str(request.stop_loss)) if request.stop_loss else None,
        take_profit=Decimal(str(request.take_profit)) if request.take_profit else None,
        strategy=request.strategy,
    )
    
    db.add(trade)
    await db.commit()
    await db.refresh(trade)
    
    return trade


@router.put("/{trade_id}/modify", response_model=TradeResponse)
async def modify_trade(
    trade_id: int,
    request: ModifyTradeRequest,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    """Modify an open trade."""
    query = select(Trade).join(TradingAccount).where(
        Trade.id == trade_id,
        TradingAccount.user_id == current_user.id,
        Trade.status == TradeStatus.OPEN,
    )
    result = await db.execute(query)
    trade = result.scalar_one_or_none()
    
    if not trade:
        raise HTTPException(status_code=404, detail="Trade not found")
    
    if request.stop_loss is not None:
        trade.stop_loss = Decimal(str(request.stop_loss))
    if request.take_profit is not None:
        trade.take_profit = Decimal(str(request.take_profit))
    
    await db.commit()
    await db.refresh(trade)
    
    return trade


@router.delete("/{trade_id}/close", response_model=TradeResponse)
async def close_trade(
    trade_id: int,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    """Close an open trade."""
    query = select(Trade).join(TradingAccount).where(
        Trade.id == trade_id,
        TradingAccount.user_id == current_user.id,
        Trade.status == TradeStatus.OPEN,
    )
    result = await db.execute(query)
    trade = result.scalar_one_or_none()
    
    if not trade:
        raise HTTPException(status_code=404, detail="Trade not found")
    
    # TODO: Get actual exit price from broker
    trade.exit_price = Decimal("2655.00")
    trade.status = TradeStatus.CLOSED
    trade.closed_at = datetime.utcnow()
    
    # Calculate P&L
    if trade.direction == TradeDirection.BUY:
        trade.profit_loss = (trade.exit_price - trade.entry_price) * trade.volume * 100
    else:
        trade.profit_loss = (trade.entry_price - trade.exit_price) * trade.volume * 100
    
    await db.commit()
    await db.refresh(trade)
    
    return trade
