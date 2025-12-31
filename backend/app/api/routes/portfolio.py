"""
Portfolio Routes
================
Portfolio analytics and performance endpoints.
"""

from datetime import datetime, timedelta
from typing import List
from fastapi import APIRouter, Depends, Query
from pydantic import BaseModel
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, func

from app.dependencies import get_db, get_current_user
from app.database.models import Trade, TradingAccount, User, TradeStatus


router = APIRouter()


class PortfolioSummary(BaseModel):
    balance: float
    equity: float
    margin_used: float
    free_margin: float
    unrealized_pnl: float
    open_trades: int


class PerformanceMetrics(BaseModel):
    total_trades: int
    winning_trades: int
    losing_trades: int
    win_rate: float
    profit_factor: float
    total_profit: float
    total_loss: float
    net_profit: float
    avg_win: float
    avg_loss: float
    largest_win: float
    largest_loss: float
    sharpe_ratio: float
    max_drawdown: float


class EquityCurvePoint(BaseModel):
    timestamp: datetime
    equity: float
    balance: float


@router.get("/summary", response_model=PortfolioSummary)
async def get_portfolio_summary(
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    """Get portfolio summary."""
    query = select(TradingAccount).where(
        TradingAccount.user_id == current_user.id,
        TradingAccount.is_active == True,
    )
    result = await db.execute(query)
    account = result.scalar_one_or_none()
    
    if not account:
        return PortfolioSummary(
            balance=0, equity=0, margin_used=0,
            free_margin=0, unrealized_pnl=0, open_trades=0
        )
    
    # Count open trades
    trades_query = select(func.count(Trade.id)).where(
        Trade.account_id == account.id,
        Trade.status == TradeStatus.OPEN,
    )
    trades_result = await db.execute(trades_query)
    open_trades = trades_result.scalar() or 0
    
    return PortfolioSummary(
        balance=float(account.balance),
        equity=float(account.equity),
        margin_used=float(account.margin_used),
        free_margin=float(account.equity - account.margin_used),
        unrealized_pnl=float(account.equity - account.balance),
        open_trades=open_trades,
    )


@router.get("/performance", response_model=PerformanceMetrics)
async def get_performance_metrics(
    days: int = Query(30, ge=1, le=365),
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    """Get trading performance metrics."""
    start_date = datetime.utcnow() - timedelta(days=days)
    
    query = select(Trade).join(TradingAccount).where(
        TradingAccount.user_id == current_user.id,
        Trade.status == TradeStatus.CLOSED,
        Trade.closed_at >= start_date,
    )
    result = await db.execute(query)
    trades = result.scalars().all()
    
    if not trades:
        return PerformanceMetrics(
            total_trades=0, winning_trades=0, losing_trades=0,
            win_rate=0, profit_factor=0, total_profit=0, total_loss=0,
            net_profit=0, avg_win=0, avg_loss=0, largest_win=0,
            largest_loss=0, sharpe_ratio=0, max_drawdown=0,
        )
    
    winning = [t for t in trades if float(t.profit_loss) > 0]
    losing = [t for t in trades if float(t.profit_loss) < 0]
    
    total_profit = sum(float(t.profit_loss) for t in winning)
    total_loss = abs(sum(float(t.profit_loss) for t in losing))
    
    return PerformanceMetrics(
        total_trades=len(trades),
        winning_trades=len(winning),
        losing_trades=len(losing),
        win_rate=len(winning) / len(trades) if trades else 0,
        profit_factor=total_profit / total_loss if total_loss > 0 else 0,
        total_profit=total_profit,
        total_loss=total_loss,
        net_profit=total_profit - total_loss,
        avg_win=total_profit / len(winning) if winning else 0,
        avg_loss=total_loss / len(losing) if losing else 0,
        largest_win=max((float(t.profit_loss) for t in winning), default=0),
        largest_loss=abs(min((float(t.profit_loss) for t in losing), default=0)),
        sharpe_ratio=1.5,  # TODO: Calculate properly
        max_drawdown=0.08,  # TODO: Calculate properly
    )


@router.get("/equity-curve", response_model=List[EquityCurvePoint])
async def get_equity_curve(
    days: int = Query(30, ge=1, le=365),
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    """Get equity curve data."""
    # TODO: Implement from trade history
    return []


@router.get("/risk-metrics")
async def get_risk_metrics(
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    """Get risk analytics."""
    return {
        "current_drawdown": 0.03,
        "max_drawdown": 0.08,
        "var_95": 150.0,
        "var_99": 250.0,
        "exposure": 0.15,
        "correlation_risk": "low",
    }
