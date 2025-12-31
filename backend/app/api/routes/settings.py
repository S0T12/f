"""
Settings Routes
===============
User settings and configuration endpoints.
"""

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select

from app.dependencies import get_db, get_current_user
from app.database.models import TradingAccount, User


router = APIRouter()


class RiskParams(BaseModel):
    max_risk_per_trade: float
    max_drawdown: float
    max_open_trades: int
    default_stop_loss_pips: float
    default_take_profit_pips: float


class StrategySettings(BaseModel):
    enabled_strategies: list[str]
    auto_trade: bool
    min_confidence: float
    preferred_timeframe: str


@router.get("/risk-params", response_model=RiskParams)
async def get_risk_params(
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    """Get risk management parameters."""
    query = select(TradingAccount).where(
        TradingAccount.user_id == current_user.id,
        TradingAccount.is_active == True,
    )
    result = await db.execute(query)
    account = result.scalar_one_or_none()
    
    if not account:
        return RiskParams(
            max_risk_per_trade=0.01,
            max_drawdown=0.15,
            max_open_trades=5,
            default_stop_loss_pips=50,
            default_take_profit_pips=100,
        )
    
    return RiskParams(
        max_risk_per_trade=account.max_risk_per_trade,
        max_drawdown=account.max_drawdown,
        max_open_trades=account.max_open_trades,
        default_stop_loss_pips=50,
        default_take_profit_pips=100,
    )


@router.put("/risk-params", response_model=RiskParams)
async def update_risk_params(
    params: RiskParams,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    """Update risk management parameters."""
    query = select(TradingAccount).where(
        TradingAccount.user_id == current_user.id,
        TradingAccount.is_active == True,
    )
    result = await db.execute(query)
    account = result.scalar_one_or_none()
    
    if not account:
        raise HTTPException(status_code=404, detail="No active account")
    
    account.max_risk_per_trade = params.max_risk_per_trade
    account.max_drawdown = params.max_drawdown
    account.max_open_trades = params.max_open_trades
    
    await db.commit()
    
    return params


@router.get("/strategies", response_model=StrategySettings)
async def get_strategy_settings(
    current_user: User = Depends(get_current_user),
):
    """Get strategy settings."""
    # TODO: Load from user settings
    return StrategySettings(
        enabled_strategies=["trend_following", "volatility_breakout", "swing"],
        auto_trade=False,
        min_confidence=0.7,
        preferred_timeframe="1h",
    )


@router.put("/strategies", response_model=StrategySettings)
async def update_strategy_settings(
    settings: StrategySettings,
    current_user: User = Depends(get_current_user),
):
    """Update strategy settings."""
    # TODO: Save to user settings
    return settings
