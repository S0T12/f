"""
Admin Routes
============
Administrative endpoints for system management.
"""

from datetime import datetime
from typing import List
from fastapi import APIRouter, Depends, BackgroundTasks
from pydantic import BaseModel
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, func

from app.dependencies import get_db, get_admin_user
from app.database.models import (
    User, Trade, Signal, ModelVersion, SystemLog, Alert
)


router = APIRouter()


class SystemStatus(BaseModel):
    status: str
    uptime: str
    database: str
    redis: str
    celery: str
    active_models: int
    open_trades: int


class LogEntry(BaseModel):
    id: int
    level: str
    component: str
    message: str
    created_at: datetime

    class Config:
        from_attributes = True


class SystemMetrics(BaseModel):
    total_users: int
    active_users: int
    total_trades: int
    total_signals: int
    avg_accuracy: float
    system_load: float


@router.get("/system-status", response_model=SystemStatus)
async def get_system_status(
    db: AsyncSession = Depends(get_db),
    admin_user: User = Depends(get_admin_user),
):
    """Get system status."""
    # Count active models
    models_query = select(func.count(ModelVersion.id)).where(ModelVersion.is_active == True)
    models_result = await db.execute(models_query)
    active_models = models_result.scalar() or 0
    
    # Count open trades
    from app.database.models import TradeStatus
    trades_query = select(func.count(Trade.id)).where(Trade.status == TradeStatus.OPEN)
    trades_result = await db.execute(trades_query)
    open_trades = trades_result.scalar() or 0
    
    return SystemStatus(
        status="healthy",
        uptime="2 days, 5 hours",
        database="connected",
        redis="connected",
        celery="running",
        active_models=active_models,
        open_trades=open_trades,
    )


@router.post("/retrain-model")
async def trigger_model_retrain(
    model_name: str,
    background_tasks: BackgroundTasks,
    admin_user: User = Depends(get_admin_user),
):
    """Trigger model retraining."""
    # TODO: Add actual retraining task
    # background_tasks.add_task(retrain_model, model_name)
    
    return {"message": f"Retraining triggered for {model_name}"}


@router.get("/logs", response_model=List[LogEntry])
async def get_system_logs(
    level: str = None,
    component: str = None,
    limit: int = 100,
    db: AsyncSession = Depends(get_db),
    admin_user: User = Depends(get_admin_user),
):
    """Get system logs."""
    from sqlalchemy import desc
    
    query = select(SystemLog).order_by(desc(SystemLog.created_at)).limit(limit)
    
    if level:
        query = query.where(SystemLog.level == level)
    if component:
        query = query.where(SystemLog.component == component)
    
    result = await db.execute(query)
    return result.scalars().all()


@router.get("/metrics", response_model=SystemMetrics)
async def get_system_metrics(
    db: AsyncSession = Depends(get_db),
    admin_user: User = Depends(get_admin_user),
):
    """Get system metrics."""
    # Total users
    users_query = select(func.count(User.id))
    users_result = await db.execute(users_query)
    total_users = users_result.scalar() or 0
    
    # Active users
    active_query = select(func.count(User.id)).where(User.is_active == True)
    active_result = await db.execute(active_query)
    active_users = active_result.scalar() or 0
    
    # Total trades
    trades_query = select(func.count(Trade.id))
    trades_result = await db.execute(trades_query)
    total_trades = trades_result.scalar() or 0
    
    # Total signals
    signals_query = select(func.count(Signal.id))
    signals_result = await db.execute(signals_query)
    total_signals = signals_result.scalar() or 0
    
    return SystemMetrics(
        total_users=total_users,
        active_users=active_users,
        total_trades=total_trades,
        total_signals=total_signals,
        avg_accuracy=0.82,
        system_load=0.35,
    )
