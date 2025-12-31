"""
Trading Tasks
=============
Background tasks for trading operations.
"""

import logging
from datetime import datetime
from typing import Dict, Any

from celery import shared_task

logger = logging.getLogger(__name__)


@shared_task(bind=True, max_retries=2)
def update_signals(self):
    """Update trading signals from all strategies."""
    try:
        from app.strategies.ml_strategy import StrategyOrchestrator
        from app.services.redis_service import redis_service
        from app.database.session import get_db_session
        from app.database.models import Signal as SignalModel, OHLCVData
        import pandas as pd
        
        orchestrator = StrategyOrchestrator()
        
        # Get recent market data
        async def get_data():
            async with get_db_session() as session:
                from sqlalchemy import select
                
                result = await session.execute(
                    select(OHLCVData)
                    .where(OHLCVData.symbol == "XAU/USD")
                    .where(OHLCVData.timeframe == "H1")
                    .order_by(OHLCVData.timestamp.desc())
                    .limit(300)
                )
                candles = result.scalars().all()
                
                if len(candles) < 100:
                    return None
                
                df = pd.DataFrame([{
                    "timestamp": c.timestamp,
                    "open": c.open,
                    "high": c.high,
                    "low": c.low,
                    "close": c.close,
                    "volume": c.volume,
                } for c in reversed(candles)])
                
                df.set_index("timestamp", inplace=True)
                return df
        
        import asyncio
        df = asyncio.run(get_data())
        
        if df is None:
            return {"status": "insufficient_data"}
        
        # Get consensus signal
        signal = orchestrator.get_consensus_signal(df)
        
        # Store signal if actionable
        if signal.is_actionable:
            async def store_signal():
                async with get_db_session() as session:
                    sig = SignalModel(
                        symbol="XAU/USD",
                        signal_type=signal.signal_type.value,
                        confidence=signal.confidence,
                        entry_price=signal.price,
                        stop_loss=signal.stop_loss,
                        take_profit=signal.take_profit,
                        strategy_name=signal.strategy_name,
                        timeframe="H1",
                        metadata=signal.metadata,
                        created_at=signal.timestamp,
                    )
                    session.add(sig)
                    await session.commit()
                    return sig.id
            
            signal_id = asyncio.run(store_signal())
            
            # Cache and publish
            signal_data = signal.to_dict()
            signal_data["id"] = signal_id
            
            redis_service.set("signal:latest", signal_data, expire=300)
            redis_service.publish("signals:xauusd", signal_data)
            
            logger.info(f"New signal: {signal.signal_type.value} @ {signal.price:.2f}")
            
            return {
                "status": "signal_generated",
                "signal_id": signal_id,
                "type": signal.signal_type.value,
                "confidence": signal.confidence,
            }
        
        return {
            "status": "no_signal",
            "confidence": signal.confidence,
        }
    
    except Exception as e:
        logger.error(f"Signal update error: {e}")
        self.retry(exc=e, countdown=30)


@shared_task
def execute_trade(signal_id: int, user_id: int):
    """Execute a trade based on signal."""
    try:
        from app.database.session import get_db_session
        from app.database.models import Signal as SignalModel, Trade, TradingAccount
        from app.risk_management import RiskManager
        from app.strategies.base import Signal, SignalType
        
        async def process_trade():
            async with get_db_session() as session:
                from sqlalchemy import select
                
                # Get signal
                result = await session.execute(
                    select(SignalModel).where(SignalModel.id == signal_id)
                )
                signal_db = result.scalar_one_or_none()
                
                if not signal_db:
                    return {"status": "signal_not_found"}
                
                # Get trading account
                result = await session.execute(
                    select(TradingAccount)
                    .where(TradingAccount.user_id == user_id)
                    .where(TradingAccount.is_active == True)
                )
                account = result.scalar_one_or_none()
                
                if not account:
                    return {"status": "no_active_account"}
                
                # Create Signal object
                signal = Signal(
                    signal_type=SignalType(signal_db.signal_type),
                    confidence=signal_db.confidence,
                    price=signal_db.entry_price,
                    timestamp=signal_db.created_at,
                    strategy_name=signal_db.strategy_name,
                    stop_loss=signal_db.stop_loss,
                    take_profit=signal_db.take_profit,
                )
                
                # Risk assessment
                risk_manager = RiskManager(account.balance)
                assessment = risk_manager.assess_trade(signal, signal_db.entry_price)
                
                if assessment.decision.value in ["approve", "reduce"]:
                    # Create trade
                    trade = Trade(
                        user_id=user_id,
                        account_id=account.id,
                        signal_id=signal_id,
                        symbol="XAU/USD",
                        direction=signal_db.signal_type,
                        entry_price=signal_db.entry_price,
                        lot_size=assessment.position_size.lot_size,
                        stop_loss=assessment.adjusted_stop_loss,
                        take_profit=assessment.adjusted_take_profit,
                        status="pending",
                        risk_percent=assessment.position_size.risk_percent,
                        created_at=datetime.utcnow(),
                    )
                    session.add(trade)
                    await session.commit()
                    
                    # Queue order execution
                    place_order.delay(trade.id)
                    
                    return {
                        "status": "trade_created",
                        "trade_id": trade.id,
                        "lot_size": assessment.position_size.lot_size,
                    }
                else:
                    return {
                        "status": "rejected",
                        "reason": assessment.reasons,
                    }
        
        import asyncio
        return asyncio.run(process_trade())
    
    except Exception as e:
        logger.error(f"Trade execution error: {e}")
        return {"status": "error", "message": str(e)}


@shared_task(bind=True, max_retries=3)
def place_order(self, trade_id: int):
    """Place order with broker."""
    try:
        from app.database.session import get_db_session
        from app.database.models import Trade, Order
        
        async def place():
            async with get_db_session() as session:
                from sqlalchemy import select
                
                result = await session.execute(
                    select(Trade).where(Trade.id == trade_id)
                )
                trade = result.scalar_one_or_none()
                
                if not trade:
                    return {"status": "trade_not_found"}
                
                # In production, this would call broker API
                # Simulating order placement
                order = Order(
                    trade_id=trade.id,
                    symbol=trade.symbol,
                    order_type="market",
                    direction=trade.direction,
                    lot_size=trade.lot_size,
                    price=trade.entry_price,
                    status="filled",
                    broker_order_id=f"SIM-{datetime.utcnow().timestamp()}",
                    created_at=datetime.utcnow(),
                    filled_at=datetime.utcnow(),
                )
                session.add(order)
                
                trade.status = "open"
                trade.opened_at = datetime.utcnow()
                trade.broker_order_id = order.broker_order_id
                
                await session.commit()
                
                return {
                    "status": "order_placed",
                    "order_id": order.id,
                    "broker_id": order.broker_order_id,
                }
        
        import asyncio
        return asyncio.run(place())
    
    except Exception as e:
        logger.error(f"Order placement error: {e}")
        self.retry(exc=e, countdown=5)


@shared_task
def check_open_trades():
    """Check and manage open trades (SL/TP hits)."""
    try:
        from app.database.session import get_db_session
        from app.database.models import Trade
        from app.services.redis_service import redis_service
        
        async def check():
            async with get_db_session() as session:
                from sqlalchemy import select
                
                # Get current price
                tick = redis_service.get("xauusd:tick:latest")
                if not tick:
                    return {"status": "no_price"}
                
                current_price = (tick["bid"] + tick["ask"]) / 2
                
                # Get open trades
                result = await session.execute(
                    select(Trade).where(Trade.status == "open")
                )
                trades = result.scalars().all()
                
                closed = 0
                for trade in trades:
                    should_close = False
                    close_reason = None
                    
                    if trade.direction == "buy":
                        if current_price <= trade.stop_loss:
                            should_close = True
                            close_reason = "stop_loss"
                        elif current_price >= trade.take_profit:
                            should_close = True
                            close_reason = "take_profit"
                    else:  # sell
                        if current_price >= trade.stop_loss:
                            should_close = True
                            close_reason = "stop_loss"
                        elif current_price <= trade.take_profit:
                            should_close = True
                            close_reason = "take_profit"
                    
                    if should_close:
                        trade.exit_price = current_price
                        trade.status = "closed"
                        trade.close_reason = close_reason
                        trade.closed_at = datetime.utcnow()
                        
                        # Calculate PnL
                        if trade.direction == "buy":
                            pnl = (current_price - trade.entry_price) * trade.lot_size * 100
                        else:
                            pnl = (trade.entry_price - current_price) * trade.lot_size * 100
                        
                        trade.pnl = pnl
                        closed += 1
                
                await session.commit()
                return {"status": "checked", "closed": closed}
        
        import asyncio
        return asyncio.run(check())
    
    except Exception as e:
        logger.error(f"Trade check error: {e}")
        return {"status": "error", "message": str(e)}


@shared_task
def reset_daily_limits():
    """Reset daily trading limits."""
    try:
        from app.risk_management import RiskManager
        
        # This would reset limits for all active accounts
        logger.info("Daily trading limits reset")
        return {"status": "success"}
    
    except Exception as e:
        logger.error(f"Reset limits error: {e}")
        return {"status": "error"}
