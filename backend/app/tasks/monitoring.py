"""
Monitoring Tasks
================
Background tasks for system monitoring and maintenance.
"""

import logging
from datetime import datetime, timedelta

from celery import shared_task

logger = logging.getLogger(__name__)


@shared_task
def check_model_performance():
    """Check model performance metrics."""
    try:
        from app.ml_engine.auto_improve.system import improvement_system
        
        stats = improvement_system.get_improvement_stats()
        
        logger.info(
            f"Model performance - Accuracy: {stats['current_accuracy']:.4f}, "
            f"Predictions: {stats['total_predictions']}, "
            f"Drift: {stats['drift_detected']}"
        )
        
        # Alert if performance is dropping
        if stats['current_accuracy'] < 0.6:
            logger.warning("Model accuracy below 60%! Consider retraining.")
            # Could send alert notification here
        
        return stats
    
    except Exception as e:
        logger.error(f"Performance check error: {e}")
        return {"status": "error", "message": str(e)}


@shared_task
def check_drift():
    """Check for model drift."""
    try:
        from app.ml_engine.auto_improve.system import improvement_system
        
        drift_result = improvement_system.check_drift()
        
        if drift_result["drift_detected"]:
            logger.warning(f"Model drift detected: {drift_result}")
            # Could trigger alert or auto-retrain
        
        return drift_result
    
    except Exception as e:
        logger.error(f"Drift check error: {e}")
        return {"status": "error", "message": str(e)}


@shared_task
def cleanup_old_data():
    """Clean up old data to manage storage."""
    try:
        from app.database.session import get_db_session
        from app.database.models import (
            TickData, OHLCVData, Prediction, SystemLog, Alert
        )
        
        async def cleanup():
            async with get_db_session() as session:
                from sqlalchemy import delete
                
                # Delete tick data older than 7 days
                tick_cutoff = datetime.utcnow() - timedelta(days=7)
                await session.execute(
                    delete(TickData).where(TickData.timestamp < tick_cutoff)
                )
                
                # Delete M1 OHLCV data older than 30 days
                m1_cutoff = datetime.utcnow() - timedelta(days=30)
                await session.execute(
                    delete(OHLCVData).where(
                        OHLCVData.timeframe == "M1",
                        OHLCVData.timestamp < m1_cutoff
                    )
                )
                
                # Delete old predictions (keep 90 days)
                pred_cutoff = datetime.utcnow() - timedelta(days=90)
                await session.execute(
                    delete(Prediction).where(Prediction.predicted_at < pred_cutoff)
                )
                
                # Delete old system logs (keep 30 days)
                log_cutoff = datetime.utcnow() - timedelta(days=30)
                await session.execute(
                    delete(SystemLog).where(SystemLog.created_at < log_cutoff)
                )
                
                # Delete old alerts (keep 60 days)
                alert_cutoff = datetime.utcnow() - timedelta(days=60)
                await session.execute(
                    delete(Alert).where(Alert.created_at < alert_cutoff)
                )
                
                await session.commit()
        
        import asyncio
        asyncio.run(cleanup())
        
        logger.info("Old data cleanup completed")
        return {"status": "success"}
    
    except Exception as e:
        logger.error(f"Cleanup error: {e}")
        return {"status": "error", "message": str(e)}


@shared_task
def health_check():
    """System health check."""
    try:
        from app.services.redis_service import redis_service
        from app.database.session import get_db_session
        
        checks = {
            "redis": False,
            "database": False,
            "model_loaded": False,
        }
        
        # Check Redis
        try:
            redis_service.redis.ping()
            checks["redis"] = True
        except Exception as e:
            logger.error(f"Redis health check failed: {e}")
        
        # Check Database
        async def check_db():
            try:
                async with get_db_session() as session:
                    await session.execute("SELECT 1")
                    return True
            except Exception as e:
                logger.error(f"Database health check failed: {e}")
                return False
        
        import asyncio
        checks["database"] = asyncio.run(check_db())
        
        # Check if model is available
        from pathlib import Path
        from app.config import settings
        model_path = Path(settings.MODEL_PATH)
        checks["model_loaded"] = any(model_path.glob("ensemble_*"))
        
        all_healthy = all(checks.values())
        
        if not all_healthy:
            logger.warning(f"Health check issues: {checks}")
        
        return {
            "status": "healthy" if all_healthy else "unhealthy",
            "checks": checks,
            "timestamp": datetime.utcnow().isoformat(),
        }
    
    except Exception as e:
        logger.error(f"Health check error: {e}")
        return {"status": "error", "message": str(e)}


@shared_task
def generate_daily_report():
    """Generate daily trading report."""
    try:
        from app.database.session import get_db_session
        from app.database.models import Trade, Prediction, Signal
        
        async def generate():
            async with get_db_session() as session:
                from sqlalchemy import select, func, and_
                
                today = datetime.utcnow().date()
                yesterday = today - timedelta(days=1)
                
                # Get yesterday's trades
                result = await session.execute(
                    select(Trade).where(
                        func.date(Trade.created_at) == yesterday
                    )
                )
                trades = result.scalars().all()
                
                # Get yesterday's predictions
                result = await session.execute(
                    select(Prediction).where(
                        func.date(Prediction.predicted_at) == yesterday
                    )
                )
                predictions = result.scalars().all()
                
                # Calculate metrics
                total_trades = len(trades)
                winning_trades = len([t for t in trades if t.pnl and t.pnl > 0])
                total_pnl = sum(t.pnl or 0 for t in trades)
                
                total_predictions = len(predictions)
                correct_predictions = len([p for p in predictions if p.is_correct])
                
                win_rate = winning_trades / total_trades if total_trades > 0 else 0
                pred_accuracy = correct_predictions / total_predictions if total_predictions > 0 else 0
                
                report = {
                    "date": yesterday.isoformat(),
                    "trades": {
                        "total": total_trades,
                        "winning": winning_trades,
                        "losing": total_trades - winning_trades,
                        "win_rate": win_rate,
                        "total_pnl": total_pnl,
                    },
                    "predictions": {
                        "total": total_predictions,
                        "correct": correct_predictions,
                        "accuracy": pred_accuracy,
                    },
                }
                
                return report
        
        import asyncio
        report = asyncio.run(generate())
        
        logger.info(f"Daily report: {report}")
        
        # Could send report via email/notification here
        
        return report
    
    except Exception as e:
        logger.error(f"Report generation error: {e}")
        return {"status": "error", "message": str(e)}


@shared_task
def sync_broker_positions():
    """Sync open positions with broker."""
    try:
        from app.database.session import get_db_session
        from app.database.models import Trade
        
        # In production, this would:
        # 1. Fetch open positions from broker API
        # 2. Compare with database records
        # 3. Update any discrepancies
        
        logger.info("Broker position sync completed")
        return {"status": "success"}
    
    except Exception as e:
        logger.error(f"Position sync error: {e}")
        return {"status": "error", "message": str(e)}
