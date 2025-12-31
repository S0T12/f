"""
Training Tasks
==============
Background tasks for model training and retraining.
"""

import logging
from datetime import datetime, timedelta

from celery import shared_task

logger = logging.getLogger(__name__)


@shared_task(bind=True, time_limit=3600)  # 1 hour limit
def train_model(self, epochs: int = 100):
    """Train new model version."""
    try:
        from app.ml_engine.training.pipeline import TrainingPipeline
        from app.database.session import get_db_session
        from app.database.models import OHLCVData, ModelVersion
        import pandas as pd
        
        logger.info("Starting model training...")
        
        async def get_training_data():
            async with get_db_session() as session:
                from sqlalchemy import select
                
                # Get last 6 months of H1 data
                result = await session.execute(
                    select(OHLCVData)
                    .where(OHLCVData.symbol == "XAU/USD")
                    .where(OHLCVData.timeframe == "H1")
                    .order_by(OHLCVData.timestamp.desc())
                    .limit(4320)  # ~6 months of H1 data
                )
                candles = result.scalars().all()
                
                if len(candles) < 1000:
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
        df = asyncio.run(get_training_data())
        
        if df is None:
            return {"status": "insufficient_data"}
        
        # Train model
        pipeline = TrainingPipeline()
        result = asyncio.run(pipeline.train(df, epochs=epochs))
        
        # Store model version in database
        async def store_model_version():
            async with get_db_session() as session:
                model_version = ModelVersion(
                    version=result["version"],
                    model_type="ensemble",
                    accuracy=result["ensemble_metrics"]["accuracy"],
                    precision=result["ensemble_metrics"]["precision"],
                    recall=result["ensemble_metrics"]["recall"],
                    f1_score=result["ensemble_metrics"]["f1_score"],
                    training_samples=result["training_samples"],
                    validation_samples=result["validation_samples"],
                    trained_at=datetime.fromisoformat(result["trained_at"]),
                    is_active=True,
                    metadata={
                        "lstm_metrics": result["lstm_metrics"],
                        "xgboost_metrics": result["xgboost_metrics"],
                        "weights": result["weights"],
                    },
                )
                session.add(model_version)
                
                # Deactivate previous versions
                from sqlalchemy import update
                await session.execute(
                    update(ModelVersion)
                    .where(ModelVersion.version != result["version"])
                    .values(is_active=False)
                )
                
                await session.commit()
        
        asyncio.run(store_model_version())
        
        logger.info(f"Training complete. Version: {result['version']}, Accuracy: {result['ensemble_metrics']['accuracy']:.4f}")
        
        return {
            "status": "success",
            "version": result["version"],
            "accuracy": result["ensemble_metrics"]["accuracy"],
            "duration": result["duration_seconds"],
        }
    
    except Exception as e:
        logger.error(f"Training error: {e}")
        return {"status": "error", "message": str(e)}


@shared_task
def check_and_retrain():
    """Check if retraining is needed and trigger if so."""
    try:
        from app.ml_engine.auto_improve.system import improvement_system
        
        if improvement_system.should_retrain():
            logger.info("Auto-retraining triggered")
            train_model.delay(epochs=50)  # Shorter training for auto-retrain
            return {"status": "retrain_triggered"}
        
        return {"status": "no_retrain_needed"}
    
    except Exception as e:
        logger.error(f"Retrain check error: {e}")
        return {"status": "error", "message": str(e)}


@shared_task
def optimize_ensemble_weights():
    """Optimize ensemble model weights based on recent performance."""
    try:
        from app.database.session import get_db_session
        from app.database.models import Prediction
        from app.ml_engine.training.pipeline import TrainingPipeline
        
        async def get_recent_predictions():
            async with get_db_session() as session:
                from sqlalchemy import select, and_
                
                result = await session.execute(
                    select(Prediction)
                    .where(
                        and_(
                            Prediction.is_correct.isnot(None),
                            Prediction.predicted_at >= datetime.utcnow() - timedelta(days=7)
                        )
                    )
                )
                return result.scalars().all()
        
        import asyncio
        predictions = asyncio.run(get_recent_predictions())
        
        if len(predictions) < 100:
            return {"status": "insufficient_predictions"}
        
        # Calculate accuracy by model component (would need to track per-model predictions)
        # For now, just report overall accuracy
        correct = sum(1 for p in predictions if p.is_correct)
        accuracy = correct / len(predictions)
        
        logger.info(f"Recent accuracy: {accuracy:.4f} from {len(predictions)} predictions")
        
        return {
            "status": "analyzed",
            "accuracy": accuracy,
            "predictions": len(predictions),
        }
    
    except Exception as e:
        logger.error(f"Weight optimization error: {e}")
        return {"status": "error", "message": str(e)}


@shared_task
def backtest_strategy(strategy_name: str, days: int = 30):
    """Backtest a strategy on historical data."""
    try:
        from app.database.session import get_db_session
        from app.database.models import OHLCVData
        from app.strategies import VolatilityStrategy, TrendFollowingStrategy, SwingStrategy
        import pandas as pd
        
        strategies = {
            "volatility": VolatilityStrategy(),
            "trend": TrendFollowingStrategy(),
            "swing": SwingStrategy(),
        }
        
        strategy = strategies.get(strategy_name)
        if not strategy:
            return {"status": "invalid_strategy"}
        
        async def get_data():
            async with get_db_session() as session:
                from sqlalchemy import select
                
                result = await session.execute(
                    select(OHLCVData)
                    .where(OHLCVData.symbol == "XAU/USD")
                    .where(OHLCVData.timeframe == "H1")
                    .order_by(OHLCVData.timestamp.desc())
                    .limit(days * 24)
                )
                candles = result.scalars().all()
                
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
        
        if len(df) < 200:
            return {"status": "insufficient_data"}
        
        # Run backtest
        trades = []
        for i in range(200, len(df)):
            subset = df.iloc[:i]
            signal = strategy.generate_signal(subset)
            
            if signal and signal.is_actionable:
                trades.append({
                    "timestamp": df.index[i-1],
                    "type": signal.signal_type.value,
                    "price": signal.price,
                    "confidence": signal.confidence,
                    "sl": signal.stop_loss,
                    "tp": signal.take_profit,
                })
        
        # Calculate basic statistics
        total_trades = len(trades)
        if total_trades == 0:
            return {"status": "no_trades"}
        
        avg_confidence = sum(t["confidence"] for t in trades) / total_trades
        
        return {
            "status": "success",
            "strategy": strategy_name,
            "total_trades": total_trades,
            "avg_confidence": avg_confidence,
            "period_days": days,
        }
    
    except Exception as e:
        logger.error(f"Backtest error: {e}")
        return {"status": "error", "message": str(e)}
