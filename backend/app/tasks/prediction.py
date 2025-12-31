"""
Prediction Tasks
================
Background tasks for generating ML predictions.
"""

import logging
from datetime import datetime
from typing import Dict, Any

from celery import shared_task
import numpy as np

logger = logging.getLogger(__name__)


@shared_task(bind=True, max_retries=2)
def generate_predictions(self):
    """Generate predictions using ensemble model."""
    try:
        from app.ml_engine.training.pipeline import TrainingPipeline
        from app.services.redis_service import redis_service
        from app.database.session import get_db_session
        from app.database.models import Prediction, OHLCVData
        import pandas as pd
        
        pipeline = TrainingPipeline()
        
        # Load latest model
        version = pipeline.load_latest_model()
        if not version:
            return {"status": "no_model"}
        
        # Get recent data
        async def get_data_and_predict():
            async with get_db_session() as session:
                from sqlalchemy import select
                
                # Get last 200 H1 candles
                result = await session.execute(
                    select(OHLCVData)
                    .where(OHLCVData.symbol == "XAU/USD")
                    .where(OHLCVData.timeframe == "H1")
                    .order_by(OHLCVData.timestamp.desc())
                    .limit(200)
                )
                candles = result.scalars().all()
                
                if len(candles) < 100:
                    return None
                
                # Convert to DataFrame
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
        df = asyncio.run(get_data_and_predict())
        
        if df is None:
            return {"status": "insufficient_data"}
        
        # Prepare features
        features_df = pipeline.feature_engineer.create_features(df)
        X = features_df[pipeline.feature_engineer.feature_columns].values
        X_scaled = pipeline.feature_engineer.transform(X)
        
        # Create sequence
        seq_length = 60
        X_seq = X_scaled[-seq_length:].reshape(1, seq_length, -1)
        
        # Predict
        result = pipeline.model.predict(X_seq)
        prediction = result[0]
        predicted_class = int(np.argmax(prediction))
        confidence = float(prediction[predicted_class])
        
        # Map to direction
        directions = {0: "down", 1: "neutral", 2: "up"}
        direction = directions[predicted_class]
        
        # Get current price
        current_price = float(df["close"].iloc[-1])
        
        # Store prediction
        async def store_prediction():
            async with get_db_session() as session:
                pred = Prediction(
                    model_version=version,
                    symbol="XAU/USD",
                    timeframe="H1",
                    direction=direction,
                    confidence=confidence,
                    current_price=current_price,
                    predicted_at=datetime.utcnow(),
                    probabilities={
                        "down": float(prediction[0]),
                        "neutral": float(prediction[1]),
                        "up": float(prediction[2]),
                    },
                )
                session.add(pred)
                await session.commit()
                return pred.id
        
        pred_id = asyncio.run(store_prediction())
        
        # Cache in Redis
        redis_service.set(
            "prediction:latest",
            {
                "id": pred_id,
                "direction": direction,
                "confidence": confidence,
                "current_price": current_price,
                "timestamp": datetime.utcnow().isoformat(),
                "model_version": version,
            },
            expire=120,  # 2 minutes
        )
        
        # Publish prediction
        redis_service.publish("predictions:xauusd", {
            "direction": direction,
            "confidence": confidence,
            "price": current_price,
        })
        
        logger.info(f"Generated prediction: {direction} ({confidence:.2%})")
        
        return {
            "status": "success",
            "prediction_id": pred_id,
            "direction": direction,
            "confidence": confidence,
        }
    
    except Exception as e:
        logger.error(f"Prediction error: {e}")
        self.retry(exc=e, countdown=30)


@shared_task
def record_prediction_outcome(prediction_id: int, actual_direction: str):
    """Record the actual outcome of a prediction for model improvement."""
    try:
        from app.database.session import get_db_session
        from app.database.models import Prediction
        from app.ml_engine.auto_improve.system import improvement_system
        
        async def update_prediction():
            async with get_db_session() as session:
                from sqlalchemy import select
                
                result = await session.execute(
                    select(Prediction).where(Prediction.id == prediction_id)
                )
                pred = result.scalar_one_or_none()
                
                if pred:
                    pred.actual_direction = actual_direction
                    pred.is_correct = pred.direction == actual_direction
                    await session.commit()
                    
                    return pred.direction, actual_direction, pred.confidence
        
        import asyncio
        result = asyncio.run(update_prediction())
        
        if result:
            predicted, actual, confidence = result
            improvement_system.record_prediction(predicted, actual, confidence)
        
        return {"status": "success"}
    
    except Exception as e:
        logger.error(f"Record outcome error: {e}")
        return {"status": "error", "message": str(e)}
