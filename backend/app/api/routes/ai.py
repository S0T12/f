"""
AI/ML Routes
============
AI predictions and model performance endpoints.
"""

from datetime import datetime, timedelta
from typing import List
from fastapi import APIRouter, Depends, Query
from pydantic import BaseModel
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, desc

from app.dependencies import get_db, get_current_user
from app.database.models import ModelVersion, Prediction, ModelPerformance, User


router = APIRouter()


class PredictionResponse(BaseModel):
    predicted_direction: str
    predicted_magnitude: float
    confidence: float
    prob_up: float
    prob_down: float
    prob_neutral: float
    model_version: str
    created_at: datetime


class ModelPerformanceResponse(BaseModel):
    model_name: str
    version: str
    accuracy: float
    precision: float
    recall: float
    f1_score: float
    profit_factor: float
    is_production: bool


class FeatureImportance(BaseModel):
    feature: str
    importance: float


@router.get("/predictions", response_model=PredictionResponse)
async def get_current_prediction(
    timeframe: str = Query("1h"),
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    """Get current AI prediction."""
    query = select(Prediction).where(
        Prediction.timeframe == timeframe,
    ).order_by(desc(Prediction.created_at)).limit(1)
    
    result = await db.execute(query)
    prediction = result.scalar_one_or_none()
    
    if prediction:
        return PredictionResponse(
            predicted_direction=prediction.predicted_direction,
            predicted_magnitude=prediction.predicted_magnitude,
            confidence=prediction.confidence,
            prob_up=prediction.prob_up,
            prob_down=prediction.prob_down,
            prob_neutral=prediction.prob_neutral,
            model_version="v1.0",
            created_at=prediction.created_at,
        )
    
    # Default response when no prediction exists
    return PredictionResponse(
        predicted_direction="neutral",
        predicted_magnitude=0,
        confidence=0.5,
        prob_up=0.33,
        prob_down=0.33,
        prob_neutral=0.34,
        model_version="v1.0",
        created_at=datetime.utcnow(),
    )


@router.get("/model-performance", response_model=List[ModelPerformanceResponse])
async def get_model_performance(
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    """Get performance metrics for all models."""
    query = select(ModelVersion).where(ModelVersion.is_active == True)
    result = await db.execute(query)
    models = result.scalars().all()
    
    return [
        ModelPerformanceResponse(
            model_name=m.name,
            version=m.version,
            accuracy=m.accuracy,
            precision=m.precision,
            recall=m.recall,
            f1_score=m.f1_score,
            profit_factor=m.profit_factor,
            is_production=m.is_production,
        )
        for m in models
    ]


@router.get("/feature-importance", response_model=List[FeatureImportance])
async def get_feature_importance(
    model_name: str = Query("ensemble"),
    current_user: User = Depends(get_current_user),
):
    """Get feature importance from the model."""
    # TODO: Load from actual model
    return [
        FeatureImportance(feature="rsi_14", importance=0.12),
        FeatureImportance(feature="macd_signal", importance=0.10),
        FeatureImportance(feature="sentiment_score", importance=0.09),
        FeatureImportance(feature="atr_14", importance=0.08),
        FeatureImportance(feature="ema_50", importance=0.07),
    ]


@router.get("/confidence-levels")
async def get_confidence_levels(
    current_user: User = Depends(get_current_user),
):
    """Get confidence levels breakdown."""
    return {
        "overall_confidence": 0.82,
        "technical_confidence": 0.85,
        "fundamental_confidence": 0.78,
        "sentiment_confidence": 0.80,
        "model_agreement": 0.75,
    }
