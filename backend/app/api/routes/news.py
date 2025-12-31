"""
News Routes
===========
News and sentiment analysis endpoints.
"""

from datetime import datetime, timedelta
from typing import List, Optional
from fastapi import APIRouter, Depends, Query
from pydantic import BaseModel
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, desc

from app.dependencies import get_db, get_current_user
from app.database.models import NewsArticle, EconomicEvent, SentimentScore, User


router = APIRouter()


class NewsResponse(BaseModel):
    id: int
    title: str
    summary: Optional[str]
    source: str
    url: str
    category: str
    importance: str
    sentiment_score: float
    sentiment_label: str
    gold_impact: float
    published_at: datetime

    class Config:
        from_attributes = True


class EconomicEventResponse(BaseModel):
    id: int
    name: str
    country: str
    currency: str
    event_datetime: datetime
    importance: str
    actual: Optional[str]
    forecast: Optional[str]
    previous: Optional[str]
    expected_gold_impact: float

    class Config:
        from_attributes = True


class SentimentResponse(BaseModel):
    overall_sentiment: float
    news_sentiment: float
    social_sentiment: float
    fear_greed_index: float
    retail_sentiment: float
    institutional_sentiment: float
    timestamp: datetime


@router.get("/latest", response_model=List[NewsResponse])
async def get_latest_news(
    limit: int = Query(20, ge=1, le=100),
    category: Optional[str] = None,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    """Get latest news articles."""
    query = select(NewsArticle).order_by(desc(NewsArticle.published_at)).limit(limit)
    
    if category:
        query = query.where(NewsArticle.category == category)
    
    result = await db.execute(query)
    return result.scalars().all()


@router.get("/calendar", response_model=List[EconomicEventResponse])
async def get_economic_calendar(
    days_ahead: int = Query(7, ge=1, le=30),
    importance: Optional[str] = None,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    """Get upcoming economic events."""
    end_date = datetime.utcnow() + timedelta(days=days_ahead)
    
    query = select(EconomicEvent).where(
        EconomicEvent.event_datetime >= datetime.utcnow(),
        EconomicEvent.event_datetime <= end_date,
    ).order_by(EconomicEvent.event_datetime)
    
    if importance:
        query = query.where(EconomicEvent.importance == importance)
    
    result = await db.execute(query)
    return result.scalars().all()


@router.get("/sentiment", response_model=SentimentResponse)
async def get_current_sentiment(
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    """Get current market sentiment."""
    query = select(SentimentScore).order_by(desc(SentimentScore.timestamp)).limit(1)
    result = await db.execute(query)
    sentiment = result.scalar_one_or_none()
    
    if sentiment:
        return SentimentResponse(
            overall_sentiment=sentiment.overall_sentiment,
            news_sentiment=sentiment.news_sentiment,
            social_sentiment=sentiment.social_sentiment,
            fear_greed_index=sentiment.fear_greed_index,
            retail_sentiment=sentiment.retail_sentiment,
            institutional_sentiment=sentiment.institutional_sentiment,
            timestamp=sentiment.timestamp,
        )
    
    return SentimentResponse(
        overall_sentiment=0,
        news_sentiment=0,
        social_sentiment=0,
        fear_greed_index=50,
        retail_sentiment=50,
        institutional_sentiment=50,
        timestamp=datetime.utcnow(),
    )


@router.get("/impact-analysis")
async def get_news_impact_analysis(
    current_user: User = Depends(get_current_user),
):
    """Get news impact analysis."""
    return {
        "upcoming_high_impact": [],
        "recent_impacts": [],
        "predicted_volatility": "medium",
    }
