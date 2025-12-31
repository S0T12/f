"""
Data Collection Tasks
=====================
Background tasks for collecting market data, news, and economic events.
"""

import logging
from datetime import datetime, timedelta
from typing import Optional

from celery import shared_task

logger = logging.getLogger(__name__)


@shared_task(bind=True, max_retries=3)
def collect_tick_data(self):
    """Collect real-time tick data."""
    try:
        from app.data_collectors.market.oanda import OANDACollector
        from app.services.redis_service import redis_service
        
        collector = OANDACollector()
        
        # This would be replaced with actual streaming in production
        # For now, get latest price
        tick = collector.get_current_price()
        
        if tick:
            # Store in Redis for real-time access
            redis_service.set(
                "xauusd:tick:latest",
                {
                    "bid": tick.bid,
                    "ask": tick.ask,
                    "timestamp": tick.timestamp.isoformat(),
                },
                expire=10,
            )
            
            # Publish to subscribers
            redis_service.publish("ticks:xauusd", tick.__dict__)
        
        return {"status": "success", "tick": tick is not None}
    
    except Exception as e:
        logger.error(f"Tick collection error: {e}")
        self.retry(exc=e, countdown=5)


@shared_task(bind=True, max_retries=3)
def collect_ohlcv(self, timeframe: str = "M1"):
    """Collect OHLCV candle data."""
    try:
        from app.data_collectors.market.oanda import OANDACollector
        from app.database.session import get_db_session
        from app.database.models import OHLCVData
        
        collector = OANDACollector()
        
        # Get latest candles
        candles = collector.get_historical_data(
            timeframe=timeframe,
            count=10,
        )
        
        if not candles:
            return {"status": "no_data"}
        
        # Store in database
        async def store_candles():
            async with get_db_session() as session:
                for candle in candles:
                    ohlcv = OHLCVData(
                        symbol="XAU/USD",
                        timeframe=timeframe,
                        timestamp=candle.timestamp,
                        open=candle.open,
                        high=candle.high,
                        low=candle.low,
                        close=candle.close,
                        volume=candle.volume,
                    )
                    session.add(ohlcv)
                
                await session.commit()
        
        import asyncio
        asyncio.run(store_candles())
        
        return {"status": "success", "candles": len(candles)}
    
    except Exception as e:
        logger.error(f"OHLCV collection error: {e}")
        self.retry(exc=e, countdown=30)


@shared_task(bind=True, max_retries=2)
def collect_news(self):
    """Collect gold-related news articles."""
    try:
        from app.data_collectors.news.collector import NewsCollector
        from app.data_collectors.sentiment.analyzer import SentimentAnalyzer
        from app.database.session import get_db_session
        from app.database.models import NewsArticle, SentimentScore
        
        news_collector = NewsCollector()
        sentiment_analyzer = SentimentAnalyzer()
        
        # Collect news
        articles = news_collector.collect_all()
        
        if not articles:
            return {"status": "no_news"}
        
        # Analyze sentiment and store
        async def store_news():
            async with get_db_session() as session:
                for article in articles:
                    # Analyze sentiment
                    sentiment = sentiment_analyzer.analyze(
                        article.get("title", "") + " " + article.get("content", "")
                    )
                    
                    # Store article
                    news = NewsArticle(
                        title=article.get("title"),
                        content=article.get("content"),
                        source=article.get("source"),
                        url=article.get("url"),
                        published_at=article.get("published_at"),
                        sentiment_score=sentiment.get("compound", 0),
                    )
                    session.add(news)
                    
                    # Store sentiment
                    sent = SentimentScore(
                        source="news",
                        score=sentiment.get("compound", 0),
                        confidence=sentiment.get("confidence", 0.5),
                        metadata=sentiment,
                    )
                    session.add(sent)
                
                await session.commit()
        
        import asyncio
        asyncio.run(store_news())
        
        return {"status": "success", "articles": len(articles)}
    
    except Exception as e:
        logger.error(f"News collection error: {e}")
        self.retry(exc=e, countdown=60)


@shared_task(bind=True, max_retries=2)
def update_economic_calendar(self):
    """Update economic calendar events."""
    try:
        from app.data_collectors.news.economic_calendar import EconomicCalendarCollector
        from app.database.session import get_db_session
        from app.database.models import EconomicEvent
        
        collector = EconomicCalendarCollector()
        
        # Get upcoming events
        events = collector.get_upcoming_events(days=7)
        
        if not events:
            return {"status": "no_events"}
        
        async def store_events():
            async with get_db_session() as session:
                for event in events:
                    eco_event = EconomicEvent(
                        event_name=event.get("name"),
                        country=event.get("country"),
                        currency=event.get("currency"),
                        scheduled_at=event.get("datetime"),
                        impact=event.get("impact"),
                        actual=event.get("actual"),
                        forecast=event.get("forecast"),
                        previous=event.get("previous"),
                    )
                    session.add(eco_event)
                
                await session.commit()
        
        import asyncio
        asyncio.run(store_events())
        
        return {"status": "success", "events": len(events)}
    
    except Exception as e:
        logger.error(f"Economic calendar error: {e}")
        self.retry(exc=e, countdown=300)
