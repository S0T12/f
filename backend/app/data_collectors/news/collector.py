"""
News Collector
==============
Fetch news from various sources for fundamental analysis.
"""

import asyncio
import logging
from datetime import datetime, timedelta
from typing import List, Dict, Any, Optional
from dataclasses import dataclass
import httpx

from app.config import settings

logger = logging.getLogger(__name__)


@dataclass
class NewsArticle:
    """News article data structure."""
    title: str
    source: str
    url: str
    published_at: datetime
    content: Optional[str] = None
    summary: Optional[str] = None
    category: str = "general"
    importance: str = "medium"


class NewsCollector:
    """Collect news from multiple sources."""
    
    def __init__(self):
        self.news_api_key = settings.NEWS_API_KEY
        self._client: Optional[httpx.AsyncClient] = None
    
    async def connect(self) -> None:
        """Initialize HTTP client."""
        self._client = httpx.AsyncClient(timeout=30.0)
    
    async def disconnect(self) -> None:
        """Close HTTP client."""
        if self._client:
            await self._client.aclose()
    
    async def fetch_all(self) -> List[NewsArticle]:
        """Fetch news from all sources."""
        tasks = [
            self._fetch_newsapi(),
            self._fetch_forex_factory(),
        ]
        
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        articles = []
        for result in results:
            if isinstance(result, list):
                articles.extend(result)
            elif isinstance(result, Exception):
                logger.error(f"News fetch error: {result}")
        
        return articles
    
    async def _fetch_newsapi(self) -> List[NewsArticle]:
        """Fetch from NewsAPI."""
        if not self.news_api_key or not self._client:
            return []
        
        try:
            response = await self._client.get(
                "https://newsapi.org/v2/everything",
                params={
                    "apiKey": self.news_api_key,
                    "q": "gold OR XAUUSD OR dollar OR Federal Reserve OR inflation",
                    "language": "en",
                    "sortBy": "publishedAt",
                    "pageSize": 50,
                }
            )
            
            if response.status_code == 200:
                data = response.json()
                articles = []
                
                for item in data.get("articles", []):
                    articles.append(NewsArticle(
                        title=item["title"],
                        source=item["source"]["name"],
                        url=item["url"],
                        published_at=datetime.fromisoformat(
                            item["publishedAt"].replace("Z", "+00:00")
                        ),
                        content=item.get("content"),
                        summary=item.get("description"),
                        category=self._categorize_news(item["title"]),
                    ))
                
                return articles
                
        except Exception as e:
            logger.error(f"NewsAPI error: {e}")
        
        return []
    
    async def _fetch_forex_factory(self) -> List[NewsArticle]:
        """Fetch economic calendar from Forex Factory (scraping)."""
        # Note: In production, use proper API or respect robots.txt
        return []
    
    def _categorize_news(self, title: str) -> str:
        """Categorize news based on title keywords."""
        title_lower = title.lower()
        
        if any(w in title_lower for w in ["fed", "federal reserve", "interest rate", "fomc"]):
            return "central_bank"
        elif any(w in title_lower for w in ["inflation", "cpi", "ppi"]):
            return "inflation"
        elif any(w in title_lower for w in ["employment", "jobs", "nfp", "unemployment"]):
            return "employment"
        elif any(w in title_lower for w in ["gdp", "growth", "recession"]):
            return "economic"
        elif any(w in title_lower for w in ["gold", "xau", "precious metal"]):
            return "gold"
        elif any(w in title_lower for w in ["dollar", "usd", "dxy"]):
            return "usd"
        elif any(w in title_lower for w in ["war", "conflict", "geopolitical"]):
            return "geopolitical"
        
        return "general"
