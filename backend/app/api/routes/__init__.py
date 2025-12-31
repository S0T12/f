"""API routes subpackage."""

from fastapi import APIRouter

from app.api.routes import auth, market, signals, trades, portfolio, news, ai, settings, admin

api_router = APIRouter()

# Include all route modules
api_router.include_router(auth.router, prefix="/auth", tags=["Authentication"])
api_router.include_router(market.router, prefix="/market", tags=["Market Data"])
api_router.include_router(signals.router, prefix="/signals", tags=["Trading Signals"])
api_router.include_router(trades.router, prefix="/trades", tags=["Trades"])
api_router.include_router(portfolio.router, prefix="/portfolio", tags=["Portfolio"])
api_router.include_router(news.router, prefix="/news", tags=["News & Sentiment"])
api_router.include_router(ai.router, prefix="/ai", tags=["AI & Predictions"])
api_router.include_router(settings.router, prefix="/settings", tags=["Settings"])
api_router.include_router(admin.router, prefix="/admin", tags=["Admin"])

__all__ = ["api_router"]
