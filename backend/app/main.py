"""
FastAPI Application Entry Point
===============================
Main application setup with middleware, routes, and event handlers.
"""

import logging
from contextlib import asynccontextmanager
from typing import AsyncGenerator

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.gzip import GZipMiddleware
from prometheus_fastapi_instrumentator import Instrumentator

from app.config import settings
from app.api.routes import api_router
from app.websocket.handlers import router as ws_router
from app.websocket.pubsub import pubsub_listener
from app.database.session import init_db, close_db
from app.services.redis_service import redis_manager
from app.core.logging import setup_logging

# Setup logging
setup_logging()
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator:
    """Application lifespan handler for startup and shutdown events."""
    # Startup
    logger.info("Starting XAU/USD AI Trading System...")
    
    # Initialize database
    await init_db()
    logger.info("Database initialized")
    
    # Initialize Redis
    await redis_manager.connect()
    logger.info("Redis connected")
    
    # Start Redis PubSub listener for WebSocket broadcasting
    await pubsub_listener.start()
    logger.info("PubSub listener started")
    
    # Start background tasks
    logger.info("Background tasks started")
    
    logger.info("Application startup complete")
    
    yield
    
    # Shutdown
    logger.info("Shutting down application...")
    
    # Stop PubSub listener
    await pubsub_listener.stop()
    logger.info("PubSub listener stopped")
    
    # Close Redis
    await redis_manager.disconnect()
    logger.info("Redis disconnected")
    
    # Close database
    await close_db()
    logger.info("Database connections closed")
    
    logger.info("Application shutdown complete")


def create_application() -> FastAPI:
    """Create and configure FastAPI application."""
    
    app = FastAPI(
        title=settings.APP_NAME,
        version=settings.APP_VERSION,
        description="AI-Powered XAU/USD Trading System with ML predictions and automated execution",
        docs_url="/docs" if settings.DEBUG else None,
        redoc_url="/redoc" if settings.DEBUG else None,
        openapi_url="/openapi.json" if settings.DEBUG else None,
        lifespan=lifespan,
    )
    
    # CORS middleware
    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.ALLOWED_ORIGINS,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    
    # Gzip compression
    app.add_middleware(GZipMiddleware, minimum_size=1000)
    
    # Prometheus metrics
    if settings.PROMETHEUS_ENABLED:
        Instrumentator().instrument(app).expose(app)
    
    # Include API routes
    app.include_router(api_router, prefix=settings.API_V1_PREFIX)
    
    # Include WebSocket routes
    app.include_router(ws_router)
    
    # Health check endpoint
    @app.get("/health")
    async def health_check():
        return {
            "status": "healthy",
            "version": settings.APP_VERSION,
            "environment": settings.ENVIRONMENT,
        }
    
    return app


app = create_application()


if __name__ == "__main__":
    import uvicorn
    
    uvicorn.run(
        "app.main:app",
        host=settings.API_HOST,
        port=settings.API_PORT,
        reload=settings.DEBUG,
        workers=4 if not settings.DEBUG else 1,
    )
