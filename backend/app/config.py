"""
Application Configuration
=========================
Central configuration management using Pydantic Settings.
"""

import os
from functools import lru_cache
from typing import List, Optional, Union
from pydantic_settings import BaseSettings
from pydantic import Field, field_validator


class Settings(BaseSettings):
    """Application settings loaded from environment variables."""
    
    # Application
    APP_NAME: str = "XAU/USD AI Trading System"
    APP_VERSION: str = "1.0.0"
    DEBUG: bool = False
    ENVIRONMENT: str = "development"
    
    # API
    API_V1_PREFIX: str = "/api/v1"
    API_HOST: str = "0.0.0.0"
    API_PORT: int = 8000
    ALLOWED_ORIGINS: str = "http://localhost:3000,http://localhost:8080"
    
    @property
    def allowed_origins_list(self) -> List[str]:
        """Get ALLOWED_ORIGINS as a list."""
        return [origin.strip() for origin in self.ALLOWED_ORIGINS.split(',')]
    
    # Database
    DATABASE_URL: str = "postgresql+asyncpg://trading:trading@localhost:5432/trading_db"
    DATABASE_POOL_SIZE: int = 20
    DATABASE_MAX_OVERFLOW: int = 10
    
    # Redis
    REDIS_URL: str = "redis://localhost:6379/0"
    REDIS_CACHE_TTL: int = 300
    
    # RabbitMQ
    RABBITMQ_URL: str = "amqp://guest:guest@localhost:5672/"
    
    # JWT Authentication
    JWT_SECRET_KEY: str = Field(default="your-super-secret-key-change-in-production")
    JWT_ALGORITHM: str = "HS256"
    JWT_ACCESS_TOKEN_EXPIRE_MINUTES: int = 30
    JWT_REFRESH_TOKEN_EXPIRE_DAYS: int = 7
    
    # Trading
    DEFAULT_RISK_PER_TRADE: float = 0.01  # 1% risk per trade
    MAX_DRAWDOWN_PERCENT: float = 0.15    # 15% max drawdown
    MAX_OPEN_TRADES: int = 5
    DEFAULT_LEVERAGE: int = 100
    
    # Market Data APIs
    OANDA_API_KEY: Optional[str] = None
    OANDA_ACCOUNT_ID: Optional[str] = None
    OANDA_ENVIRONMENT: str = "practice"  # practice or live
    
    ALPHA_VANTAGE_API_KEY: Optional[str] = None
    TWELVE_DATA_API_KEY: Optional[str] = None
    
    # News & Sentiment APIs
    NEWS_API_KEY: Optional[str] = None
    TWITTER_BEARER_TOKEN: Optional[str] = None
    REDDIT_CLIENT_ID: Optional[str] = None
    REDDIT_CLIENT_SECRET: Optional[str] = None
    
    # ML Model Settings
    MODEL_PATH: str = "./models"
    RETRAIN_THRESHOLD_ACCURACY: float = 0.75  # Retrain if accuracy drops below
    MIN_TRAINING_SAMPLES: int = 10000
    PREDICTION_CONFIDENCE_THRESHOLD: float = 0.7
    
    # Celery
    CELERY_BROKER_URL: str = "redis://localhost:6379/1"
    CELERY_RESULT_BACKEND: str = "redis://localhost:6379/2"
    
    # Monitoring
    PROMETHEUS_ENABLED: bool = True
    SENTRY_DSN: Optional[str] = None
    
    # Additional trading settings
    TRADING_SYMBOL: str = "XAU_USD"
    DEFAULT_TIMEFRAME: str = "H1"
    MAX_POSITION_SIZE: float = 0.1
    MAX_DAILY_LOSS: float = 0.02
    RISK_PER_TRADE: float = 0.01
    
    # Additional ML settings
    MODEL_RETRAIN_INTERVAL: int = 86400
    
    # Server settings (these are mapped to API_HOST/API_PORT)
    SERVER_HOST: str = "0.0.0.0"
    SERVER_PORT: int = 8000
    
    # Password settings for external services
    RABBITMQ_PASSWORD: Optional[str] = None
    REDIS_PASSWORD: Optional[str] = None
    GRAFANA_PASSWORD: Optional[str] = None
    JWT_SECRET: Optional[str] = None
    SECRET_KEY: Optional[str] = None
    APP_ENV: Optional[str] = None
    
    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
        case_sensitive = True
        extra = "ignore"  # Ignore extra environment variables


@lru_cache()
def get_settings() -> Settings:
    """Get cached settings instance."""
    return Settings()


settings = get_settings()
