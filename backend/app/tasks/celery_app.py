"""
Celery Application
==================
Celery configuration and app initialization.
"""

from celery import Celery
from app.config import settings

celery_app = Celery(
    "trading_system",
    broker=settings.CELERY_BROKER_URL,
    backend=settings.CELERY_RESULT_BACKEND,
    include=[
        "app.tasks.data_collection",
        "app.tasks.prediction",
        "app.tasks.trading",
        "app.tasks.training",
        "app.tasks.monitoring",
    ],
)

# Celery configuration
celery_app.conf.update(
    task_serializer="json",
    accept_content=["json"],
    result_serializer="json",
    timezone="UTC",
    enable_utc=True,
    task_track_started=True,
    task_time_limit=300,  # 5 minutes
    task_soft_time_limit=240,
    worker_prefetch_multiplier=1,
    worker_concurrency=4,
)

# Beat schedule for periodic tasks
celery_app.conf.beat_schedule = {
    # Market data collection
    "collect-tick-data": {
        "task": "app.tasks.data_collection.collect_tick_data",
        "schedule": 1.0,  # Every second
    },
    "collect-ohlcv-m1": {
        "task": "app.tasks.data_collection.collect_ohlcv",
        "schedule": 60.0,  # Every minute
        "args": ["M1"],
    },
    "collect-ohlcv-m15": {
        "task": "app.tasks.data_collection.collect_ohlcv",
        "schedule": 900.0,  # Every 15 minutes
        "args": ["M15"],
    },
    "collect-ohlcv-h1": {
        "task": "app.tasks.data_collection.collect_ohlcv",
        "schedule": 3600.0,  # Every hour
        "args": ["H1"],
    },
    
    # News and sentiment
    "collect-news": {
        "task": "app.tasks.data_collection.collect_news",
        "schedule": 300.0,  # Every 5 minutes
    },
    "update-economic-calendar": {
        "task": "app.tasks.data_collection.update_economic_calendar",
        "schedule": 3600.0,  # Every hour
    },
    
    # Predictions
    "generate-predictions": {
        "task": "app.tasks.prediction.generate_predictions",
        "schedule": 60.0,  # Every minute
    },
    "update-signals": {
        "task": "app.tasks.trading.update_signals",
        "schedule": 60.0,  # Every minute
    },
    
    # Model monitoring
    "check-model-performance": {
        "task": "app.tasks.monitoring.check_model_performance",
        "schedule": 3600.0,  # Every hour
    },
    "check-drift": {
        "task": "app.tasks.monitoring.check_drift",
        "schedule": 1800.0,  # Every 30 minutes
    },
    
    # Daily tasks
    "daily-model-retrain-check": {
        "task": "app.tasks.training.check_and_retrain",
        "schedule": 86400.0,  # Daily
    },
    "daily-reset-limits": {
        "task": "app.tasks.trading.reset_daily_limits",
        "schedule": 86400.0,  # Daily
    },
    "cleanup-old-data": {
        "task": "app.tasks.monitoring.cleanup_old_data",
        "schedule": 86400.0,  # Daily
    },
}
