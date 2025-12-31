"""
Database Models
===============
SQLAlchemy ORM models for the trading system.
"""

from datetime import datetime
from typing import Optional, List
from decimal import Decimal
import enum

from sqlalchemy import (
    Column, Integer, String, Float, Boolean, DateTime, Text,
    ForeignKey, Enum, Numeric, JSON, Index, UniqueConstraint,
    BigInteger,
)
from sqlalchemy.orm import DeclarativeBase, relationship, Mapped, mapped_column
from sqlalchemy.sql import func


class Base(DeclarativeBase):
    """Base model class."""
    pass


# ============== ENUMS ==============

class TradeDirection(str, enum.Enum):
    """Trade direction enum."""
    BUY = "buy"
    SELL = "sell"


class TradeStatus(str, enum.Enum):
    """Trade status enum."""
    PENDING = "pending"
    OPEN = "open"
    CLOSED = "closed"
    CANCELLED = "cancelled"


class OrderType(str, enum.Enum):
    """Order type enum."""
    MARKET = "market"
    LIMIT = "limit"
    STOP = "stop"
    STOP_LIMIT = "stop_limit"


class SignalType(str, enum.Enum):
    """Signal type enum."""
    BUY = "buy"
    SELL = "sell"
    HOLD = "hold"


class Timeframe(str, enum.Enum):
    """Trading timeframe enum."""
    M1 = "1m"
    M5 = "5m"
    M15 = "15m"
    M30 = "30m"
    H1 = "1h"
    H4 = "4h"
    D1 = "1d"
    W1 = "1w"


class MarketRegime(str, enum.Enum):
    """Market regime enum."""
    TRENDING_UP = "trending_up"
    TRENDING_DOWN = "trending_down"
    RANGING = "ranging"
    HIGH_VOLATILITY = "high_volatility"
    LOW_VOLATILITY = "low_volatility"


# ============== USER & AUTH MODELS ==============

class User(Base):
    """User model."""
    __tablename__ = "users"
    
    id: Mapped[int] = mapped_column(Integer, primary_key=True, index=True)
    email: Mapped[str] = mapped_column(String(255), unique=True, index=True, nullable=False)
    username: Mapped[str] = mapped_column(String(100), unique=True, index=True, nullable=False)
    hashed_password: Mapped[str] = mapped_column(String(255), nullable=False)
    full_name: Mapped[Optional[str]] = mapped_column(String(255))
    is_active: Mapped[bool] = mapped_column(Boolean, default=True)
    is_admin: Mapped[bool] = mapped_column(Boolean, default=False)
    is_verified: Mapped[bool] = mapped_column(Boolean, default=False)
    created_at: Mapped[datetime] = mapped_column(DateTime, server_default=func.now())
    updated_at: Mapped[datetime] = mapped_column(DateTime, server_default=func.now(), onupdate=func.now())
    last_login: Mapped[Optional[datetime]] = mapped_column(DateTime)
    
    # Relationships
    accounts: Mapped[List["TradingAccount"]] = relationship("TradingAccount", back_populates="user")
    api_keys: Mapped[List["APIKey"]] = relationship("APIKey", back_populates="user")


class APIKey(Base):
    """API Key model for external integrations."""
    __tablename__ = "api_keys"
    
    id: Mapped[int] = mapped_column(Integer, primary_key=True, index=True)
    user_id: Mapped[int] = mapped_column(Integer, ForeignKey("users.id"), nullable=False)
    name: Mapped[str] = mapped_column(String(100), nullable=False)
    key_hash: Mapped[str] = mapped_column(String(255), nullable=False)
    is_active: Mapped[bool] = mapped_column(Boolean, default=True)
    permissions: Mapped[dict] = mapped_column(JSON, default=dict)
    created_at: Mapped[datetime] = mapped_column(DateTime, server_default=func.now())
    last_used: Mapped[Optional[datetime]] = mapped_column(DateTime)
    expires_at: Mapped[Optional[datetime]] = mapped_column(DateTime)
    
    # Relationships
    user: Mapped["User"] = relationship("User", back_populates="api_keys")


# ============== TRADING ACCOUNT MODELS ==============

class TradingAccount(Base):
    """Trading account model."""
    __tablename__ = "trading_accounts"
    
    id: Mapped[int] = mapped_column(Integer, primary_key=True, index=True)
    user_id: Mapped[int] = mapped_column(Integer, ForeignKey("users.id"), nullable=False)
    name: Mapped[str] = mapped_column(String(100), nullable=False)
    broker: Mapped[str] = mapped_column(String(50), nullable=False)
    account_number: Mapped[str] = mapped_column(String(100))
    balance: Mapped[Decimal] = mapped_column(Numeric(18, 8), default=0)
    equity: Mapped[Decimal] = mapped_column(Numeric(18, 8), default=0)
    margin_used: Mapped[Decimal] = mapped_column(Numeric(18, 8), default=0)
    leverage: Mapped[int] = mapped_column(Integer, default=100)
    currency: Mapped[str] = mapped_column(String(10), default="USD")
    is_live: Mapped[bool] = mapped_column(Boolean, default=False)
    is_active: Mapped[bool] = mapped_column(Boolean, default=True)
    created_at: Mapped[datetime] = mapped_column(DateTime, server_default=func.now())
    updated_at: Mapped[datetime] = mapped_column(DateTime, server_default=func.now(), onupdate=func.now())
    
    # Risk settings
    max_risk_per_trade: Mapped[float] = mapped_column(Float, default=0.01)
    max_drawdown: Mapped[float] = mapped_column(Float, default=0.15)
    max_open_trades: Mapped[int] = mapped_column(Integer, default=5)
    
    # Relationships
    user: Mapped["User"] = relationship("User", back_populates="accounts")
    trades: Mapped[List["Trade"]] = relationship("Trade", back_populates="account")
    orders: Mapped[List["Order"]] = relationship("Order", back_populates="account")


# ============== MARKET DATA MODELS ==============

class OHLCVData(Base):
    """OHLCV candlestick data model."""
    __tablename__ = "ohlcv_data"
    __table_args__ = (
        Index("idx_ohlcv_symbol_timeframe_timestamp", "symbol", "timeframe", "timestamp"),
        UniqueConstraint("symbol", "timeframe", "timestamp", name="uq_ohlcv_symbol_tf_ts"),
    )
    
    id: Mapped[int] = mapped_column(BigInteger, primary_key=True, index=True)
    symbol: Mapped[str] = mapped_column(String(20), nullable=False, index=True)
    timeframe: Mapped[str] = mapped_column(String(10), nullable=False)
    timestamp: Mapped[datetime] = mapped_column(DateTime, nullable=False, index=True)
    open: Mapped[Decimal] = mapped_column(Numeric(18, 8), nullable=False)
    high: Mapped[Decimal] = mapped_column(Numeric(18, 8), nullable=False)
    low: Mapped[Decimal] = mapped_column(Numeric(18, 8), nullable=False)
    close: Mapped[Decimal] = mapped_column(Numeric(18, 8), nullable=False)
    volume: Mapped[Decimal] = mapped_column(Numeric(18, 8), default=0)
    tick_volume: Mapped[int] = mapped_column(Integer, default=0)
    spread: Mapped[int] = mapped_column(Integer, default=0)


class TickData(Base):
    """High-frequency tick data model."""
    __tablename__ = "tick_data"
    __table_args__ = (
        Index("idx_tick_symbol_timestamp", "symbol", "timestamp"),
    )
    
    id: Mapped[int] = mapped_column(BigInteger, primary_key=True, index=True)
    symbol: Mapped[str] = mapped_column(String(20), nullable=False, index=True)
    timestamp: Mapped[datetime] = mapped_column(DateTime, nullable=False, index=True)
    bid: Mapped[Decimal] = mapped_column(Numeric(18, 8), nullable=False)
    ask: Mapped[Decimal] = mapped_column(Numeric(18, 8), nullable=False)
    bid_volume: Mapped[Decimal] = mapped_column(Numeric(18, 8), default=0)
    ask_volume: Mapped[Decimal] = mapped_column(Numeric(18, 8), default=0)


# ============== TRADING MODELS ==============

class Trade(Base):
    """Trade model."""
    __tablename__ = "trades"
    __table_args__ = (
        Index("idx_trade_account_status", "account_id", "status"),
        Index("idx_trade_opened_at", "opened_at"),
    )
    
    id: Mapped[int] = mapped_column(Integer, primary_key=True, index=True)
    account_id: Mapped[int] = mapped_column(Integer, ForeignKey("trading_accounts.id"), nullable=False)
    signal_id: Mapped[Optional[int]] = mapped_column(Integer, ForeignKey("signals.id"))
    
    symbol: Mapped[str] = mapped_column(String(20), nullable=False, default="XAUUSD")
    direction: Mapped[TradeDirection] = mapped_column(Enum(TradeDirection), nullable=False)
    status: Mapped[TradeStatus] = mapped_column(Enum(TradeStatus), default=TradeStatus.PENDING)
    
    # Entry
    entry_price: Mapped[Decimal] = mapped_column(Numeric(18, 8), nullable=False)
    volume: Mapped[Decimal] = mapped_column(Numeric(18, 8), nullable=False)  # Lot size
    
    # Risk management
    stop_loss: Mapped[Optional[Decimal]] = mapped_column(Numeric(18, 8))
    take_profit: Mapped[Optional[Decimal]] = mapped_column(Numeric(18, 8))
    trailing_stop: Mapped[Optional[Decimal]] = mapped_column(Numeric(18, 8))
    
    # Exit
    exit_price: Mapped[Optional[Decimal]] = mapped_column(Numeric(18, 8))
    
    # P&L
    profit_loss: Mapped[Decimal] = mapped_column(Numeric(18, 8), default=0)
    profit_loss_pips: Mapped[float] = mapped_column(Float, default=0)
    commission: Mapped[Decimal] = mapped_column(Numeric(18, 8), default=0)
    swap: Mapped[Decimal] = mapped_column(Numeric(18, 8), default=0)
    
    # Timestamps
    opened_at: Mapped[datetime] = mapped_column(DateTime, server_default=func.now())
    closed_at: Mapped[Optional[datetime]] = mapped_column(DateTime)
    
    # Additional info
    strategy: Mapped[Optional[str]] = mapped_column(String(100))
    notes: Mapped[Optional[str]] = mapped_column(Text)
    extra_data: Mapped[dict] = mapped_column(JSON, default=dict)
    
    # Relationships
    account: Mapped["TradingAccount"] = relationship("TradingAccount", back_populates="trades")
    signal: Mapped[Optional["Signal"]] = relationship("Signal", back_populates="trades")


class Order(Base):
    """Pending order model."""
    __tablename__ = "orders"
    
    id: Mapped[int] = mapped_column(Integer, primary_key=True, index=True)
    account_id: Mapped[int] = mapped_column(Integer, ForeignKey("trading_accounts.id"), nullable=False)
    
    symbol: Mapped[str] = mapped_column(String(20), nullable=False, default="XAUUSD")
    order_type: Mapped[OrderType] = mapped_column(Enum(OrderType), nullable=False)
    direction: Mapped[TradeDirection] = mapped_column(Enum(TradeDirection), nullable=False)
    
    price: Mapped[Decimal] = mapped_column(Numeric(18, 8), nullable=False)
    volume: Mapped[Decimal] = mapped_column(Numeric(18, 8), nullable=False)
    stop_loss: Mapped[Optional[Decimal]] = mapped_column(Numeric(18, 8))
    take_profit: Mapped[Optional[Decimal]] = mapped_column(Numeric(18, 8))
    
    status: Mapped[str] = mapped_column(String(20), default="pending")
    expires_at: Mapped[Optional[datetime]] = mapped_column(DateTime)
    
    created_at: Mapped[datetime] = mapped_column(DateTime, server_default=func.now())
    executed_at: Mapped[Optional[datetime]] = mapped_column(DateTime)
    cancelled_at: Mapped[Optional[datetime]] = mapped_column(DateTime)
    
    # Relationships
    account: Mapped["TradingAccount"] = relationship("TradingAccount", back_populates="orders")


# ============== SIGNAL MODELS ==============

class Signal(Base):
    """Trading signal model."""
    __tablename__ = "signals"
    __table_args__ = (
        Index("idx_signal_created_at", "created_at"),
    )
    
    id: Mapped[int] = mapped_column(Integer, primary_key=True, index=True)
    
    symbol: Mapped[str] = mapped_column(String(20), nullable=False, default="XAUUSD")
    timeframe: Mapped[str] = mapped_column(String(10), nullable=False)
    signal_type: Mapped[SignalType] = mapped_column(Enum(SignalType), nullable=False)
    
    # Price levels
    entry_price: Mapped[Decimal] = mapped_column(Numeric(18, 8), nullable=False)
    stop_loss: Mapped[Decimal] = mapped_column(Numeric(18, 8), nullable=False)
    take_profit_1: Mapped[Decimal] = mapped_column(Numeric(18, 8), nullable=False)
    take_profit_2: Mapped[Optional[Decimal]] = mapped_column(Numeric(18, 8))
    take_profit_3: Mapped[Optional[Decimal]] = mapped_column(Numeric(18, 8))
    
    # Confidence and scoring
    confidence: Mapped[float] = mapped_column(Float, nullable=False)
    technical_score: Mapped[float] = mapped_column(Float, default=0)
    fundamental_score: Mapped[float] = mapped_column(Float, default=0)
    sentiment_score: Mapped[float] = mapped_column(Float, default=0)
    
    # Market context
    market_regime: Mapped[Optional[MarketRegime]] = mapped_column(Enum(MarketRegime))
    volatility: Mapped[float] = mapped_column(Float, default=0)
    
    # Analysis breakdown
    analysis: Mapped[dict] = mapped_column(JSON, default=dict)
    indicators_used: Mapped[List[str]] = mapped_column(JSON, default=list)
    
    # Model info
    model_version: Mapped[str] = mapped_column(String(50))
    
    # Timestamps
    created_at: Mapped[datetime] = mapped_column(DateTime, server_default=func.now())
    expires_at: Mapped[Optional[datetime]] = mapped_column(DateTime)
    
    # Outcome tracking
    is_executed: Mapped[bool] = mapped_column(Boolean, default=False)
    actual_outcome: Mapped[Optional[str]] = mapped_column(String(20))  # win/loss/breakeven
    actual_profit_pips: Mapped[Optional[float]] = mapped_column(Float)
    
    # Relationships
    trades: Mapped[List["Trade"]] = relationship("Trade", back_populates="signal")


# ============== AI/ML MODELS ==============

class ModelVersion(Base):
    """ML model version tracking."""
    __tablename__ = "model_versions"
    
    id: Mapped[int] = mapped_column(Integer, primary_key=True, index=True)
    name: Mapped[str] = mapped_column(String(100), nullable=False)
    version: Mapped[str] = mapped_column(String(50), nullable=False)
    model_type: Mapped[str] = mapped_column(String(50), nullable=False)  # lstm, transformer, xgboost, etc.
    
    # Performance metrics
    accuracy: Mapped[float] = mapped_column(Float, default=0)
    precision: Mapped[float] = mapped_column(Float, default=0)
    recall: Mapped[float] = mapped_column(Float, default=0)
    f1_score: Mapped[float] = mapped_column(Float, default=0)
    profit_factor: Mapped[float] = mapped_column(Float, default=0)
    
    # Training info
    training_samples: Mapped[int] = mapped_column(Integer, default=0)
    validation_samples: Mapped[int] = mapped_column(Integer, default=0)
    hyperparameters: Mapped[dict] = mapped_column(JSON, default=dict)
    feature_columns: Mapped[List[str]] = mapped_column(JSON, default=list)
    
    # File paths
    model_path: Mapped[str] = mapped_column(String(500))
    scaler_path: Mapped[Optional[str]] = mapped_column(String(500))
    
    # Status
    is_active: Mapped[bool] = mapped_column(Boolean, default=False)
    is_production: Mapped[bool] = mapped_column(Boolean, default=False)
    
    # Timestamps
    trained_at: Mapped[datetime] = mapped_column(DateTime, server_default=func.now())
    deployed_at: Mapped[Optional[datetime]] = mapped_column(DateTime)


class Prediction(Base):
    """ML model prediction tracking."""
    __tablename__ = "predictions"
    __table_args__ = (
        Index("idx_prediction_timestamp", "created_at"),
    )
    
    id: Mapped[int] = mapped_column(Integer, primary_key=True, index=True)
    model_version_id: Mapped[int] = mapped_column(Integer, ForeignKey("model_versions.id"), nullable=False)
    
    symbol: Mapped[str] = mapped_column(String(20), nullable=False)
    timeframe: Mapped[str] = mapped_column(String(10), nullable=False)
    
    # Prediction
    predicted_direction: Mapped[str] = mapped_column(String(10), nullable=False)  # up/down/neutral
    predicted_magnitude: Mapped[float] = mapped_column(Float, default=0)  # Expected move in pips
    confidence: Mapped[float] = mapped_column(Float, nullable=False)
    
    # Probabilities
    prob_up: Mapped[float] = mapped_column(Float, default=0)
    prob_down: Mapped[float] = mapped_column(Float, default=0)
    prob_neutral: Mapped[float] = mapped_column(Float, default=0)
    
    # Feature values at prediction time
    feature_values: Mapped[dict] = mapped_column(JSON, default=dict)
    
    # Actual outcome (filled later)
    actual_direction: Mapped[Optional[str]] = mapped_column(String(10))
    actual_magnitude: Mapped[Optional[float]] = mapped_column(Float)
    is_correct: Mapped[Optional[bool]] = mapped_column(Boolean)
    
    # Timestamps
    created_at: Mapped[datetime] = mapped_column(DateTime, server_default=func.now())
    evaluated_at: Mapped[Optional[datetime]] = mapped_column(DateTime)


class ModelPerformance(Base):
    """Daily model performance tracking."""
    __tablename__ = "model_performance"
    
    id: Mapped[int] = mapped_column(Integer, primary_key=True, index=True)
    model_version_id: Mapped[int] = mapped_column(Integer, ForeignKey("model_versions.id"), nullable=False)
    date: Mapped[datetime] = mapped_column(DateTime, nullable=False)
    
    # Daily metrics
    total_predictions: Mapped[int] = mapped_column(Integer, default=0)
    correct_predictions: Mapped[int] = mapped_column(Integer, default=0)
    accuracy: Mapped[float] = mapped_column(Float, default=0)
    
    # Confidence analysis
    avg_confidence: Mapped[float] = mapped_column(Float, default=0)
    high_confidence_accuracy: Mapped[float] = mapped_column(Float, default=0)  # >80% confidence
    
    # Trading metrics
    signals_generated: Mapped[int] = mapped_column(Integer, default=0)
    trades_executed: Mapped[int] = mapped_column(Integer, default=0)
    profitable_trades: Mapped[int] = mapped_column(Integer, default=0)
    total_pips: Mapped[float] = mapped_column(Float, default=0)


# ============== NEWS & SENTIMENT MODELS ==============

class NewsArticle(Base):
    """News article model."""
    __tablename__ = "news_articles"
    __table_args__ = (
        Index("idx_news_published_at", "published_at"),
    )
    
    id: Mapped[int] = mapped_column(Integer, primary_key=True, index=True)
    
    title: Mapped[str] = mapped_column(String(500), nullable=False)
    content: Mapped[Optional[str]] = mapped_column(Text)
    summary: Mapped[Optional[str]] = mapped_column(Text)
    source: Mapped[str] = mapped_column(String(100), nullable=False)
    url: Mapped[str] = mapped_column(String(1000))
    
    # Classification
    category: Mapped[str] = mapped_column(String(50))  # economic, geopolitical, gold, usd, etc.
    importance: Mapped[str] = mapped_column(String(20), default="medium")  # low, medium, high
    
    # Sentiment analysis
    sentiment_score: Mapped[float] = mapped_column(Float, default=0)  # -1 to 1
    sentiment_label: Mapped[str] = mapped_column(String(20), default="neutral")
    gold_impact: Mapped[float] = mapped_column(Float, default=0)  # Expected impact on gold
    usd_impact: Mapped[float] = mapped_column(Float, default=0)  # Expected impact on USD
    
    # Entities extracted
    entities: Mapped[dict] = mapped_column(JSON, default=dict)
    keywords: Mapped[List[str]] = mapped_column(JSON, default=list)
    
    # Timestamps
    published_at: Mapped[datetime] = mapped_column(DateTime, nullable=False)
    fetched_at: Mapped[datetime] = mapped_column(DateTime, server_default=func.now())
    analyzed_at: Mapped[Optional[datetime]] = mapped_column(DateTime)


class EconomicEvent(Base):
    """Economic calendar event model."""
    __tablename__ = "economic_events"
    __table_args__ = (
        Index("idx_event_datetime", "event_datetime"),
    )
    
    id: Mapped[int] = mapped_column(Integer, primary_key=True, index=True)
    
    name: Mapped[str] = mapped_column(String(200), nullable=False)
    country: Mapped[str] = mapped_column(String(10), nullable=False)  # Country code
    currency: Mapped[str] = mapped_column(String(10), nullable=False)
    
    event_datetime: Mapped[datetime] = mapped_column(DateTime, nullable=False, index=True)
    importance: Mapped[str] = mapped_column(String(20), nullable=False)  # low, medium, high
    
    # Values
    actual: Mapped[Optional[str]] = mapped_column(String(50))
    forecast: Mapped[Optional[str]] = mapped_column(String(50))
    previous: Mapped[Optional[str]] = mapped_column(String(50))
    
    # Impact analysis
    expected_gold_impact: Mapped[float] = mapped_column(Float, default=0)
    actual_gold_impact: Mapped[Optional[float]] = mapped_column(Float)
    
    created_at: Mapped[datetime] = mapped_column(DateTime, server_default=func.now())


class SentimentScore(Base):
    """Aggregated sentiment scores."""
    __tablename__ = "sentiment_scores"
    __table_args__ = (
        Index("idx_sentiment_timestamp", "timestamp"),
    )
    
    id: Mapped[int] = mapped_column(Integer, primary_key=True, index=True)
    
    timestamp: Mapped[datetime] = mapped_column(DateTime, nullable=False, index=True)
    symbol: Mapped[str] = mapped_column(String(20), nullable=False, default="XAUUSD")
    
    # Aggregated scores (-100 to 100)
    overall_sentiment: Mapped[float] = mapped_column(Float, default=0)
    news_sentiment: Mapped[float] = mapped_column(Float, default=0)
    social_sentiment: Mapped[float] = mapped_column(Float, default=0)
    
    # Market indicators
    fear_greed_index: Mapped[float] = mapped_column(Float, default=50)
    retail_sentiment: Mapped[float] = mapped_column(Float, default=0)  # % long
    institutional_sentiment: Mapped[float] = mapped_column(Float, default=0)
    
    # COT data
    cot_net_position: Mapped[Optional[float]] = mapped_column(Float)
    cot_change: Mapped[Optional[float]] = mapped_column(Float)
    
    # Source breakdown
    breakdown: Mapped[dict] = mapped_column(JSON, default=dict)


# ============== SYSTEM MODELS ==============

class SystemLog(Base):
    """System log entries."""
    __tablename__ = "system_logs"
    
    id: Mapped[int] = mapped_column(Integer, primary_key=True, index=True)
    
    level: Mapped[str] = mapped_column(String(20), nullable=False)  # info, warning, error, critical
    component: Mapped[str] = mapped_column(String(100), nullable=False)
    message: Mapped[str] = mapped_column(Text, nullable=False)
    details: Mapped[dict] = mapped_column(JSON, default=dict)
    
    created_at: Mapped[datetime] = mapped_column(DateTime, server_default=func.now(), index=True)


class Alert(Base):
    """System and trading alerts."""
    __tablename__ = "alerts"
    
    id: Mapped[int] = mapped_column(Integer, primary_key=True, index=True)
    
    alert_type: Mapped[str] = mapped_column(String(50), nullable=False)
    severity: Mapped[str] = mapped_column(String(20), nullable=False)  # info, warning, critical
    title: Mapped[str] = mapped_column(String(200), nullable=False)
    message: Mapped[str] = mapped_column(Text, nullable=False)
    
    is_read: Mapped[bool] = mapped_column(Boolean, default=False)
    is_dismissed: Mapped[bool] = mapped_column(Boolean, default=False)
    
    created_at: Mapped[datetime] = mapped_column(DateTime, server_default=func.now())
    read_at: Mapped[Optional[datetime]] = mapped_column(DateTime)
