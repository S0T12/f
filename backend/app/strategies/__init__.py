"""Trading Strategies Package."""
from app.strategies.base import BaseStrategy, Signal, SignalType
from app.strategies.volatility import VolatilityStrategy
from app.strategies.trend_following import TrendFollowingStrategy
from app.strategies.swing import SwingStrategy
from app.strategies.ml_strategy import MLStrategy
