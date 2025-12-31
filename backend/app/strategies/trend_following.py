"""
Trend Following Strategy
========================
Strategy for trend-based XAU/USD trading.
"""

import logging
from datetime import datetime
from typing import Dict, Any
import pandas as pd
import numpy as np

from app.strategies.base import BaseStrategy, Signal, SignalType
from app.technical_analysis.indicators import trend, momentum, volatility

logger = logging.getLogger(__name__)


class TrendFollowingStrategy(BaseStrategy):
    """
    Trend following strategy using multiple timeframe analysis.
    
    Combines moving averages, ADX, and MACD for strong trend identification.
    Best for H1 and H4 timeframes.
    """
    
    def __init__(
        self,
        fast_ema: int = 12,
        slow_ema: int = 26,
        trend_ema: int = 200,
        adx_threshold: float = 25.0,
        timeframe: str = "H1",
    ):
        super().__init__("TrendFollowing", timeframe)
        self.fast_ema = fast_ema
        self.slow_ema = slow_ema
        self.trend_ema = trend_ema
        self.adx_threshold = adx_threshold
    
    def analyze(self, data: pd.DataFrame) -> Signal:
        """Analyze for trend following signals."""
        if len(data) < self.trend_ema + 10:
            return self._hold_signal(data)
        
        close = data["close"]
        high = data["high"]
        low = data["low"]
        current_price = close.iloc[-1]
        
        # Calculate indicators
        ema_fast = trend.ema(close, self.fast_ema)
        ema_slow = trend.ema(close, self.slow_ema)
        ema_trend = trend.ema(close, self.trend_ema)
        
        macd_line, signal_line, histogram = trend.macd(close)
        adx_value, plus_di, minus_di = trend.adx(high, low, close)
        
        # Supertrend for additional confirmation
        supertrend_line, supertrend_dir = trend.supertrend(high, low, close)
        
        # RSI for momentum confirmation
        rsi = momentum.rsi(close, 14)
        
        # Current values
        current_adx = adx_value.iloc[-1]
        current_plus_di = plus_di.iloc[-1]
        current_minus_di = minus_di.iloc[-1]
        current_rsi = rsi.iloc[-1]
        current_supertrend_dir = supertrend_dir.iloc[-1]
        
        # Trend determination
        is_uptrend = (
            ema_fast.iloc[-1] > ema_slow.iloc[-1] and
            close.iloc[-1] > ema_trend.iloc[-1]
        )
        is_downtrend = (
            ema_fast.iloc[-1] < ema_slow.iloc[-1] and
            close.iloc[-1] < ema_trend.iloc[-1]
        )
        
        # Strong trend detection
        strong_trend = current_adx > self.adx_threshold
        
        # EMA crossover detection
        ema_bullish_cross = (
            ema_fast.iloc[-2] <= ema_slow.iloc[-2] and
            ema_fast.iloc[-1] > ema_slow.iloc[-1]
        )
        ema_bearish_cross = (
            ema_fast.iloc[-2] >= ema_slow.iloc[-2] and
            ema_fast.iloc[-1] < ema_slow.iloc[-1]
        )
        
        # MACD confirmation
        macd_bullish = histogram.iloc[-1] > 0 and histogram.iloc[-1] > histogram.iloc[-2]
        macd_bearish = histogram.iloc[-1] < 0 and histogram.iloc[-1] < histogram.iloc[-2]
        
        # Generate signal
        signal_type = SignalType.HOLD
        confidence = 0.0
        metadata: Dict[str, Any] = {
            "adx": current_adx,
            "plus_di": current_plus_di,
            "minus_di": current_minus_di,
            "rsi": current_rsi,
            "is_uptrend": is_uptrend,
            "is_downtrend": is_downtrend,
            "strong_trend": strong_trend,
            "supertrend_direction": current_supertrend_dir,
        }
        
        # Bullish trend entry
        if is_uptrend and strong_trend:
            if (ema_bullish_cross or macd_bullish) and current_supertrend_dir == 1:
                if current_plus_di > current_minus_di and 30 < current_rsi < 70:
                    signal_type = SignalType.BUY
                    confidence = self._calculate_confidence(
                        current_adx, is_uptrend, macd_bullish, current_supertrend_dir == 1
                    )
        
        # Bearish trend entry
        elif is_downtrend and strong_trend:
            if (ema_bearish_cross or macd_bearish) and current_supertrend_dir == -1:
                if current_minus_di > current_plus_di and 30 < current_rsi < 70:
                    signal_type = SignalType.SELL
                    confidence = self._calculate_confidence(
                        current_adx, is_downtrend, macd_bearish, current_supertrend_dir == -1
                    )
        
        # Trend continuation signals (pullback entries)
        if signal_type == SignalType.HOLD and strong_trend:
            # Bullish pullback to EMA
            if is_uptrend and close.iloc[-1] <= ema_fast.iloc[-1] * 1.005:
                if current_rsi < 60 and current_rsi > 40:
                    signal_type = SignalType.BUY
                    confidence = 0.65
                    metadata["signal_reason"] = "bullish_pullback"
            
            # Bearish pullback to EMA
            elif is_downtrend and close.iloc[-1] >= ema_fast.iloc[-1] * 0.995:
                if current_rsi > 40 and current_rsi < 60:
                    signal_type = SignalType.SELL
                    confidence = 0.65
                    metadata["signal_reason"] = "bearish_pullback"
        
        return Signal(
            signal_type=signal_type,
            confidence=confidence,
            price=current_price,
            timestamp=datetime.utcnow(),
            strategy_name=self.name,
            metadata=metadata,
        )
    
    def _calculate_confidence(
        self,
        adx: float,
        trend_aligned: bool,
        macd_aligned: bool,
        supertrend_aligned: bool,
    ) -> float:
        """Calculate signal confidence."""
        confidence = 0.5
        
        if trend_aligned:
            confidence += 0.15
        if macd_aligned:
            confidence += 0.1
        if supertrend_aligned:
            confidence += 0.1
        if adx > 30:
            confidence += 0.05
        if adx > 40:
            confidence += 0.05
        
        return min(0.9, confidence)
    
    def calculate_stop_loss(
        self,
        price: float,
        signal_type: SignalType,
        data: pd.DataFrame,
    ) -> float:
        """Calculate stop loss based on recent swing high/low."""
        # Use recent swing points
        lookback = 20
        
        if signal_type == SignalType.BUY:
            # Stop below recent swing low
            recent_low = data["low"].iloc[-lookback:].min()
            return recent_low * 0.999  # Small buffer
        else:
            # Stop above recent swing high
            recent_high = data["high"].iloc[-lookback:].max()
            return recent_high * 1.001
    
    def calculate_take_profit(
        self,
        price: float,
        signal_type: SignalType,
        data: pd.DataFrame,
    ) -> float:
        """Calculate take profit with 2:1 reward/risk."""
        stop_loss = self.calculate_stop_loss(price, signal_type, data)
        risk = abs(price - stop_loss)
        
        if signal_type == SignalType.BUY:
            return price + (risk * 2)
        else:
            return price - (risk * 2)
    
    def _hold_signal(self, data: pd.DataFrame) -> Signal:
        """Return hold signal."""
        return Signal(
            signal_type=SignalType.HOLD,
            confidence=0.0,
            price=data["close"].iloc[-1] if len(data) > 0 else 0,
            timestamp=datetime.utcnow(),
            strategy_name=self.name,
        )
