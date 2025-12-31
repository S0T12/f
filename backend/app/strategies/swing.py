"""
Swing Trading Strategy
======================
Strategy for swing trading XAU/USD on higher timeframes.
"""

import logging
from datetime import datetime
from typing import Dict, Any, List, Tuple
import pandas as pd
import numpy as np

from app.strategies.base import BaseStrategy, Signal, SignalType
from app.technical_analysis.indicators import trend, momentum, volatility

logger = logging.getLogger(__name__)


class SwingStrategy(BaseStrategy):
    """
    Swing trading strategy for longer-term positions.
    
    Uses support/resistance levels, Fibonacci retracements,
    and divergence detection for swing trade entries.
    Best for H4 and D1 timeframes.
    """
    
    def __init__(
        self,
        swing_lookback: int = 20,
        fib_levels: List[float] = None,
        timeframe: str = "H4",
    ):
        super().__init__("Swing", timeframe)
        self.swing_lookback = swing_lookback
        self.fib_levels = fib_levels or [0.236, 0.382, 0.5, 0.618, 0.786]
    
    def analyze(self, data: pd.DataFrame) -> Signal:
        """Analyze for swing trading signals."""
        if len(data) < 100:
            return self._hold_signal(data)
        
        close = data["close"]
        high = data["high"]
        low = data["low"]
        current_price = close.iloc[-1]
        
        # Find swing highs and lows
        swing_highs, swing_lows = self._find_swing_points(data)
        
        # Calculate Fibonacci levels
        fib_support, fib_resistance = self._calculate_fib_levels(swing_highs, swing_lows)
        
        # Support and resistance levels
        support = self._find_support(data)
        resistance = self._find_resistance(data)
        
        # Indicators
        rsi = momentum.rsi(close, 14)
        macd_line, signal_line, histogram = trend.macd(close)
        
        # Divergence detection
        bullish_div = self._detect_bullish_divergence(close, rsi)
        bearish_div = self._detect_bearish_divergence(close, rsi)
        
        # Current values
        current_rsi = rsi.iloc[-1]
        
        # Distance to support/resistance
        dist_to_support = (current_price - support) / current_price if support else 1
        dist_to_resistance = (resistance - current_price) / current_price if resistance else 1
        
        # Generate signal
        signal_type = SignalType.HOLD
        confidence = 0.0
        metadata: Dict[str, Any] = {
            "support": support,
            "resistance": resistance,
            "fib_support": fib_support,
            "fib_resistance": fib_resistance,
            "rsi": current_rsi,
            "bullish_divergence": bullish_div,
            "bearish_divergence": bearish_div,
            "swing_highs": len(swing_highs),
            "swing_lows": len(swing_lows),
        }
        
        # Bullish swing setup
        if bullish_div or (dist_to_support < 0.01 and current_rsi < 40):
            if self._confirm_reversal(data, "bullish"):
                signal_type = SignalType.BUY
                confidence = 0.7 if bullish_div else 0.65
                metadata["signal_reason"] = "bullish_divergence" if bullish_div else "support_bounce"
        
        # Bearish swing setup
        elif bearish_div or (dist_to_resistance < 0.01 and current_rsi > 60):
            if self._confirm_reversal(data, "bearish"):
                signal_type = SignalType.SELL
                confidence = 0.7 if bearish_div else 0.65
                metadata["signal_reason"] = "bearish_divergence" if bearish_div else "resistance_rejection"
        
        # Fibonacci pullback entries
        if signal_type == SignalType.HOLD:
            fib_signal = self._check_fib_pullback(data, fib_support, fib_resistance)
            if fib_signal:
                signal_type = fib_signal[0]
                confidence = fib_signal[1]
                metadata["signal_reason"] = "fibonacci_pullback"
        
        return Signal(
            signal_type=signal_type,
            confidence=confidence,
            price=current_price,
            timestamp=datetime.utcnow(),
            strategy_name=self.name,
            metadata=metadata,
        )
    
    def _find_swing_points(self, data: pd.DataFrame) -> Tuple[List[float], List[float]]:
        """Find swing highs and lows."""
        highs = []
        lows = []
        
        for i in range(self.swing_lookback, len(data) - self.swing_lookback):
            # Swing high
            if data["high"].iloc[i] == data["high"].iloc[i-self.swing_lookback:i+self.swing_lookback+1].max():
                highs.append(data["high"].iloc[i])
            
            # Swing low
            if data["low"].iloc[i] == data["low"].iloc[i-self.swing_lookback:i+self.swing_lookback+1].min():
                lows.append(data["low"].iloc[i])
        
        return highs[-5:], lows[-5:]  # Last 5 swing points
    
    def _calculate_fib_levels(
        self,
        swing_highs: List[float],
        swing_lows: List[float],
    ) -> Tuple[float, float]:
        """Calculate Fibonacci support and resistance."""
        if not swing_highs or not swing_lows:
            return None, None
        
        high = max(swing_highs)
        low = min(swing_lows)
        diff = high - low
        
        # 0.618 level as support, 0.382 as resistance
        fib_support = high - (diff * 0.618)
        fib_resistance = low + (diff * 0.618)
        
        return fib_support, fib_resistance
    
    def _find_support(self, data: pd.DataFrame) -> float:
        """Find nearest support level."""
        lows = data["low"].iloc[-50:]
        
        # Find clusters of lows
        bins = np.linspace(lows.min(), lows.max(), 20)
        hist, _ = np.histogram(lows, bins)
        
        # Most common price zone
        support_idx = np.argmax(hist)
        support = (bins[support_idx] + bins[support_idx + 1]) / 2
        
        return support
    
    def _find_resistance(self, data: pd.DataFrame) -> float:
        """Find nearest resistance level."""
        highs = data["high"].iloc[-50:]
        
        # Find clusters of highs
        bins = np.linspace(highs.min(), highs.max(), 20)
        hist, _ = np.histogram(highs, bins)
        
        # Most common price zone
        resistance_idx = np.argmax(hist)
        resistance = (bins[resistance_idx] + bins[resistance_idx + 1]) / 2
        
        return resistance
    
    def _detect_bullish_divergence(
        self,
        price: pd.Series,
        indicator: pd.Series,
        lookback: int = 14,
    ) -> bool:
        """Detect bullish divergence (lower price lows, higher indicator lows)."""
        price_recent = price.iloc[-lookback:]
        indicator_recent = indicator.iloc[-lookback:]
        
        # Find lowest points
        price_low_idx = price_recent.idxmin()
        
        # Check if price making lower lows but indicator making higher lows
        prev_period = price.iloc[-lookback*2:-lookback]
        
        if len(prev_period) == 0:
            return False
        
        if price_recent.min() < prev_period.min():
            if indicator_recent.min() > indicator.iloc[-lookback*2:-lookback].min():
                return True
        
        return False
    
    def _detect_bearish_divergence(
        self,
        price: pd.Series,
        indicator: pd.Series,
        lookback: int = 14,
    ) -> bool:
        """Detect bearish divergence (higher price highs, lower indicator highs)."""
        price_recent = price.iloc[-lookback:]
        indicator_recent = indicator.iloc[-lookback:]
        
        prev_period = price.iloc[-lookback*2:-lookback]
        
        if len(prev_period) == 0:
            return False
        
        if price_recent.max() > prev_period.max():
            if indicator_recent.max() < indicator.iloc[-lookback*2:-lookback].max():
                return True
        
        return False
    
    def _confirm_reversal(self, data: pd.DataFrame, direction: str) -> bool:
        """Confirm reversal with candlestick patterns."""
        if len(data) < 3:
            return False
        
        last_candle = data.iloc[-1]
        prev_candle = data.iloc[-2]
        
        if direction == "bullish":
            # Bullish engulfing or hammer
            is_bullish_engulfing = (
                prev_candle["close"] < prev_candle["open"] and
                last_candle["close"] > last_candle["open"] and
                last_candle["close"] > prev_candle["open"] and
                last_candle["open"] < prev_candle["close"]
            )
            
            is_hammer = (
                last_candle["close"] > last_candle["open"] and
                (last_candle["open"] - last_candle["low"]) > (last_candle["close"] - last_candle["open"]) * 2
            )
            
            return is_bullish_engulfing or is_hammer
        
        else:
            # Bearish engulfing or shooting star
            is_bearish_engulfing = (
                prev_candle["close"] > prev_candle["open"] and
                last_candle["close"] < last_candle["open"] and
                last_candle["close"] < prev_candle["open"] and
                last_candle["open"] > prev_candle["close"]
            )
            
            is_shooting_star = (
                last_candle["close"] < last_candle["open"] and
                (last_candle["high"] - last_candle["open"]) > (last_candle["open"] - last_candle["close"]) * 2
            )
            
            return is_bearish_engulfing or is_shooting_star
    
    def _check_fib_pullback(
        self,
        data: pd.DataFrame,
        fib_support: float,
        fib_resistance: float,
    ) -> Tuple[SignalType, float]:
        """Check for Fibonacci pullback entry."""
        if not fib_support or not fib_resistance:
            return None
        
        current_price = data["close"].iloc[-1]
        
        # Near Fibonacci support
        if abs(current_price - fib_support) / current_price < 0.003:
            return (SignalType.BUY, 0.6)
        
        # Near Fibonacci resistance
        if abs(current_price - fib_resistance) / current_price < 0.003:
            return (SignalType.SELL, 0.6)
        
        return None
    
    def calculate_stop_loss(
        self,
        price: float,
        signal_type: SignalType,
        data: pd.DataFrame,
    ) -> float:
        """Calculate stop loss based on swing points."""
        if signal_type == SignalType.BUY:
            # Below recent swing low
            swing_low = data["low"].iloc[-self.swing_lookback:].min()
            return swing_low * 0.998
        else:
            # Above recent swing high
            swing_high = data["high"].iloc[-self.swing_lookback:].max()
            return swing_high * 1.002
    
    def calculate_take_profit(
        self,
        price: float,
        signal_type: SignalType,
        data: pd.DataFrame,
    ) -> float:
        """Calculate take profit with 2.5:1 reward/risk for swing trades."""
        stop_loss = self.calculate_stop_loss(price, signal_type, data)
        risk = abs(price - stop_loss)
        
        if signal_type == SignalType.BUY:
            return price + (risk * 2.5)
        else:
            return price - (risk * 2.5)
    
    def _hold_signal(self, data: pd.DataFrame) -> Signal:
        """Return hold signal."""
        return Signal(
            signal_type=SignalType.HOLD,
            confidence=0.0,
            price=data["close"].iloc[-1] if len(data) > 0 else 0,
            timestamp=datetime.utcnow(),
            strategy_name=self.name,
        )
