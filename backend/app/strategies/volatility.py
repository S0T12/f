"""
Volatility Trading Strategy
============================
Strategy for high-volatility XAU/USD trading.
"""

import logging
from datetime import datetime
from typing import Dict, Any
import pandas as pd
import numpy as np

from app.strategies.base import BaseStrategy, Signal, SignalType
from app.technical_analysis.indicators import volatility, momentum, trend

logger = logging.getLogger(__name__)


class VolatilityStrategy(BaseStrategy):
    """
    Volatility-based trading strategy.
    
    Uses Bollinger Bands, ATR, and Keltner Channels to identify
    breakout opportunities during high volatility periods.
    """
    
    def __init__(
        self,
        atr_multiplier: float = 2.0,
        bb_period: int = 20,
        bb_std: float = 2.0,
        timeframe: str = "M15",
    ):
        super().__init__("Volatility", timeframe)
        self.atr_multiplier = atr_multiplier
        self.bb_period = bb_period
        self.bb_std = bb_std
    
    def analyze(self, data: pd.DataFrame) -> Signal:
        """Analyze for volatility breakout signals."""
        if len(data) < 50:
            return self._hold_signal(data)
        
        close = data["close"]
        high = data["high"]
        low = data["low"]
        current_price = close.iloc[-1]
        
        # Calculate indicators
        bb_upper, bb_middle, bb_lower = volatility.bollinger_bands(
            close, self.bb_period, self.bb_std
        )
        atr = volatility.atr(high, low, close, 14)
        kc_upper, kc_middle, kc_lower = volatility.keltner_channel(
            high, low, close
        )
        
        # Bollinger Band squeeze detection
        bb_width = (bb_upper - bb_lower) / bb_middle
        is_squeeze = bb_width.iloc[-1] < bb_width.rolling(50).mean().iloc[-1]
        
        # Breakout detection
        broke_upper_bb = close.iloc[-1] > bb_upper.iloc[-1]
        broke_lower_bb = close.iloc[-1] < bb_lower.iloc[-1]
        
        # Keltner Channel confirmation
        above_kc = close.iloc[-1] > kc_upper.iloc[-1]
        below_kc = close.iloc[-1] < kc_lower.iloc[-1]
        
        # RSI for confirmation
        rsi = momentum.rsi(close, 14)
        rsi_value = rsi.iloc[-1]
        
        # Volatility expansion
        current_atr = atr.iloc[-1]
        avg_atr = atr.rolling(20).mean().iloc[-1]
        volatility_expanding = current_atr > avg_atr * 1.2
        
        # Generate signal
        signal_type = SignalType.HOLD
        confidence = 0.0
        metadata: Dict[str, Any] = {
            "bb_width": bb_width.iloc[-1],
            "is_squeeze": is_squeeze,
            "atr": current_atr,
            "rsi": rsi_value,
            "volatility_expanding": volatility_expanding,
        }
        
        # Bullish breakout
        if broke_upper_bb and above_kc and volatility_expanding:
            if rsi_value < 80:  # Not too overbought
                signal_type = SignalType.BUY
                confidence = min(0.9, 0.6 + (current_atr / avg_atr - 1) * 0.3)
        
        # Bearish breakout
        elif broke_lower_bb and below_kc and volatility_expanding:
            if rsi_value > 20:  # Not too oversold
                signal_type = SignalType.SELL
                confidence = min(0.9, 0.6 + (current_atr / avg_atr - 1) * 0.3)
        
        # Mean reversion signals during low volatility
        elif not volatility_expanding:
            if close.iloc[-1] < bb_lower.iloc[-1] and rsi_value < 30:
                signal_type = SignalType.BUY
                confidence = 0.65
                metadata["signal_reason"] = "mean_reversion_oversold"
            elif close.iloc[-1] > bb_upper.iloc[-1] and rsi_value > 70:
                signal_type = SignalType.SELL
                confidence = 0.65
                metadata["signal_reason"] = "mean_reversion_overbought"
        
        return Signal(
            signal_type=signal_type,
            confidence=confidence,
            price=current_price,
            timestamp=datetime.utcnow(),
            strategy_name=self.name,
            metadata=metadata,
        )
    
    def calculate_stop_loss(
        self,
        price: float,
        signal_type: SignalType,
        data: pd.DataFrame,
    ) -> float:
        """Calculate ATR-based stop loss."""
        atr = volatility.atr(data["high"], data["low"], data["close"], 14)
        atr_value = atr.iloc[-1]
        
        if signal_type == SignalType.BUY:
            return price - (atr_value * self.atr_multiplier)
        else:
            return price + (atr_value * self.atr_multiplier)
    
    def calculate_take_profit(
        self,
        price: float,
        signal_type: SignalType,
        data: pd.DataFrame,
    ) -> float:
        """Calculate take profit with 2:1 reward/risk ratio."""
        stop_loss = self.calculate_stop_loss(price, signal_type, data)
        risk = abs(price - stop_loss)
        
        if signal_type == SignalType.BUY:
            return price + (risk * 2)
        else:
            return price - (risk * 2)
    
    def _hold_signal(self, data: pd.DataFrame) -> Signal:
        """Return hold signal with current price."""
        return Signal(
            signal_type=SignalType.HOLD,
            confidence=0.0,
            price=data["close"].iloc[-1] if len(data) > 0 else 0,
            timestamp=datetime.utcnow(),
            strategy_name=self.name,
        )
