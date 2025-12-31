"""
Base Strategy
=============
Abstract base class for all trading strategies.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Dict, Any, Optional, List
import pandas as pd
import numpy as np


class SignalType(Enum):
    """Trading signal types."""
    BUY = "buy"
    SELL = "sell"
    HOLD = "hold"


@dataclass
class Signal:
    """Trading signal with metadata."""
    signal_type: SignalType
    confidence: float
    price: float
    timestamp: datetime
    strategy_name: str
    stop_loss: Optional[float] = None
    take_profit: Optional[float] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    @property
    def is_actionable(self) -> bool:
        """Check if signal should trigger a trade."""
        return self.signal_type != SignalType.HOLD and self.confidence >= 0.6
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "signal_type": self.signal_type.value,
            "confidence": self.confidence,
            "price": self.price,
            "timestamp": self.timestamp.isoformat(),
            "strategy_name": self.strategy_name,
            "stop_loss": self.stop_loss,
            "take_profit": self.take_profit,
            "metadata": self.metadata,
        }


class BaseStrategy(ABC):
    """Abstract base class for trading strategies."""
    
    def __init__(self, name: str, timeframe: str = "H1"):
        self.name = name
        self.timeframe = timeframe
        self.is_active = True
        self.signals_history: List[Signal] = []
        self.performance_metrics: Dict[str, float] = {}
    
    @abstractmethod
    def analyze(self, data: pd.DataFrame) -> Signal:
        """Analyze market data and generate signal."""
        pass
    
    @abstractmethod
    def calculate_stop_loss(self, price: float, signal_type: SignalType, data: pd.DataFrame) -> float:
        """Calculate stop loss level."""
        pass
    
    @abstractmethod
    def calculate_take_profit(self, price: float, signal_type: SignalType, data: pd.DataFrame) -> float:
        """Calculate take profit level."""
        pass
    
    def generate_signal(
        self,
        data: pd.DataFrame,
        min_confidence: float = 0.6,
    ) -> Optional[Signal]:
        """Generate trading signal with risk levels."""
        signal = self.analyze(data)
        
        if signal.confidence < min_confidence:
            signal.signal_type = SignalType.HOLD
        
        if signal.signal_type != SignalType.HOLD:
            signal.stop_loss = self.calculate_stop_loss(
                signal.price, signal.signal_type, data
            )
            signal.take_profit = self.calculate_take_profit(
                signal.price, signal.signal_type, data
            )
        
        self.signals_history.append(signal)
        return signal
    
    def calculate_risk_reward_ratio(self, signal: Signal) -> float:
        """Calculate risk/reward ratio."""
        if not signal.stop_loss or not signal.take_profit:
            return 0.0
        
        risk = abs(signal.price - signal.stop_loss)
        reward = abs(signal.take_profit - signal.price)
        
        return reward / risk if risk > 0 else 0.0
    
    def update_performance(self, result: Dict[str, float]) -> None:
        """Update strategy performance metrics."""
        self.performance_metrics.update(result)
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get strategy statistics."""
        if not self.signals_history:
            return {"total_signals": 0}
        
        signals = self.signals_history
        actionable = [s for s in signals if s.is_actionable]
        
        return {
            "total_signals": len(signals),
            "actionable_signals": len(actionable),
            "buy_signals": len([s for s in actionable if s.signal_type == SignalType.BUY]),
            "sell_signals": len([s for s in actionable if s.signal_type == SignalType.SELL]),
            "avg_confidence": np.mean([s.confidence for s in actionable]) if actionable else 0,
            "performance": self.performance_metrics,
        }
