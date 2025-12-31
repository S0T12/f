"""
ML-Based Trading Strategy
=========================
Strategy that uses ML models for signal generation.
"""

import logging
from datetime import datetime
from typing import Dict, Any, Optional
import pandas as pd
import numpy as np

from app.strategies.base import BaseStrategy, Signal, SignalType
from app.ml_engine.models.ensemble import EnsembleModel
from app.ml_engine.features.engineer import FeatureEngineer
from app.technical_analysis.indicators import volatility

logger = logging.getLogger(__name__)


class MLStrategy(BaseStrategy):
    """
    ML-based trading strategy using ensemble model predictions.
    
    Combines technical analysis with ML predictions for
    high-confidence signal generation.
    """
    
    def __init__(
        self,
        model: Optional[EnsembleModel] = None,
        confidence_threshold: float = 0.7,
        sequence_length: int = 60,
        timeframe: str = "H1",
    ):
        super().__init__("ML", timeframe)
        self.model = model
        self.feature_engineer = FeatureEngineer()
        self.confidence_threshold = confidence_threshold
        self.sequence_length = sequence_length
        self._model_loaded = False
    
    def load_model(self, model_path: str) -> None:
        """Load pre-trained model."""
        if self.model is None:
            self.model = EnsembleModel()
        
        self.model.load(model_path)
        self._model_loaded = True
        logger.info(f"ML Strategy loaded model from {model_path}")
    
    def analyze(self, data: pd.DataFrame) -> Signal:
        """Analyze using ML model predictions."""
        if not self._model_loaded or self.model is None:
            logger.warning("ML model not loaded, returning hold signal")
            return self._hold_signal(data)
        
        if len(data) < self.sequence_length + 50:
            return self._hold_signal(data)
        
        current_price = data["close"].iloc[-1]
        
        try:
            # Prepare features
            features_df = self.feature_engineer.create_features(data)
            feature_values = features_df[self.feature_engineer.feature_columns].values
            
            # Scale features
            X_scaled = self.feature_engineer.transform(feature_values)
            
            # Create sequence for LSTM
            if len(X_scaled) >= self.sequence_length:
                X_seq = X_scaled[-self.sequence_length:].reshape(1, self.sequence_length, -1)
            else:
                return self._hold_signal(data)
            
            # Get prediction
            result = self.model.predict(X_seq)
            
            # Interpret prediction
            # result shape: (1, 3) for [down, neutral, up]
            prediction = result[0]
            predicted_class = np.argmax(prediction)
            confidence = float(prediction[predicted_class])
            
            # Map to signal
            class_to_signal = {
                0: SignalType.SELL,
                1: SignalType.HOLD,
                2: SignalType.BUY,
            }
            
            signal_type = class_to_signal[predicted_class]
            
            # Only signal if confidence above threshold
            if confidence < self.confidence_threshold:
                signal_type = SignalType.HOLD
            
            # Get additional metadata from feature importance
            metadata: Dict[str, Any] = {
                "ml_confidence": confidence,
                "ml_prediction": predicted_class,
                "probabilities": {
                    "down": float(prediction[0]),
                    "neutral": float(prediction[1]),
                    "up": float(prediction[2]),
                },
                "model_version": self.model.version,
            }
            
            # Add key technical indicators for context
            metadata["rsi"] = features_df["rsi_14"].iloc[-1]
            metadata["macd"] = features_df["macd"].iloc[-1]
            metadata["bb_percent"] = features_df["bb_percent"].iloc[-1]
            
            return Signal(
                signal_type=signal_type,
                confidence=confidence,
                price=current_price,
                timestamp=datetime.utcnow(),
                strategy_name=self.name,
                metadata=metadata,
            )
        
        except Exception as e:
            logger.error(f"ML prediction error: {e}")
            return self._hold_signal(data)
    
    def calculate_stop_loss(
        self,
        price: float,
        signal_type: SignalType,
        data: pd.DataFrame,
    ) -> float:
        """Calculate ATR-based stop loss."""
        atr = volatility.atr(data["high"], data["low"], data["close"], 14)
        atr_value = atr.iloc[-1]
        
        # Dynamic multiplier based on volatility
        avg_atr = atr.iloc[-50:].mean()
        multiplier = 2.0 if atr_value <= avg_atr else 2.5
        
        if signal_type == SignalType.BUY:
            return price - (atr_value * multiplier)
        else:
            return price + (atr_value * multiplier)
    
    def calculate_take_profit(
        self,
        price: float,
        signal_type: SignalType,
        data: pd.DataFrame,
    ) -> float:
        """Calculate take profit with dynamic R:R ratio."""
        stop_loss = self.calculate_stop_loss(price, signal_type, data)
        risk = abs(price - stop_loss)
        
        # Higher confidence = higher R:R target
        rr_ratio = 2.0  # Default
        
        if hasattr(self, "_last_confidence"):
            if self._last_confidence > 0.8:
                rr_ratio = 3.0
            elif self._last_confidence > 0.7:
                rr_ratio = 2.5
        
        if signal_type == SignalType.BUY:
            return price + (risk * rr_ratio)
        else:
            return price - (risk * rr_ratio)
    
    def _hold_signal(self, data: pd.DataFrame) -> Signal:
        """Return hold signal."""
        return Signal(
            signal_type=SignalType.HOLD,
            confidence=0.0,
            price=data["close"].iloc[-1] if len(data) > 0 else 0,
            timestamp=datetime.utcnow(),
            strategy_name=self.name,
        )


class StrategyOrchestrator:
    """
    Orchestrates multiple strategies and combines their signals.
    
    Weights strategy signals based on their historical performance.
    """
    
    def __init__(self):
        from app.strategies import VolatilityStrategy, TrendFollowingStrategy, SwingStrategy
        
        self.strategies = {
            "volatility": VolatilityStrategy(),
            "trend": TrendFollowingStrategy(),
            "swing": SwingStrategy(),
            "ml": MLStrategy(),
        }
        
        self.weights = {
            "volatility": 0.2,
            "trend": 0.25,
            "swing": 0.15,
            "ml": 0.4,
        }
    
    def analyze_all(self, data: pd.DataFrame) -> Dict[str, Signal]:
        """Get signals from all strategies."""
        signals = {}
        
        for name, strategy in self.strategies.items():
            try:
                signals[name] = strategy.generate_signal(data)
            except Exception as e:
                logger.error(f"Strategy {name} error: {e}")
        
        return signals
    
    def get_consensus_signal(self, data: pd.DataFrame) -> Signal:
        """Get weighted consensus signal from all strategies."""
        signals = self.analyze_all(data)
        
        if not signals:
            return Signal(
                signal_type=SignalType.HOLD,
                confidence=0.0,
                price=data["close"].iloc[-1],
                timestamp=datetime.utcnow(),
                strategy_name="Consensus",
            )
        
        # Weight votes
        buy_score = 0.0
        sell_score = 0.0
        
        for name, signal in signals.items():
            weight = self.weights.get(name, 0.2)
            
            if signal.signal_type == SignalType.BUY:
                buy_score += weight * signal.confidence
            elif signal.signal_type == SignalType.SELL:
                sell_score += weight * signal.confidence
        
        # Determine consensus
        if buy_score > sell_score and buy_score > 0.5:
            signal_type = SignalType.BUY
            confidence = buy_score
        elif sell_score > buy_score and sell_score > 0.5:
            signal_type = SignalType.SELL
            confidence = sell_score
        else:
            signal_type = SignalType.HOLD
            confidence = max(buy_score, sell_score)
        
        # Calculate average SL/TP from actionable signals
        actionable = [s for s in signals.values() if s.signal_type == signal_type]
        
        stop_loss = None
        take_profit = None
        
        if actionable:
            if all(s.stop_loss for s in actionable):
                stop_loss = np.mean([s.stop_loss for s in actionable])
            if all(s.take_profit for s in actionable):
                take_profit = np.mean([s.take_profit for s in actionable])
        
        return Signal(
            signal_type=signal_type,
            confidence=confidence,
            price=data["close"].iloc[-1],
            timestamp=datetime.utcnow(),
            strategy_name="Consensus",
            stop_loss=stop_loss,
            take_profit=take_profit,
            metadata={
                "buy_score": buy_score,
                "sell_score": sell_score,
                "individual_signals": {k: v.to_dict() for k, v in signals.items()},
            },
        )
    
    def update_weights(self, performance: Dict[str, float]) -> None:
        """Update strategy weights based on performance."""
        total = sum(performance.values())
        if total > 0:
            self.weights = {k: v / total for k, v in performance.items()}
