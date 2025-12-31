"""
Feature Engineering Pipeline
=============================
Create features from market data for ML models.
"""

import logging
from typing import List, Optional, Dict, Tuple
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, MinMaxScaler

from app.technical_analysis.indicators import trend, momentum, volatility, volume

logger = logging.getLogger(__name__)


class FeatureEngineer:
    """Feature engineering for trading ML models."""
    
    def __init__(self, scaler_type: str = "standard"):
        self.scaler_type = scaler_type
        self.scaler = None
        self.feature_columns: List[str] = []
    
    def create_features(
        self,
        df: pd.DataFrame,
        include_lagged: bool = True,
        lag_periods: List[int] = [1, 2, 3, 5, 10],
    ) -> pd.DataFrame:
        """Create all features from OHLCV data."""
        features = df.copy()
        
        # Basic price features
        features["returns"] = features["close"].pct_change()
        features["log_returns"] = np.log(features["close"] / features["close"].shift(1))
        features["hl_range"] = (features["high"] - features["low"]) / features["close"]
        features["co_range"] = (features["close"] - features["open"]) / features["open"]
        
        # Trend indicators
        for period in [10, 20, 50, 100, 200]:
            features[f"sma_{period}"] = trend.sma(features["close"], period)
            features[f"ema_{period}"] = trend.ema(features["close"], period)
            features[f"price_sma_{period}_ratio"] = features["close"] / features[f"sma_{period}"]
        
        # MACD
        macd_line, signal_line, histogram = trend.macd(features["close"])
        features["macd"] = macd_line
        features["macd_signal"] = signal_line
        features["macd_histogram"] = histogram
        
        # ADX
        adx_val, plus_di, minus_di = trend.adx(features["high"], features["low"], features["close"])
        features["adx"] = adx_val
        features["plus_di"] = plus_di
        features["minus_di"] = minus_di
        
        # Momentum indicators
        for period in [7, 14, 21]:
            features[f"rsi_{period}"] = momentum.rsi(features["close"], period)
        
        stoch_k, stoch_d = momentum.stochastic(features["high"], features["low"], features["close"])
        features["stoch_k"] = stoch_k
        features["stoch_d"] = stoch_d
        
        features["cci"] = momentum.cci(features["high"], features["low"], features["close"])
        features["williams_r"] = momentum.williams_r(features["high"], features["low"], features["close"])
        features["roc"] = momentum.roc(features["close"], 12)
        
        if "volume" in features.columns:
            features["mfi"] = momentum.mfi(
                features["high"], features["low"],
                features["close"], features["volume"]
            )
        
        # Volatility indicators
        for period in [14, 21]:
            features[f"atr_{period}"] = volatility.atr(
                features["high"], features["low"], features["close"], period
            )
            features[f"natr_{period}"] = volatility.natr(
                features["high"], features["low"], features["close"], period
            )
        
        bb_upper, bb_middle, bb_lower = volatility.bollinger_bands(features["close"])
        features["bb_upper"] = bb_upper
        features["bb_middle"] = bb_middle
        features["bb_lower"] = bb_lower
        features["bb_width"] = (bb_upper - bb_lower) / bb_middle
        features["bb_percent"] = (features["close"] - bb_lower) / (bb_upper - bb_lower)
        
        features["historical_vol"] = volatility.historical_volatility(features["close"])
        
        # Volume indicators
        if "volume" in features.columns:
            features["obv"] = volume.obv(features["close"], features["volume"])
            features["vwap"] = volume.vwap(
                features["high"], features["low"],
                features["close"], features["volume"]
            )
            features["cmf"] = volume.chaikin_money_flow(
                features["high"], features["low"],
                features["close"], features["volume"]
            )
        
        # Time features
        if isinstance(features.index, pd.DatetimeIndex):
            features["hour"] = features.index.hour
            features["dayofweek"] = features.index.dayofweek
            features["month"] = features.index.month
            
            # Trading sessions (simplified)
            features["is_asian"] = ((features["hour"] >= 0) & (features["hour"] < 8)).astype(int)
            features["is_london"] = ((features["hour"] >= 8) & (features["hour"] < 16)).astype(int)
            features["is_ny"] = ((features["hour"] >= 13) & (features["hour"] < 22)).astype(int)
        
        # Lagged features
        if include_lagged:
            for col in ["returns", "rsi_14", "macd", "atr_14"]:
                if col in features.columns:
                    for lag in lag_periods:
                        features[f"{col}_lag_{lag}"] = features[col].shift(lag)
        
        # Drop NaN rows
        features = features.dropna()
        
        self.feature_columns = [
            col for col in features.columns
            if col not in ["open", "high", "low", "close", "volume", "timestamp"]
        ]
        
        return features
    
    def create_labels(
        self,
        df: pd.DataFrame,
        prediction_horizon: int = 1,
        threshold: float = 0.001,
    ) -> pd.Series:
        """Create classification labels."""
        future_returns = df["close"].shift(-prediction_horizon) / df["close"] - 1
        
        # 0: down, 1: neutral, 2: up
        labels = pd.Series(1, index=df.index)  # Default neutral
        labels[future_returns > threshold] = 2   # Up
        labels[future_returns < -threshold] = 0  # Down
        
        return labels
    
    def fit_scaler(self, X: np.ndarray) -> None:
        """Fit the scaler on training data."""
        if self.scaler_type == "standard":
            self.scaler = StandardScaler()
        else:
            self.scaler = MinMaxScaler()
        
        self.scaler.fit(X)
    
    def transform(self, X: np.ndarray) -> np.ndarray:
        """Transform features using fitted scaler."""
        if self.scaler is None:
            self.fit_scaler(X)
        return self.scaler.transform(X)
    
    def fit_transform(self, X: np.ndarray) -> np.ndarray:
        """Fit and transform features."""
        self.fit_scaler(X)
        return self.transform(X)
    
    def create_sequences(
        self,
        X: np.ndarray,
        y: np.ndarray,
        sequence_length: int = 60,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Create sequences for LSTM/Transformer models."""
        sequences = []
        labels = []
        
        for i in range(sequence_length, len(X)):
            sequences.append(X[i - sequence_length:i])
            labels.append(y[i])
        
        return np.array(sequences), np.array(labels)
    
    def prepare_data(
        self,
        df: pd.DataFrame,
        sequence_length: int = 60,
        train_ratio: float = 0.8,
        prediction_horizon: int = 1,
    ) -> Dict[str, np.ndarray]:
        """Prepare data for training."""
        # Create features
        features_df = self.create_features(df)
        labels = self.create_labels(features_df, prediction_horizon)
        
        # Get feature matrix
        X = features_df[self.feature_columns].values
        y = labels.values
        
        # Remove last rows where we don't have labels
        X = X[:-prediction_horizon]
        y = y[:-prediction_horizon]
        
        # Train/validation split
        split_idx = int(len(X) * train_ratio)
        
        X_train, X_val = X[:split_idx], X[split_idx:]
        y_train, y_val = y[:split_idx], y[split_idx:]
        
        # Scale features
        X_train_scaled = self.fit_transform(X_train)
        X_val_scaled = self.transform(X_val)
        
        # Create sequences for LSTM
        X_train_seq, y_train_seq = self.create_sequences(X_train_scaled, y_train, sequence_length)
        X_val_seq, y_val_seq = self.create_sequences(X_val_scaled, y_val, sequence_length)
        
        return {
            "X_train": X_train_scaled,
            "X_val": X_val_scaled,
            "y_train": y_train,
            "y_val": y_val,
            "X_train_seq": X_train_seq,
            "X_val_seq": X_val_seq,
            "y_train_seq": y_train_seq,
            "y_val_seq": y_val_seq,
            "feature_columns": self.feature_columns,
        }
