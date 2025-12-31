"""
LSTM Model
==========
Long Short-Term Memory network for time series prediction.
"""

import logging
from typing import Dict, Any, Optional, Tuple, List
import numpy as np
from pathlib import Path

from app.ml_engine.models.base import BaseModel

logger = logging.getLogger(__name__)


class LSTMModel(BaseModel):
    """LSTM-based prediction model."""
    
    def __init__(
        self,
        name: str = "lstm",
        version: str = "1.0",
        units: int = 64,
        layers: int = 2,
        dropout: float = 0.2,
        sequence_length: int = 60,
    ):
        super().__init__(name, version)
        self.units = units
        self.layers = layers
        self.dropout = dropout
        self.sequence_length = sequence_length
        self.history = None
    
    def build(self, input_shape: Tuple[int, ...]) -> None:
        """Build LSTM architecture."""
        try:
            import tensorflow as tf
            from tensorflow.keras.models import Sequential
            from tensorflow.keras.layers import LSTM, Dense, Dropout, BatchNormalization
            
            model = Sequential()
            
            # First LSTM layer
            model.add(LSTM(
                self.units,
                return_sequences=(self.layers > 1),
                input_shape=input_shape,
            ))
            model.add(BatchNormalization())
            model.add(Dropout(self.dropout))
            
            # Additional LSTM layers
            for i in range(1, self.layers):
                return_seq = i < self.layers - 1
                model.add(LSTM(self.units, return_sequences=return_seq))
                model.add(BatchNormalization())
                model.add(Dropout(self.dropout))
            
            # Dense layers
            model.add(Dense(32, activation="relu"))
            model.add(Dropout(self.dropout / 2))
            
            # Output layer (3 classes: down, neutral, up)
            model.add(Dense(3, activation="softmax"))
            
            model.compile(
                optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
                loss="categorical_crossentropy",
                metrics=["accuracy"],
            )
            
            self.model = model
            logger.info(f"LSTM model built with shape {input_shape}")
            
        except ImportError:
            logger.error("TensorFlow not available, using mock model")
            self.model = None
    
    def train(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: Optional[np.ndarray] = None,
        y_val: Optional[np.ndarray] = None,
        epochs: int = 100,
        batch_size: int = 32,
        **kwargs,
    ) -> Dict[str, float]:
        """Train the LSTM model."""
        if self.model is None:
            logger.error("Model not built")
            return {}
        
        try:
            from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
            
            callbacks = [
                EarlyStopping(
                    monitor="val_loss",
                    patience=10,
                    restore_best_weights=True,
                ),
                ReduceLROnPlateau(
                    monitor="val_loss",
                    factor=0.5,
                    patience=5,
                    min_lr=1e-6,
                ),
            ]
            
            validation_data = (X_val, y_val) if X_val is not None else None
            
            self.history = self.model.fit(
                X_train, y_train,
                validation_data=validation_data,
                epochs=epochs,
                batch_size=batch_size,
                callbacks=callbacks,
                verbose=1,
            )
            
            self.is_trained = True
            
            # Return final metrics
            return {
                "loss": self.history.history["loss"][-1],
                "accuracy": self.history.history["accuracy"][-1],
                "val_loss": self.history.history.get("val_loss", [0])[-1],
                "val_accuracy": self.history.history.get("val_accuracy", [0])[-1],
            }
            
        except Exception as e:
            logger.error(f"Training error: {e}")
            return {}
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Make predictions."""
        if self.model is None:
            return np.array([[0.33, 0.34, 0.33]])  # Default neutral
        
        return self.model.predict(X, verbose=0)
    
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Get prediction probabilities."""
        return self.predict(X)
    
    def save(self, path: Path) -> None:
        """Save LSTM model."""
        path = Path(path)
        super().save(path)
        
        if self.model:
            self.model.save(path / "model.keras")
    
    def load(self, path: Path) -> None:
        """Load LSTM model."""
        path = Path(path)
        super().load(path)
        
        try:
            import tensorflow as tf
            self.model = tf.keras.models.load_model(path / "model.keras")
        except Exception as e:
            logger.error(f"Error loading model: {e}")
