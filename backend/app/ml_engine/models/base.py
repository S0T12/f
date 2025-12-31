"""
Base Model Interface
====================
Abstract base class for all ML models.
"""

from abc import ABC, abstractmethod
from typing import Dict, Any, Optional, List, Tuple
from dataclasses import dataclass
import numpy as np
import pandas as pd
import logging
from pathlib import Path

logger = logging.getLogger(__name__)


@dataclass
class PredictionResult:
    """Model prediction result."""
    direction: str        # up, down, neutral
    confidence: float     # 0 to 1
    magnitude: float      # Expected price change
    probabilities: Dict[str, float]  # up, down, neutral probabilities
    features_used: List[str]
    model_version: str


class BaseModel(ABC):
    """Abstract base class for prediction models."""
    
    def __init__(self, name: str, version: str = "1.0"):
        self.name = name
        self.version = version
        self.is_trained = False
        self.model = None
        self.scaler = None
        self.feature_columns: List[str] = []
        self.metrics: Dict[str, float] = {}
    
    @abstractmethod
    def build(self, input_shape: Tuple[int, ...]) -> None:
        """Build the model architecture."""
        pass
    
    @abstractmethod
    def train(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: Optional[np.ndarray] = None,
        y_val: Optional[np.ndarray] = None,
        **kwargs,
    ) -> Dict[str, float]:
        """Train the model."""
        pass
    
    @abstractmethod
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Make predictions."""
        pass
    
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Predict probabilities (for classification models)."""
        return self.predict(X)
    
    def get_prediction_result(
        self,
        X: np.ndarray,
        feature_names: List[str],
    ) -> PredictionResult:
        """Get structured prediction result."""
        proba = self.predict_proba(X)
        
        if len(proba.shape) > 1 and proba.shape[1] == 3:
            # Multi-class output [down, neutral, up]
            prob_down, prob_neutral, prob_up = proba[0]
        else:
            # Binary or single output
            prob_up = float(proba[0])
            prob_down = 1 - prob_up
            prob_neutral = 0.0
        
        # Determine direction
        if prob_up > prob_down and prob_up > prob_neutral:
            direction = "up"
            confidence = prob_up
        elif prob_down > prob_up and prob_down > prob_neutral:
            direction = "down"
            confidence = prob_down
        else:
            direction = "neutral"
            confidence = prob_neutral
        
        return PredictionResult(
            direction=direction,
            confidence=confidence,
            magnitude=0.0,  # Calculated separately
            probabilities={"up": prob_up, "down": prob_down, "neutral": prob_neutral},
            features_used=feature_names,
            model_version=f"{self.name}_{self.version}",
        )
    
    def save(self, path: Path) -> None:
        """Save model to disk."""
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)
        
        # Save metadata
        import json
        metadata = {
            "name": self.name,
            "version": self.version,
            "feature_columns": self.feature_columns,
            "metrics": self.metrics,
        }
        with open(path / "metadata.json", "w") as f:
            json.dump(metadata, f)
        
        logger.info(f"Model saved to {path}")
    
    def load(self, path: Path) -> None:
        """Load model from disk."""
        path = Path(path)
        
        # Load metadata
        import json
        with open(path / "metadata.json", "r") as f:
            metadata = json.load(f)
        
        self.name = metadata["name"]
        self.version = metadata["version"]
        self.feature_columns = metadata["feature_columns"]
        self.metrics = metadata["metrics"]
        
        self.is_trained = True
        logger.info(f"Model loaded from {path}")
    
    def evaluate(
        self,
        X_test: np.ndarray,
        y_test: np.ndarray,
    ) -> Dict[str, float]:
        """Evaluate model performance."""
        from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
        
        predictions = self.predict(X_test)
        
        # For regression, convert to classification
        if predictions.dtype == float:
            pred_classes = np.where(predictions > 0.5, 1, 0)
            true_classes = np.where(y_test > 0.5, 1, 0)
        else:
            pred_classes = predictions
            true_classes = y_test
        
        metrics = {
            "accuracy": accuracy_score(true_classes, pred_classes),
            "precision": precision_score(true_classes, pred_classes, average="weighted", zero_division=0),
            "recall": recall_score(true_classes, pred_classes, average="weighted", zero_division=0),
            "f1_score": f1_score(true_classes, pred_classes, average="weighted", zero_division=0),
        }
        
        self.metrics = metrics
        return metrics
