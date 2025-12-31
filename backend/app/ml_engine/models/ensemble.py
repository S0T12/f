"""
Ensemble Model
==============
Combines multiple models for robust predictions.
"""

import logging
from typing import Dict, Any, Optional, List
import numpy as np
from pathlib import Path

from app.ml_engine.models.base import BaseModel, PredictionResult
from app.ml_engine.models.lstm import LSTMModel
from app.ml_engine.models.xgboost_model import XGBoostModel

logger = logging.getLogger(__name__)


class EnsembleModel(BaseModel):
    """Ensemble of multiple models."""
    
    def __init__(
        self,
        name: str = "ensemble",
        version: str = "1.0",
        weights: Optional[Dict[str, float]] = None,
    ):
        super().__init__(name, version)
        self.models: Dict[str, BaseModel] = {}
        self.weights = weights or {}
        self._initialize_models()
    
    def _initialize_models(self) -> None:
        """Initialize component models."""
        self.models = {
            "lstm": LSTMModel(),
            "xgboost": XGBoostModel(),
        }
        
        # Default equal weights
        if not self.weights:
            self.weights = {name: 1.0 / len(self.models) for name in self.models}
    
    def build(self, input_shape=None) -> None:
        """Build all component models."""
        for name, model in self.models.items():
            try:
                model.build(input_shape)
                logger.info(f"Built {name} model")
            except Exception as e:
                logger.error(f"Error building {name}: {e}")
    
    def train(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: Optional[np.ndarray] = None,
        y_val: Optional[np.ndarray] = None,
        **kwargs,
    ) -> Dict[str, float]:
        """Train all component models."""
        all_metrics = {}
        
        for name, model in self.models.items():
            logger.info(f"Training {name}...")
            try:
                metrics = model.train(X_train, y_train, X_val, y_val, **kwargs)
                all_metrics[name] = metrics
            except Exception as e:
                logger.error(f"Error training {name}: {e}")
        
        self.is_trained = True
        return all_metrics
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Make weighted ensemble prediction."""
        predictions = []
        weights = []
        
        for name, model in self.models.items():
            if model.is_trained:
                try:
                    pred = model.predict_proba(X)
                    predictions.append(pred)
                    weights.append(self.weights.get(name, 1.0))
                except Exception as e:
                    logger.error(f"Prediction error for {name}: {e}")
        
        if not predictions:
            return np.array([[0.33, 0.34, 0.33]])
        
        # Weighted average
        weights = np.array(weights) / sum(weights)
        ensemble_pred = sum(w * p for w, p in zip(weights, predictions))
        
        return ensemble_pred
    
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Get ensemble prediction probabilities."""
        return self.predict(X)
    
    def get_prediction_result(
        self,
        X: np.ndarray,
        feature_names: List[str],
    ) -> PredictionResult:
        """Get ensemble prediction with individual model contributions."""
        result = super().get_prediction_result(X, feature_names)
        
        # Add individual model predictions to metadata
        individual_preds = {}
        for name, model in self.models.items():
            if model.is_trained:
                try:
                    pred = model.predict_proba(X)
                    individual_preds[name] = {
                        "up": float(pred[0][2]) if len(pred[0]) > 2 else float(pred[0][0]),
                        "down": float(pred[0][0]) if len(pred[0]) > 2 else float(1 - pred[0][0]),
                    }
                except:
                    pass
        
        return result
    
    def update_weights(self, new_weights: Dict[str, float]) -> None:
        """Update model weights based on performance."""
        # Normalize weights
        total = sum(new_weights.values())
        self.weights = {k: v / total for k, v in new_weights.items()}
        logger.info(f"Updated ensemble weights: {self.weights}")
    
    def optimize_weights(
        self,
        X_val: np.ndarray,
        y_val: np.ndarray,
    ) -> Dict[str, float]:
        """Optimize weights based on validation performance."""
        from scipy.optimize import minimize
        
        model_preds = {}
        for name, model in self.models.items():
            if model.is_trained:
                model_preds[name] = model.predict_proba(X_val)
        
        if len(model_preds) < 2:
            return self.weights
        
        def objective(weights):
            weights = np.abs(weights) / np.sum(np.abs(weights))
            ensemble = sum(w * p for w, p in zip(weights, model_preds.values()))
            pred_classes = np.argmax(ensemble, axis=1)
            accuracy = (pred_classes == y_val).mean()
            return -accuracy  # Minimize negative accuracy
        
        n_models = len(model_preds)
        initial_weights = np.ones(n_models) / n_models
        
        result = minimize(objective, initial_weights, method="Nelder-Mead")
        
        if result.success:
            optimal_weights = np.abs(result.x) / np.sum(np.abs(result.x))
            self.weights = dict(zip(model_preds.keys(), optimal_weights))
        
        return self.weights
    
    def save(self, path: Path) -> None:
        """Save ensemble and all component models."""
        path = Path(path)
        super().save(path)
        
        for name, model in self.models.items():
            model_path = path / name
            model.save(model_path)
        
        import json
        with open(path / "weights.json", "w") as f:
            json.dump(self.weights, f)
    
    def load(self, path: Path) -> None:
        """Load ensemble and all component models."""
        path = Path(path)
        super().load(path)
        
        for name, model in self.models.items():
            model_path = path / name
            if model_path.exists():
                model.load(model_path)
        
        import json
        weights_path = path / "weights.json"
        if weights_path.exists():
            with open(weights_path, "r") as f:
                self.weights = json.load(f)
