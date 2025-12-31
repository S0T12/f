"""
XGBoost Ensemble Model
======================
Gradient boosting model for classification.
"""

import logging
from typing import Dict, Any, Optional, Tuple, List
import numpy as np
from pathlib import Path
import joblib

from app.ml_engine.models.base import BaseModel

logger = logging.getLogger(__name__)


class XGBoostModel(BaseModel):
    """XGBoost classification model."""
    
    def __init__(
        self,
        name: str = "xgboost",
        version: str = "1.0",
        n_estimators: int = 100,
        max_depth: int = 6,
        learning_rate: float = 0.1,
        subsample: float = 0.8,
        colsample_bytree: float = 0.8,
    ):
        super().__init__(name, version)
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.learning_rate = learning_rate
        self.subsample = subsample
        self.colsample_bytree = colsample_bytree
        self.feature_importance_: Optional[np.ndarray] = None
    
    def build(self, input_shape: Tuple[int, ...] = None) -> None:
        """Build XGBoost model."""
        try:
            import xgboost as xgb
            
            self.model = xgb.XGBClassifier(
                n_estimators=self.n_estimators,
                max_depth=self.max_depth,
                learning_rate=self.learning_rate,
                subsample=self.subsample,
                colsample_bytree=self.colsample_bytree,
                objective="multi:softprob",
                num_class=3,  # down, neutral, up
                eval_metric="mlogloss",
                use_label_encoder=False,
                random_state=42,
                n_jobs=-1,
            )
            
            logger.info("XGBoost model built")
            
        except ImportError:
            logger.error("XGBoost not available")
            self.model = None
    
    def train(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: Optional[np.ndarray] = None,
        y_val: Optional[np.ndarray] = None,
        **kwargs,
    ) -> Dict[str, float]:
        """Train the XGBoost model."""
        if self.model is None:
            self.build()
        
        try:
            eval_set = [(X_train, y_train)]
            if X_val is not None:
                eval_set.append((X_val, y_val))
            
            self.model.fit(
                X_train, y_train,
                eval_set=eval_set,
                verbose=False,
            )
            
            self.is_trained = True
            self.feature_importance_ = self.model.feature_importances_
            
            # Evaluate
            train_pred = self.model.predict(X_train)
            train_accuracy = (train_pred == y_train).mean()
            
            metrics = {"train_accuracy": train_accuracy}
            
            if X_val is not None:
                val_pred = self.model.predict(X_val)
                val_accuracy = (val_pred == y_val).mean()
                metrics["val_accuracy"] = val_accuracy
            
            return metrics
            
        except Exception as e:
            logger.error(f"Training error: {e}")
            return {}
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Make predictions."""
        if self.model is None or not self.is_trained:
            return np.array([1])  # Default neutral
        
        return self.model.predict(X)
    
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Get prediction probabilities."""
        if self.model is None or not self.is_trained:
            return np.array([[0.33, 0.34, 0.33]])
        
        return self.model.predict_proba(X)
    
    def get_feature_importance(
        self,
        feature_names: List[str],
    ) -> Dict[str, float]:
        """Get feature importance scores."""
        if self.feature_importance_ is None:
            return {}
        
        importance = dict(zip(feature_names, self.feature_importance_))
        return dict(sorted(importance.items(), key=lambda x: x[1], reverse=True))
    
    def save(self, path: Path) -> None:
        """Save XGBoost model."""
        path = Path(path)
        super().save(path)
        
        if self.model:
            joblib.dump(self.model, path / "model.joblib")
            if self.feature_importance_ is not None:
                np.save(path / "feature_importance.npy", self.feature_importance_)
    
    def load(self, path: Path) -> None:
        """Load XGBoost model."""
        path = Path(path)
        super().load(path)
        
        try:
            self.model = joblib.load(path / "model.joblib")
            importance_path = path / "feature_importance.npy"
            if importance_path.exists():
                self.feature_importance_ = np.load(importance_path)
        except Exception as e:
            logger.error(f"Error loading model: {e}")
