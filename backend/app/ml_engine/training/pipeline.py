"""
Training Pipeline
=================
End-to-end training pipeline for ML models.
"""

import logging
from typing import Dict, Any, Optional, List
from datetime import datetime
from pathlib import Path
import numpy as np
import pandas as pd

from app.ml_engine.models.ensemble import EnsembleModel
from app.ml_engine.features.engineer import FeatureEngineer
from app.config import settings

logger = logging.getLogger(__name__)


class TrainingPipeline:
    """End-to-end model training pipeline."""
    
    def __init__(
        self,
        model_name: str = "ensemble",
        model_path: Optional[Path] = None,
    ):
        self.model_name = model_name
        self.model_path = model_path or Path(settings.MODEL_PATH)
        self.feature_engineer = FeatureEngineer()
        self.model = EnsembleModel()
        self.training_history: List[Dict[str, Any]] = []
    
    async def train(
        self,
        data: pd.DataFrame,
        epochs: int = 100,
        batch_size: int = 32,
        sequence_length: int = 60,
        train_ratio: float = 0.8,
    ) -> Dict[str, Any]:
        """Run full training pipeline."""
        logger.info("Starting training pipeline...")
        
        start_time = datetime.utcnow()
        
        # Prepare data
        logger.info("Preparing features...")
        prepared = self.feature_engineer.prepare_data(
            data,
            sequence_length=sequence_length,
            train_ratio=train_ratio,
        )
        
        logger.info(f"Training samples: {len(prepared['X_train'])}")
        logger.info(f"Validation samples: {len(prepared['X_val'])}")
        logger.info(f"Features: {len(prepared['feature_columns'])}")
        
        # Build models
        logger.info("Building models...")
        input_shape = (sequence_length, len(prepared["feature_columns"]))
        self.model.build(input_shape)
        
        # Train ensemble
        logger.info("Training ensemble...")
        
        # Train LSTM with sequences
        lstm_metrics = self.model.models["lstm"].train(
            prepared["X_train_seq"],
            self._to_categorical(prepared["y_train_seq"]),
            prepared["X_val_seq"],
            self._to_categorical(prepared["y_val_seq"]),
            epochs=epochs,
            batch_size=batch_size,
        )
        
        # Train XGBoost with flat features
        xgb_metrics = self.model.models["xgboost"].train(
            prepared["X_train"],
            prepared["y_train"],
            prepared["X_val"],
            prepared["y_val"],
        )
        
        # Optimize ensemble weights
        self.model.optimize_weights(
            prepared["X_val_seq"],
            prepared["y_val_seq"],
        )
        
        # Evaluate final ensemble
        logger.info("Evaluating ensemble...")
        final_metrics = self._evaluate_ensemble(
            prepared["X_val_seq"],
            prepared["y_val_seq"],
        )
        
        # Save models
        version = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
        save_path = self.model_path / f"ensemble_{version}"
        self.model.version = version
        self.model.save(save_path)
        
        # Record training
        training_record = {
            "version": version,
            "trained_at": start_time.isoformat(),
            "duration_seconds": (datetime.utcnow() - start_time).total_seconds(),
            "training_samples": len(prepared["X_train"]),
            "validation_samples": len(prepared["X_val"]),
            "features": len(prepared["feature_columns"]),
            "lstm_metrics": lstm_metrics,
            "xgboost_metrics": xgb_metrics,
            "ensemble_metrics": final_metrics,
            "weights": self.model.weights,
        }
        
        self.training_history.append(training_record)
        logger.info(f"Training complete. Accuracy: {final_metrics.get('accuracy', 0):.4f}")
        
        return training_record
    
    def _to_categorical(self, y: np.ndarray, num_classes: int = 3) -> np.ndarray:
        """Convert labels to one-hot encoding."""
        from tensorflow.keras.utils import to_categorical
        return to_categorical(y, num_classes=num_classes)
    
    def _evaluate_ensemble(
        self,
        X_val: np.ndarray,
        y_val: np.ndarray,
    ) -> Dict[str, float]:
        """Evaluate ensemble performance."""
        predictions = self.model.predict(X_val)
        pred_classes = np.argmax(predictions, axis=1)
        
        accuracy = (pred_classes == y_val).mean()
        
        # Calculate per-class metrics
        from sklearn.metrics import precision_score, recall_score, f1_score
        
        return {
            "accuracy": accuracy,
            "precision": precision_score(y_val, pred_classes, average="weighted", zero_division=0),
            "recall": recall_score(y_val, pred_classes, average="weighted", zero_division=0),
            "f1_score": f1_score(y_val, pred_classes, average="weighted", zero_division=0),
        }
    
    def load_model(self, version: str) -> None:
        """Load a specific model version."""
        model_path = self.model_path / f"ensemble_{version}"
        self.model.load(model_path)
        logger.info(f"Loaded model version {version}")
    
    def load_latest_model(self) -> Optional[str]:
        """Load the most recent model."""
        model_dirs = list(self.model_path.glob("ensemble_*"))
        if not model_dirs:
            logger.warning("No saved models found")
            return None
        
        latest = max(model_dirs, key=lambda x: x.name)
        version = latest.name.replace("ensemble_", "")
        self.load_model(version)
        return version
