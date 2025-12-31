"""
Self-Improvement System
=======================
Automated model monitoring, drift detection, and retraining.
"""

import logging
from typing import Dict, Any, Optional, List
from datetime import datetime, timedelta
from dataclasses import dataclass
import numpy as np
from collections import deque

from app.config import settings

logger = logging.getLogger(__name__)


@dataclass
class PerformanceMetric:
    """Performance metric with timestamp."""
    timestamp: datetime
    accuracy: float
    predictions: int
    correct: int


class SelfImprovementSystem:
    """Automated model improvement system."""
    
    def __init__(
        self,
        accuracy_threshold: float = 0.75,
        drift_window: int = 100,
        retrain_cooldown_hours: int = 24,
    ):
        self.accuracy_threshold = accuracy_threshold
        self.drift_window = drift_window
        self.retrain_cooldown = timedelta(hours=retrain_cooldown_hours)
        
        self.performance_history: deque = deque(maxlen=1000)
        self.last_retrain: Optional[datetime] = None
        self.drift_detected = False
    
    def record_prediction(
        self,
        predicted: str,
        actual: str,
        confidence: float,
    ) -> None:
        """Record a prediction outcome."""
        is_correct = predicted == actual
        
        metric = PerformanceMetric(
            timestamp=datetime.utcnow(),
            accuracy=1.0 if is_correct else 0.0,
            predictions=1,
            correct=1 if is_correct else 0,
        )
        
        self.performance_history.append(metric)
    
    def get_current_accuracy(self, window: int = None) -> float:
        """Calculate current accuracy over recent predictions."""
        window = window or self.drift_window
        recent = list(self.performance_history)[-window:]
        
        if not recent:
            return 1.0  # Assume good if no data
        
        correct = sum(m.correct for m in recent)
        total = sum(m.predictions for m in recent)
        
        return correct / total if total > 0 else 0.0
    
    def check_drift(self) -> Dict[str, Any]:
        """Check for model performance drift."""
        current_accuracy = self.get_current_accuracy()
        
        # Compare to longer history
        historical_accuracy = self.get_current_accuracy(window=self.drift_window * 3)
        
        accuracy_drop = historical_accuracy - current_accuracy
        drift_detected = accuracy_drop > 0.05 or current_accuracy < self.accuracy_threshold
        
        result = {
            "current_accuracy": current_accuracy,
            "historical_accuracy": historical_accuracy,
            "accuracy_drop": accuracy_drop,
            "drift_detected": drift_detected,
            "below_threshold": current_accuracy < self.accuracy_threshold,
        }
        
        if drift_detected:
            logger.warning(f"Model drift detected: {result}")
            self.drift_detected = True
        
        return result
    
    def should_retrain(self) -> bool:
        """Determine if model should be retrained."""
        # Check cooldown
        if self.last_retrain:
            time_since_retrain = datetime.utcnow() - self.last_retrain
            if time_since_retrain < self.retrain_cooldown:
                return False
        
        # Check performance
        drift_result = self.check_drift()
        
        if drift_result["drift_detected"]:
            logger.info("Retrain recommended due to drift detection")
            return True
        
        if drift_result["below_threshold"]:
            logger.info(f"Retrain recommended: accuracy {drift_result['current_accuracy']:.4f} below threshold")
            return True
        
        return False
    
    def trigger_retrain(self) -> bool:
        """Trigger model retraining."""
        if not self.should_retrain():
            return False
        
        logger.info("Triggering model retrain...")
        self.last_retrain = datetime.utcnow()
        self.drift_detected = False
        
        # In production, this would queue a Celery task
        # from app.tasks.training import retrain_model
        # retrain_model.delay()
        
        return True
    
    def get_improvement_stats(self) -> Dict[str, Any]:
        """Get self-improvement statistics."""
        return {
            "total_predictions": len(self.performance_history),
            "current_accuracy": self.get_current_accuracy(),
            "drift_detected": self.drift_detected,
            "last_retrain": self.last_retrain.isoformat() if self.last_retrain else None,
            "retrain_recommended": self.should_retrain(),
        }
    
    def suggest_weight_adjustments(
        self,
        model_accuracies: Dict[str, float],
    ) -> Dict[str, float]:
        """Suggest new ensemble weights based on performance."""
        total = sum(model_accuracies.values())
        if total == 0:
            return {k: 1.0 / len(model_accuracies) for k in model_accuracies}
        
        # Weight proportional to accuracy
        return {k: v / total for k, v in model_accuracies.items()}
    
    def analyze_feature_importance(
        self,
        feature_importance: Dict[str, float],
        accuracy_by_feature: Dict[str, float],
    ) -> List[str]:
        """Suggest features to add or remove."""
        suggestions = []
        
        # Find low importance features
        avg_importance = np.mean(list(feature_importance.values()))
        for feature, importance in feature_importance.items():
            if importance < avg_importance * 0.1:
                suggestions.append(f"Consider removing low-importance feature: {feature}")
        
        return suggestions


# Global instance
improvement_system = SelfImprovementSystem()
