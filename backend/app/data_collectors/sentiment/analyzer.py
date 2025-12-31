"""
Sentiment Analyzer
==================
Analyze sentiment from news and social media.
"""

import logging
from typing import List, Dict, Optional
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass  
class SentimentResult:
    """Sentiment analysis result."""
    text: str
    score: float  # -1 to 1
    label: str    # bearish, neutral, bullish
    confidence: float
    
    @classmethod
    def from_score(cls, text: str, score: float, confidence: float = 1.0):
        if score > 0.2:
            label = "bullish"
        elif score < -0.2:
            label = "bearish"
        else:
            label = "neutral"
        return cls(text=text, score=score, label=label, confidence=confidence)


class SentimentAnalyzer:
    """Analyze sentiment using NLP models."""
    
    def __init__(self):
        self._model = None
        self._tokenizer = None
    
    async def initialize(self) -> None:
        """Load the sentiment model."""
        try:
            from transformers import pipeline
            self._model = pipeline(
                "sentiment-analysis",
                model="ProsusAI/finbert",
                device=-1,  # CPU
            )
            logger.info("Sentiment model loaded")
        except Exception as e:
            logger.warning(f"Could not load FinBERT, using fallback: {e}")
            self._model = None
    
    def analyze(self, text: str) -> SentimentResult:
        """Analyze sentiment of text."""
        if not self._model:
            return self._fallback_analyze(text)
        
        try:
            result = self._model(text[:512])[0]  # FinBERT max length
            
            # Convert FinBERT labels to scores
            label = result["label"].lower()
            confidence = result["score"]
            
            if label == "positive":
                score = confidence
            elif label == "negative":
                score = -confidence
            else:
                score = 0.0
            
            return SentimentResult.from_score(text, score, confidence)
            
        except Exception as e:
            logger.error(f"Sentiment analysis error: {e}")
            return self._fallback_analyze(text)
    
    def _fallback_analyze(self, text: str) -> SentimentResult:
        """Fallback keyword-based sentiment analysis."""
        text_lower = text.lower()
        
        bullish_words = [
            "bullish", "surge", "rally", "gain", "rise", "up",
            "buy", "growth", "higher", "positive", "strong"
        ]
        bearish_words = [
            "bearish", "crash", "fall", "drop", "down", "sell",
            "decline", "lower", "negative", "weak", "slump"
        ]
        
        bullish_count = sum(1 for w in bullish_words if w in text_lower)
        bearish_count = sum(1 for w in bearish_words if w in text_lower)
        
        total = bullish_count + bearish_count
        if total == 0:
            score = 0.0
        else:
            score = (bullish_count - bearish_count) / total
        
        return SentimentResult.from_score(text, score, 0.5)
    
    def analyze_batch(self, texts: List[str]) -> List[SentimentResult]:
        """Analyze multiple texts."""
        return [self.analyze(text) for text in texts]
    
    def aggregate_sentiment(self, results: List[SentimentResult]) -> float:
        """Aggregate multiple sentiment results."""
        if not results:
            return 0.0
        
        weighted_sum = sum(r.score * r.confidence for r in results)
        total_weight = sum(r.confidence for r in results)
        
        return weighted_sum / total_weight if total_weight > 0 else 0.0
