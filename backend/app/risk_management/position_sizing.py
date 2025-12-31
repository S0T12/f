"""
Position Sizing
===============
Dynamic position sizing based on risk parameters.
"""

import logging
from dataclasses import dataclass
from typing import Optional
from enum import Enum

logger = logging.getLogger(__name__)


class PositionSizeMethod(Enum):
    """Position sizing methods."""
    FIXED = "fixed"
    PERCENT_RISK = "percent_risk"
    KELLY = "kelly"
    ATR_BASED = "atr_based"


@dataclass
class PositionSize:
    """Position size calculation result."""
    lot_size: float
    risk_amount: float
    risk_percent: float
    method: str
    metadata: dict = None


class PositionSizer:
    """
    Dynamic position sizing calculator.
    
    Calculates appropriate position sizes based on account balance,
    risk parameters, and market conditions.
    """
    
    def __init__(
        self,
        max_risk_percent: float = 2.0,
        max_position_size: float = 10.0,
        min_position_size: float = 0.01,
        pip_value: float = 10.0,  # Standard lot pip value for XAU/USD
    ):
        self.max_risk_percent = max_risk_percent
        self.max_position_size = max_position_size
        self.min_position_size = min_position_size
        self.pip_value = pip_value
    
    def calculate_fixed(
        self,
        account_balance: float,
        fixed_lot: float = 0.1,
    ) -> PositionSize:
        """Fixed position size."""
        lot_size = min(fixed_lot, self.max_position_size)
        lot_size = max(lot_size, self.min_position_size)
        
        return PositionSize(
            lot_size=lot_size,
            risk_amount=0,  # Unknown without stop loss
            risk_percent=0,
            method="fixed",
        )
    
    def calculate_percent_risk(
        self,
        account_balance: float,
        entry_price: float,
        stop_loss: float,
        risk_percent: float = None,
    ) -> PositionSize:
        """
        Calculate position size based on percent risk.
        
        Formula: Lot Size = (Account Balance * Risk%) / (SL Distance in pips * Pip Value)
        """
        risk_percent = risk_percent or self.max_risk_percent
        risk_percent = min(risk_percent, self.max_risk_percent)
        
        # Calculate risk amount
        risk_amount = account_balance * (risk_percent / 100)
        
        # Calculate stop loss distance in dollars per lot
        # For XAU/USD, 1 pip = $0.01 movement
        sl_distance = abs(entry_price - stop_loss)
        sl_distance_pips = sl_distance * 100  # Convert to pips
        
        # Calculate lot size
        if sl_distance_pips > 0:
            lot_size = risk_amount / (sl_distance_pips * self.pip_value)
        else:
            lot_size = self.min_position_size
        
        # Apply limits
        lot_size = min(lot_size, self.max_position_size)
        lot_size = max(lot_size, self.min_position_size)
        
        # Round to standard lot increments
        lot_size = round(lot_size, 2)
        
        return PositionSize(
            lot_size=lot_size,
            risk_amount=risk_amount,
            risk_percent=risk_percent,
            method="percent_risk",
            metadata={
                "sl_distance_pips": sl_distance_pips,
                "pip_value": self.pip_value,
            },
        )
    
    def calculate_kelly(
        self,
        account_balance: float,
        entry_price: float,
        stop_loss: float,
        take_profit: float,
        win_rate: float = 0.55,
        fraction: float = 0.25,  # Use 25% Kelly for safety
    ) -> PositionSize:
        """
        Kelly Criterion position sizing.
        
        Kelly % = W - [(1-W) / R]
        Where:
        W = Win rate
        R = Win/Loss ratio (average win / average loss)
        """
        # Calculate risk/reward ratio
        risk = abs(entry_price - stop_loss)
        reward = abs(take_profit - entry_price)
        
        if risk == 0:
            return self.calculate_fixed(account_balance)
        
        rr_ratio = reward / risk
        
        # Kelly formula
        kelly_percent = (win_rate * rr_ratio - (1 - win_rate)) / rr_ratio
        
        # Apply fraction (fractional Kelly)
        kelly_percent = kelly_percent * fraction
        
        # Ensure within bounds
        kelly_percent = max(0, min(kelly_percent, self.max_risk_percent / 100))
        
        # Calculate position size using percent risk method
        position = self.calculate_percent_risk(
            account_balance,
            entry_price,
            stop_loss,
            kelly_percent * 100,
        )
        
        position.method = "kelly"
        position.metadata = position.metadata or {}
        position.metadata.update({
            "kelly_raw": kelly_percent * 100,
            "win_rate": win_rate,
            "rr_ratio": rr_ratio,
            "fraction": fraction,
        })
        
        return position
    
    def calculate_atr_based(
        self,
        account_balance: float,
        entry_price: float,
        atr_value: float,
        atr_multiplier: float = 2.0,
        risk_percent: float = None,
    ) -> PositionSize:
        """
        ATR-based position sizing.
        
        Stop loss is calculated using ATR, then position sized accordingly.
        """
        risk_percent = risk_percent or self.max_risk_percent
        
        # Calculate stop loss distance
        sl_distance = atr_value * atr_multiplier
        
        # Determine stop loss price (assuming buy direction for calculation)
        stop_loss = entry_price - sl_distance
        
        position = self.calculate_percent_risk(
            account_balance,
            entry_price,
            stop_loss,
            risk_percent,
        )
        
        position.method = "atr_based"
        position.metadata = position.metadata or {}
        position.metadata.update({
            "atr": atr_value,
            "atr_multiplier": atr_multiplier,
        })
        
        return position
    
    def adjust_for_correlation(
        self,
        position: PositionSize,
        correlation_factor: float = 1.0,
    ) -> PositionSize:
        """
        Adjust position size for correlated positions.
        
        If trading multiple correlated pairs, reduce size accordingly.
        """
        adjusted_lot = position.lot_size / correlation_factor
        adjusted_lot = max(adjusted_lot, self.min_position_size)
        
        return PositionSize(
            lot_size=round(adjusted_lot, 2),
            risk_amount=position.risk_amount / correlation_factor,
            risk_percent=position.risk_percent / correlation_factor,
            method=f"{position.method}_corr_adjusted",
            metadata={
                "original_lot": position.lot_size,
                "correlation_factor": correlation_factor,
            },
        )
    
    def scale_for_confidence(
        self,
        position: PositionSize,
        confidence: float,
        min_confidence: float = 0.6,
        max_confidence: float = 0.9,
    ) -> PositionSize:
        """
        Scale position size based on signal confidence.
        
        Higher confidence = larger position (within limits).
        """
        if confidence < min_confidence:
            scale = 0.5
        elif confidence > max_confidence:
            scale = 1.0
        else:
            # Linear scaling between min and max
            scale = 0.5 + 0.5 * (confidence - min_confidence) / (max_confidence - min_confidence)
        
        scaled_lot = position.lot_size * scale
        scaled_lot = max(scaled_lot, self.min_position_size)
        
        return PositionSize(
            lot_size=round(scaled_lot, 2),
            risk_amount=position.risk_amount * scale,
            risk_percent=position.risk_percent * scale,
            method=f"{position.method}_confidence_scaled",
            metadata={
                "original_lot": position.lot_size,
                "confidence": confidence,
                "scale_factor": scale,
            },
        )
