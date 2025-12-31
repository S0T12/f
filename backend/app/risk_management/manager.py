"""
Risk Manager
============
Central risk management orchestrator.
"""

import logging
from dataclasses import dataclass
from datetime import datetime
from typing import Dict, Any, Optional, List
from enum import Enum

from app.risk_management.position_sizing import PositionSizer, PositionSize
from app.risk_management.drawdown import DrawdownController, DrawdownState
from app.strategies.base import Signal, SignalType

logger = logging.getLogger(__name__)


class RiskDecision(Enum):
    """Risk-based trading decisions."""
    APPROVE = "approve"
    REDUCE = "reduce"
    REJECT = "reject"
    PAUSE = "pause"


@dataclass
class RiskAssessment:
    """Risk assessment result."""
    decision: RiskDecision
    position_size: PositionSize
    adjusted_stop_loss: float
    adjusted_take_profit: float
    risk_reward_ratio: float
    reasons: List[str]
    metadata: Dict[str, Any]


class RiskManager:
    """
    Central risk management system.
    
    Coordinates position sizing, drawdown control, and trade risk assessment.
    """
    
    def __init__(
        self,
        account_balance: float,
        max_risk_per_trade: float = 2.0,
        max_daily_risk: float = 6.0,
        max_total_exposure: float = 10.0,
        min_risk_reward: float = 1.5,
        max_open_trades: int = 5,
    ):
        self.account_balance = account_balance
        self.max_risk_per_trade = max_risk_per_trade
        self.max_daily_risk = max_daily_risk
        self.max_total_exposure = max_total_exposure
        self.min_risk_reward = min_risk_reward
        self.max_open_trades = max_open_trades
        
        # Components
        self.position_sizer = PositionSizer(
            max_risk_percent=max_risk_per_trade,
        )
        self.drawdown_controller = DrawdownController(
            initial_balance=account_balance,
        )
        
        # Tracking
        self.open_positions: List[Dict[str, Any]] = []
        self.daily_risk_used = 0.0
        self.trade_history: List[Dict[str, Any]] = []
    
    def assess_trade(
        self,
        signal: Signal,
        current_price: float,
        atr: float = None,
    ) -> RiskAssessment:
        """Perform comprehensive risk assessment for a trade."""
        reasons = []
        metadata = {}
        
        # Check if trading is allowed
        if not self.drawdown_controller.can_trade():
            return RiskAssessment(
                decision=RiskDecision.PAUSE,
                position_size=PositionSize(0, 0, 0, "rejected"),
                adjusted_stop_loss=0,
                adjusted_take_profit=0,
                risk_reward_ratio=0,
                reasons=["Trading paused due to drawdown limits"],
                metadata={"drawdown_status": self.drawdown_controller.get_status()},
            )
        
        # Check open positions limit
        if len(self.open_positions) >= self.max_open_trades:
            return RiskAssessment(
                decision=RiskDecision.REJECT,
                position_size=PositionSize(0, 0, 0, "rejected"),
                adjusted_stop_loss=0,
                adjusted_take_profit=0,
                risk_reward_ratio=0,
                reasons=[f"Max open trades ({self.max_open_trades}) reached"],
                metadata={"open_positions": len(self.open_positions)},
            )
        
        # Validate stop loss and take profit
        if not signal.stop_loss or not signal.take_profit:
            return RiskAssessment(
                decision=RiskDecision.REJECT,
                position_size=PositionSize(0, 0, 0, "rejected"),
                adjusted_stop_loss=0,
                adjusted_take_profit=0,
                risk_reward_ratio=0,
                reasons=["Missing stop loss or take profit"],
                metadata={},
            )
        
        # Calculate risk/reward ratio
        risk = abs(current_price - signal.stop_loss)
        reward = abs(signal.take_profit - current_price)
        rr_ratio = reward / risk if risk > 0 else 0
        
        # Check minimum R:R
        if rr_ratio < self.min_risk_reward:
            reasons.append(f"R:R ratio {rr_ratio:.2f} below minimum {self.min_risk_reward}")
            
            # Adjust take profit to meet minimum R:R
            if signal.signal_type == SignalType.BUY:
                adjusted_tp = current_price + (risk * self.min_risk_reward)
            else:
                adjusted_tp = current_price - (risk * self.min_risk_reward)
            
            signal.take_profit = adjusted_tp
            rr_ratio = self.min_risk_reward
        
        # Calculate position size
        position = self.position_sizer.calculate_percent_risk(
            self.account_balance,
            current_price,
            signal.stop_loss,
        )
        
        # Apply drawdown scaling
        dd_scale = self.drawdown_controller.get_position_scale()
        if dd_scale < 1.0:
            position.lot_size = round(position.lot_size * dd_scale, 2)
            position.risk_amount *= dd_scale
            reasons.append(f"Position reduced by {(1-dd_scale)*100:.0f}% due to drawdown")
        
        # Apply confidence scaling
        position = self.position_sizer.scale_for_confidence(
            position,
            signal.confidence,
        )
        
        # Check daily risk limit
        new_daily_risk = self.daily_risk_used + position.risk_percent
        if new_daily_risk > self.max_daily_risk:
            available_risk = self.max_daily_risk - self.daily_risk_used
            if available_risk <= 0.5:
                return RiskAssessment(
                    decision=RiskDecision.REJECT,
                    position_size=position,
                    adjusted_stop_loss=signal.stop_loss,
                    adjusted_take_profit=signal.take_profit,
                    risk_reward_ratio=rr_ratio,
                    reasons=["Daily risk limit reached"],
                    metadata={"daily_risk_used": self.daily_risk_used},
                )
            else:
                # Reduce position to fit within daily limit
                reduction = available_risk / position.risk_percent
                position.lot_size = round(position.lot_size * reduction, 2)
                position.risk_percent = available_risk
                position.risk_amount = self.account_balance * (available_risk / 100)
                reasons.append(f"Position reduced to fit daily risk limit")
        
        # Check total exposure
        current_exposure = sum(p.get("risk_percent", 0) for p in self.open_positions)
        if current_exposure + position.risk_percent > self.max_total_exposure:
            available_exposure = self.max_total_exposure - current_exposure
            if available_exposure <= 0.5:
                return RiskAssessment(
                    decision=RiskDecision.REJECT,
                    position_size=position,
                    adjusted_stop_loss=signal.stop_loss,
                    adjusted_take_profit=signal.take_profit,
                    risk_reward_ratio=rr_ratio,
                    reasons=["Total exposure limit reached"],
                    metadata={"current_exposure": current_exposure},
                )
        
        # Determine final decision
        if len(reasons) > 0:
            decision = RiskDecision.REDUCE
        else:
            decision = RiskDecision.APPROVE
        
        # Ensure minimum position size
        if position.lot_size < self.position_sizer.min_position_size:
            return RiskAssessment(
                decision=RiskDecision.REJECT,
                position_size=position,
                adjusted_stop_loss=signal.stop_loss,
                adjusted_take_profit=signal.take_profit,
                risk_reward_ratio=rr_ratio,
                reasons=["Calculated position size below minimum"],
                metadata={},
            )
        
        metadata = {
            "original_rr_ratio": reward / risk if risk > 0 else 0,
            "adjusted_rr_ratio": rr_ratio,
            "confidence": signal.confidence,
            "drawdown_scale": dd_scale,
            "daily_risk_after": self.daily_risk_used + position.risk_percent,
        }
        
        return RiskAssessment(
            decision=decision,
            position_size=position,
            adjusted_stop_loss=signal.stop_loss,
            adjusted_take_profit=signal.take_profit,
            risk_reward_ratio=rr_ratio,
            reasons=reasons,
            metadata=metadata,
        )
    
    def approve_trade(self, assessment: RiskAssessment, trade_id: str) -> bool:
        """Approve and record a trade."""
        if assessment.decision in [RiskDecision.APPROVE, RiskDecision.REDUCE]:
            self.open_positions.append({
                "trade_id": trade_id,
                "risk_percent": assessment.position_size.risk_percent,
                "risk_amount": assessment.position_size.risk_amount,
                "lot_size": assessment.position_size.lot_size,
                "opened_at": datetime.utcnow(),
            })
            
            self.daily_risk_used += assessment.position_size.risk_percent
            return True
        
        return False
    
    def close_trade(self, trade_id: str, pnl: float) -> None:
        """Close a trade and update risk tracking."""
        # Remove from open positions
        self.open_positions = [p for p in self.open_positions if p["trade_id"] != trade_id]
        
        # Update drawdown controller
        self.drawdown_controller.record_trade_result(pnl)
        
        # Update account balance
        self.account_balance += pnl
        
        # Record in history
        self.trade_history.append({
            "trade_id": trade_id,
            "pnl": pnl,
            "closed_at": datetime.utcnow(),
            "balance_after": self.account_balance,
        })
    
    def update_balance(self, new_balance: float) -> None:
        """Update account balance."""
        self.account_balance = new_balance
        self.drawdown_controller.update_balance(new_balance)
    
    def reset_daily_limits(self) -> None:
        """Reset daily risk limits (call at start of new trading day)."""
        self.daily_risk_used = 0.0
        logger.info("Daily risk limits reset")
    
    def get_status(self) -> Dict[str, Any]:
        """Get current risk management status."""
        return {
            "account_balance": self.account_balance,
            "open_positions": len(self.open_positions),
            "total_exposure": sum(p.get("risk_percent", 0) for p in self.open_positions),
            "daily_risk_used": self.daily_risk_used,
            "daily_risk_remaining": self.max_daily_risk - self.daily_risk_used,
            "can_trade": self.drawdown_controller.can_trade(),
            "drawdown_status": self.drawdown_controller.get_status(),
            "position_scale": self.drawdown_controller.get_position_scale(),
        }
    
    def calculate_optimal_position(
        self,
        entry_price: float,
        stop_loss: float,
        take_profit: float,
        win_rate: float = 0.55,
        confidence: float = 0.7,
    ) -> PositionSize:
        """Calculate optimal position size considering all factors."""
        # Start with Kelly criterion
        position = self.position_sizer.calculate_kelly(
            self.account_balance,
            entry_price,
            stop_loss,
            take_profit,
            win_rate,
        )
        
        # Apply drawdown scaling
        dd_scale = self.drawdown_controller.get_position_scale()
        if dd_scale < 1.0:
            position.lot_size = round(position.lot_size * dd_scale, 2)
            position.risk_percent *= dd_scale
        
        # Apply confidence scaling
        position = self.position_sizer.scale_for_confidence(position, confidence)
        
        return position
