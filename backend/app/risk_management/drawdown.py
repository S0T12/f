"""
Drawdown Controller
===================
Monitor and control drawdown to protect account.
"""

import logging
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import List, Optional, Dict, Any
from enum import Enum
from collections import deque

logger = logging.getLogger(__name__)


class DrawdownState(Enum):
    """Drawdown state levels."""
    NORMAL = "normal"
    WARNING = "warning"
    CRITICAL = "critical"
    EMERGENCY = "emergency"
    RECOVERY = "recovery"


@dataclass
class DrawdownLevel:
    """Drawdown level configuration."""
    state: DrawdownState
    threshold_percent: float
    position_scale: float
    max_trades_per_day: int
    action: str


class DrawdownController:
    """
    Drawdown monitoring and control system.
    
    Automatically reduces exposure and pauses trading
    when drawdown thresholds are exceeded.
    """
    
    def __init__(
        self,
        initial_balance: float,
        warning_threshold: float = 5.0,
        critical_threshold: float = 10.0,
        emergency_threshold: float = 15.0,
        daily_loss_limit: float = 3.0,
        max_consecutive_losses: int = 5,
    ):
        self.initial_balance = initial_balance
        self.peak_balance = initial_balance
        self.current_balance = initial_balance
        
        # Thresholds
        self.warning_threshold = warning_threshold
        self.critical_threshold = critical_threshold
        self.emergency_threshold = emergency_threshold
        self.daily_loss_limit = daily_loss_limit
        self.max_consecutive_losses = max_consecutive_losses
        
        # State tracking
        self.current_state = DrawdownState.NORMAL
        self.consecutive_losses = 0
        self.daily_pnl = 0.0
        self.last_reset_date = datetime.utcnow().date()
        
        # History
        self.balance_history: deque = deque(maxlen=1000)
        self.drawdown_history: List[Dict[str, Any]] = []
        
        # Define levels
        self.levels = [
            DrawdownLevel(
                state=DrawdownState.NORMAL,
                threshold_percent=0.0,
                position_scale=1.0,
                max_trades_per_day=20,
                action="normal_trading",
            ),
            DrawdownLevel(
                state=DrawdownState.WARNING,
                threshold_percent=warning_threshold,
                position_scale=0.75,
                max_trades_per_day=10,
                action="reduce_position_size",
            ),
            DrawdownLevel(
                state=DrawdownState.CRITICAL,
                threshold_percent=critical_threshold,
                position_scale=0.5,
                max_trades_per_day=5,
                action="defensive_trading",
            ),
            DrawdownLevel(
                state=DrawdownState.EMERGENCY,
                threshold_percent=emergency_threshold,
                position_scale=0.0,
                max_trades_per_day=0,
                action="stop_trading",
            ),
        ]
    
    def update_balance(self, new_balance: float) -> DrawdownState:
        """Update balance and recalculate drawdown state."""
        self.current_balance = new_balance
        
        # Update peak
        if new_balance > self.peak_balance:
            self.peak_balance = new_balance
        
        # Record history
        self.balance_history.append({
            "timestamp": datetime.utcnow(),
            "balance": new_balance,
            "peak": self.peak_balance,
        })
        
        # Calculate current drawdown
        current_dd = self.get_current_drawdown()
        
        # Update state
        previous_state = self.current_state
        self.current_state = self._determine_state(current_dd)
        
        # Log state changes
        if self.current_state != previous_state:
            logger.warning(
                f"Drawdown state changed: {previous_state.value} -> {self.current_state.value} "
                f"(DD: {current_dd:.2f}%)"
            )
            
            self.drawdown_history.append({
                "timestamp": datetime.utcnow(),
                "from_state": previous_state.value,
                "to_state": self.current_state.value,
                "drawdown_percent": current_dd,
                "balance": new_balance,
            })
        
        return self.current_state
    
    def record_trade_result(self, pnl: float) -> None:
        """Record trade result for consecutive loss tracking."""
        # Reset daily if new day
        today = datetime.utcnow().date()
        if today != self.last_reset_date:
            self.daily_pnl = 0.0
            self.last_reset_date = today
        
        self.daily_pnl += pnl
        
        if pnl < 0:
            self.consecutive_losses += 1
        else:
            self.consecutive_losses = 0
        
        # Update balance
        new_balance = self.current_balance + pnl
        self.update_balance(new_balance)
    
    def get_current_drawdown(self) -> float:
        """Calculate current drawdown percentage."""
        if self.peak_balance == 0:
            return 0.0
        
        return ((self.peak_balance - self.current_balance) / self.peak_balance) * 100
    
    def get_max_drawdown(self) -> float:
        """Calculate maximum historical drawdown."""
        if not self.balance_history:
            return 0.0
        
        peak = self.initial_balance
        max_dd = 0.0
        
        for record in self.balance_history:
            if record["balance"] > peak:
                peak = record["balance"]
            
            dd = ((peak - record["balance"]) / peak) * 100
            max_dd = max(max_dd, dd)
        
        return max_dd
    
    def _determine_state(self, drawdown_percent: float) -> DrawdownState:
        """Determine drawdown state based on current drawdown."""
        # Check emergency first
        if drawdown_percent >= self.emergency_threshold:
            return DrawdownState.EMERGENCY
        elif drawdown_percent >= self.critical_threshold:
            return DrawdownState.CRITICAL
        elif drawdown_percent >= self.warning_threshold:
            return DrawdownState.WARNING
        elif self.current_state == DrawdownState.EMERGENCY:
            # Recovery state after emergency
            if drawdown_percent < self.critical_threshold:
                return DrawdownState.RECOVERY
            return DrawdownState.EMERGENCY
        else:
            return DrawdownState.NORMAL
    
    def can_trade(self) -> bool:
        """Check if trading is allowed based on current state."""
        if self.current_state == DrawdownState.EMERGENCY:
            return False
        
        # Check consecutive losses
        if self.consecutive_losses >= self.max_consecutive_losses:
            logger.warning(f"Trading paused: {self.consecutive_losses} consecutive losses")
            return False
        
        # Check daily loss limit
        daily_loss_percent = abs(self.daily_pnl) / self.current_balance * 100
        if self.daily_pnl < 0 and daily_loss_percent >= self.daily_loss_limit:
            logger.warning(f"Trading paused: Daily loss limit reached ({daily_loss_percent:.2f}%)")
            return False
        
        return True
    
    def get_current_level(self) -> DrawdownLevel:
        """Get current drawdown level configuration."""
        for level in reversed(self.levels):
            if self.current_state == level.state:
                return level
        return self.levels[0]
    
    def get_position_scale(self) -> float:
        """Get position size scale factor based on current state."""
        level = self.get_current_level()
        
        # Additional scaling for consecutive losses
        if self.consecutive_losses >= 3:
            loss_scale = 1.0 - (self.consecutive_losses - 2) * 0.1
            return max(0.3, level.position_scale * loss_scale)
        
        return level.position_scale
    
    def get_max_trades_allowed(self) -> int:
        """Get maximum trades allowed today based on state."""
        level = self.get_current_level()
        return level.max_trades_per_day
    
    def reset_to_normal(self) -> None:
        """Reset to normal state (e.g., after recovery period)."""
        self.current_state = DrawdownState.NORMAL
        self.consecutive_losses = 0
        logger.info("Drawdown controller reset to normal state")
    
    def get_status(self) -> Dict[str, Any]:
        """Get current drawdown controller status."""
        return {
            "state": self.current_state.value,
            "current_balance": self.current_balance,
            "peak_balance": self.peak_balance,
            "initial_balance": self.initial_balance,
            "current_drawdown_percent": self.get_current_drawdown(),
            "max_drawdown_percent": self.get_max_drawdown(),
            "consecutive_losses": self.consecutive_losses,
            "daily_pnl": self.daily_pnl,
            "can_trade": self.can_trade(),
            "position_scale": self.get_position_scale(),
            "max_trades_today": self.get_max_trades_allowed(),
        }
