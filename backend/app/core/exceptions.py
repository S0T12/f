"""
Custom Exceptions
=================
Application-specific exception classes.
"""

from typing import Any, Optional, Dict


class TradingSystemException(Exception):
    """Base exception for the trading system."""
    
    def __init__(
        self,
        message: str,
        code: Optional[str] = None,
        details: Optional[Dict[str, Any]] = None,
    ):
        self.message = message
        self.code = code or "TRADING_ERROR"
        self.details = details or {}
        super().__init__(self.message)


class DatabaseException(TradingSystemException):
    """Database-related exceptions."""
    
    def __init__(self, message: str, details: Optional[Dict[str, Any]] = None):
        super().__init__(message, "DATABASE_ERROR", details)


class AuthenticationException(TradingSystemException):
    """Authentication-related exceptions."""
    
    def __init__(self, message: str, details: Optional[Dict[str, Any]] = None):
        super().__init__(message, "AUTH_ERROR", details)


class AuthorizationException(TradingSystemException):
    """Authorization-related exceptions."""
    
    def __init__(self, message: str, details: Optional[Dict[str, Any]] = None):
        super().__init__(message, "AUTHZ_ERROR", details)


class MarketDataException(TradingSystemException):
    """Market data-related exceptions."""
    
    def __init__(self, message: str, details: Optional[Dict[str, Any]] = None):
        super().__init__(message, "MARKET_DATA_ERROR", details)


class TradingException(TradingSystemException):
    """Trading operation exceptions."""
    
    def __init__(self, message: str, details: Optional[Dict[str, Any]] = None):
        super().__init__(message, "TRADING_OP_ERROR", details)


class RiskManagementException(TradingSystemException):
    """Risk management violations."""
    
    def __init__(self, message: str, details: Optional[Dict[str, Any]] = None):
        super().__init__(message, "RISK_ERROR", details)


class ModelException(TradingSystemException):
    """ML model-related exceptions."""
    
    def __init__(self, message: str, details: Optional[Dict[str, Any]] = None):
        super().__init__(message, "MODEL_ERROR", details)


class ValidationException(TradingSystemException):
    """Data validation exceptions."""
    
    def __init__(self, message: str, details: Optional[Dict[str, Any]] = None):
        super().__init__(message, "VALIDATION_ERROR", details)


class ExternalAPIException(TradingSystemException):
    """External API call exceptions."""
    
    def __init__(self, message: str, details: Optional[Dict[str, Any]] = None):
        super().__init__(message, "EXTERNAL_API_ERROR", details)


class InsufficientFundsException(TradingException):
    """Insufficient funds for trading operation."""
    
    def __init__(self, required: float, available: float):
        super().__init__(
            f"Insufficient funds: required {required}, available {available}",
            {"required": required, "available": available},
        )


class MaxDrawdownExceededException(RiskManagementException):
    """Maximum drawdown limit exceeded."""
    
    def __init__(self, current_drawdown: float, max_drawdown: float):
        super().__init__(
            f"Maximum drawdown exceeded: {current_drawdown:.2%} > {max_drawdown:.2%}",
            {"current_drawdown": current_drawdown, "max_drawdown": max_drawdown},
        )


class PositionSizeTooLargeException(RiskManagementException):
    """Position size exceeds risk limits."""
    
    def __init__(self, requested_size: float, max_size: float):
        super().__init__(
            f"Position size too large: {requested_size} > {max_size}",
            {"requested_size": requested_size, "max_size": max_size},
        )
