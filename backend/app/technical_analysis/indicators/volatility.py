"""
Volatility Indicators
=====================
Bollinger Bands, ATR, Keltner Channels and volatility measures.
"""

import numpy as np
import pandas as pd


def bollinger_bands(
    data: pd.Series,
    period: int = 20,
    std_dev: float = 2.0,
) -> tuple[pd.Series, pd.Series, pd.Series]:
    """
    Bollinger Bands.
    Returns: (upper_band, middle_band, lower_band)
    """
    middle = data.rolling(window=period).mean()
    std = data.rolling(window=period).std()
    
    upper = middle + std_dev * std
    lower = middle - std_dev * std
    
    return upper, middle, lower


def atr(
    high: pd.Series,
    low: pd.Series,
    close: pd.Series,
    period: int = 14,
) -> pd.Series:
    """Average True Range."""
    tr1 = high - low
    tr2 = abs(high - close.shift(1))
    tr3 = abs(low - close.shift(1))
    
    true_range = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr_value = true_range.rolling(window=period).mean()
    
    return atr_value


def keltner_channels(
    high: pd.Series,
    low: pd.Series,
    close: pd.Series,
    ema_period: int = 20,
    atr_period: int = 10,
    multiplier: float = 2.0,
) -> tuple[pd.Series, pd.Series, pd.Series]:
    """
    Keltner Channels.
    Returns: (upper_band, middle_band, lower_band)
    """
    middle = close.ewm(span=ema_period, adjust=False).mean()
    atr_value = atr(high, low, close, atr_period)
    
    upper = middle + multiplier * atr_value
    lower = middle - multiplier * atr_value
    
    return upper, middle, lower


def donchian_channels(
    high: pd.Series,
    low: pd.Series,
    period: int = 20,
) -> tuple[pd.Series, pd.Series, pd.Series]:
    """
    Donchian Channels.
    Returns: (upper_band, middle_band, lower_band)
    """
    upper = high.rolling(window=period).max()
    lower = low.rolling(window=period).min()
    middle = (upper + lower) / 2
    
    return upper, middle, lower


def standard_deviation(data: pd.Series, period: int = 20) -> pd.Series:
    """Rolling Standard Deviation."""
    return data.rolling(window=period).std()


def historical_volatility(
    close: pd.Series,
    period: int = 20,
    trading_days: int = 252,
) -> pd.Series:
    """Historical Volatility (annualized)."""
    log_returns = np.log(close / close.shift(1))
    hv = log_returns.rolling(window=period).std() * np.sqrt(trading_days) * 100
    return hv


def bollinger_bandwidth(
    data: pd.Series,
    period: int = 20,
    std_dev: float = 2.0,
) -> pd.Series:
    """Bollinger Bandwidth (volatility measure)."""
    upper, middle, lower = bollinger_bands(data, period, std_dev)
    bandwidth = (upper - lower) / middle * 100
    return bandwidth


def bollinger_percent_b(
    data: pd.Series,
    period: int = 20,
    std_dev: float = 2.0,
) -> pd.Series:
    """Bollinger %B."""
    upper, middle, lower = bollinger_bands(data, period, std_dev)
    percent_b = (data - lower) / (upper - lower)
    return percent_b


def natr(
    high: pd.Series,
    low: pd.Series,
    close: pd.Series,
    period: int = 14,
) -> pd.Series:
    """Normalized Average True Range."""
    atr_value = atr(high, low, close, period)
    natr_value = (atr_value / close) * 100
    return natr_value


def chaikin_volatility(
    high: pd.Series,
    low: pd.Series,
    ema_period: int = 10,
    roc_period: int = 10,
) -> pd.Series:
    """Chaikin Volatility."""
    hl_diff = high - low
    ema_hl = hl_diff.ewm(span=ema_period, adjust=False).mean()
    cv = ((ema_hl - ema_hl.shift(roc_period)) / ema_hl.shift(roc_period)) * 100
    return cv
