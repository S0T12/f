"""
Trend Indicators
================
Moving averages and trend-following indicators.
"""

import numpy as np
import pandas as pd
from typing import Union


def sma(data: pd.Series, period: int = 20) -> pd.Series:
    """Simple Moving Average."""
    return data.rolling(window=period).mean()


def ema(data: pd.Series, period: int = 20) -> pd.Series:
    """Exponential Moving Average."""
    return data.ewm(span=period, adjust=False).mean()


def wma(data: pd.Series, period: int = 20) -> pd.Series:
    """Weighted Moving Average."""
    weights = np.arange(1, period + 1)
    return data.rolling(window=period).apply(
        lambda x: np.dot(x, weights) / weights.sum(), raw=True
    )


def vwma(close: pd.Series, volume: pd.Series, period: int = 20) -> pd.Series:
    """Volume Weighted Moving Average."""
    return (close * volume).rolling(window=period).sum() / volume.rolling(window=period).sum()


def macd(
    data: pd.Series,
    fast_period: int = 12,
    slow_period: int = 26,
    signal_period: int = 9,
) -> tuple[pd.Series, pd.Series, pd.Series]:
    """
    MACD - Moving Average Convergence Divergence.
    Returns: (macd_line, signal_line, histogram)
    """
    fast_ema = ema(data, fast_period)
    slow_ema = ema(data, slow_period)
    macd_line = fast_ema - slow_ema
    signal_line = ema(macd_line, signal_period)
    histogram = macd_line - signal_line
    return macd_line, signal_line, histogram


def adx(
    high: pd.Series,
    low: pd.Series,
    close: pd.Series,
    period: int = 14,
) -> tuple[pd.Series, pd.Series, pd.Series]:
    """
    ADX - Average Directional Index.
    Returns: (adx, plus_di, minus_di)
    """
    # True Range
    tr1 = high - low
    tr2 = abs(high - close.shift(1))
    tr3 = abs(low - close.shift(1))
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr = tr.rolling(window=period).mean()
    
    # Directional Movement
    plus_dm = high.diff()
    minus_dm = -low.diff()
    
    plus_dm = plus_dm.where((plus_dm > minus_dm) & (plus_dm > 0), 0)
    minus_dm = minus_dm.where((minus_dm > plus_dm) & (minus_dm > 0), 0)
    
    # Smoothed DM
    plus_di = 100 * (plus_dm.rolling(window=period).mean() / atr)
    minus_di = 100 * (minus_dm.rolling(window=period).mean() / atr)
    
    # DX and ADX
    dx = 100 * abs(plus_di - minus_di) / (plus_di + minus_di)
    adx_value = dx.rolling(window=period).mean()
    
    return adx_value, plus_di, minus_di


def parabolic_sar(
    high: pd.Series,
    low: pd.Series,
    af_start: float = 0.02,
    af_increment: float = 0.02,
    af_max: float = 0.2,
) -> pd.Series:
    """Parabolic SAR."""
    length = len(high)
    sar = pd.Series(index=high.index, dtype=float)
    
    # Initialize
    is_uptrend = True
    af = af_start
    ep = low.iloc[0]
    sar.iloc[0] = high.iloc[0]
    
    for i in range(1, length):
        if is_uptrend:
            sar.iloc[i] = sar.iloc[i-1] + af * (ep - sar.iloc[i-1])
            sar.iloc[i] = min(sar.iloc[i], low.iloc[i-1], low.iloc[i-2] if i > 1 else low.iloc[i-1])
            
            if low.iloc[i] < sar.iloc[i]:
                is_uptrend = False
                sar.iloc[i] = ep
                ep = low.iloc[i]
                af = af_start
            else:
                if high.iloc[i] > ep:
                    ep = high.iloc[i]
                    af = min(af + af_increment, af_max)
        else:
            sar.iloc[i] = sar.iloc[i-1] + af * (ep - sar.iloc[i-1])
            sar.iloc[i] = max(sar.iloc[i], high.iloc[i-1], high.iloc[i-2] if i > 1 else high.iloc[i-1])
            
            if high.iloc[i] > sar.iloc[i]:
                is_uptrend = True
                sar.iloc[i] = ep
                ep = high.iloc[i]
                af = af_start
            else:
                if low.iloc[i] < ep:
                    ep = low.iloc[i]
                    af = min(af + af_increment, af_max)
    
    return sar


def supertrend(
    high: pd.Series,
    low: pd.Series,
    close: pd.Series,
    period: int = 10,
    multiplier: float = 3.0,
) -> tuple[pd.Series, pd.Series]:
    """
    Supertrend indicator.
    Returns: (supertrend, direction)
    """
    # ATR
    tr = pd.concat([
        high - low,
        abs(high - close.shift(1)),
        abs(low - close.shift(1))
    ], axis=1).max(axis=1)
    atr = tr.rolling(window=period).mean()
    
    # Basic bands
    hl2 = (high + low) / 2
    upper_band = hl2 + multiplier * atr
    lower_band = hl2 - multiplier * atr
    
    supertrend = pd.Series(index=close.index, dtype=float)
    direction = pd.Series(index=close.index, dtype=int)
    
    supertrend.iloc[0] = upper_band.iloc[0]
    direction.iloc[0] = -1
    
    for i in range(1, len(close)):
        if close.iloc[i] > supertrend.iloc[i-1]:
            supertrend.iloc[i] = lower_band.iloc[i]
            direction.iloc[i] = 1
        else:
            supertrend.iloc[i] = upper_band.iloc[i]
            direction.iloc[i] = -1
    
    return supertrend, direction
