"""
Volume Indicators
=================
OBV, VWAP, and other volume-based indicators.
"""

import numpy as np
import pandas as pd


def obv(close: pd.Series, volume: pd.Series) -> pd.Series:
    """On-Balance Volume."""
    direction = np.sign(close.diff())
    return (direction * volume).cumsum()


def vwap(
    high: pd.Series,
    low: pd.Series,
    close: pd.Series,
    volume: pd.Series,
) -> pd.Series:
    """Volume Weighted Average Price."""
    typical_price = (high + low + close) / 3
    vwap_value = (typical_price * volume).cumsum() / volume.cumsum()
    return vwap_value


def accumulation_distribution(
    high: pd.Series,
    low: pd.Series,
    close: pd.Series,
    volume: pd.Series,
) -> pd.Series:
    """Accumulation/Distribution Line."""
    clv = ((close - low) - (high - close)) / (high - low)
    clv = clv.fillna(0)
    ad = (clv * volume).cumsum()
    return ad


def chaikin_money_flow(
    high: pd.Series,
    low: pd.Series,
    close: pd.Series,
    volume: pd.Series,
    period: int = 20,
) -> pd.Series:
    """Chaikin Money Flow."""
    clv = ((close - low) - (high - close)) / (high - low)
    clv = clv.fillna(0)
    
    cmf = (clv * volume).rolling(window=period).sum() / volume.rolling(window=period).sum()
    return cmf


def force_index(
    close: pd.Series,
    volume: pd.Series,
    period: int = 13,
) -> pd.Series:
    """Force Index."""
    fi = close.diff() * volume
    return fi.ewm(span=period, adjust=False).mean()


def ease_of_movement(
    high: pd.Series,
    low: pd.Series,
    volume: pd.Series,
    period: int = 14,
) -> pd.Series:
    """Ease of Movement."""
    distance_moved = ((high + low) / 2) - ((high.shift(1) + low.shift(1)) / 2)
    box_ratio = (volume / 1e6) / (high - low)
    emv = distance_moved / box_ratio
    return emv.rolling(window=period).mean()


def volume_price_trend(close: pd.Series, volume: pd.Series) -> pd.Series:
    """Volume Price Trend."""
    pct_change = close.pct_change()
    vpt = (pct_change * volume).cumsum()
    return vpt


def negative_volume_index(close: pd.Series, volume: pd.Series) -> pd.Series:
    """Negative Volume Index."""
    nvi = pd.Series(index=close.index, dtype=float)
    nvi.iloc[0] = 1000
    
    for i in range(1, len(close)):
        if volume.iloc[i] < volume.iloc[i-1]:
            nvi.iloc[i] = nvi.iloc[i-1] * (1 + (close.iloc[i] - close.iloc[i-1]) / close.iloc[i-1])
        else:
            nvi.iloc[i] = nvi.iloc[i-1]
    
    return nvi


def positive_volume_index(close: pd.Series, volume: pd.Series) -> pd.Series:
    """Positive Volume Index."""
    pvi = pd.Series(index=close.index, dtype=float)
    pvi.iloc[0] = 1000
    
    for i in range(1, len(close)):
        if volume.iloc[i] > volume.iloc[i-1]:
            pvi.iloc[i] = pvi.iloc[i-1] * (1 + (close.iloc[i] - close.iloc[i-1]) / close.iloc[i-1])
        else:
            pvi.iloc[i] = pvi.iloc[i-1]
    
    return pvi


def volume_oscillator(
    volume: pd.Series,
    fast_period: int = 5,
    slow_period: int = 10,
) -> pd.Series:
    """Volume Oscillator."""
    fast_ema = volume.ewm(span=fast_period, adjust=False).mean()
    slow_ema = volume.ewm(span=slow_period, adjust=False).mean()
    vo = ((fast_ema - slow_ema) / slow_ema) * 100
    return vo
