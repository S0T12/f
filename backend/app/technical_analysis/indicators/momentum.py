"""
Momentum Indicators
===================
RSI, Stochastic, CCI and other momentum indicators.
"""

import numpy as np
import pandas as pd


def rsi(data: pd.Series, period: int = 14) -> pd.Series:
    """Relative Strength Index."""
    delta = data.diff()
    
    gain = delta.where(delta > 0, 0)
    loss = -delta.where(delta < 0, 0)
    
    avg_gain = gain.rolling(window=period).mean()
    avg_loss = loss.rolling(window=period).mean()
    
    rs = avg_gain / avg_loss
    rsi_value = 100 - (100 / (1 + rs))
    
    return rsi_value


def stochastic(
    high: pd.Series,
    low: pd.Series,
    close: pd.Series,
    k_period: int = 14,
    d_period: int = 3,
) -> tuple[pd.Series, pd.Series]:
    """
    Stochastic Oscillator.
    Returns: (%K, %D)
    """
    lowest_low = low.rolling(window=k_period).min()
    highest_high = high.rolling(window=k_period).max()
    
    stoch_k = 100 * (close - lowest_low) / (highest_high - lowest_low)
    stoch_d = stoch_k.rolling(window=d_period).mean()
    
    return stoch_k, stoch_d


def cci(
    high: pd.Series,
    low: pd.Series,
    close: pd.Series,
    period: int = 20,
) -> pd.Series:
    """Commodity Channel Index."""
    typical_price = (high + low + close) / 3
    sma_tp = typical_price.rolling(window=period).mean()
    mad = typical_price.rolling(window=period).apply(
        lambda x: np.mean(np.abs(x - x.mean())), raw=True
    )
    
    cci_value = (typical_price - sma_tp) / (0.015 * mad)
    return cci_value


def williams_r(
    high: pd.Series,
    low: pd.Series,
    close: pd.Series,
    period: int = 14,
) -> pd.Series:
    """Williams %R."""
    highest_high = high.rolling(window=period).max()
    lowest_low = low.rolling(window=period).min()
    
    wr = -100 * (highest_high - close) / (highest_high - lowest_low)
    return wr


def roc(data: pd.Series, period: int = 12) -> pd.Series:
    """Rate of Change."""
    return ((data - data.shift(period)) / data.shift(period)) * 100


def mfi(
    high: pd.Series,
    low: pd.Series,
    close: pd.Series,
    volume: pd.Series,
    period: int = 14,
) -> pd.Series:
    """Money Flow Index."""
    typical_price = (high + low + close) / 3
    raw_money_flow = typical_price * volume
    
    # Positive and negative money flow
    price_change = typical_price.diff()
    positive_flow = raw_money_flow.where(price_change > 0, 0)
    negative_flow = raw_money_flow.where(price_change < 0, 0)
    
    positive_mf = positive_flow.rolling(window=period).sum()
    negative_mf = negative_flow.rolling(window=period).sum()
    
    money_ratio = positive_mf / negative_mf
    mfi_value = 100 - (100 / (1 + money_ratio))
    
    return mfi_value


def awesome_oscillator(
    high: pd.Series,
    low: pd.Series,
    fast_period: int = 5,
    slow_period: int = 34,
) -> pd.Series:
    """Awesome Oscillator."""
    median_price = (high + low) / 2
    ao = median_price.rolling(window=fast_period).mean() - median_price.rolling(window=slow_period).mean()
    return ao


def momentum(data: pd.Series, period: int = 10) -> pd.Series:
    """Momentum indicator."""
    return data - data.shift(period)


def tsi(
    close: pd.Series,
    long_period: int = 25,
    short_period: int = 13,
) -> pd.Series:
    """True Strength Index."""
    price_change = close.diff()
    
    # Double smoothed price change
    first_smooth = price_change.ewm(span=long_period, adjust=False).mean()
    double_smooth = first_smooth.ewm(span=short_period, adjust=False).mean()
    
    # Double smoothed absolute price change
    abs_first_smooth = price_change.abs().ewm(span=long_period, adjust=False).mean()
    abs_double_smooth = abs_first_smooth.ewm(span=short_period, adjust=False).mean()
    
    tsi_value = 100 * double_smooth / abs_double_smooth
    return tsi_value
