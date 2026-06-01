import pandas as pd
import numpy as np

"""
So far the primary purpose of this file is to hold functions that analyze 
and return values for use in our regression models in backtest_simulation.
Note: format of calculation functions used in backtest_simulation must have:
  - input: df_close (Pandas DataFrame)
  - output: same, we'll do more manipulation in backtest_simulation
"""


def calculate_sma(df_close, window="5d"):
    return df_close.rolling(window=window).mean()

def calculate_sma_crossover(df_close, fast_window=5, slow_window=20, binary=True):
    """
    It probably makes more sense for us to see the relationship 
    between fast & slow sma instead of their individual values.
    """
    fast_sma = calculate_sma(df_close, window=fast_window)
    slow_sma = calculate_sma(df_close, window=slow_window)

    if binary:
        return (fast_sma > slow_sma).astype(int)
    else:
        return (fast_sma - slow_sma) / slow_sma

def calculate_bollinger_position(df_close, window=20, num_std=2):
    """
    Calculates the position of the price relative to the Bollinger Bands.
    Returns a continuous scale where 0 is the middle band, >0 is upper, <0 is lower.
    """
    rolling_mean = df_close.rolling(window=window).mean()
    rolling_std = df_close.rolling(window=window).std()
    
    upper_band = rolling_mean + (num_std * rolling_std)
    lower_band = rolling_mean - (num_std * rolling_std)
    
    # Calculate relative position within the bandwidth
    position = (df_close - rolling_mean) / (upper_band - rolling_mean)
    return position

def calculate_roc(df_close, window=10):
    """
    Calculates the percentage Rate of Change over a specified trailing window.
    """
    return df_close.pct_change(periods=window)

def calculate_volume_spread(df_volume, fast_window=5, slow_window=20):
    """
    Measures if current trading volume is accelerating compared to its baseline average.
    """
    fast_vol = df_volume.rolling(window=fast_window).mean()
    slow_vol = df_volume.rolling(window=slow_window).mean()

    spread = (fast_vol - slow_vol) / slow_vol
    spread = spread.replace([np.inf, -np.inf], np.nan).fillna(0)
    
    # Percentage distance above or below the volume baseline
    return spread

def detect_golden_cross(sma_fast, sma_slow):
    """
    Scans the latest row of data to find which stocks 
    currently have a "Golden Cross" (Fast SMA > Slow SMA).
    """
    
    latest_fast = sma_fast.iloc[-1]
    latest_slow = sma_slow.iloc[-1]
    
    has_cross = latest_fast > latest_slow
    
    tickers = has_cross[has_cross == True].index.tolist()
    
    return tickers

def calculate_rsi(df_close, window=14):
    """
    RSI is basically an index of how accurate the current pricing is
    High RSI (>70) = overvalued, a lot more gains than losses (ppl buying)
    Low RSI (<30) = undervalued, a lot more losses (ppl selling)
    """

    # Daily difference in prices
    delta = df_close.diff()

    gains = delta.clip(lower=0)
    losses = -delta.clip(upper=0)

    # Calculate EMA (exponential moving avg)
    avg_gain = gains.ewm(com=window-1, adjust=False).mean()
    avg_loss = losses.ewm(com=window-1, adjust=False).mean()

    rs = avg_gain/avg_loss

    rsi = 100 - (100 / (1 + rs))

    return rsi