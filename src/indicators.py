import pandas as pd

"""
So far the primary purpose of this file is to hold functions that analyze 
and return values for use in our regression models in backtest_simulation.
Note: format of calculation functions used in backtest_simulation must have:
  - input: df_close (Pandas DataFrame)
  - output: same, we'll do more manipulation in backtest_simulation
"""


def calculate_sma(df_close, window="5d"):
    return df_close.rolling(window=window).mean()

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