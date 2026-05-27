import pandas as pd

def calculate_sma(df_close, window):
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