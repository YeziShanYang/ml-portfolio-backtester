"""
Stock Screener: finds potential stocks based on criteria
  - fetch_screener_data -> gets stock data
  - find_percentage_drops -> looks for stocks that are down in price from peak
  - find_volume_spikes -> looks for stocks that are heavily bought compared to avg
  - find_new_highs -> looks for stocks that are trading at new highs
"""

import pandas as pd
import yfinance as yf

def fetch_screener_data(tickers):
    print(f"Fetching historical data for companies: {tickers}...")
    
    data = yf.download(tickers, period="30d", interval="1d")
    return data

def find_percentage_drops(data):
    print(f"\nSearching for stocks down >5% from 30d peak...")

    # Fetch closing prices for given tickers
    close_prices = data["Close"]
    
    # Calc %age drop from max close price
    drawdown_pct = (close_prices - close_prices.max()) / close_prices.max() * 100
    drawdown_pct = drawdown_pct.dropna()

    # Convert to pandas Series
    today_drawdown = drawdown_pct.iloc[-1]
    today_drawdown = today_drawdown[today_drawdown < -5]

    return today_drawdown.sort_values()

def find_volume_spikes(data):
    print(f"Searching for stocks trading at volume > 150% 20-day rolling avg...")

    volume_data = data["Volume"]

    # Calc %age inc from rolling 20-day avg volume
    avg_vol = volume_data.rolling(window=20).mean()
    vol_pct = (volume_data - avg_vol) / avg_vol * 100
    vol_pct = vol_pct.dropna()

    # Convert to pandas Series
    today_vol_pct = vol_pct.iloc[-1]
    today_vol_pct = today_vol_pct[today_vol_pct >= 50]

    return today_vol_pct.sort_values(ascending=False)

def find_new_highs(data):
    print(f"Searching for stocks trading at new highs...")

    close_prices = data["Close"]

    # Get maximums
    max_close_prices = close_prices.max()

    # Get today's closing prices
    today_close = close_prices.iloc[-1]
    
    # Filter out & return
    today_highs = today_close[today_close >= max_close_prices]

    return today_highs.sort_values(ascending=False)
    

if __name__ == "__main__":
    # Stocks tested
    stock_pool = ["AAPL", "MSFT", "GOOGL", "AMZN", "NVDA", "TSLA"]
    
    raw_data = fetch_screener_data(stock_pool)

    pct_drop_list = find_percentage_drops(raw_data)
    volume_spike_list = find_volume_spikes(raw_data)
    new_high_list = find_new_highs(raw_data)
    
    print("\n" + "-" * 50)
    print("                SCREENER REPORT                ")
    print("-" * 50)
    
    print("\nPotential Bargains (Drawdown %):")
    print(pct_drop_list if not pct_drop_list.empty else "No stocks match this criteria.")
    
    print("\nVolume Spikes (% Above Avg):")
    print(volume_spike_list if not volume_spike_list.empty else "No stocks match this criteria.")
    
    print("\nNew Highs (Current Price):")
    print(new_high_list if not new_high_list.empty else "No stocks match this criteria.")
    print("-" * 50)