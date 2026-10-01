import pandas as pd
import yfinance as yf

"""
Moving Average Simulation
Strategy: 
  - Buy when the closing price drops below the 5-day moving average (underpriced)
  - Sell when the closing price rises above the 5-day moving average (overpriced)
  - Simulation defaults to 10000 to make %ages easy
"""

def get_stock_data(ticker, period="60d", interval="1d"):
    print(f"Fetching historical data for {ticker}...")
    
    # We use yf.Ticker instead of yf.download because we might want to get other info later
    stock = yf.Ticker(ticker)
    data = stock.history(period=period, interval=interval)
    
    return data

def run_moving_avg_simulation(ticker, period="60d", interval="1d", window="5d"):
    # Fetch historical stock data
    pandas_df = get_stock_data(ticker, period=period, interval=interval)
    pandas_df['MA'] = pandas_df['Close'].rolling(window=window).mean()

    # Clear rows w/ empty cells (first few rows can't calculate)
    pandas_df=pandas_df.dropna()

    # Setup initial conditions
    starting_capital = 10000
    capital = starting_capital
    shares_owned = 0
    stock_owned = False

    print(f"\nStarting Simulation for {ticker} with starting capital ${starting_capital:,.2f}")
    print("------------------------------------")
    
    # Run simulation
    for (date, row) in pandas_df.iterrows():
        close_price = row['Close']
        ma_price = row['MA']

         # Checks to see if we should buy or sell
        if close_price < ma_price and not stock_owned:
            shares_owned = capital / close_price
            capital = 0
            stock_owned = True
            print(f"BUY on {date.date()}: Bought {shares_owned:.2f} shares at ${close_price:.2f}")
        elif close_price > ma_price and stock_owned:
            capital = shares_owned * close_price
            shares_owned = 0
            stock_owned = False
            print(f"SELL on {date.date()}: Sold shares at ${close_price:.2f}, Capital now ${capital:,.2f}")
    
    # Calculate final statistics
    final_balance = capital if not stock_owned else (shares_owned * pandas_df["Close"].iloc[-1])
    total_return = ((final_balance - starting_capital) / starting_capital) * 100

    print("\n--- Simulation Results ---")
    print(f"Starting Capital: ${starting_capital:,.2f}")
    print(f"Ending Capital:   ${final_balance:,.2f}")
    print(f"Total Return:     {total_return:.2f}%")

if __name__ == "__main__":
    run_moving_avg_simulation("AAPL")


