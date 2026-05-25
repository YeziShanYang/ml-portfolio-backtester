import yfinance as yf

def get_stock_data(ticker):
    print(f"Fetching historical data for {ticker}...")
    
    stock = yf.Ticker(ticker)
    data = stock.history(period="60d", interval="1d")
    
    return data

if __name__ == "__main__":
    data = get_stock_data("AAPL")
    
    print("\n--- Data (Last 5 Days) ---")
    print(data.tail())