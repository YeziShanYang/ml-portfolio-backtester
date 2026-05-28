import pandas as pd
import yfinance as yf
from src import stock_screener, indicators
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report

"""
Backtesting Simulation
Strategy: 
  - We will use data from a specific stock and see our returns
  - Our ML models (Linear SVM & Random Forest) will tell us whether to invest or not
  - We will first implement Random Forst as it performed better in 03_model_training
ConclusionL
  - Our simulation tested AAPL with training data MSFT, GOOGL, NVDA. 
  - We experienced +38.42% returns over 1 year.
Potential next step is to expand to pool of multiple stocks
"""

def run_backtest_simulation(ticker, training_tickers, period="60d", interval="1d"):
    # Our first step is to get the data for training.
    # Here, we will use training_tickers in order to guess ticker.

    # Fetch historical stock data
    pandas_df = stock_screener.fetch_screener_data(training_tickers, period=period, interval=interval)
    close_prices = pandas_df['Close']
    

    # Setup training matrix:
    # 3 features come from indicators: SMA_5, SMA_20, RSI
    sma_5 = indicators.calculate_sma(close_prices, window=5)
    sma_20 = indicators.calculate_sma(close_prices, window=20)
    rsi_14 = indicators.calculate_rsi(close_prices, window=14)

    # To get the answers for our training data, we setup target labels
    # We have set it to 7 days into the future on purpose here
    # Note: 1 means price went up, should've bought, 0 means it went down, shouldn't have bought
    future_price = close_prices.shift(-7)
    target_labels = (future_price > close_prices).astype(int)

    # Initialize training dataframe
    training_list = []
    for t_ticker in training_tickers:
        training_stock_df = pd.DataFrame({
            'sma_5': sma_5[t_ticker],
            'sma_20': sma_20[t_ticker],
            'rsi': rsi_14[t_ticker],
            'target': target_labels[t_ticker]
        })
        training_stock_df['ticker'] = t_ticker
        training_list.append(training_stock_df)

    training_df = pd.concat(training_list)
    training_df = training_df.dropna()

    # Set up training features & solutions (label)
    X = training_df.drop(columns=['target', 'ticker'])
    y = training_df['target']

    # Setup random forest model
    rf_model = RandomForestClassifier(n_estimators=100, random_state=42, min_samples_split=10)
    rf_model.fit(X, y)

    # Our second step is to setup the data that we want to run the model on (ticker)
    # Download ticker data. We have to redownload this because it's different
    # Must be different because we can't test it on the same training data
    ticker_prices = stock_screener.fetch_screener_data([ticker], period=period, interval=interval)
    ticker_close_prices = ticker_prices['Close']
    # Setup target stock
    target_stock_df = pd.DataFrame({
        'sma_5': indicators.calculate_sma(ticker_close_prices, window=5)[ticker],
        'sma_20': indicators.calculate_sma(ticker_close_prices, window=20)[ticker],
        'rsi': indicators.calculate_rsi(ticker_close_prices, window=14)[ticker]
    }).dropna()

    predictions = rf_model.predict(target_stock_df)
    # Note: squeeze flattens the 2D dataframe into a 1D pandas series so the string stuff later works
    test_close_prices = ticker_close_prices.loc[target_stock_df.index].squeeze()

    # Our third step is to actually simulate buying and selling day-by-day.

    # Setup initial conditions
    starting_capital = 10000
    capital = starting_capital
    shares_owned = 0
    stock_owned = False

    print(f"\nStarting Backtest Simulation for {ticker} with starting capital ${starting_capital:,.2f}")
    print("------------------------------------")
    
    for i in range(len(test_close_prices)):
        date = test_close_prices.index[i]
        current_price = test_close_prices.iloc[i]
        prediction = predictions[i]

        # Buy if model predicts it'll go up and we don't have it
        if prediction == 1 and not stock_owned:
            shares_owned = capital / current_price
            capital = 0
            stock_owned = True
            print(f"BUY  on {date.date()}: Bought {shares_owned:.2f} shares at ${current_price:.2f}")

        # Sell if model predicts it won't go up and we have it.
        elif prediction == 0 and stock_owned:
            capital = shares_owned * current_price
            shares_owned = 0
            stock_owned = False
            print(f"SELL on {date.date()}: Sold shares at ${current_price:.2f}, Capital now ${capital:,.2f}")
    
    # Now we can calculate our final statistics
    final_price = test_close_prices.iloc[-1]
    final_balance = capital if not stock_owned else (shares_owned * final_price)
    total_return = ((final_balance - starting_capital) / starting_capital) * 100

    print("\nSimulation Results:")
    print(f"Starting Capital:   ${starting_capital:,.2f}")
    print(f"Ending   Capital:   ${final_balance:,.2f} ({total_return:+.2f}%)")

if __name__ == "__main__":
    run_backtest_simulation(
        ticker="AAPL", 
        training_tickers=["MSFT", "GOOGL", "NVDA"], 
        period="1y",
        interval="1d"
    )