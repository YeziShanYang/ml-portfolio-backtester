import pandas as pd
import yfinance as yf
from src import stock_screener, indicators
from sklearn import ensemble, linear_model, metrics

class BacktestEngine:
    def __init__(self, model, feature_configs, target_shift=-7):
        """
        Here, 'model' is the kind of scikit-learn model we want to use.
        Since different types of models work differently, we would like to encompass
        both 0/1 classifiers as well as regression models.
        'feature_configs' is our dictionary of features we want the model to consider.
        Each entry is formatted 'feature_name': {'func': function_name, 'params': param_dict}.
        So for example if we wanted sma_5: 'sma_5':  {'func': indicators.calculate_sma, 'params': {'window': 5}}
        Note: target_shift must be negative because of the way we shift data for the labels for training
        """
        self.model = model
        self.feature_configs = feature_configs
        self.target_shift = target_shift
        self.is_regressor = "Classifier" not in type(model).__name__
    
    def build_features(self, close_df):
        """
        This function basically allows us to get a dictionary of Pandas Dataframes with the info we need
        and handles all of the interactions with indicators.py so we don't have to worry about it later.
        """
        features_dict = {}
        for feat, config in self.feature_configs.items():
            features_dict[feat] = config['func'](close_df, **config['params'])
        return features_dict
    
    def get_training_data(self, training_tickers, close_prices, features, target_labels):
        """
        This function is almost a copy-paste of what I wrote for run_backtest_simulation.py
        in the first version (when everything was still just one function).
        It takes in all of the info we want and returns the correctly formatted features and label.
        """
        training_list = []
        for ticker in training_tickers:
            ticker_data = {}

            for feat in self.feature_configs.keys():
                ticker_data[feat] = features[feat][ticker]
            
            ticker_data['target'] = target_labels[ticker]
            training_stock_df = pd.DataFrame(ticker_data)
            training_stock_df['ticker'] = ticker
            training_list.append(training_stock_df)
        
        training_df = pd.concat(training_list).dropna()
        X = training_df.drop(columns=['target', 'ticker'])
        y = training_df['target']
        return X, y
    
    def run_simulation(self, target_ticker, training_tickers, period="1y", interval="1d"):
        """
        This function actually runs our simulation. Thanks to the class/OOP structure we have now,
        it just requires us to provide its target ticker and training tickers, assuming the
        features are already loaded.
        """
        # First, we get the data for training
        pandas_df = stock_screener.fetch_screener_data(training_tickers, period=period, interval=interval)
        close_prices = pandas_df['Close']

        # Thanks to our previous functions, we can build our training models really easily:
        features = self.build_features(close_prices)
        future_price = close_prices.shift(self.target_shift)
        if self.is_regressor:
            target_labels = (future_price - close_prices) / close_prices
        else:
            target_labels = (future_price > close_prices).astype(int)

        X, y = self.get_training_data(training_tickers, close_prices, features, target_labels)
        self.model.fit(X, y)

        # Now we get the data that we'll be testing it against
        ticker_df = stock_screener.fetch_screener_data(target_ticker, period=period, interval=interval)
        ticker_close_prices = ticker_df['Close']

        target_features = self.build_features(ticker_close_prices)

        # We need to flatten this list, because yfinance returns dataframes with a multi-index structure.
        # We could probably solve this by building another function in stock_screener, but that's a problem for another day.
        target_stock_dict = {}
        for feat_name in self.feature_configs.keys():
            target_stock_dict[feat_name] = target_features[feat_name][target_ticker]
        target_stock_df = pd.DataFrame(target_stock_dict).dropna()

        # We can now generate model predictions & flatten our data, similar to what we did with the regular backtest function
        raw_predictions = self.model.predict(target_stock_df)
        if self.is_regressor:
            predictions = (raw_predictions > 0).astype(int)
        else:
            predictions = raw_predictions
        test_close_prices = ticker_close_prices[target_ticker].loc[target_stock_df.index].squeeze()

        # Now we use a loop to actually run the simulation
        starting_capital = 10000.0
        capital = starting_capital
        shares_owned = 0
        stock_owned = False

        for i in range(len(test_close_prices)):
            current_price = test_close_prices.iloc[i]
            prediction = predictions[i]

            if prediction == 1 and not stock_owned:
                shares_owned = capital / current_price
                capital = 0
                stock_owned = True
            elif prediction == 0 and stock_owned:
                capital = shares_owned * current_price
                shares_owned = 0
                stock_owned = False
        
        # Now we can evaluate how well our simulation did:
        final_price = test_close_prices.iloc[-1]
        final_balance = capital if not stock_owned else (shares_owned * final_price)
        total_return = ((final_balance - starting_capital) / starting_capital) * 100

        # We calculate the returns for buy-and-hold so we can compare & calculate alpha
        bh_shares = starting_capital / test_close_prices.iloc[0]
        bh_final_balance = bh_shares * final_price
        bh_return = ((bh_final_balance - starting_capital) / starting_capital) * 100

        print(f"\nResults of simulation for {target_ticker} based on {training_tickers} ({type(self.model).__name__})")
        print(f"Features Utilized       : {list(self.feature_configs.keys())}")
        print(f"Ending Model Capital    : ${final_balance:,.2f} ({total_return:+.2f}%)")
        print(f"Baseline Buy & Hold     : ${bh_final_balance:,.2f} ({bh_return:+.2f}%)")

if __name__ == "__main__":
    # We set binary to True for random forest because it returns a categorical 0/1
    features_rf = {
        'sma_trend_regime': {
            'func': indicators.calculate_sma_crossover, 
            'params': {'fast_window': 5, 'slow_window': 20, 'binary': True}
        },
        'rsi': {
            'func': indicators.calculate_rsi,
            'params': {'window': 14}
        }
    }
    # binary is False for linreg because decimals are handled better
    features_lr = {
        'sma_trend_distance': {
            'func': indicators.calculate_sma_crossover,
            'params': {'fast_window': 5, 'slow_window': 20, 'binary': False}
        },
        'rsi': {
            'func': indicators.calculate_rsi,
            'params': {'window': 14}
        }
    }
    
    train_pool = ["MSFT", "GOOGL", "NVDA"]
    target = "AAPL"

    rf_classifier = ensemble.RandomForestClassifier(n_estimators=100, random_state=42, min_samples_split=10)
    engine_rf = BacktestEngine(model=rf_classifier, feature_configs=features_rf)
    engine_rf.run_simulation(target_ticker=target, training_tickers=train_pool, period="1y")

    lin_regressor = linear_model.LinearRegression()
    engine_lr = BacktestEngine(model=lin_regressor, feature_configs=features_lr)
    engine_lr.run_simulation(target_ticker=target, training_tickers=train_pool, period="1y")


# I'll leave this here as it's still interesting. May move to a different file later on.
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
    rf_model = ensemble.RandomForestClassifier(n_estimators=100, random_state=42, min_samples_split=10)
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