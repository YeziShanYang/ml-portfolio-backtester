import pandas as pd
import yfinance as yf
import numpy as np
from src import stock_screener, indicators
from sklearn import ensemble, linear_model, metrics, preprocessing

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
    
    def build_features(self, full_df):
        """
        This function basically allows us to get a dictionary of Pandas Dataframes with the info we need
        and handles all of the interactions with indicators.py so we don't have to worry about it later.
        """
        features_dict = {}
        for feat, config in self.feature_configs.items():
            func = config['func']
            params = config.get('params', {})
            data_type = config.get('data_type', 'Close')
            target_data = full_df[data_type]
            features_dict[feat] = func(target_data, **params)
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
        features = self.build_features(pandas_df)
        future_price = close_prices.shift(self.target_shift)
        if self.is_regressor:
            target_labels = (future_price - close_prices) / close_prices
        else:
            target_labels = (future_price > close_prices).astype(int)

        X, y = self.get_training_data(training_tickers, close_prices, features, target_labels)
        scaler = preprocessing.StandardScaler()
        X_scaled = scaler.fit_transform(X)
        self.model.fit(X_scaled, y)

        # Now we get the data that we'll be testing it against
        ticker_df = stock_screener.fetch_screener_data(target_ticker, period=period, interval=interval)
        ticker_close_prices = ticker_df['Close']

        target_features = self.build_features(ticker_df)

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

        trade_log = [] # Tracks: {'type': 'BUY/SELL', 'price': X, 'date': Y}
        completed_trades = [] # Tracks percentage return of each closed trade

        for i in range(len(test_close_prices)):
            current_price = test_close_prices.iloc[i]
            current_date = test_close_prices.index[i]
            prediction = predictions[i]

            if prediction == 1 and not stock_owned:
                shares_owned = capital / current_price
                capital = 0
                stock_owned = True
                trade_log.append({'type': 'BUY', 'price': current_price, 'date': current_date})
            elif prediction == 0 and stock_owned:
                capital = shares_owned * current_price
                shares_owned = 0
                stock_owned = False
                buy_price = trade_log[-1]['price']
                trade_return = ((current_price - buy_price) / buy_price) * 100
                completed_trades.append(trade_return)
                trade_log.append({'type': 'SELL', 'price': current_price, 'date': current_date, 'return': trade_return})
        
        # Clean up our open position at the end
        if stock_owned:
            final_price = test_close_prices.iloc[-1]
            capital = shares_owned * final_price
            trade_return = ((final_price - trade_log[-1]['price']) / trade_log[-1]['price']) * 100
            completed_trades.append(trade_return)
            trade_log.append({'type': 'SELL_END', 'price': final_price, 'date': test_close_prices.index[-1], 'return': trade_return})
        
        # Now we can evaluate how well our simulation did:
        final_balance = capital
        total_return = ((final_balance - starting_capital) / starting_capital) * 100
        
        bh_shares = starting_capital / test_close_prices.iloc[0]
        bh_final_balance = bh_shares * test_close_prices.iloc[-1]
        bh_return = ((test_close_prices.iloc[-1] - test_close_prices.iloc[0]) / test_close_prices.iloc[0]) * 100

        # We calculate the returns for buy-and-hold so we can compare & calculate alpha
        total_trades = len(completed_trades)
        winning_trades = sum(1 for r in completed_trades if r > 0)
        win_rate = (winning_trades / total_trades * 100) if total_trades > 0 else 0.0

        print(f"\nResults of simulation for {target_ticker} based on {training_tickers} ({type(self.model).__name__})")
        print(f"Total Executed Trades   : {total_trades}")
        print(f"Strategy Win Rate       : {win_rate:.2f}%")
        if total_trades > 0:
            print(f"Avg Return Per Trade    : {np.mean(completed_trades):+.2f}%")
            print(f"Best Trade Performance  : {max(completed_trades):+.2f}%")
            print(f"Worst Trade Performance : {min(completed_trades):+.2f}%")
        print(f"Features Utilized       : {list(self.feature_configs.keys())}")
        print(f"Ending Model Capital    : ${final_balance:,.2f} ({total_return:+.2f}%)")
        print(f"Baseline Buy & Hold     : ${bh_final_balance:,.2f} ({bh_return:+.2f}%)")

        # Save logs to instance
        self.last_trade_log = pd.DataFrame(trade_log)

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
        },
        'bollinger_position': {
            'func': indicators.calculate_bollinger_position,
            'params': {'window': 20, 'num_std': 2}
        },
        'price_roc': {
            'func': indicators.calculate_roc,
            'params': {'window': 10}
        },
        'volume_acceleration': {
            'func': indicators.calculate_volume_spread,
            'params': {'fast_window': 5, 'slow_window': 20},
            'data_type': 'Volume' 
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
        },
        'bollinger_position': {
            'func': indicators.calculate_bollinger_position,
            'params': {'window': 20, 'num_std': 2}
        },
        'price_roc': {
            'func': indicators.calculate_roc,
            'params': {'window': 10}
        },
        'volume_acceleration': {
            'func': indicators.calculate_volume_spread,
            'params': {'fast_window': 5, 'slow_window': 20},
            'data_type': 'Volume' 
        }
    }
    
    train_pool = [
        "NVDA", "MSFT", "AVGO", "NOW", 
        "ORCL", "CRM", "TEAM", "INTC", 
        "SNOW", "WIX", "AMD", "CSCO", 
        "SHOP", "AMZN"
    ]
    target = "AAPL"

    rf_classifier = ensemble.RandomForestClassifier(n_estimators=100, random_state=42, min_samples_split=10)
    engine_rf = BacktestEngine(model=rf_classifier, feature_configs=features_rf)
    engine_rf.run_simulation(target_ticker=target, training_tickers=train_pool, period="1y")

    lin_regressor = linear_model.LinearRegression()
    engine_lr = BacktestEngine(model=lin_regressor, feature_configs=features_lr)
    engine_lr.run_simulation(target_ticker=target, training_tickers=train_pool, period="1y")