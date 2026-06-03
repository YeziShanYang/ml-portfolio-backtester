import pandas as pd
import numpy as np
from src import stock_screener, indicators
from sklearn import ensemble, preprocessing

class BacktestEngine:
    def __init__(self, model, feature_configs, target_shift=-7, confidence_threshold=0.60, stop_loss=0.10, min_hold_days=7):
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
        self.confidence_threshold = confidence_threshold
        self.stop_loss = stop_loss
        self.min_hold_days = min_hold_days
        self.scaler = preprocessing.StandardScaler()
        self.is_regressor = "Classifier" not in type(model).__name__
    
    def build_features(self, full_df):
        """
        This function basically allows us to get a dictionary of Pandas Dataframes with the info we need
        and handles all of the interactions with indicators.py so we don't have to worry about it later.
        """
        features = {}
        for feat, config in self.feature_configs.items():
            func = config['func']
            params = config.get('params', {})
            data_type = config.get('data_type', 'Close')
            features[feat] = func(full_df[data_type], **params)
        return features
    
    def build_features_df(self, features, ticker=None):
        # I was using this a lot so might as well make it a function
        row = {}
        for name in self.feature_configs:
            series = features[name]
            row[name] = series[ticker] if ticker else series
        return pd.DataFrame(row).dropna()

    def get_training_data(self, training_tickers, features, target_labels):
        """
        This function is almost a copy-paste of what I wrote for run_backtest_simulation.py
        in the first version (when everything was still just one function).
        It takes in all of the info we want and returns the correctly formatted features and label.
        """
        frames = []
        for ticker in training_tickers:
            ticker_df = self.build_features_df(features, ticker=ticker)
            ticker_df['target'] = target_labels[ticker]
            ticker_df['ticker'] = ticker
            frames.append(ticker_df)
 
        combined = pd.concat(frames).dropna()
        X = combined.drop(columns=['target', 'ticker'])
        y = combined['target']
        return X, y
    
    def build_labels(self, close_prices):
        future_15day_avg = close_prices.shift(-1).rolling(window=15, min_periods=1).mean()
        if self.is_regressor:
            return (future_15day_avg - close_prices) / close_prices
        else:
            return (future_15day_avg > close_prices * 1.01).astype(int)

    
    def run_simulation(self, target_ticker, training_tickers, pre_downloaded_df, benchmark_prices):
        """
        This function actually runs our simulation. Thanks to the class/OOP structure we have now,
        it just requires us to provide its target ticker and training tickers, assuming the
        features are already loaded.
        """
        # First, we get the data for training
        close_prices = pre_downloaded_df['Close']

        # Thanks to our previous functions, we can build our training models really easily:
        features = self.build_features(pre_downloaded_df)
        target_labels = self.build_labels(close_prices)

        # Train the model
        X, y = self.get_training_data(training_tickers, features, target_labels)
        X_scaled = self.scaler.fit_transform(X)
        self.model.fit(X_scaled, y)

        # Now we get the data that we'll be testing it against
        ticker_df = pre_downloaded_df.xs(target_ticker, level=1, axis=1)
        test_features = self.build_features(ticker_df)
        test_df = self.build_features_df(test_features)
        test_scaled = self.scaler.transform(test_df)

        if not self.is_regressor and hasattr(self.model, 'predict_proba'):
            # Use predicted probability so we can apply a confidence threshold,
            # which filters out low-conviction signals and reduces overtrading.
            proba = self.model.predict_proba(test_scaled)[:, 1]
            predictions = (proba >= self.confidence_threshold).astype(int)
        else:
            raw = self.model.predict(test_scaled)
            predictions = (raw > 0).astype(int) if self.is_regressor else raw
 
        test_close_prices = close_prices[target_ticker].loc[test_df.index].squeeze()

        # Now we use a loop to actually run the simulation
        starting_capital = 10000.0
        capital = starting_capital
        shares_owned = 0
        stock_owned = False
        days_held = 0

        trade_log = [] # {'type', 'price', 'date', ['return']}
        completed_trades = [] # Tracks percentage return of each closed trade

        for i in range(len(test_close_prices)):
            current_price = test_close_prices.iloc[i]
            current_date = test_close_prices.index[i]
            prediction = predictions[i]

            if stock_owned:
                days_held += 1
                buy_price = trade_log[-1]['price']
                unrealized_loss = (current_price - buy_price) / buy_price
                if unrealized_loss <= -self.stop_loss:  # 7% stop-loss
                    # force sell
                    prediction = 0
                    days_held = self.min_hold_days  # bypass the min-hold check

            if prediction == 1 and not stock_owned:
                shares_owned = capital / current_price
                capital = 0
                stock_owned = True
                days_held = 0
                trade_log.append({'type': 'BUY', 'price': current_price, 'date': current_date})

            elif prediction == 0 and stock_owned and days_held >= self.min_hold_days:
                capital = shares_owned * current_price
                shares_owned = 0
                stock_owned = False
                ret = ((current_price - trade_log[-1]['price']) / trade_log[-1]['price']) * 100
                completed_trades.append(ret)
                trade_log.append({'type': 'SELL', 'price': current_price, 'date': current_date, 'return': ret})

        
        # Clean up our open position at the end
        if stock_owned:
            final_price = test_close_prices.iloc[-1]
            capital = shares_owned * final_price
            ret = ((final_price - trade_log[-1]['price']) / trade_log[-1]['price']) * 100
            completed_trades.append(ret)
            trade_log.append({'type': 'SELL_END', 'price': final_price, 'date': test_close_prices.index[-1], 'return': ret})
        
        # Now we can evaluate how well our simulation did:
        total_return = ((capital - starting_capital) / starting_capital) * 100

        start_date = test_close_prices.index[0]
        end_date   = test_close_prices.index[-1]
        benchmark_start  = benchmark_prices.loc[start_date]
        benchmark_end    = benchmark_prices.loc[end_date]
        benchmark_return  = ((benchmark_end - benchmark_start) / benchmark_start) * 100

        alpha = total_return - benchmark_return
 
        total_trades = len(completed_trades)
        winning_trades = sum(1 for r in completed_trades if r > 0)
        win_rate = (winning_trades / total_trades * 100) if total_trades > 0 else 0

        print(f"Target: {target_ticker:<5} | Strategy: {total_return:+.2f}% | Alpha: {alpha:+.2f}% | Trades: {total_trades}")

        self.last_strategy_return = total_return
        self.last_benchmark_return       = benchmark_return
        self.last_alpha           = alpha
        self.last_win_rate        = win_rate
        self.last_total_trades    = total_trades
        self.last_trade_log       = pd.DataFrame(trade_log)

if __name__ == "__main__":
    # We set binary to True for random forest because it returns a categorical 0/1
    features_rf = {
        'sma_trend_regime': {
            'func': indicators.calculate_sma_crossover, 
            'params': {'fast_window': 5, 'slow_window': 20, 'binary': True}
        },
        'sma_position_50': {
            'func':   indicators.calculate_sma_position,
            'params': {'window': 50},
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
        "ORCL", "AAPL", "TEAM", "INTC", 
        "SNOW", "WIX", "AMD", "CSCO", 
        "SHOP", "AMZN", "CRM"
    ]

    benchmark = "SPY"

    print(f"Downloading data for {len(train_pool)} assets & benchmark {benchmark} from yfinance...")
    master_df = stock_screener.fetch_screener_data(train_pool + [benchmark], period="2y", interval="1d")
    print("Download Complete.")

    benchmark_prices = master_df['Close'][benchmark]

    rf_classifier = ensemble.RandomForestClassifier(
        n_estimators=100,
        random_state=42,
        min_samples_split=10,
        max_features='sqrt',
        max_depth=10
    )

    engine = BacktestEngine(
        model=rf_classifier,
        feature_configs=features_rf,
        confidence_threshold=0.60,
        stop_loss=0.10,
        min_hold_days=7
    )


    results = {k: [] for k in ('strategy', 'benchmark', 'alpha', 'win_rate', 'trades')}

    for target in train_pool:
        oos_pool = [t for t in train_pool if t != target]
        engine.run_simulation(target_ticker=target, training_tickers=oos_pool, pre_downloaded_df=master_df, benchmark_prices=benchmark_prices)
 
        results['strategy'].append(engine.last_strategy_return)
        results['benchmark'].append(engine.last_benchmark_return)
        results['alpha'].append(engine.last_alpha)
        results['win_rate'].append(engine.last_win_rate)
        results['trades'].append(engine.last_total_trades)
 
    print("\nSummary Statistics:")
    print(f"  Average Strategy Return   : {np.mean(results['strategy']):+.2f}%")
    print(f"  Average Benchmark Return : {np.mean(results['benchmark']):+.2f}%")
    print(f"  Mean Alpha                : {np.mean(results['alpha']):+.2f}%")
    print(f"  Average Win Rate          : {np.mean(results['win_rate']):.2f}%")
    print(f"  Average Trades Executed   : {np.mean(results['trades']):.1f}")




    """
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

    lin_regressor = linear_model.LinearRegression()
    engine_lr = BacktestEngine(model=lin_regressor, feature_configs=features_lr)
    engine_lr.run_simulation(target_ticker=target, training_tickers=train_pool, period="1y")
    """