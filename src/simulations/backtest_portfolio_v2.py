import pandas as pd
import numpy as np
from src import stock_screener, indicators
from sklearn import ensemble, preprocessing

class PortfolioBacktestEngine:
    def __init__(self, model, feature_configs, confidence_threshold=0.60, 
                 stop_loss=0.10, min_hold_days=7, adx_threshold=20, 
                 training_years=2, testing_years=1, offset_years=0,
                 bear_confidence_threshold=0.75):
        # Same as v1 but we don't need to check for which model 
        # it is since we're going to be focusing on RFC
        self.model = model
        self.feature_configs = feature_configs
        self.confidence_threshold = confidence_threshold
        self.stop_loss = stop_loss
        self.min_hold_days = min_hold_days
        self.adx_threshold = adx_threshold
        self.training_years = training_years
        self.testing_years = testing_years
        self.offset_years = offset_years
        self.bear_confidence_threshold = bear_confidence_threshold
        self.scaler = preprocessing.StandardScaler()
    
    def build_features(self, full_df):
        # Same as v1
        features = {}
        for feat, config in self.feature_configs.items():
            func      = config['func']
            params    = config.get('params', {})
            data_type = config.get('data_type', 'Close')
            features[feat] = func(full_df[data_type], **params)
        return features

    def build_features_df(self, features, ticker):
        # +1 function to make code more readable
        row = {}
        for name in self.feature_configs:
            series = features[name]
            row[name] = series[ticker] if ticker else series
        return pd.DataFrame(row).dropna()
    
    def build_labels(self, close_prices):
        # Same as v1
        future_day_avg = close_prices.shift(-1).rolling(window=15, min_periods=1).mean()
        return (future_day_avg > close_prices * 1.01).astype(int)

    def split_data(self, full_df):
        """
        Splits full_df into a training window and a test window based on
        training_years and testing_years, with no overlap between them.
        Returns (train_df, test_df) as separate DataFrames.
        """
        today      = full_df.index.max() - pd.DateOffset(years=self.offset_years)
        test_start = today - pd.DateOffset(years=self.testing_years)
        train_end  = test_start
        train_start = train_end - pd.DateOffset(years=self.training_years)
 
        train_df = full_df.loc[train_start:train_end]
        test_df  = full_df.loc[test_start:today]
        return train_df, test_df
    
    def get_market_regime(self, benchmark_prices, date):
        # Returns 'bull', 'bear', or 'neutral' based on SPY

        if date not in benchmark_prices.index:
            return 'bull'
        
        sma50 = benchmark_prices.rolling(50).mean()
        sma200 = benchmark_prices.rolling(200).mean()
        roc20 = benchmark_prices.pct_change(20)  # 1-month momentum

        price = benchmark_prices.loc[date]
        s50 = sma50.loc[date]
        s200 = sma200.loc[date]
        r20 = roc20.loc[date]

        if price > s50 and s50 > s200 and r20 > 0:
            return 'bull'
        elif price < s200 and r20 < -0.05:
            return 'bear'
        else:
            return 'neutral'
    
    def run_simulation(self, ticker_pool, pre_downloaded_df, benchmark_prices):
        """
        Runs a portfolio simulation across the entire ticker pool.
        Each day during the test window, the model:
          1. Scores every ticker
          2. Applies ADX filter, skipping choppy tickers
          3. Pick the ticker with the highest predicted probability
             above confidence_threshold.
          4. If currently holding a different ticker (and min_hold_days met),
             sell it and buy the new one.
          5. If nothing clears the threshold, sit in cash.
        The model is trained once on the training window across all tickers,
        with no overlap with the test window.
        """
        # Split into train & testing data
        train_df, test_df = self.split_data(pre_downloaded_df)

        # build features & labels on training data
        train_features = self.build_features(train_df)
        train_labels = self.build_labels(train_df['Close'])

        # Concatenate all tickers into one training set; same as v1
        frames = []
        for ticker in ticker_pool:
            ticker_feat_df = self.build_features_df(train_features, ticker=ticker)
            ticker_feat_df['target'] = train_labels[ticker]
            ticker_feat_df['ticker'] = ticker
            frames.append(ticker_feat_df)
 
        combined = pd.concat(frames).dropna()
        X_train = combined.drop(columns=['target', 'ticker'])
        y_train = combined['target']
        X_scaled = self.scaler.fit_transform(X_train)
        self.model.fit(X_scaled, y_train)

        # Now that we're done training the model, we can build our testing features
        test_features = self.build_features(test_df)

        ticker_test_dfs = {}
        ticker_adx = {}
        for ticker in ticker_pool:
            # Single-ticker slice needed for ADX (requires High/Low/Close)
            single_ticker_df = test_df.xs(ticker, level=1, axis=1)
            feat_df = self.build_features_df(test_features, ticker=ticker)
            ticker_test_dfs[ticker] = feat_df
            ticker_adx[ticker] = indicators.calculate_adx(single_ticker_df).reindex(feat_df.index)
        
        # Build predicted probabilities for every ticker across the test window.
        # We align all tickers to a common date index so we can compare them daily.
        common_index = ticker_test_dfs[ticker_pool[0]].index
        for df in ticker_test_dfs.values():
            common_index = common_index.intersection(df.index)

        ticker_proba = {}
        for ticker in ticker_pool:
            feat_df = ticker_test_dfs[ticker].reindex(common_index)
            scaled = self.scaler.transform(feat_df)
            proba = self.model.predict_proba(scaled)[:, 1]
            ticker_proba[ticker] = pd.Series(proba, index=common_index)
 
        # proba_df: rows = dates, columns = tickers, values = P(buy signal)
        proba_df = pd.DataFrame(ticker_proba)
 
        # adx_df: same shape, True where ADX clears the threshold
        adx_df = pd.DataFrame({
            ticker: ticker_adx[ticker].reindex(common_index)
            for ticker in ticker_pool
        }) >= self.adx_threshold

        # Close prices for testing window
        test_close = pre_downloaded_df['Close'].reindex(common_index)

        # We can start the actual simulation now
        starting_capital = 10_000.0
        capital = starting_capital
        shares_owned = 0.0
        held_ticker = None
        days_held = 0
 
        trade_log = []
        completed_trades = []
        portfolio_values = []

        spy_sma200 = benchmark_prices.rolling(window=200).mean()
        in_bull_series = benchmark_prices > spy_sma200

        for i in range(len(common_index)):
            date = common_index[i]
 
            if held_ticker is not None:
                days_held += 1
                current_price = test_close.loc[date, held_ticker]
                buy_price = trade_log[-1]['price']
                unrealized_pct = (current_price - buy_price) / buy_price
 
                # Stop-loss: force exit current position
                if unrealized_pct <= -self.stop_loss:
                    capital = shares_owned * current_price
                    shares_owned = 0.0
                    ret = unrealized_pct * 100
                    completed_trades.append(ret)
                    trade_log.append({'type': 'STOP_LOSS', 'ticker': held_ticker,
                                      'price': current_price, 'date': date, 'return': ret})
                    held_ticker = None
                    days_held = 0

            # Score today's candidates: must clear ADX filter and confidence threshold
            regime = self.get_market_regime(benchmark_prices, date)
            if regime == 'bear':
                if held_ticker is not None and days_held >= self.min_hold_days:
                    sell_price = test_close.loc[date, held_ticker]
                    capital = shares_owned * sell_price
                    shares_owned = 0.0
                    ret = ((sell_price - trade_log[-1]['price']) / trade_log[-1]['price']) * 100
                    completed_trades.append(ret)
                    trade_log.append({'type': 'SELL_BEAR', 'ticker': held_ticker,
                                      'price': sell_price, 'date': date, 'return': ret})
                    held_ticker = None
                    days_held = 0
 
                current_value = shares_owned * test_close.loc[date, held_ticker] if held_ticker else capital
                portfolio_values.append(current_value)
                continue

            threshold = self.confidence_threshold if regime == 'bull' else self.bear_confidence_threshold
            today_proba = proba_df.loc[date].where(adx_df.loc[date])
            candidates  = today_proba[today_proba >= threshold]
            best_ticker = candidates.idxmax() if len(candidates) > 0 else None
 
            # Only act if min_hold_days met or we're not holding anything
            can_switch = (held_ticker is None) or (days_held >= self.min_hold_days)
 
            if can_switch and best_ticker != held_ticker:
                # Sell current position if we have one
                if held_ticker is not None:
                    sell_price = test_close.loc[date, held_ticker]
                    capital = shares_owned * sell_price
                    shares_owned = 0.0
                    ret = ((sell_price - trade_log[-1]['price']) / trade_log[-1]['price']) * 100
                    completed_trades.append(ret)
                    trade_log.append({'type': 'SELL', 'ticker': held_ticker,
                                      'price': sell_price, 'date': date, 'return': ret})
                    held_ticker = None
                    days_held = 0
 
                # Buy new position if there's a candidate
                if best_ticker is not None:
                    buy_price = test_close.loc[date, best_ticker]
                    shares_owned = capital / buy_price
                    capital = 0.0
                    held_ticker = best_ticker
                    days_held = 0
                    trade_log.append({'type': 'BUY', 'ticker': best_ticker,
                                      'price': buy_price, 'date': date})
            current_value = shares_owned * test_close.loc[date, held_ticker] if held_ticker else capital
            portfolio_values.append(current_value)
 
        # Close any open position at end of window
        if held_ticker is not None:
            final_price = test_close.loc[common_index[-1], held_ticker]
            capital = shares_owned * final_price
            ret = ((final_price - trade_log[-1]['price']) / trade_log[-1]['price']) * 100
            completed_trades.append(ret)
            trade_log.append({'type': 'SELL_END', 'ticker': held_ticker,
                               'price': final_price, 'date': common_index[-1], 'return': ret})
 
        total_return = ((capital - starting_capital) / starting_capital) * 100
        self.last_equity_curve = pd.Series(portfolio_values, index=common_index)
 
        bm_start = benchmark_prices.loc[common_index[0]]
        bm_end = benchmark_prices.loc[common_index[-1]]
        benchmark_return = ((bm_end - bm_start) / bm_start) * 100
        alpha = total_return - benchmark_return
 
        total_trades = len(completed_trades)
        winning_trades = sum(1 for r in completed_trades if r > 0)
        win_rate = (winning_trades / total_trades * 100) if total_trades > 0 else 0.0
 
        print(f"\nPortfolio Simulation Results")
        print(f"  Train window : {train_df.index.min().date()} to {train_df.index.max().date()}")
        print(f"  Test window  : {test_df.index.min().date()} to {test_df.index.max().date()}")
        print(f"  Strategy Return   : {total_return:+.2f}%")
        print(f"  Benchmark ({benchmark_prices.name}) : {benchmark_return:+.2f}%")
        print(f"  Alpha             : {alpha:+.2f}%")
        print(f"  Win Rate          : {win_rate:.2f}%")
        print(f"  Total Trades      : {total_trades}")
 
        self.last_strategy_return  = total_return
        self.last_benchmark_return = benchmark_return
        self.last_alpha            = alpha
        self.last_win_rate         = win_rate
        self.last_total_trades     = total_trades
        self.last_trade_log        = pd.DataFrame(trade_log)

if __name__ == "__main__":
    features_rf = {
        'sma_trend_regime': {
            'func': indicators.calculate_sma_crossover,
            'params': {'fast_window': 10, 'slow_window': 30, 'binary': True},
        },
        'sma_position': {
            'func': indicators.calculate_sma_position,
            'params': {'window': 25},
        },
        'rsi': {
            'func': indicators.calculate_rsi,
            'params': {'window': 14},
        },
        'bollinger_position': {
            'func': indicators.calculate_bollinger_position,
            'params': {'window': 20, 'num_std': 2},
        },
        'price_roc': {
            'func': indicators.calculate_roc,
            'params': {'window': 5}
        }
    }
 
    ticker_pool = [
        "NVDA", "MSFT", "AVGO", "NOW",
        "ORCL", "AAPL", "TEAM", "INTC",
        "SNOW", "WIX",  "AMD",  "CSCO",
        "SHOP", "AMZN", "CRM",
    ]
 
    benchmark = "SPY"

    rf_classifier = ensemble.RandomForestClassifier(
        n_estimators=100,
        random_state=42,
        min_samples_split=10,
        max_features='sqrt',
        max_depth=10
    )
 
    engine = PortfolioBacktestEngine(
        model=rf_classifier,
        feature_configs=features_rf,
        confidence_threshold=0.75,
        stop_loss=0.05,
        min_hold_days=2,
        adx_threshold=20,
        training_years=2,
        testing_years=1,
        offset_years=0
    )
 
    total_years = engine.training_years + engine.testing_years + engine.offset_years
    print(f"Downloading {total_years}y of data for {len(ticker_pool)} tickers + {benchmark}...")
    master_df = stock_screener.fetch_screener_data(
        ticker_pool + [benchmark], period=f"{total_years}y", interval="1d"
    )
    print("Download Complete.")
 
    benchmark_prices = master_df['Close'][benchmark]
    benchmark_prices.name = benchmark
 
    engine.run_simulation(
        ticker_pool=ticker_pool,
        pre_downloaded_df=master_df,
        benchmark_prices=benchmark_prices,
    )
 
    print("\nTrade Log:")
    print(engine.last_trade_log.to_string(index=False))