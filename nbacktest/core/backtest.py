import numpy as np
import pandas as pd
import statsmodels
import scipy
from ..utils import *
from .broker import BacktestBroker




@auto_properties({"strategy": "_strategy",
                  "result": "_result",
                  "data": "_data",
                  "universe": "_universe",
                  "price_column": "_price_column",
                  "alternative_data": "_alternative_data",
                  "slicing_column": "_slicing_column",
                  "orderbook": "_orderbook",
                  "tradebook": "_tradebook"
})
class Backtest:
    def __init__(self,
                 data: pd.DataFrame,
                 universe: list[str],
                 strategy_class: type,
                 price_column: str = "Close",
                 cash: float = 100_000,
                 alternative_data: pd.DataFrame = None,
                 slicing_column: str = None):
        """
        Initialize a Backtest instance for running trading strategy simulations.

        This class coordinates the execution of a trading strategy over historical data,
        managing the broker, strategy, and optional alternative data sources.

        Parameters
        ----------
        data : pd.DataFrame
            Historical price data. For a single asset, columns should include at least the `price_column`.
            For multiple assets, columns should be a MultiIndex with (field, symbol).
            Example (single asset):
            Date        Open   High    Low   Close
            2020-01-01  100    105     99    104
            Example (multi-asset):
            (Close, AAPL)  (Close, MSFT)
            2020-01-01     104            150
        universe : list[str]
            List of asset symbols to backtest (e.g., ["AAPL", "MSFT"]).
        strategy_class : type
            The class of the trading strategy to use (should accept a broker instance).
        price_column : str, default "Close"
            Name of the price column to use for trading and valuation.
        cash : float, default 100_000
            Initial cash balance for the backtest.
        alternative_data : pd.DataFrame, optional
            Optional DataFrame with alternative data for the strategy (e.g., signals, factors).
        slicing_column : str, optional
            Column in `alternative_data` used to filter rows up to the current iteration.

        Examples
        --------
        >>> bt = Backtest(
        ...     data=price_df,
        ...     universe=["AAPL", "MSFT"],
        ...     strategy_class=MyStrategy,
        ...     price_column="Close",
        ...     cash=1_000_000,
        ...     alternative_data=signals_df,
        ...     slicing_column="date"
        ... )
        >>> results = bt.run()
        >>> stats = bt.statistics()
        """

        self._data = self._validate_parameters(universe, data, price_column)
        self._universe = universe
        self._price_column = price_column
        self._alternative_data = alternative_data
        self._slicing_column = slicing_column
        self._broker = BacktestBroker(universe=universe, cash=cash, data=self._data)
        self._strategy = strategy_class(self._broker)
        self._result = None
        self._orderbook = None
        self._tradebook = None

    
    def _validate_parameters(self, universe, data, price_column) -> pd.DataFrame:
        """
        Validates inputs and formats the data appropriately.
        """
        data = data.copy()

        if len(universe) < 1:
            raise Exception("Your universe of stocks is lower than one.")
        if price_column is None:
            raise Exception("price_column was not defined!")
        if price_column not in data.columns:
            raise Exception("price_column does not exist!")

        if not isinstance(data.columns, pd.MultiIndex):
            if len(universe) == 1:
                data.columns = pd.MultiIndex.from_product([data.columns, [universe[0]]])
                print("Automatically converting to multi index format!")
            else:
                raise Exception("You are backtesting more than one stock, but your data is not in multi index format")

        return data

    def run(self):
        """
        Runs the backtest simulation by iterating over each time step.
        At each iteration:
          - The strategy is fed historical data up to the current point.
          - The broker is updated with the latest price data.
          - The strategy's logic is executed (on_start, next, on_end).
        Collects and stores results including balance, equity, and position info.
        """
        results = []

        for iteration in self._data.index:
            # Slice data up to and including the current index
            self._strategy._data = self._data.loc[:iteration].copy()

            # Slice alternative data if present
            if self._alternative_data is not None and self._slicing_column is not None:
                self._strategy._alternative_data = self._alternative_data.loc[
                    self._alternative_data[self._slicing_column] <= iteration 
                ].copy()

            # Set iteration, price
            self._strategy._iteration = iteration
            self._strategy._prices = self._strategy._data.loc[:, self._price_column]

            # Update the broker with latest prices
            self._broker._update(iteration=iteration, last_prices=self._strategy._prices.iloc[-1])

            # Run strategy lifecycle hooks
            try:
                if self._data.index.get_loc(iteration) == 0:
                    self._strategy.on_start()
                self._strategy.next()
                if self._data.index.get_loc(iteration) == len(self._data) - 1:
                    self._strategy.on_end()
            except IndexError as e:
                print(f"Index Error: {e}")
            except Exception as e:
                print(e)
                raise e

            # Collect performance metrics
            results.append({
                'ITERATION': self._broker._iteration,
                'BALANCE': self._broker._balance,
                'POSITIONS_TOTAL': self._broker._positions_total,
                'EQUITY': self._broker._equity
            })

        self._result = pd.DataFrame(results)
        self._orderbook = build_orderbook(self._broker._orders)
        self._tradebook = build_tradebook(self._broker._trades)
        return self._result


    def statistics(self):
        """
        Return key statistics and advanced deviation metrics from backtest
        """
    
        if self._result is None:
            print("Run the backtest before accessing statistics!")
            return
    
        pnl = self._tradebook["PNL"]
        df_wins = self._tradebook.query("PNL > 0")
        df_losses = self._tradebook.query("PNL <= 0")
    
        n_won = len(df_wins)
        n_lost = len(df_losses)
        n_total = n_won + n_lost
    
        if n_total > 0:
    
            win_rate = n_won / n_total
    
            # Means
            avg_abs_return = pnl.mean()
            avg_abs_return_per_win = df_wins["PNL"].mean() if n_won > 0 else np.nan
            avg_abs_return_per_lost = df_losses["PNL"].mean() if n_lost > 0 else np.nan
    
            # Medians
            median_abs_return = pnl.median()
            median_abs_return_per_win = df_wins["PNL"].median() if n_won > 0 else np.nan
            median_abs_return_per_lost = df_losses["PNL"].median() if n_lost > 0 else np.nan
    
            # Standard Deviation
            std_return = pnl.std()
    
            # Median Absolute Deviation
            mad_return = scipy.stats.median_abs_deviation(pnl, scale='normal')
    
            # Downside deviation (only negative returns)
            downside_dev = np.sqrt(np.mean(np.minimum(pnl, 0)**2))
    
            # Upside deviation (only positive returns)
            upside_dev = np.sqrt(np.mean(np.maximum(pnl, 0)**2))
    
            # Skewness and Kurtosis
            skewness = scipy.stats.skew(pnl)
            kurt = scipy.stats.kurtosis(pnl)
    
            # Mean vs Median gap
            mean_median_gap = avg_abs_return - median_abs_return
    
        else:
            win_rate = np.nan
            avg_abs_return = np.nan
            avg_abs_return_per_win = np.nan
            avg_abs_return_per_lost = np.nan
            median_abs_return = np.nan
            median_abs_return_per_win = np.nan
            median_abs_return_per_lost = np.nan
            std_return = np.nan
            mad_return = np.nan
            downside_dev = np.nan
            upside_dev = np.nan
            skewness = np.nan
            kurt = np.nan
            mean_median_gap = np.nan
    
        statistics = {
            "n_won": n_won,
            "n_lost": n_lost,
            "n_total": n_total,
            "win_rate": win_rate,
            "avg_abs_return": avg_abs_return,
            "avg_abs_return_per_win": avg_abs_return_per_win,
            "avg_abs_return_per_lost": avg_abs_return_per_lost,
            "median_abs_return": median_abs_return,
            "median_abs_return_per_win": median_abs_return_per_win,
            "median_abs_return_per_lost": median_abs_return_per_lost,
            "std_return": std_return,
            "mad_return": mad_return,
            "downside_deviation": downside_dev,
            "upside_deviation": upside_dev,
            "skewness": skewness,
            "kurtosis": kurt,
            "mean_median_gap": mean_median_gap
        }
    
        return statistics

