class Backtest:

    def __init__(self,
                 data: pd.DataFrame,
                 universe: list[str],
                 strategy: object,
                 price_column: str = "Close",
                 cash: float = 100_000,
                 safety_lock: bool = False,
                 alternative_data: pd.DataFrame = None,
                 slicing_column: str = None
                 ):
        """
        Create Backtest instance

        data: pd.DataFrame, no default value
            Pandas dataframe containing OHLC data (must have the following columns: Open, High, Low, Close, Adj Close, Volume)

        universe: list[str], no default value
            List containing all the tradeable assets you wish to backtest.
            For example, if you want to create a strategy to trade FAANGs (Facebook/Meta, Amazon, Apple, Netflix and Alphabet), your universe is:
            ["META", "AMZN", "AAPL", "NFLX", "GOOG"]

        strategy: object, no default value
            Your strategy class name (not instance, but class name).

        price_column: str, default "Close"
            Name of the reference price column used in backtesting.
            It is recommended that price_column is the closing price, the price taken from this column will be used for filling order price and calculating position value.
            Example: price_column = "Close" or price_column = "Adj Close"

        cash: float, default 100_000
            Initial amount of cash you want to start with

        safety_lock: bool, default False
            Special feature that blocks the user from altering backtest data inside a strategy. This is useful if the user is inexperienced, but on the other side, it makes it harder to export features created during strategy execution.
            Example: Let's suppose you want to create a custom indicator directly on the self.data, setting the safety_lock to False will let you get the custom indicator with the
            result dataframe, making it much easier to debug and analyze your backtest! 
            If you set the safety_lock to True, the custom indicator data will not be returned with the result dataframe.

        alternative_data: pd.DataFrame, default None
            Dataframe having alternate data for your strategy. Alternative data is sliced according to the condition that alternative_data[slicing_column] <= data.index, where both alternative_data's and data's indexes are datetimes.

        slicing_column: str, default None
            Column used for slicing the alternative_data, following the condition that alternative_data[slicing_column] <= data.index
        """

        data = self.check_parameters(universe, data, price_column) # Check source data and return a copy of data

        self.full_data = data # Simply a copy of dataframe
        self.result = None # Result dataframe (returned on the end of backtest)
        self.broker = Broker(universe=universe, cash=cash, data=self.full_data)
        self.universe = universe
        self.strategy = strategy(self.broker)
        self.price_column = price_column
        self.safety_lock = safety_lock
        self.alternative_data = alternative_data
        self.slicing_column = slicing_column


    def check_parameters(self,
                          universe: list[str],
                          data: pd.DataFrame,
                          price_column: str
                         ) -> pd.DataFrame:
        """Check and fix input parameters"""

        data = data.copy()

        # Check universe
        if len(universe) < 1:
            raise Exception("Your universe of stocks is lower than one.")

        # Check price column
        if price_column is None:
            raise Exception("price_column was not defined!")

        if price_column not in data.columns:
            raise Exception("price_column does not exist!")

        # Check if data is in multindex format, in case not, then convert it assuming you are backtesting only one stock
        if not isinstance(data.columns, pd.MultiIndex):
            if len(universe) == 1:
                data.columns = pd.MultiIndex.from_product([data.columns, [universe[0]]])
                print("Automatically converting to multi index format!")
            else:
                raise Exception("You are backtesting more than one stock, but your data is not in multi index format!")

        return data


    def run (self):

        """ 
        Run backtest (Main loop)
        """

        # Create lists to hold the values to be inserted as columns to the final result dataframe at the end of the backtest
        iterations = []
        equities = []

        # Set strategy's fixed variables
        self.strategy.universe = self.universe

        # ----> MAIN LOOP

        for i in range (0, len(self.full_data), 1):

            if self.safety_lock:
                # Main data
                self.strategy.data = self.full_data.iloc[0:i+1].copy()
                # Alternative data
                if self.alternative_data is not None: self.strategy.alternative_data = self.alternative_data.loc[self.alternative_data[self.slicing_column]  <= self.strategy.data.index[-1]].copy()

            else:
                # Main data
                self.strategy.data = self.full_data.iloc[0:i+1]
                # Alternative data
                if self.alternative_data is not None: self.strategy.alternative_data = self.alternative_data.loc[self.alternative_data[self.slicing_column] <= self.strategy.data.index[-1]]

            self.strategy.index = self.strategy.data.index
            self.strategy.price = self.strategy.data.loc[:, self.price_column]
            self.strategy.iteration = i # Tracks simulation step (which iteration are we running?)

            # ----> RUN STRATEGY (Run inside a try so in case an error happens, backtest does not fail/stop)

            # Broker must be updated before strategy, so when the strategy runs on close, it already knows the updated df_positions and equity
            self.broker.update(iteration=i, last_price=self.strategy.price.iloc[-1]) # Update broker (send current prices so broker can update df_positions)

            self.strategy.df_positions = self.broker.df_positions.transpose().to_dict() # strategy.position is the broker.position but with few changes for user friendliness

            # Call strategy.on_start(), strategy.next() and strategy.on_end() at this stage, because all the data/variables are available for the strategy
            try:
                # This is the first function run (Ran when the first candle of data is available - iteration=1)
                if i == 0: self.strategy.on_start()
                
                self.strategy.next() # Call strategy.next() which has the MAIN LOGIC (This is also ran on the same candle of strategy.on_start() and strategy.on_end())

                # This is the last function run (Ran when the iteration is at the last candle of data)
                if i == len(self.full_data) - 1: self.strategy.on_end()

            # This is normally raised when next() tries something like: self.close[-3] when the index is not yet available
            # Index Error won't stop backtest, you are probably trying to access data that is not available yet!
            except IndexError as e:
                print("Index Error: %s" % e)

            except Exception as e:
                print(e)
                raise e
                

            # Stop backtest if broker has no equity left!
            if self.broker.equity <= 0:
                #raise Exception("Out of money") # Disabled for now
                pass

            # Append params from backtest to result dataframe
            iterations.append(self.broker.iteration)
            equities.append(self.broker.equity)


        # Set dataframe result
        self.result = self.strategy.data.copy()

        # Create columns for result dataframe
        self.result.loc[:, "ITERATION"] = iterations
        self.result.loc[:, "EQUITY"] = equities

        return self.result # Return result


    def statistics (self):
        """
        Return some key statistics from backtest
        """

        if self.result is None:
            print("Run the backtest before accessing statistics!")
            return
        
        df_wins = self.broker.df_tradebook.query("PNL > 0")
        df_losses = self.broker.df_tradebook.query("PNL <= 0")
        n_won = len(df_wins)
        n_lost = len(df_losses)
        n_total = n_won + n_lost
        if n_total > 0:
            win_rate = n_won/n_total
            avg_abs_return = self.broker.df_tradebook["PNL"].sum() / n_total # This is also the expected value per trade (EV)
            avg_abs_return_per_win = df_wins["PNL"].sum() / n_won
            avg_abs_return_per_lost = df_losses["PNL"].sum() / n_lost
        else:
            win_rate = 0
            avg_abs_return = 0
            avg_abs_return_per_win = 0
            avg_abs_return_per_lost = 0

        statistics = {
                      "n_won": n_won,
                      "n_lost": n_lost,
                      "n_total": n_total,
                      "win_rate": win_rate,
                      "avg_abs_return": avg_abs_return,
                      "avg_abs_return_per_win": avg_abs_return_per_win,
                      "avg_abs_return_per_lost": avg_abs_return_per_lost
                     }
        
        return statistics

