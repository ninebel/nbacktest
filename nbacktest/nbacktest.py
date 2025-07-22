from abc import ABCMeta, abstractmethod
import pandas as pd
import numpy as np
import datetime


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


class Broker:

    def __init__(self,
                 universe: list,
                 cash: float,
                 data: pd.DataFrame
                 ):

        # -----> STRATEGY MANAGEMENT

        self.universe = universe # Universe of tradeable stocks
        self.balance = cash # Balance
        self.equity = self.balance # At the start, equity is equal to balance, as you have df_positions
        self.iteration = 0 # Keep track of simulation iteration
        self.last_price = None # Last close that is available to strategy (contains index and closing prices)


        # -----> ORDER MANAGEMENT

        self.orders: list[Order] = [] # All orders are stored here

        # Order history
        self.df_orderbook = None

        # -----> POSITION MANAGEMENT

        # Positions - all df_positions are considered to be OPEN, when a position is closed, it is removed from this dataframe
        self.df_positions = pd.DataFrame(data={
                                            'QUANTITY':[],
                                            'VALUE':[]
                                           }
                                     )


         # -----> TRADE MANAGEMENT

        self.trades: list[Trade] = [] # All trades are stored here

        # Trade history
        self.df_tradebook = None
    

    def update (self,
                iteration: int,
                last_price: pd.Series
                ):
        
        """
        This function is responsible for keeping the broker updated. Each iteration this should be called so the broker can update its parameters.

        last_price is a Pandas Series, like:
            AAPL    164.076080
            GOOG    127.960999
            Name: 2022-04-18 00:00:00, dtype: float64

        Examples:
            last_price.name = 2022-04-18 00:00:00
            last_price["AAPL"] = 164.076080
            last_price.loc["AAPL"] = 164.076080
        """

        self.iteration = iteration
        self.last_price = last_price

        # Update df_orderbook
        self.df_orderbook = pd.DataFrame(
            data={
                'ID': [order.id for order in self.orders],
                'ITERATION': [order.iteration for order in self.orders],
                'ACTION': [order.action for order in self.orders],
                'TICKER': [order.ticker for order in self.orders],
                'QUANTITY': [order.quantity for order in self.orders],
                'PRICE': [order.price for order in self.orders],
                'COMMISSION': [order.commission for order in self.orders],
                'SLIPPAGE': [order.slippage for order in self.orders],
                'TOTAL': [order.total for order in self.orders],
                'STATUS': [order.status for order in self.orders],
            }).set_index('ID')

        # Update df_positions
        self.df_positions = self._calc_positions(df_orderbook=self.df_orderbook, last_price=self.last_price)

        # Update equity
        self.equity = self._calc_equity(balance=self.balance, df_positions=self.df_positions)

        # Update trades that are still OPEN (running)
        for trade in self.trades:
            if trade.status == "open":
                trade.update()

        # Update df_tradebook
        self.df_tradebook = pd.DataFrame(
            data={
                'ID': [trade.id for trade in self.trades],
                'STATUS': [trade.status for trade in self.trades],
                'DESCRIPTION': [trade.description for trade in self.trades],
                'PNL': [trade.pl for trade in self.trades],
                'CREATED_AT_ITERATION': [trade.created_at_iteration for trade in self.trades],
                'CLOSED_AT_ITERATION': [trade.closed_at_iteration for trade in self.trades],
                'STOP_LOSS': [trade.sl for trade in self.trades],
                'TAKE_PROFIT': [trade.tp for trade in self.trades],
                'MAX_AGE': [trade.max_age for trade in self.trades],
                'REASON_CLOSED': [trade.reason_closed for trade in self.trades],
            }).set_index('ID')


    def place_order (self,
                     action: str,
                     ticker: str,
                     quantity: int,
                     commission: float = 0,
                     slippage: float = 0
                     ):
        """
        Place an order, this function checks all parameters and if everything is correct, then execute the order.
        If action is "sell" its a SELL/SHORT order, else if action is "buy" its BUY/LONG order. 
        In case it is a "sell" order, quantity is negative (you lose stocks) and total is positive (you receive money from selling).
        If it is a "buy" order, quantity is positive (you get stocks) and total is negative (you lose money).
        """

        # ----> CHECK PRIMARY PARAMS

        if action != "sell" and action != "buy":
            raise Exception("action must be either 'buy' or 'sell'")
        
        # If user is buying/selling a non-existing stock
        if ticker not in self.universe: raise Exception("%s is not in given universe of stocks!" % ticker)

        # Check ticker
        if type(ticker) != str: raise Exception("ticker must be a string!")
        
        # Check quantity
        if type(int(quantity)) != int: raise Exception("quantity must be an int!")
        else:
            if quantity == 0: raise Exception("quantity must not be zero or null!")
        
        # ----> CALCULATE PARAMS

        id = len(self.orders)
        iteration = self.iteration
        action = action.lower() # Format action
        quantity = abs(quantity) if (action == "buy") else -abs(quantity)
        price = (self.last_price[ticker] + slippage) if (action == "buy") else (self.last_price[ticker] - slippage)
        total = -abs(quantity*price) - commission if (action == "buy") else abs(quantity*price) - commission

        # ----> CHECK CALCULATED PARAMS
        if price < 0:
            raise Exception("price is lower than 0, check price and slippage")

        # Check whether broker has enough cash to buy a stock (in case its a sell order, nothing is checked because the broker is borrowing money)
        if action == "buy" and total >= self.balance:
            raise Exception("Broker does not have enough cash to execute buy order")

        # ----> PROCESS ORDER (Assume orders are instantaneously filled)

        # Create new order instance
        order = Order(id=id,
                      iteration=iteration,
                      action=action,
                      ticker=ticker,
                      quantity=quantity,
                      price=price,
                      commission=commission,
                      slippage=slippage,
                      total=total,
                      status="filled"  # Assume orders are instantaneously filled
                     )

        # Update cash balance. Total<0 if buying, Total>0 if selling
        self.balance += order.total

        # Append newly created order to a list of orders
        self.orders.append(order)

        # Update broker equity, df_positions and trades
        self.update(iteration=self.iteration, last_price=self.last_price)

        return order


    def create_trade (self,
                      orders: list,
                      description: str
                      ):

        trade = Trade(id=len(self.trades), broker=self, orders=orders, description=description)

        self.trades.append(trade)

        self.df_tradebook = pd.DataFrame(
            data={
                'ID': [trade.id for trade in self.trades],
                'STATUS': [trade.status for trade in self.trades], 
                'DESCRIPTION': [trade.description for trade in self.trades], 
                'PNL': [trade.pl for trade in self.trades], 
                'CREATED_AT_ITERATION': [trade.created_at_iteration for trade in self.trades],
                'CLOSED_AT_ITERATION': [trade.closed_at_iteration for trade in self.trades],
                'STOP_LOSS': [trade.sl for trade in self.trades],
                'TAKE_PROFIT': [trade.tp for trade in self.trades],
                'MAX_AGE': [trade.max_age for trade in self.trades],
                'REASON_CLOSED': [trade.reason_closed for trade in self.trades],
            }
        ).set_index('ID')
        
        return trade


    @staticmethod
    def _calc_positions (
                         df_orderbook: pd.DataFrame,
                         last_price: pd.Series
                        ):
        """
        Calculate current open df_positions from df_orderbook

        Get df_positions organized by ticker, like:
        +----------+------------+------------------+----------+
        | TICKER   |   QUANTITY |   TOTAL_INVESTED |    VALUE |
        |----------+------------+------------------+----------|
        | AAPL     |        -10 |           451.2  | -1642.95 |
        | GOOG     |        -10 |           549.13 | -1272.53 |
        +----------+------------+------------------+----------+
        """
        df_positions = df_orderbook.groupby(['TICKER']).sum(numeric_only=True).drop(columns=["ITERATION", "PRICE"]).rename(columns={"TOTAL":"TOTAL_INVESTED"})

        df_positions = df_positions[df_positions["QUANTITY"] != 0] # Remove all rows where quantity is 0

        df_positions["VALUE"] = np.nan

        for ticker in df_positions.index:

            price = last_price[ticker]
            quantity = df_positions.loc[ticker, "QUANTITY"]
            df_positions.loc[ticker, "VALUE"] = price * quantity

        return df_positions


    @staticmethod
    def _calc_equity (
                      balance: float,
                      df_positions: pd.DataFrame
                      ):
        """
        Calculate equity based on cash and position value.
        Equity = Cash + Value of all positions
        """
        return balance + df_positions["VALUE"].sum()


class Order:

    def __init__ (self,
                  id: int,
                  iteration: int,
                  action: str,
                  ticker: str,
                  quantity: int,
                  price: float,
                  commission: float,
                  slippage: float,
                  total: float,
                  status: str = "open"
                  ) -> None:
        """
        Creates an order, all parameters are checked and edited
        """

        self.id = id
        self.iteration = iteration
        self.action = action
        self.ticker = ticker
        self.quantity = quantity
        self.price = price
        self.commission = commission
        self.slippage = slippage
        self.total = total
        self.status = status


class Trade:
    """
    A trade is a container for grouped orders representing a single operation (e.g., long or short).
    Tracks status, PNL, and closure conditions such as stop loss, take profit, max age, or manual.

    Attributes:
        id (int): Unique identifier of the trade.
        broker (Broker): Reference to the broker managing the trade.
        orders (list[Order]): List of executed orders in the trade.
        df_orderbook (pd.DataFrame): Log of all orders related to the trade.
        df_positions (pd.DataFrame): Current open positions in the trade.
        description (str): Optional text describing the trade purpose or logic.
        stop_loss (float): Stop loss threshold.
        take_profit (float): Take profit threshold.
        max_age (int): Maximum duration in iterations before the trade is closed.
        reason_closed (str): Reason the trade was closed ("manual", "stop_loss", "take_profit", or "max_age").
    """

    def __init__ (self,
                  id: int,
                  broker: Broker,
                  orders: list[Order],
                  description: str = ""
                  ):
        
        self.id = id
        self.broker = broker
        self.status = "open" # open / closed
        self.description = description
        self.balance = 0
        self.pnl = None
        self.created_at_iteration = min(order.iteration for order in orders) # earliest order
        self.closed_at_iteration = np.inf # latest order
        self.reason_closed = None

        # Initialize empty order list and orderbook dataframe
        self.orders = []
        self.df_orderbook = None

        # Track current net positions by ticker
        self.df_positions = pd.DataFrame(data={
            'QUANTITY':[], 'VALUE':[]
        })

        for order in orders:
            self.add_order(order)

        # Set default stop conditions to disabled
        self.stop_loss = -np.inf
        self.take_profit = np.inf
        self.max_age = np.inf


    def add_order(self, order: Order):
        
        self.orders.append(order)
        # Update trade cash balance with order total
        self.balance += order.total
        return self.orders


    def update(self):
        """
        Update position values and check if stop conditions are triggered
        """
        self.df_orderbook = pd.DataFrame(
            data={
                'ID': [order.id for order in self.orders],
                'ITERATION': [order.iteration for order in self.orders],
                'ACTION': [order.action for order in self.orders],
                'TICKER': [order.ticker for order in self.orders],
                'QUANTITY': [order.quantity for order in self.orders],
                'PRICE': [order.price for order in self.orders],
                'COMMISSION': [order.commission for order in self.orders],
                'SLIPPAGE': [order.slippage for order in self.orders],
                'TOTAL': [order.total for order in self.orders],
                'STATUS': [order.status for order in self.orders],
            }).set_index('ID')

        self.df_positions = Broker._calc_positions(df_orderbook=self.df_orderbook, last_price=self.broker.last_price)
        self.pnl = Broker._calc_equity(balance=self.balance, df_positions=self.df_positions)
        self.check_stop()


    def check_stop(self):
        """
        Check if any of the stop conditions are met and close the trade accordingly.
        """
        if self.pnl <= self.stop_loss:
            self.reason_closed = "stop_loss"
            self.close()
            return
        if self.pnl >= self.take_profit:
            self.reason_closed = "take_profit"
            self.close()
            return
        if self.broker.iteration - self.created_at_iteration >= self.max_age:
            self.reason_closed = "max_age"
            self.close()

    def close(self):
        """
        Close the trade, this will execute counter orders for all open positions and mark the trade as closed. 
        """

        if self.reason_closed is None:
            self.reason_closed = "manual"

        # Mark trade as closed and execute counter orders for open positions
        self.status = "closed"

        for i in range(len(self.df_positions)):
            position = self.df_positions.iloc[i]
            quantity = position["QUANTITY"]
            ticker = position.name

            # If trade is long (buy), sell the position
            if quantity > 0:
                order = self.broker.place_order(action="sell", ticker=ticker, quantity=-1 * quantity)
            # If trade is short (sell), buy to cover the position
            if quantity < 0:
                order = self.broker.place_order(action="buy", ticker=ticker, quantity=-1 * quantity)

            self.add_order(order)

        self.closed_at_iteration = max(order.iteration for order in self.orders)
        self.df_positions = Broker._calc_positions(df_orderbook=self.df_orderbook, last_price=self.broker.last_price)
        self.pnl = Broker._calc_equity(balance=self.balance, df_positions=self.df_positions)


class Strategy (metaclass=ABCMeta):

    def __init__(self,
                 broker: Broker
                 ):
        
        self.broker: Broker = broker
        self.df_positions: dict = None
        self.data: pd.DataFrame = None
        self.index: pd.DataFrame = None
        self.price: pd.DataFrame = None
        self.iteration: int = None


    @abstractmethod
    def on_start (self):
        pass


    @abstractmethod
    def next (self):
        pass


    @abstractmethod
    def on_end (self):
        pass


    def buy (self,
             ticker: str,
             quantity: int,
             commission: float = 0,
             slippage: float = 0
             ) -> Order:
        """
        Request broker to place a BUY/LONG order
        """ 
        return self.broker.place_order(action="buy", ticker=ticker, quantity=int(quantity), commission=commission, slippage=slippage)


    def sell (self,
              ticker: str,
              quantity: int,
              commission: float = 0,
              slippage: float = 0
              ) -> Order:
        """
        Request broker to place a SELL/SHORT order
        """
        return self.broker.place_order(action="sell", ticker=ticker, quantity=int(quantity), commission=commission, slippage=slippage)


    def create_trade (self,
                      orders: list[Order],
                      description: str = ""
                      ) -> Trade:
        """
        Create a new trade
        """
        if not orders:  # If there are no orders in the list
            return None

        return self.broker.create_trade(orders=orders, description=description)



    def close_trade (self,
                     trade: Trade
                     ):
        """
        Close a trade
        """

        return trade.close()
    
