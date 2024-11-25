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
                 alternate_data: pd.DataFrame = None,
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

        alternate_data: pd.DataFrame, default None
            Dataframe having alternate data for your strategy. Alternate data is sliced according to the condition that alternate_data[slicing_column] <= data.index, where both alternate_data's and data's indexes are datetimes.

        slicing_column: str, default None
            Column used for slicing the alternate_data, following the condition that alternate_data[slicing_column] <= data.index
        """

        data = self.check_parameters(universe, data, price_column) # Check source data and return a copy of data

        self.full_data = data # Simply a copy of dataframe
        self.result = None # Result dataframe (returned on the end of backtest)
        self.broker = Broker(universe=universe, cash=cash, data=self.full_data)
        self.universe = universe #list(data.columns.levels[1]) #data.columns = pd.MultiIndex.from_product([data.columns, ['AAPL']])
        self.strategy = strategy(self.broker)
        self.price_column = price_column
        self.safety_lock = safety_lock
        self.alternate_data = alternate_data
        self.slicing_column = slicing_column


    def check_parameters (self, universe, data, price_column):
        """ Check and fix input parameters """

        data = data.copy() # Make a copy of data

        # Check universe
        if len(universe) < 1: raise Exception("Your universe of stocks is lower than one. Please fill in the ticker of the stocks you wish to backtest.")


        # Check price column
        if price_column is None: 

            raise Exception("price_column was not defined!")

        elif price_column not in data.columns:

            raise Exception("price_column does not exist!")


        # Check if data is in multindex format, in case not, then convert it assuming you are backtesting only one stock
        if type(data.columns) != pd.core.indexes.multi.MultiIndex:
            if len(universe) == 1:
                print("Converting data to multi index!")
                data.columns = pd.MultiIndex.from_product([ data.columns, [universe[0]] ]) # Convert index to multindex          
            else:
                raise Exception("You are backtesting more than one stock, but your data is not in multi index format!")


        return data



    def run (self):

        """ 
        Main Loop for running the backtest
        """

        # Create lists to hold the values to be inserted as columns to the final result dataframe at the end of the backtest
        iteration_list = []
        equity_list = []

        # Set strategy's fixed variables
        self.strategy.universe = self.universe

        # ----> MAIN LOOP

        for i in range (0, len(self.full_data), 1):
            
            # Set strategy's variables

            if self.safety_lock:
                # Main data
                self.strategy.data = self.full_data.iloc[0:i+1].copy()
                # Alternate data
                if self.alternate_data is not None: self.strategy.alternate_data = self.alternate_data.loc[self.alternate_data[self.slicing_column]  <= self.strategy.data.index[-1]].copy()

            else:
                # Main data
                self.strategy.data = self.full_data.iloc[0:i+1]
                # Alternate data
                if self.alternate_data is not None: self.strategy.alternate_data = self.alternate_data.loc[self.alternate_data[self.slicing_column] <= self.strategy.data.index[-1]]

            self.strategy.index = self.strategy.data.index
            self.strategy.price = self.strategy.data.loc[:, self.price_column]
            self.strategy.iteration = i # Tracks simulation step (which iteration are we running?)

            # ----> RUN STRATEGY (Run inside a try so in case an error happens, backtest does not fail/stop)

            # Broker must be updated before strategy, so when the strategy runs on close, it already knows the updated positions and equity
            self.broker.update(iteration=i, last_price=self.strategy.price.iloc[-1]) # Update broker (send current prices so broker can update positions)

            self.strategy.positions = self.broker.positions.transpose().to_dict() # strategy.position is the broker.position but with few changes for user friendliness

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
            iteration_list.append(self.broker.iteration)
            equity_list.append(self.broker.equity)


        # Set dataframe result
        self.result = self.strategy.data.copy()

        # Create columns for result dataframe
        self.result.loc[:, "_iteration"] = iteration_list
        self.result.loc[:, "_equity"] = equity_list

        # Add orders markers
        self.result.loc[:, "_buy"] = np.nan
        self.result.loc[:, "_sell"] = np.nan

        for order in self.broker.orders:

            index = self.full_data.index[order.iteration]

            if order.action == "buy":
                self.result.loc[index , "_buy"] = 1
            elif order.action == "sell":
                self.result.loc[index , "_sell"] = 1

        # Add trades markers
        self.result.loc[:, "_trade"] = np.nan

        for trade in self.broker.trades:
            self.result.loc[ 
                            (self.result["_iteration"] >= trade.created_at_iteration) & (self.result["_iteration"] <= trade.closed_at_iteration),
                            "_trade"
                            ] = 1



        return self.result # Return result


    def statistics (self):
        """
        Return some key statistics from backtest
        """

        if self.result is None:
            print("Run the backtest before accessing statistics!")
            return
        
        df_wins = self.broker.tradebook.query("pl > 0")
        df_losses = self.broker.tradebook.query("pl <= 0")

        n_won = len(df_wins)
        n_lost = len(df_losses)
        n_total = n_won + n_lost
        win_rate = n_won/n_total
        avg_abs_return = self.broker.tradebook["pl"].sum() / n_total # This is also the expected value per trade (EV)

        avg_abs_return_per_win = df_wins["pl"].sum() / n_won
        avg_abs_return_per_lost = df_losses["pl"].sum() / n_lost


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
        self.equity = self.balance # At the start, equity is equal to balance, as you have positions
        self.iteration = 0 # Keep track of simulation iteration
        self.last_price = None # Last close that is available to strategy (contains index and closing prices)


        # -----> ORDER MANAGEMENT

        self.orders: list[Order] = [] # All orders are stored here

        # Order history
        self.orderbook = pd.DataFrame(data={
                                            'id':[],
                                            'iteration':[],
                                            'action':[],
                                            'ticker':[],
                                            'quantity':[],
                                            'price':[],
                                            'commission':[],
                                            'slippage':[],
                                            'total':[]
                                           }
                                     ).set_index("id")


        # -----> POSITION MANAGEMENT

        # Positions - all positions are considered to be OPEN, when a position is closed, it is removed from this dataframe
        self.positions = pd.DataFrame(data={
                                            "quantity":[],
                                            "value":[]
                                           }
                                     )


         # -----> TRADE MANAGEMENT

        self.trades: list[Trade] = [] # All trades are stored here

        # Trade history
        self.tradebook = pd.DataFrame(data={
                                                   'id': [],
                                                   'status': [], 
                                                   'description': [], 
                                                   'pl': [], 
                                                   'created_at_iteration': [],
                                                   'closed_at_iteration': [],
                                                   'sl': [],
                                                   'tp': [],
                                                   'max_age': []
                                                   }
                                            ).set_index("id")
    

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

        # Update positions
        self.positions = self._calc_positions(orderbook=self.orderbook, last_price=self.last_price)

        # Update equity
        self.equity = self._calc_equity(balance=self.balance, positions=self.positions)

        # Update trades that are still OPEN (running)
        for trade in self.trades:
            if trade.status == "open":
                trade.update()

        # Update tradebook
        if len(self.trades) >= 1:

            self.tradebook = pd.DataFrame(data={
                                                'id': [trade.id for trade in self.trades],
                                                'status': [trade.status for trade in self.trades], 
                                                'description': [trade.description for trade in self.trades], 
                                                'pl': [trade.pl for trade in self.trades], 
                                                'created_at_iteration': [trade.created_at_iteration for trade in self.trades],
                                                'closed_at_iteration': [trade.closed_at_iteration for trade in self.trades],
                                                'sl': [trade.sl for trade in self.trades],
                                                'tp': [trade.tp for trade in self.trades],
                                                'max_age': [trade.max_age for trade in self.trades]
                                                }
                                        ).set_index("id")


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
                      total=total
                     )

        # Update cash balance. Total<0 if buying, Total>0 if selling
        self.balance += order.total

        # Append newly created order to a list of orders
        self.orders.append(order)

        # Add new 'order row' to orderbook (this works as an incremental load)
        self.orderbook.loc[order.id] = [
                                        order.iteration,
                                        order.action,
                                        order.ticker,
                                        order.quantity,
                                        order.price,
                                        order.commission,
                                        order.slippage,
                                        order.total
                                       ]

        # Update broker equity, positions and trades
        self.update(iteration=self.iteration, last_price=self.last_price)

        return order


    def create_trade (self,
                      orders: list,
                      description: str
                      ):

        trade = Trade(id=len(self.trades), broker=self, orders=orders, description=description)

        self.trades.append(trade)

        self.tradebook = pd.DataFrame(data={
                                            'id': [trade.id for trade in self.trades],
                                            'status': [trade.status for trade in self.trades], 
                                            'description': [trade.description for trade in self.trades], 
                                            'pl': [trade.pl for trade in self.trades], 
                                            'created_at_iteration': [trade.created_at_iteration for trade in self.trades],
                                            'closed_at_iteration': [trade.closed_at_iteration for trade in self.trades],
                                            'sl': [trade.sl for trade in self.trades],
                                            'tp': [trade.tp for trade in self.trades],
                                            'max_age': [trade.max_age for trade in self.trades]
                                            }
                                    ).set_index("id")
        
        return trade


    @staticmethod
    def _calc_positions (
                         orderbook: pd.DataFrame,
                         last_price: pd.Series
                        ):
        """
        Calculate current open positions from orderbook

        Get positions organized by ticker, like:
        +----------+------------+------------------+----------+
        | ticker   |   quantity |   total_invested |    value |
        |----------+------------+------------------+----------|
        | AAPL     |        -10 |           451.2  | -1642.95 |
        | GOOG     |        -10 |           549.13 | -1272.53 |
        +----------+------------+------------------+----------+
        """
        positions = orderbook.groupby(['ticker']).sum(numeric_only=True).drop(columns=["iteration","price"]).rename(columns={"total":"total_invested"})

        positions = positions[positions["quantity"] != 0] # Remove all rows where quantity is 0

        positions["value"] = np.nan

        for ticker in positions.index:

            price = last_price[ticker]
            quantity = positions.loc[ticker, "quantity"]
            positions.loc[ticker, "value"] = price * quantity

        return positions


    @staticmethod
    def _calc_equity (
                      balance: float,
                      positions: pd.DataFrame
                      ):
        """
        Calculate equity based on cash and position value.
        Equity = Cash + Value of all positions
        """
        return balance + positions["value"].sum()


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
                  total: float
                  ) -> None:
        """
        Creates an order, all parameters are checked and edited. The current "status" parameter is not implemented, so it doesn't affect the backtest at all!
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


class Trade:

    def __init__ (self,
                  id: int,
                  broker: Broker,
                  orders: list[Order],
                  description: str = ""
                  ):
        """
        A trade is the same as an operation in the common sense, they are ways of grouping orders (like a buy, then sell order are a LONG trade).
        Trades are supposed to be very flexible and customizable, so you track a lot of different trading strategies using them.
        """

        # Variables
        self.id = id
        self.broker = broker
        self.status = "open" # possible options: "open" / "closed"
        self.description = description # A simple description for the trade. Reason for the trade? What are you trading?
        self.balance = 0
        self.pl = None # Profit/Loss per share (example: buy price - sell price)
        self.created_at_iteration = min(order.iteration for order in orders)
        self.closed_at_iteration = np.inf

        # Order and position handling
        self.orders = []

        # Order history
        self.orderbook = pd.DataFrame(data={
                                            'id':[],
                                            'iteration':[],
                                            'action':[],
                                            'ticker':[],
                                            'quantity':[],
                                            'price':[],
                                            'commission':[],
                                            'slippage':[],
                                            'total':[]
                                           }
                                     ).set_index("id")

        # All open positions
        self.positions = pd.DataFrame(data={
                                            "quantity":[],
                                            "value":[]
                                           }
                                     )

        for order in orders:
            self.add_order(order)

        # Stops - Automatically close trade/operation
        self.sl = -np.inf # Stop Loss
        self.tp = np.inf # Take Profit
        self.max_age = np.inf # Max age in bars/candles/iterations
        

    def add_order (self,
                   order: Order
                   ):
        """
        Adds an order to the trade and update orderbook.
        """
        # Append newly created order to a list of orders
        self.orders.append(order)

        # Add new 'order row' to orderbook (this works as an incremental load)
        self.orderbook.loc[order.id] = [
                                        order.iteration,
                                        order.action,
                                        order.ticker,
                                        order.quantity,
                                        order.price,
                                        order.commission,
                                        order.slippage,
                                        order.total
                                       ]

        self.balance += order.total

        return self.orders


    def update (self):
        """
        Update trade positions and PL. This function is called every iteration by the broker.
        """       

        self.positions = Broker._calc_positions(orderbook=self.orderbook, last_price=self.broker.last_price) # Helper function from broker to calculate new positions
        self.pl = Broker._calc_equity(balance=self.balance, positions=self.positions) # Helper function from broker to calculate PL/trade equity
        self.check_stop()


    # NOT BEING USED - Check if we can close a trade
    def check_close (self):
        """
        Check if a trade can be closed. This function is NOT BEING USED
        """

        if len(self.positions) == 0: 
            return True
        elif len(self.positions) != 0:
            for i in range (len(self.positions)):
                if self.positions.iloc[0]["quantity"] != 0:
                    return False
            return True
        return False


    def check_stop (self):
        """
        Checks if any stop condition is met, if so, call self.close() in order to fully close the trade.
        a) Price check (sl and tp): check if our PL hit the stop loss (sl) or tp (take profit)
        b) Trade max. age: check if trade has been opened a self.max_age bars/iterations ago, if so close the trade
        """

        # Price stop
        if self.pl <= self.sl: self.close()
        elif self.pl >= self.tp: self.close()

        # Max age stop (if trade has been running for max_age iterations/bars, close it!)
        if self.broker.iteration - self.created_at_iteration >= self.max_age: self.close()


    def close (self):
        """
        Close a trade. If there any open positions, orders are placed to close positions. When these are completed, the trade is considered to be closed.
        If there are no open positions, final PL and status will be updated, closing the trade.
        """
        self.status = "closed" # Keep this line here, if it is at the end of the function the broker will be stuck in a loop trying to close this trade if sl or tp is activated!

        # Close still open positions
        for i in range (len(self.positions)):

            position = self.positions.iloc[i]
            quantity = position["quantity"]
            ticker = position.name

            # If its a LONG position, close with a SELL/SHORT order
            if quantity > 0: 
                order = self.broker.place_order(action="sell", ticker=ticker, quantity=-1*quantity)

            # If its a SHORT position, close with a BUY/SHORT order
            if quantity < 0: 
                order = self.broker.place_order(action="buy", ticker=ticker, quantity=-1*quantity)
            
            self.add_order(order)

        # Effectively close trade, calculating final PL and updating status
        self.closed_at_iteration = max(order.iteration for order in self.orders)
        self.positions = Broker._calc_positions(orderbook=self.orderbook, last_price=self.broker.last_price) # Helper function from broker to calculate new positions
        self.pl = Broker._calc_equity(balance=self.balance, positions=self.positions) # Helper function from broker to calculate equity


class Strategy (metaclass=ABCMeta):

    def __init__(self,
                 broker: Broker
                 ):
        
        self.broker: Broker = broker
        self.positions: dict = None
        self.data: pd.DataFrame = None
        self.index: pd.DataFrame = None
        self.open: pd.DataFrame = None
        self.high: pd.DataFrame = None
        self.low: pd.DataFrame = None
        self.close: pd.DataFrame = None
        self.volume: pd.DataFrame = None
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
             ):
        """
        Request broker to place a BUY/LONG order
        """ 
        return self.broker.place_order(action="buy", ticker=ticker, quantity=int(quantity), commission=commission, slippage=slippage)


    def sell (self,
              ticker: str,
              quantity: int,
              commission: float = 0,
              slippage: float = 0
              ):
        """
        Request broker to place a SELL/SHORT order
        """
        return self.broker.place_order(action="sell", ticker=ticker, quantity=int(quantity), commission=commission, slippage=slippage)


    def create_trade (self,
                      orders: list[Order],
                      description: str = ""
                      ):
        """
        Create a new trade
        """
        if not len(orders): # If there are no orders in the list
            return None
        
        trade = self.broker.create_trade(orders=orders,
                                 description=description
                                 )

        return trade


    def close_trade (self,
                     trade: Trade
                     ):
        """
        Close a trade
        """

        return trade.close()
    
