from abc import ABCMeta, abstractmethod
import pandas as pd
import numpy as np
import datetime
#from exceptions import *


class Backtest:

    def __init__(self,
                 data: pd.DataFrame,
                 universe: list,
                 strategy: object,
                 cash: float = 10_000,
                 ):

        data = self.check_params(universe, data.copy())

        self.full_data = data.copy() # Simply a copy of dataframe
        self.result = None # Result dataframe (returned on the end of backtest)
        self.broker = Broker(universe=universe, cash=cash, data=self.full_data)
        self.universe = universe #list(data.columns.levels[1]) #data.columns = pd.MultiIndex.from_product([data.columns, ['AAPL']])
        self.strategy = strategy(self.broker)
        

    def check_params (self, universe, data):
        """ Check and fix input params """

        if len(universe) < 1:
            raise Exception("Your universe of stocks is lower than one. Please fill in the ticker of the stocks you wish to backtest.")
        
        # Check if data is in multindex format, in case not, then convert it assuming you are backtesting only one stock
        if type(data.columns) != pd.core.indexes.multi.MultiIndex:
            if len(universe) == 1:
                print("Converting data to multi index!")
                data.columns = pd.MultiIndex.from_product([ data.columns, [universe[0]] ]) # Convert index to multindex          
            else:
                raise Exception("You are backtesting more than one stock, but your data is not in multi index format")

        return data



    def run (self):

        iteration_list = []
        equity_list = []

        # Set strategy's fixed variables
        self.strategy.universe = self.universe

        # ----> MAIN LOOP

        for i in range (0, len(self.full_data), 1):
            
            # Set strategy's variables
            self.strategy.data = self.full_data.iloc[0:i+1].copy()
            self.strategy.index = self.strategy.data.index
            self.strategy.open = self.strategy.data.loc[:, "Open"]
            self.strategy.high = self.strategy.data.loc[:, "High"]
            self.strategy.low = self.strategy.data.loc[:, "Low"]
            self.strategy.close = self.strategy.data.loc[:, "Adj Close"]
            self.strategy.volume = self.strategy.data.loc[:, "Volume"]
            self.strategy.iteration = i # Tracks simulation step (which iteration are we running?)

            # ----> RUN STRATEGY (Run inside a try so in case an error happens, backtest does not fail/stop)

            # Broker must be updated before strategy, so when the strategy runs on close, it already knows the updated positions and equity
            self.broker.update(iteration=i, last_close=self.strategy.close.iloc[-1]) # Update broker (send current prices so broker can update positions)

            self.strategy.positions = self.broker.positions.transpose().to_dict() # strategy.position is the broker.position but with few changes for user friendliness

            # Call strategy.on_start(), strategy.next() and strategy.on_end() at this stage, because all the data/variables are available for the strategy
            try:
                # This is the first function run (Ran when the first candle of data is available - iteration=1)
                if i == 0: self.strategy.on_start()
                
                self.strategy.next() # Call strategy.next() which has the MAIN LOGIC (This is also ran on the same candle of strategy.on_start() and strategy.on_end())

                # This is the last function run (Ran when the iteration is at the last candle of data)
                if i == len(self.full_data) - 1: self.strategy.on_end()

            # This is normally raised when next() tries something like: self.close[-3] when the index is not yet available 
            except IndexError:
                pass
            except Exception as e:
                print(e)
                raise e
                

            # Stop backtest if broker has no equity left!
            if self.broker.equity <= 0:
                raise Exception("Out of money")

            # Append params from backtest to result dataframe
            iteration_list.append(self.broker.iteration)
            equity_list.append(self.broker.equity)


        # Set dataframe result
        self.result = self.strategy.data.copy()

        # Create columns for result dataframe
        self.result.loc[:, "_iteration"] = iteration_list
        self.result.loc[:, "_equity"] = equity_list

        return self.result # Return result


    def statistics (self):

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
        
        self.universe = universe
        self.balance = cash
        self.equity = self.balance

        self.orders = [] # All orders are stored here

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

        self.trades = []
        self.tradebook = pd.DataFrame()

        self.iteration = 0 # Keep track of simulation iteration
        self.last_close = None # Last close that is available to strategy (contains index and closing prices)


    def update (self,
                iteration: int,
                last_close: pd.Series
                ):
        
        """
        This function is responsible for keeping the broker updated. Each iteration this should be called so the broker can update its parameters.

        last_close is going to be a Pandas Series, like:
            AAPL    164.076080
            GOOG    127.960999
            Name: 2022-04-18 00:00:00, dtype: float64

        Examples:
            last_close.name = 2022-04-18 00:00:00
            last_close["AAPL"] = 164.076080
            last_close.loc["AAPL"] = 164.076080
        """
        self.iteration = iteration
        self.last_close = last_close

        # Update position and equity for the broker
        self.positions = self._calc_positions(orderbook=self.orderbook, last_close=self.last_close)
        self.equity = self._calc_equity(balance=self.balance, positions=self.positions)

        # Update trades that are still OPEN (running)
        for trade in self.trades:
            if trade.status == "open":
                trade.update()


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
        price = (self.last_close[ticker] + slippage) if (action == "buy") else (self.last_close[ticker] - slippage)
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
        self.update(iteration=self.iteration, last_close=self.last_close)

        return order


    @staticmethod
    def _calc_positions (
                         orderbook: pd.DataFrame,
                         last_close: pd.Series
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

            price = last_close[ticker]
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
        self.closed_at_iteration = -1

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

        self.positions = Broker._calc_positions(orderbook=self.orderbook, last_close=self.broker.last_close) # Helper function from broker to calculate new positions
        self.pl = Broker._calc_equity(balance=self.balance, positions=self.positions) # Helper function from broker to calculate PL/trade equity
        self.check_stop()


    # NOT BEING USED - Check if we can close a trade
    def check_close (self):
        """
        Check is a trade caan  be closed. This function is NOT BEING USED
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
        self.positions = Broker._calc_positions(orderbook=self.orderbook, last_close=self.broker.last_close) # Helper function from broker to calculate new positions
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
        
        trade = Trade(id=len(self.broker.trades), broker=self.broker, orders=orders, description=description)

        self.broker.trades.append(trade)
        self.broker.tradebook = pd.DataFrame(data={
                                                   'id': [trade.id for trade in self.broker.trades],
                                                   'status': [trade.status for trade in self.broker.trades], 
                                                   'description': [trade.description for trade in self.broker.trades], 
                                                   'pl': [trade.pl for trade in self.broker.trades], 
                                                   'created_at_iteration': [trade.created_at_iteration for trade in self.broker.trades],
                                                   'closed_at_iteration': [trade.closed_at_iteration for trade in self.broker.trades],
                                                   'sl': [trade.sl for trade in self.broker.trades],
                                                   'tp': [trade.tp for trade in self.broker.trades],
                                                   'max_age': [trade.max_age for trade in self.broker.trades]
                                                   }
                                            ).set_index("id")

        return trade


    def close_trade (self,
                     trade: Trade
                     ):
        """
        Close a trade
        """

        return trade.close()
    
