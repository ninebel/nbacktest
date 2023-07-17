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

        self.full_data = data.copy()
        self.broker = Broker(universe=universe, cash=cash, data=self.full_data)
        self.universe = universe #list(data.columns.levels[1]) #data.columns = pd.MultiIndex.from_product([data.columns, ['AAPL']])
        self.strategy = strategy(self.broker)
        

    def run (self):

        self.full_data["Equity"] = np.nan

        # ----> MAIN LOOP

        for i in range (0, len(self.full_data)):
            
            # Set strategy's variables!
            self.strategy.data = self.full_data[0:i+1]
            self.strategy.index = self.strategy.data.index
            self.strategy.open = self.strategy.data["Open"]
            self.strategy.high = self.strategy.data["High"]
            self.strategy.low = self.strategy.data["Low"]
            self.strategy.close = self.strategy.data["Adj Close"]
            self.strategy.volume = self.strategy.data["Volume"]
            self.strategy.universe = self.universe
            self.strategy.iteration = i # Tracks simulation step (which iteration are we running?)

            # ----> RUN STRATEGY (Run inside a try so in case an error happens, backtest doesnt fail)

            # Broker must be updated before strategy, so when the strategy runs on close, it already knows the updated positions and equity
            self.broker.update(iteration=i, last_row=self.strategy.close.iloc[-1]) # Update broker (send current prices so broker can update positions)

            self.strategy.positions = self.broker.positions.transpose().to_dict() # strategy.position is the broker.position but with few changes for user friendliness

            # Call strategy.on_start(), strategy.next() and strategy.on_end() at this stage, because all the data/variables are available for the strategy
            try:
                # This is the first function run (Ran when the first candle of data is available - iteration=1)
                if i == 0: self.strategy.on_start()
                
                self.strategy.next() # Call strategy.next() which has the MAIN LOGIC (This is also ran on the same candle of strategy.on_start() and strategy.on_end())

                # This is the last function run (Ran when the iteration is at the last candle of data)
                if i == len(self.full_data) - 1: self.strategy.on_end()
                
            except IndexError:
                # This is normally raised when next() tries something like: self.close[-3] when the index is not yet available
                pass
            except Exception as e:
                print(e)
                raise e
                print("ERROR:", e)
                

            # Stop backtest if broker has no equity left!
            if self.broker.equity <= 0:
                #break
                raise Exception("Out of money")

            # Append equity data for backtest result
            self.full_data.iloc[i, self.full_data.columns.get_loc('Equity')] = self.broker.equity

        #self.full_data = self.full_data.dropna() # Remove all rows that have NaN (this happens when the simulation didnt finish, most likely went bankrupt)

        return self.full_data


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
        self.orderbook = pd.DataFrame(data={'id':[],'iteration':[], 'timestamp':[], 'action':[], 'ticker':[], 'quantity':[], 'price':[], 'total':[]}).set_index("id") # Order history

        self.positions = pd.DataFrame(data={"quantity":[], "value":[]}) # All open positions

        self.trades = []
        self.tradebook = pd.DataFrame()

        self.iteration = 0 # Keep track of simulation iteration
        self.last_row = None # Last dat row that is available to strategy (containing index, prices, etc)


    def update (self,
                iteration: int,
                last_row: pd.Series
                ):
        
        """
        This function is responsible for keeping the broker updated. Each iteration this should be called so the broker can update its parameters.

        last_prices is going to be a Pandas Series, like:
            AAPL    164.076080
            GOOG    127.960999
            Name: 2022-04-18 00:00:00, dtype: float64

        Examples:
            last_prices.name = 2022-04-18 00:00:00
            last_prices["AAPL"] = 164.076080
            last_prices.loc["AAPL"] = 164.076080
        """
        self.iteration = iteration
        self.last_row = last_row

        # Update position and equity for the broker
        self.positions = self._calc_positions(self.last_row, self.orderbook)
        self.equity = self._calc_equity(self.balance, self.positions)

        # Update trades that are still OPEN (running)
        for trade in self.trades:
            if trade.status == "open":
                trade.update()


    def place_order (self,
                     action: str,
                     quantity: int,
                     ticker: str = ""
                     ):
        """
        Place an order, this function checks all parameters and if everything is correct, then execute the order.
        If action is "sell" its a SELL/SHORT order, else if action is "buy" its BUY/LONG order. 
        In case it is a "sell" order, quantity is negative (you lose stocks) and total is positive (you receive money from selling).
        If it is a "buy" order, quantity is positive (you get stocks) and total is negative (you lose money).
        """
        id = len(self.orders)
        iteration = self.iteration
        timestamp = self.last_row.name
        price = self.last_row[ticker]

        # If user is buying/selling a non-existing stock
        if ticker not in self.universe: raise Exception("%s is not in given universe of stocks!" % ticker)

        # Check whether broker has enough cash to buy a stock (in case its a sell order, nothing is checked because the broker is borrowing money)
        if action == "buy" and quantity*price >= self.balance:
                return None


        order = Order(id=id, iteration=iteration, timestamp=timestamp, action=action, ticker=ticker, quantity=quantity, price=price)

        # ----> PROCESS ORDER (Assume orders are instantaneously filled)

        self.balance += order.total # Update cash balance. Total<0 if buying, Total>0 if selling
        
        # Updates broker's orders
        self.orders.append(order) # Append order
        self.orderbook.loc[order.id] = [order.iteration, order.timestamp, order.action, order.ticker, order.quantity, order.price, order.total] # Add new 'order row' to orderbook (this works as an incremental load)

        # Update broker equity, positions and trades
        self.update(self.iteration, self.last_row)

        return order


    @staticmethod
    def _calc_positions (
                         last_row: pd.Series,
                         orderbook: pd.DataFrame
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

            price = last_row[ticker]
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
                  timestamp: datetime.datetime,
                  action: str,
                  ticker: str,
                  quantity: int,
                  price: float
                  ) -> None:
        """
        Creates an order, all parameters are checked and edited. The current "status" parameter is not implemented, so it doesn't affect the backtest at all!
        """

        self.id = id
        self.iteration = iteration
        self.timestamp = timestamp
        self.action = action
        self.ticker = ticker
        self.quantity = quantity
        self.price = price
        self.total = self.quantity * self.price

        self.check_fix_params() # Check and fix parameters. If this fails an exception will be raised


    def check_fix_params (self):
        """
        This function checks all parameters and if anything fails, it raises an exception
        """

        # ----> CHECK PARAMS

        # Check ticker
        if type(self.ticker) != str: raise Exception("ticker must be a string!")
        
        # Check quantity
        if type(int(self.quantity)) != int: raise Exception("quantity must be an int!")
        else:
            if self.quantity == 0: raise Exception("quantity must not be zero or null!")

        if type(int(self.price)) != int and type(float(self.price)) != float: raise Exception("price must be an int or float!")

        if self.action.lower() != "sell" and self.action.lower() != "buy":
            raise Exception("action must be either 'buy' or 'sell'")
        
        # ----> FIX PARAMS

        self.price = abs(self.price) # Price must be positive
        self.action = self.action.lower() # Format action

        # If its a SELL/SHORT order
        if self.action == "sell": self.quantity = -abs(self.quantity); self.total = abs(self.price*self.quantity)

        # If its a BUY/LONG order
        elif self.action == "buy": self.quantity = abs(self.quantity); self.total = -abs(self.price*self.quantity)


class Trade:

    def __init__ (self,
                  id: int,
                  broker: Broker,
                  orders: list[Order]
                  ):
        """
        A trade is the same as an operation in the common sense, they are ways of grouping orders (like a buy, then sell order are a LONG trade).
        Trades are supposed to be very flexible and customizable, so you track a lot of different trading strategies using them.
        """

        # Variables
        self.id = id
        self.broker = broker
        self.status = "open" # possible options: "open" / "closed"
        self.balance = 0
        self.pl = None # Profit/Loss per share (example: buy price - sell price)
        self.created_at_iteration = min(order.iteration for order in orders)

        # Order and position handling
        self.orders = []
        self.orderbook = pd.DataFrame(data={'id':[],'iteration':[], 'timestamp':[], 'action':[], 'ticker':[], 'quantity':[], 'price':[], 'total':[]}).set_index("id") # Order history
        self.positions = pd.DataFrame(data={"quantity":[], "value":[]}) # All open positions
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
        self.orders.append(order)

        self.orderbook.loc[order.id] = [order.iteration, order.timestamp, order.action, order.ticker, order.quantity, order.price, order.total] # Add new 'order row' to orderbook

        return self.orders


    def update (self):
        """
        Update trade positions and PL. This function is called every iteration by the broker.
        """       

        self.positions = Broker._calc_positions(self.broker.last_row, self.orderbook) # Helper function from broker to calculate new positions
        self.pl = Broker._calc_equity(self.balance, self.positions) # Helper function from broker to calculate PL/trade equity
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
        self.status = "closed"
        # Close still open positions
        for i in range (len(self.positions)):

            position = self.positions.iloc[i]
            quantity = position["quantity"]
            ticker = position.name

            # If its a LONG position, close with a SELL/SHORT order
            if quantity > 0: 
                order = self.broker.place_order(action="sell", ticker=ticker, quantity=-1*quantity)
                self.balance += order.total

            # If its a SHORT position, close with a BUY/SHORT order
            if quantity < 0: 
                order = self.broker.place_order(action="buy", ticker=ticker, quantity=-1*quantity)
                self.balance += order.total
            
            self.add_order(order)

        # Effectively close trade, calculating final PL and updating status
        self.positions = Broker._calc_positions(last_row=self.broker.last_row, orderbook=self.orderbook) # Helper function from broker to calculate new positions
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
             ):
        """
        Place a BUY/LONG order
        """ 
        return self.broker.place_order(action="buy", ticker=ticker, quantity=int(quantity))


    def sell (self,
              ticker: str,
              quantity: int,
              ):
        """
        Place a SELL/SHORT order
        """
        return self.broker.place_order(action="sell", ticker=ticker, quantity=int(quantity))


    def new_trade (self,
                   orders: list[Order]
                   ):
        """
        Create a new trade
        """
        if not len(orders): # If there are no orders in the list
            return None
        
        trade = Trade(id=len(self.broker.trades), broker=self.broker, orders=orders)

        self.broker.trades.append(trade)

        return trade


    def close_trade (self,
                     trade: Trade
                     ):
        """
        Close a trade
        """

        return trade.close()
    






