from abc import ABCMeta, abstractmethod
import numpy as np
import pandas as pd
import datetime
import uuid
from nbacktest.core.entities import Order, Trade


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
            })

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
            })


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

        id = str(uuid.uuid4())
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

        trade = Trade(id=str(uuid.uuid4()), broker=self, orders=orders, description=description)

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
        )
        
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
        df_positions = df_orderbook.groupby(['TICKER']).sum(numeric_only=True)[["QUANTITY", "TOTAL"]].rename(columns={"TOTAL":"TOTAL_INVESTED"})
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
