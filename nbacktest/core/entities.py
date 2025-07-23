
import numpy as np
import pandas as pd
import datetime

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
            })

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

