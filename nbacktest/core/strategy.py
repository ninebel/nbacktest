from abc import ABCMeta, abstractmethod
from ..utils import *


@auto_properties({
    "data": "_data",
    "alternative_data": "_alternative_data",
    "iteration": "_iteration",
    "prices": "_prices",
    "universe": "_broker._universe",
    "balance": "_broker._balance",
    "positions": "_broker._positions",
    "positions_total": "_broker._positions_total",
    "equity": "_broker._equity",
    "orders": ("_broker._orders", tuple),
    "trades": ("_broker._trades", tuple),
})
class Strategy (metaclass=ABCMeta):

    def __init__(self,
                 broker: "BacktestBroker"
                 ):
        
        self._broker: "BacktestBroker" = broker
        self._data: pd.DataFrame = None
        self._alternative_data: pd.DataFrame = None
        self._iteration = None
        self._prices: pd.DataFrame = None

    
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
             price: float = None,
             fee: float = 0.0
             ):
        """
        Request broker to place a BUY/LONG order.
        """ 
        price = self._broker._last_prices[ticker] if price is None else price

        if ticker not in self._broker._universe:
            raise Exception(f"Ticker {ticker} not in universe")
        if quantity == 0:
            raise Exception("Quantity must be non-zero")
        if price <= 0:
            raise Exception("Price must be positive")

        return self._broker._place_order(action="BUY",
                                         ticker=ticker,
                                         quantity=abs(quantity),
                                         price=price,
                                         fee=fee
                                        )


    def sell (self,
              ticker: str,
              quantity: int,
              price: float = None,
              fee: float = 0.0
              ):
        """
        Request broker to place a SELL/SHORT order.
        """
        price = self._broker._last_prices[ticker] if price is None else price
        
        if ticker not in self._broker._universe:
            raise Exception(f"Ticker {ticker} not in universe")
        if quantity == 0:
            raise Exception("Quantity must be non-zero")
        if price <= 0:
            raise Exception("Price must be positive")
        
        return self._broker._place_order(action="SELL",
                                         ticker=ticker,
                                         quantity=-abs(quantity),
                                         price=price,
                                         fee=fee
                                        )


    def fill_order(order: "Order",
                   quantity: int,
                   price: float,
                   fee: float = 0.0
                  ):
        """
        Fills an order with a given quantity, price, and optional fee, while enforcing validity checks.
        """
        if quantity == 0:
            raise ValueError("Filled quantity must be non-zero")
            
        if abs(order.filled_quantity) + abs(quantity) > abs(order.requested_quantity):
            raise ValueError("Filled quantity exceeds requested quantity")

        quantity = abs(quantity) if order.action == "BUY" else -abs(quantity)
        
        return order._fill(quantity=quantity, price=price, fee=fee)

    
    def cancel_order(self, order):
        """
        Cancel an order.
        """
        return order._cancel()

    
    def reject_order(self, order):
        """
        Reject an order.
        """
        return order._reject()

    
    def expire_order(self, order):
        """
        Expire an order.
        """
        return order._expire()
    
    
    def create_trade(self,
                     orders: list["Order"],
                     notes: str = ""
                    ):
        """
        Create a new trade.
        """
        if not orders:  # If there are no orders in the list
            raise Exception("Can not create a trade with no orders")
        return self._broker._create_trade(orders=orders, notes=notes)


    def close_trade (self,
                     trade: "Trade"
                    ):
        """
        Close a trade.
        """
        return trade._close()

    
    def add_order_to_trade (self,
                            trade: "Trade",
                            order: list["Order"],
                           ):
        """
        Add an order to trade.
        """
        return trade._add_order(order)