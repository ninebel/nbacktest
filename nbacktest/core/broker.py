import pandas as pd
from ..utils import *
from .entities import Order, Trade


class BaseBroker:
    
    def __init__(self, universe: list[str], cash: float, data: pd.DataFrame = None):
        self._universe = universe
        self._balance = cash
        self._equity = cash
        self._iteration = 0
        self._last_prices = None
        self._orders: list["Order"] = []
        self._trades: list["Trade"] = []
        self._positions = {}
        self._positions_total = 0.0

    
    def _update(self, iteration: int, last_prices: pd.Series):
        
        self._iteration = iteration
        self._last_prices = last_prices

        # Update positions based on filled orders
        self._positions = self._get_positions(self._orders, self._last_prices)
        self._positions_total = sum(pos["value"] for pos in self._positions.values())
        self._equity = self._balance + self._positions_total

        # Update open trades
        for trade in self._trades:
            if trade._status == "OPEN":
                trade._update()

    
    def _place_order(self,
                    action: str,
                    ticker: str,
                    quantity: int,
                    price: float,
                    fee: float = 0.0
                   ):
        """
        Base place_order only creates order object
        """

        order = Order(
            broker=self,
            action=action,
            ticker=ticker,
            quantity=quantity,
            price=price,
            fee=fee,
            status="pending"
        )

        self._orders.append(order)
        
        return order

    
    def _create_trade(self, orders: list, notes: str):
        trade = Trade(broker=self, orders=orders, notes=notes)
        self._trades.append(trade)

        return trade


    @staticmethod
    def _get_positions(orders: list, last_prices: pd.Series):
        positions = {}
        for order in orders:
            ticker = order._ticker
            if ticker not in positions:
                positions[ticker] = {"quantity": 0, "value": 0.0}
            positions[ticker]["quantity"] += order._filled_quantity

        # Remove zero qty positions
        positions = {k: v for k, v in positions.items() if v["quantity"] != 0}

        # Calculate position value based on last price
        for ticker, position in positions.items():
            position["value"] = position["quantity"] * last_prices.get(ticker, 0)

        return positions


class BacktestBroker(BaseBroker):
    
    def _place_order(self,
                     action: str,
                     ticker: str,
                     quantity: int,
                     price: float,
                     fee: float = 0.0
                    ):
        
        order = super()._place_order(action=action, 
                                    ticker=ticker,
                                    quantity=quantity,
                                    price=price,
                                    fee=fee
                                   )

        # Fill immediately (simulate instant fill)
        order._fill(quantity=quantity,
                   price=price,
                   fee=fee
                  )

        return order



class RealBroker(BaseBroker):
    
    def _place_order(self, action: str, ticker: str, quantity: int, price: float, fee: float = 0.0):
        """
        Real broker would send order to external system, might be async.
        Here we just create order but do NOT fill immediately.
        Filling logic happens elsewhere (e.g., via market updates or API callbacks).
        """
        order = super()._place_order(action, ticker, quantity, price, fee)
        # Real implementation: send order to broker API
        # No immediate fill or balance update here.
        return order
