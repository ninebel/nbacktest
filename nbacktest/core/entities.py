import uuid
from ..utils import *


@auto_properties({
    "id": "_id",
    "ticker": "_ticker",
    "action": "_action",
    "status": "_status",
    "broker": "_broker",
    "requested_iteration": "_requested_iteration",
    "requested_quantity": "_requested_quantity",
    "requested_price": "_requested_price",
    "requested_fee": "_requested_fee",
    "requested_gross_total": "_requested_gross_total",
    "requested_total": "_requested_total",
    "filled_iteration": "_filled_iteration",
    "filled_quantity": "_filled_quantity",
    "filled_price": "_filled_price",
    "filled_fee": "_filled_fee",
    "filled_gross_total": "_filled_gross_total",
    "filled_total": "_filled_total",
})
class Order:
    """
    This class models the lifecycle of a buy or sell order, including requested details,
    partial or full fills, and status transitions (e.g., filled, cancelled, rejected).
    """

    def __init__(self,
                 broker: "Broker",
                 action: str,
                 ticker: str,
                 quantity: int,
                 price: float,
                 fee: float = 0.0,
                 status: str = "PENDING"
                 ) -> None:

        self._id = str(uuid.uuid4())
        self._ticker = ticker
        self._action = action
        self._status = status
        self._broker = broker
        self._trade_id = None
        
        self._requested_iteration = self._broker._iteration
        self._requested_quantity = quantity
        self._requested_price = price
        self._requested_fee = fee
        self._requested_gross_total = self._requested_quantity * self._requested_price
        self._requested_total = self._requested_gross_total - self._requested_fee

        self._filled_iteration = None
        self._filled_quantity = 0
        self._filled_price = None
        self._filled_fee = 0.0
        self._filled_gross_total = 0.0
        self._filled_total = 0

    
    def _fill(self,
             quantity: int,
             price: float,
             fee: float = 0.0
            ):
        """
        Record a fill event, updating filled quantities, costs, fees, and order status.
        """
        self._filled_iteration = self._broker._iteration
        self._filled_quantity += quantity
        self._filled_fee += fee
        fill_cost = price * quantity

        # Update average weighted price
        if self._filled_price is None:
            self._filled_price = price
        else:
            current_total_cost = self._filled_gross_total + fill_cost
            self._filled_price = current_total_cost / self._filled_quantity

        # Update totals
        self._filled_gross_total -= fill_cost
        self._filled_total -= fill_cost + fee

        # Update Status
        if abs(self._filled_quantity) < abs(self._requested_quantity):
            self._status = "LIVE"
        else:
            self._status = "FILLED"
            self._broker._balance += self._filled_total

        # Update Broker
        self._broker._update(iteration=self._broker._iteration, last_prices=self._broker._last_prices)

        return True

    
    def _cancel(self):
        if self._status in {"FILLED", "CANCELLED", "REJECTED", "EXPIRED"}:
            raise RuntimeError(f"Cannot cancel order in status {self._status}.")
        self._status = "CANCELLED"
        return True

    
    def _reject(self):
        if self._status in {"FILLED", "CANCELLED", "REJECTED", "EXPIRED"}:
            raise RuntimeError(f"Cannot reject order in status {self._status}.")
        self._status = "REJECTED"
        return True

    
    def _expire(self):
        if self._status in {"FILLED", "CANCELLED", "REJECTED", "EXPIRED"}:
            raise RuntimeError(f"Cannot expire order in status {self._status}.")
        self._status = "EXPIRED"
        return True

    
    def __repr__(self):
        return (f"<Order {self._action} {self._requested_quantity} {self._ticker} "
                f"at {self._requested_price:.2f} | Filled Quantity: {self._filled_quantity} | Filled Price: {self._filled_price:.2f} "
                f"| Filled Total: {self._filled_total:.2f} | Status: {self._status}>")


@auto_properties({
    "id": "_id",
    "status": "_status",
    "notes": "_notes",
    "balance": "_balance",
    "pnl": "_pnl",
    "created_iteration": "_created_iteration",
    "closed_iteration": "_closed_iteration",
    "reason_closed": "_reason_closed",
    "orders": ("_orders", tuple),
    "positions": ("_positions", tuple),
    "positions_total": "_positions_total",
    "stop_loss": "_stop_loss",
    "take_profit": "_take_profit",
    "max_age": "_max_age"
})
class Trade:
    """
    A trade is a container for grouped orders representing a single operation (e.g., long or short).
    Tracks status, PNL, and closure conditions such as stop loss, take profit, max age, or manual.
    """

    def __init__ (self,
                  broker: "Broker",
                  orders: list["Order"],
                  notes: str = ""
                  ):
        
        self._id = str(uuid.uuid4())
        self._broker = broker
        self._status = "OPEN" # OPEN / CLOSED
        self._notes = notes
        self._balance = 0
        self._pnl = 0
        self._created_iteration = None
        self._closed_iteration = None
        self._reason_closed = None
        self._orders = []
        self._positions = {}
        self._positions_total = 0
        self._stop_loss = None
        self._take_profit = None
        self._max_age = None

        for order in orders:
            self._add_order(order)


    @property
    def stop_loss(self):
        return self._stop_loss
    

    @stop_loss.setter
    def stop_loss(self, value):
        self._stop_loss = value


    @property
    def take_profit(self):
        return self._take_profit
    

    @take_profit.setter
    def take_profit(self, value):
        self._take_profit = value


    @property
    def max_age(self):
        return self._max_age
    

    @max_age.setter
    def max_age(self, value):
        self._max_age = value

    
    def _add_order(self, order: Order):
        """
        Add an order to the internal order list and update the balance accordingly.
        """
        self._orders.append(order)
        order._trade_id = self._id
        self._created_iteration = min(order.requested_iteration for order in self._orders)
        self._balance += order._filled_total
        self._update()
        return True


    def _update(self):
        """
        Update values and check if stop conditions are triggered.
        """
        self._positions = self._broker._get_positions(self._orders, self._broker._last_prices)
        self._positions_total = sum(position["value"] for position in self._positions.values())
        self._pnl = self._balance + self._positions_total
        if self._status == "OPEN":
            self._check_stop()

    
    def _check_stop(self):
        """
        Check if any of the stop conditions are met and close the trade accordingly.
        """
        if self._stop_loss is not None:
            if self._pnl <= self._stop_loss:
                self._close(reason_closed="STOP_LOSS")
                return
        if self._take_profit is not None:
            if self._pnl >= self._take_profit:
                self._close(reason_closed="TAKE_PROFIT")
                return
        if self._max_age is not None:
            if (self._broker._iteration - self._created_iteration) >= self._max_age:
                self._close(reason_closed="MAX_AGE")
                return
 
    
    def _close(self, reason_closed="MANUAL"):
        """
        Close the trade, this will execute counter orders for all open positions and mark the trade as closed. 
        """
        if self._status == "CLOSED":
            raise Exception("Trade is already closed")

        self._reason_closed = reason_closed
        self._status = "CLOSED"
        # Use the latest positions dict to close all open positions
        for ticker, pos in self._positions.items():
            quantity = pos["quantity"]
            if quantity > 0:
                order = self._broker._place_order(action="SELL", ticker=ticker, quantity=-abs(quantity), price=self._broker._last_prices[ticker])
                self._add_order(order)
            elif quantity < 0:
                order = self._broker._place_order(action="BUY", ticker=ticker, quantity=abs(quantity), price=self._broker._last_prices[ticker])
                self._add_order(order)

        self._closed_iteration = max(order._filled_iteration for order in self._orders)
        self._update()
        return True


