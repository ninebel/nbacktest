import uuid
from ..utils import *


@auto_properties({
    "id": "_id",
    "ticker": "_ticker",
    "action": "_action",
    "status": "_status",
    "broker": "_broker",
    "notes": "_notes",
    "iteration_requested": "_iteration_requested",
    "quantity_requested": "_quantity_requested",
    "price_requested": "_price_requested",
    "fee_requested": "_fee_requested",
    "gross_total_requested": "_gross_total_requested",
    "total_requested": "_total_requested",
    "iteration_filled": "_iteration_filled",
    "quantity_filled": "_quantity_filled",
    "price_filled": "_price_filled",
    "fee_filled": "_fee_filled",
    "gross_total_filled": "_gross_total_filled",
    "total_filled": "_total_filled",
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
                 status: str = "PENDING",
                 notes: str = ""
                 ) -> None:

        self._id = str(uuid.uuid4())
        self._ticker = ticker
        self._action = action
        self._status = status
        self._broker = broker
        self._notes = notes
        self._trade_id = None
        
        self._iteration_requested = self._broker._iteration
        print('order',self._iteration_requested)
        self._quantity_requested = quantity
        self._price_requested = price
        self._fee_requested = -abs(fee) if fee > 0 or fee < 0 else 0  # fees are always money going out
        self._gross_total_requested = -(self._quantity_requested * self._price_requested) # quantity > 0 (BUY) and quantity < 0 (SELL), meaning money going out for BUY and money coming in for SELL
        self._total_requested = self._gross_total_requested + self._fee_requested

        self._iteration_filled = None
        self._quantity_filled = 0
        self._price_filled = None
        self._fee_filled = 0.0
        self._gross_total_filled = 0.0
        self._total_filled = 0

    
    def _fill(self,
             quantity: int,
             price: float,
             fee: float = 0.0
            ):
        """
        Record a fill event, updating filled quantities, costs, fees, and order status.
        """
        self._iteration_filled = self._broker._iteration
        self._quantity_filled += quantity
        self._fee_filled += fee
        fill_cost = -(price * quantity) # quantity > 0 (BUY) and quantity < 0 (SELL), meaning money going out for BUY and money coming in for SELL
        fee = -abs(fee) # fees are always money going out

        # Update average weighted price
        if self._price_filled is None:
            self._price_filled = price
        else:
            current_total_cost = self._gross_total_filled + fill_cost
            self._price_filled = abs(current_total_cost / self._quantity_filled)

        # Update totals
        self._gross_total_filled += fill_cost
        self._total_filled += fill_cost + fee

        # Update Status
        if abs(self._quantity_filled) < abs(self._quantity_requested):
            self._status = "UNFILLED"
        else:
            self._status = "FILLED"
            self._broker._balance += self._total_filled

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
        # Guard against None values for partially filled/unfilled orders when formatting
        filled_price_str = f"{self._price_filled:.2f}" if self._price_filled is not None else "-"
        return (f"<Order {self._action} {self._quantity_requested} {self._ticker} "
            f"at {self._price_requested:.2f} | Filled Quantity: {self._quantity_filled} | Filled Price: {filled_price_str} "
            f"| Filled Total: {self._total_filled:.2f} | Status: {self._status}>")


@auto_properties({
    "id": "_id",
    "status": "_status",
    "notes": "_notes",
    "balance": "_balance",
    "pnl": "_pnl",
    "created_iteration": "_created_iteration",
    "closed_iteration": "_closed_iteration",
    "age": "_age",
    "reason_closed": "_reason_closed",
    "orders": ("_orders", tuple),
    "positions_filled": ("_positions_filled", dict),
    "positions_filled_total": "_positions_filled_total",
    "positions_unfilled": ("_positions_unfilled", dict),
    "positions_unfilled_total": "_positions_unfilled_total",
    "stop_loss": "_stop_loss",
    "take_profit": "_take_profit",
    "max_age": "_max_age",
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
        self._age = None
        self._reason_closed = None
        self._orders = []
        self._positions_filled = {}
        self._positions_filled_total = 0
        self._positions_unfilled = {}
        self._positions_unfilled_total = 0
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
        self._created_iteration = min(order.iteration_requested for order in self._orders)
        self._balance += order._total_filled
        self._update()
        return True


    def _update(self):
        """
        Update values and check if stop conditions are triggered.
        """
        self._positions_filled = self._broker._get_positions(self._orders, self._broker._last_prices, quantity="filled")
        self._positions_unfilled = self._broker._get_positions(self._orders, self._broker._last_prices, quantity="unfilled")
        self._positions_filled_total = sum(position["value"] for position in self._positions_filled.values())
        self._positions_unfilled_total = sum(position["value"] for position in self._positions_unfilled.values())
        self._pnl = self._balance + self._positions_filled_total
        self._age = self._broker._iteration - self._created_iteration
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
            if self._age >= self._max_age:
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
        for ticker, pos in self._positions_filled.items():
            quantity = pos["quantity"]
            if quantity > 0:
                order = self._broker._place_order(action="SELL", ticker=ticker, quantity=-abs(quantity), price=self._broker._last_prices[ticker])
                self._add_order(order)
            elif quantity < 0:
                order = self._broker._place_order(action="BUY", ticker=ticker, quantity=abs(quantity), price=self._broker._last_prices[ticker])
                self._add_order(order)

        self._closed_iteration = max(order._iteration_filled for order in self._orders)
        self._update()
        return True


