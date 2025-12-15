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
        self._positions_filled = {}
        self._positions_unfilled = {}
        self._positions_filled_total = 0.0
        self._positions_unfilled_total = 0.0

    
    def _update(self, iteration: int, last_prices: pd.Series):
        
        self._iteration = iteration
        self._last_prices = last_prices

        # Update positions based on filled and unfilled portions of orders
        self._positions_filled = self._get_positions(self._orders, self._last_prices, quantity="filled")
        self._positions_unfilled = self._get_positions(self._orders, self._last_prices, quantity="unfilled")
        self._positions_filled_total = sum(position["value"] for position in self._positions_filled.values())
        self._positions_unfilled_total = sum(position["value"] for position in self._positions_unfilled.values())
        self._equity = self._balance + self._positions_filled_total

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
    def _get_positions(orders: list, last_prices: pd.Series, quantity: str = "filled"):
        """
        Compute position quantities and values based on a quantity selector.

        quantity options:
        - "filled": use filled quantities (current actual positions)
        - "unfilled": use requested minus filled (outstanding/pending quantities)
        """

        if quantity not in {"filled", "unfilled"}:
            raise ValueError("quantity must be 'filled' or 'unfilled'")

        positions = {}
        for order in orders:
            ticker = order._ticker
            if ticker not in positions:
                positions[ticker] = {"quantity": 0, "value": 0.0}
            if quantity == "filled":
                qty = order._filled_quantity
            else:
                qty = order._requested_quantity - order._filled_quantity

            positions[ticker]["quantity"] += qty

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
    def __init__(self, universe: list[str], cash: float, api_client, data: pd.DataFrame = None):
        super().__init__(universe=universe, cash=cash, data=data)
        self._api = api_client
        self._in_fill_update = False  # guard to avoid recursion when order._fill triggers broker._update

    def _place_order(self, action: str, ticker: str, quantity: int, price: float, fee: float = 0.0):
        """
        Send a market order to the external broker API and register the local order.
        Fees are ignored (set to 0 for now).
        """
        side = "buy" if quantity > 0 else "sell"
        try:
            resp = self._api.market_order(symbol=ticker, side=side, volume=abs(quantity))
        except Exception as exc:
            raise RuntimeError(f"Order placement failed: {exc}")

        # Use price hint from response if provided, otherwise requested price or last known price
        price_hint = None
        if isinstance(resp, dict):
            price_hint = resp.get("price") or resp.get("price_open") or resp.get("price_current")
        resolved_price = price_hint or price or (self._last_prices.get(ticker, 0) if self._last_prices is not None else 0)

        order = super()._place_order(action, ticker, quantity, resolved_price, fee=0.0)
        # Keep a reference to provider-side id for reconciliation
        if isinstance(resp, dict):
            if "id" in resp:
                order._provider_id = resp["id"]
            elif "ticket" in resp:
                order._provider_id = resp["ticket"]
        return order

    def _sync_orders(self):
        """
        Pull remote order states and update local orders with fills or terminal statuses.

        Supports MetaTrader-like responses containing keys such as:
        ticket, state, volume_initial, volume_current, price_open, price_current, symbol.
        """
        try:
            remote_orders = self._api.list_orders()
        except Exception:
            return  # fail soft; keep last known state

        if not remote_orders:
            return

        remote_index = {}
        for ro in remote_orders:
            if not isinstance(ro, dict):
                continue
            rid = ro.get("id") or ro.get("ticket")
            if rid is None:
                continue
            remote_index[rid] = ro

        # MetaTrader state mapping (best-effort defaults)
        mt_state = {
            0: "STARTED",
            1: "PLACED",
            2: "CANCELLED",
            3: "PARTIAL",
            4: "FILLED",
            5: "REJECTED",
            6: "EXPIRED",
        }

        for order in self._orders:
            provider_id = getattr(order, "_provider_id", None)
            if not provider_id or provider_id not in remote_index:
                continue

            ro = remote_index[provider_id]

            volume_initial = ro.get("volume_initial")
            volume_current = ro.get("volume_current")
            if volume_initial is not None and volume_current is not None:
                filled_abs = volume_initial - volume_current
            else:
                filled_abs = ro.get("filled_volume")

            if filled_abs is None:
                continue

            local_filled_abs = abs(order._filled_quantity)
            delta_abs = filled_abs - local_filled_abs
            if delta_abs > 0:
                sign = 1 if order._requested_quantity > 0 else -1
                qty_delta = delta_abs * sign

                price = ro.get("avg_price") or ro.get("price_current") or ro.get("price_open") or order._requested_price
                try:
                    self._in_fill_update = True
                    order._fill(quantity=qty_delta, price=price, fee=0.0)
                finally:
                    self._in_fill_update = False

            status_code = ro.get("state")
            status = mt_state.get(status_code, str(ro.get("status", "")).upper())
            if status in {"CANCELLED", "CANCELED"}:
                order._cancel()
            elif status == "REJECTED":
                order._reject()
            elif status == "EXPIRED":
                order._expire()

    def _update(self, iteration: int, last_prices: pd.Series):
        # If this update is re-entered via order._fill, skip to avoid recursion; main caller will run after sync.
        if self._in_fill_update:
            return

        self._iteration = iteration
        self._last_prices = last_prices

        # First, reconcile with remote broker
        self._sync_orders()

        # Then proceed with standard bookkeeping
        return super()._update(iteration=iteration, last_prices=last_prices)
