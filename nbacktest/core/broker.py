import pandas as pd
import requests
from ..utils import *
from .entities import Order, Trade
import time


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
                    fee: float = 0.0,
                    notes: str = ""
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
            status="pending",
            notes=notes
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
                qty = order._quantity_filled
            else:
                qty = order._quantity_requested - order._quantity_filled

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
                     fee: float = 0.0,
                     notes: str = ""
                    ):
        
        order = super()._place_order(action=action, 
                        ticker=ticker,
                        quantity=quantity,
                        price=price,
                        fee=fee,
                        notes=notes
                       )

        # Fill immediately (simulate instant fill)
        order._fill(quantity=quantity,
                   price=price,
                   fee=fee
                  )

        return order



class RealBroker(BaseBroker):
    
    def __init__(self, universe: list[str], cash: float = 0.0, base_url: str = "", data: pd.DataFrame = None, session: requests.sessions.Session | None = None):
        """
        Real broker that talks to HTTP endpoints directly.

        base_url example: "http://metatrader_api:8080"
        session can be a configured requests.Session; defaults to requests module.
        """
        super().__init__(universe=universe, cash=cash, data=data)
        self._base_url = base_url.rstrip("/")
        self._in_fill_update = False  # guard to avoid recursion when order._fill triggers broker._update

    
    def _post_market_order(self, *, symbol: str, side: str, volume: float, deviation: int = 1, magic: int = 20250, comment: str = "nbacktest", take_profit: float = 0.0, stop_loss: float = 0.0) -> dict:
        url = f"{self._base_url}/orders/market"
        payload = {
            "symbol": symbol,
            "side": side,
            "volume": volume,
            "deviation": deviation,
            "magic": magic,
            "comment": comment,
            "take_profit": take_profit,
            "stop_loss": stop_loss,
        }
        resp = requests.post(url, json=payload, headers={"accept": "application/json"})
        if resp.status_code >= 400:
            raise RuntimeError(f"Order placement failed: HTTP {resp.status_code} {resp.text}")
        try:
            #print('post_market_order response:', resp.text)
            return resp.json()
        except Exception:
            return {}

    
    def _get_orders(self, ticket: int | None = None) -> list[dict]:
        url = f"{self._base_url}/orders"
        params = {"ticket": ticket} if ticket is not None else None
        try:
            resp = requests.get(url, params=params, headers={"accept": "application/json"})
            if resp.status_code >= 400:
                return []
            #print('get_orders response:', resp.text)
            return resp.json() or []
        except Exception:
            return []

    
    def _get_orders_history(self, ticket: int) -> list[dict]:
        """
        Fetch closed orders history to retrieve execution details (e.g., price_current)
        when the open orders endpoint no longer returns the ticket.
        """
        url = f"{self._base_url}/orders/history"
        params = {"ticket": ticket}
        try:
            resp = requests.get(url, params=params, headers={"accept": "application/json"})
            if resp.status_code >= 400:
                return []
            #print('get_orders_history response:', resp.text)
            return resp.json() or []
        except Exception:
            return []

    
    def _place_order(self, action: str, ticker: str, quantity: int, price: float, fee: float = 0.0, notes: str = ""):
        """
        Send a market order via HTTP and register the local order. Fees are ignored (set to 0).
        """
        side = "buy" if quantity > 0 else "sell"
        resp = self._post_market_order(symbol=ticker, side=side, volume=abs(quantity))

        # Prefer broker-executed price; if missing, raise since executed price is required
        if not isinstance(resp, dict):
            raise RuntimeError("Order placement failed: unexpected response format")

        resolved_price = resp.get("price")
        if resolved_price is None:
            raise RuntimeError("Order placement failed: missing execution price in response")

        time.sleep(1)
        order = super()._place_order(action, ticker, quantity, resolved_price, fee=0.0, notes=notes)
        # Keep a reference to provider-side id for reconciliation (use order field explicitly)
        order._provider_id = resp["order"]
        return order


    def _sync_orders(self):
        """
        Pull remote order states and update local orders with fills or terminal statuses.

        Supports MetaTrader-like responses containing keys such as:
        ticket, state, volume_initial, volume_current, price_open, price_current, symbol.
        """
        # Query per-order to respect ticket filtering
        remote_index = {}
        for order in self._orders:
            provider_id = getattr(order, "_provider_id", None)
            if provider_id is None:
                continue
            remote_list = self._get_orders(ticket=provider_id)
            if not remote_list:
                # Try history endpoint for filled/closed orders
                history_list = self._get_orders_history(ticket=provider_id)
                remote_index[provider_id] = history_list or []
            else:
                # assume first entry corresponds to the ticket
                remote_index[provider_id] = remote_list

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
            if provider_id is None:
                continue

            ros = remote_index.get(provider_id, [])

            if ros == []:
                # API returns only open/partial; empty means fully filled
                filled_abs = abs(order._quantity_requested)
                price = self._last_prices.get(order._ticker, order._price_requested) if self._last_prices is not None else order._price_requested
            else:
                ro = ros[0]
                volume_initial = ro.get("volume_initial")
                volume_current = ro.get("volume_current")
                if volume_initial is not None and volume_current is not None:
                    filled_abs = volume_initial - volume_current
                else:
                    filled_abs = ro.get("filled_volume")

                if filled_abs is None:
                    continue

                price = ro.get("price_current") or ro.get("price_open") or (self._last_prices.get(order._ticker, order._price_requested) if self._last_prices is not None else order._price_requested)

            local_filled_abs = abs(order._quantity_filled)
            delta_abs = filled_abs - local_filled_abs
            if delta_abs > 0:
                sign = 1 if order._quantity_requested > 0 else -1
                qty_delta = delta_abs * sign
                try:
                    self._in_fill_update = True
                    order._fill(quantity=qty_delta, price=price, fee=0.0)
                finally:
                    self._in_fill_update = False

            if ros and ros != []:
                status_code = ro.get("state")
                status = mt_state.get(status_code, str(ro.get("status", "")).upper())
                if status in {"CANCELLED", "CANCELED"}:
                    order._cancel()
                elif status == "REJECTED":
                    order._reject()
                elif status == "EXPIRED":
                    order._expire()

    
    def _update(self, iteration: int, last_prices: pd.Series):
        # If re-entered via order._fill, run base bookkeeping immediately to reflect fills in the same iteration
        # without triggering another sync cycle.
        if self._in_fill_update:
            return BaseBroker._update(self, iteration=iteration, last_prices=last_prices)

        self._iteration = iteration
        self._last_prices = last_prices

        # First, reconcile with remote broker
        self._sync_orders()

        # Then proceed with standard bookkeeping
        return super()._update(iteration=iteration, last_prices=last_prices)
