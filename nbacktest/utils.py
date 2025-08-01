import pandas as pd


def auto_properties(mapping):
    """
    mapping: dict of property name -> attribute path or (path, transform_fn)
    """
    def make_property(path, transform):
        def prop(self):
            val = self
            for part in path.split('.'):
                val = getattr(val, part)
            return transform(val) if transform else val
        return property(prop)

    def decorator(cls):
        for name, value in mapping.items():
            if hasattr(cls, name):
                continue # Skip if already defined
            if isinstance(value, tuple):
                path, transform = value
            else:
                path, transform = value, None
            setattr(cls, name, make_property(path, transform))
        return cls
    return decorator


def build_orderbook(orders: list["Order"]):
    order_dicts = []
    for order in orders:
        d = order.__dict__.copy()

        # Remove unneeded or complex objects
        d.pop('_broker', None)

        order_dicts.append(d)

    orderbook = pd.DataFrame(order_dicts)

    orderbook.columns = [col.lstrip('_').upper() for col in orderbook.columns]
    return orderbook


def build_tradebook(trades: list["Trade"]):
    trade_dicts = []
    for trade in trades:
        d = trade.__dict__.copy()

        # Remove unneeded or complex objects
        d.pop('_broker', None)
        d.pop('_orders', None)
        d.pop('_positions', None)
        d.pop('_balance', None)

        trade_dicts.append(d)

    tradebook = pd.DataFrame(trade_dicts)

    tradebook.columns = [col.lstrip('_').upper() for col in tradebook.columns]
    return tradebook
