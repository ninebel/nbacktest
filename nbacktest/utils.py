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
            if isinstance(value, tuple):
                path, transform = value
            else:
                path, transform = value, None
            setattr(cls, name, make_property(path, transform))
        return cls
    return decorator

