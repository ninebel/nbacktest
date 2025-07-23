# Version of your package (standard practice)
__version__ = "1.0.0" # Example version, update as needed

# Import key classes/functions from sub-packages to expose them at the top level
# This allows users to do, e.g., 'from nbacktest import Backtest, ManualBroker'
from .backtest import Backtest
from .core import Order, Trade, Broker, ManualBroker, Strategy
