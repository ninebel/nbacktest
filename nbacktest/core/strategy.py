
from abc import ABCMeta, abstractmethod

class Strategy (metaclass=ABCMeta):

    def __init__(self,
                 broker: Broker
                 ):
        
        self.broker: Broker = broker
        self.df_positions: dict = None
        self.data: pd.DataFrame = None
        self.index: pd.DataFrame = None
        self.price: pd.DataFrame = None
        self.iteration: int = None


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
             commission: float = 0,
             slippage: float = 0
             ) -> Order:
        """
        Request broker to place a BUY/LONG order
        """ 
        return self.broker.place_order(action="buy", ticker=ticker, quantity=int(quantity), commission=commission, slippage=slippage)


    def sell (self,
              ticker: str,
              quantity: int,
              commission: float = 0,
              slippage: float = 0
              ) -> Order:
        """
        Request broker to place a SELL/SHORT order
        """
        return self.broker.place_order(action="sell", ticker=ticker, quantity=int(quantity), commission=commission, slippage=slippage)


    def create_trade (self,
                      orders: list[Order],
                      description: str = ""
                      ) -> Trade:
        """
        Create a new trade
        """
        if not orders:  # If there are no orders in the list
            return None

        return self.broker.create_trade(orders=orders, description=description)



    def close_trade (self,
                     trade: Trade
                     ):
        """
        Close a trade
        """

        return trade.close()