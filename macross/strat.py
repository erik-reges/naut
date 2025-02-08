from decimal import Decimal
from collections import deque

from nautilus_trader.core.nautilus_pyo3 import BarType, InstrumentId
from nautilus_trader.trading.strategy import Strategy
from nautilus_trader.config import StrategyConfig
from nautilus_trader.indicators.average.ma_factory import (
    MovingAverageFactory,
    MovingAverageType,
)
from nautilus_trader.model.enums import OrderSide

class MACrossConfig(StrategyConfig, frozen=True):
    instrument_id: InstrumentId      # e.g. "AAPL"
    bar_type: BarType                # e.g. "AAPL-1MINUTE"
    trade_size: str                  # e.g. "100"

class MACrossStrategy(Strategy):
    def __init__(self, config: MACrossConfig) -> None:
        super().__init__(config)
        self.trade_size = Decimal(config.trade_size)
        # Create indicator instances using the built-in factory.
        self.ma5 = MovingAverageFactory.create(5, MovingAverageType.SIMPLE)
        self.ma10 = MovingAverageFactory.create(10, MovingAverageType.SIMPLE)
        self.ma20 = MovingAverageFactory.create(20, MovingAverageType.SIMPLE)

        # Optionally, maintain historical values if needed.
        self.ma5_history = deque()
        self.ma10_history = deque()
        self.ma20_history = deque()

    def on_start(self) -> None:
        self.instrument = self.cache.instrument(self.config.instrument_id)
        if self.instrument is None:
            self.log.error(f"Could not find instrument {self.config.instrument_id}")
            self.stop()
            return
        self.subscribe_bars(self.config.bar_type)

        # Register indicators to be automatically updated with new bar data.
        self.register_indicator_for_bars(self.config.bar_type, self.ma5)
        self.register_indicator_for_bars(self.config.bar_type, self.ma10)
        self.register_indicator_for_bars(self.config.bar_type, self.ma20)
        self.log.info("MACrossStrategy started using built-in MA classes.")

    def on_bar(self, bar) -> None:
        # Optionally, store historical indicator values if you need them later.
        self.ma5_history.appendleft(self.ma5.value)
        self.ma10_history.appendleft(self.ma10.value)
        self.ma20_history.appendleft(self.ma20.value)

        # Wait until indicators are properly initialized.
        if (not self.ma5.initialized or
            not self.ma10.initialized or
            not self.ma20.initialized):
            return

        # If not in a position, check the entry condition.
        if self.portfolio.is_flat(self.config.instrument_id):
            if self.ma5.value > self.ma10.value and self.ma5.value > self.ma20.value:
                self.log.info(f"Entry condition met: MA5={self.ma5.value:.2f} > MA10={self.ma10.value:.2f} and MA20={self.ma20.value:.2f}")
                self.buy()
        else:
            # Exit as soon as the relationship breaks.
            if self.ma5.value <= self.ma10.value or self.ma5.value <= self.ma20.value:
                self.log.info(f"Exit condition met: MA5={self.ma5.value:.2f} <= MA10={self.ma10.value:.2f} or MA5={self.ma5.value:.2f} <= MA20={self.ma20.value:.2f}")
                self.close_all_positions(self.config.instrument_id)


    def buy(self) -> None:
        qty = self.instrument.make_qty(self.trade_size)
        self.log.info(f"make_qty output: {qty} (type: {type(qty)})")
        # Do not convert qty to int; pass the Quantity object directly.
        order = self.order_factory.market(
            instrument_id=self.config.instrument_id,
            order_side=OrderSide.BUY,  # Use the enum, not a string.
            quantity=qty
        )
        self.submit_order(order)
        self.log.info("Submitted BUY order using MA indicator signal.")


    def on_stop(self) -> None:
        self.log.info("Stopping MACrossStrategy; closing positions and canceling orders.")
        self.cancel_all_orders(self.config.instrument_id)
        self.close_all_positions(self.config.instrument_id)
