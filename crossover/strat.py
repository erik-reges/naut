#!/usr/bin/env python3
# -------------------------------------------------------------------------------------------------
#  This file implements a simple Moving Average Crossover strategy for equities.
#  The strategy maintains a state variable (in_position) to track whether it currently
#  holds a long position. When the short moving average (MA) (over a short_window) crosses
#  above the long moving average (over a long_window) and no position is held, it submits a
#  BUY market order. When the short MA falls below the long MA and a long position is held,
#  it submits a SELL market order.
# -------------------------------------------------------------------------------------------------

import numpy as np
from decimal import Decimal
from typing import Optional

# NautilusTrader imports
from nautilus_trader.config import StrategyConfig
from nautilus_trader.trading.strategy import Strategy
from nautilus_trader.model.enums import OrderSide, TimeInForce
from nautilus_trader.model.data import BarType
from nautilus_trader.model.identifiers import PositionId
from nautilus_trader.model.currencies import USD

# --- Strategy configuration ---
class MACrossoverStrategyConfig(StrategyConfig, frozen=True):
    instrument_id: str         # e.g. "SPY.XNAS"
    bar_type: BarType          # e.g. BarType.from_str("SPY.XNAS-1-MINUTE-LAST-EXTERNAL")
    short_window: int = 10     # period for the short moving average
    long_window: int = 30      # period for the long moving average

# --- MA Crossover Strategy implementation ---
class MACrossoverStrategy(Strategy):
    def __init__(self, config: MACrossoverStrategyConfig) -> None:
        super().__init__(config)
        self.prices_history = []       # List to store closing prices
        self.in_position = False       # Tracks whether a position is held
        self.position_id: Optional[PositionId] = None
        self.entry_order = None
        self.exit_order = None

    def on_start(self) -> None:
        self.instrument = self.cache.instrument(self.config.instrument_id)
        if self.instrument is None:
            self.log.error(f"Could not find instrument for {self.config.instrument_id}")
            self.stop()
            return
        self.subscribe_bars(self.config.bar_type)
        self.request_bars(self.config.bar_type)
        self.log.info("MACrossoverStrategy started successfully.")

    def calculate_ma(self, window: int) -> Optional[float]:
        if len(self.prices_history) < window:
            return None
        return float(np.mean(self.prices_history[-window:]))

    def on_bar(self, bar) -> None:
        ts_event = bar.ts_event  # Timestamp in ns
        current_price = bar.close.as_double()
        self.prices_history.append(current_price)

        # Only proceed if enough data has been collected.
        if len(self.prices_history) < self.config.long_window:
            return

        # Calculate the moving averages.
        short_ma = self.calculate_ma(self.config.short_window)
        long_ma = self.calculate_ma(self.config.long_window)
        if short_ma is None or long_ma is None:
            return

        # State-based logic:
        # If not in a position and short MA > long MA, then BUY.
        if not self.in_position and short_ma > long_ma:
            venue = self.instrument.id.venue
            account = self.portfolio.account(venue)
            cash = account.balance_free(USD).as_double()
            investment = cash * 0.2  # invest 20% of available cash
            qty = investment / current_price
            qty = self.instrument.make_qty(qty)
            order = self.order_factory.market(
                instrument_id=self.instrument.id,
                order_side=OrderSide.BUY,
                quantity=qty,
                time_in_force=TimeInForce.GTC,
            )
            self.entry_order = order
            self.submit_order(order)
            self.log.info(f"Submitted BUY order for {qty} units at price {current_price}.")
            self.in_position = True

        # If in a position and short MA < long MA, then SELL.
        elif self.in_position and short_ma < long_ma:
            if self.position_id is None:
                return
            pos = self.cache.position(self.position_id)
            if pos is None:
                return
            sell_qty = pos.quantity  # Sell the entire position
            sell_qty = self.instrument.make_qty(sell_qty)
            order = self.order_factory.market(
                instrument_id=self.instrument.id,
                order_side=OrderSide.SELL,
                quantity=sell_qty,
                time_in_force=TimeInForce.GTC,
            )
            self.exit_order = order
            self.submit_order(order)
            self.log.info(f"Submitted SELL order for {sell_qty} units at price {current_price}.")
            self.in_position = False

    def on_order_filled(self, event) -> None:
        if event.order_side == OrderSide.BUY:
            if event.position_id is not None:
                self.position_id = PositionId(str(event.position_id))
        elif event.order_side == OrderSide.SELL:
            self.position_id = None

        if self.entry_order and event.client_order_id == self.entry_order.client_order_id:
            self.entry_order = None
        if self.exit_order and event.client_order_id == self.exit_order.client_order_id:
            self.exit_order = None

    def on_stop(self) -> None:
        self.cancel_all_orders(self.instrument.id)
        self.close_all_positions(self.instrument.id)
        self.log.info("MACrossoverStrategy stopped and cleanup complete.")
