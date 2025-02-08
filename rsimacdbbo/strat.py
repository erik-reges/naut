#!/usr/bin/env python3
"""
RSIMACDBBStrategy
-----------------
A NautilusTrader strategy that uses three key technical indicators:
  - RSI (Relative Strength Index)
  - Bollinger Bands (upper, middle, lower)
  - MACD (Moving Average Convergence Divergence)

Modified entry/exit logic:
  - Long entry: When RSI < 30 (oversold), current close is near the lower Bollinger Band,
    and MACD > MACD signal.
  - Long exit: Exit when RSI > 70 (overbought) OR if the current price has risen at least
    profit_target above entry OR fallen below stop_loss of entry.
"""

import numpy as np
import pandas as pd
from typing import List, Optional

# NautilusTrader imports
from nautilus_trader.config import StrategyConfig
from nautilus_trader.trading.strategy import Strategy
from nautilus_trader.model.data import Bar, BarType
from nautilus_trader.model.enums import OrderSide, TimeInForce
from nautilus_trader.model.currencies import USDT

# --- Helper function: Compute RSI without TA-Lib ---
def compute_rsi_np(prices: np.ndarray, period: int) -> np.ndarray:
    """
    Compute the Relative Strength Index (RSI) for an array of prices using the Wilder method.
    Returns an array of RSI values (the first 'period' values will be NaN).
    """
    delta = np.diff(prices)
    gains = np.where(delta > 0, delta, 0)
    losses = np.where(delta < 0, -delta, 0)
    avg_gain = np.full_like(prices, np.nan, dtype=float)
    avg_loss = np.full_like(prices, np.nan, dtype=float)
    avg_gain[period] = np.mean(gains[:period])
    avg_loss[period] = np.mean(losses[:period])
    for i in range(period + 1, len(prices)):
        avg_gain[i] = (avg_gain[i - 1] * (period - 1) + gains[i - 1]) / period
        avg_loss[i] = (avg_loss[i - 1] * (period - 1) + losses[i - 1]) / period
    rs = avg_gain / (avg_loss + 1e-10)
    rsi = 100 - (100 / (1 + rs))
    return rsi

# --- Extended Configuration Class ---
class RSIMACDBBStrategyConfig(StrategyConfig, frozen=True):
    instrument_id: str
    bar_type: BarType
    # Indicator periods
    rsi_period: int = 14
    bb_period: int = 20
    # Signal thresholds (modified for long-only trading)
    rsi_long_threshold: float = 30.0  # buy when RSI is below 30 (oversold)
    rsi_exit_threshold: float = 70.0  # exit when RSI exceeds 70 (overbought)
    # Investment settings
    investment_fraction: float = 0.1
    # Minimum number of bars required before signals are generated
    bars_required: int = 50
    # New exit parameters
    profit_target: float = 0.02    # exit if price rises 2% above entry
    stop_loss: float = 0.01        # exit if price falls 1% below entry

# --- Strategy Class ---
class RSIMACDBBStrategy(Strategy):
    def __init__(self, config: RSIMACDBBStrategyConfig) -> None:
        super().__init__(config)
        self.history: List[Bar] = []  # internal storage for Bar objects
        self.position_id: Optional[str] = None  # to store PositionId when a position is opened
        self.entry_price: Optional[float] = None  # record the entry price
        self.log.info("RSIMACDBBStrategy initialized.")

    def on_start(self) -> None:
        self.instrument = self.cache.instrument(self.config.instrument_id)
        if self.instrument is None:
            self.log.error(f"Instrument {self.config.instrument_id} not found.")
            self.stop()
            return
        self.subscribe_bars(self.config.bar_type)
        self.request_bars(self.config.bar_type)
        self.log.info("RSIMACDBBStrategy started: subscribed to bars and requested historical data.")

    def on_bar(self, bar: Bar) -> None:
        self.history.append(bar)
        # Limit history to the last 500 bars (or whatever is sufficient for your indicators)
        max_history = 500
        if len(self.history) > max_history:
            self.history = self.history[-max_history:]

        if len(self.history) < self.config.bars_required:
            return

        df = self._history_to_dataframe(self.history)
        df = self._compute_indicators(df)
        latest = df.iloc[-1]
        current_price = float(latest["close"].as_double())

        # If we are in a position, check for exit conditions:
        if self.position_id is not None and self.entry_price is not None:
            # Exit if profit target is met
            if current_price >= self.entry_price * (1 + self.config.profit_target):
                self.log.info(f"Profit target met: current price {current_price} >= {self.entry_price*(1+self.config.profit_target)}")
                self.close_position()
                return
            # Exit if stop-loss is triggered
            if current_price <= self.entry_price * (1 - self.config.stop_loss):
                self.log.info(f"Stop-loss triggered: current price {current_price} <= {self.entry_price*(1-self.config.stop_loss)}")
                self.close_position()
                return
            # Also exit if RSI exceeds the exit threshold (overbought)
            if latest["RSI"] > self.config.rsi_exit_threshold:
                self.log.info(f"RSI exit condition met: RSI {latest['RSI']} > {self.config.rsi_exit_threshold}")
                self.close_position()
                return

        # Only check for entry if no position is held.
        if self.portfolio.is_flat(self.instrument.id):
            # Modified long entry: buy if RSI is below 30 (oversold), current price is near (or below) the lower band,
            # and MACD > MACD signal.
            entry_condition = (
                (latest["RSI"] < self.config.rsi_long_threshold) and
                (current_price <= latest["lower_band"] * 1.02) and  # within 2% of the lower band
                (latest["macd"] > latest["macd_signal"])
            )
            if entry_condition:
                venue = self.instrument.id.venue
                account = self.portfolio.account(venue)
                cash = account.balance_free(USDT).as_double()
                investment = cash * self.config.investment_fraction
                qty = investment / current_price
                qty = self.instrument.make_qty(qty)
                order = self.order_factory.market(
                    instrument_id=self.instrument.id,
                    order_side=OrderSide.BUY,
                    quantity=qty,
                    time_in_force=TimeInForce.GTC,
                )
                self.submit_order(order)
                self.log.info(f"Submitted BUY order for {qty} units at price {current_price:.2f}")

    def _history_to_dataframe(self, history: List[Bar]) -> pd.DataFrame:
        data = {
            "timestamp": [bar.ts_event for bar in history],
            "open": [bar.open for bar in history],
            "high": [bar.high for bar in history],
            "low": [bar.low for bar in history],
            "close": [bar.close for bar in history],
            "volume": [bar.volume for bar in history],
        }
        df = pd.DataFrame(data)
        df.sort_values("timestamp", inplace=True)
        df.reset_index(drop=True, inplace=True)
        return df

    def _compute_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        prices = df["close"].values.astype(float)
        df["RSI"] = compute_rsi_np(prices, self.config.rsi_period)
        df["middle_band"] = df["close"].rolling(window=self.config.bb_period).mean()
        df["std"] = df["close"].rolling(window=self.config.bb_period).std()
        df["upper_band"] = df["middle_band"] + 2 * df["std"]
        df["lower_band"] = df["middle_band"] - 2 * df["std"]
        ema_fast = df["close"].ewm(span=12, adjust=False).mean()
        ema_slow = df["close"].ewm(span=26, adjust=False).mean()
        df["macd"] = ema_fast - ema_slow
        df["macd_signal"] = df["macd"].ewm(span=9, adjust=False).mean()
        return df

    def close_position(self) -> None:
        if self.position_id is None:
            return
        pos = self.cache.position(self.position_id)
        if pos:
            order = self.order_factory.market(
                instrument_id=self.instrument.id,
                order_side=OrderSide.SELL,
                quantity=self.instrument.make_qty(pos.quantity),
                time_in_force=TimeInForce.GTC,
            )
            self.submit_order(order)
            self.log.info(f"Submitted SELL order to close position of {pos.quantity} units.")
            self.position_id = None
            self.entry_price = None
        else:
            self.log.info("No open position found to close.")

    def on_order_filled(self, event) -> None:
        if event.order_side == OrderSide.BUY:
            self.position_id = event.position_id  # store PositionId
            if hasattr(event, 'last_px'):
                self.entry_price = float(event.last_px)
            else:
                self.entry_price = None
        elif event.order_side == OrderSide.SELL:
            self.position_id = None
            self.entry_price = None

    def on_stop(self) -> None:
        self.cancel_all_orders(self.instrument.id)
        self.close_all_positions(self.instrument.id)
        self.log.info("RSIMACDBBStrategy stopped and cleanup complete.")
