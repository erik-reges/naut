#!/usr/bin/env python3
"""
Improved SimpleMACrossoverStrategy + Daily Trend Filter (Long-Only)
---------------------------------------------------------------------
1) M30-based moving average crossover: short vs. long.
2) Daily SMA filter: only go long if daily close > daily SMA.
3) ATR-based stop loss with dynamic trailing stop update based on momentum.
4) Trade duration limit: exit if position held for too many bars.
5) History trimming to conserve memory.
"""

from __future__ import annotations
from typing import List, Optional
import pandas as pd
import numpy as np

from nautilus_trader.config import StrategyConfig
from nautilus_trader.model.data import Bar, BarType
from nautilus_trader.model.enums import OrderSide, TimeInForce
from nautilus_trader.model.identifiers import InstrumentId
from nautilus_trader.model.currencies import USDT
from nautilus_trader.trading.strategy import Strategy


# --- Utility Functions ---
def compute_ma(series: pd.Series, window: int) -> pd.Series:
    """Compute simple moving average (SMA) over the specified window."""
    return series.rolling(window=window).mean()


def compute_atr(df: pd.DataFrame, period: int) -> pd.Series:
    """
    Compute the Average True Range (ATR) using high, low, and close prices.
    """
    high = df["high"]
    low = df["low"]
    close = df["close"]
    tr = pd.concat([
        high - low,
        (high - close.shift(1)).abs(),
        (low - close.shift(1)).abs()
    ], axis=1).max(axis=1)
    return tr.rolling(window=period).mean()


# --- Configuration Class ---
class SimpleMACrossoverStrategyConfig(StrategyConfig, frozen=True):
    instrument_id: str
    bar_type: BarType
    short_window: int = 25
    long_window: int = 100
    atr_period: int = 14
    atr_stop_multiple: float = 1.5
    max_risk_pct: float = 0.1

    # Higher timeframe daily filter
    use_daily_filter: bool = True
    daily_sma_period: int = 20



    # --- New: Maximum bars allowed in a trade ---
    max_trade_bars: int = 100  # exit trade if held for more than 100 bars

    # --- New: Trailing stop dynamic multiplier boundaries ---
    trailing_multiplier_min: float = 1.5
    trailing_multiplier_max: float = 3.0


# --- Strategy Class (Long-Only) ---
class SimpleMACrossoverStrategy(Strategy):
    def __init__(self, config: SimpleMACrossoverStrategyConfig) -> None:
        super().__init__(config)
        self.history: List[Bar] = []
        self.position_id: Optional[str] = None
        self.entry_price: Optional[float] = None
        self.stop_price: Optional[float] = None
        self.bars_counter: int = 0  # Counts how many bars the trade has been active

    def on_start(self) -> None:
        self.instrument = self.cache.instrument(self.config.instrument_id)
        if not self.instrument:
            self.log.error(f"Instrument {self.config.instrument_id} not found.")
            self.stop()
            return

        self.subscribe_bars(self.config.bar_type)
        self.request_bars(self.config.bar_type)
        self.log.info(f"SimpleMACrossoverStrategy started for {self.instrument.symbol}")

    def on_bar(self, bar: Bar) -> None:
        self.history.append(bar)
        # Trim history to avoid unbounded memory usage
        max_bars_needed = max(self.config.long_window, self.config.atr_period) + 2
        if len(self.history) > max_bars_needed * 2:
            self.history = self.history[-max_bars_needed * 2:]

        if len(self.history) < max_bars_needed:
            return

        df = self._history_to_dataframe(self.history)

        # Daily filter: only proceed if the daily trend is bullish.
        is_bullish = True
        if self.config.use_daily_filter:
            is_bullish = self._check_daily_filter(df)
        if not is_bullish:
            self.log.debug("Daily filter not bullish. Closing any open position.")
            if self.position_id is not None:
                self.close_position()
            return

        # Compute short/long moving averages and ATR on 30-min bars.
        short_ma = compute_ma(df["close"], window=self.config.short_window)
        long_ma = compute_ma(df["close"], window=self.config.long_window)
        current_close = df["close"].iloc[-1]
        short_val = short_ma.iloc[-1]
        long_val = long_ma.iloc[-1]
        atr_val = compute_atr(df, self.config.atr_period).iloc[-1]

        should_be_long = short_val > long_val

        if self.position_id is not None:
            self.bars_counter += 1

            # Check if stop loss is hit.
            if self.stop_price is not None and current_close < self.stop_price:
                self.log.info(
                    f"Stop hit: current {current_close:.2f} < stop {self.stop_price:.2f}. Exiting position."
                )
                self.close_position()
                return

            # Exit if the moving average crossover reverses.
            if not should_be_long:
                self.log.info(
                    f"Exit signal: short MA ({short_val:.2f}) < long MA ({long_val:.2f}). Closing position."
                )
                self.close_position()
                return

            # Exit if trade duration has exceeded maximum allowed bars.
            if self.bars_counter >= self.config.max_trade_bars:
                self.log.info(
                    f"Trade duration exceeded: {self.bars_counter} bars. Closing position."
                )
                self.close_position()
                return

            # --- Dynamic Trailing Stop Update ---
            if self.entry_price:
                momentum = current_close / self.entry_price - 1
                dynamic_multiplier = self.config.atr_stop_multiple * (1 + momentum)
                # Clip the multiplier between configured bounds.
                dynamic_multiplier = np.clip(
                    dynamic_multiplier,
                    self.config.trailing_multiplier_min,
                    self.config.trailing_multiplier_max,
                )
                new_stop = current_close - (dynamic_multiplier * atr_val)
                if new_stop > (self.stop_price or 0.0):
                    self.stop_price = new_stop
                    self.log.debug(
                        f"Updated trailing stop to {self.stop_price:.2f} using dynamic multiplier {dynamic_multiplier:.2f}."
                    )
        else:
            # Consider entering a new long position.
            if should_be_long:
                self._enter_position(current_close, atr_val)

    def _check_daily_filter(self, df_30min: pd.DataFrame) -> bool:
        """
        Resample 30-min bars to daily data and apply a SMA filter.
        Returns True if the last daily close exceeds its SMA.
        """
        df_daily = df_30min.resample("1D", label="right", closed="right").agg({
            "open": "first",
            "high": "max",
            "low": "min",
            "close": "last",
            "volume": "sum",
        })
        df_daily.dropna(subset=["close"], inplace=True)
        if len(df_daily) < self.config.daily_sma_period:
            self.log.debug("Not enough daily data for SMA filter; defaulting to bullish.")
            return True
        df_daily["daily_sma"] = df_daily["close"].rolling(self.config.daily_sma_period).mean()
        last_close = df_daily["close"].iloc[-1]
        last_sma = df_daily["daily_sma"].iloc[-1]
        self.log.debug(f"Daily filter: last close = {last_close:.2f}, SMA = {last_sma:.2f}.")
        return last_close > last_sma

    def _enter_position(self, current_close: float, atr_val: float) -> None:
        if self.position_id is not None:
            return  # already in a position

        venue = self.instrument.id.venue
        account = self.portfolio.account(venue)
        cash_balance = account.balance_free(USDT).as_double()
        risk_amount = cash_balance * self.config.max_risk_pct
        stop_distance = atr_val * self.config.atr_stop_multiple

        if stop_distance <= 0:
            self.log.warning("Stop distance <= 0, skipping entry.")
            return

        # Calculate base quantity
        qty = (risk_amount / stop_distance) / current_close

        # Check if computed qty is below the instrument's minimum quantity.
        if qty < self.instrument.min_quantity:
            self.log.warning(
                f"Calculated qty {qty:.6f} is below minimum allowed {self.instrument.min_quantity}. Order not placed.")
            return

        # Use the quantity multiplier to adjust the final order size.
        order_qty = self.instrument.make_qty(qty )

        # Final safeguard: ensure the final order quantity is positive.
        if order_qty <= 0:
            self.log.warning(f"Final order quantity {order_qty} is not positive. Order aborted.")
            return

        order = self.order_factory.market(
            instrument_id=self.instrument.id,
            order_side=OrderSide.BUY,
            quantity=order_qty,
            time_in_force=TimeInForce.GTC,
        )
        self.submit_order(order)
        self.log.info(
            f"Entering LONG: qty={order_qty}, entry={current_close:.2f}, stop distance={stop_distance:.2f}."
        )
        self.entry_price = current_close
        self.stop_price = current_close - stop_distance

    def close_position(self) -> None:
        if self.position_id:
            pos = self.cache.position(self.position_id)
            if pos is None or pos.quantity <= 0:
                return
            order = self.order_factory.market(
                instrument_id=self.instrument.id,
                order_side=OrderSide.SELL,
                quantity=self.instrument.make_qty(pos.quantity),
                time_in_force=TimeInForce.GTC,
            )
            self.submit_order(order)
            self.log.info(f"Closing position {self.position_id} of {pos.quantity} units.")
            # Reset trade-related state.
            self.position_id = None
            self.entry_price = None
            self.stop_price = None
            self.bars_counter = 0

    def on_order_filled(self, event) -> None:
        if not hasattr(event, "position_id"):
            return
        if self.position_id is None:
            self.position_id = event.position_id
            self.log.info(f"Opened position {self.position_id}.")
        elif str(event.position_id) == str(self.position_id):
            self.log.info(f"Closed position {self.position_id}.")
            self.position_id = None
            self.entry_price = None
            self.stop_price = None
            self.bars_counter = 0

    def on_stop(self) -> None:
        self.cancel_all_orders(self.instrument.id)
        if self.position_id is not None:
            self.close_position()
        self.log.info("SimpleMACrossoverStrategy stopped.")

    def _history_to_dataframe(self, history: List[Bar]) -> pd.DataFrame:
        data = {
            "timestamp": [bar.ts_event for bar in history],
            "open": [float(bar.open) for bar in history],
            "high": [float(bar.high) for bar in history],
            "low": [float(bar.low) for bar in history],
            "close": [float(bar.close) for bar in history],
            "volume": [float(bar.volume) for bar in history],
        }
        df = pd.DataFrame(data)
        df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ns")
        df.set_index("timestamp", inplace=True)
        df.sort_index(inplace=True)
        return df
