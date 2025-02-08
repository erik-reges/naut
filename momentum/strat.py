#!/usr/bin/env python3
"""
MomentumBreakoutStrategy
--------------------------
A NautilusTrader strategy based on a momentum breakout approach for 1‑minute ETHUSDT data.

Strategy logic:
- Use a lookback window (e.g. 20 bars) to compute the highest high.
- When the current close price exceeds the highest high from the lookback window,
  this signals a momentum breakout.
- Enter a long position if not already in one.
- Once in a position, update the trailing stop loss based on the highest price achieved.
- Exit the position if any of the following occur:
    1. The current price falls below the trailing stop level.
    2. The current price exceeds a predefined profit target (relative to entry).
    3. The current price falls below a fixed stop-loss threshold (relative to entry).

Risk management is implemented via a trailing stop and fixed profit/stop-loss levels.
"""

from decimal import Decimal
import numpy as np
import pandas as pd
from typing import List, Optional
from dataclasses import dataclass

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
class MomentumBreakoutStrategyConfig(StrategyConfig, frozen=True):
    instrument_id: str
    bar_type: BarType
    lookback_period: int = 20
    trailing_stop_percent: float = 0.01
    profit_target: float = 0.03
    stop_loss: float = 0.01
    investment_fraction: float = 0.1
    bars_required: int = 21
    rsi_period: int = 14
    bb_period: int = 20                   # Bollinger Bands period

# --- Strategy Class ---
class MomentumBreakoutStrategy(Strategy):
    def __init__(self, config: MomentumBreakoutStrategyConfig) -> None:
        super().__init__(config)
        self.history: List[Bar] = []  # maintain a sliding window of recent bars
        self.position_id: Optional[str] = None  # will store the PositionId when a position is open
        self.entry_price: Optional[float] = None  # record entry price (as a float)
        self.highest_since_entry: Optional[float] = None  # track highest price since entering
        self.log.info("MomentumBreakoutStrategy initialized.")

    def on_start(self) -> None:
        self.instrument = self.cache.instrument(self.config.instrument_id)
        if self.instrument is None:
            self.log.error(f"Instrument {self.config.instrument_id} not found.")
            self.stop()
            return
        self.subscribe_bars(self.config.bar_type)
        self.request_bars(self.config.bar_type)
        self.log.info("MomentumBreakoutStrategy started: subscribed to bars and requested historical data.")

    def on_bar(self, bar: Bar) -> None:
        # Append the new bar to the sliding window history.
        self.history.append(bar)
        # Limit history to speed up processing
        max_history = self.config.lookback_period + 50  # extra buffer
        if len(self.history) > max_history:
            self.history = self.history[-max_history:]
        if len(self.history) < self.config.bars_required:
            return

        # Convert the recent history to a DataFrame and compute all indicators.
        df = self._history_to_dataframe(self.history)
        df = self._compute_indicators(df)
        latest = df.iloc[-1]
        current_price = float(latest["close"].as_double())

        # -------------------------------
        # If already in a position, check exit conditions.
        # -------------------------------
        if self.position_id is not None and self.entry_price is not None:
            # Update the highest price reached since entry
            if self.highest_since_entry is None or current_price > self.highest_since_entry:
                self.highest_since_entry = current_price

            # --- ATR-based trailing stop implementation ---
            # Compute ATR over the last 14 bars.
            atr_period = 14
            if len(df) >= atr_period + 1:
                tr_values = []
                # Compute True Range for the last 'atr_period' bars
                for i in range(1, atr_period + 1):
                    current_high = float(df.iloc[-i]["high"])
                    current_low = float(df.iloc[-i]["low"])
                    prev_close = float(df.iloc[-i-1]["close"])
                    tr = max(
                        current_high - current_low,
                        abs(current_high - prev_close),
                        abs(current_low - prev_close)
                    )
                    tr_values.append(tr)
                atr = Decimal(str(np.mean(tr_values)))  # Convert to Decimal
            else:
                atr = Decimal('0.0')

            # Set trailing stop level using an ATR multiplier
            atr_multiplier = Decimal('1.5')
            if atr > 0:
                trailing_stop_level = Decimal(str(self.highest_since_entry)) - (atr_multiplier * atr)
                # # Log the ATR and trailing stop for debugging
                # self.log.info(f"ATR: {atr:.4f}, Highest: {self.highest_since_entry:.4f}, "
                #               f"Trailing Stop Level: {trailing_stop_level:.4f}")
                if Decimal(str(current_price)) <= trailing_stop_level:
                    self.close_position()
                    return
            # Profit target and fixed stop loss conditions remain.
            if current_price >= self.entry_price * (1 + self.config.profit_target):
                self.close_position()
                return
            if current_price <= self.entry_price * (1 - self.config.stop_loss):
                self.close_position()
                return
            return

        # -------------------------------
        # Entry Condition: Revised Momentum Breakout with Enhanced Confirmation
        # -------------------------------
        lookback = self.config.lookback_period
        if len(df) <= lookback:
            return

        # Compute the highest high from the previous lookback bars (excluding the current bar)
        previous_bars = df.iloc[-(lookback + 1):-1]
        highest_high = previous_bars["high"].max()

        # Optional: Uncomment to add a small breakout buffer (e.g., 0.5% above the highest high)
        # breakout_threshold = highest_high * 1.005
        # breakout = current_price > breakout_threshold

        # Primary breakout condition: current price must exceed the previous highest high.
        breakout = current_price > highest_high

        # Additional confirmations:
        macd_confirm = latest["macd"] > latest["macd_signal"]
        rsi_confirm = latest["RSI"] > 55  # Increased threshold for stronger bullish momentum

        # Instead of the upper band, require the current price to be above the middle band (simple moving average)
        bb_confirm = current_price >= latest["middle_band"]

        # Volume confirmation: require that the current bar's volume is at least 1.2 times the rolling average
        volume_avg = df["volume"].rolling(window=self.config.bb_period).mean().iloc[-1]
        volume_confirm = bar.volume >= (1.2 * volume_avg)

        # If all conditions are met, submit a BUY order.
        if breakout and macd_confirm and rsi_confirm and bb_confirm and volume_confirm:
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

    def close_position(self, instrument_id=None, order_side=None, quantity=None, time_in_force=None) -> None:
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
            self.highest_since_entry = None
        else:
            self.log.info("No open position found to close.")

    def on_order_filled(self, event) -> None:
        if event.order_side == OrderSide.BUY:
            from nautilus_trader.model.identifiers import PositionId
            self.position_id = PositionId(str(event.position_id))
            if hasattr(event, 'last_px'):
                self.entry_price = float(event.last_px)
            else:
                self.entry_price = None
            self.highest_since_entry = self.entry_price
        elif event.order_side == OrderSide.SELL:
            self.position_id = None
            self.entry_price = None
            self.highest_since_entry = None

    def on_stop(self) -> None:
        self.cancel_all_orders(self.instrument.id)
        self.close_all_positions(self.instrument.id)
        self.log.info("MomentumBreakoutStrategy stopped and cleanup complete.")
