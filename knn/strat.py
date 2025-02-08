#!/usr/bin/env python3
"""
KNNBasedStrategy (Long-Only Version)
-------------------------------------
A NautilusTrader strategy implementing a kNN‐based market direction predictor.
In this version the strategy is long only: it will only enter long positions and exit
when the aggregated prediction turns non‐positive.
"""

from __future__ import annotations
from math import sqrt
from typing import List, Optional

import numpy as np
import pandas as pd

from nautilus_trader.config import StrategyConfig
from nautilus_trader.trading.strategy import Strategy
from nautilus_trader.model.data import Bar, BarType
from nautilus_trader.model.enums import OrderSide, TimeInForce
from nautilus_trader.model.currencies import USDT

# --- Helper Functions for Indicators ---
def compute_rsi_np(prices: np.ndarray, period: int) -> np.ndarray:
    # Ensure there is at least period+1 data points
    if len(prices) <= period:
        return np.full(len(prices), np.nan)

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

def compute_cci(df: pd.DataFrame, period: int) -> pd.Series:
    tp = (df["high"] + df["low"] + df["close"]) / 3
    sma = tp.rolling(window=period).mean()
    mad = tp.rolling(window=period).apply(lambda x: np.mean(np.abs(x - np.mean(x))), raw=True)
    cci = (tp - sma) / (0.015 * mad + 1e-10)
    return cci

def compute_roc(df: pd.DataFrame, period: int) -> pd.Series:
    roc = df["close"].pct_change(periods=period) * 100
    return roc

def normalize_volume(df: pd.DataFrame, period: int) -> pd.Series:
    roll_min = df["volume"].rolling(window=period).min()
    roll_max = df["volume"].rolling(window=period).max()
    norm = 100 * (df["volume"] - roll_min) / (roll_max - roll_min + 1e-10)
    return norm

def compute_atr(df: pd.DataFrame, period: int) -> pd.Series:
    high = df["high"]
    low = df["low"]
    close = df["close"]
    tr = pd.concat([
        high - low,
        (high - close.shift()).abs(),
        (low - close.shift()).abs()
    ], axis=1).max(axis=1)
    atr = tr.rolling(window=period).mean()
    return atr

# --- Configuration Class ---
class KNNBasedStrategyConfig(StrategyConfig, frozen=True):
    instrument_id: str
    bar_type: BarType
    start_date: str                   # ISO‐8601 string
    stop_date: str                    # ISO‐8601 string
    indicator: str = "All"            # Options: "RSI", "CCI", "ROC", "Volume", "All"
    short_window: int = 14
    long_window: int = 28
    base_k: int = 252
    volatility_filter: bool = False
    bars_threshold: int = 300         # maximum bars to remain in a trade
    bars_required: int = 30
    max_risk_pct: float = 0.01        # Maximum risk per trade (1% of account)
    margin_factor: float = 0.2        # minimum bars needed to start generating signals
    trailing_multiplier: float = 1.5  # Use the same multiplier for both entry and update
    trailing_atr_period: int = 14     # ATR period to be used for the trailing stop
    trailing_min_profit_multiple: float = 1.5  # Mi
    use_trend_filter: bool = False
    use_volume_filter: bool = False
    sma_period: int = 10
    volume_ma_period: int = 20
    neighbour_pct: float = 0.5


# --- Strategy Class (Long-Only) ---
class KNNBasedStrategy(Strategy):
    def __init__(self, config: KNNBasedStrategyConfig) -> None:
        super().__init__(config)

        self.history: List[Bar] = []
        self.feature1: List[float] = []
        self.feature2: List[float] = []
        self.directions: List[int] = []

        self.bars_counter: int = 0
        self.position_id: Optional[str] = None
        self.entry_price: Optional[float] = None
        self.trailing_stop: Optional[float] = None   # NEW: stores the current trailing stop

        self.k: int = int(sqrt(self.config.base_k))
        self.log.info("KNNBasedStrategy (Long Only) initialized.")

    def on_start(self) -> None:
        self.instrument = self.cache.instrument(self.config.instrument_id)
        if self.instrument is None:
            self.log.error(f"Instrument {self.config.instrument_id} not found.")
            self.stop()
            return
        self.subscribe_bars(self.config.bar_type)
        self.request_bars(self.config.bar_type)
        self.log.info("KNNBasedStrategy (Long Only) started: subscribed to bars and requested historical data.")

    def on_bar(self, bar: Bar) -> None:
        self.history.append(bar)
        max_history = self.config.bars_required + 50
        if len(self.history) > max_history:
            self.history = self.history[-max_history:]
        if len(self.history) < self.config.bars_required:
            return

        df = self._history_to_dataframe(self.history)

        # --- Compute Technical Indicators ---
        prices = df["close"].to_numpy(dtype=float)
        rsi_long = compute_rsi_np(prices, self.config.long_window)
        rsi_short = compute_rsi_np(prices, self.config.short_window)
        cci_long = compute_cci(df, self.config.long_window)
        cci_short = compute_cci(df, self.config.short_window)
        roc_long = compute_roc(df, self.config.long_window)
        roc_short = compute_roc(df, self.config.short_window)
        vol_long = normalize_volume(df, self.config.long_window)
        vol_short = normalize_volume(df, self.config.short_window)

        idx = df.index[-1]
        def safe(val):
            return float(val) if pd.notna(val) else 50.0

        def z_score(series, window):
            series = pd.Series(series)
            mean_series = series.rolling(window).mean()
            std_series = series.rolling(window).std()
            # Replace any zero standard deviations with NaN so that the division yields NaN,
            # then fill those NaNs with 0 (neutral value)
            std_series = std_series.replace(0, np.nan)
            z = (series - mean_series) / std_series
            # For bars with insufficient data or zero variance, return 0 (neutral deviation)
            return z.fillna(0)

        if self.config.indicator.upper() == "RSI":
            f1 = safe(rsi_long[idx])
            f2 = safe(rsi_short[idx])
        elif self.config.indicator.upper() == "CCI":
            f1 = safe(cci_long[idx])
            f2 = safe(cci_short[idx])
        elif self.config.indicator.upper() == "ROC":
            f1 = safe(roc_long[idx])
            f2 = safe(roc_short[idx])
        elif self.config.indicator.upper() == "VOLUME":
            f1 = safe(vol_long[idx])
            f2 = safe(vol_short[idx])
        else:
            rsi_long_z = z_score(rsi_long, self.config.long_window)
            cci_long_z = z_score(cci_long, self.config.long_window)
            roc_long_z = z_score(roc_long, self.config.long_window)
            vol_long_z = z_score(vol_long, self.config.long_window)

            f1 = np.nanmean([rsi_long_z.iloc[-1], cci_long_z.iloc[-1], roc_long_z.iloc[-1], vol_long_z.iloc[-1]])

            rsi_short_z = z_score(rsi_short, self.config.short_window)
            cci_short_z = z_score(cci_short, self.config.short_window)
            roc_short_z = z_score(roc_short, self.config.short_window)
            vol_short_z = z_score(vol_short, self.config.short_window)

            f2 = np.nanmean([rsi_short_z.iloc[-1], cci_short_z.iloc[-1], roc_short_z.iloc[-1], vol_short_z.iloc[-1]])

        if len(df) < 2:
            return
        prev_close = float(df["close"].iloc[-2])
        current_close = float(df["close"].iloc[-1])
        # For long-only, we treat positive moves as our signal to be in the market.
        class_label = 1 if current_close > prev_close else 0

        max_features = self.config.base_k * 2
        if len(self.feature1) >= max_features:
            self.feature1 = self.feature1[-max_features:]
            self.feature2 = self.feature2[-max_features:]
            self.directions = self.directions[-max_features:]
        self.feature1.append(float(f1))
        self.feature2.append(float(f2))
        self.directions.append(class_label)

        # --- kNN Prediction ---
        features = np.column_stack((np.array(self.feature1), np.array(self.feature2)))
        current_feature = np.array([f1, f2])
        if len(features) <= self.k:
            self.log.debug(f"Not enough data points ({len(features)}) for kNN calculation (k={self.k})")
            return
        distances = np.sqrt(np.sum((features - current_feature) ** 2, axis=1))
        k_indices = np.argpartition(distances, self.k)[:self.k]
        k_neighbors = np.array(self.directions)[k_indices]
        prediction = np.sum(k_neighbors)

        # --- Volatility Filter (Optional) ---
        filter_pass = True
        if self.config.volatility_filter:
            if len(df) >= 41:
                atr_short = compute_atr(df, 10).iloc[-1]
                atr_long = compute_atr(df, 40).iloc[-1]
                filter_pass = atr_short > atr_long
            else:
                filter_pass = False

        # In long-only, only a positive prediction (and filter passing) leads to an entry.
        # Adjust signal condition to require > 50% of neighbors as 'up'

        signal = 1 if (filter_pass and prediction >= (self.k * self.config.neighbour_pct)) else 0

        # Apply trend filter if enabled
        if signal == 1 and self.config.use_trend_filter:
            sma_50 = df["close"].rolling(window=self.config.sma_period).mean().iloc[-1]
            if current_close < sma_50:
                self.log.debug(
                    f"Signal rejected by trend filter: price {current_close} below SMA{self.config.sma_period} {sma_50}")
                signal = 0

        # Apply volume filter if enabled
        if signal == 1 and self.config.use_volume_filter:
            volume_avg = df["volume"].rolling(window=self.config.volume_ma_period).mean().iloc[-1]
            current_volume = float(df["volume"].iloc[-1])
            if current_volume < volume_avg:
                self.log.debug(
                    f"Signal rejected by volume filter: volume {current_volume} below {self.config.volume_ma_period}-period average {volume_avg}")
                signal = 0


        # --- Trade Management with Trailing Stop ---
        if self.position_id is not None:
            self.bars_counter += 1
            current_close = float(df["close"].iloc[-1])

            # Calculate ATR and potential new stop level
            trailing_atr = compute_atr(df, self.config.trailing_atr_period).iloc[-1]
            momentum = current_close / self.entry_price - 1
            dynamic_multiplier = np.clip(self.config.trailing_multiplier * (1 + momentum), 1.5, 3.0)
            new_stop = current_close - (dynamic_multiplier * trailing_atr)


            # Check if we've met the minimum profit threshold
            if self.entry_price is not None:
                profit_distance = current_close - self.entry_price
                min_profit_required = trailing_atr * self.config.trailing_min_profit_multiple

                if profit_distance >= min_profit_required:
                    # Update trailing stop if new stop is higher
                    if self.trailing_stop is None or new_stop > self.trailing_stop:
                        self.trailing_stop = new_stop
        #                self.log.info(f"Updated trailing stop to {self.trailing_stop:.2f} "
        #                            f"(profit: {profit_distance:.2f})")

            # Check for stop hit
            if self.trailing_stop is not None and current_close < self.trailing_stop:
         #       self.log.info(f"Trailing stop hit at {current_close:.2f} < {self.trailing_stop:.2f}")
                self.close_position()
                return

            # Check bars threshold
            if self.bars_counter >= self.config.bars_threshold:
          #      self.log.info("Maximum bars threshold reached")
                self.close_position()
                return
        else:
            if signal == 1:
                self._enter_position(OrderSide.BUY, current_close)

    def _enter_position(self, side: OrderSide, current_price: float) -> None:
        if self.position_id is not None:
            return

        venue = self.instrument.id.venue
        account = self.portfolio.account(venue)
        cash_balance = account.balance_free(USDT).as_double()

        # Calculate ATR for position sizing and initial stop
        df = self._history_to_dataframe(self.history)
        trailing_atr = compute_atr(df, self.config.trailing_atr_period).iloc[-1]

        # Risk-based position sizing
        risk_amount = cash_balance * self.config.max_risk_pct
        stop_loss_distance = trailing_atr * self.config.trailing_multiplier
        max_qty_risk = risk_amount / stop_loss_distance

        # Additional position size constraints
        max_trade_value = cash_balance * 0.05  # Max 5% of account per trade
        max_qty_value = max_trade_value / current_price
        max_qty = min(max_qty_risk, max_qty_value)

        # Validate position size
        if max_qty < self.instrument.min_quantity:
            self.log.warning(f"Calculated quantity too small: {max_qty}")
            return

        qty = self.instrument.make_qty(max_qty)

        # Submit market order
        order = self.order_factory.market(
            instrument_id=self.instrument.id,
            order_side=side,
            quantity=qty,
            time_in_force=TimeInForce.GTC,
        )

        self.submit_order(order)
     #   self.log.info(f"Submitted {side.name} order: {qty} units at ~{current_price}")

        # Initialize trailing stop
        self.trailing_stop = current_price - stop_loss_distance
        self.entry_price = current_price
        self.bars_counter = 0

        # NEW: Initialize trailing stop using the current price minus an offset.
        # For example, use the latest ATR value as the trailing offset.
        df = self._history_to_dataframe(self.history)
        # Calculate the stop loss distance using the ATR for the configured period.
        stop_loss_distance = compute_atr(df, self.config.trailing_atr_period).iloc[-1]
        trailing_multiplier = self.config.trailing_multiplier  # e.g., 1.5
        self.trailing_stop = current_price - trailing_multiplier * stop_loss_distance
      #  self.log.info(f"Initial trailing stop set at {self.trailing_stop}")


    def  _history_to_dataframe(self, history: List[Bar]) -> pd.DataFrame:
        data = {
            "timestamp": [bar.ts_event for bar in history],
            "open": [float(bar.open) for bar in history],
            "high": [float(bar.high) for bar in history],
            "low": [float(bar.low) for bar in history],
            "close": [float(bar.close) for bar in history],
            "volume": [float(bar.volume) for bar in history],
        }
        df = pd.DataFrame(data)
        df.sort_values("timestamp", inplace=True)
        df.reset_index(drop=True, inplace=True)
        return df

    def close_position(self, instrument_id=None, order_side=None, quantity=None, time_in_force=None) -> None:
        if self.position_id is None:
            return
        pos = self.cache.position(self.position_id)
        if pos is None:
            self.position_id = None
            self.entry_price = None
            self.bars_counter = 0
            return
        # For a long position, we close by selling.
        order = self.order_factory.market(
            instrument_id=self.instrument.id,
            order_side=OrderSide.SELL,
            quantity=self.instrument.make_qty(pos.quantity),
            time_in_force=TimeInForce.GTC,
        )
        self.submit_order(order)
  #      self.log.info(f"Submitted order to close long position of {pos.quantity} units")

    def on_order_filled(self, event) -> None:
        if not hasattr(event, 'position_id'):
            return
        if self.position_id is None:
            self.position_id = event.position_id
            self.entry_price = float(event.last_px) if hasattr(event, 'last_px') else None
            self.bars_counter = 0
       #     self.log.info(f"Entered new position {self.position_id}")
        elif str(event.position_id) == str(self.position_id):
      #      self.log.info(f"Closed position {self.position_id}")
            self.position_id = None
            self.entry_price = None
            self.bars_counter = 0

    def on_stop(self) -> None:
        self.cancel_all_orders(self.instrument.id)
        self.close_all_positions(self.instrument.id)
    #    self.log.info("KNNBasedStrategy (Long Only) stopped and cleanup complete.")
