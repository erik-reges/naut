#!/usr/bin/env python3
"""
KNN-Based Strategy (Long Only Version)
----------------------------------------
This NautilusTrader strategy implements a kNN-based market direction predictor
using two features computed from one of several possible technical indicators.
Training data is collected over a specified window, and on each new bar the
current features are computed and compared with historical values. Based on a
volatility filter and a maximum bars-in-trade rule, a BUY signal will open a long
position, while a SELL signal will now only trigger a position exit.
"""

from __future__ import annotations
from math import sqrt, copysign
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
    delta = np.diff(prices)
    gains = np.where(delta > 0, delta, 0)
    losses = np.where(delta < 0, -delta, 0)
    avg_gain = np.full(len(prices), np.nan, dtype=float)
    avg_loss = np.full(len(prices), np.nan, dtype=float)
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
    start_date: str                   # ISO‐8601 string (e.g. "2000-01-01T00:00:00")
    stop_date: str                    # ISO‐8601 string (e.g. "2025-12-31T23:59:00")
    indicator: str = "All"            # Options: "RSI", "CCI", "ROC", "Volume", "All"
    short_window: int = 14
    long_window: int = 28
    base_k: int = 252                 # Base number of neighbours; actual k = floor(sqrt(base_k))
    volatility_filter: bool = False
    bars_threshold: int = 300         # Maximum bars to remain in a trade
    bars_required: int = 30           # Minimum number of bars in history to compute indicators


    # New parameters for additional filters:
    neighbour_pct: float = 0.5        # e.g. require >=50% of neighbors as 'up'
    use_trend_filter: bool = False
    sma_period: int = 50              # period for trend (SMA) filter
    use_volume_filter: bool = False
    volume_ma_period: int = 20

# --- Strategy Class (Long-Only) ---
class KNNBasedStrategy(Strategy):
    BUY = 1
    SELL = -1
    CLEAR = 0

    def __init__(self, config: KNNBasedStrategyConfig) -> None:
        super().__init__(config)
        self.history: List[Bar] = []
        self.feature1: List[float] = []  # First feature values
        self.feature2: List[float] = []  # Second feature values
        self.directions: List[int] = []  # Class labels (+1 for up, -1 for down, 0 for flat)
        self.k: int = int(sqrt(self.config.base_k))
        self.current_signal: int = KNNBasedStrategy.CLEAR
        self.bars_counter: int = 0
        self.position_id: Optional[str] = None
        self.position_side: Optional[OrderSide] = None
        self.log.info(f"KNNBasedStrategy initialized with k={self.k}")

    def on_start(self) -> None:
        self.instrument = self.cache.instrument(self.config.instrument_id)
        if self.instrument is None:
            self.log.error(f"Instrument {self.config.instrument_id} not found.")
            self.stop()
            return
        self.subscribe_bars(self.config.bar_type)
        self.request_bars(self.config.bar_type)
        self.log.info("KNNBasedStrategy started: subscribed to bars and requested historical data.")

    def on_bar(self, bar: Bar) -> None:
        self.history.append(bar)
        max_history = self.config.bars_required + 50
        if len(self.history) > max_history:
            self.history = self.history[-max_history:]
        if len(self.history) < self.config.bars_required:
            return

        # Convert history to DataFrame for indicator calculations.
        df = self._history_to_dataframe(self.history)
        idx = df.index[-1]

        # Compute technical indicators.
        prices = df["close"].to_numpy(dtype=float)
        rsi_long = compute_rsi_np(prices, self.config.long_window)
        rsi_short = compute_rsi_np(prices, self.config.short_window)
        cci_long = compute_cci(df, self.config.long_window)
        cci_short = compute_cci(df, self.config.short_window)
        roc_long = compute_roc(df, self.config.long_window)
        roc_short = compute_roc(df, self.config.short_window)
        vol_long = normalize_volume(df, self.config.long_window)
        vol_short = normalize_volume(df, self.config.short_window)

        def safe(val):
            return float(val) if pd.notna(val) else 50.0

        # Select features based on the chosen indicator.
        ind = self.config.indicator.upper()
        if ind == "RSI":
            f1, f2 = safe(rsi_long[idx]), safe(rsi_short[idx])
        elif ind == "CCI":
            f1, f2 = safe(cci_long[idx]), safe(cci_short[idx])
        elif ind == "ROC":
            f1, f2 = safe(roc_long[idx]), safe(roc_short[idx])
        elif ind == "VOLUME":
            f1, f2 = safe(vol_long[idx]), safe(vol_short[idx])
        else:  # "ALL": average values
            f1 = np.nanmean([safe(rsi_long[idx]), safe(cci_long[idx]), safe(roc_long[idx]), safe(vol_long[idx])])
            f2 = np.nanmean([safe(rsi_short[idx]), safe(cci_short[idx]), safe(roc_short[idx]), safe(vol_short[idx])])

        # Update training data if within the configured date window.
        bar_time = pd.to_datetime(bar.ts_event, utc=True)
        start_dt = pd.to_datetime(self.config.start_date, utc=True)
        stop_dt = pd.to_datetime(self.config.stop_date, utc=True)
        if start_dt <= bar_time <= stop_dt:
            if len(df) >= 2:
                prev_close = float(df["close"].iloc[-2])
                current_close = float(df["close"].iloc[-1])
                move = current_close - prev_close
                class_label = int(copysign(1, move)) if move != 0 else 0
            else:
                class_label = 0
            self.feature1.append(f1)
            self.feature2.append(f2)
            self.directions.append(class_label)

        # ---------------------------
        # kNN Prediction (only if enough training data)
        if len(self.feature1) <= self.k:
            self.log.debug(f"Not enough training data ({len(self.feature1)}) for kNN (requires > {self.k})")
            return

        features = np.column_stack((np.array(self.feature1), np.array(self.feature2)))
        current_feature = np.array([f1, f2])
        distances = np.sqrt(np.sum((features - current_feature) ** 2, axis=1))
        k_indices = np.argpartition(distances, self.k)[:self.k]
        k_neighbors = np.array(self.directions)[k_indices]
        prediction = np.sum(k_neighbors)

        # Volatility filter (if enabled)
        filter_pass = True
        if self.config.volatility_filter and len(df) >= 41:
            atr_short = compute_atr(df, 10).iloc[-1]
            atr_long = compute_atr(df, 40).iloc[-1]
            filter_pass = atr_short > atr_long
        elif self.config.volatility_filter:
            filter_pass = False

        # Compute current close price for later comparisons.
        current_close = float(df["close"].iloc[-1])

        # ---- New Signal Generation with Trend and Volume Filters ----
        # Only allow an entry if filter passes and a sufficient number of neighbors are 'up'.
        # (For long-only, we treat any non-entry as a CLEAR signal.)
        signal = 1 if (filter_pass and prediction >= (self.k * self.config.neighbour_pct)) else 0

        # Apply trend filter if enabled
        if signal == 1 and self.config.use_trend_filter:
            sma_val = df["close"].rolling(window=self.config.sma_period).mean().iloc[-1]
            if current_close < sma_val:
                self.log.debug(
                    f"Signal rejected by trend filter: price {current_close} below SMA{self.config.sma_period} {sma_val}"
                )
                signal = 0

        # Apply volume filter if enabled
        if signal == 1 and self.config.use_volume_filter:
            volume_avg = df["volume"].rolling(window=self.config.volume_ma_period).mean().iloc[-1]
            current_volume = float(df["volume"].iloc[-1])
            if current_volume < volume_avg:
                self.log.debug(
                    f"Signal rejected by volume filter: volume {current_volume} below {self.config.volume_ma_period}-period average {volume_avg}"
                )
                signal = 0

        # Translate signal into a strategy command.
        # For a long-only strategy, a valid signal (signal==1) means BUY.
        new_signal = KNNBasedStrategy.BUY if signal == 1 else KNNBasedStrategy.CLEAR

        # Trade duration management.
        if self.bars_counter >= self.config.bars_threshold:
            new_signal = KNNBasedStrategy.CLEAR
            self.bars_counter = 0

        # Signal Change – For a SELL signal in a cash account, only exit.
        if new_signal != self.current_signal:
            self.log.debug(f"Signal change from {self.current_signal} to {new_signal} (prediction: {prediction})")
            if self.position_id is not None:
                self.close_position()
            # Only open a new long position on a BUY signal.
            if new_signal == KNNBasedStrategy.BUY:
                self._enter_position(OrderSide.BUY, float(df["close"].iloc[-1]))
            # Do not enter a short position in a cash account.
            self.current_signal = new_signal
            self.bars_counter = 0
        else:
            if self.position_id is not None:
                self.bars_counter += 1
                if self.bars_counter >= self.config.bars_threshold:
                    self.log.info("Bars threshold reached, closing position.")
                    self.close_position()

    def _enter_position(self, side: OrderSide, current_price: float) -> None:
        # Only enter long positions.
        if self.position_id is not None or side != OrderSide.BUY:
            return

        venue = self.instrument.id.venue
        account = self.portfolio.account(venue)
        cash_balance = account.balance_free(USDT).as_double()

        df = self._history_to_dataframe(self.history)
        atr = compute_atr(df, 14).iloc[-1]
        risk_amount = cash_balance * 0.01  # 1% risk per trade
        stop_loss_distance = atr * 1.5
        if stop_loss_distance <= 0:
            self.log.warning("Invalid stop loss distance computed.")
            return
        max_qty_risk = risk_amount / stop_loss_distance
        max_trade_value = cash_balance * 0.05  # 5% of account per trade
        max_qty_value = max_trade_value / current_price
        max_qty = min(max_qty_risk, max_qty_value)
        self.log.debug(f"cash_balance: {cash_balance}, atr: {atr}, risk_amount: {risk_amount}, "
                       f"stop_loss_distance: {stop_loss_distance}, max_qty_risk: {max_qty_risk}, "
                       f"max_qty_value: {max_qty_value}, max_qty: {max_qty}")
        if max_qty < self.instrument.min_quantity or max_qty <= 0:
            self.log.warning(f"Calculated quantity too small: {max_qty}")
            return

        qty = self.instrument.make_qty(max_qty)
        if qty <= 0:
            self.log.warning(f"Final order quantity computed as {qty}; not submitting order.")
            return

        order = self.order_factory.market(
            instrument_id=self.instrument.id,
            order_side=side,
            quantity=qty,
            time_in_force=TimeInForce.GTC,
        )
        self.submit_order(order)
        self.log.info(f"Submitted {side.name} order: {qty} units at ~{current_price}")
        self.position_side = side
        self.current_signal = side.value
        self.bars_counter = 0

    def close_position(self, instrument_id=None, order_side=None, quantity=None, time_in_force=None) -> None:
        if self.position_id is None:
            return
        pos = self.cache.position(self.position_id)
        if pos is None or pos.quantity <= 0:
            self.position_id = None
            self.current_signal = KNNBasedStrategy.CLEAR
            self.bars_counter = 0
            return

        # For a long position, the exit order is SELL.
        exit_side = OrderSide.SELL
        qty = self.instrument.make_qty(pos.quantity)
        if qty <= 0:
            self.log.warning(f"Calculated close quantity is too small: {qty}; skipping close order.")
            self.position_id = None
            self.current_signal = KNNBasedStrategy.CLEAR
            self.bars_counter = 0
            return

        order = self.order_factory.market(
            instrument_id=self.instrument.id,
            order_side=exit_side,
            quantity=qty,
            time_in_force=TimeInForce.GTC,
        )
        self.submit_order(order)
        self.log.info(f"Submitted order to close position of {pos.quantity} units")
        self.position_id = None
        self.current_signal = KNNBasedStrategy.CLEAR
        self.bars_counter = 0

    def on_order_filled(self, event) -> None:
        if not hasattr(event, 'position_id'):
            return
        if self.position_id is None:
            self.position_id = event.position_id
            self.log.info(f"Entered new position {self.position_id}")
        elif str(event.position_id) == str(self.position_id):
            self.log.info(f"Closed position {self.position_id}")
            self.position_id = None
            self.current_signal = KNNBasedStrategy.CLEAR
            self.bars_counter = 0

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
        df.sort_values("timestamp", inplace=True)
        df.reset_index(drop=True, inplace=True)
        return df

    def on_stop(self) -> None:
        self.cancel_all_orders(self.instrument.id)
        self.close_all_positions(self.instrument.id)
        self.log.info("KNNBasedStrategy stopped and cleanup complete.")
