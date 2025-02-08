# strategy.py
from typing import Optional
from decimal import Decimal

from nautilus_trader.config import StrategyConfig
from nautilus_trader.trading.strategy import Strategy
from nautilus_trader.model.data import Bar, BarType
from nautilus_trader.model.enums import OrderSide, PriceType, TimeInForce
from nautilus_trader.model.currencies import USD
from nautilus_trader.indicators.average.ma_factory import MovingAverageFactory, MovingAverageType
from nautilus_trader.indicators.bollinger_bands import BollingerBands
from nautilus_trader.indicators.rsi import RelativeStrengthIndex
from nautilus_trader.indicators.atr import AverageTrueRange

class BBRSIStrategyConfig(StrategyConfig, frozen=True):
    instrument_id: str
    bar_type: BarType
    sma_period: int = 50
    rsi_period: int = 14
    rsi_threshold_low: float = 30
    rsi_threshold_high: float = 70
    bb_period: int = 20
    k: float = 2.0
    bb_width_threshold: float = 0.0015

    atr_period: int = 9
    sl_atr_multiple: float = 3.0
    tp_atr_multiple: float = 2.0
    position_size: float = 0.1

class BBRSIStrategy(Strategy):
    def __init__(self, config: BBRSIStrategyConfig):
        super().__init__(config)

        # Initialize state
        self.position_id: Optional[str] = None
        self.position_side: Optional[OrderSide] = None  # Track the side of our open position
        self.entry_price: Optional[float] = None
        self.trailing_stop: Optional[float] = None
        self.bars_counter: int = 0

        # Initialize indicators for SPY bars.
        self.sma = MovingAverageFactory.create(
            period=self.config.sma_period,
            ma_type=MovingAverageType.SIMPLE,
            price_type=PriceType.LAST,
        )

        self.bb = BollingerBands(
            period=self.config.bb_period,
            k=self.config.k,
            ma_type=MovingAverageType.SIMPLE,
        )

        self.rsi = RelativeStrengthIndex(
            period=self.config.rsi_period,
            ma_type=MovingAverageType.EXPONENTIAL,
        )

        self.atr = AverageTrueRange(
            period=self.config.atr_period,
            ma_type=MovingAverageType.SIMPLE,
        )

    def on_start(self):
        self.log.info("Strategy starting...")

        # Get our instrument
        self.instrument = self.cache.instrument(self.config.instrument_id)
        if self.instrument is None:
            self.log.error(f"Instrument {self.config.instrument_id} not found.")
            self.stop()
            return

        # Register our indicators
        self.register_indicator_for_bars(self.config.bar_type, self.bb)
        self.register_indicator_for_bars(self.config.bar_type, self.rsi)
        self.register_indicator_for_bars(self.config.bar_type, self.atr)
        self.register_indicator_for_bars(self.config.bar_type, self.sma)

        # Subscribe to bars
        self.subscribe_bars(self.config.bar_type)

        # Debug logs
        self.log.info(f"""
        Strategy Initialized:
        Instrument: {self.instrument}
        Bar Type: {self.config.bar_type}
        Initial Indicator Status:
        BB initialized: {self.bb.initialized}
        RSI initialized: {self.rsi.initialized}
        ATR initialized: {self.atr.initialized}
        Config Parameters:
        RSI Thresholds: {self.config.rsi_threshold_low}/{self.config.rsi_threshold_high}
        BB Period: {self.config.bb_period}
        BB Width Threshold: {self.config.bb_width_threshold}
        Position Size: {self.config.position_size}
        """)

    def on_bar(self, bar: Bar) -> None:
        if not all([
            self.bb.initialized,
            self.rsi.initialized,
            self.atr.initialized,
        ]):
            return

        current_price = float(bar.close)

        # If in position, check trailing stop and exit conditions
        if self.position_id is not None:
            pos = self.cache.position(self.position_id)
            if pos is not None:
                # Long position logic
                if self.position_side == OrderSide.BUY:
                    tp = self.entry_price + (self.config.tp_atr_multiple * self.atr.value)
                    # Initialize trailing stop on first update if not already set
                    if self.trailing_stop is None:
                        self.trailing_stop = self.entry_price - (self.config.sl_atr_multiple * self.atr.value)
                    else:
                        # Calculate new potential trailing stop level
                        new_stop = current_price - (self.config.sl_atr_multiple * self.atr.value)
                        # Only update if it moves upward (for long positions)
                        if new_stop > self.trailing_stop:
                            self.trailing_stop = new_stop
                    # Check exit conditions for a long position
                    if current_price <= self.trailing_stop or current_price >= tp or self.rsi.value > 0.88:
                        self.close_position()
                        return

                # Short position logic
                elif self.position_side == OrderSide.SELL:
                    tp = self.entry_price - (self.config.tp_atr_multiple * self.atr.value)
                    if self.trailing_stop is None:
                        self.trailing_stop = self.entry_price + (self.config.sl_atr_multiple * self.atr.value)
                    else:
                        new_stop = current_price + (self.config.sl_atr_multiple * self.atr.value)
                        # For shorts, update if the stop moves downward
                        if new_stop < self.trailing_stop:
                            self.trailing_stop = new_stop
                    if current_price >= self.trailing_stop or current_price <= tp or self.rsi.value < 0.12:
                        self.close_position()
                        return

            return

        # Check for buy signal to enter a long position
        if (self.position_id is None and
            bar.close < self.bb.lower and
            self.rsi.value < self.config.rsi_threshold_low and
            (self.bb.upper - self.bb.lower) / self.bb.middle > self.config.bb_width_threshold):
            self._enter_long(current_price)

    def _enter_long(self, current_price: float) -> None:
        # Calculate proper quantity based on position size
        account = self.portfolio.account(self.instrument.id.venue)
        equity = account.balance_total(USD).as_double()
        position_value = equity * self.config.position_size
        quantity = int(position_value / current_price)

        if quantity < 1:
            self.log.warning(f"Position size too small: {quantity} shares")
            return

        order = self.order_factory.market(
            instrument_id=self.config.instrument_id,
            order_side=OrderSide.BUY,
            quantity=self.instrument.make_qty(quantity),
            time_in_force=TimeInForce.GTC,
        )

        self.submit_order(order)
        self.log.info(f"Entering long position at {current_price:.2f}")

    def _enter_short(self, current_price: float) -> None:
        # Calculate proper quantity based on position size
        account = self.portfolio.account(self.instrument.id.venue)
        equity = account.balance_total(USD).as_double()
        position_value = equity * self.config.position_size
        quantity = int(position_value / current_price)

        if quantity < 1:
            self.log.warning(f"Position size too small: {quantity} shares")
            return

        order = self.order_factory.market(
            instrument_id=self.config.instrument_id,
            order_side=OrderSide.SELL,
            quantity=self.instrument.make_qty(quantity),
            time_in_force=TimeInForce.GTC,
        )

        self.submit_order(order)
        self.log.info(f"Entering short position at {current_price:.2f}")

    def close_position(self, instrument_id=None, order_side=None, quantity=None, time_in_force=None) -> None:
        if self.position_id is None:
            return
        pos = self.cache.position(self.position_id)
        if pos is None or pos.quantity <= 0:
            return

        # Determine closing order side based on stored position side.
        if self.position_side == OrderSide.BUY:
            closing_side = OrderSide.SELL
        elif self.position_side == OrderSide.SELL:
            closing_side = OrderSide.BUY
        else:
            self.log.error("Unknown position side. Cannot close position properly.")
            return

        close_order = self.order_factory.market(
            instrument_id=self.config.instrument_id,
            order_side=closing_side,
            quantity=pos.quantity,
            time_in_force=TimeInForce.GTC,
        )

        self.submit_order(close_order)
        self.log.info(f"Closing position of {pos.quantity} units")

    def on_order_filled(self, event) -> None:
        # Record the position ID, side, and entry price from the event.
        self.position_id = event.position_id
        self.position_side = event.order_side
        self.entry_price = float(event.last_px)
        self.trailing_stop = None  # Reset trailing stop on new entry
        self.log.info(f"Position {self.position_id} opened at {self.entry_price}")

    def on_position_closed(self, event) -> None:
        self.position_id = None
        self.position_side = None
        self.entry_price = None
        self.trailing_stop = None
        self.log.info("Position closed")

    def on_stop(self) -> None:
        self.cancel_all_orders(self.config.instrument_id)
        self.close_all_positions(self.config.instrument_id)
        self.log.info("Strategy stopped")
