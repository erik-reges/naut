from typing import Optional
from nautilus_trader.config import StrategyConfig
from nautilus_trader.indicators.rsi import RelativeStrengthIndex
from nautilus_trader.trading.strategy import Strategy
from nautilus_trader.model.data import Bar, BarType
from nautilus_trader.model.enums import OrderSide, TimeInForce, PriceType
from nautilus_trader.indicators.average.ma_factory import MovingAverageFactory, MovingAverageType

class EMA528StrategyConfig(StrategyConfig, frozen=True):
    instrument_id: str
    bar_type: BarType
    ema_period: int = 500             # EMA period (e.g. 500 bars)
    stop_pct: float = 0.016           # Fixed stop: 1.6% below entry price
    trailing_gap_pct: float = 0.034    # Trailing gap: 3% below the maximum price reached
    position_size: float = 1          # Fraction of account equity to use per trade
    confirm_bars: int = 18            # Number of consecutive bars confirming the signal
    rsi_period: int = 20

class EMA528Strategy(Strategy):
    def __init__(self, config: EMA528StrategyConfig):
        super().__init__(config)
        self.position_id: Optional[str] = None
        self.entry_price: Optional[float] = None
        self.trailing_max: Optional[float] = None
        self.trailing_tp: Optional[float] = None
        self.position_side: Optional[OrderSide] = None

        # Counters for consecutive confirmation bars.
        self.entry_confirmation_count: int = 0
        self.exit_confirmation_count: int = 0

        # Create an EMA indicator using an exponential moving average.
        self.ema = MovingAverageFactory.create(
            period=self.config.ema_period,
            ma_type=MovingAverageType.EXPONENTIAL,
            price_type=PriceType.LAST,
        )

        self.rsi = RelativeStrengthIndex(
            period=self.config.rsi_period,
            ma_type=MovingAverageType.EXPONENTIAL,
        )

    def on_start(self):
        self.log.info("EMA528 Strategy starting...")
        self.instrument = self.cache.instrument(self.config.instrument_id)
        if self.instrument is None:
            self.log.error(f"Instrument {self.config.instrument_id} not found.")
            self.stop()
            return

        # Register our EMA indicator and subscribe to bars.
        self.register_indicator_for_bars(self.config.bar_type, self.ema)
        self.register_indicator_for_bars(self.config.bar_type, self.rsi)

        self.subscribe_bars(self.config.bar_type)
        self.log.info("Strategy initialized.")

    def on_bar(self, bar: Bar) -> None:
        # Ensure the EMA is fully initialized.
        if not self.ema.initialized:
            return

        current_price = float(bar.close)

        # ========= ENTRY LOGIC ==========
        if self.position_id is None:
            # If the current bar closes below the EMA, increment the confirmation counter.
            if current_price < self.ema.value:
                self.entry_confirmation_count += 1
                self.log.debug(
                    f"Entry confirmation count increased to {self.entry_confirmation_count} (current price {current_price:.2f} < EMA {self.ema.value:.2f})."
                )
            else:
                if self.entry_confirmation_count > 0:
                    self.log.debug("Entry confirmation reset to 0.")
                self.entry_confirmation_count = 0

            # If we have enough consecutive confirmation bars, enter the long position.
            if self.entry_confirmation_count >= self.config.confirm_bars:
                self.log.info(
                    f"Entry confirmed for {self.config.confirm_bars} consecutive bars (price {current_price:.2f} < EMA {self.ema.value:.2f})."
                )
                self._enter_long(current_price)
                # Reset the confirmation counter after entering.
                self.entry_confirmation_count = 0

        else:
            # ========= EXIT LOGIC ==========
            # Update the trailing maximum if the current price is a new high.
            if self.trailing_max is None or current_price > self.trailing_max:
                self.trailing_max = current_price
                self.trailing_tp = self.trailing_max * (1 - self.config.trailing_gap_pct)
            #    self.log.info(
      #              f"Updated trailing max: {self.trailing_max:.2f}, trailing TP set to: {self.trailing_tp:.2f}"
            #    )
                # Reset the exit confirmation counter since we have a new high.
                self.exit_confirmation_count = 0

            # Calculate the fixed stop loss level.
            fixed_stop_level = self.entry_price * (1 - self.config.stop_pct)
            # Exit immediately if the fixed stop is breached.
            if current_price < fixed_stop_level:
                self.log.info(
                    f"Fixed stop triggered: current price {current_price:.2f} < fixed stop level {fixed_stop_level:.2f}. Exiting position."
                )
                self.close_position()
                return

            # Check the trailing take profit condition.
            if self.trailing_tp is not None and current_price < self.trailing_tp:
                self.exit_confirmation_count += 1
                self.log.debug(
                    f"Exit confirmation count increased to {self.exit_confirmation_count} (current price {current_price:.2f} < trailing TP {self.trailing_tp:.2f})."
                )
            else:
                if self.exit_confirmation_count > 0:
                    self.log.debug("Exit confirmation reset to 0.")
                self.exit_confirmation_count = 0

            # Exit if we have enough consecutive bars breaching the trailing TP.
            if self.exit_confirmation_count >= self.config.confirm_bars:
                self.log.info(
                    f"Trailing exit confirmed for {self.config.confirm_bars} consecutive bars (price {current_price:.2f} < trailing TP {self.trailing_tp:.2f}). Exiting position."
                )
                self.close_position()
                return

        # (Optional) You can add additional logging or state updates here.

    def _enter_long(self, current_price: float) -> None:
        # Calculate the position quantity based on account equity and position size.
        print(f"Entering long: {self.rsi.value:.2f}")
        account = self.portfolio.account(self.instrument.id.venue)
        equity = account.balance_total().as_double()  # Returns the account's total equity.
        position_value = equity * self.config.position_size
        quantity = int(position_value / current_price)

        if quantity < 1:
            self.log.warning("Calculated quantity less than 1. Not entering trade.")
            return

        order = self.order_factory.market(
            instrument_id=self.config.instrument_id,
            order_side=OrderSide.BUY,
            quantity=self.instrument.make_qty(quantity),
            time_in_force=TimeInForce.GTC,
        )
        self.submit_order(order)
        self.log.info(f"Submitted order to enter long position at {current_price:.2f}")

    def close_position(self, instrument_id=None, order_side=None, quantity=None, time_in_force=None) -> None:
        if self.position_id is None:
            return

        pos = self.cache.position(self.position_id)
        if pos is None or pos.quantity <= 0:
            return

        # Determine the closing side based on our open position.
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
        self.log.info(f"Submitted order to close position of {pos.quantity} units.")

    def on_order_filled(self, event) -> None:
        # Record the position details when the order is filled.
        self.position_id = event.position_id
        self.entry_price = float(event.last_px)
        self.trailing_max = self.entry_price
        self.trailing_tp = self.entry_price * (1 - self.config.trailing_gap_pct)
        self.position_side = event.order_side
        self.log.info(
            f"Order filled. Entry price: {self.entry_price:.2f}, initial trailing TP set to: {self.trailing_tp:.2f}"
        )

    def on_position_closed(self, event) -> None:
        # Reset state when the position is closed.
        self.log.info("Position closed. Resetting state.")
        self.position_id = None
        self.entry_price = None
        self.trailing_max = None
        self.trailing_tp = None
        self.position_side = None
        self.entry_confirmation_count = 0
        self.exit_confirmation_count = 0

    def on_stop(self) -> None:
        self.cancel_all_orders(self.config.instrument_id)
        self.close_all_positions(self.config.instrument_id)
        self.log.info("Strategy stopped.")
