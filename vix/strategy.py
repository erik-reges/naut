from typing import List, Optional

from nautilus_trader.config import StrategyConfig
from nautilus_trader.trading.strategy import Strategy
from nautilus_trader.model.data import Bar, BarType
from nautilus_trader.model.enums import OrderSide, TimeInForce, PriceType
from nautilus_trader.model.currencies import USD
from nautilus_trader.indicators.average.ma_factory import MovingAverageFactory, MovingAverageType
from nautilus_trader.indicators.atr import AverageTrueRange
from nautilus_trader.indicators.macd import MovingAverageConvergenceDivergence
from nautilus_trader.indicators.rsi import RelativeStrengthIndex

class ESVIXStrategyConfig(StrategyConfig, frozen=True):
    instrument_id: str
    bar_type: BarType
    vix_bar_type: BarType
    ema_period: int = 20
    macd_fast: int = 10
    macd_slow: int = 20
    macd_signal: int = 7
    rsi_period: int = 14
    vix_ma_period: int = 20
    max_risk_pct: float = 0.03
    trailing_multiplier: float = 1.8
    trailing_atr_period: int = 14
    bars_required: int = 50
    bars_threshold: int = 500
    # New parameters:
    profit_target_atr_multiple: float = 1.5   # Exit if profit exceeds 1.5 ATR above entry
    stop_loss_atr_multiple: float = 0.7       # Exit if price drops 0.7 ATR below entry

class ESVIXStrategy(Strategy):
    def __init__(self, config: ESVIXStrategyConfig):
        super().__init__(config)

        # History and state management
        self.es_history: List[Bar] = []
        self.vix_history: List[Bar] = []
        self.position_id: Optional[str] = None
        self.entry_price: Optional[float] = None
        self.trailing_stop: Optional[float] = None
        self.bars_counter: int = 0

        # Initialize indicators for SPY bars.
        self.es_ema = MovingAverageFactory.create(
            period=self.config.sma_period,
            ma_type=MovingAverageType.EXPONENTIAL,
            price_type=PriceType.LAST,
        )

        self.macd = MovingAverageConvergenceDivergence(
            fast_period=self.config.macd_fast,
            slow_period=self.config.macd_slow,
            ma_type=MovingAverageType.EXPONENTIAL,
            price_type=PriceType.LAST,
        )

        self.rsi = RelativeStrengthIndex(
            period=self.config.rsi_period,
            ma_type=MovingAverageType.EXPONENTIAL,
        )

        self.atr = AverageTrueRange(
            period=self.config.trailing_atr_period,
            ma_type=MovingAverageType.SIMPLE,
        )

        # Initialize the moving average for UVXY (volatility indicator).
        self.vix_ma = MovingAverageFactory.create(
            period=self.config.vix_ma_period,
            ma_type=MovingAverageType.EXPONENTIAL,
            price_type=PriceType.LAST,
        )

    def on_start(self):
        self.instrument = self.cache.instrument(self.config.instrument_id)
        if self.instrument is None:
            self.log.error(f"Instrument {self.config.instrument_id} not found.")
            self.stop()
            return

        self.register_indicator_for_bars(self.config.bar_type, self.es_ema)
        self.register_indicator_for_bars(self.config.bar_type, self.macd)
        self.register_indicator_for_bars(self.config.bar_type, self.rsi)
        self.register_indicator_for_bars(self.config.bar_type, self.atr)
        self.register_indicator_for_bars(self.config.vix_bar_type, self.vix_ma)

        self.subscribe_bars(self.config.bar_type)
        self.subscribe_bars(self.config.vix_bar_type)
        self.log.info("ESVIXStrategy started.")

    def on_bar(self, bar: Bar) -> None:
        if bar.bar_type == self.config.bar_type:
            self.es_history.append(bar)
            if len(self.es_history) > self.config.bars_required:
                self.es_history = self.es_history[-self.config.bars_required:]
            self._process_es_bar(bar)
        elif bar.bar_type == self.config.vix_bar_type:
            self.vix_history.append(bar)
            if len(self.vix_history) > self.config.bars_required:
                self.vix_history = self.vix_history[-self.config.bars_required:]
            # The vix_ma indicator is automatically updated via the registered subscription.
            # Removed: self.vix_ma.on_bar(bar)

    def _process_es_bar(self, bar: Bar) -> None:
        if not all([
            self.es_ema.initialized,
            self.macd.initialized,
            self.rsi.initialized,
            self.atr.initialized,
            self.vix_ma.initialized,
        ]):
            return

        current_price = float(bar.close)

        # If in a position, manage exit conditions.
        if self.position_id is not None:
            self.bars_counter += 1
            if self.entry_price is not None:
                momentum = current_price / self.entry_price - 1
                dynamic_multiplier = min(max(self.config.trailing_multiplier * (1 + momentum), 1.5), 3.0)
                new_stop = current_price - (dynamic_multiplier * self.atr.value)
                if self.trailing_stop is None or new_stop > self.trailing_stop:
                    self.trailing_stop = new_stop
                    self.log.debug(f"Updated trailing stop to {self.trailing_stop}")

                # Exit if profit target is met.
                profit_target = self.entry_price + self.config.profit_target_atr_multiple * self.atr.value
                if current_price >= profit_target:
                    self.log.info(f"Profit target reached: {current_price:.2f} >= {profit_target:.2f}")
                    self.close_position()
                    return

                # Hard stop-loss check.
                stop_loss_price = self.entry_price - self.config.stop_loss_atr_multiple * self.atr.value
                if current_price <= stop_loss_price:
                    self.log.info(f"Stop loss triggered: {current_price:.2f} <= {stop_loss_price:.2f}")
                    self.close_position()
                    return

            if self.trailing_stop is not None and current_price < self.trailing_stop:
                self.log.info(f"Trailing stop hit at {current_price:.2f}")
                self.close_position()
                return

            if self.bars_counter >= self.config.bars_threshold:
                self.log.info("Maximum bars threshold reached")
                self.close_position()
                return

        # Entry conditions.
        trend_condition = current_price > self.es_ema.value
        momentum_condition = self.macd.value > 0
        rsi_condition = self.rsi.value < 62.5

        # VIX condition: enter only if UVXY price is below its moving average (indicating receding volatility).
        if self.vix_history:
            current_vix_price = float(self.vix_history[-1].close)
            vix_condition = current_vix_price < self.vix_ma.value
        else:
            vix_condition = False

        self.log.debug(f"Entry check: price {current_price:.2f}, EMA {self.es_ema.value:.2f}, "
                       f"MACD {self.macd.value:.4f}, RSI {self.rsi.value:.2f}, "
                       f"UVXY {current_vix_price if self.vix_history else 'N/A'}, VIX MA {self.vix_ma.value:.2f}")

        if trend_condition and momentum_condition and rsi_condition and vix_condition:
            self.log.info(f"Entry conditions met: BUY at {current_price:.2f}")
            self._enter_position(OrderSide.BUY, current_price)

    def _enter_position(self, side: OrderSide, current_price: float) -> None:
        if self.position_id is not None:
            return

        account = self.portfolio.account(self.instrument.id.venue)
        free_balance = account.balance_free(USD).as_double()
        risk_amount = free_balance * self.config.max_risk_pct
        stop_distance = self.atr.value * self.config.trailing_multiplier
        if stop_distance <= 0:
            self.log.warning("Invalid stop distance calculated")
            return

        # Determine position size based on risk.
        position_size = risk_amount / stop_distance
        # Enforce a maximum trade value constraint (e.g., 12.5% of free balance).
        max_trade_value = free_balance * 0.125
        max_qty_value = max_trade_value / current_price
        position_size = min(position_size, max_qty_value)

        self.log.info(f"Buying for: {position_size * current_price:.2f}")
        order = self.order_factory.market(
            instrument_id=self.config.instrument_id,
            order_side=side,
            quantity=self.instrument.make_qty(position_size),
            time_in_force=TimeInForce.GTC,
        )
        self.submit_order(order)
        self.entry_price = current_price
        self.trailing_stop = current_price - stop_distance
        self.bars_counter = 0
        self.log.info(f"Entered {side} position at {current_price:.2f}")

    def close_position(self, instrument_id=None, order_side=None, quantity=None, time_in_force=None) -> None:
        if self.position_id is None:
            return
        pos = self.cache.position(self.position_id)
        if pos is None or pos.quantity <= 0:
            self._reset_position_state()
            return
        qty = self.instrument.make_qty(pos.quantity)
        if qty <= 0:
            self.log.warning(f"Calculated close quantity is too small: {qty}; skipping close order.")
            self.position_id = None
            self._reset_position_state()
            return
        order = self.order_factory.market(
            instrument_id=self.config.instrument_id,
            order_side=OrderSide.SELL,
            quantity=qty,
            time_in_force=TimeInForce.GTC,
        )
        self.submit_order(order)
        self.log.info(f"Closing position of {pos.quantity} units")

    def _reset_position_state(self) -> None:
        self.position_id = None
        self.entry_price = None
        self.trailing_stop = None
        self.bars_counter = 0

    def on_order_filled(self, event) -> None:
        if not hasattr(event, 'position_id'):
            return
        if self.position_id is None:
            self.position_id = event.position_id
            self.entry_price = float(event.last_px) if hasattr(event, 'last_px') else None
            self.bars_counter = 0
            self.log.info(f"Position {self.position_id} opened at {self.entry_price}")
        elif str(event.position_id) == str(self.position_id):
            self._reset_position_state()
            self.log.info("Position closed")

    def on_stop(self) -> None:
        self.cancel_all_orders(self.config.instrument_id)
        self.close_all_positions(self.config.instrument_id)
        self.log.info("Strategy stopped")
