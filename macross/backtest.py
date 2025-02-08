from pathlib import Path
import time
import pandas as pd
from decimal import Decimal
from nautilus_trader.backtest.engine import BacktestEngine, BacktestEngineConfig
from nautilus_trader.config import LoggingConfig, RiskEngineConfig
from nautilus_trader.model.currencies import ETH, USDT
from nautilus_trader.model.data import BarType
from nautilus_trader.model.enums import AccountType, OmsType
from nautilus_trader.model.identifiers import TraderId, Venue
from nautilus_trader.model.objects import Money
from nautilus_trader.test_kit.providers import TestInstrumentProvider
from utils import load_trades_to_bars
from strat import MACrossConfig, MACrossStrategy

# setup
instrument = TestInstrumentProvider.ethusdt_binance()
engine_config = BacktestEngineConfig(
    trader_id=TraderId("BACKTESTER-ETH"),
    logging=LoggingConfig(log_level="INFO"),
    risk_engine=RiskEngineConfig(bypass=True)
)
engine = BacktestEngine(config=engine_config)

# venue
BINANCE = Venue("BINANCE")
engine.add_venue(
    venue=BINANCE,
    oms_type=OmsType.NETTING,
    account_type=AccountType.MARGIN,
    base_currency=None,
    starting_balances=[Money(1_000_000, USDT), Money(10, ETH)],
)

# instrument
engine.add_instrument(instrument)

# bars
eth_bar_type = BarType.from_str(f"{instrument.id}-1-MINUTE-LAST-EXTERNAL")
bars = load_trades_to_bars(
    folder_path="../data",
    instrument=instrument,
    bar_type=eth_bar_type,
    timeframe="1min"
)
engine.add_data(bars)

# strat
ma_config = MACrossConfig(
    instrument_id=instrument.id,
    bar_type=eth_bar_type,
    trade_size="100"
)
strategy = MACrossStrategy(config=ma_config)
engine.add_strategy(strategy=strategy)

# optional pause
time.sleep(0.1)
input("Press Enter to continue...")

# run
engine.run()

# reports
with pd.option_context("display.max_rows", 100, "display.max_columns", None, "display.width", 300):
    print(engine.trader.generate_account_report(BINANCE))
    print(engine.trader.generate_order_fills_report())
    print(engine.trader.generate_positions_report())

# cleanup
engine.reset()
engine.dispose()
