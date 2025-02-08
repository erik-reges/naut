from pathlib import Path
import time
import pandas as pd

from nautilus_trader.adapters.databento.loaders import DatabentoDataLoader
from nautilus_trader.backtest.engine import BacktestEngine, BacktestEngineConfig
from nautilus_trader.config import LoggingConfig, RiskEngineConfig
from nautilus_trader.model.currencies import USD
from nautilus_trader.model.data import BarType
from nautilus_trader.model.enums import AccountType, OmsType
from nautilus_trader.model.identifiers import TraderId, Venue
from nautilus_trader.model.objects import Money
from nautilus_trader.test_kit.providers import TestInstrumentProvider

# Import the new MA Crossover strategy and its configuration.
from strat import MACrossoverStrategy, MACrossoverStrategyConfig

# Backtest engine configuration
engine_config = BacktestEngineConfig(
    trader_id=TraderId("BACKTESTER-002"),
    logging=LoggingConfig(log_level="INFO"),
    risk_engine=RiskEngineConfig(bypass=True),
)
engine = BacktestEngine(config=engine_config)

# Add trading venue
NASDAQ = Venue("XNAS")
engine.add_venue(
    venue=NASDAQ,
    oms_type=OmsType.NETTING,
    account_type=AccountType.CASH,
    base_currency=USD,
    starting_balances=[Money(1_000_000.0, USD)],
)

# Add instrument for SPY
SPY_XNAS = TestInstrumentProvider.equity(symbol="SPY", venue="XNAS")
engine.add_instrument(SPY_XNAS)

# Load data from Databento
loader = DatabentoDataLoader()
filenames = [
    "spy-xnas-202401-202403.trades.dbn.zst"
]
DATABENTO_DATA_DIR = Path("../databento")
for filename in filenames:
    trades = loader.from_dbn_file(
        path=DATABENTO_DATA_DIR / filename,
        instrument_id=SPY_XNAS.id,
    )
    engine.add_data(trades)

# Configure the MA Crossover strategy.
import time
start_date_ns = int(time.mktime(time.strptime("2024-01-01", "%Y-%m-%d"))) * 1_000_000_000
stop_date_ns  = int(time.mktime(time.strptime("2024-02-29", "%Y-%m-%d"))) * 1_000_000_000

SPY_bar_type = BarType.from_str(f"{SPY_XNAS.id}-1-MINUTE-LAST-EXTERNAL")

ma_config = MACrossoverStrategyConfig(
    instrument_id=SPY_XNAS.id,
    bar_type=SPY_bar_type,
    short_window=10,
    long_window=30,
)
strategy = MACrossoverStrategy(config=ma_config)
engine.add_strategy(strategy=strategy)

time.sleep(0.1)
input("Press Enter to continue...")

engine.run()

with pd.option_context("display.max_rows", 100, "display.max_columns", None, "display.width", 300):
    print(engine.trader.generate_account_report(NASDAQ))
    print(engine.trader.generate_order_fills_report())
    print(engine.trader.generate_positions_report())

engine.reset()
engine.dispose()
