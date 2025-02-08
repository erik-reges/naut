#!/usr/bin/env python3
# -------------------------------------------------------------------------------------------------
#  This backtest file sets up the NautilusTrader engine, registers an instrument and venue,
#  loads bar data from a Parquet 15min, and adds the RSIMACDBBStrategy (defined in strat.py)
#  for backtesting.
# -------------------------------------------------------------------------------------------------

import pandas as pd
from decimal import Decimal
from nautilus_trader.backtest.engine import BacktestEngine, BacktestEngineConfig
from nautilus_trader.config import LoggingConfig, RiskEngineConfig
from nautilus_trader.model.currencies import USDT
from nautilus_trader.model.data import BarType
from nautilus_trader.model.enums import AccountType, OmsType
from nautilus_trader.model.identifiers import TraderId, Venue
from nautilus_trader.model.objects import Money
from nautilus_trader.persistence.catalog.parquet import ParquetDataCatalog
from nautilus_trader.test_kit.providers import TestInstrumentProvider

from strat import RSIMACDBBStrategy, RSIMACDBBStrategyConfig

# --- Instrument and Engine Setup ---
instrument = TestInstrumentProvider.ethusdt_binance()
engine_config = BacktestEngineConfig(
    trader_id=TraderId("BACKTESTER-RSIMA"),
    logging=LoggingConfig(log_level="INFO"),
    risk_engine=RiskEngineConfig(bypass=True)
)
engine = BacktestEngine(config=engine_config)

# --- Define Venue ---
BINANCE = Venue("BINANCE")
engine.add_venue(
    venue=BINANCE,
    oms_type=OmsType.NETTING,
    account_type=AccountType.MARGIN,
    base_currency=None,
    starting_balances=[Money(1_000_0, USDT)]
)

# --- Register Instrument ---
engine.add_instrument(instrument)

# --- Load Bar Data from Parquet ---
catalog = ParquetDataCatalog("../catalogs/ETHUSDT/15min")  # Adjust this path as needed
rsima_bar_type = BarType.from_str(f"{instrument.id}-1-MINUTE-LAST-EXTERNAL")
bars = catalog.bars([str(rsima_bar_type)])
engine.add_data(bars)

# --- Strategy Setup ---
rsima_config = RSIMACDBBStrategyConfig(
    instrument_id=instrument.id,
    bar_type=rsima_bar_type,
    bars_required=50,
    rsi_period=14,
    bb_period=20,
    rsi_long_threshold=30.0,
    rsi_exit_threshold=70.0,
    investment_fraction=0.1
)
strategy = RSIMACDBBStrategy(config=rsima_config)
engine.add_strategy(strategy=strategy)

# --- Run Backtest ---
engine.run()

# --- Generate and Display Reports ---
with pd.option_context("display.max_rows", 100, "display.max_columns", None, "display.width", 300):
    print(engine.trader.generate_account_report(BINANCE))
    print(engine.trader.generate_order_fills_report())
    print(engine.trader.generate_positions_report())

# --- Cleanup ---
engine.reset()
engine.dispose()
