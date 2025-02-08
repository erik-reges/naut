#!/usr/bin/env python3
import pandas as pd
from nautilus_trader.backtest.engine import BacktestEngine, BacktestEngineConfig
from nautilus_trader.config import LoggingConfig, RiskEngineConfig
from nautilus_trader.model.currencies import USDT, ETH
from nautilus_trader.model.data import BarType
from nautilus_trader.model.enums import AccountType, OmsType
from nautilus_trader.model.identifiers import TraderId, Venue
from nautilus_trader.model.objects import Money
from nautilus_trader.persistence.catalog.parquet import ParquetDataCatalog
from nautilus_trader.test_kit.providers import TestInstrumentProvider

# Import the strategy and its configuration.
from strat import KNNBasedStrategy, KNNBasedStrategyConfig

# --- Instrument and Engine Setup ---
instrument = TestInstrumentProvider.ethusdt_binance()
engine_config = BacktestEngineConfig(
    trader_id=TraderId("BACKTESTER-KNN"),
    logging=LoggingConfig(log_level="INFO", print_config=True),
    risk_engine=RiskEngineConfig(bypass=True),
)
engine = BacktestEngine(config=engine_config)

# --- Define Venue ---
BINANCE = Venue("BINANCE")
engine.add_venue(
    venue=BINANCE,
    oms_type=OmsType.NETTING,
    account_type=AccountType.CASH,  # Using a cash account (no shorting)
    base_currency=None,
    starting_balances=[Money(1_000_000, USDT), Money(25, ETH)],
)

# --- Register Instrument ---
engine.add_instrument(instrument)

# --- Load Bar Data from Parquet ---
catalog = ParquetDataCatalog("../catalogs/ETHUSDT/1h")  # Adjust path as needed
knn_bar_type = BarType.from_str(f"{instrument.id}-60-MINUTE-LAST-EXTERNAL")
bars = catalog.bars([str(knn_bar_type)])
engine.add_data(bars)

# --- Strategy Configuration ---
knn_config = KNNBasedStrategyConfig(
    instrument_id=instrument.id,
    bar_type=knn_bar_type,
    start_date="2024-07-01T00:00:00Z",
    stop_date="2024-10-31T23:45:00Z",
    indicator="ALL",
    short_window=7,
    long_window=14,
    base_k=252,              # k becomes floor(sqrt(20)) = 4
    volatility_filter=True,
    bars_required=30,
    bars_threshold=50,
    neighbour_pct=0.5,  # e.g. require >=50% of neighbors as 'up'
    use_trend_filter=True,
    sma_period=50,# period for trend (SMA) filter
    use_volume_filter=True,
    volume_ma_period= 50
)

# --- Create and Register the Strategy ---
strategy = KNNBasedStrategy(config=knn_config)
engine.add_strategies([strategy])

# --- Run Backtest ---
engine.run()

# --- Generate and Display Reports ---
with pd.option_context("display.max_rows", 100, "display.max_columns", None, "display.width", 300):
    print(engine.trader.generate_account_report(BINANCE))
    # Optionally:
    # print(engine.trader.generate_order_fills_report())
    # print(engine.trader.generate_positions_report())

# --- Cleanup ---
engine.reset()
engine.dispose()
