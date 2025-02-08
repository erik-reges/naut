import pandas as pd

from nautilus_trader.backtest.engine import BacktestEngine, BacktestEngineConfig
from nautilus_trader.config import LoggingConfig, RiskEngineConfig
from nautilus_trader.model.currencies import USDT
from nautilus_trader.model.data import BarType
from nautilus_trader.model.enums import AccountType, OmsType
from nautilus_trader.model.identifiers import TraderId, Venue
from nautilus_trader.model.objects import Money
from nautilus_trader.persistence.catalog.parquet import ParquetDataCatalog
from nautilus_trader.test_kit.providers import TestInstrumentProvider

# Import the simple strategy classes
from strat import SimpleMACrossoverStrategy, SimpleMACrossoverStrategyConfig

# 1) Setup Instrument and Engine
instrument = TestInstrumentProvider.ethusdt_binance()
engine_config = BacktestEngineConfig(
    trader_id=TraderId("BACKTESTER-MA"),
    logging=LoggingConfig(log_level="INFO"),
    risk_engine=RiskEngineConfig(bypass=True),
)
engine = BacktestEngine(config=engine_config)

# 2) Define Venue
BINANCE = Venue("BINANCE")
engine.add_venue(
    venue=BINANCE,
    oms_type=OmsType.NETTING,
    account_type=AccountType.CASH,
    base_currency=None,
    starting_balances=[Money(1_000_000, USDT)]
)

# 3) Register Instrument
engine.add_instrument(instrument)

# 4) Load Bar Data
catalog = ParquetDataCatalog("../catalogs/ETHUSDT/1h")  # Adjust this path
bar_type_60m = BarType.from_str(f"{instrument.id}-60-MINUTE-LAST-EXTERNAL")
bars = catalog.bars([str(bar_type_60m)])
engine.add_data(bars)

# 5) Create Strategy Config & Strategy
config = SimpleMACrossoverStrategyConfig(
    instrument_id=instrument.id,
    bar_type=bar_type_60m,
    short_window=25,         # Short-term moving average period for entry signals
    long_window=100,         # Long-term moving average period for trend confirmation
    atr_period=14,           # ATR period for calculating volatility and stop-loss levels
    atr_stop_multiple=1.5,   # Multiplier for setting the stop-loss distance based on ATR
    max_risk_pct=0.1,        # Maximum risk per trade (10% of account balance; consider lowering for live trading)
    use_daily_filter=True,   # Enable daily trend filter for additional confirmation
    daily_sma_period=20,     # Daily SMA period used in the higher timeframe filter
)

strategy = SimpleMACrossoverStrategy(config)
engine.add_strategies([strategy])

# 6) Run Backtest
engine.run()

# 7) Reporting
with pd.option_context("display.max_rows", 100, "display.max_columns", None, "display.width", 300):
    print(engine.trader.generate_account_report(BINANCE))
    # print(engine.trader.generate_order_fills_report())
    # print(engine.trader.generate_positions_report())

# Cleanup
engine.reset()
engine.dispose()
