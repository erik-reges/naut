import time
from decimal import Decimal
import pandas as pd

from nautilus_trader.backtest.engine import BacktestEngine, BacktestEngineConfig
from nautilus_trader.config import LoggingConfig, RiskEngineConfig
from nautilus_trader.model.currencies import USD
from nautilus_trader.model.data import BarType
from nautilus_trader.model.enums import AccountType, OmsType
from nautilus_trader.model.identifiers import TraderId, Venue
from nautilus_trader.model.objects import Money
from nautilus_trader.persistence.catalog.parquet import ParquetDataCatalog
from nautilus_trader.test_kit.providers import TestInstrumentProvider

# Import the adapted strategy.
from strategy import ESVIXStrategy, ESVIXStrategyConfig

if __name__ == "__main__":
    # Configure the backtest engine.
    engine_config = BacktestEngineConfig(
        trader_id=TraderId("BACKTESTER-ESVIX"),
        logging=LoggingConfig(log_level="INFO", print_config=True),
        risk_engine=RiskEngineConfig(bypass=True),
    )
    engine = BacktestEngine(config=engine_config)

    # Add a trading venue.
    NASDAQ = Venue("XNAS")
    engine.add_venue(
        venue=NASDAQ,
        oms_type=OmsType.NETTING,
        account_type=AccountType.CASH,
        base_currency=USD,
        starting_balances=[Money(1_000_000.0, USD)],
    )

    # Register the primary instrument (SPY)
    SPY = TestInstrumentProvider.equity(symbol="SPY", venue="XNAS")
    engine.add_instrument(SPY)

    # Register the UVXY instrument as well.
    UVXY = TestInstrumentProvider.equity(symbol="UVXY", venue="XNAS")
    engine.add_instrument(UVXY)

    timeframe = "1-HOUR"

    # Create bar types using the registered instruments' IDs.
    spy_bar_type = BarType.from_str(f"{SPY.id}-{timeframe}-LAST-EXTERNAL")
    uvxy_bar_type = BarType.from_str(f"{UVXY.id}-{timeframe}-LAST-EXTERNAL")

    # Load bar data from your local catalogs.
    spy_catalog = ParquetDataCatalog("../catalogs/spy/1H")
    uvxy_catalog = ParquetDataCatalog("../catalogs/uvxy/1H")

    spy_bars = spy_catalog.bars([str(spy_bar_type)])
    uvxy_bars = uvxy_catalog.bars([str(uvxy_bar_type)])
    # Add both data streams to the engine.
    engine.add_data(spy_bars + uvxy_bars)

    # Updated configuration with tighter risk and additional volatility filtering.
    strategy_config = ESVIXStrategyConfig(
        instrument_id=SPY.id,
        bar_type=spy_bar_type,
        vix_bar_type=uvxy_bar_type,
        ema_period=50,              # Longer EMA for smoother trend filtering
        macd_fast=7,               # Adjusted MACD settings to reduce noise
        macd_slow=14,
        macd_signal=7,
        rsi_period=14,
        vix_ma_period=20,           # Moving average period for UVXY (volatility filter)
        max_risk_pct=0.16,          # Lower risk per trade (3% of free balance)
        trailing_multiplier=2,    # Tighter trailing stop multiplier
        trailing_atr_period=14,
        bars_required=50,
        bars_threshold=500,
        profit_target_atr_multiple=3.4,  # Reduced profit target ATR multiple
        stop_loss_atr_multiple=1.3      # Tighter stop loss ATR multiple
    )
    strategy = ESVIXStrategy(config=strategy_config)
    engine.add_strategy(strategy)

    # Run the backtest engine.
    engine.run()

    print(engine.get_result())
    # Reset and dispose of the engine for clean-up.
    engine.reset()
    engine.dispose()
