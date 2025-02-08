from decimal import Decimal
from nautilus_trader.persistence.catalog.parquet import ParquetDataCatalog
import pandas as pd

from nautilus_trader.backtest.engine import BacktestEngine, BacktestEngineConfig
from nautilus_trader.config import LoggingConfig
from nautilus_trader.model.currencies import USD
from nautilus_trader.model.data import BarType
from nautilus_trader.model.enums import AccountType, OmsType
from nautilus_trader.model.identifiers import TraderId, Venue
from nautilus_trader.model.objects import Money
from nautilus_trader.test_kit.providers import TestInstrumentProvider
from ma528 import EMA528Strategy, EMA528StrategyConfig

if __name__ == "__main__":
    # Configure backtest engine.
    engine_config = BacktestEngineConfig(
        trader_id=TraderId("BACKTESTER-001"),
        logging=LoggingConfig(log_level="INFO"),
    )
    engine = BacktestEngine(config=engine_config)

    # Add trading venue.
    NASDAQ = Venue("XNAS")
    engine.add_venue(
        venue=NASDAQ,
        oms_type=OmsType.NETTING,
        account_type=AccountType.MARGIN,
        base_currency=USD,
        starting_balances=[Money(10_000, USD)],
    )

    # Register the primary instrument (SPY).
    SPY = TestInstrumentProvider.equity(symbol="SPY", venue="XNAS")
    engine.add_instrument(SPY)

    timeframe = "30-MINUTE"
    # Create bar types using the registered instrument's ID.
    spy_bar_type = BarType.from_str(f"{SPY.id}-{timeframe}-LAST-EXTERNAL")

    # Load bar data from local catalogs.
    spy_catalog = ParquetDataCatalog("../catalogs/spy/30min")
    spy_bars = spy_catalog.bars([str(spy_bar_type)])
    engine.add_data(spy_bars)

    # Use a shorter EMA period for the backtest (and adjust other parameters if desired).
    config = EMA528StrategyConfig(
        ema_period=15,  # shorter EMA for backtesting purposes
        instrument_id=SPY.id,
        bar_type=spy_bar_type,
        confirm_bars=17,
        trailing_gap_pct=0.025,
        stop_pct=0.025# For a quicker backtest, you might reduce confirmation requirement
    )

    strategy = EMA528Strategy(config=config)
    engine.add_strategy(strategy)

    # Run the backtest.
    engine.run()

    # Print results.
    print(engine.get_result())
