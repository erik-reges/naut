# backtest.py
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
from strat import BBRSIStrategy, BBRSIStrategyConfig

if __name__ == "__main__":
    # Configure backtest engine
    engine_config = BacktestEngineConfig(
        trader_id=TraderId("BACKTESTER-001"),
        logging=LoggingConfig(log_level="INFO"),
    )
    engine = BacktestEngine(config=engine_config)

    # Add trading venue
    NASDAQ = Venue("XNAS")
    engine.add_venue(
        venue=NASDAQ,
        oms_type=OmsType.NETTING,
        account_type=AccountType.MARGIN,
        base_currency=USD,
        starting_balances=[Money(10_000, USD)],
    )

    # Register the primary instrument (SPY)
    SPY = TestInstrumentProvider.equity(symbol="SPY", venue="XNAS")
    engine.add_instrument(SPY)

    timeframe = "1-HOUR"
    # Create bar types using the registered instrument's ID.
    spy_bar_type = BarType.from_str(f"{SPY.id}-{timeframe}-LAST-EXTERNAL")

    # Load bar data from your local catalogs.
    spy_catalog = ParquetDataCatalog("../catalogs/spy/1H")
    spy_bars = spy_catalog.bars([str(spy_bar_type)])
    engine.add_data(spy_bars)

    # Updated strategy configuration with best parameters so far.
    strategy_config = BBRSIStrategyConfig(
        instrument_id=SPY.id,
        bar_type=spy_bar_type,
        rsi_period=9,
        rsi_threshold_low=0.35,
        rsi_threshold_high=0.65,
        bb_period=20,
        k=2.47,
        bb_width_threshold=0.0010,
        atr_period=9,
        sl_atr_multiple=1.16,
        tp_atr_multiple=3.4,
        sma_period=50,
        position_size=1,


    )

    strategy = BBRSIStrategy(config=strategy_config)
    engine.add_strategy(strategy)

    # Run backtest
    engine.run()

    # Print results
    print(engine.get_result())
