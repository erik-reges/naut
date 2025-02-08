import pandas as pd
from sklearn.model_selection import ParameterGrid

from nautilus_trader.backtest.engine import BacktestEngine, BacktestEngineConfig
from nautilus_trader.config import LoggingConfig, RiskEngineConfig
from nautilus_trader.model.currencies import USD
from nautilus_trader.model.data import BarType
from nautilus_trader.model.enums import AccountType, OmsType
from nautilus_trader.model.identifiers import TraderId, Venue
from nautilus_trader.model.objects import Money
from nautilus_trader.persistence.catalog.parquet import ParquetDataCatalog
from nautilus_trader.test_kit.providers import TestInstrumentProvider

from macross2 import EMA528Strategy, EMA528StrategyConfig


def run_backtest(strategy_config: EMA528StrategyConfig) -> float:
    engine_config = BacktestEngineConfig(
        trader_id=TraderId("BACKTESTER-EMA528"),
        logging=LoggingConfig(log_level="INFO", print_config=False),
        risk_engine=RiskEngineConfig(bypass=True),
    )
    engine = BacktestEngine(config=engine_config)

    # Set up the trading venue.
    VENUE = Venue("XNAS")
    engine.add_venue(
        venue=VENUE,
        oms_type=OmsType.NETTING,
        account_type=AccountType.MARGIN,
        base_currency=USD,
        starting_balances=[Money(10_000, USD)],
    )

    # Register the primary instrument (SPY).
    SPY = TestInstrumentProvider.equity(symbol="SPY", venue="XNAS")
    engine.add_instrument(SPY)

    # Define the bar type (using a 30-minute timeframe).
    timeframe = "30-MINUTE"
    spy_bar_type = BarType.from_str(f"{SPY.id}-{timeframe}-LAST-EXTERNAL")

    # Load bar data from the local catalog.
    catalog = ParquetDataCatalog("../catalogs/spy/30min")
    bars = catalog.bars([str(spy_bar_type)])
    engine.add_data(bars)

    # Add the strategy with the current configuration.
    strategy = EMA528Strategy(config=strategy_config)
    engine.add_strategy(strategy)

    engine.run()

    # Generate account report and extract the final free balance.
    report_df = engine.trader.generate_account_report(VENUE)
    report_df = report_df.sort_index()
    usd_report = report_df[report_df['currency'] == 'USD']
    final_value = float(usd_report.iloc[-1]['free']) if not usd_report.empty else 10_000
    profit = final_value - 10_000

    engine.reset()
    engine.dispose()

    return profit


param_grid = {
    "ema_period": [15],
    "stop_pct": [ 0.025, 0.027, 0.029],
    "trailing_gap_pct": [0.015, 0.025],
    "confirm_bars": [15, 17],
    "position_size": [1],
}

grid = list(ParameterGrid(param_grid))
results = []

for params in grid:
    print("Running backtest with parameters:", params)

    # Create instrument and bar type.
    SPY = TestInstrumentProvider.equity(symbol="SPY", venue="XNAS")
    timeframe = "30-MINUTE"
    spy_bar_type = BarType.from_str(f"{SPY.id}-{timeframe}-LAST-EXTERNAL")

    # Build strategy configuration using current grid parameters.
    strategy_config = EMA528StrategyConfig(
        instrument_id=SPY.id,
        bar_type=spy_bar_type,
        ema_period=params["ema_period"],
        stop_pct=params["stop_pct"],
        trailing_gap_pct=params["trailing_gap_pct"],
        confirm_bars=params["confirm_bars"],
        position_size=params["position_size"],
    )

    profit = run_backtest(strategy_config)
    print("Profit for these parameters:", profit)
    results.append({**params, "profit": profit})

df_results = pd.DataFrame(results)
print("\nGrid search results:")
print(df_results)

top_configs = df_results.nlargest(20, "profit")
print("\nTop configurations:")
with pd.option_context('display.max_columns', None, 'display.width', None):
    print(top_configs)
