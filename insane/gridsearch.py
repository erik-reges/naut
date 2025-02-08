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

from strat import BBRSIStrategy, BBRSIStrategyConfig


def run_backtest(bbrsi_config: BBRSIStrategyConfig) -> float:
    engine_config = BacktestEngineConfig(
        trader_id=TraderId("BACKTESTER-BBRSI"),
        logging=LoggingConfig(log_level="INFO", print_config=False),
        risk_engine=RiskEngineConfig(bypass=True),
    )
    engine = BacktestEngine(config=engine_config)

    # Set up the trading venue.
    VENUE = Venue("XNAS")
    engine.add_venue(
        venue=VENUE,
        oms_type=OmsType.NETTING,
        account_type=AccountType.CASH,
        base_currency=USD,
        starting_balances=[Money(10_000, USD)],
    )

    # Register the primary instrument (SPY).
    SPY = TestInstrumentProvider.equity(symbol="SPY", venue="XNAS")
    engine.add_instrument(SPY)

    timeframe = "1-HOUR"
    # Create bar type from SPY's ID.
    spy_bar_type = BarType.from_str(f"{SPY.id}-{timeframe}-LAST-EXTERNAL")

    # Load bar data from the local catalog.
    catalog = ParquetDataCatalog("../catalogs/spy/1H")
    bars = catalog.bars([str(spy_bar_type)])
    engine.add_data(bars)

    # Add strategy with the current configuration.
    strategy = BBRSIStrategy(config=bbrsi_config)
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


# Further refined grid search with very small increments around the best settings:
param_grid = {
    # Fine-tune the lookback period around the best-performing values (7 and 10)
    "rsi_period": [7, 8, 9, 10],
    # Test slight variations around the optimal low threshold (0.35)
    "rsi_threshold_low": [0.35, 0.36, 0.37],
    # Fine-tune the overbought threshold around 0.65–0.75
    "rsi_threshold_high": [0.65, 0.68, 0.70, 0.72, 0.75],
    # Keep the other parameters fixed at their best values
    "bb_period": [20],
    "k": [2.47],
    "bb_width_threshold": [0.0010],
    "atr_period": [9],
    "sl_atr_multiple": [1.16],
    "tp_atr_multiple": [3.4],
    "position_size": [1],
}


grid = list(ParameterGrid(param_grid))
results = []

for params in grid:
    print("Running backtest with parameters:", params)

    SPY = TestInstrumentProvider.equity(symbol="SPY", venue="XNAS")
    timeframe = "1-HOUR"
    spy_bar_type = BarType.from_str(f"{SPY.id}-{timeframe}-LAST-EXTERNAL")

    bbrsi_config = BBRSIStrategyConfig(
        instrument_id=SPY.id,
        bar_type=spy_bar_type,
        rsi_period=params["rsi_period"],
        rsi_threshold_low=params["rsi_threshold_low"],
        rsi_threshold_high=params["rsi_threshold_high"],
        bb_period=params["bb_period"],
        k=params["k"],
        bb_width_threshold=params["bb_width_threshold"],
        atr_period=params["atr_period"],
        sl_atr_multiple=params["sl_atr_multiple"],
        tp_atr_multiple=params["tp_atr_multiple"],
        position_size=params["position_size"],
    )

    profit = run_backtest(bbrsi_config)
    print("Profit for these parameters:", profit)
    results.append({**params, "profit": profit})

df_results = pd.DataFrame(results)
print("\nFurther refined grid search results:")
print(df_results)

top_configs = df_results.nlargest(50, "profit")
print("\nTop configurations from further refined grid search:")
with pd.option_context('display.max_columns', None, 'display.width', None):
    print(top_configs)
