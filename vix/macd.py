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

from strategy import ESVIXStrategy, ESVIXStrategyConfig

def run_backtest(config: ESVIXStrategyConfig) -> float:
    # Configure engine.
    engine_config = BacktestEngineConfig(
        trader_id=TraderId("BACKTESTER-ESVIX"),
        logging=LoggingConfig(log_level="INFO", print_config=False),
        risk_engine=RiskEngineConfig(bypass=True),
    )
    engine = BacktestEngine(config=engine_config)

    # Set up venue.
    VENUE = Venue("XNAS")
    engine.add_venue(
        venue=VENUE,
        oms_type=OmsType.NETTING,
        account_type=AccountType.CASH,
        base_currency=USD,
        starting_balances=[Money(1_000_000.0, USD)],
    )

    # Register instruments: SPY and UVXY.
    SPY = TestInstrumentProvider.equity(symbol="SPY", venue="XNAS")
    UVXY = TestInstrumentProvider.equity(symbol="UVXY", venue="XNAS")
    engine.add_instrument(SPY)
    engine.add_instrument(UVXY)

    # Create bar types.
    spy_bar_type = BarType.from_str(f"{SPY.id}-15-MINUTE-LAST-EXTERNAL")
    uvxy_bar_type = BarType.from_str(f"{UVXY.id}-15-MINUTE-LAST-EXTERNAL")

    # Load data from local catalogs.
    spy_catalog = ParquetDataCatalog("../catalogs/spy/15min")
    uvxy_catalog = ParquetDataCatalog("../catalogs/uvxy/15min")
    spy_bars = spy_catalog.bars([str(spy_bar_type)])
    uvxy_bars = uvxy_catalog.bars([str(uvxy_bar_type)])
    engine.add_data(spy_bars + uvxy_bars)

    # Instantiate strategy with the provided configuration.
    strategy = ESVIXStrategy(config=config)
    engine.add_strategy(strategy)

    # Run the backtest.
    engine.run()
    result = engine.get_result()
    # Extract total PnL (for USD).
    pnl = result.stats_pnls['USD']['PnL (total)']
    engine.reset()
    engine.dispose()
    return pnl
# Define baseline configuration
baseline_config = {
    "ema_period": 14,
    "macd_fast": 7,
    "macd_slow": 14,
    "macd_signal": 10,
    "rsi_period": 14,
    "max_risk_pct": 0.05,
    "trailing_multiplier": 2.5,
    "profit_target_atr_multiple": 2.0,
    "stop_loss_atr_multiple": 1.0,
    "bars_threshold": 500,
}

# Define the grid of parameters focusing only on MACD
param_grid = {
    "macd_fast": [7, 12],
    "macd_slow": [ 14,24],
    "macd_signal": [9, 15, 27],
}

grid = list(ParameterGrid(param_grid))
results = []

for params in grid:
    print("Running backtest with MACD parameters:", params)

    # Combine baseline config with MACD parameters
    config = ESVIXStrategyConfig(
        instrument_id=TestInstrumentProvider.equity(symbol="SPY", venue="XNAS").id,
        bar_type=BarType.from_str(f"{TestInstrumentProvider.equity(symbol='SPY', venue='XNAS').id}-15-MINUTE-LAST-EXTERNAL"),
        vix_bar_type=BarType.from_str(f"{TestInstrumentProvider.equity(symbol='UVXY', venue='XNAS').id}-15-MINUTE-LAST-EXTERNAL"),
        ema_period=baseline_config["ema_period"],
        macd_fast=params["macd_fast"],
        macd_slow=params["macd_slow"],
        macd_signal=params["macd_signal"],
        rsi_period=baseline_config["rsi_period"],
        vix_ma_period=20,  # fixed
        max_risk_pct=baseline_config["max_risk_pct"],
        trailing_multiplier=baseline_config["trailing_multiplier"],
        trailing_atr_period=14,
        bars_required=50,
        bars_threshold=baseline_config["bars_threshold"],
        profit_target_atr_multiple=baseline_config["profit_target_atr_multiple"],
        stop_loss_atr_multiple=baseline_config["stop_loss_atr_multiple"],
    )
    pnl = run_backtest(config)
    print("PnL for these parameters:", pnl)
    results.append({**params, "pnl": pnl})

df_results = pd.DataFrame(results)
print("\nGrid search results:")
print(df_results)

# Sort and show the top 5 configurations
top_configs = df_results.nlargest(5, "pnl")
print("\nTop 5 MACD configurations found:")
with pd.option_context('display.max_columns', None, 'display.width', None):
    print(top_configs)

# Save results to CSV
df_results.to_csv('macd_gridsearch_results.csv', index=False)
print("\nResults saved to 'macd_gridsearch_results.csv'")
