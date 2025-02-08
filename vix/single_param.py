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

from strategy import ESVIXStrategy, ESVIXStrategyConfig

def run_backtest(config: ESVIXStrategyConfig) -> float:
    # Configure the backtest engine.
    engine_config = BacktestEngineConfig(
        trader_id=TraderId("BACKTESTER-ESVIX"),
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
        starting_balances=[Money(1_000_000.0, USD)],
    )

    # Register instruments: SPY and UVXY.
    SPY = TestInstrumentProvider.equity(symbol="SPY", venue="XNAS")
    UVXY = TestInstrumentProvider.equity(symbol="UVXY", venue="XNAS")
    engine.add_instrument(SPY)
    engine.add_instrument(UVXY)

    # Create bar types using the registered instruments' IDs.
    spy_bar_type = BarType.from_str(f"{SPY.id}-15-MINUTE-LAST-EXTERNAL")
    uvxy_bar_type = BarType.from_str(f"{UVXY.id}-15-MINUTE-LAST-EXTERNAL")

    # Load bar data from local catalogs.
    spy_catalog = ParquetDataCatalog("../catalogs/spy/15min")
    uvxy_catalog = ParquetDataCatalog("../catalogs/uvxy/15min")
    spy_bars = spy_catalog.bars([str(spy_bar_type)])
    uvxy_bars = uvxy_catalog.bars([str(uvxy_bar_type)])
    engine.add_data(spy_bars + uvxy_bars)

    # Instantiate the strategy with the given configuration.
    strategy = ESVIXStrategy(config=config)
    engine.add_strategy(strategy)

    # Run the backtest.
    engine.run()
    result = engine.get_result()
    pnl = result.stats_pnls['USD']['PnL (total)']
    engine.reset()
    engine.dispose()
    return pnl

# Baseline configuration (based on your top results)
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

# Focus on one parameter at a time.
# For example, vary 'ema_period' over a small range.
param_to_vary = "macd_signal"
values = [2,9, 10,12, 13, 14, 15, 16, 34]

results = []
for value in values:
    # Create a configuration copy and update the parameter to vary.
    config_params = baseline_config.copy()
    config_params[param_to_vary] = value
    print(f"Running backtest with {param_to_vary} = {value}")
    config = ESVIXStrategyConfig(
        instrument_id=TestInstrumentProvider.equity(symbol="SPY", venue="XNAS").id,
        bar_type=BarType.from_str(f"{TestInstrumentProvider.equity(symbol='SPY', venue='XNAS').id}-15-MINUTE-LAST-EXTERNAL"),
        vix_bar_type=BarType.from_str(f"{TestInstrumentProvider.equity(symbol='UVXY', venue='XNAS').id}-15-MINUTE-LAST-EXTERNAL"),
        ema_period=config_params["ema_period"],
        macd_fast=config_params["macd_fast"],
        macd_slow=config_params["macd_slow"],
        macd_signal=config_params["macd_signal"],
        rsi_period=config_params["rsi_period"],
        vix_ma_period=20,  # fixed value for this analysis
        max_risk_pct=config_params["max_risk_pct"],
        trailing_multiplier=config_params["trailing_multiplier"],
        trailing_atr_period=14,
        bars_required=50,
        bars_threshold=config_params["bars_threshold"],
        profit_target_atr_multiple=config_params["profit_target_atr_multiple"],
        stop_loss_atr_multiple=config_params["stop_loss_atr_multiple"],
    )
    pnl = run_backtest(config)
    print(f"ema_period {value}: PnL = {pnl}")
    results.append({param_to_vary: value, "pnl": pnl})

df_results = pd.DataFrame(results)
print("One-Parameter Sensitivity Analysis Results:")
print(df_results)
