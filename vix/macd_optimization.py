# macd_optimization.py

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
    """Run a single backtest with the given configuration."""
    engine_config = BacktestEngineConfig(
        trader_id=TraderId("BACKTESTER-ESVIX"),
        logging=LoggingConfig(log_level="INFO", print_config=False),
        risk_engine=RiskEngineConfig(bypass=True),
    )
    engine = BacktestEngine(config=engine_config)

    # Set up venue
    VENUE = Venue("XNAS")
    engine.add_venue(
        venue=VENUE,
        oms_type=OmsType.NETTING,
        account_type=AccountType.CASH,
        base_currency=USD,
        starting_balances=[Money(1_000_000.0, USD)],
    )

    # Register instruments
    SPY = TestInstrumentProvider.equity(symbol="SPY", venue="XNAS")
    UVXY = TestInstrumentProvider.equity(symbol="UVXY", venue="XNAS")
    engine.add_instrument(SPY)
    engine.add_instrument(UVXY)

    # Create bar types
    spy_bar_type = BarType.from_str(f"{SPY.id}-15-MINUTE-LAST-EXTERNAL")
    uvxy_bar_type = BarType.from_str(f"{UVXY.id}-15-MINUTE-LAST-EXTERNAL")

    # Load data
    spy_catalog = ParquetDataCatalog("../catalogs/spy/15min")
    uvxy_catalog = ParquetDataCatalog("../catalogs/uvxy/15min")
    spy_bars = spy_catalog.bars([str(spy_bar_type)])
    uvxy_bars = uvxy_catalog.bars([str(uvxy_bar_type)])
    engine.add_data(spy_bars + uvxy_bars)

    strategy = ESVIXStrategy(config=config)
    engine.add_strategy(strategy)

    engine.run()
    result = engine.get_result()
    pnl = result.stats_pnls['USD']['PnL (total)']
    engine.reset()
    engine.dispose()
    return pnl

def create_config(**kwargs) -> ESVIXStrategyConfig:
    """Create strategy configuration with baseline values and optional overrides."""
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
        "vix_ma_period": 20,
        "trailing_atr_period": 14,
        "bars_required": 50,
    }

    # Update baseline with provided kwargs
    config_params = {**baseline_config, **kwargs}

    return ESVIXStrategyConfig(
        instrument_id=TestInstrumentProvider.equity(symbol="SPY", venue="XNAS").id,
        bar_type=BarType.from_str(f"{TestInstrumentProvider.equity(symbol='SPY', venue='XNAS').id}-15-MINUTE-LAST-EXTERNAL"),
        vix_bar_type=BarType.from_str(f"{TestInstrumentProvider.equity(symbol='UVXY', venue='XNAS').id}-15-MINUTE-LAST-EXTERNAL"),
        **config_params
    )

def optimize_macd_parameters():
    """Perform MACD parameter optimization with paired fast/slow testing."""
    results = {}

    # Step 1: Optimize MACD fast/slow pairs
    print("\n=== Optimizing MACD Fast/Slow Pairs ===")

    # Define pairs of fast/slow periods to test
    period_pairs = [
        (5, 14),  # Traditional short
        (7, 14),
        (9, 14),
        (12, 14),
        (5, 21),  # Medium term
        (7, 21),
        (9, 21),
        (12, 21),
        (5, 26),  # Longer term
        (7, 26),
        (9, 26),
        (12, 26),
    ]

    pair_results = []

    for fast, slow in period_pairs:
        print(f"\nTesting MACD pair - Fast: {fast}, Slow: {slow}")
        config = create_config(
            macd_fast=fast,
            macd_slow=slow,
            macd_signal=10  # Keep signal constant initially
        )
        pnl = run_backtest(config)
        pair_results.append({
            "macd_fast": fast,
            "macd_slow": slow,
            "spread": slow - fast,  # Add spread information
            "pnl": pnl
        })
        print(f"PnL: {pnl}")

    pair_df = pd.DataFrame(pair_results)
    print("\nFast/Slow pair results:")
    print(pair_df.sort_values('pnl', ascending=False))

    # Get best pair
    best_pair = pair_df.nlargest(1, "pnl").iloc[0]
    best_fast = int(best_pair["macd_fast"])
    best_slow = int(best_pair["macd_slow"])
    results["pair_optimization"] = pair_df

    print(f"\nBest MACD pair - Fast: {best_fast}, Slow: {best_slow}")

    # Step 2: Optimize signal period using best fast/slow pair
    print("\n=== Optimizing MACD Signal Period ===")
    signal_periods = [7, 9, 10, 12, 14, 16]
    signal_results = []

    for signal in signal_periods:
        print(f"\nTesting MACD signal period: {signal}")
        config = create_config(
            macd_fast=best_fast,
            macd_slow=best_slow,
            macd_signal=signal
        )
        pnl = run_backtest(config)
        signal_results.append({
            "macd_signal": signal,
            "pnl": pnl
        })
        print(f"PnL: {pnl}")

    signal_df = pd.DataFrame(signal_results)
    best_signal = signal_df.nlargest(1, "pnl")["macd_signal"].iloc[0]
    results["signal_optimization"] = signal_df

    # Final optimization summary
    print("\n=== MACD Optimization Results ===")
    print(f"Best MACD Fast Period: {best_fast}")
    print(f"Best MACD Slow Period: {best_slow}")
    print(f"Best MACD Signal Period: {best_signal}")
    print(f"Best Spread: {best_slow - best_fast}")

    # Save results to CSV
    for name, df in results.items():
        df.to_csv(f'macd_{name}_results.csv', index=False)
        print(f"\n{name} results:")
        print(df.sort_values('pnl', ascending=False))

    return {
        "best_fast": best_fast,
        "best_slow": best_slow,
        "best_signal": best_signal,
        "results": results
    }

def __main__():
  optimize_macd_parameters()

if __name__ == '__main__':
    __main__()