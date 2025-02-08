import pandas as pd
from sklearn.model_selection import ParameterGrid

# Import necessary NautilusTrader modules and your strategy classes.
from nautilus_trader.backtest.engine import BacktestEngine, BacktestEngineConfig
from nautilus_trader.config import LoggingConfig, RiskEngineConfig
from nautilus_trader.model.currencies import USDT
from nautilus_trader.model.data import BarType
from nautilus_trader.model.enums import AccountType, OmsType
from nautilus_trader.model.identifiers import TraderId, Venue
from nautilus_trader.model.objects import Money
from nautilus_trader.persistence.catalog.parquet import ParquetDataCatalog
from nautilus_trader.test_kit.providers import TestInstrumentProvider

from strat import KNNBasedStrategy, KNNBasedStrategyConfig

instrument = TestInstrumentProvider.ethusdt_binance()
knn_bar_type = BarType.from_str(f"{instrument.id}-30-MINUTE-LAST-EXTERNAL")


# --- Define a function that creates and runs a backtest given a configuration ---
def run_backtest(knn_config: KNNBasedStrategyConfig) -> float:
    # --- Create Engine Setup ---
    engine_config = BacktestEngineConfig(
        trader_id=TraderId("BACKTESTER-KNN"),
        logging=LoggingConfig(log_level="INFO", print_config=True),
        risk_engine=RiskEngineConfig(bypass=True),
    )
    engine = BacktestEngine(config=engine_config)

    # Define the venue and add it to the engine.
    BINANCE = Venue("BINANCE")
    engine.add_venue(
        venue=BINANCE,
        oms_type=OmsType.NETTING,
        account_type=AccountType.CASH,
        base_currency=None,
        starting_balances=[Money(1_000_000, USDT)]
    )

    # --- Register Instrument ---
    engine.add_instrument(instrument)

    # --- Load Bar Data from Parquet ---
    catalog = ParquetDataCatalog("../catalogs/ETHUSDT/30min")  # Adjust this path as needed
    bars = catalog.bars([str(knn_bar_type)])
    engine.add_data(bars)

    # --- Create Strategy Instance ---
    strategy = KNNBasedStrategy(config=knn_config)
    engine.add_strategies([strategy])

    # --- Run Backtest ---
    engine.run()

    # --- Generate Report ---
    report_df = engine.trader.generate_account_report(BINANCE)
    # Sort by date to ensure we get the final state
    report_df = report_df.sort_index()
    # Filter for USDT rows (assuming USDT is your base currency)
    usdt_report = report_df[report_df['currency'] == 'USDT']
    if not usdt_report.empty:
        final_value = float(usdt_report.iloc[-1]['free'])
    else:
        final_value = 1_000_000  # default if no report is available

    starting_balance = 1_000_000
    profit = final_value - starting_balance

    # --- Cleanup ---
    engine.reset()
    engine.dispose()

    return profit

# --- Define the grid of parameters to search over ---
param_grid = {
    'short_window': [5, 7, 10, 12],
    'long_window': [10, 14, 20, 25, 30],
    'base_k': [150, 169, 200, 225, 252],
    'trailing_multiplier': [1.0, 1.5, 2.0, 2.5]
}

grid = list(ParameterGrid(param_grid))
results = []

# --- Run grid search ---
for params in grid:
    print("Running backtest with parameters:", params)
    knn_config = KNNBasedStrategyConfig(
        instrument_id=TestInstrumentProvider.ethusdt_binance().id,
        bar_type=knn_bar_type,
        start_date="2024-07-01T00:00:00Z",
        stop_date="2024-10-31T23:45:00Z",
        indicator="ALL",  # Use multiple indicators for robustness
        short_window=params['short_window'],
        long_window=params['long_window'],
        base_k=params['base_k'],
        volatility_filter=True,
        bars_required=30,
        bars_threshold=50,
        max_risk_pct=0.02,
        margin_factor=0.2,
        trailing_multiplier=params['trailing_multiplier'],
        trailing_atr_period=14,
        trailing_min_profit_multiple=1.5,
        use_trend_filter=True,
        use_volume_filter=True,
        sma_period=50,
        volume_ma_period=50
    )

    profit = run_backtest(knn_config)
    print("Profit for these parameters:", profit)
    results.append({**params, 'profit': profit})

# --- Analyze the results ---
df_results = pd.DataFrame(results)
print("Grid search results:")
print(df_results)

best_config = df_results.loc[df_results['profit'].idxmax()]
print("Best configuration found:")
print(best_config)
