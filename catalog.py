from pathlib import Path
import shutil
from nautilus_trader.model.data import BarType
from nautilus_trader.persistence.catalog import ParquetDataCatalog
from nautilus_trader.test_kit.providers import TestInstrumentProvider
from utils import load_trades_to_bars

DATABENTO_DATA_DIR = Path("databento")
CATALOG_PATH = Path("catalogs/ETHUSDT/1h")  # Update catalog path if needed

# Clear if it already exists
if CATALOG_PATH.exists():
    shutil.rmtree(CATALOG_PATH)
CATALOG_PATH.mkdir()

# Create a new catalog instance for 1-hour bars
catalog = ParquetDataCatalog(CATALOG_PATH)

instrument = TestInstrumentProvider.ethusdt_binance()

# Update the bar type to represent 1-hour bars
eth_bar_type = BarType.from_str(f"{instrument.id}-30-MINUTE-LAST-EXTERNAL")

bars = load_trades_to_bars(
    folder_path="data",
    instrument=instrument,
    bar_type=eth_bar_type,
    timeframe="30min"  # or "1h" if that’s what your loader expects
)
bars = sorted(bars, key=lambda x: x.ts_init)
catalog.write_data(bars)




# instrument_id = InstrumentId.from_str("SPY.XNAS")
# path = DATABENTO_DATA_DIR / "spy-xnas-202401-202403.trades.dbn.zst"
# # Decode data to pyo3 objects
# loader = DatabentoDataLoader()
# trades = loader.from_dbn_file(
#     path=path,
#     instrument_id=instrument_id,
#     as_legacy_cython=False,
# )

# # Write data
# 15min.write_data(trades)
# # Test reading from 15min
# depths = 15min.order_book_depth10()
# print(len(depths))
