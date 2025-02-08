from nautilus_trader.model.instruments import Instrument
import databento as db
import pandas as pd
from pathlib import Path
import shutil
from typing import List, Tuple
from nautilus_trader.model.data import BarType
from nautilus_trader.persistence.catalog import ParquetDataCatalog
from nautilus_trader.test_kit.providers import TestInstrumentProvider
from nautilus_trader.persistence.wranglers import BarDataWrangler
from nautilus_trader.adapters.databento import DatabentoDataLoader
import os


# Constants
DATABENTO_KEY = ""  # Add your key here
DATA_DIR = Path("data")
CATALOG_DIR = Path("catalogs")

class DataPipeline:
    def __init__(self):
        api_key = os.environ.get('DATABENTO_API_KEY')
        if not api_key:
            raise ValueError("DATABENTO_API_KEY environment variable is not set")
        self.client = db.Historical(api_key)
        self.ensure_directories()

    def ensure_directories(self):
        """Create necessary directories if they don't exist"""
        DATA_DIR.mkdir(exist_ok=True)
        CATALOG_DIR.mkdir(exist_ok=True)

    def download_data(
        self,
        symbol: str,
        dataset: str,
        start: str,
        end: str,
    ) -> Path:
        """Download data from Databento if not already present"""
        path = DATA_DIR / f"{symbol.lower()}-{dataset.lower()}-{start[:6]}-{end[:6]}.trades.dbn.zst"

        if not path.exists():
            # Get cost estimate
            cost = self.client.metadata.get_cost(
                dataset=dataset,
                symbols=[symbol],
                schema="trades",
                start=start,
                end=end,
            )
            print(f"Estimated cost for {symbol}: ${cost:.2f}")

            # Download data
            self.client.timeseries.get_range(
                dataset=dataset,
                symbols=[symbol],
                schema="trades",
                start=start,
                end=end,
                path=path,
            )
            print(f"Downloaded {symbol} data to {path}")
        else:
            print(f"Found existing {symbol} data at {path}")

        return path

    def setup_instrument(self, symbol: str, venue: str) -> Tuple[Instrument, BarType]:
        """Setup instrument and bar type"""
        if symbol in ["SPY", "UVXY"]:
            instrument = TestInstrumentProvider.equity(symbol=symbol, venue=venue)
        else:
            raise ValueError(f"Unsupported instrument: {symbol}")

        bar_type = BarType.from_str(f"{symbol}.{venue}-30-MINUTE-LAST-EXTERNAL")
        return instrument, bar_type

    def create_catalog(self, symbol: str) -> ParquetDataCatalog:
        """Create fresh catalog for instrument"""
        catalog_path = CATALOG_DIR / symbol.lower() / "30min"
        if catalog_path.exists():
            shutil.rmtree(catalog_path)
        catalog_path.mkdir(parents=True)
        return ParquetDataCatalog(catalog_path)

    def process_dbn_to_bars(
        self,
        dbn_path: Path,
        instrument,
        bar_type: BarType,
        timeframe: str
    ) -> List:
        """Process DBN file to bars"""
        loader = DatabentoDataLoader()
        # Get the instrument ID using the instrument's ID property
        trades = loader.from_dbn_file(
            path=dbn_path,
            instrument_id=instrument.id,  # Use instrument.id directly
            as_legacy_cython=False,
        )

        # Convert trades to DataFrame
        df = pd.DataFrame([
            {
                'timestamp': trade.ts_event,
                'price': float(trade.price),
                'size': float(trade.size)
            }
            for trade in trades
        ])

        # Create OHLCV bars
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ns')
        df.set_index('timestamp', inplace=True)

        bars_df = df.resample(timeframe).agg({
            'price': ['first', 'max', 'min', 'last'],
            'size': 'sum'
        }).dropna()

        bars_df.columns = ['open', 'high', 'low', 'close', 'volume']

        # Convert to Bar objects using wrangler
        wrangler = BarDataWrangler(bar_type, instrument)
        return wrangler.process(bars_df)

    def process_instrument(
        self,
        symbol: str,
        venue: str,
        dataset: str,
        start: str,
        end: str,
        timeframe: str
    ):
        """Complete pipeline for one instrument"""
        print(f"\nProcessing {symbol}...")

        # Download data
        dbn_path = self.download_data(symbol, dataset, start, end)

        # Setup instrument
        instrument, bar_type = self.setup_instrument(symbol, venue)

        # Create catalog
        catalog = self.create_catalog(symbol)

        # Process data
        bars = self.process_dbn_to_bars(
            dbn_path=dbn_path,
            instrument=instrument,  # Changed from instrument_id
            bar_type=bar_type,
            timeframe=timeframe
        )

        # Sort and write to catalog
        bars = sorted(bars, key=lambda x: x.ts_init)
        catalog.write_data(bars)
        print(f"Written {len(bars)} bars to catalog for {symbol}")

def main():
    pipeline = DataPipeline()

    # Process SPY
    pipeline.process_instrument(
       symbol="SPY",
       venue="XNAS",
       dataset="XNAS.ITCH",
       start="2022-01",
       end="2025-01",
       timeframe="30min"
   )


if __name__ == "__main__":
    main()
