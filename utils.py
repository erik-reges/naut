from nautilus_trader.core.nautilus_pyo3 import Bar, BarType
from nautilus_trader.model.instruments import Instrument
import pandas as pd
from pathlib import Path
from nautilus_trader.persistence.wranglers import BarDataWrangler

def load_trades_to_bars(
    folder_path: str,
    instrument: Instrument,
    bar_type: BarType,
    timeframe: str  # e.g., "1T" for 1-minute bars
) -> list[Bar]:
    # glob all csvs in the folder
    csv_files = Path(folder_path).glob("*.csv")
    all_bars = []

    for file in csv_files:
        # read the trades csv
        df = pd.read_csv(
            file,
            names=["trade_id", "price", "size", "notional", "timestamp", "is_buyer_maker", "is_best_match"],
            header=None
        )

        # convert timestamp to datetime (epoch ns -> datetime)
        df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms")

        # resample to bars (OHLCV)
        bars_df = df.resample(timeframe, on="timestamp").agg({
            "price": ["first", "max", "min", "last"],
            "size": "sum"
        })
        bars_df.columns = ["open", "high", "low", "close", "volume"]

        # wrangle into Bar objects
        wrangler = BarDataWrangler(bar_type, instrument)
        bars = wrangler.process(bars_df)  # assuming `process` is the correct method
        all_bars.extend(bars)

    return all_bars
