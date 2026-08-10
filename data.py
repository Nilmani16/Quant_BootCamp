import yfinance as yf
import pandas as pd


TICKERS = [
    "HDFCBANK.NS",
    "ICICIBANK.NS",
    "SBIN.NS",
    "AXISBANK.NS",
    "KOTAKBANK.NS"
]

START = "2018-01-01"
END = "2025-01-01"


data = {}

for ticker in TICKERS:

    print(f"Downloading {ticker}...")

    df = yf.download(
        ticker,
        start=START,
        end=END,
        auto_adjust=False,
        progress=False
    )

    # Remove unnecessary MultiIndex
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)

    # Keep OHLCV
    df = df[
        ["Open", "High", "Low", "Close", "Volume"]
    ].dropna()

    # Split
    train = df.loc["2018-01-01":"2023-12-31"]
    test = df.loc["2024-01-01":"2024-12-31"]

    data[ticker] = {
        "train": train,
        "test": test
    }

    print(
        f"Train: {len(train)} rows | "
        f"Test: {len(test)} rows"
    )


print("\nData loading complete.")
