import yfinance as yf
import pandas as pd


TICKERS = [
    "HDFCBANK.NS",
    "ICICIBANK.NS",
    "SBIN.NS",
    "AXISBANK.NS",
    "KOTAKBANK.NS"
]

BENCHMARK = "^NSEBANK"

START = "2018-01-01"
END = "2025-01-01"


def load_all_data():
    """Download and return all ticker data split into train/test sets."""

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

        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)

        df = df[
            ["Open", "High", "Low", "Close", "Volume"]
        ].dropna()

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

    return data

def load_benchmark_data():
    """Download NIFTY Bank benchmark data."""

    print(f"Downloading benchmark {BENCHMARK}...")

    df = yf.download(
        BENCHMARK,
        start=START,
        end=END,
        auto_adjust=False,
        progress=False
    )

    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)

    df = df[["Open", "High", "Low", "Close", "Volume"]].dropna()

    train = df.loc["2018-01-01":"2023-12-31"]
    test = df.loc["2024-01-01":"2024-12-31"]

    print(
        f"NIFTY Bank | Train: {len(train)} rows | "
        f"Test: {len(test)} rows"
    )

    return {
        "train": train,
        "test": test
    }

    return data
