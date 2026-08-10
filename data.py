import yfinance as yf
import pandas as pd

# CONFIGURATION 

TICKERS = [
    "HDFCBANK.NS",
    "ICICIBANK.NS",
    "SBIN.NS",
    "AXISBANK.NS",
    "KOTAKBANK.NS"
]

START_DATE = "2018-01-01"
END_DATE = "2025-01-01"  

TRAIN_START = "2018-01-01"
TRAIN_END = "2023-12-31"

TEST_START = "2024-01-01"
TEST_END = "2024-12-31"

# DOWNLOAD DATA

def download_data(ticker):
    # Download daily OHLCV data for one stock.

    df = yf.download(
        ticker,
        start=START_DATE,
        end=END_DATE,
        interval="1d",
        auto_adjust=False,
        progress=False
    )

    if df.empty:
        raise ValueError(f"No data downloaded for {ticker}")

    # Handle yfinance MultiIndex columns
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)

    # Keep only required OHLCV columns
    required_columns = [
        "Open",
        "High",
        "Low",
        "Close",
        "Volume"
    ]

    df = df[required_columns].copy()

    # Remove rows containing missing values
    df.dropna(inplace=True)

    # Make sure index is datetime
    df.index = pd.to_datetime(df.index)

    return df

# TRAIN / TEST SPLIT

def split_data(df):
    #Split data into training and testing periods.

    train = df.loc[
        TRAIN_START:TRAIN_END
    ].copy()

    test = df.loc[
        TEST_START:TEST_END
    ].copy()

    return train, test

# LOAD ALL STOCKS

def load_all_data():
    """
    Download and split data for all five stocks.

    Returns
    -------
    data : dict
        {
            ticker: {
                "full": full_data,
                "train": training_data,
                "test": testing_data
            }
        }
    """

    data = {}

    for ticker in TICKERS:

        print(f"Downloading {ticker}...")

        df = download_data(ticker)

        train, test = split_data(df)

        data[ticker] = {
            "full": df,
            "train": train,
            "test": test
        }

        print(
            f"{ticker}: "
            f"{len(train)} training rows | "
            f"{len(test)} testing rows"
        )

    return data

# BASIC VALIDATION

def validate_data(data):
    """
    Perform basic checks before passing data
    to the strategy/backtester.
    """

    for ticker, datasets in data.items():

        train = datasets["train"]
        test = datasets["test"]

        # Check training period
        assert train.index.min() >= pd.Timestamp(TRAIN_START)
        assert train.index.max() <= pd.Timestamp(TRAIN_END)

        # Check testing period
        assert test.index.min() >= pd.Timestamp(TEST_START)
        assert test.index.max() <= pd.Timestamp(TEST_END)

        # Check required columns
        required = ["Open", "High", "Low", "Close", "Volume"]

        assert all(col in train.columns for col in required)
        assert all(col in test.columns for col in required)
        
        # Check missing values
        assert not train[required].isnull().any().any()
        assert not test[required].isnull().any().any()

        print(f"{ticker}: validation passed")

# MAIN

if __name__ == "__main__":

    data = load_all_data()

    validate_data(data)

    print("\nData loading completed successfully.")
