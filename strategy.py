import pandas as pd

# PARAMETERS

SHORT_TERM_MA = 20
LONG_TERM_MA = 50

# REGIME DETECTION

def detect_regime(df):
    """
    Detect market regime using moving averages.

    Regimes:
        1  = Bullish
        0  = Neutral
       -1  = Bearish
    """

    data = df.copy()

    # 1. Calculate Short-Term and Long-Term Moving Averages

    data["Short_Term_MA"] = (
        data["Close"]
        .rolling(SHORT_TERM_MA)
        .mean()
    )

    data["Long_Term_MA"] = (
        data["Close"]
        .rolling(LONG_TERM_MA)
        .mean()
    )

    # 2. Calculate Long-Term MA Slope

    data["Long_Term_MA_Slope"] = (
        data["Long_Term_MA"]
        .diff()
    )

    # 3. Start with Neutral Regime

    data["Regime"] = 0

    # 4. Bullish Regime

    bullish = (
        (data["Short_Term_MA"] > data["Long_Term_MA"]) &
        (data["Long_Term_MA_Slope"] > 0) &
        (data["Close"] > data["Long_Term_MA"])
    )

    data.loc[bullish, "Regime"] = 1

    # 5. Bearish Regime

    bearish = (
        (data["Short_Term_MA"] < data["Long_Term_MA"]) &
        (data["Long_Term_MA_Slope"] < 0) &
        (data["Close"] < data["Long_Term_MA"])
    )

    data.loc[bearish, "Regime"] = -1

    return data
