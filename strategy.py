import pandas as pd

# PARAMETERS

SHORT_TERM_MA = 20
LONG_TERM_MA = 50

RSI_PERIOD = 14
ROC_PERIOD = 20
ATR_PERIOD = 14

# Risk management parameters
ATR_STOP_MULTIPLIER = 2.0
ATR_TRAILING_MULTIPLIER = 3.0

MAX_POSITION = 1.0          # Maximum 100% of allocated capital
TARGET_ATR_PERCENT = 0.01   # Target 1% daily ATR

# FEATURE ENGINEERING

def calculate_features(df):

    data = df.copy()

    # 1. Moving Averages

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

    # 2. Long-Term MA Slope

    data["Long_Term_MA_Slope"] = (
        data["Long_Term_MA"].pct_change(5)
    )

    # 3. RSI

    price_change = data["Close"].diff()

    gains = price_change.clip(lower=0)
    losses = -price_change.clip(upper=0)

    avg_gain = gains.rolling(RSI_PERIOD).mean()
    avg_loss = losses.rolling(RSI_PERIOD).mean()

    rs = avg_gain / avg_loss

    data["RSI"] = 100 - (100 / (1 + rs))

    # 4. ROC

    data["ROC"] = (
        data["Close"].pct_change(ROC_PERIOD) * 100
    )

    # 5. True Range

    previous_close = data["Close"].shift(1)

    true_range_1 = (
        data["High"] - data["Low"]
    )

    true_range_2 = (
        data["High"] - previous_close
    ).abs()

    true_range_3 = (
        data["Low"] - previous_close
    ).abs()

    data["True_Range"] = pd.concat(
        [
            true_range_1,
            true_range_2,
            true_range_3
        ],
        axis=1
    ).max(axis=1)

    # 6. ATR

    data["ATR"] = (
        data["True_Range"]
        .rolling(ATR_PERIOD)
        .mean()
    )

    # 7. Normalized Volatility

    data["ATR_Percent"] = (
        data["ATR"] / data["Close"]
    )

    return data


# ============================================================  
# LAYER 1 — REGIME DETECTION
# ============================================================

def detect_regime(data):

    data = data.copy()

    data["Regime"] = 0

    # Bullish regime
    bullish = (
        (data["Short_Term_MA"] > data["Long_Term_MA"]) &
        (data["Long_Term_MA_Slope"] > 0) &
        (data["Close"] > data["Long_Term_MA"])
    )

    data.loc[bullish, "Regime"] = 1

    # Bearish regime
    bearish = (
        (data["Short_Term_MA"] < data["Long_Term_MA"]) &
        (data["Long_Term_MA_Slope"] < 0) &
        (data["Close"] < data["Long_Term_MA"])
    )

    data.loc[bearish, "Regime"] = -1

    return data


# ============================================================
# LAYER 2 — MOMENTUM CONFIRMATION
# ============================================================

def calculate_momentum_signal(data):

    data = data.copy()

    data["Momentum_Signal"] = 0

    bullish_momentum = (
        (data["RSI"] > 50) &
        (data["ROC"] > 0)
    )

    bearish_momentum = (
        (data["RSI"] < 50) &
        (data["ROC"] < 0)
    )

    data.loc[
        bullish_momentum,
        "Momentum_Signal"
    ] = 1

    data.loc[
        bearish_momentum,
        "Momentum_Signal"
    ] = -1

    return data


# ============================================================
# LAYER 3 — RISK MANAGEMENT
# ============================================================

def calculate_risk_parameters(data):

    data = data.copy()

    # --------------------------------------------------------
    # 1. ATR Stop-Loss Distance
    # --------------------------------------------------------

    data["Stop_Distance"] = (
        ATR_STOP_MULTIPLIER * data["ATR"]
    )

    # --------------------------------------------------------
    # 2. Long Stop Price
    # --------------------------------------------------------

    data["Long_Stop"] = (
        data["Close"] -
        data["Stop_Distance"]
    )

    # --------------------------------------------------------
    # 3. Short Stop Price
    # --------------------------------------------------------

    data["Short_Stop"] = (
        data["Close"] +
        data["Stop_Distance"]
    )

    # --------------------------------------------------------
    # 4. Trailing Stop Distance
    # --------------------------------------------------------

    data["Trailing_Stop_Distance"] = (
        ATR_TRAILING_MULTIPLIER * data["ATR"]
    )

    # --------------------------------------------------------
    # 5. Volatility-Adjusted Position Size
    # --------------------------------------------------------
    #
    # If ATR% = 2%, position = 100%
    #
    # If ATR% = 4%, position = 50%
    #
    # If ATR% = 1%, raw position = 200%,
    # but MAX_POSITION caps it at 100%.
    # --------------------------------------------------------

    data["Position_Size"] = (
        TARGET_ATR_PERCENT /
        data["ATR_Percent"]
    )

    data["Position_Size"] = (
        data["Position_Size"]
        .clip(upper=MAX_POSITION)
    )

    # Don't allow negative or invalid values
    data["Position_Size"] = (
        data["Position_Size"]
        .clip(lower=0)
    )

    return data

# ============================================================
# LAYER 4 — BREAKOUT + VOLUME CONFIRMATION
# ============================================================

BREAKOUT_PERIOD = 20
VOLUME_PERIOD = 20


def calculate_breakout_volume_signal(data):
    """
    Detect price breakouts and measure volume strength.

    Breakout:
        1  = bullish breakout
        0  = no breakout
       -1  = bearish breakout

    Volume Score:
        0  = below average volume
        1  = 1.0x - 1.5x average volume
        2  = 1.5x - 2.0x average volume
        3  = above 2.0x average volume
    """

    data = data.copy()

    # --------------------------------------------------------
    # 1. Previous N-day High / Low
    # --------------------------------------------------------

    data["Previous_High"] = (
        data["High"]
        .rolling(BREAKOUT_PERIOD)
        .max()
        .shift(1)
    )

    data["Previous_Low"] = (
        data["Low"]
        .rolling(BREAKOUT_PERIOD)
        .min()
        .shift(1)
    )

    # --------------------------------------------------------
    # 2. Breakout Signal
    # --------------------------------------------------------

    data["Breakout_Signal"] = 0

    bullish_breakout = (
        data["Close"] > data["Previous_High"]
    )

    bearish_breakout = (
        data["Close"] < data["Previous_Low"]
    )

    data.loc[
        bullish_breakout,
        "Breakout_Signal"
    ] = 1

    data.loc[
        bearish_breakout,
        "Breakout_Signal"
    ] = -1
    # --------------------------------------------------------
    # 2B. Breakout Strength
    # --------------------------------------------------------

    data["Bullish_Breakout_Strength"] = 0.0
    data["Bearish_Breakout_Strength"] = 0.0

    # Bullish breakout strength
    data.loc[
        bullish_breakout,
        "Bullish_Breakout_Strength"
    ] = (
        (data.loc[bullish_breakout, "Close"]
        - data.loc[bullish_breakout, "Previous_High"])
        / data.loc[bullish_breakout, "ATR"]
    )

    # Bearish breakout strength
    data.loc[
        bearish_breakout,
        "Bearish_Breakout_Strength"
    ] = (
        (data.loc[bearish_breakout, "Previous_Low"]
        - data.loc[bearish_breakout, "Close"])
        / data.loc[bearish_breakout, "ATR"]
    )
    # --------------------------------------------------------
    # 3. Average Volume
    # --------------------------------------------------------

    data["Average_Volume"] = (
        data["Volume"]
        .rolling(VOLUME_PERIOD)
        .mean()
    )

    # --------------------------------------------------------
    # 4. Volume Ratio
    # --------------------------------------------------------

    data["Volume_Ratio"] = (
        data["Volume"] /
        data["Average_Volume"]
    )

    # --------------------------------------------------------
    # 5. Volume Score
    # --------------------------------------------------------

    data["Volume_Score"] = 0

    data.loc[
        (data["Volume_Ratio"] >= 1.0) &
        (data["Volume_Ratio"] < 1.5),
        "Volume_Score"
    ] = 1

    data.loc[
        (data["Volume_Ratio"] >= 1.5) &
        (data["Volume_Ratio"] < 2.0),
        "Volume_Score"
    ] = 2

    data.loc[
        data["Volume_Ratio"] >= 2.0,
        "Volume_Score"
    ] = 3

    return data
# ============================================================
# LAYER 5 — CALCULATE SIGNAL SCORE
# ============================================================
def calculate_signal_score(data):

    data = data.copy()

    # Start with regime
    data["Signal_Score"] = (
        2 * data["Regime"]
    )

    # Momentum contribution
    data["Signal_Score"] += (
        2 * data["Momentum_Signal"]
    )

    # Breakout contribution
    data["Signal_Score"] += (
        2 * data["Breakout_Signal"]
    )

    # Volume contributes only when there is a breakout
    bullish_breakout = data["Breakout_Signal"] == 1
    bearish_breakout = data["Breakout_Signal"] == -1

    data.loc[
        bullish_breakout,
        "Signal_Score"
    ] += data.loc[
        bullish_breakout,
        "Volume_Score"
    ]

    data.loc[
        bearish_breakout,
        "Signal_Score"
    ] -= data.loc[
        bearish_breakout,
        "Volume_Score"
    ]

    return data

# ============================================================
# LAYER 6 — GENERATING TRADE SIGNAL 
# ============================================================

def generate_trade_signal(data):

    data = data.copy()

    data["Trade_Signal"] = 0

    data.loc[
        data["Signal_Score"] >= 5,
        "Trade_Signal"
    ] = 1

    data.loc[
        data["Signal_Score"] <= -5,
        "Trade_Signal"
    ] = -1

    return data

# ============================================================
# COMPLETE STRATEGY PIPELINE
# ============================================================

def build_strategy(df):

    data = calculate_features(df)

    data = detect_regime(data)

    data = calculate_momentum_signal(data)

    data = calculate_risk_parameters(data)

    data = calculate_breakout_volume_signal(data)

    data = calculate_signal_score(data)

    data = generate_trade_signal(data)

    return data
