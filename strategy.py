# ============================================================
# INTER-IIT QUANT BOOTCAMP
# STRATEGY V2
# Adaptive Multi-Factor Momentum + Breakout
# ============================================================

import numpy as np
import pandas as pd


# ============================================================
# 1. PARAMETERS
# ============================================================

SHORT_MA = 20
LONG_MA = 50

RSI_PERIOD = 14
ROC_PERIOD = 10
ATR_PERIOD = 14

BREAKOUT_PERIOD = 10
VOLUME_PERIOD = 20

# Minimum directional score
LONG_ENTRY_SCORE = 4
SHORT_ENTRY_SCORE = 4

# Minimum separation between bullish and bearish scores
MIN_SCORE_EDGE = 2

# Risk management
ATR_STOP_MULTIPLIER = 2.5
ATR_TARGET_MULTIPLIER = 5.0
ATR_TRAILING_MULTIPLIER = 3.0

# Position sizing
MIN_POSITION = 0.25
MAX_POSITION = 0.60

TARGET_ATR_PERCENT = 0.01


# ============================================================
# 2. FEATURE ENGINEERING
# ============================================================

def calculate_features(df):

    data = df.copy()

    # --------------------------------------------------------
    # MOVING AVERAGES
    # --------------------------------------------------------

    data["Short_MA"] = (
        data["Close"]
        .rolling(SHORT_MA)
        .mean()
    )

    data["Long_MA"] = (
        data["Close"]
        .rolling(LONG_MA)
        .mean()
    )

    # Keep trend information, but DO NOT create
    # a hard Regime gate.
    data["Long_MA_Slope"] = (
        data["Long_MA"]
        .pct_change(5)
    )

    # --------------------------------------------------------
    # RSI
    # --------------------------------------------------------

    change = data["Close"].diff()

    gain = change.clip(lower=0)
    loss = -change.clip(upper=0)

    avg_gain = (
        gain
        .rolling(RSI_PERIOD)
        .mean()
    )

    avg_loss = (
        loss
        .rolling(RSI_PERIOD)
        .mean()
    )

    rs = avg_gain / avg_loss

    data["RSI"] = (
        100 -
        (100 / (1 + rs))
    )

    # --------------------------------------------------------
    # ROC
    # --------------------------------------------------------

    data["ROC"] = (
        data["Close"]
        .pct_change(ROC_PERIOD)
        * 100
    )

    # --------------------------------------------------------
    # ATR
    # --------------------------------------------------------

    previous_close = (
        data["Close"]
        .shift(1)
    )

    tr1 = (
        data["High"] -
        data["Low"]
    )

    tr2 = (
        data["High"] -
        previous_close
    ).abs()

    tr3 = (
        data["Low"] -
        previous_close
    ).abs()

    data["True_Range"] = pd.concat(
        [tr1, tr2, tr3],
        axis=1
    ).max(axis=1)

    data["ATR"] = (
        data["True_Range"]
        .rolling(ATR_PERIOD)
        .mean()
    )

    data["ATR_Percent"] = (
        data["ATR"] /
        data["Close"]
    )

    # --------------------------------------------------------
    # BREAKOUT
    # --------------------------------------------------------

    # IMPORTANT:
    # shift(1) means today's price is compared only against
    # the PREVIOUS completed breakout window.
    #
    # No look-ahead bias.

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

    data["Breakout_Signal"] = 0

    bullish_breakout = (
        data["Close"] >
        data["Previous_High"]
    )

    bearish_breakout = (
        data["Close"] <
        data["Previous_Low"]
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
    # BREAKOUT STRENGTH
    # --------------------------------------------------------

    data["Bullish_Breakout_Strength"] = 0.0
    data["Bearish_Breakout_Strength"] = 0.0

    valid_atr = (
        data["ATR"] > 0
    )

    bullish_valid = (
        bullish_breakout &
        valid_atr
    )

    bearish_valid = (
        bearish_breakout &
        valid_atr
    )

    data.loc[
        bullish_valid,
        "Bullish_Breakout_Strength"
    ] = (
        (
            data.loc[
                bullish_valid,
                "Close"
            ]
            -
            data.loc[
                bullish_valid,
                "Previous_High"
            ]
        )
        /
        data.loc[
            bullish_valid,
            "ATR"
        ]
    )

    data.loc[
        bearish_valid,
        "Bearish_Breakout_Strength"
    ] = (
        (
            data.loc[
                bearish_valid,
                "Previous_Low"
            ]
            -
            data.loc[
                bearish_valid,
                "Close"
            ]
        )
        /
        data.loc[
            bearish_valid,
            "ATR"
        ]
    )

    # --------------------------------------------------------
    # VOLUME
    # --------------------------------------------------------

    data["Average_Volume"] = (
        data["Volume"]
        .rolling(VOLUME_PERIOD)
        .mean()
    )

    data["Volume_Ratio"] = (
        data["Volume"] /
        data["Average_Volume"]
    )

    # Volume is NOT part of directional Long/Short score.
    #
    # It is treated as market-activity confirmation.
    #
    # < 1.0x  -> 0
    # 1.0x+   -> 1
    # 1.5x+   -> 2
    # 2.0x+   -> 3

    data["Volume_Score"] = 0

    data.loc[
        data["Volume_Ratio"] >= 1.0,
        "Volume_Score"
    ] = 1

    data.loc[
        data["Volume_Ratio"] >= 1.5,
        "Volume_Score"
    ] = 2

    data.loc[
        data["Volume_Ratio"] >= 2.0,
        "Volume_Score"
    ] = 3

    return data


# ============================================================
# 3. DIRECTIONAL SCORING
# ============================================================

def calculate_scores(data):

    data = data.copy()

    long_score = pd.Series(
        0.0,
        index=data.index
    )

    short_score = pd.Series(
        0.0,
        index=data.index
    )

    # ========================================================
    # LONG SCORE
    # ========================================================

    # 1. Price above long-term MA
    long_score += (
        data["Close"] >
        data["Long_MA"]
    ).astype(int)

    # 2. Short MA above long MA
    long_score += (
        data["Short_MA"] >
        data["Long_MA"]
    ).astype(int)

    # 3. Positive long MA slope
    long_score += (
        data["Long_MA_Slope"] > 0
    ).astype(int)

    # 4. RSI bullish
    long_score += (
        data["RSI"] > 50
    ).astype(int)

    # 5. ROC bullish
    long_score += (
        data["ROC"] > 0
    ).astype(int)

    # 6. Bullish breakout
    long_score += (
        data["Breakout_Signal"] == 1
    ).astype(int)

    # IMPORTANT:
    # NO VOLUME SCORE HERE.
    #
    # Volume does not tell us direction.

    # ========================================================
    # SHORT SCORE
    # ========================================================

    # 1. Price below long-term MA
    short_score += (
        data["Close"] <
        data["Long_MA"]
    ).astype(int)

    # 2. Short MA below long MA
    short_score += (
        data["Short_MA"] <
        data["Long_MA"]
    ).astype(int)

    # 3. Negative long MA slope
    short_score += (
        data["Long_MA_Slope"] < 0
    ).astype(int)

    # 4. RSI bearish
    short_score += (
        data["RSI"] < 50
    ).astype(int)

    # 5. ROC bearish
    short_score += (
        data["ROC"] < 0
    ).astype(int)

    # 6. Bearish breakout
    short_score += (
        data["Breakout_Signal"] == -1
    ).astype(int)

    # NO VOLUME HERE

    # --------------------------------------------------------
    # Store scores
    # --------------------------------------------------------

    data["Long_Score"] = long_score
    data["Short_Score"] = short_score

    data["Signal_Score"] = (
        long_score -
        short_score
    )

    return data


# ============================================================
# 4. TRADE SIGNAL
# ============================================================

def generate_signal(data):

    data = data.copy()

    data["Trade_Signal"] = 0

    # --------------------------------------------------------
    # LONG
    # --------------------------------------------------------

    long_condition = (
        (data["Long_Score"] >= LONG_ENTRY_SCORE)
        &
        (data["Long_Score"] - data["Short_Score"]>= MIN_SCORE_EDGE)
        
    )

    # --------------------------------------------------------
    # SHORT
    # --------------------------------------------------------

    # short_condition = (
    #     (data["Short_Score"] >= SHORT_ENTRY_SCORE)
    #     &
    #     (
    #         (
    #             data["Short_Score"] -
    #             data["Long_Score"]>=MIN_SCORE_EDGE
    #         )
    #     )
    # )

    data.loc[
        long_condition,
        "Trade_Signal"
    ] = 1

    # data.loc[
    #     short_condition,
    #     "Trade_Signal"
    # ] = -1

    return data


# ============================================================
# 5. POSITION SIZING
# ============================================================

def calculate_position_size(data):

    data = data.copy()

    # --------------------------------------------------------
    # Base volatility-adjusted position
    # --------------------------------------------------------

    safe_atr_percent = (
        data["ATR_Percent"]
        .replace(0, np.nan)
    )

    volatility_position = (
        TARGET_ATR_PERCENT /
        safe_atr_percent
    ).clip(
        lower=MIN_POSITION,
        upper=MAX_POSITION
    )

    # --------------------------------------------------------
    # Confidence adjustment
    # --------------------------------------------------------

    confidence = pd.Series(
        0.0,
        index=data.index
    )

    # --------------------------------------------------------
    # LONG confidence
    # --------------------------------------------------------

    long_edge = (
        data["Long_Score"] -
        data["Short_Score"]
    )

    confidence += np.where(
        data["Trade_Signal"] == 1,

        np.select(
            [
                long_edge >= 4,
                long_edge >= 3,
                long_edge >= 2
            ],
            [
                0.15,
                0.10,
                0.05
            ],
            default=0.0
        ),

        0.0
    )

    # --------------------------------------------------------
    # SHORT confidence
    # --------------------------------------------------------

    short_edge = (
        data["Short_Score"] -
        data["Long_Score"]
    )

    confidence += np.where(
        data["Trade_Signal"] == -1,

        np.select(
            [
                short_edge >= 4,
                short_edge >= 3,
                short_edge >= 2
            ],
            [
                0.15,
                0.10,
                0.05
            ],
            default=0.0
        ),

        0.0
    )

    # --------------------------------------------------------
    # Breakout confirmation
    # --------------------------------------------------------

    breakout_confirmed = (
        (
            data["Trade_Signal"] != 0
        )
        &
        (
            data["Breakout_Signal"] ==
            data["Trade_Signal"]
        )
    )

    confidence += np.where(
        breakout_confirmed,
        0.05,
        0.0
    )

    # --------------------------------------------------------
    # Volume confirmation
    # --------------------------------------------------------

    # High volume DOES NOT create the trade.
    #
    # It only increases position confidence when the
    # direction is already valid.

    active_volume = (
        data["Volume_Ratio"] >= 1.2
    )

    confidence += np.where(
        (
            data["Trade_Signal"] != 0
        )
        &
        active_volume,
        0.05,
        0.0
    )

    # Very strong volume gets another small increment.
    confidence += np.where(
        (
            data["Trade_Signal"] != 0
        )
        &
        (
            data["Volume_Ratio"] >= 1.5
        ),
        0.05,
        0.0
    )

    # --------------------------------------------------------
    # Final position size
    # --------------------------------------------------------

    data["Position_Size"] = (
        volatility_position +
        confidence
    ).clip(
        lower=MIN_POSITION,
        upper=MAX_POSITION
    )

    # No position when no signal
    data.loc[
        data["Trade_Signal"] == 0,
        "Position_Size"
    ] = 0.0

    return data


# ============================================================
# 6. COMPLETE PIPELINE
# ============================================================

def build_strategy(df):

    data = calculate_features(df)

    data = calculate_scores(data)

    data = generate_signal(data)

    data = calculate_position_size(data)

    return data


# ============================================================
# 7. DIAGNOSTICS
# ============================================================

def strategy_diagnostics(data):

    diagnostics = {}

    diagnostics["Long_Signals"] = int(
        (
            data["Trade_Signal"] == 1
        ).sum()
    )

    diagnostics["Short_Signals"] = int(
        (
            data["Trade_Signal"] == -1
        ).sum()
    )

    diagnostics["Flat_Bars"] = int(
        (
            data["Trade_Signal"] == 0
        ).sum()
    )

    volume_ratio = (
        data["Volume_Ratio"]
        .replace(
            [np.inf, -np.inf],
            np.nan
        )
    )

    diagnostics["Average_Volume_Ratio"] = float(
        volume_ratio.mean()
    )

    diagnostics["Active_Volume_Bars_%"] = float(
        (
            volume_ratio >= 1.2
        ).mean() * 100
    )

    diagnostics["Strong_Volume_Bars_%"] = float(
        (
            volume_ratio >= 1.5
        ).mean() * 100
    )

    return diagnostics
