import numpy as np
import pandas as pd


# ============================================================
# 1. PARAMETERS
# ============================================================

TRANSACTION_COST = 0.001       # 10 bps per side
SLIPPAGE = 0.001               # 10 bps per side

COOLDOWN_BARS = 1

ATR_STOP_MULTIPLIER = 2.5
ATR_TARGET_MULTIPLIER = 5.0
ATR_TRAILING_MULTIPLIER = 3.0

# ============================================================
# 2. BACKTEST
# ============================================================

def backtest(
    data,
    initial_capital=1_000_000.0,
):
    """
    Event-driven backtester.

    IMPORTANT:
    - Signal on day t is executed on day t+1 OPEN.
    - No look-ahead is used.
    - Position size is based on signal day's Position_Size.
    - Stops/targets use intraday High/Low.
    - Final open position is liquidated at the final close.
    - All costs are explicitly accounted for.

    Required columns from strategy.py:
        Open
        High
        Low
        Close
        ATR
        Trade_Signal
        Position_Size
    """

    data = data.copy().sort_index()

    required = [
        "Open",
        "High",
        "Low",
        "Close",
        "ATR",
        "Trade_Signal",
        "Position_Size",
    ]

    missing = [
        col for col in required
        if col not in data.columns
    ]

    if missing:
        raise ValueError(
            f"Missing required columns: {missing}"
        )

    cash = float(initial_capital)

    position = 0
    shares = 0.0

    entry_price = np.nan
    entry_date = None
    
    cooldown_remaining = 0

    entry_cost = 0.0

    stop_price = np.nan
    target_price = np.nan

    highest_since_entry = np.nan
    lowest_since_entry = np.nan

    trades = []
    equity_records = []

    # --------------------------------------------------------
    # Helper: record daily equity
    # --------------------------------------------------------

    def mark_to_market(close_price):
        if position == 1:
            return cash + shares * close_price

        if position == -1:
            return cash - shares * close_price

        return cash

    # ========================================================
    # MAIN EVENT LOOP
    # ========================================================

    for i in range(1, len(data)):

        today = data.iloc[i]
        previous = data.iloc[i - 1]

        date = data.index[i]

        open_price = float(today["Open"])
        high = float(today["High"])
        low = float(today["Low"])
        close = float(today["Close"])

        signal = int(previous["Trade_Signal"])

        position_size = float(
            previous["Position_Size"]
        )

        atr = float(previous["ATR"])

        if not np.isfinite(atr) or atr <= 0:

            equity_records.append({
                "Date": date,
                "Equity": mark_to_market(close),
                "Cash": cash,
                "Position": position,
            })

            continue

        # ====================================================
        # A. MANAGE EXISTING LONG
        # ====================================================

        if position == 1:

            exit_price = None
            exit_reason = None

            # ------------------------------------------------
            # Conservative same-bar priority:
            # STOP is checked before TARGET.
            #
            # If both could have occurred inside the same
            # candle, we do not assume the favourable ordering.
            # ------------------------------------------------

            if low <= stop_price:

                exit_price = (
                    stop_price *
                    (1 - SLIPPAGE)
                )

                exit_reason = "ATR_STOP"

            elif high >= target_price:

                exit_price = (
                    target_price *
                    (1 - SLIPPAGE)
                )

                exit_reason = "TARGET"

            elif signal == -1:

                exit_price = (
                    open_price *
                    (1 - SLIPPAGE)
                )

                exit_reason = "SIGNAL_REVERSAL"

            if exit_price is not None:

                exit_cost = (
                    shares *
                    exit_price *
                    TRANSACTION_COST
                )

                gross_pnl = (
                    exit_price -
                    entry_price
                ) * shares

                total_pnl = (
                    gross_pnl -
                    entry_cost -
                    exit_cost
                )

                # Sell shares and receive proceeds.
                cash += (
                    shares *
                    exit_price
                )

                cash -= exit_cost

                trades.append({
                    "Date": date,
                    "Direction": "LONG",
                    "Entry_Date": entry_date,
                    "Entry": entry_price,
                    "Exit": exit_price,
                    "Shares": shares,
                    "PnL": total_pnl,
                    "Reason": exit_reason,
                })

                cooldown_remaining = COOLDOWN_BARS

                position = 0
                shares = 0.0

                entry_price = np.nan
                entry_date = None
                entry_cost = 0.0

                stop_price = np.nan
                target_price = np.nan

                highest_since_entry = np.nan
                lowest_since_entry = np.nan

            # ------------------------------------------------
            # Update trailing stop only if still LONG.
            # ------------------------------------------------

            if position == 1:

                highest_since_entry = max(
                    highest_since_entry,
                    high
                )

                trailing_stop = (
                    highest_since_entry -
                    ATR_TRAILING_MULTIPLIER * atr
                )

                stop_price = max(
                    stop_price,
                    trailing_stop
                )

        # ====================================================
        # B. MANAGE EXISTING SHORT
        # ====================================================

        elif position == -1:

            exit_price = None
            exit_reason = None

            if high >= stop_price:

                exit_price = (
                    stop_price *
                    (1 + SLIPPAGE)
                )

                exit_reason = "ATR_STOP"

            elif low <= target_price:

                exit_price = (
                    target_price *
                    (1 + SLIPPAGE)
                )

                exit_reason = "TARGET"

            elif signal == 1:

                exit_price = (
                    open_price *
                    (1 + SLIPPAGE)
                )

                exit_reason = "SIGNAL_REVERSAL"

            if exit_price is not None:

                exit_cost = (
                    shares *
                    exit_price *
                    TRANSACTION_COST
                )

                gross_pnl = (
                    entry_price -
                    exit_price
                ) * shares

                total_pnl = (
                    gross_pnl -
                    entry_cost -
                    exit_cost
                )

                # Buy back the short.
                cash -= (
                    shares *
                    exit_price
                )

                cash -= exit_cost

                trades.append({
                    "Date": date,
                    "Direction": "SHORT",
                    "Entry_Date": entry_date,
                    "Entry": entry_price,
                    "Exit": exit_price,
                    "Shares": shares,
                    "PnL": total_pnl,
                    "Reason": exit_reason,
                })

                cooldown_remaining = COOLDOWN_BARS

                position = 0
                shares = 0.0

                entry_price = np.nan
                entry_date = None
                entry_cost = 0.0

                stop_price = np.nan
                target_price = np.nan

                highest_since_entry = np.nan
                lowest_since_entry = np.nan

            # ------------------------------------------------
            # Update trailing stop only if still SHORT.
            # ------------------------------------------------

            if position == -1:

                lowest_since_entry = min(
                    lowest_since_entry,
                    low
                )

                trailing_stop = (
                    lowest_since_entry +
                    ATR_TRAILING_MULTIPLIER * atr
                )

                stop_price = min(
                    stop_price,
                    trailing_stop
                )

        # ====================================================
        # C. OPEN NEW POSITION
        # ====================================================

        if (
            position == 0
            and cooldown_remaining == 0
            and signal in (1, -1)
            and position_size > 0
        ):

            # ------------------------------------------------
            # LONG ENTRY
            # ------------------------------------------------

            if signal == 1:

                entry = (
                    open_price *
                    (1 + SLIPPAGE)
                )

                allocation = (
                    cash *
                    min(position_size, 1.0)
                )

                shares_candidate = (
                    allocation /
                    entry
                )

                entry_cost_candidate = (
                    shares_candidate *
                    entry *
                    TRANSACTION_COST
                )

                total_required = (
                    shares_candidate * entry
                    +
                    entry_cost_candidate
                )

                if total_required <= cash:

                    cash -= total_required

                    position = 1
                    shares = shares_candidate

                    entry_price = entry
                    entry_date = date
                    entry_cost = entry_cost_candidate

                    highest_since_entry = high

                    stop_price = (
                        entry -
                        ATR_STOP_MULTIPLIER * atr
                    )

                    target_price = (
                        entry +
                        ATR_TARGET_MULTIPLIER * atr
                    )

            # ------------------------------------------------
            # SHORT ENTRY
            # ------------------------------------------------

            elif signal == -1:

                entry = (
                    open_price *
                    (1 - SLIPPAGE)
                )

                allocation = (
                    cash *
                    min(position_size, 1.0)
                )

                shares_candidate = (
                    allocation /
                    entry
                )

                entry_cost_candidate = (
                    shares_candidate *
                    entry *
                    TRANSACTION_COST
                )

                # Short-sale proceeds enter cash.
                cash += (
                    shares_candidate *
                    entry
                )

                cash -= entry_cost_candidate

                position = -1
                shares = shares_candidate

                entry_price = entry
                entry_date = date
                entry_cost = entry_cost_candidate

                lowest_since_entry = low

                stop_price = (
                    entry +
                    ATR_STOP_MULTIPLIER * atr
                )

                target_price = (
                    entry -
                    ATR_TARGET_MULTIPLIER * atr
                )

        # ====================================================
        # D. DAILY MARK-TO-MARKET
        # ====================================================

        equity = mark_to_market(close)

        equity_records.append({
            "Date": date,
            "Equity": equity,
            "Cash": cash,
            "Position": position,
        })
        # ========================================================
        # E. COOLDOWN COUNTDOWN
        # ========================================================

        if cooldown_remaining > 0:
            cooldown_remaining -= 1

    # ========================================================
    # 3. FINAL LIQUIDATION
    # ========================================================

    if position != 0:

        final_date = data.index[-1]
        final_close = float(
            data["Close"].iloc[-1]
        )

        if position == 1:

            exit_price = (
                final_close *
                (1 - SLIPPAGE)
            )

            exit_cost = (
                shares *
                exit_price *
                TRANSACTION_COST
            )

            gross_pnl = (
                exit_price -
                entry_price
            ) * shares

            total_pnl = (
                gross_pnl -
                entry_cost -
                exit_cost
            )

            cash += (
                shares *
                exit_price
            )

            cash -= exit_cost

            trades.append({
                "Date": final_date,
                "Direction": "LONG",
                "Entry_Date": entry_date,
                "Entry": entry_price,
                "Exit": exit_price,
                "Shares": shares,
                "PnL": total_pnl,
                "Reason": "FINAL_CLOSE",
            })

        else:

            exit_price = (
                final_close *
                (1 + SLIPPAGE)
            )

            exit_cost = (
                shares *
                exit_price *
                TRANSACTION_COST
            )

            gross_pnl = (
                entry_price -
                exit_price
            ) * shares

            total_pnl = (
                gross_pnl -
                entry_cost -
                exit_cost
            )

            cash -= (
                shares *
                exit_price
            )

            cash -= exit_cost

            trades.append({
                "Date": final_date,
                "Direction": "SHORT",
                "Entry_Date": entry_date,
                "Entry": entry_price,
                "Exit": exit_price,
                "Shares": shares,
                "PnL": total_pnl,
                "Reason": "FINAL_CLOSE",
            })

        position = 0
        shares = 0.0

    # ========================================================
    # 4. OUTPUTS
    # ========================================================

    equity_curve = pd.DataFrame(
        equity_records
    )

    if not equity_curve.empty:

        equity_curve["Date"] = pd.to_datetime(
            equity_curve["Date"]
        )

        equity_curve = (
            equity_curve
            .set_index("Date")
            .sort_index()
        )

        # Final liquidation is the final account value.
        equity_curve.iloc[-1, equity_curve.columns.get_loc(
            "Equity"
        )] = cash

        equity_curve.iloc[-1, equity_curve.columns.get_loc(
            "Cash"
        )] = cash

        equity_curve.iloc[-1, equity_curve.columns.get_loc(
            "Position"
        )] = 0

    trades_df = pd.DataFrame(
        trades
    )

    if not trades_df.empty:

        trades_df["Date"] = pd.to_datetime(
            trades_df["Date"]
        )

        trades_df["PnL"] = pd.to_numeric(
            trades_df["PnL"]
        )

    final_equity = float(cash)

    realized_pnl = (
        trades_df["PnL"].sum()
        if not trades_df.empty
        else 0.0
    )

    accounting_difference = (
        final_equity -
        (
            initial_capital +
            realized_pnl
        )
    )

    result = {
        "Initial Capital": float(initial_capital),
        "Final Equity": final_equity,
        "Realized PnL": float(realized_pnl),
        "Unrealized PnL": 0.0,
        "Total PnL": float(realized_pnl),
        "Return (%)": (
            (
                final_equity /
                initial_capital
            ) - 1
        ) * 100,
        "Trades": len(trades_df),
        "Accounting Diff": float(
            accounting_difference
        ),
    }

    return (
        equity_curve,
        trades_df,
        result,
    )


# ============================================================
# 3. REPORTING HELPERS
# ============================================================

def print_backtest_summary(
    result,
    trades_df=None,
):
    """
    Clean terminal report.
    """

    print("=" * 70)
    print("BACKTEST")
    print("=" * 70)

    print(
        f"Initial Capital : "
        f"₹{result['Initial Capital']:,.2f}"
    )

    print(
        f"Final Equity    : "
        f"₹{result['Final Equity']:,.2f}"
    )

    print(
        f"Realized PnL    : "
        f"₹{result['Realized PnL']:,.2f}"
    )

    print(
        f"Unrealized PnL  : "
        f"₹{result['Unrealized PnL']:,.2f}"
    )

    print(
        f"Total PnL       : "
        f"₹{result['Total PnL']:,.2f}"
    )

    print(
        f"Return          : "
        f"{result['Return (%)']:.2f}%"
    )

    print(
        f"Trades          : "
        f"{result['Trades']}"
    )

    if trades_df is not None and not trades_df.empty:

        wins = (
            trades_df["PnL"] > 0
        ).sum()

        losses = (
            trades_df["PnL"] < 0
        ).sum()

        win_rate = (
            wins /
            len(trades_df)
            * 100
        )

        gross_profit = (
            trades_df.loc[
                trades_df["PnL"] > 0,
                "PnL"
            ].sum()
        )

        gross_loss = abs(
            trades_df.loc[
                trades_df["PnL"] < 0,
                "PnL"
            ].sum()
        )

        profit_factor = (
            gross_profit /
            gross_loss
            if gross_loss > 0
            else np.inf
        )

        print(
            f"Win Rate        : "
            f"{win_rate:.2f}%"
        )

        print(
            f"Profit Factor   : "
            f"{profit_factor:.3f}"
        )

        print("\nExit Reasons:")

        print(
            trades_df["Reason"]
            .value_counts()
        )

    print("\n")
    print("=" * 70)
    print("ACCOUNTING CHECK")
    print("=" * 70)

    expected = (
        result["Initial Capital"] +
        result["Total PnL"]
    )

    print(
        f"Initial Capital + Total PnL : "
        f"₹{expected:,.2f}"
    )

    print(
        f"Final Equity                : "
        f"₹{result['Final Equity']:,.2f}"
    )

    print(
        f"Difference                  : "
        f"₹{result['Accounting Diff']:,.6f}"
    )


def print_trade_log(trades_df):

    if trades_df is None or trades_df.empty:

        print("\nNo trades generated.")
        return

    print("\nTrade Log:")

    columns = [
        "Date",
        "Direction",
        "Entry",
        "Exit",
        "Shares",
        "PnL",
        "Reason",
    ]

    print(
        trades_df[
            columns
        ].to_string(index=False)
    )
