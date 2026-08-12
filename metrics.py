# ============================================================
# metrics.py
# Performance Evaluation Module
# ============================================================

import numpy as np
import pandas as pd


TRADING_DAYS = 252


# ============================================================
# BASIC RETURN METRICS
# ============================================================

def cumulative_return(equity_curve, initial_capital):
    """
    Cumulative return of the strategy.
    """

    if len(equity_curve) == 0:
        return 0.0

    final_equity = float(equity_curve.iloc[-1])

    return (
        final_equity / initial_capital - 1
    ) * 100


def annualized_return(equity_curve, initial_capital):
    """
    Annualized geometric return.
    """

    if len(equity_curve) < 2:
        return 0.0

    final_equity = float(equity_curve.iloc[-1])

    years = len(equity_curve) / TRADING_DAYS

    if years <= 0 or final_equity <= 0:
        return 0.0

    return (
        (final_equity / initial_capital) ** (1 / years) - 1
    ) * 100


# ============================================================
# RISK METRICS
# ============================================================

def sharpe_ratio(equity_curve):
    """
    Annualized Sharpe ratio.

    Assumes daily returns and zero risk-free rate.
    """

    returns = equity_curve.pct_change().dropna()

    if len(returns) < 2:
        return 0.0

    std = returns.std()

    if std == 0 or pd.isna(std):
        return 0.0

    return (
        returns.mean() / std
    ) * np.sqrt(TRADING_DAYS)


def sortino_ratio(equity_curve):
    """
    Annualized Sortino ratio.

    Uses downside deviation of negative daily returns.
    """

    returns = equity_curve.pct_change().dropna()

    if len(returns) < 2:
        return 0.0

    downside_returns = returns[returns < 0]

    if len(downside_returns) == 0:
        return np.inf

    downside_deviation = np.sqrt(
        (downside_returns ** 2).mean()
    )

    if downside_deviation == 0 or pd.isna(downside_deviation):
        return 0.0

    return (
        returns.mean() / downside_deviation
    ) * np.sqrt(TRADING_DAYS)


def maximum_drawdown(equity_curve):
    """
    Maximum drawdown in percentage terms.
    """

    if len(equity_curve) == 0:
        return 0.0

    running_max = equity_curve.cummax()

    drawdown = (
        equity_curve / running_max - 1
    )

    return drawdown.min() * 100


# ============================================================
# TRADE METRICS
# ============================================================

def number_of_trades(trades):
    """
    Total completed trades.
    """

    if trades is None or len(trades) == 0:
        return 0

    return len(trades)


def win_rate(trades):
    """
    Percentage of profitable trades.
    """

    if trades is None or len(trades) == 0:
        return 0.0

    return (
        (trades["PnL"] > 0).mean()
    ) * 100


def profit_factor(trades):
    """
    Gross profit / gross loss.

    Loss is converted to positive magnitude.
    """

    if trades is None or len(trades) == 0:
        return 0.0

    gross_profit = trades.loc[
        trades["PnL"] > 0,
        "PnL"
    ].sum()

    gross_loss = abs(
        trades.loc[
            trades["PnL"] < 0,
            "PnL"
        ].sum()
    )

    if gross_loss == 0:

        if gross_profit > 0:
            return np.inf

        return 0.0

    return gross_profit / gross_loss


def net_pnl(trades):
    """
    Total realized PnL from completed trades.
    """

    if trades is None or len(trades) == 0:
        return 0.0

    return float(trades["PnL"].sum())


# ============================================================
# TRADE DIAGNOSTICS
# ============================================================

def trade_statistics(trades):
    """
    Returns useful trade-level statistics.
    """

    if trades is None or len(trades) == 0:

        return {
            "Trades": 0,
            "Win Rate (%)": 0.0,
            "Profit Factor": 0.0,
            "Net PnL": 0.0,
            "Average PnL": 0.0,
            "Average Winning Trade": 0.0,
            "Average Losing Trade": 0.0,
            "Best Trade": 0.0,
            "Worst Trade": 0.0,
        }

    pnl = trades["PnL"]

    winning = pnl[pnl > 0]
    losing = pnl[pnl < 0]

    return {
        "Trades": len(trades),

        "Win Rate (%)": (
            len(winning) / len(pnl)
        ) * 100,

        "Profit Factor": profit_factor(trades),

        "Net PnL": pnl.sum(),

        "Average PnL": pnl.mean(),

        "Average Winning Trade": (
            winning.mean()
            if len(winning) > 0
            else 0.0
        ),

        "Average Losing Trade": (
            losing.mean()
            if len(losing) > 0
            else 0.0
        ),

        "Best Trade": (
            pnl.max()
            if len(pnl) > 0
            else 0.0
        ),

        "Worst Trade": (
            pnl.min()
            if len(pnl) > 0
            else 0.0
        ),
    }


# ============================================================
# COMPLETE STRATEGY METRICS
# ============================================================

def calculate_metrics(
    equity_curve,
    trades,
    final_equity,
    initial_capital
):
    """
    Calculate complete strategy performance metrics.

    Parameters
    ----------
    equity_curve : pd.Series or pd.DataFrame
        Daily equity curve.

    trades : pd.DataFrame
        Completed trades containing a 'PnL' column.

    final_equity : float
        Final account equity from the backtester.

    initial_capital : float
        Starting capital.

    Returns
    -------
    dict
        Complete performance summary.
    """

    # --------------------------------------------------------
    # Handle DataFrame equity curve
    # --------------------------------------------------------

    if isinstance(equity_curve, pd.DataFrame):

        if "Equity" not in equity_curve.columns:
            raise ValueError(
                "Equity curve DataFrame must contain "
                "'Equity' column."
            )

        equity = equity_curve["Equity"]

    else:
        equity = equity_curve.copy()

    # --------------------------------------------------------
    # Core metrics
    # --------------------------------------------------------

    cumulative_ret = cumulative_return(
        equity,
        initial_capital
    )

    annual_ret = annualized_return(
        equity,
        initial_capital
    )

    sharpe = sharpe_ratio(equity)

    sortino = sortino_ratio(equity)

    max_dd = maximum_drawdown(equity)

    trade_stats = trade_statistics(trades)

    # --------------------------------------------------------
    # Accounting
    # --------------------------------------------------------

    total_pnl = trade_stats["Net PnL"]

    accounting_difference = (
        final_equity
        - (initial_capital + total_pnl)
    )

    # --------------------------------------------------------
    # Final result
    # --------------------------------------------------------

    return {
        "Final Equity": float(final_equity),

        "Cumulative Return (%)": cumulative_ret,

        "Annualized Return (%)": annual_ret,

        "Trades": trade_stats["Trades"],

        "Win Rate (%)": trade_stats["Win Rate (%)"],

        "Profit Factor": trade_stats["Profit Factor"],

        "Sharpe": sharpe,

        "Sortino": sortino,

        "Max Drawdown (%)": max_dd,

        "Net PnL": total_pnl,

        "Average PnL": trade_stats["Average PnL"],

        "Average Winning Trade":
            trade_stats["Average Winning Trade"],

        "Average Losing Trade":
            trade_stats["Average Losing Trade"],

        "Best Trade":
            trade_stats["Best Trade"],

        "Worst Trade":
            trade_stats["Worst Trade"],

        "Accounting Diff":
            accounting_difference,
    }


# ============================================================
# BUY & HOLD BENCHMARK
# ============================================================

def buy_and_hold_curve(
    df,
    initial_capital
):
    """
    Calculate Buy & Hold equity curve and metrics.

    Assumes investment at the first available closing price.
    """

    if df is None or len(df) == 0:
        raise ValueError(
            "Cannot calculate Buy & Hold on empty data."
        )

    close = df["Close"].astype(float)

    first_price = float(close.iloc[0])

    shares = initial_capital / first_price

    equity = close * shares

    equity.name = "Equity"

    cumulative_ret = (
        equity.iloc[-1] /
        initial_capital - 1
    ) * 100

    years = len(equity) / TRADING_DAYS

    if years > 0:
        annual_ret = (
            (equity.iloc[-1] / initial_capital)
            ** (1 / years) - 1
        ) * 100
    else:
        annual_ret = 0.0

    max_dd = maximum_drawdown(equity)

    return (
        equity.to_frame(),
        {
            "Final Equity": float(equity.iloc[-1]),

            "Cumulative Return (%)":
                cumulative_ret,

            "Annualized Return (%)":
                annual_ret,

            "Max Drawdown (%)":
                max_dd,
        }
    )


# ============================================================
# COMPARISON TABLE
# ============================================================

def compare_strategy_vs_benchmark(
    strategy_metrics,
    benchmark_metrics
):
    """
    Create a compact strategy vs benchmark comparison.
    """

    return pd.DataFrame(
        {
            "Strategy": strategy_metrics,
            "Buy & Hold": benchmark_metrics,
        }
    )


# ============================================================
# FORMATTING HELPERS
# ============================================================

def format_metrics(metrics):
    """
    Convert metrics dictionary into a readable DataFrame.
    """

    return pd.Series(metrics, name="Value")


def print_metrics(
    metrics,
    title="PERFORMANCE"
):
    """
    Pretty-print performance metrics.
    """

    print("\n" + "=" * 70)
    print(title)
    print("=" * 70)

    for key, value in metrics.items():

        if isinstance(value, (int, np.integer)):
            print(f"{key:<30}: {value}")

        elif isinstance(value, (float, np.floating)):

            if np.isinf(value):
                value_str = "inf"

            else:
                value_str = f"{value:,.3f}"

            print(
                f"{key:<30}: {value_str}"
            )

        else:
            print(
                f"{key:<30}: {value}"
            )
