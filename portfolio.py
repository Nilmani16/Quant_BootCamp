# ============================================================
# portfolio.py
# Portfolio Construction & Benchmark Comparison
# ============================================================

import numpy as np
import pandas as pd

from metrics import (
    cumulative_return,
    annualized_return,
    sharpe_ratio,
    sortino_ratio,
    maximum_drawdown,
)


TRADING_DAYS = 252


# ============================================================
# 1. ALIGN EQUITY CURVES
# ============================================================

def align_equity_curves(equity_curves):
    """
    Align individual stock equity curves on a common date index.

    Parameters
    ----------
    equity_curves : dict
        {
            "HDFCBANK.NS": pd.Series,
            "ICICIBANK.NS": pd.Series,
            ...
        }

    Returns
    -------
    pd.DataFrame
        Aligned equity curves.
    """

    if not equity_curves:
        raise ValueError(
            "equity_curves cannot be empty."
        )

    curves = {}

    for ticker, curve in equity_curves.items():

        if curve is None or len(curve) == 0:
            continue

        if isinstance(curve, pd.DataFrame):

            if "Equity" not in curve.columns:
                raise ValueError(
                    f"{ticker}: DataFrame must contain "
                    "'Equity' column."
                )

            curve = curve["Equity"]

        curves[ticker] = curve.astype(float)

    if not curves:
        raise ValueError(
            "No valid equity curves found."
        )

    aligned = pd.concat(
        curves,
        axis=1
    )

    aligned = aligned.sort_index()

    # Equity is carried forward on days where a stock
    # has no new observation.
    aligned = aligned.ffill()

    aligned = aligned.dropna()

    return aligned


# ============================================================
# 2. COMBINED PORTFOLIO
# ============================================================

def build_combined_portfolio(
    equity_curves,
    initial_capital
):
    """
    Build an equal-capital combined portfolio.

    Each stock is assumed to receive an equal share
    of the total portfolio capital.

    Example:
        ₹1,000,000 total
        5 stocks
        ₹200,000 per stock

    Since individual equity curves already contain
    their allocated capital, we simply sum them.
    """

    aligned = align_equity_curves(
        equity_curves
    )

    combined = aligned.sum(axis=1)

    combined.name = "Combined Portfolio"

    expected_initial = float(
        initial_capital
    )

    # Sanity check.
    actual_initial = float(
        combined.iloc[0]
    )

    if not np.isclose(
        actual_initial,
        expected_initial,
        rtol=0.01,
        atol=1.0
    ):
        print(
            "WARNING: Combined portfolio initial "
            f"equity ₹{actual_initial:,.2f} "
            f"differs from expected "
            f"₹{expected_initial:,.2f}."
        )

    return combined


# ============================================================
# 3. PORTFOLIO METRICS
# ============================================================

def calculate_portfolio_metrics(
    portfolio_curve,
    initial_capital
):
    """
    Calculate portfolio-level performance metrics.
    """

    if portfolio_curve is None:
        raise ValueError(
            "portfolio_curve cannot be None."
        )

    if len(portfolio_curve) == 0:
        raise ValueError(
            "portfolio_curve is empty."
        )

    return {
        "Final Equity":
            float(portfolio_curve.iloc[-1]),

        "Cumulative Return (%)":
            cumulative_return(
                portfolio_curve,
                initial_capital
            ),

        "Annualized Return (%)":
            annualized_return(
                portfolio_curve,
                initial_capital
            ),

        "Sharpe":
            sharpe_ratio(
                portfolio_curve
            ),

        "Sortino":
            sortino_ratio(
                portfolio_curve
            ),

        "Max Drawdown (%)":
            maximum_drawdown(
                portfolio_curve
            ),

        "Net PnL":
            float(
                portfolio_curve.iloc[-1]
                - initial_capital
            ),
    }


# ============================================================
# 4. BUY & HOLD PORTFOLIO
# ============================================================

def build_buy_and_hold_portfolio(
    buy_hold_curves,
    initial_capital
):
    """
    Build an equal-capital Buy & Hold portfolio
    from the five individual stock Buy & Hold curves.

    The individual curves should already represent
    equal capital allocations.
    """

    aligned = align_equity_curves(
        buy_hold_curves
    )

    combined = aligned.sum(axis=1)

    combined.name = "Buy & Hold Portfolio"

    return combined


# ============================================================
# 5. SCALE BENCHMARK TO PORTFOLIO CAPITAL
# ============================================================

def scale_benchmark_curve(
    benchmark_curve,
    initial_capital
):
    """
    Scale a benchmark equity curve so that it starts
    with the same capital as the strategy portfolio.
    """

    if benchmark_curve is None:
        raise ValueError(
            "benchmark_curve cannot be None."
        )

    benchmark_curve = (
        benchmark_curve
        .astype(float)
        .dropna()
    )

    if len(benchmark_curve) == 0:
        raise ValueError(
            "benchmark_curve is empty."
        )

    first_value = float(
        benchmark_curve.iloc[0]
    )

    if first_value <= 0:
        raise ValueError(
            "Benchmark initial value must be positive."
        )

    scaled = (
        benchmark_curve
        / first_value
        * initial_capital
    )

    scaled.name = "Benchmark"

    return scaled


# ============================================================
# 6. PORTFOLIO vs BENCHMARK
# ============================================================

def compare_portfolio_with_benchmarks(
    portfolio_curve,
    buy_hold_curve,
    benchmark_curve,
    initial_capital
):
    """
    Compare:

        Strategy Portfolio
        Buy & Hold Portfolio
        NIFTY Bank

    All curves are normalized to the same starting capital.
    """

    strategy_metrics = (
        calculate_portfolio_metrics(
            portfolio_curve,
            initial_capital
        )
    )

    buy_hold_metrics = (
        calculate_portfolio_metrics(
            buy_hold_curve,
            initial_capital
        )
    )

    benchmark_scaled = (
        scale_benchmark_curve(
            benchmark_curve,
            initial_capital
        )
    )

    benchmark_metrics = (
        calculate_portfolio_metrics(
            benchmark_scaled,
            initial_capital
        )
    )

    comparison = pd.DataFrame(
        {
            "Strategy Portfolio":
                strategy_metrics,

            "Buy & Hold Portfolio":
                buy_hold_metrics,

            "NIFTY Bank":
                benchmark_metrics,
        }
    )

    return (
        comparison,
        benchmark_scaled
    )


# ============================================================
# 7. INDIVIDUAL STOCK CONTRIBUTION
# ============================================================

def stock_contribution(
    equity_curves,
    initial_capital
):
    """
    Calculate each stock's contribution to the
    combined portfolio PnL.

    Assumes equal capital allocation.
    """

    aligned = align_equity_curves(
        equity_curves
    )

    n_stocks = len(aligned.columns)

    capital_per_stock = (
        initial_capital / n_stocks
    )

    rows = []

    for ticker in aligned.columns:

        final_equity = float(
            aligned[ticker].iloc[-1]
        )

        pnl = (
            final_equity
            - capital_per_stock
        )

        ret = (
            pnl / capital_per_stock
        ) * 100

        rows.append(
            {
                "Ticker": ticker,
                "Initial Capital":
                    capital_per_stock,
                "Final Equity":
                    final_equity,
                "Net PnL": pnl,
                "Return (%)": ret,
            }
        )

    result = pd.DataFrame(rows)

    return result


# ============================================================
# 8. PORTFOLIO SUMMARY
# ============================================================

def portfolio_summary(
    portfolio_curve,
    initial_capital,
    name="COMBINED PORTFOLIO"
):
    """
    Print a clean portfolio summary.
    """

    metrics = calculate_portfolio_metrics(
        portfolio_curve,
        initial_capital
    )

    print("\n" + "=" * 75)
    print(name)
    print("=" * 75)

    print(
        f"Initial Capital      : "
        f"₹{initial_capital:,.2f}"
    )

    print(
        f"Final Equity         : "
        f"₹{metrics['Final Equity']:,.2f}"
    )

    print(
        f"Net PnL              : "
        f"₹{metrics['Net PnL']:,.2f}"
    )

    print(
        f"Cumulative Return    : "
        f"{metrics['Cumulative Return (%)']:.2f}%"
    )

    print(
        f"Annualized Return    : "
        f"{metrics['Annualized Return (%)']:.2f}%"
    )

    print(
        f"Sharpe               : "
        f"{metrics['Sharpe']:.3f}"
    )

    print(
        f"Sortino              : "
        f"{metrics['Sortino']:.3f}"
    )

    print(
        f"Max Drawdown         : "
        f"{metrics['Max Drawdown (%)']:.2f}%"
    )


# ============================================================
# 9. EQUITY CURVE DATA FOR PLOTTING
# ============================================================

def prepare_portfolio_plot_data(
    strategy_curves,
    buy_hold_curves,
    benchmark_curve,
    initial_capital
):
    """
    Prepare all curves required for final plots.

    Returns
    -------
    dict
    """

    strategy_portfolio = (
        build_combined_portfolio(
            strategy_curves,
            initial_capital
        )
    )

    buy_hold_portfolio = (
        build_buy_and_hold_portfolio(
            buy_hold_curves,
            initial_capital
        )
    )

    benchmark_scaled = (
        scale_benchmark_curve(
            benchmark_curve,
            initial_capital
        )
    )

    return {
        "strategy": strategy_portfolio,

        "buy_hold": buy_hold_portfolio,

        "nifty_bank": benchmark_scaled,
    }
