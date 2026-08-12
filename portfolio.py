# ============================================================
# Portfolio Construction & Benchmark Comparison
# ============================================================

import numpy as np
import pandas as pd

from metrics import (
    calculate_metrics,
    cumulative_return,
    annualized_return,
    sharpe_ratio,
    sortino_ratio,
    maximum_drawdown,
)


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
    """

    if not equity_curves:
        raise ValueError("equity_curves cannot be empty.")

    curves = {}

    for ticker, curve in equity_curves.items():

        if curve is None or len(curve) == 0:
            continue

        if isinstance(curve, pd.DataFrame):

            if "Equity" not in curve.columns:
                raise ValueError(
                    f"{ticker}: DataFrame must contain 'Equity' column."
                )

            curve = curve["Equity"]

        curves[ticker] = curve.astype(float)

    if not curves:
        raise ValueError("No valid equity curves found.")

    aligned = pd.concat(curves, axis=1)

    aligned = aligned.sort_index()

    # Carry forward equity on dates where a stock
    # has no observation.
    aligned = aligned.ffill()

    aligned = aligned.dropna()

    return aligned


# ============================================================
# 2. COMBINED STRATEGY PORTFOLIO
# ============================================================

def build_combined_portfolio(
    equity_curves,
    initial_capital
):
    """
    Build equal-capital combined portfolio.

    Example:
        Total capital = ₹1,000,000
        5 stocks
        ₹200,000 per stock

    The individual equity curves are summed.
    """

    aligned = align_equity_curves(equity_curves)

    combined = aligned.sum(axis=1)

    combined.name = "Combined Portfolio"

    expected_initial = float(initial_capital)
    actual_initial = float(combined.iloc[0])

    if not np.isclose(
        actual_initial,
        expected_initial,
        rtol=0.01,
        atol=1.0
    ):
        print(
            f"WARNING: Portfolio starts at "
            f"₹{actual_initial:,.2f}, expected "
            f"₹{expected_initial:,.2f}"
        )

    return combined


def build_portfolio(
    results,
    initial_capital=1_000_000.0
):
    """Build a combined portfolio from test-period equity curves."""

    if not results:
        raise ValueError("results cannot be empty.")

    equity_curves = {}

    for ticker, payload in results.items():

        if not payload or "test" not in payload:
            continue

        equity_curve = payload["test"][0]

        if equity_curve is None or len(equity_curve) == 0:
            continue

        if isinstance(equity_curve, pd.DataFrame):
            if "Equity" not in equity_curve.columns:
                raise ValueError(
                    f"{ticker}: test equity curve must contain 'Equity' column."
                )
            equity_series = equity_curve["Equity"]
        else:
            equity_series = equity_curve.copy()

        equity_series = equity_series.astype(float)

        equity_curves[ticker] = equity_series

    if not equity_curves:
        raise ValueError(
            "No valid test equity curves available for portfolio construction."
        )

    per_stock_capital = (
        float(initial_capital) /
        len(equity_curves)
    )

    scaled_curves = {
        ticker: (
            curve / curve.iloc[0] * per_stock_capital
        )
        for ticker, curve in equity_curves.items()
    }

    return build_combined_portfolio(
        scaled_curves,
        initial_capital
    )


# ============================================================
# 3. PORTFOLIO PERFORMANCE
# ============================================================

def calculate_portfolio_metrics(
    portfolio_curve,
    initial_capital
):
    """
    Calculate portfolio-level performance metrics.
    """

    if portfolio_curve is None:
        raise ValueError("portfolio_curve cannot be None.")

    if len(portfolio_curve) == 0:
        raise ValueError("portfolio_curve is empty.")

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
            sharpe_ratio(portfolio_curve),

        "Sortino":
            sortino_ratio(portfolio_curve),

        "Max Drawdown (%)":
            maximum_drawdown(portfolio_curve),

        "Net PnL":
            float(
                portfolio_curve.iloc[-1]
                - initial_capital
            ),
    }


# ============================================================
# 4. COMBINED BUY & HOLD PORTFOLIO
# ============================================================

def build_buy_and_hold_portfolio(
    buy_hold_curves,
    initial_capital
):
    """
    Build equal-capital Buy & Hold portfolio
    from the five individual stock curves.
    """

    aligned = align_equity_curves(buy_hold_curves)

    combined = aligned.sum(axis=1)

    combined.name = "Buy & Hold Portfolio"

    return combined


# ============================================================
# 5. SCALE BENCHMARK
# ============================================================

def scale_benchmark_curve(
    benchmark_curve,
    initial_capital
):
    """
    Scale NIFTY Bank benchmark to the same
    starting capital as the strategy portfolio.
    """

    if benchmark_curve is None:
        raise ValueError("benchmark_curve cannot be None.")

    benchmark_curve = (
        benchmark_curve
        .astype(float)
        .dropna()
    )

    if len(benchmark_curve) == 0:
        raise ValueError("benchmark_curve is empty.")

    first_value = float(benchmark_curve.iloc[0])

    if first_value <= 0:
        raise ValueError(
            "Benchmark initial value must be positive."
        )

    scaled = (
        benchmark_curve
        / first_value
        * initial_capital
    )

    scaled.name = "NIFTY Bank"

    return scaled


# ============================================================
# 6. STRATEGY vs BENCHMARK
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
    """

    strategy_metrics = calculate_portfolio_metrics(
        portfolio_curve,
        initial_capital
    )

    buy_hold_metrics = calculate_portfolio_metrics(
        buy_hold_curve,
        initial_capital
    )

    benchmark_scaled = scale_benchmark_curve(
        benchmark_curve,
        initial_capital
    )

    benchmark_metrics = calculate_portfolio_metrics(
        benchmark_scaled,
        initial_capital
    )

    comparison = pd.DataFrame({
        "Strategy Portfolio":
            strategy_metrics,

        "Buy & Hold Portfolio":
            buy_hold_metrics,

        "NIFTY Bank":
            benchmark_metrics,
    })

    return comparison, benchmark_scaled


# ============================================================
# 7. STOCK-WISE CONTRIBUTION
# ============================================================

def stock_contribution(
    equity_curves,
    initial_capital
):
    """
    Calculate each stock's contribution
    to the combined portfolio.
    """

    aligned = align_equity_curves(equity_curves)

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

        rows.append({
            "Ticker": ticker,
            "Initial Capital": capital_per_stock,
            "Final Equity": final_equity,
            "Net PnL": pnl,
            "Return (%)": ret,
        })

    return pd.DataFrame(rows)


# ============================================================
# 8. PORTFOLIO SUMMARY
# ============================================================

def portfolio_summary(
    portfolio_curve,
    initial_capital,
    name="COMBINED PORTFOLIO"
):
    """
    Print clean portfolio summary.
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
# 9. PREPARE FINAL PLOT DATA
# ============================================================

def prepare_portfolio_plot_data(
    strategy_curves,
    buy_hold_curves,
    benchmark_curve,
    initial_capital
):
    """
    Prepare curves for final visualization.
    """

    strategy_portfolio = build_combined_portfolio(
        strategy_curves,
        initial_capital
    )

    buy_hold_portfolio = build_buy_and_hold_portfolio(
        buy_hold_curves,
        initial_capital
    )

    benchmark_scaled = scale_benchmark_curve(
        benchmark_curve,
        initial_capital
    )

    return {
        "strategy": strategy_portfolio,
        "buy_hold": buy_hold_portfolio,
        "nifty_bank": benchmark_scaled,
    }


def print_results(
    results,
    portfolio,
    initial_capital=1_000_000.0
):
    """Print backtest and portfolio summaries."""

    print("\n" + "=" * 80)
    print("BACKTEST SUMMARY")
    print("=" * 80)

    for ticker, payload in results.items():

        print(f"\nTicker: {ticker}")

        for period in ("train", "test"):

            if period not in payload:
                continue

            equity_curve, trades, result = payload[period]
            metrics = calculate_metrics(
                equity_curve,
                trades,
                result["Final Equity"],
                result["Initial Capital"]
            )

            print(
                f"  {period.title():<5} | "
                f"Final Equity: ₹{metrics['Final Equity']:,.2f} | "
                f"Return: {metrics['Cumulative Return (%)']:.2f}% | "
                f"Sharpe: {metrics['Sharpe']:.3f} | "
                f"Trades: {metrics['Trades']} | "
                f"Win Rate: {metrics['Win Rate (%)']:.1f}%"
            )

    print("\n" + "=" * 80)
    print("COMBINED PORTFOLIO")
    print("=" * 80)

    portfolio_metrics = calculate_portfolio_metrics(
        portfolio,
        initial_capital
    )

    for name, value in portfolio_metrics.items():

        if isinstance(value, float):
            if name.endswith("(%"):
                print(f"{name:<25}: {value:.2f}%")
            else:
                print(f"{name:<25}: {value:,.2f}")
        else:
            print(f"{name:<25}: {value}")

    print("=" * 80)
