from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

from data import load_all_data, load_benchmark_data
from strategy import build_strategy
from backtester import backtest
from metrics import (
    calculate_metrics,
    buy_and_hold_curve,
)
from portfolio import (
    build_portfolio,
    build_buy_and_hold_portfolio,
    compare_portfolio_with_benchmarks,
)


# ============================================================
# CONFIG
# ============================================================

INITIAL_CAPITAL = 1_000_000.0

# Create outputs relative to this Python file
OUTPUT_DIR = Path(__file__).resolve().parent / "outputs"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================
# PLOT HELPER
# ============================================================

def save_equity_plot(
    curves,
    title,
    filename,
):
    """
    Save one equity-curve plot.
    curves = {"Strategy": series, "Buy & Hold": series, ...}
    """

    plt.figure(figsize=(12, 6))

    for name, curve in curves.items():
        plt.plot(
            curve.index,
            curve.values,
            label=name,
            linewidth=2
        )

    plt.title(title)
    plt.xlabel("Date")
    plt.ylabel("Equity (₹)")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    path = OUTPUT_DIR / filename
    plt.savefig(path, dpi=200)
    plt.close()

    print(f"Saved plot: {path}")


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 80)
    print("INTER-IIT QUANT BOOTCAMP")
    print("=" * 80)

    print(f"\nOutput directory: {OUTPUT_DIR}")

    # --------------------------------------------------------
    # 1. LOAD STOCK DATA
    # --------------------------------------------------------

    stock_data = load_all_data()

    # --------------------------------------------------------
    # 2. LOAD NIFTY BANK BENCHMARK
    # --------------------------------------------------------

    benchmark_data = load_benchmark_data()

    # --------------------------------------------------------
    # 3. RUN TRAIN + TEST
    # --------------------------------------------------------

    results = {}

    # Curves for Buy & Hold
    buy_hold_curves = {}

    # Test strategy curves
    test_strategy_curves = {}

    for ticker, splits in stock_data.items():

        print("\n" + "=" * 80)
        print(ticker)
        print("=" * 80)

        # ====================================================
        # TRAIN
        # ====================================================

        train_strategy = build_strategy(
            splits["train"]
        )

        train_result = backtest(
            train_strategy,
            initial_capital=INITIAL_CAPITAL
        )

        # ====================================================
        # TEST WARM-UP
        # ====================================================

        # Calculate indicators using TRAIN + TEST together.
        # This gives 2024 indicators access to late-2023 history.

        combined_data = pd.concat(
            [
                splits["train"],
                splits["test"]
            ]
        )

        combined_strategy = build_strategy(
            combined_data
        )

        # Slice ONLY the actual 2024 test period
        test_strategy = combined_strategy.loc[
            splits["test"].index
        ].copy()

        test_result = backtest(
            test_strategy,
            initial_capital=INITIAL_CAPITAL
        )

        results[ticker] = {
            "train": train_result,
            "test": test_result
        }

        # ====================================================
        # BUY & HOLD
        # ====================================================

        buy_hold_curve, buy_hold_metrics = buy_and_hold_curve(
            splits["test"],
            INITIAL_CAPITAL
        )

        buy_hold_curves[ticker] = (
            buy_hold_curve["Equity"]
        )

        test_strategy_curves[ticker] = (
            test_result[0]["Equity"]
        )

        # ====================================================
        # INDIVIDUAL STOCK PLOT
        # ====================================================

        save_equity_plot(
            {
                "Strategy": test_result[0]["Equity"],
                "Buy & Hold": buy_hold_curve["Equity"],
            },
            f"{ticker} - 2024 Out-of-Sample",
            f"{ticker.replace('.NS', '')}_equity.png",
        )

    # ========================================================
    # 4. COMBINED STRATEGY PORTFOLIO
    # ========================================================

    strategy_portfolio = build_portfolio(
        results,
        initial_capital=INITIAL_CAPITAL
    )

    # ========================================================
    # 5. COMBINED BUY & HOLD
    # ========================================================

    # Each stock gets equal capital.
    per_stock_capital = (
        INITIAL_CAPITAL / len(buy_hold_curves)
    )

    scaled_buy_hold_curves = {
        ticker: (
            curve / curve.iloc[0] * per_stock_capital
        )
        for ticker, curve in buy_hold_curves.items()
    }

    buy_hold_portfolio = build_buy_and_hold_portfolio(
        scaled_buy_hold_curves,
        INITIAL_CAPITAL
    )

    # ========================================================
    # 6. NIFTY BANK
    # ========================================================

    nifty_test = benchmark_data["test"]

    nifty_close = nifty_test["Close"].astype(float)

    nifty_scaled = (
        nifty_close
        / nifty_close.iloc[0]
        * INITIAL_CAPITAL
    )

    nifty_scaled.name = "NIFTY Bank"

    # ========================================================
    # 7. BENCHMARK COMPARISON
    # ========================================================

    comparison, benchmark_curve = (
        compare_portfolio_with_benchmarks(
            strategy_portfolio,
            buy_hold_portfolio,
            nifty_scaled,
            INITIAL_CAPITAL
        )
    )

    print("\n")
    print("=" * 80)
    print("2024 OUT-OF-SAMPLE PORTFOLIO COMPARISON")
    print("=" * 80)

    print(comparison.to_string())

    # Save comparison
    comparison.to_csv(
        OUTPUT_DIR / "portfolio_comparison.csv"
    )

    # ========================================================
    # 8. COMBINED PORTFOLIO PLOT
    # ========================================================

    save_equity_plot(
        {
            "Strategy Portfolio": strategy_portfolio,
            "Buy & Hold Portfolio": buy_hold_portfolio,
            "NIFTY Bank": benchmark_curve,
        },
        "2024 Out-of-Sample Portfolio Comparison",
        "combined_portfolio.png",
    )

    # ========================================================
    # 9. SAVE RAW EQUITY CURVES
    # ========================================================

    equity_df = pd.DataFrame(
        test_strategy_curves
    )

    equity_df.to_csv(
        OUTPUT_DIR / "individual_strategy_equity.csv"
    )

    portfolio_df = pd.DataFrame({
        "Strategy Portfolio": strategy_portfolio,
        "Buy & Hold Portfolio": buy_hold_portfolio,
        "NIFTY Bank": benchmark_curve,
    })

    portfolio_df.to_csv(
        OUTPUT_DIR / "portfolio_equity_curves.csv"
    )

    # ========================================================
    # 10. FINAL MESSAGE
    # ========================================================

    print("\n" + "=" * 80)
    print("RUN COMPLETE")
    print("=" * 80)

    print(f"\nAll outputs saved to:")
    print(OUTPUT_DIR)


if __name__ == "__main__":
    main()