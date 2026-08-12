from data import load_all_data
from strategy import build_strategy
from backtester import backtest
from metrics import calculate_metrics
from portfolio import build_portfolio


def main():

    # 1. Load data
    data = load_all_data()

    # 2. Run strategy + backtest
    results = {}

    for ticker, splits in data.items():

        train = build_strategy(splits["train"])
        test = build_strategy(splits["test"])

        results[ticker] = {
            "train": backtest(train),
            "test": backtest(test)
        }

    # 3. Build combined portfolio
    portfolio = build_portfolio(results)

    # 4. Print final results
    print_results(results, portfolio)


if __name__ == "__main__":
    main()
