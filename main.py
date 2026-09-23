"""Backtest a 2 MWh / 1 MW battery on French day-ahead prices."""
import json
from datetime import date
from pathlib import Path

import plots
from backtest import backtest, perfect_forecast, summary
from battery import Battery
from forecast import DAY, train_ridge, weekly_forecast

WEAR_COSTS = (25, 40)  # EUR per MWh taken out of the battery
PERIODS = {
    # used to choose between ridge and its weekend variant
    "validation": (date(2024, 1, 1), date(2024, 7, 1)),
    # kept for the end
    "test": (date(2024, 7, 1), date(2025, 1, 1)),
    # downloaded after everything else was done
    "2025": (date(2025, 1, 1), date(2025, 7, 1)),
}


def load_prices():
    prices = {}
    for name in ("2022", "2023", "2024", "2025_h1"):
        raw = json.loads(Path(f"data/prices_{name}.json").read_text())
        prices.update(zip(raw["unix_seconds"], raw["price"]))
    return prices


def main():
    prices = load_prices()
    # both models are trained once, on 2022-2023, and never retrained
    forecasts = {"weekly": weekly_forecast, "ridge": train_ridge(prices)}
    weekend = train_ridge(prices, weekend=True)
    Path("figures").mkdir(exist_ok=True)

    for period, (start, end) in PERIODS.items():
        methods = dict(forecasts)
        if period == "validation":
            methods["ridge + weekend"] = weekend
        results = {}
        for cost in WEAR_COSTS:
            battery = Battery(wear_cost=cost)
            results[cost] = {name: backtest(prices, start, end, battery, f) for name, f in methods.items()}
            results[cost]["perfect forecast"] = perfect_forecast(prices, start, end, battery)

        print(f"\n{period}, {start} to {end - DAY} (MAE in EUR/MWh, gains in EUR)")
        print(f"{'':20}{'MAE':>8}" + "".join(f"{f'gain, wear {c}':>18}" for c in WEAR_COSTS))
        for name in results[WEAR_COSTS[0]]:
            gains = [summary(results[c][name], Battery(wear_cost=c))["gain"] for c in WEAR_COSTS]
            mae = summary(results[WEAR_COSTS[0]][name], Battery())["mae"]
            mae = "-" if name == "perfect forecast" else f"{mae:.2f}"
            print(f"{name:20}{mae:>8}" + "".join(f"{g:>18,.0f}" for g in gains))

        if period != "validation":
            title = f"Cumulative gain after wear, {start:%B %Y} to {end - DAY:%B %Y}"
            plots.cumulative_gains(results, title, f"figures/gains-{period}.png")


if __name__ == "__main__":
    main()
