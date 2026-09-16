"""Backtest a 2 MWh / 1 MW battery on French day-ahead prices."""
import json
from datetime import date
from pathlib import Path

from backtest import backtest, perfect_forecast, summary
from battery import Battery
from forecast import DAY, train_ridge, weekly_forecast

WEAR_COSTS = (25, 40)  # EUR per MWh taken out of the battery
PERIODS = {
    # to make the last choices before the test
    "validation": (date(2024, 1, 1), date(2024, 7, 1)),
}


def load_prices():
    prices = {}
    for name in ("2022", "2023", "2024"):
        raw = json.loads(Path(f"data/prices_{name}.json").read_text())
        prices.update(zip(raw["unix_seconds"], raw["price"]))
    return prices


def main():
    prices = load_prices()
    # trained once on 2022-2023
    forecasts = {"weekly": weekly_forecast, "ridge": train_ridge(prices)}

    for period, (start, end) in PERIODS.items():
        results = {}
        for cost in WEAR_COSTS:
            battery = Battery(wear_cost=cost)
            results[cost] = {name: backtest(prices, start, end, battery, f) for name, f in forecasts.items()}
            results[cost]["perfect forecast"] = perfect_forecast(prices, start, end, battery)

        print(f"\n{period}, {start} to {end - DAY} (MAE in EUR/MWh, gains in EUR)")
        print(f"{'':20}{'MAE':>8}" + "".join(f"{f'gain, wear {c}':>18}" for c in WEAR_COSTS))
        for name in results[WEAR_COSTS[0]]:
            gains = [summary(results[c][name], Battery(wear_cost=c))["gain"] for c in WEAR_COSTS]
            mae = summary(results[WEAR_COSTS[0]][name], Battery())["mae"]
            mae = "-" if name == "perfect forecast" else f"{mae:.2f}"
            print(f"{name:20}{mae:>8}" + "".join(f"{g:>18,.0f}" for g in gains))


if __name__ == "__main__":
    main()
