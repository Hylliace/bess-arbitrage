"""Backtest a 2 MWh / 1 MW battery on French day-ahead prices."""
import json
from datetime import date
from pathlib import Path

from backtest import backtest, perfect_forecast, summary
from battery import Battery
from forecast import weekly_forecast

WEAR_COSTS = (25, 40)  # EUR per MWh taken out of the battery


def load_prices():
    prices = {}
    for name in ("2022", "2023", "2024"):
        raw = json.loads(Path(f"data/prices_{name}.json").read_text())
        prices.update(zip(raw["unix_seconds"], raw["price"]))
    return prices


def main():
    prices = load_prices()
    start, end = date(2023, 1, 1), date(2024, 1, 1)
    for cost in WEAR_COSTS:
        battery = Battery(wear_cost=cost)
        weekly = summary(backtest(prices, start, end, battery, weekly_forecast), battery)
        perfect = summary(perfect_forecast(prices, start, end, battery), battery)
        print(f"2023, wear {cost}: weekly {weekly['gain']:,.0f} EUR (MAE {weekly['mae']:.2f}), "
              f"perfect forecast {perfect['gain']:,.0f} EUR")


if __name__ == "__main__":
    main()
