import numpy as np

from dispatch import optimal_dispatch
from forecast import DAY, midnight


def run(times, orders, prices, battery, forecast, stored=0.0):
    rows = []
    for t, q, f in zip(times, orders, forecast):
        stored, cash = battery.step(stored, q, prices[t])
        rows.append(dict(time=t, price=prices[t], forecast=f, q=q, stored=stored,
                         cash=cash, wear=battery.wear(q)))
    return rows


def backtest(prices, start, end, battery, forecast, levels=100):
    """Simulate the battery from `start` to `end` (dates, end excluded).

    The orders for day d are sent on d-1. At that point the day-ahead prices
    of d-1 are known but not those of d. Every day we forecast d and d+1, plan
    both days and only keep the orders for d. The battery is empty at the start
    and at the end, but it can keep energy overnight.
    """
    rows, stored, day = [], 0.0, start
    while day < end:
        known = {t: prices[t] for t in range(midnight(day - 7 * DAY), midnight(day), 3600)}
        horizon = list(range(midnight(day), midnight(min(end, day + 2 * DAY)), 3600))
        predicted = forecast(horizon, known)
        orders, _ = optimal_dispatch(predicted, battery, start=stored, end=0.0, levels=levels)
        today = range(midnight(day), midnight(day + DAY), 3600)
        rows += run(today, orders, prices, battery, predicted, stored)
        stored = rows[-1]["stored"]
        day += DAY
    assert np.isclose(stored, 0)
    return rows


def perfect_forecast(prices, start, end, battery, levels=100):
    """Same battery, but with all the prices of the period known in advance.
    Not doable in practice, it gives the best we could hope for."""
    times = range(midnight(start), midnight(end), 3600)
    actual = [prices[t] for t in times]
    orders, _ = optimal_dispatch(actual, battery, levels=levels)
    return run(times, orders, prices, battery, actual)


def summary(rows, battery):
    cash = sum(r["cash"] for r in rows)
    wear = sum(r["wear"] for r in rows)
    discharged = sum(-r["q"] / battery.eta for r in rows if r["q"] < 0)
    return dict(gain=cash - wear, cash=cash, wear=wear,
                mae=np.mean([abs(r["forecast"] - r["price"]) for r in rows]),
                cycles=discharged / battery.capacity)
