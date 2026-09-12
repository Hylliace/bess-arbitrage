import numpy as np


def optimal_dispatch(prices, battery, start=0.0, end=0.0, levels=100):
    """Best plan for a list of hourly prices, assuming they are right.

    Backward dynamic programming (Bellman) with the stored energy on a grid
    of levels + 1 values between 0 and the capacity. The battery starts at
    `start` and has to finish at `end`, both on the grid.
    Returns the hourly exchanges with the grid (MWh) and the planned gain.
    """
    prices = np.asarray(prices, dtype=float)
    grid = np.linspace(0, battery.capacity, levels + 1)
    step = battery.capacity / levels
    first, last = round(start / step), round(end / step)
    assert np.isclose(grid[first], start) and np.isclose(grid[last], end), "start or end not on the grid"

    # q[i, j] is the exchange needed to go from grid[i] to grid[j] in one hour
    delta = grid[None, :] - grid[:, None]
    q = np.where(delta >= 0, delta / battery.eta, delta * battery.eta)
    allowed = np.abs(q) <= battery.power + 1e-10
    wear = battery.wear_cost * np.maximum(-delta, 0)

    value = np.full(levels + 1, -np.inf)
    value[last] = 0.0
    best = np.empty((len(prices), levels + 1), dtype=int)
    for t in range(len(prices) - 1, -1, -1):
        gain = -prices[t] * q - wear + value[None, :]
        gain[~allowed] = -np.inf
        best[t] = np.argmax(gain, axis=1)
        value = gain[np.arange(levels + 1), best[t]]
    assert np.isfinite(value[first]), "cannot reach the final level in time"

    orders, i = np.empty(len(prices)), first
    for t in range(len(prices)):
        orders[t] = q[i, best[t, i]]
        i = best[t, i]
    return orders, value[first]
