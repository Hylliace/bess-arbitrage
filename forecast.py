from collections import defaultdict
from datetime import date, datetime, timedelta
from zoneinfo import ZoneInfo

import numpy as np

PARIS = ZoneInfo("Europe/Paris")
DAY = timedelta(days=1)


def midnight(day):
    return int(datetime.combine(day, datetime.min.time(), PARIS).timestamp())


def by_local_hour(prices):
    hours = defaultdict(list)
    for t, p in prices.items():
        local = datetime.fromtimestamp(t, PARIS)
        hours[local.date(), local.hour].append(p)
    return hours


def lookup(hours, day, hour):
    # In October hour 2 happens twice, we take the mean. In March it does not
    # exist, we take the mean of hours 1 and 3.
    if (day, hour) in hours:
        return np.mean(hours[day, hour])
    return (np.mean(hours[day, hour - 1]) + np.mean(hours[day, hour + 1])) / 2


def weekly_forecast(times, known):
    """Baseline: same local hour, one week before."""
    hours = by_local_hour(known)
    forecast = []
    for t in times:
        local = datetime.fromtimestamp(t, PARIS)
        forecast.append(lookup(hours, local.date() - 7 * DAY, local.hour))
    return np.array(forecast)


def features(times, known):
    """One row per target hour, for a plan starting on the day of times[0].

    Only prices from before that day can be used (`known`). The second day of
    the plan gets the same two recent prices as the first one.
    """
    hours = by_local_hour(known)
    first = datetime.fromtimestamp(times[0], PARIS).date()
    assert max(day for day, _ in hours) < first, "future prices in the inputs"
    rows = []
    for t in times:
        local = datetime.fromtimestamp(t, PARIS)
        day, hour = local.date(), local.hour
        horizon = (day - first).days
        assert horizon in (0, 1)
        lags = [lookup(hours, first - DAY, hour),
                lookup(hours, first - 2 * DAY, hour),
                lookup(hours, day - 7 * DAY, hour)]
        row = (lags
               + [float(hour == h) for h in range(24)]
               + [float(day.weekday() == d) for d in range(7)]
               + [float(horizon)])
        rows.append(row)
    return np.array(rows)


class Ridge:
    def __init__(self, alpha):
        self.alpha = alpha

    def fit(self, x, y):
        self.mean = x.mean(axis=0)
        self.std = x.std(axis=0)
        self.std[self.std < 1e-12] = 1  # constant columns
        z = (x - self.mean) / self.std
        self.intercept = y.mean()
        a = z.T @ z + self.alpha * np.eye(z.shape[1])
        self.coef = np.linalg.solve(a, z.T @ (y - self.intercept))
        return self

    def predict(self, x):
        return self.intercept + ((x - self.mean) / self.std) @ self.coef

    def __call__(self, times, known):
        return self.predict(features(times, known))


def training_set(prices, end):
    """One two-day plan per day from 2022-01-09, with all targets before `end`."""
    xs, ys, times = [], [], []
    day = date(2022, 1, 9)
    while day < end - DAY:
        targets = list(range(midnight(day), midnight(day + 2 * DAY), 3600))
        known = {t: prices[t] for t in range(midnight(day - 7 * DAY), midnight(day), 3600)}
        xs.extend(features(targets, known))
        ys.extend(prices[t] for t in targets)
        times.extend(targets)
        day += DAY
    return np.array(xs), np.array(ys), np.array(times)


def train_ridge(prices, verbose=True):
    """Pick alpha on three windows of 2022, then fit on 2022-2023."""
    x, y, times = training_set(prices, date(2023, 1, 1))
    windows = [(date(2022, 7, 1), date(2022, 9, 1)),
               (date(2022, 9, 1), date(2022, 11, 1)),
               (date(2022, 11, 1), date(2023, 1, 1))]
    scores = {}
    for alpha in (0.1, 1, 10, 100, 1000):
        errors = []
        for start, end in windows:
            # train only on what was known before the window
            train = times < midnight(start)
            valid = (times >= midnight(start)) & (times < midnight(end))
            model = Ridge(alpha).fit(x[train], y[train])
            errors.append(np.mean(np.abs(model.predict(x[valid]) - y[valid])))
        scores[alpha] = np.mean(errors)
    alpha = min(scores, key=scores.get)
    if verbose:
        print(f"ridge: alpha = {alpha:g}  "
              f"(MAE on 2022: {', '.join(f'{a:g} -> {s:.2f}' for a, s in scores.items())})")
    x, y, _ = training_set(prices, date(2024, 1, 1))
    return Ridge(alpha).fit(x, y)
