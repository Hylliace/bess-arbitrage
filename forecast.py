from collections import defaultdict
from datetime import datetime, timedelta
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
