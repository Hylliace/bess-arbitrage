import itertools
import unittest
from datetime import date, datetime, timedelta

import numpy as np

from backtest import backtest
from battery import Battery
from dispatch import optimal_dispatch
from forecast import PARIS, Ridge, features, weekly_forecast


def timestamp(day, hour=0):
    return int(datetime(day.year, day.month, day.day, hour, tzinfo=PARIS).timestamp())


def flat_prices(start, end, price=50.0):
    """Hourly prices from a week before start (for the forecasts) to end."""
    return {t: price for t in range(timestamp(start - timedelta(days=7)), timestamp(end), 3600)}


class BatteryTests(unittest.TestCase):
    def test_full_cycle(self):
        battery = Battery()
        stored, bought = battery.step(0, 1, 30)
        stored, sold = battery.step(stored, -0.9, 100)
        self.assertAlmostEqual(stored, 0)
        self.assertAlmostEqual(bought + sold, 60)

    def test_negative_prices(self):
        battery = Battery()
        stored, bought = battery.step(0, 1, -20)
        stored, sold = battery.step(stored, -0.9, -20)
        self.assertAlmostEqual(bought, 20)  # we get paid to buy
        self.assertAlmostEqual(sold, -18)  # and pay to sell
        self.assertAlmostEqual(stored, 0)

    def test_limits(self):
        for stored, q in [(0, -0.1), (2, 0.1), (0, 1.1)]:
            with self.assertRaises(ValueError):
                Battery().step(stored, q, 50)

    def test_wear(self):
        battery = Battery(1, 1, 0.81, 10)
        self.assertAlmostEqual(battery.wear(-0.9), 10)  # 0.9 sold = 1 MWh out of the battery
        self.assertEqual(battery.wear(1), 0)


class DispatchTests(unittest.TestCase):
    def test_buy_before_selling(self):
        battery = Battery(capacity=1, power=1, efficiency=1)
        self.assertAlmostEqual(optimal_dispatch([150, 30], battery, levels=2)[1], 0)
        self.assertAlmostEqual(optimal_dispatch([30, 150], battery, levels=2)[1], 120)

    def test_same_as_brute_force(self):
        battery = Battery(capacity=1, power=0.6, efficiency=0.81)
        prices = [-20, 70, 10, 120]
        best = -np.inf
        for middle in itertools.product([0, 0.5, 1], repeat=3):
            levels = [0, *middle, 0]
            stored, gain = 0.0, 0.0
            try:
                for t, price in enumerate(prices):
                    delta = levels[t + 1] - stored
                    stored, cash = battery.step(stored, delta / 0.9 if delta >= 0 else delta * 0.9, price)
                    gain += cash
            except ValueError:
                continue
            best = max(best, gain)
        self.assertAlmostEqual(optimal_dispatch(prices, battery, levels=2)[1], best)

    def test_no_charge_and_discharge_in_the_same_hour(self):
        orders, gain = optimal_dispatch([-100], Battery(), levels=20)
        self.assertAlmostEqual(gain, 0)
        np.testing.assert_allclose(orders, [0])

    def test_wear_can_make_a_cycle_not_worth_it(self):
        self.assertAlmostEqual(optimal_dispatch([30, 35], Battery(1, 1, 1, 0))[1], 5)
        np.testing.assert_allclose(optimal_dispatch([30, 35], Battery(1, 1, 1, 8))[0], [0, 0])

    def test_finer_grid_is_not_worse(self):
        prices = [35, -10, 40, 120, 5, 85]
        coarse = optimal_dispatch(prices, Battery(), levels=10)[1]
        fine = optimal_dispatch(prices, Battery(), levels=20)[1]
        self.assertGreaterEqual(fine + 1e-8, coarse)


class ForecastTests(unittest.TestCase):
    def test_no_future_prices(self):
        start = timestamp(date(2023, 2, 1))
        known = {start - i * 3600: float(i) for i in range(1, 169)}
        times = list(range(start, start + 48 * 3600, 3600))
        x = features(times, known)
        self.assertEqual(x.shape, (48, 35))
        self.assertEqual(x[0, 0], x[24, 0])  # day 2 gets the same recent prices as day 1
        with self.assertRaises(AssertionError):
            features(times, {**known, start: 999.0})

    def test_constant_inputs(self):
        x = np.ones((10, 5))
        np.testing.assert_allclose(Ridge(1.0).fit(x, np.full(10, 42.0)).predict(x), 42)

    def test_missing_spring_hour(self):
        known = {timestamp(date(2023, 3, 26), 1): 20, timestamp(date(2023, 3, 26), 3): 40}
        self.assertAlmostEqual(weekly_forecast([timestamp(date(2023, 4, 2), 2)], known)[0], 30)


class BacktestTests(unittest.TestCase):
    def test_orders_do_not_see_the_actual_prices(self):
        start, end = date(2023, 1, 8), date(2023, 1, 10)
        prices = {t: float((t // 3600) % 24 * 10) for t in flat_prices(start, end)}
        changed = {t: 10000 - p if t >= timestamp(start) else p for t, p in prices.items()}
        a = backtest(prices, start, end, Battery(), weekly_forecast, levels=10)
        b = backtest(changed, start, end, Battery(), weekly_forecast, levels=10)
        np.testing.assert_allclose([r["q"] for r in a[:24]], [r["q"] for r in b[:24]])

    def test_energy_can_stay_overnight(self):
        start, end = date(2023, 1, 8), date(2023, 1, 10)
        prices = flat_prices(start, end)
        prices[timestamp(date(2023, 1, 1), 23)] = -100
        prices[timestamp(date(2023, 1, 2), 7)] = 200
        rows = backtest(prices, start, end, Battery(), weekly_forecast, levels=10)
        self.assertGreater(rows[23]["stored"], 0)
        self.assertAlmostEqual(rows[-1]["stored"], 0)

    def test_clock_changes(self):
        for day, hours in ((date(2023, 3, 26), 23), (date(2023, 10, 29), 25)):
            rows = backtest(flat_prices(day, day + timedelta(days=1)), day, day + timedelta(days=1),
                            Battery(), weekly_forecast, levels=10)
            self.assertEqual(len(rows), hours)


if __name__ == "__main__":
    unittest.main()
