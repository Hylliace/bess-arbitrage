import itertools
import unittest

import numpy as np

from battery import Battery
from dispatch import optimal_dispatch


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


if __name__ == "__main__":
    unittest.main()
