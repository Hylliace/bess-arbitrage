from dataclasses import dataclass
from math import sqrt


@dataclass(frozen=True)
class Battery:
    capacity: float = 2.0  # MWh
    power: float = 1.0  # MW
    efficiency: float = 0.9  # round trip
    wear_cost: float = 0.0  # EUR per MWh taken out of the battery

    @property
    def eta(self):
        # the round trip loss is split evenly between charging and discharging
        return sqrt(self.efficiency)

    def wear(self, q):
        """Wear cost of an hour where we exchange q MWh with the grid (q > 0 buys, q < 0 sells)."""
        return self.wear_cost * max(-q, 0) / self.eta

    def step(self, stored, q, price):
        """Exchange q MWh during one hour. Returns the new stored energy and the cash flow."""
        tol = 1e-10
        if abs(q) > self.power + tol:
            raise ValueError("exchange above the power limit")
        stored = stored + (q * self.eta if q >= 0 else q / self.eta)
        if stored < -tol or stored > self.capacity + tol:
            raise ValueError("stored energy out of bounds")
        return min(self.capacity, max(0.0, stored)), -price * q
