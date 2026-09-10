"""Download the French day-ahead prices from energy-charts.info and check them."""
import json
from pathlib import Path
from urllib.request import urlopen

PERIODS = {
    "2022": ("2022-01-01", "2022-12-31"),
    "2023": ("2023-01-01", "2023-12-31"),
    "2024": ("2024-01-01", "2024-12-31"),
}

for name, (start, end) in PERIODS.items():
    path = Path(f"data/prices_{name}.json")
    if not path.exists():
        url = f"https://api.energy-charts.info/price?bzn=FR&start={start}&end={end}"
        with urlopen(url, timeout=60) as response:
            path.write_bytes(response.read())
    data = json.loads(path.read_text())
    times, prices = data["unix_seconds"], data["price"]
    assert data["unit"] == "EUR / MWh"
    assert len(times) == len(prices) and None not in prices
    # one price per hour, no gap and no duplicate
    assert all(b - a == 3600 for a, b in zip(times, times[1:])), f"{path} is not hourly"
    print(f"{path}: {len(times)} hours")
