from datetime import datetime

import matplotlib
matplotlib.use("Agg")
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np

from forecast import PARIS

COLORS = {"weekly": "#315e91", "ridge": "#bc5528", "ridge + weekend": "#76518e",
          "perfect forecast": "#43816c"}


def local(t):
    return datetime.fromtimestamp(t, PARIS)


def cumulative_gains(results, title, path):
    """results[wear_cost][method] = rows of a backtest"""
    fig, axes = plt.subplots(len(results), 1, figsize=(10, 7), sharex=True, layout="constrained")
    for ax, (cost, methods) in zip(axes, results.items()):
        for method, rows in methods.items():
            times = [local(rows[0]["time"])] + [local(r["time"] + 3600) for r in rows]
            gain = np.cumsum([0] + [r["cash"] - r["wear"] for r in rows])
            ax.plot(times, gain, color=COLORS[method], label=f"{method}: {gain[-1]:,.0f} €")
        ax.axhline(0, color="gray", lw=0.6)
        ax.set_title(f"wear cost {cost} €/MWh", loc="left", fontsize=11)
        ax.set_ylabel("cumulative gain (€)")
        ax.grid(alpha=0.2)
        ax.legend(frameon=False)
    axes[-1].xaxis.set_major_locator(mdates.MonthLocator(tz=PARIS))
    axes[-1].xaxis.set_major_formatter(mdates.DateFormatter("%b %Y", tz=PARIS))
    fig.suptitle(title)
    fig.savefig(path, dpi=120)
    plt.close(fig)
