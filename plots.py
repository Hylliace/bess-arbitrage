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


def example(rows, title, path):
    """Prices, orders and stored energy hour by hour. The rows have to start
    with an empty battery."""
    edges = [local(r["time"]) for r in rows] + [local(rows[-1]["time"] + 3600)]
    fig, axes = plt.subplots(3, 1, figsize=(10, 7), sharex=True, layout="constrained")

    for key, label in (("price", "actual price"), ("forecast", "forecast from the day before")):
        values = [r[key] for r in rows]
        axes[0].step(edges, values + values[-1:], where="post", label=label,
                     color=COLORS["weekly" if key == "price" else "ridge"])
    axes[0].set_ylabel("€/MWh")
    axes[0].legend(frameon=False)

    q = [r["q"] for r in rows]
    axes[1].bar(edges[:-1], q, width=1 / 24, align="edge",
                color=["#548973" if x >= 0 else "#bc5528" for x in q])
    axes[1].set_ylabel("bought (+) / sold (-)\nMWh")

    axes[2].plot(edges, [0.0] + [r["stored"] for r in rows], color="#416857")
    axes[2].set_ylabel("stored energy\nMWh")
    axes[2].set_ylim(-0.1, 2.1)

    for ax in axes:
        ax.grid(alpha=0.2)
        ax.axhline(0, color="gray", lw=0.6)
    axes[-1].xaxis.set_major_locator(mdates.HourLocator(byhour=(0, 12), tz=PARIS))
    axes[-1].xaxis.set_major_formatter(mdates.DateFormatter("%d/%m %Hh", tz=PARIS))
    fig.suptitle(title)
    fig.savefig(path, dpi=120)
    plt.close(fig)
