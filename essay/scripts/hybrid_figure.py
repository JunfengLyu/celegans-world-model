"""Render the paired phase-embedding comparison from the held-out metrics."""

import argparse
import csv
from pathlib import Path

import cmasher as cmr
import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import FixedFormatter, FixedLocator, MaxNLocator, NullLocator

ROOT = Path(__file__).resolve().parents[2]
ARMS = ("GRU", "Phase-embedded GRU")
SEEDS = (17, 29, 43)
FPS = 16
HORIZONS = np.array((1, 4, 8, 16, 32, 64)) / FPS
COLORS = (cmr.iceburn(0.21), cmr.iceburn(0.86))
STATISTICS = ("radius_wasserstein", "signed_phase_speed_wasserstein_cycles_per_s")


def load_scores(directory, worms):
    curves = {(arm, worm): np.empty((len(SEEDS), len(HORIZONS))) for arm in ARMS for worm in worms}
    for curve in curves.values():
        curve.fill(np.nan)
    horizon_index = {horizon: index for index, horizon in enumerate(HORIZONS)}
    with (directory / "metrics_per_worm.csv").open(newline="") as input_file:
        for row in csv.DictReader(input_file):
            key = (row["model"], int(row["worm"]))
            index = (SEEDS.index(int(row["seed"])), horizon_index[float(row["seconds"])])
            curves[key][index] = float(row["rmse_z"])
    return curves


def load_distances(directory):
    distances = {}
    with (directory / "rollout_statistics.csv").open(newline="") as input_file:
        for row in csv.DictReader(input_file):
            if row["rollout"] == "stochastic":
                key = (int(row["worm"]), int(row["seed"]), row["arm"])
                distances[key] = [float(row[field]) for field in STATISTICS]
    return distances


def render(directory, output):
    distances = load_distances(directory)
    worms = sorted({key[0] for key in distances})
    curves = load_scores(directory, worms)
    plt.rcParams.update({
        "font.family": "DejaVu Sans", "font.size": 10, "axes.linewidth": 0.65,
        "axes.spines.top": False, "axes.spines.right": False,
        "xtick.major.width": 0.6, "ytick.major.width": 0.6,
        "xtick.major.size": 2.5, "ytick.major.size": 2.5,
        "xtick.major.pad": 2, "ytick.major.pad": 2,
        "pdf.fonttype": 42,
    })
    fig, axes = plt.subplots(2, 2, figsize=(6.85, 3.02))
    fig.subplots_adjust(left=0.054, right=0.99, bottom=0.083, top=0.93, wspace=0.22, hspace=0.44)
    maximum_rmse = max(curve.max() for curve in curves.values())

    for panel, worm in zip(axes[0], worms):
        for arm, color, marker in zip(ARMS, COLORS, ("o", "s")):
            samples = curves[(arm, worm)]
            panel.fill_between(HORIZONS, samples.min(axis=0), samples.max(axis=0),
                               color=color, alpha=0.14, linewidth=0)
            panel.plot(HORIZONS, samples.mean(axis=0), color=color, marker=marker,
                       markersize=3.0, linewidth=1.2)
        panel.set_xscale("log", base=2)
        panel.set_xlim(0.056, 4.5)
        panel.set_ylim(0, maximum_rmse * 1.05)
        panel.xaxis.set_major_locator(FixedLocator(HORIZONS))
        panel.xaxis.set_major_formatter(FixedFormatter(("0.063", "0.25", "0.5", "1", "2", "4")))
        panel.xaxis.set_minor_locator(NullLocator())
        panel.yaxis.set_major_locator(MaxNLocator(4))

    for measure, panel in enumerate(axes[1]):
        for group, worm in enumerate(worms):
            for jitter, seed in zip((-0.025, 0.0, 0.025), SEEDS):
                x = (group + jitter - 0.16, group + jitter + 0.16)
                y = [distances[(worm, seed, arm)][measure] for arm in ARMS]
                panel.plot(x, y, color="0.68", linewidth=0.75, zorder=1)
                for position, value, color, marker in zip(x, y, COLORS, ("o", "s")):
                    panel.plot(position, value, color=color, marker=marker, markersize=3.8, zorder=2)
        panel.set_xlim(-0.4, len(worms) - 0.6)
        panel.set_xticks(range(len(worms)), [str(worm) for worm in worms])
        panel.yaxis.set_major_locator(MaxNLocator(4))

    for panel, letter in zip(axes.flat, "ABCD"):
        panel.annotate(letter, (0, 1), xycoords="axes fraction", xytext=(-1, 2),
                       textcoords="offset points", va="bottom", ha="left",
                       fontsize=11, fontweight="normal")
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=ROOT / "artifacts" / "hybrid")
    parser.add_argument("--out", type=Path, default=ROOT / "essay" / "figures" / "hybrid.pdf")
    args = parser.parse_args()
    render(args.data, args.out)


if __name__ == "__main__":
    main()
