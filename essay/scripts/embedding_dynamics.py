"""Draw fixed held-out stochastic examples for both recurrent posture models."""

import argparse
from pathlib import Path
import sys

import cmasher as cmr
import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

from worm_render import draw_worm, load_cuticle_profile

ESSAY = Path(__file__).resolve().parents[1]
PROJECT = ESSAY.parent
sys.path.insert(0, str(PROJECT))
from wormlab.data import FPS, centerlines, read_basis

COLORS = ("#252525", cmr.iceburn(0.21), cmr.iceburn(0.86))
SNAPSHOTS = np.array((8, 16, 32, 64))
SEED = 17
PHASE_RADIUS_FLOOR = 0.05


def load_examples(path):
    with np.load(path) as examples:
        seed_index = int(np.flatnonzero(examples["seeds"] == SEED)[0])
        futures = np.concatenate((examples["truth"][None], examples["stochastic"][:, seed_index]))
        last_observation = np.broadcast_to(examples["history"][:, -1:], futures[:, :, :1].shape)
        trajectories = np.concatenate((last_observation, futures), axis=2)
    return trajectories


def panel_letter(axis, letter):
    axis.annotate(letter, (0, 1), xycoords="axes fraction", xytext=(-1, 3),
                  textcoords="offset points", va="bottom", ha="left",
                  fontsize=11, fontweight="normal")


def draw_posture(axis, shapes, profile, letter, show_scale=False):
    scale = 0.85
    row_centers = (1.02, 0.51, 0.0)
    for mode, (height, color) in enumerate(zip(row_centers, COLORS)):
        for column, contour in enumerate(shapes[mode]):
            xy = contour * scale + (column, height)
            draw_worm(axis, xy, profile, color, linewidth=.43)
    for height in (0.255, 0.765):
        axis.plot((-.43, 3.44), (height, height), color="0.9", lw=.55)
    for column, frame in enumerate(SNAPSHOTS):
        axis.text(column, 1.29, f"{frame / FPS:g}", ha="center", va="bottom")
    if show_scale:
        axis.plot((3.40 - 0.25 * scale, 3.40), (-.25, -.25), color="#252525",
                  lw=1.2, solid_capstyle="butt")
    axis.set(xlim=(-.45, 3.46), ylim=(-.28, 1.55), aspect="equal")
    axis.set_anchor("W")
    axis.axis("off")
    panel_letter(axis, letter)


def render(data, output):
    trajectories = load_examples(data)
    points = centerlines(trajectories[:, :, SNAPSHOTS], read_basis())
    profile = load_cuticle_profile()
    time = np.arange(trajectories.shape[2]) / FPS
    modes = trajectories[..., 0] + 1j * trajectories[..., 1]
    radius = np.abs(modes)
    if np.any(radius < PHASE_RADIUS_FLOOR):
        raise ValueError("A selected example has frames below the radius floor for phase analysis.")
    phase = np.unwrap(np.angle(modes), axis=-1) / (2 * np.pi)
    phase -= phase[..., :1]
    plt.rcParams.update({
        "font.family": "DejaVu Sans", "font.size": 10, "axes.linewidth": .65,
        "axes.spines.top": False, "axes.spines.right": False,
        "xtick.major.width": .6, "ytick.major.width": .6,
        "xtick.major.size": 2.5, "ytick.major.size": 2.5,
        "xtick.major.pad": 2, "ytick.major.pad": 2, "pdf.fonttype": 42,
    })
    fig = plt.figure(figsize=(6.85, 3.3))
    outer = fig.add_gridspec(2, 2, height_ratios=(1.25, 1), hspace=.29, wspace=.18,
                            left=.109, right=.975, bottom=.16, top=.845)
    palette = [Line2D([], [], color=color, lw=1.7, label=label)
               for color, label in zip(COLORS, ("Observed", "GRU", "Phase-embedded GRU"))]
    fig.legend(handles=palette, loc="upper center", ncols=3, frameon=False,
               handlelength=1.3, columnspacing=1.2, handletextpad=.45, borderaxespad=.25)
    fig.supxlabel("Time (s)", fontsize=10, y=.035)
    for animal in range(2):
        posture = fig.add_subplot(outer[0, animal])
        draw_posture(posture, points[:, animal], profile, "AB"[animal], show_scale=animal == 1)
        series = outer[1, animal].subgridspec(1, 2, wspace=.66)
        rad_axis, phase_axis = [fig.add_subplot(series[0, i]) for i in range(2)]
        for ax in (rad_axis, phase_axis):
            ax.set(xlim=(0., 4.0), xticks=(0., 2., 4.))
        rad_axis.set(ylim=(0.0, 11.2), yticks=(0, 5, 10))
        phase_axis.set(ylim=(-2.55, .23), yticks=(-2, -1, 0))
        rad_axis.set_ylabel("Bend radius", labelpad=2)
        phase_axis.set_ylabel("Phase (cycles)", labelpad=2)
        for source, color in enumerate(COLORS):
            rad_axis.plot(time, radius[source, animal], color=color, lw=1.15)
            phase_axis.plot(time, phase[source, animal], color=color, lw=1.15)
        panel_letter(rad_axis, "CD"[animal])
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=PROJECT / "artifacts" / "hybrid" / "example_rollouts.npz")
    parser.add_argument("--out", type=Path, default=ESSAY / "figures" / "embedding-dynamics.pdf")
    args = parser.parse_args()
    render(args.data, args.out)


if __name__ == "__main__":
    main()
