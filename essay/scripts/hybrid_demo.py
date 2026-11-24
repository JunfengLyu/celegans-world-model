"""Animate fixed held-out stochastic examples with identical observed histories."""

from pathlib import Path
import sys

import cmasher  # Registers the cmr maps.
import imageio_ffmpeg
import matplotlib as mpl

mpl.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.animation import FFMpegWriter, FuncAnimation
import numpy as np

ESSAY = Path(__file__).resolve().parents[1]
PROJECT = ESSAY.parent
sys.path.insert(0, str(PROJECT))
from wormlab.data import FPS, centerlines, read_basis


def main():
    with np.load(PROJECT / "artifacts" / "hybrid" / "example_rollouts.npz") as examples:
        history = examples["history"]
        real = examples["truth"]
        samples = examples["stochastic"][:, 0]
        worms = examples["worms"]
        seed = int(examples["seeds"][0])
    coefficients = [real, *samples]
    atlas = read_basis()
    paths = [centerlines(np.concatenate((history, body), axis=1), atlas) for body in coefficients]
    colors = ["#2b2c30", mpl.colormaps["cmr.ocean"](.35), mpl.colormaps["cmr.ember"](.68)]
    fig, axes = plt.subplots(2, 3, figsize=(8, 4.4), dpi=140)
    fig.subplots_adjust(left=.11, right=.98, bottom=.055, top=.85, wspace=.07, hspace=.10)
    headings = ["Observed", "GRU", "Phase-embedded GRU"]
    for i, ax in enumerate(axes[0]):
        ax.text(.5, 1.08, headings[i], transform=ax.transAxes, ha="center", fontsize=10, color=colors[i])
    lines, tips = [], []
    for animal, row in enumerate(axes):
        row[0].text(-.16, .5, f"Worm {worms[animal]}", transform=row[0].transAxes,
                    ha="right", va="center", fontsize=10, color="#2b2c30")
        for ax, color in zip(row, colors):
            ax.set(xlim=(-.54,.54), ylim=(-.54,.54), aspect="equal")
            ax.set_xticks([])
            ax.set_yticks([])
            for spine in ax.spines.values():
                spine.set_color("#e6e4e1")
                spine.set_linewidth(.6)
            lines.append(ax.plot([], [], color=color, lw=2.7, solid_capstyle="round")[0])
            tips.append(ax.plot([], [], marker="o", color=color, ms=3, linestyle="none")[0])
    clock = fig.text(.5, .975, "", ha="center", va="top", fontsize=10, color="#2b2c30")

    def update(frame):
        hlen = history.shape[1]
        time = (frame - hlen + 1) / FPS
        context = "Shared observed history" if frame < hlen else f"Autonomous rollout, seed {seed}"
        clock.set_text(f"{context}    {time:+.2f} s")
        for index, (line, tip) in enumerate(zip(lines, tips)):
            animal, model = divmod(index, 3)
            xy = paths[model][animal, frame]
            line.set_data(*xy.T)
            tip.set_data(*xy[:1].T)
        return [clock, *lines, *tips]

    mpl.rcParams["animation.ffmpeg_path"] = imageio_ffmpeg.get_ffmpeg_exe()
    clip = FuncAnimation(fig, update, frames=paths[0].shape[1], interval=1000/FPS)
    clip.save(ESSAY / "demos" / "hybrid-rollout.mp4", writer=FFMpegWriter(
        fps=FPS, codec="libx264", extra_args=["-pix_fmt", "yuv420p", "-movflags", "+faststart"]))
    plt.close(fig)


if __name__ == "__main__":
    main()
