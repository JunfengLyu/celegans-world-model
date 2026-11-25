"""Animate fixed held-out stochastic examples with identical observed histories."""

from pathlib import Path
import sys
from tempfile import TemporaryDirectory

import cmasher  # Registers the cmr maps.
import imageio_ffmpeg
import matplotlib as mpl

mpl.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.animation import FFMpegWriter, FuncAnimation
import numpy as np

from worm_render import draw_worm, load_cuticle_profile

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
    colors = ["#252525", mpl.colormaps["cmr.iceburn"](.21), mpl.colormaps["cmr.iceburn"](.86)]
    profile = load_cuticle_profile()
    fig, axes = plt.subplots(2, 3, figsize=(8, 4.4), dpi=140)
    fig.subplots_adjust(left=.11, right=.98, bottom=.055, top=.85, wspace=.07, hspace=.10)
    headings = ["Observed", "GRU", "Phase-embedded GRU"]
    for i, ax in enumerate(axes[0]):
        ax.text(.5, 1.08, headings[i], transform=ax.transAxes, ha="center", fontsize=10, color=colors[i])
    bodies = []
    for animal, row in enumerate(axes):
        row[0].text(-.16, .5, f"Worm {worms[animal]}", transform=row[0].transAxes,
                    ha="right", va="center", fontsize=10, color="#2b2c30")
        for model, (ax, color) in enumerate(zip(row, colors)):
            ax.set(xlim=(-.54,.54), ylim=(-.54,.54), aspect="equal")
            ax.set_xticks([])
            ax.set_yticks([])
            for spine in ax.spines.values():
                spine.set_color("#e6e4e1")
                spine.set_linewidth(.6)
            bodies.append(draw_worm(ax, paths[model][animal, 0], profile, color, linewidth=.85))
    clock = fig.text(.5, .975, "", ha="center", va="top", fontsize=10, color="#2b2c30")

    def update(frame):
        hlen = history.shape[1]
        time = (frame - hlen + 1) / FPS
        context = "Shared observed history" if frame < hlen else f"Autonomous rollout, seed {seed}"
        clock.set_text(f"{context}    {time:+.2f} s")
        for index, body in enumerate(bodies):
            animal, model = divmod(index, 3)
            body.set_centerline(paths[model][animal, frame])
        return [clock, *(artist for body in bodies for artist in body.artists)]

    mpl.rcParams["animation.ffmpeg_path"] = imageio_ffmpeg.get_ffmpeg_exe()
    clip = FuncAnimation(fig, update, frames=paths[0].shape[1], interval=1000/FPS)
    destination = ESSAY / "demos" / "hybrid-rollout.mp4"
    with TemporaryDirectory(dir=destination.parent) as directory:
        temporary = Path(directory) / destination.name
        clip.save(temporary, writer=FFMpegWriter(
            fps=FPS, codec="libx264", extra_args=["-pix_fmt", "yuv420p", "-movflags", "+faststart"]))
        temporary.replace(destination)
    plt.close(fig)


if __name__ == "__main__":
    main()
