"""Render one observed worm and its synchronized first-pair phase trajectory."""

import argparse
from pathlib import Path

import imageio_ffmpeg
from matplotlib.animation import FFMpegWriter
from matplotlib.collections import LineCollection
from matplotlib.colors import LogNorm
from matplotlib.patches import Circle
import numpy as np

from behavior_figure import ESSAY, example_contours, plt, style_axis
from worm_render import draw_worm, load_cuticle_profile
from wormlab.data import FPS, SPLIT, centerlines, read_basis, read_worms


def make_video(destination, seconds):
    worms = read_worms()
    basis = read_basis()
    individual = SPLIT["train"][0]
    recordings = worms[individual]
    start, *_ = example_contours(recordings, basis)
    target_frames = round(seconds * FPS)
    coefficients = recordings[start:start + target_frames]
    gaps = np.flatnonzero(~np.isfinite(coefficients).all(axis=1))
    if gaps.size:
        coefficients = coefficients[:gaps[0]]
    if len(coefficients) < 25:
        raise ValueError("No sufficiently long, contiguous clip at the example frame.")
    bodies = centerlines(coefficients, basis)
    reduced = centerlines(coefficients[:, :2], basis[:, :2])
    phase_path = coefficients[:, :2]
    training = np.concatenate([worms[index] for index in SPLIT["train"]])
    training = training[np.isfinite(training).all(axis=1)]
    radius = float(np.median(np.linalg.norm(training[:, :2], axis=1)))

    plt.rcParams.update({
        "font.family": "STIXGeneral", "mathtext.fontset": "stix",
        "font.size": 18, "axes.labelsize": 19,
        "animation.ffmpeg_path": imageio_ffmpeg.get_ffmpeg_exe(),
    })
    fig, (body, phase) = plt.subplots(1, 2, figsize=(10.4, 4.2), dpi=120)
    fig.subplots_adjust(left=0.07, right=0.96, top=0.89, bottom=0.19, wspace=0.42)
    color = plt.colormaps["cmr.cosmic"](0.59)

    body.set(xlim=(-0.59, 0.59), ylim=(-0.38, 0.38), aspect="equal")
    body.set_axis_off()
    body_cuticle = draw_worm(body, bodies[0], load_cuticle_profile(), color, linewidth=1.05)
    two_mode_line, = body.plot([], [], c="#272727", lw=1.25, ls=(0, (3, 2)), zorder=4)
    endpoint, = body.plot([], [], c=color, marker="o", ms=5, linestyle="none", zorder=5)
    body.plot([-0.51, -0.26], [-0.30, -0.30], c="#3e3e3e", lw=1.6)
    body.text(-0.23, -0.30, "$0.25L$", fontsize=16, va="center", ha="left")
    time_label = fig.text(0.10, 0.82, "", va="center")

    limits = max(10.5, float(np.ceil(np.abs(phase_path).max() + 1.1)))
    bins = np.linspace(-limits, limits, 75)
    counts, _, _ = np.histogram2d(training[:, 0], training[:, 1], bins=(bins, bins))
    density = counts.T / counts.max()
    phase.imshow(np.ma.masked_less(density, 0.01), origin="lower",
                 extent=(-limits, limits, -limits, limits),
                 cmap="cmr.ember_r", norm=LogNorm(0.01, 1), alpha=0.31)
    phase.add_patch(Circle((0, 0), radius, fill=False, ls=(0, (3, 3)), lw=1.1, color="#777777"))
    phase.set(xlim=(-limits, limits), ylim=(-limits, limits),
              xlabel="$a_1$", ylabel="$a_2$", xticks=(-8, 0, 8), yticks=(-8, 0, 8), aspect="equal")
    style_axis(phase)
    trail = LineCollection([], linewidths=2.8, capstyle="round")
    phase.add_collection(trail)
    moving, = phase.plot([], [], "o", c=color, markeredgecolor="white", markeredgewidth=1.4, ms=10)

    destination.parent.mkdir(parents=True, exist_ok=True)
    writer = FFMpegWriter(fps=FPS, codec="libx264", extra_args=["-pix_fmt", "yuv420p", "-movflags", "+faststart"])
    with writer.saving(fig, str(destination), dpi=120):
        for frame, (xy, reduced_xy, value) in enumerate(zip(bodies, reduced, phase_path)):
            body_cuticle.set_centerline(xy)
            two_mode_line.set_data(reduced_xy.T)
            endpoint.set_data(xy[:1].T)
            moving.set_data(value[None].T)
            recent = phase_path[max(0, frame - FPS):frame + 1]
            trail.set_segments(np.stack((recent[:-1], recent[1:]), axis=1))
            colors = np.tile(color, (len(recent) - 1, 1))
            colors[:, 3] = np.linspace(0.12, 0.94, len(recent) - 1)
            trail.set_color(colors)
            time_label.set_text(f"{frame / FPS:.2f} s")
            writer.grab_frame()
    plt.close(fig)
    print(f"{destination} | animal {individual}, frames {start}:{start + len(coefficients)} at {FPS} Hz")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=ESSAY / "demos" / "eigenworm-phase.mp4")
    parser.add_argument("--seconds", type=float, default=8)
    args = parser.parse_args()
    if not 1.6 <= args.seconds <= 30:
        parser.error("--seconds must be in [1.6, 30]")
    make_video(args.output, args.seconds)


if __name__ == "__main__":
    main()
