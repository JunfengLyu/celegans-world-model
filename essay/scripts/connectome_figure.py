"""Draw registered Cook-model anatomy and matched touch and sham body paths."""

import argparse
import json
import sys
from pathlib import Path

import cmasher  # Registers the cmr maps.
import imageio_ffmpeg
import matplotlib as mpl

mpl.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.animation import FFMpegWriter, FuncAnimation
from matplotlib.collections import LineCollection
from matplotlib.colors import Normalize, SymLogNorm
from matplotlib.lines import Line2D
from matplotlib.ticker import FixedLocator
import numpy as np

ESSAY = Path(__file__).resolve().parents[1]
PROJECT = ESSAY.parent
sys.path.insert(0, str(PROJECT))
from wormlab.connectome import read_cook, reorder

INK = "#242731"
TRACE = mpl.colormaps["cmr.rainforest"]
ACTIVITY = mpl.colormaps["cmr.fusion_r"]
LIMIT = SymLogNorm(linthresh=0.1, linscale=0.6, vmin=-50, vmax=50)
TIME_LEVELS = (0.50, 2.50, 13.99)
TRACE_NORM = Normalize(0, 14)

mpl.rcParams.update({
    "font.family": "DejaVu Sans", "font.size": 8.5,
    "axes.labelsize": 8.5, "xtick.labelsize": 8, "ytick.labelsize": 8,
    "axes.linewidth": 0.55, "axes.spines.top": False, "axes.spines.right": False,
    "xtick.major.width": 0.55, "ytick.major.width": 0.55,
    "xtick.major.size": 2.3, "ytick.major.size": 2.3,
    "pdf.fonttype": 42, "animation.ffmpeg_path": imageio_ffmpeg.get_ffmpeg_exe(),
})


def load():
    directory = PROJECT / "artifacts" / "virtual"
    runs = []
    for name in ("cook_posterior", "cook_sham"):
        with np.load(directory / f"{name}.npz") as src:
            runs.append({field: src[field] for field in ("time", "xy", "voltage", "neurons", "stimulus")})
    touch, sham = runs
    np.testing.assert_array_equal(touch["neurons"], sham["neurons"])
    np.testing.assert_array_equal(touch["time"], sham["time"])
    anatomy = json.loads((ESSAY / "data" / "virtual-worm-atlas.json").read_text())
    assert anatomy["names"] == touch["neurons"].tolist()
    anatomy["soma"] = np.array(anatomy["soma"])[:, :2]
    anatomy["outline"] = np.array(anatomy["outline"])[:, :2]
    anatomy["neurites"] = [np.array(path).reshape(-1, 3)[:, :2]
                           for cell in anatomy["neurites"] for path in cell]
    atlas = read_cook()
    adjustment = PROJECT / "external" / "modWorm" / "modWorm" / "data" / "conn_gap_adjust_Cook.npy"
    gap = reorder(atlas["gap"], anatomy["names"]) + np.load(adjustment)
    plm = [anatomy["names"].index(name) for name in ("PLML", "PLMR")]
    incident = {tuple(sorted((i, int(j)))) for i in plm for j in np.flatnonzero(gap[i]) if i != j}
    anatomy["edges"] = sorted(incident)
    anatomy["plm"] = plm
    return anatomy, touch, sham


def mark(ax, label, x=0.025, y=0.96):
    ax.text(x, y, label, transform=ax.transAxes, ha="left", va="top", fontsize=10, color=INK)


def atlas_panel(ax, anatomy, label=None):
    for key, points in anatomy["landmarkOutlines"].items():
        xy = np.asarray(points)[:, :2]
        fill = "#e6ebec" if key in ("pharynx", "intestine") else "#f1f1ee"
        ax.fill(*xy.T, color=fill, ec="#a8a9a5", lw=0.42, zorder=1)
    ax.fill(*anatomy["outline"].T, color="#f7f7f4", ec="#858b8e", lw=0.58, zorder=0)
    ax.add_collection(LineCollection(anatomy["neurites"], colors="#5e697a", linewidths=0.30,
                                     alpha=0.32, zorder=2))
    locations = anatomy["soma"]
    gap_paths = [[locations[i], locations[j]] for i, j in anatomy["edges"]]
    ax.add_collection(LineCollection(gap_paths, colors="#4e725a", linewidths=0.64,
                                     linestyles="--", alpha=0.90, zorder=3))
    scatter = ax.scatter(*locations.T, c=np.zeros(len(locations)), cmap=ACTIVITY, norm=LIMIT,
                         s=4.2, edgecolors="#55505c", linewidths=0.14, zorder=4)
    ax.scatter(*locations[anatomy["plm"]].T, facecolors="none", edgecolors=INK, linewidths=0.65,
               s=23, zorder=5)
    length = 100 / anatomy["source"]["lengthMicrometers"]
    ax.plot([.027, .027 + length], [-.054, -.054], c=INK, lw=.72, solid_capstyle="butt")
    ax.set(xlim=(-.005, 1.005), ylim=(-.062, .079))
    ax.axis("off")
    if label:
        mark(ax, label, x=.009, y=1.0)
    return scatter


def centerline_outline(points, atlas_outline):
    segments = np.diff(points, axis=0)
    arclength = np.concatenate(([0.0], np.cumsum(np.linalg.norm(segments, axis=1))))
    length = arclength[-1]
    x = atlas_outline[:, 0] * length
    position = np.column_stack([np.interp(x, arclength, vector) for vector in points.T])
    direction = np.gradient(points, axis=0)
    normal = np.column_stack((-direction[:, 1], direction[:, 0]))
    normal /= np.maximum(np.linalg.norm(normal, axis=1, keepdims=True), 1e-10)
    n = np.column_stack([np.interp(x, arclength, vector) for vector in normal.T])
    return position + n * (atlas_outline[:, 1] * length)[:, None]


def body_panel(ax, all_xy, label=None):
    bounds = np.concatenate([points.reshape(-1, 2) for points in all_xy])
    bottom = (np.min(bounds, axis=0) - 2.4)
    top = (np.max(bounds, axis=0) + 2.4)
    ax.set(xlim=(bottom[0], top[0]), ylim=(bottom[1], top[1]), aspect="equal")
    ax.xaxis.set_major_locator(FixedLocator([-40, -20, 0, 20, 40]))
    ax.yaxis.set_major_locator(FixedLocator([-20, 0, 20]))
    ax.grid(color="#dce0e5", linewidth=0.48, zorder=0)
    ax.spines[["left", "bottom"]].set_color("#9499a1")
    if label:
        mark(ax, label, x=.026, y=.96)


def body_pose(ax, xy, atlas_outline, color, alpha, linewidth):
    outline = centerline_outline(xy, atlas_outline)
    return ax.fill(*outline.T, facecolor=color, edgecolor=color, alpha=alpha, lw=linewidth, zorder=4)[0]


def write_figure(anatomy, touch, sham):
    times = touch["time"]
    dv = touch["voltage"] - sham["voltage"]
    paths = [case["xy"] for case in (touch, sham)]
    fig = plt.figure(figsize=(6.85, 3.02))
    grid = fig.add_gridspec(3, 2, height_ratios=[.89, .085, 1.55], hspace=.25, wspace=.25,
                           left=.104, right=.985, top=.955, bottom=.123)
    for column, (t, label) in enumerate(zip(TIME_LEVELS[:2], "AB")):
        ax = fig.add_subplot(grid[0, column])
        scatter = atlas_panel(ax, anatomy, label)
        scatter.set_array(dv[np.argmin(np.abs(times - t))])
        ax.text(.99, .96, f"{t:.2f} s", transform=ax.transAxes, ha="right", va="top")
        length = 100 / anatomy["source"]["lengthMicrometers"]
        ax.text(.044 + length, -.054, "100 μm", ha="left", va="center", fontsize=7.2)
        if column == 1:
            ax.legend(
                handles=[
                    Line2D([], [], marker="o", ls="", mfc="none", mec=INK, ms=4, mew=.65,
                           label="PLM input"),
                    Line2D([], [], color="#4e725a", lw=.7, ls="--", label="Gap junction"),
                ],
                loc="lower right", bbox_to_anchor=(.994, -.04), ncols=2, frameon=False,
                fontsize=7.2, handlelength=1.28,
                columnspacing=.85, handletextpad=.4, borderaxespad=0,
            )
    bar_ax = fig.add_subplot(grid[1, :])
    bar_box = bar_ax.get_position()
    bar_ax.set_position([.347, bar_box.y0, .306, bar_box.height])
    bar = fig.colorbar(mpl.cm.ScalarMappable(norm=LIMIT, cmap=ACTIVITY), cax=bar_ax,
                       orientation="horizontal", ticks=[-50, 0, 50], extend="both")
    bar.set_ticks([-50, 0, 50], labels=["−50", "0", "+50"])
    bar.ax.minorticks_off()
    bar.ax.tick_params(labelsize=8, length=1.8, pad=1)
    bar.outline.set_linewidth(.45)
    bar.ax.text(1.115, .5, r"$\Delta V$ (mV)", transform=bar.ax.transAxes, ha="left", va="center")
    for column, (xy, label, condition) in enumerate(zip(paths, "CD", ("Touch", "Sham"))):
        ax = fig.add_subplot(grid[2, column])
        body_panel(ax, paths, label)
        ax.set_xlabel(r"$x$ (model units)", labelpad=1)
        ax.text(.095, .937, condition, transform=ax.transAxes, ha="left", va="top", fontsize=8)
        centroid = xy.mean(axis=1)
        centroid_line, = ax.plot(*centroid.T, c="#9a9fa4", lw=.88, zorder=1, label="Centroid")
        times_legend = []
        for index, t in enumerate(TIME_LEVELS):
            frame = np.argmin(np.abs(times - t))
            paint = TRACE(.21 + .24 * index)
            body_pose(ax, xy[frame], anatomy["outline"], paint, .78, .95)
            ax.plot(*xy[frame, 0], "o", c=paint, markersize=2.5)
            times_legend.append(Line2D([], [], color=paint, lw=1, marker="o", markersize=2,
                                       label=f"{t:.2f} s"))
        legend = times_legend if column == 1 else [centroid_line]
        position = "lower left" if column == 1 else "upper right"
        ax.legend(handles=legend, loc=position, frameon=False, fontsize=7.2, handlelength=1.2,
                  handletextpad=.4, labelspacing=.2, borderaxespad=.4)
        if column == 1:
            ax.tick_params(labelleft=False)
        else:
            ax.set_ylabel(r"$y$ (model units)", labelpad=1)
    output = ESSAY / "figures" / "connectome.pdf"
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output)
    plt.close(fig)


def write_demo(anatomy, touch, sham):
    fig = plt.figure(figsize=(10, 4.0), dpi=128)
    grid = fig.add_gridspec(2, 2, height_ratios=[.93, 1.65], left=.06, right=.98, top=.89,
                           bottom=.09, hspace=.18, wspace=.20)
    times = touch["time"]
    voltage = touch["voltage"] - sham["voltage"]
    paths = [case["xy"] for case in (touch, sham)]
    atlas = fig.add_subplot(grid[0, :])
    scatter = atlas_panel(atlas, anatomy)
    cbar = fig.colorbar(mpl.cm.ScalarMappable(norm=LIMIT, cmap=ACTIVITY), ax=atlas,
                        fraction=.024, pad=.015, ticks=[-50,0,50], extend="both")
    cbar.ax.set_ylabel(r"$\Delta V$ (mV)")
    clocks = fig.text(.065, .94, "", fontsize=10, color=INK)
    caption = fig.text(.98, .94, "", fontsize=8.5, color=INK, ha="right")
    traces, bodies, heads = [], [], []
    colors = [TRACE(.45), mpl.colormaps["cmr.ember"](.62)]
    for column, (xy, color, label) in enumerate(zip(paths, colors, ("Posterior touch", "Sham"))):
        ax = fig.add_subplot(grid[1, column])
        body_panel(ax, paths)
        ax.text(.04, .96, label, transform=ax.transAxes, va="top", fontsize=9.5, color=color)
        ax.plot(*xy.mean(axis=1).T, color=color, lw=.75, alpha=.14)
        traces.append(ax.plot([], [], color=color, lw=1.1)[0])
        bodies.append(body_pose(ax, xy[0], anatomy["outline"], color, .78, 1.1))
        heads.append(ax.plot([], [], marker="o", color=color, markersize=3, ls="")[0])

    def update(index):
        scatter.set_array(voltage[index])
        t = times[index]
        clocks.set_text(f"{t:05.2f} s")
        current = np.abs(touch["stimulus"][index]).max()
        caption.set_text(f"PLML and PLMR input: {current:,.0f} pA per cell")
        for xy, trace, body, head in zip(paths, traces, bodies, heads):
            position = xy[index]
            trace.set_data(*xy[:index+1].mean(axis=1).T)
            body.set_xy(centerline_outline(position, anatomy["outline"]))
            head.set_data(*position[:1].T)
        return [scatter, clocks, caption, *traces, *bodies, *heads]

    frames = np.append(np.arange(0, len(times), 5), len(times) - 1)
    movie = FuncAnimation(fig, update, frames=frames, interval=50)
    writer = FFMpegWriter(fps=20, codec="libx264", extra_args=["-pix_fmt", "yuv420p", "-movflags", "+faststart"])
    movie.save(ESSAY / "demos" / "connectome-touch.mp4", writer=writer)
    plt.close(fig)


if __name__ == "__main__":
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("--movie", action="store_true")
    args = cli.parse_args()
    sources = load()
    write_figure(*sources)
    if args.movie:
        write_demo(*sources)
