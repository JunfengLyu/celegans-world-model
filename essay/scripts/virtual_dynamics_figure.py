"""Plot model voltage, muscle forces, and learned recurrent state dynamics."""

import sys
from pathlib import Path

import cmasher  # Register the cmr maps.
import matplotlib as mpl

mpl.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from matplotlib.colors import Normalize
from matplotlib.lines import Line2D
import numpy as np
from sklearn.decomposition import PCA
import torch

ESSAY = Path(__file__).resolve().parents[1]
PROJECT = ESSAY.parent
sys.path.insert(0, str(PROJECT))

from wormlab.data import ARTIFACTS, FPS, HISTORY, SPLIT, read_worms, standardize, take_windows, window_index
from wormlab.models import PostureGRU
from wormlab.rssm import CategoricalRSSM

INK = "#242731"
GRAY = "#9499a2"
OBSERVED_LINESTYLE = (0, (2, 1.2))
TIME_NORM = Normalize(0, 4)
TIME_MAP = mpl.colormaps["cmr.cosmic"].resampled(256)
FORCE_MAP = mpl.colormaps["cmr.fusion_r"]
VOLTAGE_COLORS = [mpl.colormaps[name](.55) for name in (
    "cmr.cosmic", "cmr.sunburst", "cmr.rainforest", "cmr.neon",
)]
FORCE_SCALE = 6000.0  # modWorm CE.scaled_factor in muscle_body_params.py.
RSSM_DRAW_SEED = 3107

mpl.rcParams.update({
    "font.family": "DejaVu Sans", "font.size": 8.5, "axes.labelsize": 8.5,
    "axes.linewidth": 0.55, "axes.spines.top": False, "axes.spines.right": False,
    "xtick.labelsize": 8, "ytick.labelsize": 8, "xtick.major.width": 0.55,
    "ytick.major.width": 0.55, "xtick.major.size": 2.3, "ytick.major.size": 2.3,
    "pdf.fonttype": 42,
})


def load_checkpoint(model, path):
    checkpoint = torch.load(path, weights_only=True, map_location="cpu")
    model.load_state_dict(checkpoint["weights"])
    print(f"{path.relative_to(PROJECT)}: seed {checkpoint['seed']}, epoch {checkpoint['epoch']}")
    return model.eval()


@torch.no_grad()
def gru_embedding(train, example):
    model = load_checkpoint(PostureGRU(), ARTIFACTS / "benchmark" / "gru_seed_17.pt")
    _, observed_train = model.recurrence(train)
    pca = PCA(n_components=2, svd_solver="full").fit(observed_train[0].numpy())
    observed, _ = model.recurrence(example)
    observed = observed[0, HISTORY - 1:].numpy()
    recurrent_calls = []
    hook = model.recurrence.register_forward_hook(
        lambda module, args, output: recurrent_calls.append(output[0][0].numpy())
    )
    try:
        model(example[:, :HISTORY], len(observed))
    finally:
        hook.remove()
    autonomous = np.concatenate(recurrent_calls[1:])
    np.testing.assert_allclose(autonomous[0], observed[0], atol=1e-6)
    return pca, observed, autonomous


@torch.no_grad()
def rssm_embedding(train, example):
    model = load_checkpoint(CategoricalRSSM(), ARTIFACTS / "rssm" / "rssm.pt")

    def pass_through(sequence, observed_frames):
        generator = torch.Generator().manual_seed(RSSM_DRAW_SEED)
        hidden, latent = model.initial(sequence)
        probabilities = []
        for t in range(sequence.shape[1]):
            hidden = model.dynamics(latent, hidden)
            logits = (model.posterior(torch.cat((hidden, sequence[:, t]), dim=-1))
                      if t < observed_frames else model.prior(hidden))
            distribution = model.distribution(logits)
            latent = model.draw(distribution, generator)
            probabilities.append(distribution.flatten(1))
        return torch.stack(probabilities, dim=1).numpy()

    observed_train = pass_through(train, HISTORY)[:, -1]
    pca = PCA(n_components=2, svd_solver="full").fit(observed_train)
    observed = pass_through(example, example.shape[1])[0, HISTORY - 1:]
    autonomous = pass_through(example, HISTORY)[0, HISTORY - 1:]
    np.testing.assert_array_equal(autonomous[0], observed[0])
    return pca, observed, autonomous


def mark(ax, label):
    ax.text(-.17, 1.14, label, transform=ax.transAxes, ha="left", va="top", fontsize=10, color=INK)


def draw_latent_panel(ax, model, projection, observed, autonomous):
    observed, autonomous = projection.transform(observed), projection.transform(autonomous)
    ax.plot(*observed.T, color=GRAY, lw=.9, ls=OBSERVED_LINESTYLE, alpha=.85)
    ax.scatter(*observed[-1], marker="x", color=GRAY, lw=.85, s=16, zorder=4)
    segments = np.stack((autonomous[:-1], autonomous[1:]), axis=1)
    paths = LineCollection(segments, cmap=TIME_MAP, norm=TIME_NORM, lw=1.25, zorder=3)
    paths.set_array((np.arange(len(segments)) + .5) / FPS)
    ax.add_collection(paths)
    ax.scatter(*autonomous[0], facecolor="white", edgecolor=INK, lw=.72, s=16, zorder=5)
    ax.scatter(*autonomous[-1], marker="x", color=TIME_MAP(1.), lw=.9, s=22, zorder=5)
    ax.autoscale()
    ax.margins(.10)
    pc1, pc2 = 100 * projection.explained_variance_ratio_
    ax.set_xlabel(f"{model} PC1 ({pc1:.0f}%)")
    ax.set_ylabel(f"PC2 ({pc2:.0f}%)")
    ax.xaxis.set_major_locator(mpl.ticker.MaxNLocator(4))
    ax.yaxis.set_major_locator(mpl.ticker.MaxNLocator(4))
    return pc1, pc2


def draw_neural_panels(fig, ax_v, ax_f, ax_f_color):
    directory = ARTIFACTS / "virtual"
    with np.load(directory / "cook_posterior.npz") as touch, np.load(directory / "cook_sham.npz") as sham:
        time = touch["time"]
        names = touch["neurons"].tolist()
        np.testing.assert_array_equal(time, sham["time"])
        np.testing.assert_array_equal(touch["neurons"], sham["neurons"])
        voltage_delta = touch["voltage"] - sham["voltage"]
        force_delta = touch["muscle_force"] - sham["muscle_force"]

    for name, color in zip(("PLML", "PVCL", "AVBL", "VA02"), VOLTAGE_COLORS):
        ax_v.plot(time, voltage_delta[:, names.index(name)], lw=.85, c=color, label=name)
    ax_v.axhline(0, color="#d8dbe0", lw=.55, zorder=0)
    for snapshot in (.5, 2.5):
        ax_v.axvline(snapshot, color="#babdc3", lw=.55, ls="--", zorder=0)
        ax_f.axvline(snapshot, color=INK, lw=.55, ls="--", alpha=.6, zorder=3)
    ax_v.set(xlim=(0, 4), ylim=(-3, 12000), xlabel=r"$t$ (s)", ylabel=r"$\Delta V$ (mV)")
    ax_v.set_yscale("symlog", linthresh=.3, linscale=.45)
    ax_v.set_yticks([-1, 0, 1, 100, 10000], [r"$-1$", "0", "1", r"$10^2$", r"$10^4$"])
    ax_v.legend(frameon=False, loc="upper right", ncols=2, fontsize=7, handlelength=1.2,
                handletextpad=.4, columnspacing=.8, borderaxespad=.25)
    # The body solver uses sites 0..22 and 24..46; the omitted site on each side
    # is the unused final force column. Upstream output concatenates dorsal then ventral.
    asymmetry = (force_delta[:, :23] - force_delta[:, 24:47]) / FORCE_SCALE
    rows = np.arange(0, 401, 2)
    heatmap = ax_f.pcolormesh(time[rows], np.arange(1, 24), asymmetry[rows].T, cmap=FORCE_MAP,
                              vmin=-2, vmax=2, shading="nearest")
    ax_f.set(xlim=(0, 4), ylim=(23.5, .5), xlabel=r"$t$ (s)", ylabel="Muscle site",
             yticks=[1, 6, 12, 18, 23])
    color = fig.colorbar(heatmap, cax=ax_f_color, ticks=[-2, 0, 2])
    color.set_label(r"$\Delta(F_\mathrm{D}-F_\mathrm{V})/F_0$", labelpad=2)
    color.ax.tick_params(labelsize=7.2, pad=2, length=1.8)
    color.outline.set_linewidth(.45)


def main():
    torch.set_num_threads(2)
    worms, _, _ = standardize(read_worms())
    reference = window_index(worms, SPLIT["train"], HISTORY, 0, stride=32)
    train = torch.from_numpy(take_windows(worms, reference, HISTORY))
    candidates = np.load(ARTIFACTS / "benchmark" / "test_windows.npy")
    example_index = candidates[candidates[:, 0] == 6][:1]
    example = torch.from_numpy(take_windows(worms, example_index, HISTORY + 4 * FPS))
    fig = plt.figure(figsize=(6.85, 4.40))
    grid = fig.add_gridspec(3, 3, width_ratios=[1, 1, .055], height_ratios=[1, 1, .06],
                           left=.114, right=.947, bottom=.085, top=.928,
                           wspace=.36, hspace=.58)
    voltage_axis = fig.add_subplot(grid[0, 0])
    force_axis = fig.add_subplot(grid[0, 1])
    force_color_axis = fig.add_subplot(grid[0, 2])
    force_box = force_color_axis.get_position()
    force_color_axis.set_position([force_box.x0-.029, force_box.y0, .012, force_box.height])
    gru_axis = fig.add_subplot(grid[1, 0])
    rssm_axis = fig.add_subplot(grid[1, 1])
    draw_neural_panels(fig, voltage_axis, force_axis, force_color_axis)
    for ax, model, method in ((gru_axis, "GRU", gru_embedding), (rssm_axis, "RSSM", rssm_embedding)):
        percentages = draw_latent_panel(ax, model, *method(train, example))
        print(f"{method.__name__}: PC1 {percentages[0]:.2f}%, PC2 {percentages[1]:.2f}%")
    for label, axis in zip("ABCD", (voltage_axis, force_axis, gru_axis, rssm_axis)):
        mark(axis, label)
    color_axis = fig.add_subplot(grid[2, :2])
    box = color_axis.get_position()
    color_axis.set_position([.610, box.y0 - .02, .180, .016])
    color = fig.colorbar(mpl.cm.ScalarMappable(norm=TIME_NORM, cmap=TIME_MAP), cax=color_axis,
                         orientation="horizontal", ticks=[0, 2, 4])
    color.ax.tick_params(labelsize=7.2, pad=1, length=1.8)
    color.outline.set_linewidth(.45)
    color.ax.text(1.15, .5, r"Autonomous $t$ (s)", transform=color.ax.transAxes,
                  ha="left", va="center", fontsize=7.2)
    fig.legend(
        handles=[
            Line2D([], [], color=GRAY, lw=.95, ls=OBSERVED_LINESTYLE, label="Observed input"),
            Line2D([], [], marker="o", ls="", mfc="white", mec=INK, ms=3.4, mew=.72, label="Start"),
            Line2D([], [], marker="x", ls="", color=INK, ms=3.4, mew=.8, label="End"),
        ],
        loc="center left", bbox_to_anchor=(.105, box.y0 - .01), frameon=False, ncols=3,
        fontsize=7.2, handlelength=1.4, handletextpad=.45, columnspacing=1.2,
    )
    print(f"PCA reference: {len(reference)} train histories, held-out example {example_index.tolist()}")
    output = ESSAY / "figures" / "virtual-dynamics.pdf"
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output)
    plt.close(fig)


if __name__ == "__main__":
    main()
