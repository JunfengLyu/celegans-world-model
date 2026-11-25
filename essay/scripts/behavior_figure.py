"""Regenerate the behavior-physics figure from released eigenworm recordings.

Run from the project root: .venv/bin/python essay/scripts/behavior_figure.py
"""

import argparse
import json
from pathlib import Path
import sys

import cmasher  # Registers the cmr colormaps.
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm, TwoSlopeNorm
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
import numpy as np
from threadpoolctl import threadpool_limits

PROJECT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT))

from wormlab.data import FPS, SPLIT, centerlines, read_basis, read_worms, valid_windows

ESSAY = PROJECT / "essay"
MARKOV = PROJECT / "artifacts" / "benchmark" / "delay_markov.npz"
PHASE_STEPS = 2
PHASE_H = PHASE_STEPS / FPS
MIN_PHASE_RADIUS = 2.0
MIN_BIN_COUNT = 60


def example_contours(worm, basis):
    """Use the first smooth 1.5 s segment of the first training individual."""
    starts = valid_windows(worm, past=25, future=0)
    coefficients = worm[starts[:, None] + np.arange(25)]
    complex_mode = coefficients[..., 0] + 1j * coefficients[..., 1]
    radius = np.abs(complex_mode)
    phases = np.angle(complex_mode[:, 1:] * complex_mode[:, :-1].conj())
    accepted = (
        (radius.min(axis=1) > 3)
        & (radius.max(axis=1) < 10)
        & (np.max(np.abs(coefficients[..., 2]), axis=1) < 5)
        & (np.abs(phases.sum(axis=1)) >= np.pi)
        & (np.abs(phases).max(axis=1) < 0.8)
        & ((np.mean(phases > 0, axis=1) >= 0.9) | (np.mean(phases < 0, axis=1) >= 0.9))
    )
    if not np.any(accepted):
        raise ValueError("No recording segment satisfies the contour-selection rule.")
    index = int(np.flatnonzero(accepted)[0])
    frames = np.arange(0, 25, 8)
    selected = coefficients[index, frames]
    full = centerlines(selected, basis)
    reduced = centerlines(selected[:, :2], basis[:, :2])
    return int(starts[index]), frames, selected, full, reduced


def phase_drift(worms):
    """Estimate a discrete-lag phase drift without interpolation across gaps."""
    phi_edges = np.linspace(-np.pi, np.pi, 37)
    omega_edges = np.linspace(-1.2, 1.2, 41)
    phase, speed, acceleration = [], [], []
    for worm in worms:
        center = valid_windows(worm, past=2 * PHASE_STEPS + 1, future=0) + PHASE_STEPS
        indices = center[:, None] + np.arange(-PHASE_STEPS, PHASE_STEPS + 1)
        a = worm[indices]
        complex_mode = a[..., 0] + 1j * a[..., 1]
        ok = (np.abs(complex_mode) >= MIN_PHASE_RADIUS).all(axis=1)
        values = complex_mode[ok]
        past, current, future = values[:, 0], values[:, PHASE_STEPS], values[:, -1]
        past_speed = np.angle(current * past.conj()) / (2.0 * np.pi * PHASE_H)
        future_speed = np.angle(future * current.conj()) / (2.0 * np.pi * PHASE_H)
        phase.append(np.angle(current))
        speed.append(past_speed)
        acceleration.append((future_speed - past_speed) / PHASE_H)
    phi, omega, dw = map(np.concatenate, (phase, speed, acceleration))
    bins = phi_edges, omega_edges
    count, _, _ = np.histogram2d(phi, omega, bins=bins)
    weights, _, _ = np.histogram2d(phi, omega, bins=bins, weights=dw)
    field = np.divide(weights, count, out=np.zeros_like(weights), where=count > 0)
    return phi_edges, omega_edges, np.ma.masked_where(count < MIN_BIN_COUNT, field), count


def measured_correlations(worm, lags):
    """Use endpoint pairs within contiguous, valid recordings, then normalize C(0)."""
    valid = np.isfinite(worm).all(axis=1)
    modes = worm[:, :2].astype(np.float64)
    modes -= modes[valid].mean(axis=0)
    variance = np.mean(np.sum(modes[valid] ** 2, axis=1))
    prefix = np.r_[0, np.cumsum(~valid)]
    correlation = []
    for lag in lags:
        starts = np.flatnonzero(prefix[lag + 1:] == prefix[:len(worm) - lag])
        products = np.sum(modes[starts] * modes[starts + lag], axis=1)
        correlation.append(products.mean() / variance)
    return np.asarray(correlation)


def markov_correlations(path, lag_count):
    with np.load(path) as file:
        transition = file["transition"].astype(np.float64)
        modes = file["centers"].reshape(len(transition), -1, 5)[:, -1, :2].astype(np.float64)
        lag_frames = int(file["lag"])
    # Convert prototype emissions from training z scores back to native coefficients.
    with np.load(path.with_name("scaling.npz")) as scale:
        modes = modes * scale["scale"][:2] + scale["mean"][:2]
    if np.any(transition < 0) or not np.allclose(transition.sum(axis=1), 1):
        raise ValueError("Expected a row-stochastic transition matrix.")
    stationary = np.full(len(transition), 1 / len(transition))
    for _ in range(10_000):
        updated = stationary @ transition
        if np.max(np.abs(updated - stationary)) < 1e-13:
            stationary = updated
            break
        stationary = updated
    else:
        raise RuntimeError("Stationary distribution did not converge.")
    modes -= stationary @ modes
    variance = np.sum(stationary[:, None] * modes**2)
    prediction = modes.copy()
    correlation = []
    for _ in range(lag_count + 1):
        correlation.append(np.sum(stationary[:, None] * modes * prediction) / variance)
        prediction = transition @ prediction
    return lag_frames, np.asarray(correlation)


def style_axis(axis):
    axis.spines[["top", "right"]].set_visible(False)
    axis.tick_params(length=2.7, pad=2)


def field_colorbar(figure, axis, mesh, ticks, labels):
    bar = figure.colorbar(mesh, ax=axis, pad=0.065, shrink=0.92, aspect=15, ticks=ticks)
    bar.outline.set_linewidth(0.45)
    bar.ax.minorticks_off()
    bar.set_ticks(ticks, labels=labels)
    bar.ax.tick_params(length=2, pad=2, labelright=True)


def draw_figure(worms, basis, markov, output):
    from worm_render import draw_worm, load_cuticle_profile

    training = [worms[index] for index in SPLIT["train"]]
    observations = np.concatenate(training)
    observations = observations[np.isfinite(observations).all(axis=1)]
    start, frames, modes, full, reduced = example_contours(training[0], basis)
    cuticle = load_cuticle_profile()
    phi_edges, omega_edges, field, counts = phase_drift(training)
    lag, markov_curve = markov_correlations(markov, lag_count=10)
    lags = lag * np.arange(len(markov_curve))
    measured = np.stack([measured_correlations(worms[index], lags) for index in SPLIT["test"]])

    plt.rcParams.update({
        "font.family": "STIXGeneral", "mathtext.fontset": "stix",
        "font.size": 11.0, "axes.labelsize": 11.0, "axes.labelpad": 2.5,
        "xtick.labelsize": 10.1, "ytick.labelsize": 10.1,
        "axes.linewidth": 0.65, "pdf.fonttype": 42,
    })
    table = plt.colormaps["cmr.cosmic"]
    contour_colors = table(np.linspace(0.30, 0.79, len(frames)))
    data_color, markov_color = plt.colormaps["cmr.fusion"]([0.28, 0.72])
    figure, axes = plt.subplots(2, 2, figsize=(6.85, 3.36))
    figure.subplots_adjust(left=0.095, right=0.97, bottom=0.112, top=0.951, wspace=0.36, hspace=0.45)
    silhouette, occupation, drift, memories = axes.flat

    offsets = ((-0.57, 0.34), (0.57, 0.34), (-0.57, -0.32), (0.57, -0.32))
    for a, b, t, color, (x, y) in zip(full, reduced, frames, contour_colors, offsets):
        draw_worm(silhouette, a + (x, y), cuticle, color, linewidth=0.48, alpha=0.73)
        silhouette.plot(b[:, 0] + x, b[:, 1] + y, color="#333338", lw=0.94,
                        ls=(0, (3.4, 2.4)), zorder=4)
        silhouette.text(x, y - 0.22, f"{t / FPS:.2f} s", ha="center", va="top")
    silhouette.plot([-0.25, 0.0], [-0.77, -0.77], color="#383838", lw=1.1)
    silhouette.text(0.05, -0.77, "$0.25L$", ha="left", va="center", fontsize=9.2)
    silhouette.set(xlim=(-1.07, 1.07), ylim=(-0.86, 0.62), aspect="equal")
    silhouette.set_axis_off()
    handles = [
        Patch(facecolor=contour_colors[1], alpha=0.73, linewidth=0, label="Five modes"),
        Line2D([], [], color="#333338", lw=0.94, ls=(0, (3.4, 2.4)), label="Two modes"),
    ]
    silhouette.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.5, -0.035),
                      ncols=2, fontsize=9.3, frameon=False, borderaxespad=0,
                      handlelength=1.36, handletextpad=0.4, columnspacing=1.0)

    edges = np.linspace(-11, 11, 100)
    frequency, _, _ = np.histogram2d(observations[:, 0], observations[:, 1], bins=(edges, edges))
    density = frequency.T / frequency.max()
    mesh = occupation.imshow(np.ma.masked_less(density, 0.003), origin="lower",
                             extent=(-11, 11, -11, 11), interpolation="nearest",
                             cmap="cmr.ember_r", norm=LogNorm(0.003, 1))
    occupation.scatter(modes[:, 0], modes[:, 1], s=18, c=contour_colors, edgecolors="white", lw=0.45)
    occupation.set(xlabel="$a_1$", ylabel="$a_2$", xticks=(-8, 0, 8), yticks=(-8, 0, 8))
    occupation.set_aspect("equal", adjustable="box")
    field_colorbar(figure, occupation, mesh, [0.01, 0.1, 1], ["0.01", "0.1", "1"])

    span = float(np.ceil(np.quantile(np.abs(field.compressed()), 0.96)))
    mesh = drift.imshow(field.T, origin="lower", interpolation="nearest", aspect="auto",
                        extent=(phi_edges[0], phi_edges[-1], omega_edges[0], omega_edges[-1]),
                        cmap="cmr.fusion", norm=TwoSlopeNorm(vcenter=0.0, vmin=-span, vmax=span))
    drift.set(xlabel=r"$\phi$ (rad)", ylabel=r"$\omega$ (cycles/s)",
              yticks=(-1, 0, 1), xticks=(-np.pi, 0, np.pi),
              xticklabels=(r"$-\pi$", "$0$", r"$\pi$"))
    field_colorbar(figure, drift, mesh, [-span, 0, span], [f"{-span:g}", "0", f"{span:g}"])

    times = lags / FPS
    memories.axhline(0, color="#aaaaaa", linewidth=0.64)
    memories.fill_between(times, measured.min(axis=0), measured.max(axis=0), color=data_color, alpha=0.17)
    memories.plot(times, measured.mean(axis=0), color=data_color, lw=1.55, label="Observed")
    memories.plot(times, markov_curve, color=markov_color, lw=1.45, ls="--",
                  marker="o", markersize=2.5, markeredgewidth=0, label="Delay Markov")
    memories.set(xlabel=r"Lag $\tau$ (s)", ylabel=r"Correlation $C(\tau)$", xlim=(0, times[-1]),
                 ylim=(-0.65, 1.06), xticks=(0, 2.5, 5.0, 7.5), yticks=(-0.5, 0, 0.5, 1))
    memories.legend(loc="upper right", fontsize=9.3, frameon=False, borderaxespad=0.25,
                    handlelength=1.7, handletextpad=0.45, labelspacing=0.2)

    for axis in (occupation, drift, memories):
        style_axis(axis)
    for label, (x, y) in zip("ABCD", ((0.014, 0.967), (0.519, 0.967),
                                     (0.014, 0.490), (0.519, 0.490))):
        figure.text(x, y, label, fontsize=11.5, va="top", ha="left", fontweight="normal")

    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output, metadata={"Title": "Statistical physics of C. elegans behavior"})
    plt.close(figure)
    radii = np.linalg.norm(observations[:, :2], axis=1)
    return {
        "contour_animal": SPLIT["train"][0],
        "contour_start_frame_zero_based": start,
        "contour_elapsed_s": (frames / FPS).tolist(),
        "observed_training_frames": len(observations),
        "observed_radius_quantiles_10_50_90": np.quantile(radii, [0.1, 0.5, 0.9]).tolist(),
        "drift_lag_s": PHASE_H,
        "minimum_drift_radius": MIN_PHASE_RADIUS,
        "minimum_drift_bin_count": MIN_BIN_COUNT,
        "drift_samples_in_supported_bins": int(counts[counts >= MIN_BIN_COUNT].sum()),
        "drift_color_saturation_cycles_per_s2": span,
        "markov_lag_s": lag / FPS,
        "autocorrelation_lags_s": times.tolist(),
        "measured_autocorrelations": dict(zip(SPLIT["test"], measured.tolist())),
        "mean_measured_autocorrelation": measured.mean(axis=0).tolist(),
        "markov_autocorrelation": markov_curve.tolist(),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--markov", type=Path, default=MARKOV)
    parser.add_argument("--output", type=Path, default=ESSAY / "figures" / "behavior.pdf")
    parser.add_argument("--stats", type=Path, default=ESSAY / "build" / "behavior_stats.json")
    args = parser.parse_args()
    with threadpool_limits(limits=2):
        summary = draw_figure(read_worms(), read_basis(), args.markov, args.output)
    args.stats.parent.mkdir(parents=True, exist_ok=True)
    args.stats.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf8")
    print(args.output)


if __name__ == "__main__":
    main()
