"""Render reproducible forecast plots and offline centerline animations."""

import argparse
import json
from pathlib import Path

import imageio_ffmpeg
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.animation import FFMpegWriter, FuncAnimation
import numpy as np
import pandas as pd

from .data import ARTIFACTS, FPS, centerlines, read_basis

COLORS = {"Observation": "#1f2937", "Persistence": "#8a919e", "Linear AR": "#577590",
          "Phase dynamics": "#ba522c", "Delay Markov": "#7656a0", "GRU": "#18787a",
          "GRU 17": "#18787a", "RSSM": "#bd3970"}
plt.rcParams.update({"font.size": 11, "axes.spines.top": False, "axes.spines.right": False,
                     "axes.labelcolor": "#243047", "text.color": "#243047",
                     "figure.dpi": 130, "savefig.dpi": 170,
                     "animation.ffmpeg_path": imageio_ffmpeg.get_ffmpeg_exe()})


def forecast_plots(benchmark, rssm, out):
    raw = pd.concat([pd.read_csv(folder / "metrics_per_worm.csv") for folder in (benchmark, rssm)])
    raw["family"] = raw["model"].str.replace(r"^GRU \d+$", "GRU", regex=True)
    by_worm = raw.groupby(["family", "worm", "seconds"], as_index=False)["rmse_z"].mean()
    summary = by_worm.groupby(["family", "seconds"])["rmse_z"].mean().unstack()
    summary.to_csv(out / "macro_rmse.csv", float_format="%.6f")
    fig, axes = plt.subplots(1, 3, figsize=(13, 4), sharex=True, sharey=True, layout="constrained")
    panels = [(by_worm, "Mean across test animals")] + [
        (by_worm[by_worm.worm == worm], f"Test animal {worm}") for worm in sorted(by_worm.worm.unique())
    ]
    for ax, (panel, title) in zip(axes, panels):
        for model in ("Persistence", "Linear AR", "Delay Markov", "Phase dynamics", "GRU", "RSSM"):
            curve = panel[panel.family == model].groupby("seconds").rmse_z.mean()
            ax.plot(curve.index, curve, marker="o", ms=3, label=model, color=COLORS[model])
        ax.set(xscale="log", xlabel="Forecast horizon (s)", title=title)
        ax.set_xticks([0.0625, 0.25, 0.5, 1, 2, 4], ["1/16", "1/4", "1/2", "1", "2", "4"])
        ax.grid(axis="y", alpha=0.22)
    axes[0].set_ylabel("RMSE in training-standardized coefficients")
    axes[2].legend(frameon=False, fontsize=8, loc="lower right")
    fig.savefig(out / "forecast_rmse.png")
    plt.close(fig)

    training = pd.read_csv(benchmark / "training.csv")
    latent = pd.read_csv(rssm / "training.csv")
    fig, axes = plt.subplots(1, 2, figsize=(9.5, 3.5), layout="constrained")
    for seed, curve in training.groupby("seed"):
        axes[0].plot(curve.epoch, curve.validation_rollout_mse, label=f"Seed {seed}")
    axes[1].plot(latent.epoch, latent.validation_rollout_mse, color=COLORS["RSSM"], label="Prior rollout MSE")
    axes[1].plot(latent.epoch, latent.train_reconstruction_mse, color="#577590", label="Posterior reconstruction MSE")
    for ax, title in zip(axes, ("GRU · 16-step validation", "RSSM · validation and reconstruction")):
        ax.set(xlabel="Epoch", ylabel="MSE (standardized)", title=title)
        ax.grid(axis="y", alpha=0.22)
        ax.legend(frameon=False, fontsize=8)
    fig.savefig(out / "training_curves.png")
    plt.close(fig)


def dynamics_plots(benchmark, rssm, out):
    from .models import PhaseDynamics

    stats = pd.concat([pd.read_csv(folder / "rollout_statistics.csv") for folder in (benchmark, rssm)])
    fig, axes = plt.subplots(1, 2, figsize=(10, 4), sharey=True, layout="constrained")
    lags = np.array([1, 8, 16, 32])
    columns = [f"acf_a1_a2_mean_lag{lag}" for lag in lags]
    for ax, (animal, group) in zip(axes, stats.groupby("worm")):
        for row in group.to_dict("records"):
            label = row["model"]
            ax.plot(lags / FPS, [row[col] for col in columns], "o-", color=COLORS[label],
                    markersize=4, label=label)
        ax.axhline(0, color="#9ca3af", linewidth=0.8)
        ax.set(xlabel="Lag (s)", title=f"Test animal {animal}")
    axes[0].set_ylabel("Mean lag correlation of a₁ and a₂")
    axes[1].legend(frameon=False, fontsize=8)
    fig.savefig(out / "rollout_dynamics.png")
    plt.close(fig)

    with np.load(benchmark / "scaling.npz") as scaling, np.load(benchmark / "phase_fit.npz") as fitted:
        phase = PhaseDynamics(scaling["mean"], scaling["scale"])
        phase.coef = fitted["coefficients"]
        phase.speed_scale = float(fitted["speed_scale"].item())
    radius = json.loads((benchmark / "run.json").read_text())["phase_reference_radius"]
    phi, cycles = np.meshgrid(np.linspace(-np.pi, np.pi, 120), np.linspace(-1, 1, 100))
    acceleration = phase.angular_acceleration(phi.ravel(), cycles.ravel() * 2 * np.pi, radius)
    acceleration = acceleration.reshape(phi.shape) / (2 * np.pi)
    limit = np.quantile(np.abs(acceleration), 0.97)
    fig, ax = plt.subplots(figsize=(7.3, 4.1), layout="constrained")
    img = ax.pcolormesh(phi, cycles, acceleration, cmap="RdBu_r", vmin=-limit, vmax=limit,
                        shading="auto", rasterized=True)
    ax.contour(phi, cycles, acceleration, levels=[0], colors="#20283b", linewidths=1)
    ax.set(xlabel="Body-wave phase φ (rad)", ylabel="Angular speed (cycles/s)",
           title="Fitted phase drift at median training radius")
    ax.set_xticks([-np.pi, -np.pi / 2, 0, np.pi / 2, np.pi], ["−π", "−π/2", "0", "π/2", "π"])
    fig.colorbar(img, ax=ax, label="Conditional angular acceleration (cycles/s²)")
    fig.savefig(out / "phase_drift.png")
    plt.close(fig)


def posture_movie(benchmark, rssm, out):
    basis = read_basis()
    with np.load(benchmark / "example_rollouts.npz") as examples, np.load(rssm / "rollouts.npz") as latent:
        panels = ("Observation", "GRU 17", "RSSM", "Phase dynamics", "Delay Markov")
        shapes = {model: centerlines(
            np.concatenate((examples["history"], latent["example"] if model == "RSSM" else examples[model])),
            basis) for model in panels}
        history = len(examples["history"])
    fig, axes = plt.subplots(1, len(panels), figsize=(11.5, 3.2), layout="constrained")
    lines, tips = [], []
    for ax, model in zip(axes, panels):
        ax.set(xlim=(-0.55, 0.55), ylim=(-0.55, 0.55), aspect="equal", title=model)
        ax.axis("off")
        lines.append(ax.plot([], [], color=COLORS[model], lw=3, solid_capstyle="round")[0])
        tips.append(ax.plot([], [], ".", color=COLORS[model], ms=7)[0])
    caption = fig.suptitle("", fontsize=13)

    def update(frame):
        relative = (frame - history + 1) / FPS
        stage = "Observed context" if frame < history else "Free rollout · one sample per model"
        caption.set_text(f"{stage}   |   {relative:+.2f} s")
        for line, tip, model in zip(lines, tips, panels):
            xy = shapes[model][frame]
            line.set_data(*xy.T)
            tip.set_data(*xy[:1].T)
        return [*lines, *tips, caption]

    movie = FuncAnimation(fig, update, frames=len(shapes[panels[0]]), interval=1000 / FPS)
    movie.save(out / "posture_rollouts.mp4", writer=FFMpegWriter(fps=FPS, codec="libx264"))
    plt.close(fig)


def virtual_figures(virtual, out):
    meta = json.loads((virtual / "modworm.json").read_text())
    scenarios = list(meta["scenarios"])
    label = {"varshney_posterior": "Varshney · posterior touch",
             "cook_posterior": "Cook · posterior touch", "cook_sham": "Cook · sham"}
    arrays = {}
    for scenario in scenarios:
        with np.load(virtual / f"{scenario}.npz") as data:
            arrays[scenario] = dict(time=data["time"], xy=data["xy"])
    fig, axes = plt.subplots(1, 3, figsize=(12, 4.4), layout="constrained", sharex=True, sharey=True)
    all_points = np.concatenate([data["xy"].reshape(-1, 2) for data in arrays.values()])
    low, high = all_points.min(axis=0) - 2, all_points.max(axis=0) + 2
    body, tip, trails = [], [], []
    for ax, scenario, color in zip(axes, scenarios, ("#18787a", "#ba522c", "#577590")):
        ax.set(xlim=(low[0], high[0]), ylim=(low[1], high[1]), aspect="equal", title=label[scenario],
               xlabel="x (model units)")
        ax.grid(alpha=0.15)
        trails.append(ax.plot([], [], color=color, alpha=0.36, lw=1.5)[0])
        body.append(ax.plot([], [], color=color, lw=3, solid_capstyle="round")[0])
        tip.append(ax.plot([], [], ".", color=color, ms=6)[0])
    axes[0].set_ylabel("y (model units)")
    title = fig.suptitle("")

    def update(frame):
        title.set_text(f"modWorm · upstream Python solver   |   {arrays[scenarios[0]]['time'][frame]:.2f} s")
        for name, line, marker, trail in zip(scenarios, body, tip, trails):
            xy = arrays[name]["xy"]
            line.set_data(*xy[frame].T)
            marker.set_data(*xy[frame, :1].T)
            trail.set_data(*xy[:frame + 1].mean(axis=1).T)
        return [*body, *tip, *trails, title]

    frames = np.arange(0, len(arrays[scenarios[0]]["time"]), 5)
    movie = FuncAnimation(fig, update, frames=frames, interval=50)
    movie.save(out / "modworm_posterior_touch.mp4", writer=FFMpegWriter(fps=20, codec="libx264"))
    update(frames[-1])
    fig.savefig(out / "modworm_centroid_paths.png")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("behavior", "virtual", "all"), default="all", nargs="?")
    parser.add_argument("--artifacts", type=Path, default=ARTIFACTS)
    args = parser.parse_args()
    out = args.artifacts / "figures"
    out.mkdir(parents=True, exist_ok=True)
    if args.mode in ("behavior", "all"):
        benchmark, rssm = args.artifacts / "benchmark", args.artifacts / "rssm"
        forecast_plots(benchmark, rssm, out)
        dynamics_plots(benchmark, rssm, out)
        posture_movie(benchmark, rssm, out)
    if args.mode in ("virtual", "all"):
        virtual_figures(args.artifacts / "virtual", out)


if __name__ == "__main__":
    main()
