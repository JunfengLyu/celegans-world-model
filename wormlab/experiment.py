"""Fit, evaluate and save a cross-animal benchmark of posture forecasts."""

import argparse
import csv
import json
from pathlib import Path

import numpy as np
import torch
from threadpoolctl import threadpool_limits

from .data import ARTIFACTS, FPS, HISTORY, SPLIT, read_worms, standardize, take_windows, window_index
from .models import PHASE_FEATURES, DelayMarkov, LinearAR, Persistence, PhaseDynamics, PostureGRU, wrap

HORIZONS = (1, 4, 8, 16, 32, 64)
TRAIN_STEPS = 8
BATCH = 256
SEEDS = (17, 29, 43)


def write_csv(path, rows):
    with path.open("w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def fit_gru(worms, seed, epochs, batches, out):
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    train = window_index(worms, SPLIT["train"], HISTORY, TRAIN_STEPS, stride=2)
    val = window_index(worms, SPLIT["validation"], HISTORY, 16, stride=29)
    validation = take_windows(worms, val, HISTORY + 16)
    val_input = torch.from_numpy(validation[:, :HISTORY])
    val_output = torch.from_numpy(validation[:, HISTORY:])
    model = PostureGRU()
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.002, weight_decay=0.0001)
    best_score = float("inf")
    log = []
    for epoch in range(1, epochs + 1):
        model.train()
        losses = []
        for _ in range(batches):
            batch = take_windows(worms, train[rng.integers(len(train), size=BATCH)], HISTORY + TRAIN_STEPS)
            batch = torch.from_numpy(batch)
            forecast = model(batch[:, :HISTORY], TRAIN_STEPS)
            loss = (forecast - batch[:, HISTORY:]).square().mean()
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            losses.append(loss.item())
        model.eval()
        with torch.no_grad():
            valid_mse = (model(val_input, 16) - val_output).square().mean().item()
        log.append({"seed": seed, "epoch": epoch, "train_rollout_mse": np.mean(losses),
                    "validation_rollout_mse": valid_mse})
        print(f"seed={seed} epoch={epoch:02d} train={np.mean(losses):.5f} val={valid_mse:.5f}", flush=True)
        if valid_mse < best_score:
            best_score = valid_mse
            best_epoch = epoch
            best_weights = {key: value.detach().clone() for key, value in model.state_dict().items()}
    model.load_state_dict(best_weights)
    residual = validation[:, HISTORY] - model.rollout(validation[:, :HISTORY], 1)[:, 0]
    model.noise = np.cov(residual.T)
    torch.save({"weights": best_weights, "innovation_covariance": torch.tensor(model.noise),
                "seed": seed, "epoch": best_epoch}, out / f"gru_seed_{seed}.pt")
    return model, log, {"seed": seed, "epoch": best_epoch, "validation_mse": best_score,
                        "parameters": sum(p.numel() for p in model.parameters()),
                        "eligible_training_windows": len(train), "validation_windows": len(val)}


def scores(truth, predicted, reference, scale, model):
    rows = []
    for animal, ref_start in reference.items():
        actual, estimate = truth[ref_start], predicted[ref_start]
        for horizon in HORIZONS:
            error = estimate[:, horizon - 1] - actual[:, horizon - 1]
            row = {"model": model, "worm": animal, "frames": horizon, "seconds": horizon / FPS,
                   "windows": len(estimate), "rmse_z": np.sqrt(np.mean(error ** 2)),
                   "rmse_eigenworm": np.sqrt(np.mean((error * scale) ** 2))}
            row.update({f"a{k}_rmse_z": val for k, val in enumerate(np.sqrt(np.mean(error ** 2, axis=0)), 1)})
            rows.append(row)
    return rows


def rollout_statistics(z, mean, scale, curvature_threshold):
    x = z * scale + mean
    radius = np.hypot(x[..., 0], x[..., 1])
    phi = np.arctan2(x[..., 1], x[..., 0])
    speed = wrap(phi[:, 1:] - phi[:, :-1]) * FPS / (2 * np.pi)
    values = {
        "radius_mean": radius.mean(),
        "phase_speed_abs_median_cycles_per_s": np.median(np.abs(speed)),
        "negative_phase_speed_fraction": np.mean(speed < 0),
        "high_a3_fraction": np.mean(np.abs(x[..., 2]) >= curvature_threshold),
        "all_modes_step_rms_z": np.sqrt(np.mean(np.diff(z, axis=1) ** 2)),
    }
    centered = z - z.mean(axis=(0, 1), keepdims=True)
    for lag in (1, 8, 16, 32):
        lead, follow = centered[:, :-lag, :2], centered[:, lag:, :2]
        cov = (lead * follow).mean(axis=(0, 1))
        norm = np.sqrt((lead ** 2).mean(axis=(0, 1)) * (follow ** 2).mean(axis=(0, 1)))
        values[f"acf_a1_a2_mean_lag{lag}"] = np.divide(cov, norm, out=np.zeros_like(cov),
                                                    where=norm > 1e-12).mean()
    return values


def run(out, epochs, batches):
    out.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(2)
    torch.use_deterministic_algorithms(True)
    source = read_worms()
    worms, mean, scale = standardize(source)
    np.savez_compressed(out / "scaling.npz", mean=mean, scale=scale)
    train_ids = SPLIT["train"]
    models = {
        "Persistence": Persistence(),
        "Linear AR": LinearAR().fit(worms, train_ids),
        "Phase dynamics": PhaseDynamics(mean, scale).fit(worms, train_ids),
        "Delay Markov": DelayMarkov().fit(worms, train_ids),
    }
    phase = models["Phase dynamics"]
    np.savez_compressed(out / "phase_fit.npz", coefficients=phase.coef, bounds=phase.bounds,
                        speed_scale=phase.speed_scale, covariance=phase.covariance,
                        feature_names=np.array(PHASE_FEATURES))
    np.savez_compressed(out / "linear_ar.npz", coefficients=models["Linear AR"].coef)
    markov = models["Delay Markov"]
    np.savez_compressed(out / "delay_markov.npz", centers=markov.cluster.cluster_centers_,
                        transition=markov.transition.toarray(), lag=markov.lag,
                        next_labels=markov.ends, snippets=markov.snippets,
                        origin_labels=markov.cluster.labels_)

    training_log, fits = [], []
    for seed in SEEDS:
        gru, log, fit = fit_gru(worms, seed, epochs, batches, out)
        models[f"GRU {seed}"] = gru
        training_log.extend(log)
        fits.append(fit)
    write_csv(out / "training.csv", training_log)

    test = window_index(worms, SPLIT["test"], HISTORY, max(HORIZONS), stride=32)
    np.save(out / "test_windows.npy", test)
    batch = take_windows(worms, test, HISTORY + max(HORIZONS))
    inputs, truth = batch[:, :HISTORY], batch[:, HISTORY:]
    masks = {animal: test[:, 0] == animal for animal in SPLIT["test"]}
    curvature_threshold = float(np.nanquantile(np.abs(np.concatenate(
        [source[animal][:, 2] for animal in train_ids])), 0.9))
    statistics = []
    metrics = []
    example_index = int(np.flatnonzero(masks[SPLIT["test"][0]])[0])
    example = {"Observation": truth[example_index] * scale + mean}
    predictions = {}
    for label, model in models.items():
        forecast = model.rollout(inputs, max(HORIZONS))
        if not np.isfinite(forecast).all():
            raise RuntimeError(f"Non-finite forecast for {label}")
        predictions[label] = forecast
        metrics.extend(scores(truth, forecast, masks, scale, label))
        if label in {"Phase dynamics", "Delay Markov", f"GRU {SEEDS[0]}"}:
            sample = model.rollout(inputs, max(HORIZONS), np.random.default_rng(2026))
            example[label] = sample[example_index] * scale + mean
            for animal, mask in masks.items():
                statistics.append(dict(model=label, worm=animal, **rollout_statistics(
                    sample[mask], mean, scale, curvature_threshold)))
    for animal, mask in masks.items():
        statistics.append(dict(model="Observation", worm=animal, **rollout_statistics(
            truth[mask], mean, scale, curvature_threshold)))

    write_csv(out / "metrics_per_worm.csv", metrics)
    write_csv(out / "rollout_statistics.csv", statistics)
    np.savez_compressed(out / "example_rollouts.npz", history=inputs[example_index] * scale + mean,
                        **example)
    np.savez_compressed(out / "forecast_examples.npz",
                        example_index=example_index, truth=truth[example_index] * scale + mean,
                        **{label: values[example_index] * scale + mean for label, values in predictions.items()})
    np.savez_compressed(out / "test_forecasts.npz",
                        truth=truth, **predictions)
    summary = {
        "split": SPLIT, "history_frames": HISTORY, "fps": FPS, "train_steps": TRAIN_STEPS,
        "epochs": epochs, "batches_per_epoch": batches, "batch_size": BATCH, "seeds": SEEDS,
        "torch_version": str(torch.__version__), "device": "CPU",
        "recording_frames": {animal: len(worm) for animal, worm in source.items()},
        "valid_frames": {animal: int(np.isfinite(worm).all(axis=1).sum()) for animal, worm in source.items()},
        "test_windows": {animal: int(mask.sum()) for animal, mask in masks.items()},
        "demo_example": {"worm": int(test[example_index, 0]), "start_frame_zero_based": int(test[example_index, 1])},
        "a3_threshold_train_abs_q90": curvature_threshold, "fits": fits,
        "phase_omega_per_frame_bounds": phase.bounds[:, 0].tolist(),
        "phase_reference_radius": float(np.median(np.hypot(
            *np.concatenate([source[i][np.isfinite(source[i]).all(axis=1), :2] for i in train_ids]).T))),
        "markov_states": markov.clusters, "markov_lag_frames": markov.lag,
        "markov_edges": markov.transition.nnz,
    }
    (out / "run.json").write_text(json.dumps(summary, indent=2) + "\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=ARTIFACTS / "benchmark")
    parser.add_argument("--epochs", type=int, default=12)
    parser.add_argument("--batches", type=int, default=80)
    args = parser.parse_args()
    if args.epochs < 1 or args.batches < 1:
        parser.error("--epochs and --batches must be positive")
    with threadpool_limits(limits=2):
        run(args.out, args.epochs, args.batches)


if __name__ == "__main__":
    main()
