"""Compare matched recurrent posture models with and without a fixed phase drift."""

import argparse
import json
import math
from pathlib import Path

import numpy as np
import torch
from scipy.stats import wasserstein_distance
from threadpoolctl import threadpool_limits

from .data import ARTIFACTS, FPS, HISTORY, SPLIT, read_worms, standardize, take_windows, window_index
from .experiment import BATCH, HORIZONS, SEEDS, TRAIN_STEPS, rollout_statistics, scores, write_csv
from .models import PHASE_FEATURES, PhaseDynamics, PostureGRU, wrap

ARMS = ("GRU", "Phase-embedded GRU")
PRIOR_WEIGHT = 0.5
MIN_RADIUS = 0.05
COHERENCE_LAGS = (1, 8, 16, 32)
SAMPLE_SEED = 2026


class PhaseEmbeddedGRU(PostureGRU):
    """Residual GRU with a fixed phase-amplitude drift in its first two modes."""

    def __init__(self, prior):
        super().__init__()
        self.register_buffer("mean_raw", torch.as_tensor(prior.mean, dtype=torch.float32))
        self.register_buffer("scale_raw", torch.as_tensor(prior.scale, dtype=torch.float32))
        self.register_buffer("basis_coef", torch.as_tensor(prior.coef[:, :2], dtype=torch.float32))
        self.register_buffer("bounds", torch.as_tensor(prior.bounds[:, :2], dtype=torch.float32))
        self.register_buffer("speed_scale", torch.as_tensor(prior.speed_scale, dtype=torch.float32))

    @staticmethod
    def polar(pair):
        x, y = pair.unbind(-1)
        squared_radius = x.square() + y.square()
        defined = squared_radius > 1e-12
        phase = torch.atan2(torch.where(defined, y, 0.0), torch.where(defined, x, 1.0))
        log_radius = 0.5 * torch.log(squared_radius.clamp_min(MIN_RADIUS ** 2))
        return phase, log_radius

    def phase_increment(self, previous_z, current_z):
        previous = previous_z[:, :2] * self.scale_raw[:2] + self.mean_raw[:2]
        current = current_z[:, :2] * self.scale_raw[:2] + self.mean_raw[:2]
        previous_phi, _ = self.polar(previous)
        phi, log_radius = self.polar(current)
        velocity = torch.remainder(phi - previous_phi + math.pi, 2 * math.pi) - math.pi
        speed = velocity / self.speed_scale
        sine, cosine = phi.sin(), phi.cos()
        a3, a4, a5 = current_z[:, 2:].unbind(1)
        features = torch.stack((
            torch.ones_like(phi), speed, speed.square(), speed.pow(3), sine, cosine,
            (2 * phi).sin(), (2 * phi).cos(), speed * sine, speed * cosine, log_radius, a3, a4, a5,
        ), dim=1)
        state = torch.stack((velocity, log_radius), dim=1)
        next_state = (state + features @ self.basis_coef).clamp(self.bounds[0], self.bounds[1])
        next_angle = phi + next_state[:, 0]
        next_pair = next_state[:, 1, None].exp() * torch.stack((next_angle.cos(), next_angle.sin()), dim=1)
        return (next_pair - current) / self.scale_raw[:2]

    def forward(self, history, steps, noise=None):
        _, hidden = self.recurrence(history[:, :-1])
        previous, value = history[:, -2], history[:, -1]
        predictions = []
        for step in range(steps):
            output, hidden = self.recurrence(value[:, None], hidden)
            residual = self.readout(output[:, 0])
            phase = PRIOR_WEIGHT * self.phase_increment(previous, value)
            increment = torch.cat((residual[:, :2] + phase, residual[:, 2:]), dim=1)
            previous, value = value, value + increment
            if noise is not None:
                value = value + noise[:, step]
            predictions.append(value)
        return torch.stack(predictions, dim=1)


def fit_arm(model, arm, worms, seed, epochs, batches, train, validation, output):
    rng = np.random.default_rng(seed)
    val_input = torch.from_numpy(validation[:, :HISTORY])
    val_output = torch.from_numpy(validation[:, HISTORY:])
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.002, weight_decay=0.0001)
    best_score = float("inf")
    log = []
    for epoch in range(1, epochs + 1):
        model.train()
        losses = []
        for _ in range(batches):
            choices = rng.integers(len(train), size=BATCH)
            batch = torch.from_numpy(take_windows(worms, train[choices], HISTORY + TRAIN_STEPS))
            forecast = model(batch[:, :HISTORY], TRAIN_STEPS)
            loss = (forecast - batch[:, HISTORY:]).square().mean()
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            losses.append(loss.item())
        model.eval()
        with torch.no_grad():
            validation_mse = (model(val_input, 16) - val_output).square().mean().item()
        training_mse = float(np.mean(losses))
        if not np.isfinite(training_mse) or not np.isfinite(validation_mse):
            raise RuntimeError(f"Non-finite training result for {arm}, seed {seed}, epoch {epoch}")
        row = {"arm": arm, "seed": seed, "epoch": epoch, "train_rollout_mse": training_mse,
               "validation_rollout_mse": validation_mse}
        log.append(row)
        print(f"{arm} seed={seed} epoch={epoch:02d} train={training_mse:.5f} val={validation_mse:.5f}", flush=True)
        if validation_mse < best_score:
            best_score, best_epoch = validation_mse, epoch
            best_weights = {key: value.detach().clone() for key, value in model.state_dict().items()}
    model.load_state_dict(best_weights)
    residual = validation[:, HISTORY] - model.rollout(validation[:, :HISTORY], 1)[:, 0]
    model.noise = np.cov(residual.T)
    filename = "gru" if arm == ARMS[0] else "phase_embedded_gru"
    torch.save({"weights": best_weights, "innovation_covariance": torch.tensor(model.noise),
                "seed": seed, "epoch": best_epoch, "arm": arm}, output / f"{filename}_seed_{seed}.pt")
    fit = {"arm": arm, "seed": seed, "epoch": best_epoch, "validation_mse": best_score,
           "parameters": sum(parameter.numel() for parameter in model.parameters())}
    return log, fit


def phase_summary(x):
    radius = np.hypot(x[..., 0], x[..., 1])
    phi = np.arctan2(x[..., 1], x[..., 0])
    valid = radius >= MIN_RADIUS
    valid_step = valid[:, 1:] & valid[:, :-1]
    signed_speed = wrap(phi[:, 1:] - phi[:, :-1])[valid_step] * FPS / (2 * np.pi)
    coherence = []
    for lag in COHERENCE_LAGS:
        valid_lag = valid[:, lag:] & valid[:, :-lag]
        if not valid_lag.any():
            raise ValueError("None of the endpoint phases is defined at a requested lag.")
        difference = (phi[:, lag:] - phi[:, :-lag])[valid_lag]
        coherence.append(np.exp(1j * difference).mean())
    return radius, signed_speed, np.array(coherence)


def phase_row(prediction, truth, mean, scale, reference, curvature_threshold):
    physical = prediction * scale + mean
    radius, speed, coherence = phase_summary(physical)
    true_radius, true_speed, true_coherence = reference
    row = rollout_statistics(prediction, mean, scale, curvature_threshold)
    row.update({
        "radius_wasserstein": wasserstein_distance(true_radius.ravel(), radius.ravel()),
        "signed_phase_speed_wasserstein_cycles_per_s": wasserstein_distance(true_speed, speed),
        "coherence_mean_abs_error": np.abs(coherence - true_coherence).mean(),
        "low_amplitude_fraction": (radius < MIN_RADIUS).mean(),
        "a3_wasserstein": wasserstein_distance(truth[..., 2].ravel(), physical[..., 2].ravel()),
    })
    for lag, value in zip(COHERENCE_LAGS, coherence):
        row[f"coherence_real_lag{lag}"] = value.real
        row[f"coherence_imag_lag{lag}"] = value.imag
    return row


def summarize(metrics, statistics):
    rmse = {}
    for arm in ARMS:
        rmse[arm] = {
            str(horizon): float(np.mean([row["rmse_z"] for row in metrics
                                        if row["model"] == arm and row["frames"] == horizon]))
            for horizon in HORIZONS
        }
    phase_metrics = ("radius_wasserstein", "signed_phase_speed_wasserstein_cycles_per_s",
                     "coherence_mean_abs_error", "a3_wasserstein")
    statistical = {}
    for mode in ("deterministic", "stochastic"):
        statistical[mode] = {
            arm: {
                name: float(np.mean([row[name] for row in statistics
                                     if row["arm"] == arm and row["rollout"] == mode]))
                for name in phase_metrics
            }
            for arm in ARMS
        }
    return {"rmse_z_equal_worm_and_seed": rmse, "distribution_metrics_equal_worm_and_seed": statistical}


def run(out, epochs, batches):
    out.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(2)
    torch.use_deterministic_algorithms(True)
    source = read_worms()
    worms, mean, scale = standardize(source)
    train = window_index(worms, SPLIT["train"], HISTORY, TRAIN_STEPS, stride=2)
    val = window_index(worms, SPLIT["validation"], HISTORY, 16, stride=29)
    validation = take_windows(worms, val, HISTORY + 16)
    test = window_index(worms, SPLIT["test"], HISTORY, max(HORIZONS), stride=32)
    batch = take_windows(worms, test, HISTORY + max(HORIZONS))
    inputs, truth = batch[:, :HISTORY], batch[:, HISTORY:]
    masks = {animal: test[:, 0] == animal for animal in SPLIT["test"]}
    examples = np.array([np.flatnonzero(mask)[0] for mask in masks.values()])
    true_physical = truth * scale + mean
    reference = {animal: phase_summary(true_physical[mask]) for animal, mask in masks.items()}
    threshold = float(np.nanquantile(np.abs(np.concatenate(
        [source[animal][:, 2] for animal in SPLIT["train"]])), 0.9))
    prior = PhaseDynamics(mean, scale).fit(worms, SPLIT["train"])
    np.savez_compressed(out / "phase_prior.npz", coefficients=prior.coef, bounds=prior.bounds,
                        speed_scale=prior.speed_scale, covariance=prior.covariance,
                        feature_names=np.array(PHASE_FEATURES), mean=mean, scale=scale)
    np.save(out / "test_windows.npy", test)

    config = {
        "arms": ARMS, "split": SPLIT, "history_frames": HISTORY, "fps": FPS,
        "train_steps": TRAIN_STEPS, "validation_steps": 16, "horizon_frames": HORIZONS,
        "epochs": epochs, "batches_per_epoch": batches, "batch_size": BATCH, "seeds": SEEDS,
        "train_stride": 2, "validation_stride": 29, "test_stride": 32,
        "optimizer": {"name": "AdamW", "learning_rate": 0.002, "weight_decay": 0.0001},
        "gradient_clip": 1.0, "phase_drift_weight": PRIOR_WEIGHT, "phase_min_amplitude": MIN_RADIUS,
        "phase_prior_ridge_penalty": 10.0, "phase_state_bounds_quantiles": [0.0025, 0.9975],
        "phase_coherence_lags": COHERENCE_LAGS, "train_loss": "mean eight-step standardized MSE",
        "checkpoint_criterion": "minimum sixteen-step validation standardized MSE",
        "stochastic_rollouts": "one Gaussian innovation sequence per held-out window and seed",
        "stochastic_innovation_covariance": "estimated separately per checkpoint using one-step validation residuals",
        "innovation_rng_seed": SAMPLE_SEED, "torch_version": str(torch.__version__), "device": "CPU",
        "windows": {"train": len(train), "validation": len(val),
                    "test": {animal: int(mask.sum()) for animal, mask in masks.items()}},
        "demo_examples": [{"worm": int(test[i, 0]), "start_frame_zero_based": int(test[i, 1])} for i in examples],
        "a3_threshold_train_abs_q90": threshold,
    }
    (out / "config.json").write_text(json.dumps(config, indent=2) + "\n")

    logs, fits, metrics, stats = [], [], [], []
    deterministic, sampled = {}, {}
    for seed in SEEDS:
        for arm in ARMS:
            torch.manual_seed(seed)
            model = PostureGRU() if arm == ARMS[0] else PhaseEmbeddedGRU(prior)
            log, fit = fit_arm(model, arm, worms, seed, epochs, batches, train, validation, out)
            logs.extend(log)
            fits.append(fit)
            write_csv(out / "training.csv", logs)
            prediction = model.rollout(inputs, max(HORIZONS))
            sample = model.rollout(inputs, max(HORIZONS), np.random.default_rng(SAMPLE_SEED))
            if not (np.isfinite(prediction).all() and np.isfinite(sample).all()):
                raise RuntimeError(f"Non-finite test forecast for {arm}, seed {seed}")
            deterministic[(arm, seed)], sampled[(arm, seed)] = prediction, sample
            metrics.extend(dict(seed=seed, **row) for row in scores(truth, prediction, masks, scale, arm))
            for label, trajectories in (("deterministic", prediction), ("stochastic", sample)):
                for animal, mask in masks.items():
                    stats.append(dict(arm=arm, seed=seed, worm=animal, rollout=label,
                                      **phase_row(trajectories[mask], true_physical[mask], mean, scale,
                                                  reference[animal], threshold)))
    for animal, mask in masks.items():
        stats.append(dict(arm="Observation", seed="", worm=animal, rollout="observation",
                          **phase_row(truth[mask], true_physical[mask], mean, scale, reference[animal], threshold)))
    write_csv(out / "metrics_per_worm.csv", metrics)
    write_csv(out / "rollout_statistics.csv", stats)

    def pack(runs):
        return np.array([[runs[(arm, seed)] for seed in SEEDS] for arm in ARMS])

    prediction, sample = pack(deterministic), pack(sampled)
    np.savez_compressed(out / "test_rollouts.npz", arms=ARMS, seeds=SEEDS, mean=mean, scale=scale,
                        history=inputs, truth=truth, deterministic=prediction, stochastic=sample)
    np.savez_compressed(out / "example_rollouts.npz", arms=ARMS, seeds=SEEDS,
                        worms=test[examples, 0], start_frames=test[examples, 1],
                        history=inputs[examples] * scale + mean, truth=true_physical[examples],
                        deterministic=prediction[:, :, examples] * scale + mean,
                        stochastic=sample[:, :, examples] * scale + mean)
    summary = summarize(metrics, stats)
    summary["fits"] = fits
    (out / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=ARTIFACTS / "hybrid")
    parser.add_argument("--epochs", type=int, default=12)
    parser.add_argument("--batches", type=int, default=80)
    args = parser.parse_args()
    if args.epochs < 1 or args.batches < 1:
        parser.error("--epochs and --batches must be positive")
    with threadpool_limits(limits=2):
        run(args.out, args.epochs, args.batches)


if __name__ == "__main__":
    main()
