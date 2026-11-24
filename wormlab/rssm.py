"""Train an observation-only categorical RSSM on the common posture benchmark."""

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from .data import ARTIFACTS, HISTORY, MODES, SPLIT, read_worms, standardize, take_windows, window_index
from .experiment import BATCH, HORIZONS, rollout_statistics, scores, write_csv

GROUPS = 8
CATEGORIES = 8
WIDTH = 64
SEED = 31
FRAMES = 32
SAMPLES = 24


class CategoricalRSSM(nn.Module):
    """Dreamer-style recurrent transition, categorical prior/posterior and decoder."""

    def __init__(self):
        super().__init__()
        latent_size = GROUPS * CATEGORIES
        self.dynamics = nn.GRUCell(latent_size, WIDTH)
        self.prior = nn.Linear(WIDTH, latent_size)
        self.posterior = nn.Sequential(nn.Linear(WIDTH + MODES, WIDTH), nn.SiLU(),
                                       nn.Linear(WIDTH, latent_size))
        self.decoder = nn.Sequential(nn.Linear(WIDTH + latent_size, WIDTH), nn.SiLU(),
                                     nn.Linear(WIDTH, MODES))

    def initial(self, reference):
        return reference.new_zeros(len(reference), WIDTH), reference.new_zeros(
            len(reference), GROUPS * CATEGORIES)

    @staticmethod
    def distribution(logits):
        return 0.99 * logits.unflatten(-1, (GROUPS, CATEGORIES)).softmax(-1) + 0.01 / CATEGORIES

    @staticmethod
    def draw(probs, generator=None):
        indices = torch.multinomial(probs.flatten(0, 1), 1, generator=generator)
        sampled = F.one_hot(indices.squeeze(-1), CATEGORIES).reshape_as(probs).to(probs.dtype)
        return (sampled + (probs - probs.detach())).flatten(1)

    @staticmethod
    def kl(q, p):
        return (q * (q.log() - p.log())).sum(dim=(-2, -1)).clamp_min(1.0).mean()

    def observe(self, observations):
        hidden, latent = self.initial(observations)
        reconstructions, dynamics_kl, representation_kl = [], [], []
        for frame in observations.unbind(1):
            hidden = self.dynamics(latent, hidden)
            prior = self.distribution(self.prior(hidden))
            posterior = self.distribution(self.posterior(torch.cat((hidden, frame), dim=-1)))
            latent = self.draw(posterior)
            reconstructions.append(self.decoder(torch.cat((hidden, latent), dim=-1)))
            dynamics_kl.append(self.kl(posterior.detach(), prior))
            representation_kl.append(self.kl(posterior, prior.detach()))
        reconstruction = F.mse_loss(torch.stack(reconstructions, dim=1), observations)
        dynamics = torch.stack(dynamics_kl).mean()
        representation = torch.stack(representation_kl).mean()
        return reconstruction + 0.5 * dynamics + 0.1 * representation, reconstruction, dynamics

    @torch.no_grad()
    def rollout(self, history, steps, samples, seed):
        context = torch.as_tensor(history, dtype=torch.float32).repeat_interleave(samples, dim=0)
        generator = torch.Generator().manual_seed(seed)
        hidden, latent = self.initial(context)
        for frame in context.unbind(1):
            hidden = self.dynamics(latent, hidden)
            posterior = self.distribution(self.posterior(torch.cat((hidden, frame), dim=-1)))
            latent = self.draw(posterior, generator)
        result = []
        for _ in range(steps):
            hidden = self.dynamics(latent, hidden)
            latent = self.draw(self.distribution(self.prior(hidden)), generator)
            prediction = self.decoder(torch.cat((hidden, latent), dim=-1))
            result.append(prediction.reshape(len(history), samples, MODES).mean(dim=1).numpy())
        return np.stack(result, axis=1)


def predict(model, x, steps, samples, seed):
    return np.concatenate([
        model.rollout(batch, steps, samples, seed + offset)
        for offset, batch in enumerate(np.array_split(x, (len(x) + 255) // 256))
    ])


def run(benchmark, out, epochs, batches):
    out.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(2)
    torch.use_deterministic_algorithms(True)
    torch.manual_seed(SEED)
    rng = np.random.default_rng(SEED)
    raw = read_worms()
    worms, mean, scale = standardize(raw)
    train_index = window_index(worms, SPLIT["train"], HISTORY, FRAMES - HISTORY, stride=2)
    validation_index = window_index(worms, SPLIT["validation"], HISTORY, 16, stride=127)
    validation = take_windows(worms, validation_index, HISTORY + 16)
    model = CategoricalRSSM()
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.001, weight_decay=0.0001)
    log = []
    best = float("inf")
    for epoch in range(1, epochs + 1):
        model.train()
        train_metrics = []
        for _ in range(batches):
            index = train_index[rng.integers(len(train_index), size=BATCH)]
            frames = torch.from_numpy(take_windows(worms, index, FRAMES))
            loss, reconstruction, dynamics = model.observe(frames)
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 10.0)
            optimizer.step()
            train_metrics.append((loss.item(), reconstruction.item(), dynamics.item()))
        model.eval()
        estimate = predict(model, validation[:, :HISTORY], 16, 8, 3100)
        val_loss = float(np.mean((estimate - validation[:, HISTORY:]) ** 2))
        objective, reconstruction, dynamics = np.mean(train_metrics, axis=0)
        log.append(dict(epoch=epoch, train_objective=objective, train_reconstruction_mse=reconstruction,
                        train_dynamics_kl_free_nats=dynamics, validation_rollout_mse=val_loss))
        print(f"RSSM epoch={epoch:02d} obj={objective:.5f} recon={reconstruction:.5f} "
              f"KL={dynamics:.4f} val={val_loss:.5f}", flush=True)
        if val_loss < best:
            best, best_epoch = val_loss, epoch
            weights = {name: value.detach().clone() for name, value in model.state_dict().items()}
    model.load_state_dict(weights)
    torch.save({"weights": weights, "epoch": best_epoch, "seed": SEED}, out / "rssm.pt")
    write_csv(out / "training.csv", log)
    test = np.load(benchmark / "test_windows.npy")
    windows = take_windows(worms, test, HISTORY + max(HORIZONS))
    context, truth = windows[:, :HISTORY], windows[:, HISTORY:]
    masks = {animal: test[:, 0] == animal for animal in SPLIT["test"]}
    point = predict(model, context, max(HORIZONS), SAMPLES, 5000)
    sample = predict(model, context, max(HORIZONS), 1, 6100)
    if not np.isfinite(point).all() or not np.isfinite(sample).all():
        raise RuntimeError("Non-finite RSSM rollout")
    benchmark_run = json.loads((benchmark / "run.json").read_text())
    threshold = benchmark_run["a3_threshold_train_abs_q90"]
    write_csv(out / "metrics_per_worm.csv", scores(truth, point, masks, scale, "RSSM"))
    write_csv(out / "rollout_statistics.csv", [
        dict(model="RSSM", worm=animal, **rollout_statistics(sample[mask], mean, scale, threshold))
        for animal, mask in masks.items()
    ])
    example = int(np.load(benchmark / "forecast_examples.npz")["example_index"])
    np.savez_compressed(out / "rollouts.npz", forecast=point, sample=sample,
                        example=sample[example] * scale + mean)
    summary = {"seed": SEED, "epochs": epochs, "batches_per_epoch": batches, "batch_size": BATCH,
               "training_frames": FRAMES, "prior_samples_for_rmse": SAMPLES,
               "categorical_groups": GROUPS, "categories": CATEGORIES, "hidden_width": WIDTH,
               "parameters": sum(p.numel() for p in model.parameters()), "eligible_train_windows": len(train_index),
               "validation_windows": len(validation_index), "best_epoch": best_epoch, "validation_mse": best}
    (out / "run.json").write_text(json.dumps(summary, indent=2) + "\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--benchmark", type=Path, default=ARTIFACTS / "benchmark")
    parser.add_argument("--out", type=Path, default=ARTIFACTS / "rssm")
    parser.add_argument("--epochs", type=int, default=14)
    parser.add_argument("--batches", type=int, default=70)
    args = parser.parse_args()
    if args.epochs < 1 or args.batches < 1:
        parser.error("--epochs and --batches must be positive")
    run(args.benchmark, args.out, args.epochs, args.batches)


if __name__ == "__main__":
    main()
