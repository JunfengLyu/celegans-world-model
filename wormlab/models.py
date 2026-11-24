"""Autoregressive posture dynamics with a shared history-to-rollout interface."""

import numpy as np
import torch
from scipy.sparse import csr_matrix
from sklearn.cluster import MiniBatchKMeans
from torch import nn

from .data import FPS, HISTORY, MODES, take_windows, window_index

PHASE_FEATURES = (
    "intercept", "omega", "omega^2", "omega^3", "sin(phi)", "cos(phi)",
    "sin(2phi)", "cos(2phi)", "omega sin(phi)", "omega cos(phi)",
    "log radius", "a3", "a4", "a5",
)


def ridge(features, target, penalty):
    gram = features.T @ features
    return np.linalg.solve(gram + penalty * np.eye(gram.shape[0]), features.T @ target)


class Persistence:
    def rollout(self, history, steps, rng=None):
        return np.repeat(history[:, -1:, :], steps, axis=1)


class LinearAR:
    def fit(self, worms, ids):
        index = window_index(worms, ids, HISTORY, 1, stride=2)
        windows = take_windows(worms, index, HISTORY + 1)
        past = windows[:, :-1].reshape(len(windows), -1).astype(np.float64)
        self.coef = ridge(np.column_stack((past, np.ones(len(past)))), windows[:, -1], 1.0)
        return self

    def rollout(self, history, steps, rng=None):
        predictions = []
        for _ in range(steps):
            latest = np.column_stack((history.reshape(len(history), -1), np.ones(len(history)))) @ self.coef
            predictions.append(latest)
            history = np.concatenate((history[:, 1:], latest[:, None]), axis=1)
        return np.stack(predictions, axis=1)


class PostureGRU(nn.Module):
    def __init__(self, width=48):
        super().__init__()
        self.recurrence = nn.GRU(MODES, width, batch_first=True)
        self.readout = nn.Linear(width, MODES)
        self.noise = None

    def forward(self, history, steps, noise=None):
        _, hidden = self.recurrence(history[:, :-1])
        value = history[:, -1]
        predictions = []
        for step in range(steps):
            output, hidden = self.recurrence(value[:, None], hidden)
            value = value + self.readout(output[:, 0])
            if noise is not None:
                value = value + noise[:, step]
            predictions.append(value)
        return torch.stack(predictions, dim=1)

    @torch.no_grad()
    def rollout(self, history, steps, rng=None):
        noise = None
        if rng is not None:
            if self.noise is None:
                raise ValueError("Fit one-step innovation covariance before sampling.")
            draws = rng.multivariate_normal(np.zeros(MODES), self.noise, size=(len(history), steps))
            noise = torch.as_tensor(draws, dtype=torch.float32)
        return self(torch.as_tensor(history, dtype=torch.float32), steps, noise).numpy()


def wrap(angle):
    return (angle + np.pi) % (2 * np.pi) - np.pi


class PhaseDynamics:
    """Discrete Langevin-style model on phase, angular speed and shape envelope."""

    def __init__(self, mean, scale):
        self.mean = mean
        self.scale = scale

    def encode(self, z):
        physical = z * self.scale + self.mean
        phi = np.arctan2(physical[..., 1], physical[..., 0])
        radius = np.hypot(physical[..., 0], physical[..., 1])
        last = z.shape[-2] - 1
        omega = wrap(phi[..., last] - phi[..., last - 1])
        shape = z[..., last, 2:]
        return phi[..., last], np.concatenate(
            (np.stack((omega, np.log(np.maximum(radius[..., last], 0.05))), axis=-1), shape), axis=-1
        )

    def features(self, phi, state):
        omega, log_radius, a3, a4, a5 = state.T
        speed = omega / self.speed_scale
        sin, cos = np.sin(phi), np.cos(phi)
        return np.column_stack((
            np.ones(len(phi)), speed, speed ** 2, speed ** 3, sin, cos,
            np.sin(2 * phi), np.cos(2 * phi), speed * sin, speed * cos,
            log_radius, a3, a4, a5,
        ))

    def fit(self, worms, ids):
        index = window_index(worms, ids, 2, 1, stride=2)
        values = take_windows(worms, index, 3)
        phi, state = self.encode(values[:, :2])
        _, next_state = self.encode(values[:, 1:])
        self.speed_scale = np.std(state[:, 0])
        self.bounds = np.quantile(state, [0.0025, 0.9975], axis=0)
        keep = ((state > self.bounds[0]) & (state < self.bounds[1])).all(axis=1)
        state, next_state = state[keep], next_state[keep]
        features = self.features(phi[keep], state)
        delta = next_state - state
        self.coef = ridge(features, delta, 10.0)
        residual = delta - features @ self.coef
        self.covariance = np.stack([
            np.cov(residual[group].T) + 1e-8 * np.eye(MODES)
            for group in (state[:, 0] < 0, state[:, 0] >= 0)
        ])
        return self

    def rollout(self, history, steps, rng=None):
        phi, state = self.encode(history)
        predictions = []
        for _ in range(steps):
            if rng is None:
                innovation = 0
            else:
                draws = np.stack([
                    rng.multivariate_normal(np.zeros(MODES), cov, size=len(history))
                    for cov in self.covariance
                ])
                innovation = draws[(state[:, 0] >= 0).astype(int), np.arange(len(history))]
            state = np.clip(state + self.features(phi, state) @ self.coef + innovation,
                            self.bounds[0], self.bounds[1])
            phi = wrap(phi + state[:, 0])
            radius = np.exp(state[:, 1])
            first = (np.column_stack((radius * np.cos(phi), radius * np.sin(phi))) -
                     self.mean[:2]) / self.scale[:2]
            predictions.append(np.column_stack((first, state[:, 2:])))
        return np.stack(predictions, axis=1)

    def angular_acceleration(self, phi, speed_radians_per_second, radius):
        state = np.column_stack((speed_radians_per_second / FPS, np.full(len(phi), np.log(radius)),
                                 np.zeros((len(phi), 3))))
        return (self.features(phi, state) @ self.coef)[:, 0] * FPS ** 2


class DelayMarkov:
    """Coarse Markov chain with empirical posture paths over each lag interval."""

    def __init__(self, clusters=256, lag=12):
        self.clusters = clusters
        self.lag = lag

    def fit(self, worms, ids):
        index = window_index(worms, ids, HISTORY, self.lag, stride=2)
        windows = take_windows(worms, index, HISTORY + self.lag)
        origin = windows[:, :HISTORY].reshape(len(windows), -1)
        target = windows[:, -HISTORY:].reshape(len(windows), -1)
        self.cluster = MiniBatchKMeans(self.clusters, random_state=23, n_init=3,
                                       batch_size=4096, max_iter=80).fit(origin)
        start = self.cluster.labels_
        end = self.cluster.predict(target)
        counts = csr_matrix((np.ones(len(start)), (start, end)), shape=(self.clusters, self.clusters))
        visits = np.asarray(counts.sum(axis=1)).ravel()
        if np.any(visits == 0):
            raise RuntimeError("Empty Markov state. Reduce the cluster count.")
        self.transition = counts.multiply(1 / visits[:, None]).tocsr()
        self.snippets = windows[:, HISTORY:]
        self.ends = end
        order = np.argsort(start, kind="stable")
        self.by_state = np.split(order, np.cumsum(visits.astype(int))[:-1])
        self.emissions = np.stack([self.snippets[rows].mean(axis=0) for rows in self.by_state])
        return self

    def rollout(self, history, steps, rng=None):
        labels = self.cluster.predict(history.reshape(len(history), -1).astype(np.float32))
        blocks = []
        if rng is None:
            probability = np.eye(self.clusters)[labels]
            for _ in range(0, steps, self.lag):
                blocks.append((probability @ self.emissions.reshape(self.clusters, -1))
                              .reshape(len(history), self.lag, MODES))
                probability = (self.transition.T @ probability.T).T
        else:
            for _ in range(0, steps, self.lag):
                rows = np.array([rng.choice(self.by_state[label]) for label in labels])
                blocks.append(self.snippets[rows])
                labels = self.ends[rows]
        return np.concatenate(blocks, axis=1)[:, :steps]
