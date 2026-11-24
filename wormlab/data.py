"""Load the released eigenworm data and construct within-animal windows."""

from pathlib import Path

import h5py
import numpy as np
from scipy.io import loadmat

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw"
ARTIFACTS = ROOT / "artifacts"
FPS = 16
HISTORY = 12
MODES = 5
# IDs are one-based positions in the published MATLAB cell array, not specimen labels.
SPLIT = {"train": (1, 2, 4, 5, 7, 8, 10, 11), "validation": (3, 9), "test": (6, 12)}


def read_worms(path=RAW / "crawl.mat"):
    with h5py.File(path, "r") as file:
        worms = {
            index: np.asarray(file[ref][:MODES].T, dtype=np.float32)
            for index, ref in enumerate(file["tr"][...].ravel(), 1)
        }
    if len(worms) != 12 or any(worm.shape != (33600, MODES) for worm in worms.values()):
        raise ValueError("Expected twelve 33,600-frame foraging recordings.")
    return worms


def read_basis(path=RAW / "EigenWorms.mat"):
    variables = loadmat(path)
    basis = np.asarray(variables["EigenWorms"], dtype=np.float64)
    if basis.shape[1] < MODES:
        raise ValueError(f"Unexpected eigenworm basis shape {basis.shape}")
    return basis[:, :MODES]


def valid_windows(worm, past, future, stride=1):
    valid = np.isfinite(worm).all(axis=1)
    return np.flatnonzero(np.lib.stride_tricks.sliding_window_view(valid, past + future)
                          .all(axis=1)[::stride]) * stride


def window_index(worms, ids, past, future, stride=1):
    return np.concatenate([
        np.column_stack((np.full(starts.size, animal), starts))
        for animal in ids
        if (starts := valid_windows(worms[animal], past, future, stride)).size
    ])


def take_windows(worms, index, length):
    return np.stack([worms[animal][start:start + length] for animal, start in index])


def standardize(worms):
    train = np.concatenate([worms[animal] for animal in SPLIT["train"]])
    train = train[np.isfinite(train).all(axis=1)]
    mean, scale = train.mean(axis=0), train.std(axis=0)
    return {animal: (worm - mean) / scale for animal, worm in worms.items()}, mean, scale


def centerlines(coefficients, basis):
    """Integrate body tangent angles at unit arc length in a body-fixed frame."""
    angles = coefficients @ basis.T
    steps = np.stack((np.cos(angles), np.sin(angles)), axis=-1) / len(basis)
    points = np.concatenate((np.zeros_like(steps[..., :1, :]), steps.cumsum(axis=-2)), axis=-2)
    return points - points.mean(axis=-2, keepdims=True)
