"""Build the authors' 2026 chemotaxis simulator and rerun one released trial."""

from concurrent.futures import ThreadPoolExecutor
from hashlib import sha1
import json
import os
from pathlib import Path
import subprocess
import tempfile
from urllib.parse import quote

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "external" / "chemotaxis-source"
OUT = ROOT / "artifacts" / "chemotaxis"
REPO = "adasquires/celegans-chemotaxis"
REV = "5f639a23d546093f1b278fa5d9c96fdc99da5f49"
TRIAL = "data/fullSimOutputs/angleSweep_conicalGradient_foodPos=25,0_duration=300_step=0.01/120.00000"
SOURCES = ("Collide", "Muscles", "NervousSystem", "StretchReceptor", "Worm", "WormBody", "random")
REFERENCE_FILES = {f"{TRIAL}/{name}" for name in ("params.json", "coll_objs.tsv", "body.dat")}


def curl(url, *args):
    return subprocess.check_output(["curl", "-fsSL", "--retry", "2", "--max-time", "60", url, *args])


def git_blob(path):
    data = path.read_bytes()
    return sha1(f"blob {len(data)}\0".encode() + data).hexdigest()


def fetch_source():
    tree = json.loads(curl(f"https://api.github.com/repos/{REPO}/git/trees/{REV}?recursive=1"))
    if tree["sha"] != REV or tree["truncated"]:
        raise ValueError("Expected complete pinned Git tree")
    requested = {"consts.h", "main.h", "singlerun.cpp"} | REFERENCE_FILES
    files = [entry for entry in tree["tree"] if entry["type"] == "blob" and
             (entry["path"].startswith("modules/") or entry["path"] in requested)]
    if not requested <= {entry["path"] for entry in files}:
        raise FileNotFoundError("Pinned source/config/reference missing upstream")

    def fetch(entry):
        name = entry["path"]
        target = SOURCE / name
        if target.exists() and git_blob(target) == entry["sha"]:
            return
        target.parent.mkdir(parents=True, exist_ok=True)
        with tempfile.NamedTemporaryFile(dir=target.parent, delete=False) as file:
            pending = Path(file.name)
        try:
            curl(f"https://raw.githubusercontent.com/{REPO}/{REV}/{quote(name)}", "-o", str(pending))
            if git_blob(pending) != entry["sha"]:
                raise ValueError(f"Git blob hash mismatch for {name}")
            pending.replace(target)
        finally:
            pending.unlink(missing_ok=True)

    with ThreadPoolExecutor(max_workers=5) as pool:
        list(pool.map(fetch, files))


def load_body(path):
    matrix = np.loadtxt(path)
    if matrix.ndim != 2 or (matrix.shape[1] - 1) % 3 or not np.isfinite(matrix).all():
        raise ValueError(f"Unexpected simulated body output in {path}")
    return matrix[:, 0], matrix[:, 1:].reshape(len(matrix), -1, 3)[..., :2] * 1000


def main():
    fetch_source()
    OUT.mkdir(parents=True, exist_ok=True)
    executable = OUT / "singlerun"
    compiler = os.environ.get("CXX", "c++")
    with (OUT / "build.log").open("w") as log:
        subprocess.run([compiler, "-std=c++17", "-O2", "singlerun.cpp",
                        *[f"modules/{name}.cpp" for name in SOURCES], "-o", str(executable)],
                       cwd=SOURCE, stdout=log, stderr=subprocess.STDOUT, check=True)

    reference = SOURCE / TRIAL
    configuration = json.loads((reference / "params.json").read_text())
    food = np.array([configuration["ChemoReceptors"]["foodPos"][axis] for axis in ("x", "y")]) * 1000
    flat_gradient = OUT / "flat_gradient.json"
    # Preserve receptor initialization and remove the spatial input by setting C(x, y) = 0.
    configuration["ChemoReceptors"]["kappa"] = 0.0
    flat_gradient.write_text(json.dumps(configuration, indent=2) + "\n")
    runs = {}
    for label, params in (("published_gradient", reference / "params.json"),
                          ("flat_gradient", flat_gradient)):
        destination = OUT / label
        destination.mkdir(exist_ok=True)
        with (OUT / f"{label}.log").open("w") as log:
            subprocess.run([str(executable), "--params", str(params),
                            "--coll", str(reference / "coll_objs.tsv"), "--output", f"{destination}/"],
                           cwd=destination, stdout=log, stderr=subprocess.STDOUT, check=True)
        runs[label] = load_body(destination / "body.dat")
    time, xy = runs["published_gradient"]
    sample_time, sample_xy = load_body(reference / "body.dat")
    if not np.array_equal(time, sample_time) or xy.shape != sample_xy.shape:
        raise ValueError("Authors' and rerun frame grids differ")
    summary = {
        "commit": REV, "trial": TRIAL, "seed": configuration["simulation"]["seed"],
        "angle_initial_deg": np.degrees(configuration["simulation"]["angle"]),
        "body_frames": len(time), "simulation_dt_s": 0.01,
        "reference_body_xy_rmse_mm": float(np.sqrt(np.mean((xy - sample_xy) ** 2))),
        "reference_body_xy_max_abs_mm": float(np.max(np.abs(xy - sample_xy))),
        "reference_xy_rmse_through_s": {
            str(limit): float(np.sqrt(np.mean((xy[time <= limit] - sample_xy[time <= limit]) ** 2)))
            for limit in (10, 60, 120, 300)
        },
        "released_reference_end_centroid_mm": sample_xy[-1].mean(axis=0).tolist(),
    }
    fig, ax = plt.subplots(figsize=(7.5, 5), layout="constrained")
    reference_path = sample_xy.mean(axis=1)
    ax.plot(*reference_path.T, color="#6b7280", lw=1.1, ls="--", label="Released reference")
    for (label, (_, body)), color, title in zip(
            runs.items(), ("#18787a", "#b45238"), ("Rerun · published gradient", "Rerun · flat gradient")):
        path = body.mean(axis=1)
        distance = np.linalg.norm(path - food, axis=1)
        summary[label] = {"start_distance_mm": distance[0], "end_distance_mm": distance[-1],
                          "minimum_distance_mm": distance.min(), "end_centroid_mm": path[-1].tolist()}
        ax.plot(*path.T, color=color, lw=1.0, label=title)
        ax.scatter(*path[-1], marker="x", s=45, color=color)
    ax.scatter(*food, color="#a87f00", marker="*", s=115, label="Attractant location")
    ax.set(xlabel="x (mm)", ylabel="y (mm)", aspect="equal",
           title="Chemotaxis circuit · published 120° trial and ablation")
    ax.legend(frameon=False, fontsize=9)
    ax.grid(alpha=0.18)
    fig.savefig(OUT / "chemotaxis_paths.png", dpi=170)
    plt.close(fig)
    (OUT / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
