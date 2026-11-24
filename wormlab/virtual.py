"""Run the upstream modWorm Python simulator and inspect BAAIWorm released arrays."""

import argparse
import importlib
import json
import os
import subprocess
import sys
import types
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from threadpoolctl import threadpool_limits

from .connectome import read_cook, reorder
from .data import ARTIFACTS, RAW, ROOT

MODWORM_COMMIT = "10912b4e09c72ba3430573f0b93a48c6e86b5614"
BAAIWORM_COMMIT = "b99ca5ab9c00aef739fceb1cf007bc50b06f2ccc"


class PythonBackend:
    """Accept import-time Julia declarations while running upstream SciPy functions."""

    importing = True

    def eval(self, _):
        if not self.importing:
            raise RuntimeError("This adapter uses modWorm's Python backend.")

    def __getattr__(self, name):
        raise RuntimeError(f"Unexpected Julia backend call in the Python demo: {name}")


def load_modworm(checkout):
    commit = subprocess.check_output(["git", "-C", str(checkout), "rev-parse", "HEAD"], text=True).strip()
    if commit != MODWORM_COMMIT:
        raise ValueError(f"modWorm adapter expects {MODWORM_COMMIT}, found {commit}")
    # The upstream package resolves adjacent assets relative to its import CWD.
    original_cwd = Path.cwd()
    sys.path.insert(0, str(checkout))
    julia_stub = types.ModuleType("julia")
    backend = julia_stub.Main = PythonBackend()
    original_julia = sys.modules.get("julia")
    sys.modules["julia"] = julia_stub
    os.chdir(checkout)
    try:
        return importlib.import_module("modWorm")
    finally:
        backend.importing = False
        os.chdir(original_cwd)
        sys.path.remove(str(checkout))
        if original_julia is None:
            del sys.modules["julia"]
        else:
            sys.modules["julia"] = original_julia


def run_modworm(out):
    checkout = ROOT / "external" / "modWorm"
    package = load_modworm(checkout)
    name_to_index = package.utils.neuron_names_2_inds
    muscle = package.utils.construct_muscle_map_Hall(RAW / "NeuronFixedPoints.xls")
    var_gap, var_syn = package.utils.construct_connectome_Varshney(RAW / "NeuronConnect.xls")
    matrices = read_cook()
    names = package.utils.neuron_names
    delta = checkout / "modWorm" / "data"
    cook_gap = reorder(matrices["gap"], names) + np.load(delta / "conn_gap_adjust_Cook.npy")
    cook_syn = reorder(matrices["chemical"], names) + np.load(delta / "conn_syn_adjust_Cook.npy")
    stimulus = np.load(checkout / "modWorm/presets_input/input_mat_gentle_post_touch.npy")
    scenarios = {
        "varshney_posterior": (var_gap, var_syn, stimulus, 0.01),
        "cook_posterior": (cook_gap, cook_syn, stimulus, 0.001),
        "cook_sham": (cook_gap, cook_syn, np.zeros_like(stimulus), 0.001),
    }

    result = {}
    for label, (gap, syn, external, viscosity) in scenarios.items():
        print(f"Running modWorm {label} ({len(external)} steps)", flush=True)
        np.random.seed(2026)
        nervous = package.predefined_classes_nv.CelegansWorm_NervousSystem_PPC(gap, syn)
        body = package.predefined_classes_mb.CelegansWorm_MuscleBody_PPC(muscle)
        body.fluid_Viscosity = viscosity
        if label.startswith("cook_"):
            # Tutorial 1.4 sets these two conductance maps explicitly.
            interaction = package.network_interactions
            active = np.ones(nervous.network_Size, dtype=bool)
            conductance = np.full_like(gap, 0.1)
            nervous.network_Electrical = interaction.init_network_Electrical(gap, conductance, active)
            nervous.network_Chemical = interaction.init_network_Chemical(
                syn, conductance, package.network_params.CE.ei_map, active)
            interaction.init_vth_Linear(nervous)
        simulation = package.proprioception_simulation.run_network(nervous, body, external)
        voltage = simulation["v_solution"]
        xy = np.stack((simulation["x_solution"], simulation["y_solution"]), axis=-1)
        if (voltage.shape != external.shape or not np.isfinite(voltage).all()
                or not np.isfinite(xy).all() or np.any(np.all(voltage[1:] == 0, axis=1))):
            raise RuntimeError(f"Incomplete numerical integration in {label}")
        # The upstream loop fills force/threshold up to the penultimate frame.
        dt = nervous.timescale
        time = np.arange(len(voltage)) * dt
        centroid = xy.mean(axis=1)
        np.savez_compressed(out / f"{label}.npz", time=time, xy=xy, voltage=voltage,
                            stimulus=external, phi=simulation["phi"],
                            muscle_force=simulation["muscle_force"][:-1],
                            neurons=np.array(names))
        result[label] = {
            "steps": len(voltage), "dt_s": dt, "neurons": len(names),
            "body_segments": body.body_SegCount, "viscosity": viscosity,
            "stimulated_neurons": [name for name, active in zip(names, external.any(axis=0)) if active],
            "peak_abs_preset_current_pA": float(np.abs(external).max()),
            "chemical_nonzero_entries": int(np.count_nonzero(syn)),
            "gap_nonzero_entries": int(np.count_nonzero(gap)),
            "net_centroid_displacement_model_units": float(np.linalg.norm(centroid[-1] - centroid[0])),
            "centroid_path_length_model_units": float(np.linalg.norm(np.diff(centroid, axis=0), axis=1).sum()),
            "final_centroid_xy_model_units": centroid[-1].tolist(),
            "plm_indices": name_to_index(["PLML", "PLMR"]).tolist(),
        }
        print(json.dumps(result[label]), flush=True)

    (out / "modworm.json").write_text(json.dumps({"commit": MODWORM_COMMIT,
                                                  "backend": "upstream Python/SciPy",
                                                  "integrators": {"neurons": "VODE/BDF", "body": "DOPRI5"},
                                                  "scenarios": result}, indent=2) + "\n")


def baaiworm_assets(out):
    directory = RAW / "baaiworm"
    reference = np.load(directory / "gt_eworm.muscle-220428-152601.npy")
    offline = np.load(directory / "video_offline_eworm.muscle-220428-152601.npy")
    stats = {
        "commit": BAAIWORM_COMMIT,
        "mode": "replot of released arrays; sample alignment not established",
        "arrays": {
            name: {"samples": len(data), "channels": data.shape[1], "minimum": float(data.min()),
                   "maximum": float(data.max()), "std": float(data.std())}
            for name, data in (("muscle_target", reference), ("offline_export", offline))
        },
    }
    (out / "baaiworm.json").write_text(json.dumps(stats, indent=2) + "\n")
    fig, axes = plt.subplots(2, 1, figsize=(8, 6), layout="constrained")
    for ax, name, data in zip(axes, ("Published muscle target", "Offline muscle export"), (reference, offline)):
        img = ax.imshow(data.T, aspect="auto", origin="lower", cmap="viridis", interpolation="nearest")
        ax.set(xlabel="Sample index", ylabel="Muscle channel", title=name)
        fig.colorbar(img, ax=ax, label="Released activation scale")
    fig.suptitle("BAAIWorm · released open-loop arrays")
    fig.savefig(out / "baaiworm_muscles.png", dpi=170)
    plt.close(fig)
    print(json.dumps(stats), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("demo", choices=["modworm", "baaiworm", "all"], default="all", nargs="?")
    parser.add_argument("--out", type=Path, default=ARTIFACTS / "virtual")
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    if args.demo in {"baaiworm", "all"}:
        baaiworm_assets(args.out)
    if args.demo in {"modworm", "all"}:
        with threadpool_limits(limits=2):
            run_modworm(args.out)


if __name__ == "__main__":
    main()
