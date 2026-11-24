"""Read the adult hermaphrodite Cook adjacency sheets with cell names intact."""

import csv
import json

import numpy as np
from openpyxl import load_workbook

from .data import ARTIFACTS, RAW

COOK = RAW / "cook2019_corrected2020.xlsx"


def read_cook(path=COOK):
    workbook = load_workbook(path, read_only=True, data_only=True)
    matrices = {}
    try:
        for kind, sheet in (("chemical", "hermaphrodite chemical"),
                            ("gap", "hermaphrodite gap jn symmetric")):
            rows = list(workbook[sheet].values)
            columns = [(i, name.strip()) for i, name in enumerate(rows[2][3:], 3)
                       if isinstance(name, str) and name.strip()]
            entries = [(row[2].strip(), row) for row in rows[3:]
                       if isinstance(row[2], str) and row[2].strip()]
            targets = [name for _, name in columns]
            sources = [name for name, _ in entries]
            if len(set(sources)) != len(sources) or len(set(targets)) != len(targets):
                raise ValueError(f"Duplicate cell labels in {sheet}")
            matrix = np.array([[row[i] or 0 for i, _ in columns] for _, row in entries], dtype=float)
            if np.any(matrix < 0) or not np.isfinite(matrix).all():
                raise ValueError(f"Unexpected adjacency values in {sheet}")
            matrices[kind] = (sources, targets, matrix)
    finally:
        workbook.close()
    pre, post, gap = matrices["gap"]
    if pre != post or not np.allclose(gap, gap.T):
        raise ValueError("Gap junction sheet is expected to be symmetric.")
    return matrices


def reorder(sheet, names):
    pre, post, weights = sheet
    return weights[np.ix_([pre.index(name) for name in names], [post.index(name) for name in names])]


def export_connectome(out=ARTIFACTS / "connectome"):
    out.mkdir(parents=True, exist_ok=True)
    matrices = read_cook()
    stats = {}
    for kind, (sources, targets, weights) in matrices.items():
        # Export each undirected gap edge once, keeping any reported self contacts.
        edges = np.triu(weights) if kind == "gap" else weights
        i, j = np.nonzero(edges)
        with (out / f"{kind}_edges.csv").open("w", newline="") as file:
            writer = csv.writer(file)
            writer.writerow(("source_cell", "target_cell", "em_section_weight"))
            writer.writerows((sources[a], targets[b], edges[a, b]) for a, b in zip(i, j))
        np.savez_compressed(out / f"{kind}.npz", sources=sources, targets=targets, weights=weights)
        stats[kind] = {"sources": len(sources), "targets": len(targets), "nonzero_edges": len(i),
                       "nonzero_diagonal": sum(sources[a] == targets[b] for a, b in zip(i, j)),
                       "total_exported_weight": float(edges.sum())}
    (out / "summary.json").write_text(json.dumps(stats, indent=2) + "\n")
    print(json.dumps(stats), flush=True)


if __name__ == "__main__":
    export_connectome()
