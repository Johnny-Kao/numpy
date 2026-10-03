"""Decision-tree diagnostics for searchsorted coarse-base locality routing.

This script does not benchmark searchsorted. It reconstructs the same coarse
binary-search base state for the synthetic workload matrix and records structural
features used to distinguish false-positive random cases from true local cases.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np

# Importing the shared workload generator should not require a benchmark mode.
# The tournament module reads this at import time, so give diagnostics a
# harmless default before importing it.
os.environ.setdefault("NPY_SEARCHSORTED_RESEARCH_MODE", "current")

from searchsorted_selector_tournament import make_queries

OUT = Path(os.environ.get("SEARCHSORTED_DIAG_OUT", "coarse-decision-tree-diagnostics.json"))


def coarse_bases(n: int, queries: np.ndarray, side: str, levels: int) -> np.ndarray:
    q = np.asarray(queries).reshape(-1)
    base = np.zeros(q.size, dtype=np.int64)
    interval = n
    half = interval >> 1
    interval -= half

    if side == "left":
        take = half < q
    else:
        take = half <= q
    base = take.astype(np.int64) * half

    completed = 1
    while interval > 1 and completed < levels:
        half = interval >> 1
        interval -= half
        pivot = base + half
        if side == "left":
            take = pivot < q
        else:
            take = pivot <= q
        base += take.astype(np.int64) * half
        completed += 1
    return base


def sample_features(base: np.ndarray, observations: int) -> dict[str, object]:
    if base.size < 2:
        sample = base.copy()
    else:
        checks = min(observations, base.size - 1)
        idx = [0]
        prev = 0
        for j in range(1, checks + 1):
            i = (j * (base.size - 1)) // checks
            if i <= prev:
                i = prev + 1
            if i >= base.size:
                i = base.size - 1
            idx.append(i)
            prev = i
        sample = base[np.asarray(idx, dtype=np.int64)]

    if sample.size == 0:
        return {}

    diffs = np.diff(sample)
    absdiff = np.abs(diffs)
    distinct = int(np.count_nonzero(diffs) + 1)
    ties = int(np.count_nonzero(diffs == 0))
    forward = int(np.count_nonzero(diffs > 0))
    backward = int(np.count_nonzero(diffs < 0))
    mn = int(sample.min())
    mx = int(sample.max())
    span = mx - mn
    tv = int(absdiff.sum())
    values, counts = np.unique(sample, return_counts=True)
    concentration = float(counts.max() / sample.size)
    strict = tv == span

    return {
        "sample": sample.tolist(),
        "distinct": distinct,
        "ties": ties,
        "tie_fraction": float(ties / max(1, sample.size - 1)),
        "forward_steps": forward,
        "backward_steps": backward,
        "min_base": mn,
        "max_base": mx,
        "range": span,
        "total_variation": tv,
        "max_step": int(absdiff.max(initial=0)),
        "strict": strict,
        "max_bucket_concentration": concentration,
        "n_unique": int(values.size),
    }


def gate_results(f: dict[str, object]) -> dict[str, bool]:
    strict = bool(f["strict"])
    zero_range = int(f["range"]) == 0
    distinct = int(f["distinct"])
    forward = int(f["forward_steps"])
    ties = int(f["ties"])
    checks = max(1, len(f["sample"]) - 1)

    return {
        "p0_strict": strict,
        "p1_distinct": strict and (zero_range or distinct >= 3),
        "p2_progress": strict and (zero_range or forward * 4 >= checks),
        "p3_ties": strict and (zero_range or ties * 4 <= checks * 3),
        "p4_combined": strict and (
            zero_range
            or (distinct >= 3 and forward * 4 >= checks and ties * 4 <= checks * 3)
        ),
    }


def main() -> None:
    rows: list[dict[str, object]] = []
    for dtype in (np.dtype("int32"), np.dtype("int64"), np.dtype("float64")):
        for n in (1_000_000, 10_000_000):
            for q in (1_024, 8_192, 100_000):
                for shape in (
                    "dense", "medium", "mostly_monotonic", "random",
                    "block_sorted", "reversal_bursts", "duplicates",
                ):
                    try:
                        queries = make_queries(n, q, shape, dtype)
                    except ValueError:
                        continue
                    for layout in ("1d", "2d"):
                        if layout == "2d":
                            rows_count = 8 if q % 8 == 0 else 10
                            qv = queries.reshape(rows_count, q // rows_count)
                        else:
                            qv = queries
                        for side in ("left", "right"):
                            for levels in (3, 4):
                                bases = coarse_bases(n, qv, side, levels)
                                for obs in (8, 16, 32):
                                    f = sample_features(bases, obs)
                                    rows.append({
                                        "dtype": dtype.name,
                                        "n": n,
                                        "q": q,
                                        "shape": shape,
                                        "layout": layout,
                                        "side": side,
                                        "levels": levels,
                                        "observations": obs,
                                        **f,
                                        **gate_results(f),
                                    })

    payload = {"rows": rows}
    OUT.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    for policy in ("p0_strict", "p1_distinct", "p2_progress", "p3_ties", "p4_combined"):
        random_rows = [r for r in rows if r["shape"] == "random"]
        local_rows = [r for r in rows if r["shape"] in {"dense", "medium", "mostly_monotonic", "duplicates"}]
        fp = sum(bool(r[policy]) for r in random_rows)
        tp = sum(bool(r[policy]) for r in local_rows)
        print(
            f"{policy}: random_routes={fp}/{len(random_rows)} "
            f"local_routes={tp}/{len(local_rows)}"
        )

    print(f"wrote {len(rows)} diagnostic rows -> {OUT}")


if __name__ == "__main__":
    main()
