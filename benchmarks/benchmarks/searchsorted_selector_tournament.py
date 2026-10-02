"""Research-only broad tournament for searchsorted selector strategies.

Each invocation runs exactly one selector mode in a fresh Python process so the
C++ research mode is fixed before any timed calls. Results are JSON and are
intended for same-runner ratio comparisons, not cross-runner absolute timing.
"""

from __future__ import annotations

import json
import os
import statistics
import time
from pathlib import Path

import numpy as np


MODE = os.environ["NPY_SEARCHSORTED_RESEARCH_MODE"]
SUITE = os.environ.get("SEARCHSORTED_TOURNAMENT_SUITE", "broad")
OUT = Path(os.environ.get("SEARCHSORTED_TOURNAMENT_OUT", f"{MODE}-{SUITE}.json"))


def make_queries(n: int, q: int, shape: str, dtype: np.dtype, seed: int = 42) -> np.ndarray:
    rng = np.random.default_rng(seed)
    if shape == "random":
        return rng.integers(-n // 10, n + n // 10, size=q, dtype=np.int64).astype(dtype)

    start = n // 10
    if shape == "dense":
        steps = np.ones(q, dtype=np.int64)
    elif shape == "medium":
        steps = np.full(q, 10, dtype=np.int64)
    elif shape == "sparse":
        steps = np.full(q, max(100, n // max(q * 4, 1)), dtype=np.int64)
    elif shape == "clustered":
        steps = np.ones(q, dtype=np.int64)
        steps[4::5] = 64
    elif shape == "mostly_monotonic":
        steps = np.ones(q, dtype=np.int64)
        pos = start + np.cumsum(steps) - 1
        pos[31::32] -= 16
        if pos.max(initial=0) >= n:
            raise ValueError("workload does not fit")
        return pos.astype(dtype)
    else:
        raise ValueError(shape)

    pos = start + np.cumsum(steps) - steps[0]
    if pos.max(initial=0) >= n:
        raise ValueError("workload does not fit")
    return pos.astype(dtype)


def suite_cases() -> tuple[tuple[int, ...], tuple[int, ...], tuple[str, ...], tuple[np.dtype, ...]]:
    if SUITE == "tiny":
        return (
            (10_000, 1_000_000),
            (1, 4, 16, 64),
            ("dense", "mostly_monotonic", "random"),
            (np.dtype("int32"), np.dtype("int64")),
        )
    if SUITE == "dtype":
        return (
            (1_000_000, 10_000_000),
            (1_000, 100_000),
            ("dense", "medium", "mostly_monotonic", "random"),
            (np.dtype("int64"),),
        )
    return (
        (1_000_000, 10_000_000),
        (1_000, 100_000),
        ("dense", "medium", "sparse", "clustered", "mostly_monotonic", "random"),
        (np.dtype("int32"),),
    )


def check(arr: np.ndarray, queries: np.ndarray, side: str) -> None:
    n = arr.size
    q64 = queries.astype(np.int64)
    expected = np.clip(q64 if side == "left" else q64 + 1, 0, n)
    actual = np.searchsorted(arr, queries, side=side)
    if not np.array_equal(actual, expected):
        raise AssertionError(
            f"correctness failure mode={MODE} suite={SUITE} side={side} "
            f"dtype={arr.dtype} n={n} q={queries.size}"
        )


def measure(arr: np.ndarray, queries: np.ndarray, side: str) -> dict[str, float | int]:
    # Correctness call also initializes the C++ static research-mode selector.
    check(arr, queries, side)

    t0 = time.perf_counter()
    np.searchsorted(arr, queries, side=side)
    one = max(time.perf_counter() - t0, 1e-9)
    loops = max(1, min(20_000, int(0.025 / one)))

    samples: list[float] = []
    for _ in range(3):
        for _ in range(max(1, loops // 10)):
            np.searchsorted(arr, queries, side=side)

    for _ in range(7):
        start = time.perf_counter_ns()
        for _ in range(loops):
            np.searchsorted(arr, queries, side=side)
        elapsed = time.perf_counter_ns() - start
        samples.append(elapsed / loops)

    return {
        "median_ns": statistics.median(samples),
        "mean_ns": statistics.fmean(samples),
        "min_ns": min(samples),
        "max_ns": max(samples),
        "loops": loops,
    }


def main() -> None:
    ns, qs, shapes, dtypes = suite_cases()
    rows: list[dict[str, object]] = []

    for dtype in dtypes:
        for n in ns:
            arr = np.arange(n, dtype=dtype)
            for q in qs:
                for shape in shapes:
                    try:
                        queries = make_queries(n, q, shape, dtype)
                    except ValueError:
                        continue
                    for side in ("left", "right"):
                        result = measure(arr, queries, side)
                        rows.append(
                            {
                                "mode": MODE,
                                "suite": SUITE,
                                "dtype": dtype.name,
                                "n": n,
                                "q": q,
                                "shape": shape,
                                "side": side,
                                **result,
                            }
                        )

    payload = {
        "mode": MODE,
        "suite": SUITE,
        "numpy_version": np.__version__,
        "rows": rows,
    }
    OUT.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"{MODE} {SUITE}: {len(rows)} cases -> {OUT}")


if __name__ == "__main__":
    main()
