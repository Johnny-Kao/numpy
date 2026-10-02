"""Standalone screening benchmark for searchsorted locality research.

This file is research-only and is not intended for upstream submission.
"""

from __future__ import annotations

import argparse

import numpy as np
import pyperf


def make_queries(array_size: int, n_queries: int, query_shape: str, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)

    if query_shape == "random":
        return rng.integers(
            -array_size // 10,
            array_size + array_size // 10,
            size=n_queries,
            dtype=np.int32,
        )

    start = array_size // 10

    if query_shape == "dense":
        steps = np.ones(n_queries, dtype=np.int64)
    elif query_shape == "medium":
        steps = np.full(n_queries, 10, dtype=np.int64)
    elif query_shape == "sparse":
        steps = np.full(
            n_queries,
            max(100, array_size // max(n_queries * 4, 1)),
            dtype=np.int64,
        )
    elif query_shape == "clustered":
        steps = np.ones(n_queries, dtype=np.int64)
        steps[4::5] = 64
    elif query_shape == "mostly_monotonic":
        steps = np.ones(n_queries, dtype=np.int64)
        positions = start + np.cumsum(steps) - 1
        positions[31::32] -= 16
        if positions.max(initial=0) >= array_size:
            raise ValueError("workload does not fit array")
        return positions.astype(np.int32)
    else:
        raise ValueError(query_shape)

    positions = start + np.cumsum(steps) - steps[0]
    if positions.max(initial=0) >= array_size:
        raise ValueError("workload does not fit array")
    return positions.astype(np.int32)


def bench(loops: int, arr: np.ndarray, queries: np.ndarray) -> float:
    timer = pyperf.perf_counter
    start = timer()
    for _ in range(loops):
        np.searchsorted(arr, queries)
    return timer() - start


def main() -> None:
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--variant", required=True)
    args, remaining = parser.parse_known_args()

    # Keep Runner in control of pyperf's own CLI options.
    import sys
    sys.argv = [sys.argv[0], *remaining]

    runner = pyperf.Runner()
    runner.metadata["variant"] = args.variant
    runner.metadata["numpy_version"] = np.__version__

    array_sizes = (1_000_000, 10_000_000)
    query_counts = (1_000, 100_000)
    query_shapes = (
        "dense",
        "medium",
        "sparse",
        "clustered",
        "mostly_monotonic",
        "random",
    )
    seeds = (42,)

    for array_size in array_sizes:
        arr = np.arange(array_size, dtype=np.int32)
        for n_queries in query_counts:
            for query_shape in query_shapes:
                for seed in seeds:
                    try:
                        queries = make_queries(array_size, n_queries, query_shape, seed)
                    except ValueError:
                        continue
                    name = (
                        f"searchsorted_N{array_size}_Q{n_queries}_"
                        f"{query_shape}_seed{seed}"
                    )
                    expected = np.searchsorted(arr, queries)
                    # Cheap correctness guard before timing.
                    if expected.shape != queries.shape:
                        raise AssertionError(name)
                    runner.bench_time_func(name, bench, arr, queries)


if __name__ == "__main__":
    main()
