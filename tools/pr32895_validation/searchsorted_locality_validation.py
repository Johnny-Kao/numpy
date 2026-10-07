#!/usr/bin/env python3
"""Validation/benchmark harness for NumPy PR #32895.

Run the same script on baseline-before, PR HEAD, and baseline-after builds.
It intentionally does not modify the production algorithm.
"""

from __future__ import annotations

import argparse
import json
import platform
import statistics
import sys
import time
from pathlib import Path

import numpy as np

KEY_COUNT = 1 << 20
DEFAULT_ARRAY_SIZE = 1 << 20
SAMPLES = 16
CORRECTNESS_SEED_COUNT = 3


def _selector_indices(n: int) -> np.ndarray:
    last = n - 1
    return np.asarray([(j * last) >> 4 for j in range(SAMPLES + 1)], dtype=np.intp)


def _coarse_base_identity(key: int, arr_len: int, side: str) -> int:
    """Emulate the first 3 PR binary-search levels for a == arange(arr_len)."""
    interval = arr_len
    half = interval >> 1
    interval -= half
    if side == "left":
        base = half if half < key else 0
    else:
        base = half if half <= key else 0

    completed = 1
    while interval > 1 and completed < 3:
        half = interval >> 1
        interval -= half
        pivot = base + half
        if side == "left":
            if pivot < key:
                base += half
        else:
            if pivot <= key:
                base += half
        completed += 1
    return base


def _selector_probe(queries: np.ndarray, arr_len: int, side: str) -> dict:
    indices = _selector_indices(queries.size)
    keys = queries[indices].astype(np.int64, copy=False)
    coarse = [_coarse_base_identity(int(k), arr_len, side) for k in keys]
    direction = 0
    reversed_ = False
    prev = coarse[0]
    for pos in coarse[1:]:
        if pos > prev:
            if direction < 0:
                reversed_ = True
                break
            direction = 1
        elif pos < prev:
            if direction > 0:
                reversed_ = True
                break
            direction = -1
        prev = pos

    if not reversed_ and direction >= 0:
        for i in indices:
            i = int(i)
            if i > 0 and queries[i] < queries[i - 1]:
                reversed_ = True
                break

    return {
        "sample_indices": indices.tolist(),
        "sample_keys": keys.tolist(),
        "coarse_bases": coarse,
        "selector_accepts_locality": bool(not reversed_ and direction >= 0),
    }


def _query_pattern_stats(queries: np.ndarray) -> dict:
    q = queries.astype(np.int64, copy=False)
    d = np.diff(q)
    if d.size == 0:
        return {"nondecreasing_fraction": 1.0, "decreasing_fraction": 0.0}
    return {
        "nondecreasing_fraction": float(np.mean(d >= 0)),
        "decreasing_fraction": float(np.mean(d < 0)),
    }


def _apply_monotonic_anchors(queries: np.ndarray, arr_len: int) -> np.ndarray:
    indices = _selector_indices(queries.size)
    anchors = np.linspace(0, arr_len - 1, indices.size, dtype=queries.dtype)
    queries[indices] = anchors
    return queries


def make_cases(array_size: int, seed: int) -> dict[str, np.ndarray]:
    rng = np.random.default_rng(seed)

    random_interiors = rng.integers(
        0, array_size, size=KEY_COUNT, dtype=np.int32
    )
    _apply_monotonic_anchors(random_interiors, array_size)

    oscillating = np.empty(KEY_COUNT, dtype=np.int32)
    oscillating[0::2] = array_size - 1
    oscillating[1::2] = 0
    _apply_monotonic_anchors(oscillating, array_size)

    block = 4096
    base = np.arange(KEY_COUNT, dtype=np.int64) % array_size
    block_id = np.arange(KEY_COUNT, dtype=np.int64) // block
    reversed_blocks = np.where(
        (block_id & 1) == 0,
        base,
        (array_size - 1) - base,
    ).astype(np.int32)
    _apply_monotonic_anchors(reversed_blocks, array_size)

    ordered = np.linspace(0, array_size - 1, KEY_COUNT, dtype=np.int32)

    return {
        "ordered_control": ordered,
        "adversarial_random_interiors": random_interiors,
        "adversarial_oscillating_interiors": oscillating,
        "adversarial_reversed_blocks": reversed_blocks,
    }


def check_searchsorted_finite(
    a: np.ndarray, v: np.ndarray, idx: np.ndarray, side: str
) -> None:
    """Check searchsorted's insertion invariant for finite numeric inputs."""
    n = len(a)
    if not ((idx >= 0) & (idx <= n)).all():
        raise AssertionError("index out of range")
    if n == 0:
        if not (idx == 0).all():
            raise AssertionError("nonzero result for empty haystack")
        return

    prev_values = a[np.maximum(idx - 1, 0)]
    next_values = a[np.minimum(idx, n - 1)]
    if side == "left":
        prev_ok = (idx == 0) | (prev_values < v)
        next_ok = (idx == n) | (v <= next_values)
    else:
        prev_ok = (idx == 0) | (prev_values <= v)
        next_ok = (idx == n) | (v < next_values)
    if not prev_ok.all():
        raise AssertionError("lower bound violated")
    if not next_ok.all():
        raise AssertionError("upper bound violated")


def randomized_correctness(seed: int) -> dict:
    results = []
    for seed_offset in range(CORRECTNESS_SEED_COUNT):
        case_seed = seed + seed_offset
        rng = np.random.default_rng(case_seed)
        for dtype in (np.int32, np.int64, np.float32, np.float64):
            if np.issubdtype(dtype, np.integer):
                raw = rng.integers(-10_000, 10_000, size=4096, dtype=np.int64)
                a = np.sort(raw.astype(dtype))
                q_raw = rng.integers(
                    -12_000, 12_000, size=KEY_COUNT, dtype=np.int64
                ).astype(dtype)
            else:
                # Rounding deliberately creates duplicate-heavy float inputs.
                raw = np.round(rng.normal(size=4096), 2).astype(dtype)
                a = np.sort(raw)
                q_raw = np.round(rng.normal(size=KEY_COUNT), 2).astype(dtype)

            for pattern, queries in (
                ("random", q_raw),
                ("ordered", np.sort(q_raw.copy())),
            ):
                for side in ("left", "right"):
                    idx = np.searchsorted(a, queries, side=side)
                    check_searchsorted_finite(a, queries, idx, side)
                    results.append({
                        "seed": case_seed,
                        "dtype": np.dtype(dtype).name,
                        "pattern": pattern,
                        "side": side,
                        "ok": True,
                    })
    return {"seed_count": CORRECTNESS_SEED_COUNT, "cases": results}


def deterministic_special_correctness() -> list[dict]:
    results = []

    nan_values = [-np.inf, 0, 1, 127, 247, np.inf, np.nan, np.nan]
    nan_expected = {
        "left": [0, 0, 1, 127, 247, 248, 250, 250],
        "right": [0, 1, 2, 128, 248, 250, 256, 256],
    }
    for dtype in (np.float16, np.float32, np.float64):
        a = np.concatenate([
            np.arange(248, dtype=dtype),
            np.asarray([np.inf, np.inf] + [np.nan] * 6, dtype=dtype),
        ])
        queries = np.repeat(
            np.asarray(nan_values, dtype=dtype), KEY_COUNT // len(nan_values)
        )
        for side in ("left", "right"):
            actual = np.searchsorted(a, queries, side=side)
            desired = np.repeat(
                np.asarray(nan_expected[side], dtype=np.intp),
                KEY_COUNT // len(nan_values),
            )
            if not np.array_equal(actual, desired):
                raise AssertionError(f"NaN correctness failed for {dtype=} {side=}")
            results.append({
                "family": "float_nan",
                "dtype": np.dtype(dtype).name,
                "side": side,
                "ok": True,
            })

    counts = np.full(9, KEY_COUNT // 9, dtype=np.intp)
    counts[-1] += KEY_COUNT - int(counts.sum())
    for dtype in (np.complex64, np.complex128):
        values = np.asarray([
            0 + 0j,
            0 + 1j,
            1 + 0j,
            1 + 1j,
            complex(0, np.nan),
            complex(1, np.nan),
            complex(np.nan, 0),
            complex(np.nan, 1),
            complex(np.nan, np.nan),
        ], dtype=dtype)
        a = np.repeat(values, 4)
        queries = np.repeat(values, counts)
        for side, expected in (
            ("left", np.arange(0, 36, 4, dtype=np.intp)),
            ("right", np.arange(4, 40, 4, dtype=np.intp)),
        ):
            actual = np.searchsorted(a, queries, side=side)
            desired = np.repeat(expected, counts)
            if not np.array_equal(actual, desired):
                raise AssertionError(
                    f"complex NaN correctness failed for {dtype=} {side=}"
                )
            results.append({
                "family": "complex_nan",
                "dtype": np.dtype(dtype).name,
                "side": side,
                "ok": True,
            })

    datetime_cases = [
        ("datetime64[D]", ["2000-01-01", "2000-01-02", "2000-01-03", "NaT"]),
        ("timedelta64[D]", [0, 1, 2, "NaT"]),
    ]
    dt_expected = {
        "left": np.asarray([0, 64, 128, 192], dtype=np.intp),
        "right": np.asarray([64, 128, 192, 256], dtype=np.intp),
    }
    for dtype, values in datetime_cases:
        values = np.asarray(values, dtype=dtype)
        a = np.repeat(values, 64)
        queries = np.repeat(values, KEY_COUNT // len(values))
        for side in ("left", "right"):
            actual = np.searchsorted(a, queries, side=side)
            desired = np.repeat(dt_expected[side], KEY_COUNT // len(values))
            if not np.array_equal(actual, desired):
                raise AssertionError(
                    f"datetime/NaT correctness failed for {dtype=} {side=}"
                )
            results.append({
                "family": "datetime_nat",
                "dtype": dtype,
                "side": side,
                "ok": True,
            })

    return results


def benchmark_case(
    a: np.ndarray, q: np.ndarray, side: str, repeat: int
) -> dict:
    # Correctness-check the exact adversarial/control input before timing it.
    checked = np.searchsorted(a, q, side=side)
    check_searchsorted_finite(a, q, checked, side)

    # Separate warm-up from measured samples.
    np.searchsorted(a, q, side=side)
    samples = []
    checksum = None
    for _ in range(repeat):
        start = time.perf_counter()
        out = np.searchsorted(a, q, side=side)
        elapsed = time.perf_counter() - start
        samples.append(elapsed)
        checksum = int(out[0]) + int(out[len(out) // 2]) + int(out[-1])
    return {
        "median_s": statistics.median(samples),
        "min_s": min(samples),
        "max_s": max(samples),
        "samples_s": samples,
        "checksum": checksum,
        "correctness_checked": True,
    }


def run(label: str, output: Path, array_size: int, seed: int, repeat: int) -> None:
    a = np.arange(array_size, dtype=np.int32)
    cases = make_cases(array_size, seed)

    selector_probes = {}
    pattern_stats = {}
    benchmarks = {}
    for name, queries in cases.items():
        selector_probes[name] = {
            side: _selector_probe(queries, array_size, side)
            for side in ("left", "right")
        }
        pattern_stats[name] = _query_pattern_stats(queries)
        benchmarks[name] = {
            side: benchmark_case(a, queries, side, repeat)
            for side in ("left", "right")
        }

    result = {
        "label": label,
        "python": sys.version,
        "platform": platform.platform(),
        "numpy_version": np.__version__,
        "numpy_file": np.__file__,
        "array_size": array_size,
        "query_count": KEY_COUNT,
        "seed": seed,
        "repeat": repeat,
        "randomized_correctness": randomized_correctness(seed),
        "special_correctness": deterministic_special_correctness(),
        "selector_probes": selector_probes,
        "pattern_stats": pattern_stats,
        "benchmarks": benchmarks,
    }
    output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(f"Wrote {output}")


def _load(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def compare(
    baseline_before_path: Path,
    pr_path: Path,
    output: Path,
    baseline_after_path: Path | None,
) -> None:
    before = _load(baseline_before_path)
    pr = _load(pr_path)
    after = _load(baseline_after_path) if baseline_after_path else None

    lines = [
        "# PR #32895 custom validation comparison",
        "",
        f"Baseline-before: `{before['label']}` / NumPy `{before['numpy_version']}`",
        f"PR: `{pr['label']}` / NumPy `{pr['numpy_version']}`",
    ]
    if after is not None:
        lines.append(
            f"Baseline-after: `{after['label']}` / NumPy `{after['numpy_version']}`"
        )
    lines += [
        "",
        "Speedup is baseline-reference median / PR median; >1.0x means the PR is faster.",
        "With baseline-after supplied, the reference is the median of the two baseline run medians.",
        "",
        "| Case | Side | Base before (s) | Base after (s) | PR (s) | Speedup | Baseline drift | Selector accepts locality |",
        "|---|---|---:|---:|---:|---:|---:|---|",
    ]

    worst = None
    for case, before_sides in before["benchmarks"].items():
        for side, before_metrics in before_sides.items():
            b1 = before_metrics["median_s"]
            if after is not None:
                b2 = after["benchmarks"][case][side]["median_s"]
                baseline_ref = statistics.median([b1, b2])
                drift = (b2 / b1 - 1.0) * 100.0
                b2_text = f"{b2:.6f}"
                drift_text = f"{drift:+.2f}%"
            else:
                b2 = None
                baseline_ref = b1
                b2_text = "—"
                drift_text = "—"

            p = pr["benchmarks"][case][side]["median_s"]
            speedup = baseline_ref / p
            accepts = pr["selector_probes"][case][side]["selector_accepts_locality"]
            lines.append(
                f"| {case} | {side} | {b1:.6f} | {b2_text} | {p:.6f} | "
                f"{speedup:.3f}x | {drift_text} | {accepts} |"
            )
            if worst is None or speedup < worst[0]:
                worst = (speedup, case, side, baseline_ref, p)

    lines += ["", "## Worst observed case", ""]
    if worst is not None:
        speedup, case, side, b, p = worst
        regression = (p / b - 1.0) * 100.0
        lines.append(
            f"`{case}` / `{side}`: {speedup:.3f}x baseline/PR; "
            f"PR time change {regression:+.2f}%."
        )

    lines += ["", "## Adversarial pattern evidence", ""]
    for case, stats in pr["pattern_stats"].items():
        if case.startswith("adversarial_"):
            left_accepts = pr["selector_probes"][case]["left"]["selector_accepts_locality"]
            right_accepts = pr["selector_probes"][case]["right"]["selector_accepts_locality"]
            lines.append(
                f"- `{case}`: adjacent nondecreasing fraction "
                f"{stats['nondecreasing_fraction']:.4f}; selector accepts "
                f"left={left_accepts}, right={right_accepts}."
            )

    output.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"Wrote {output}")


def main() -> None:
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="cmd", required=True)

    run_p = sub.add_parser("run")
    run_p.add_argument("--label", required=True)
    run_p.add_argument("--output", type=Path, required=True)
    run_p.add_argument("--array-size", type=int, default=DEFAULT_ARRAY_SIZE)
    run_p.add_argument("--seed", type=int, default=20261007)
    run_p.add_argument("--repeat", type=int, default=7)

    cmp_p = sub.add_parser("compare")
    cmp_p.add_argument("baseline_before", type=Path)
    cmp_p.add_argument("pr", type=Path)
    cmp_p.add_argument("--baseline-after", type=Path)
    cmp_p.add_argument("--output", type=Path, default=Path("comparison.md"))

    args = parser.parse_args()
    if args.cmd == "run":
        run(args.label, args.output, args.array_size, args.seed, args.repeat)
    else:
        compare(
            args.baseline_before,
            args.pr,
            args.output,
            args.baseline_after,
        )


if __name__ == "__main__":
    main()
