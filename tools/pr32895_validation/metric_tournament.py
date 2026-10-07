#!/usr/bin/env python3
from __future__ import annotations

import json
import math
import statistics
import time
from pathlib import Path

import numpy as np

Q = 1 << 20
N = 1 << 20
SEEDS = list(range(20261007, 20261022))
SAMPLE_COUNTS = (17, 33, 65)
TIMING_REPEATS = 15
TIMING_WARMUP = 3


def sample_indices(count: int) -> np.ndarray:
    bits = int(math.log2(count - 1))
    last = Q - 1
    return np.asarray([(j * last) >> bits for j in range(count)], dtype=np.intp)


def coarse_base_identity(keys: np.ndarray, side: str = "left") -> np.ndarray:
    out = np.empty(keys.size, dtype=np.int64)
    for j, key in enumerate(keys.astype(np.int64, copy=False)):
        interval = N
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
        out[j] = base
    return out


def metrics_from_positions(p: np.ndarray) -> dict[str, float]:
    p = p.astype(np.int64, copy=False)
    d1 = np.diff(p)
    d2 = np.diff(d1)
    d3 = np.diff(d2)

    nz = d1[d1 != 0]
    signs = np.sign(nz)
    flips = int(np.count_nonzero(signs[1:] != signs[:-1])) if signs.size > 1 else 0

    s1 = int(np.abs(d1).sum())
    s2 = int(np.abs(d2).sum())
    s3 = int(np.abs(d3).sum())

    desc = int(np.count_nonzero(d1 < 0))
    asc = int(np.count_nonzero(d1 > 0))
    denom = max(1, d1.size)

    # Cheap proxy for how far galloping would have to travel.
    gallop_debt = int(np.floor(np.log2(np.abs(d1) + 1)).sum()) if d1.size else 0

    return {
        "descending_fraction": desc / denom,
        "ascending_fraction": asc / denom,
        "sign_flip_rate": flips / max(1, signs.size - 1),
        "s1": float(s1),
        "s2": float(s2),
        "s3": float(s3),
        "s2_over_s1": s2 / max(1, s1),
        "s3_over_s1": s3 / max(1, s1),
        "roughness": (s2 + s3) / max(1, s1),
        "gallop_debt": float(gallop_debt),
        "current_monotone_accept": float(desc == 0),
    }


def metric_block(q: np.ndarray, count: int, scheme: str) -> dict[str, float]:
    idx = sample_indices(count)
    if scheme == "anchors":
        probes = idx
    elif scheme == "adjacent":
        probes = np.unique(np.concatenate([
            idx,
            np.clip(idx - 1, 0, Q - 1),
            np.clip(idx + 1, 0, Q - 1),
        ]))
    elif scheme == "multiscale":
        # 17 anchors plus deterministic midpoints between them, then quarter-points.
        base = sample_indices(17)
        mids = ((base[:-1].astype(np.int64) + base[1:].astype(np.int64)) // 2).astype(np.intp)
        q1 = ((3 * base[:-1].astype(np.int64) + base[1:].astype(np.int64)) // 4).astype(np.intp)
        q3 = ((base[:-1].astype(np.int64) + 3 * base[1:].astype(np.int64)) // 4).astype(np.intp)
        probes = np.unique(np.concatenate([base, mids, q1, q3]))
    else:
        raise ValueError(scheme)

    exact = np.clip(q[probes].astype(np.int64, copy=False), 0, N)
    coarse = coarse_base_identity(exact)
    out = {}
    for prefix, values in (("exact", exact), ("coarse", coarse)):
        for k, v in metrics_from_positions(values).items():
            out[f"{prefix}_{k}"] = v
    out["probe_count"] = float(len(probes))
    return out


def make_workloads(seed: int) -> dict[str, tuple[str, np.ndarray]]:
    rng = np.random.default_rng(seed)
    out = {}

    ordered = np.linspace(0, N - 1, Q, dtype=np.int32)
    out["ordered_uniform"] = ("good", ordered)

    dup = np.repeat(np.arange(0, N, 64, dtype=np.int32), 64)[:Q]
    out["ordered_duplicate_heavy"] = ("good", dup)

    plateau = np.minimum(np.arange(Q, dtype=np.int64) // 8, N - 1).astype(np.int32)
    out["ordered_plateaus"] = ("good", plateau)

    clustered = np.sort(rng.integers(0, N, size=Q, dtype=np.int32))
    out["ordered_random_distribution"] = ("good", clustered)

    local_swaps = ordered.copy()
    swap_idx = rng.choice(Q - 1, size=Q // 1000, replace=False)
    tmp = local_swaps[swap_idx].copy()
    local_swaps[swap_idx] = local_swaps[swap_idx + 1]
    local_swaps[swap_idx + 1] = tmp
    out["mostly_ordered_local_swaps"] = ("good", local_swaps)

    random_q = rng.integers(0, N, size=Q, dtype=np.int32)
    out["random"] = ("bad", random_q)

    oscillating = np.empty(Q, dtype=np.int32)
    oscillating[0::2] = N - 1
    oscillating[1::2] = 0
    out["oscillating"] = ("bad", oscillating)

    block = 4096
    base = np.arange(Q, dtype=np.int64) % N
    block_id = np.arange(Q, dtype=np.int64) // block
    reversed_blocks = np.where((block_id & 1) == 0, base, (N - 1) - base).astype(np.int32)
    out["reversed_blocks"] = ("bad", reversed_blocks)

    saw = (np.arange(Q, dtype=np.int64) % 8192).astype(np.int32)
    out["sawtooth"] = ("bad", saw)

    shuffled_blocks = ordered.reshape(-1, 4096).copy()
    rng.shuffle(shuffled_blocks, axis=0)
    out["shuffled_blocks"] = ("bad", shuffled_blocks.reshape(-1).copy())

    # Construct inputs that fool all nested 17/33/65 uniform anchor sets.
    deception = rng.integers(0, N, size=Q, dtype=np.int32)
    idx65 = sample_indices(65)
    deception[idx65] = np.linspace(0, N - 1, idx65.size, dtype=np.int32)
    out["anchor_deception_random"] = ("bad", deception)

    deception2 = oscillating.copy()
    deception2[idx65] = np.linspace(0, N - 1, idx65.size, dtype=np.int32)
    out["anchor_deception_oscillating"] = ("bad", deception2)

    deception3 = reversed_blocks.copy()
    deception3[idx65] = np.linspace(0, N - 1, idx65.size, dtype=np.int32)
    out["anchor_deception_reversed_blocks"] = ("bad", deception3)

    return out


def evaluate_threshold(rows: list[dict], field: str) -> dict:
    good = [r[field] for r in rows if r["label"] == "good"]
    bad = [r[field] for r in rows if r["label"] == "bad"]
    vals = sorted(set(good + bad))
    candidates = vals if len(vals) < 200 else [vals[int(i * (len(vals)-1) / 199)] for i in range(200)]
    best = None
    for direction in ("low_is_good", "high_is_good"):
        for t in candidates:
            if direction == "low_is_good":
                tp = sum(v <= t for v in good)
                tn = sum(v > t for v in bad)
            else:
                tp = sum(v >= t for v in good)
                tn = sum(v < t for v in bad)
            bal = 0.5 * (tp / len(good) + tn / len(bad))
            item = (bal, direction, t, tp, tn)
            if best is None or item[0] > best[0]:
                best = item
    bal, direction, threshold, tp, tn = best
    return {
        "balanced_accuracy": bal,
        "direction": direction,
        "threshold": threshold,
        "good_pass": tp,
        "good_total": len(good),
        "bad_reject": tn,
        "bad_total": len(bad),
        "good_min": min(good),
        "good_max": max(good),
        "bad_min": min(bad),
        "bad_max": max(bad),
    }


def time_metric(q: np.ndarray, count: int, scheme: str) -> dict:
    for _ in range(TIMING_WARMUP):
        metric_block(q, count, scheme)
    samples = []
    for _ in range(TIMING_REPEATS):
        t0 = time.perf_counter_ns()
        metric_block(q, count, scheme)
        samples.append(time.perf_counter_ns() - t0)
    return {
        "median_ns": statistics.median(samples),
        "min_ns": min(samples),
        "max_ns": max(samples),
        "cv": statistics.pstdev(samples) / statistics.mean(samples),
    }


def main() -> None:
    rows = []
    timing = []

    for seed in SEEDS:
        workloads = make_workloads(seed)
        for name, (label, q) in workloads.items():
            for count in SAMPLE_COUNTS:
                for scheme in ("anchors", "adjacent", "multiscale"):
                    if scheme == "multiscale" and count != 17:
                        continue
                    row = {
                        "seed": seed,
                        "workload": name,
                        "label": label,
                        "sample_count": count,
                        "scheme": scheme,
                    }
                    row.update(metric_block(q, count, scheme))
                    rows.append(row)

        if seed == SEEDS[0]:
            representative = workloads["anchor_deception_random"][1]
            for count in SAMPLE_COUNTS:
                for scheme in ("anchors", "adjacent", "multiscale"):
                    if scheme == "multiscale" and count != 17:
                        continue
                    timing.append({
                        "sample_count": count,
                        "scheme": scheme,
                        **time_metric(representative, count, scheme),
                    })

    fields = [
        "exact_descending_fraction",
        "exact_sign_flip_rate",
        "exact_s2_over_s1",
        "exact_s3_over_s1",
        "exact_roughness",
        "exact_gallop_debt",
        "coarse_descending_fraction",
        "coarse_sign_flip_rate",
        "coarse_s2_over_s1",
        "coarse_s3_over_s1",
        "coarse_roughness",
        "coarse_gallop_debt",
        "coarse_current_monotone_accept",
    ]

    summaries = []
    configs = sorted({(r["sample_count"], r["scheme"]) for r in rows})
    for count, scheme in configs:
        subset = [r for r in rows if r["sample_count"] == count and r["scheme"] == scheme]
        for field in fields:
            s = evaluate_threshold(subset, field)
            summaries.append({
                "sample_count": count,
                "scheme": scheme,
                "metric": field,
                **s,
            })

    summaries.sort(key=lambda x: (-x["balanced_accuracy"], x["sample_count"], x["scheme"], x["metric"]))

    output = {
        "query_count": Q,
        "array_size": N,
        "seeds": SEEDS,
        "rows": rows,
        "timing": timing,
        "summaries": summaries,
    }
    Path("pr32895_metric_tournament.json").write_text(json.dumps(output, indent=2))

    lines = [
        "# PR #32895 metric tournament",
        "",
        f"- Query count: {Q}",
        f"- Seeds: {len(SEEDS)}",
        f"- Workloads/seed: {len(make_workloads(SEEDS[0]))}",
        f"- Timing repeats: {TIMING_REPEATS} after {TIMING_WARMUP} warmups",
        "",
        "## Top separation candidates",
        "",
        "| BA | probes | scheme | metric | direction | threshold | good pass | bad reject |",
        "|---:|---:|---|---|---|---:|---:|---:|",
    ]
    for s in summaries[:40]:
        lines.append(
            f"| {s['balanced_accuracy']:.3f} | {s['sample_count']} | {s['scheme']} | "
            f"{s['metric']} | {s['direction']} | {s['threshold']:.6g} | "
            f"{s['good_pass']}/{s['good_total']} | {s['bad_reject']}/{s['bad_total']} |"
        )

    lines += ["", "## Timing stability", "", "| probes | scheme | median ns | min ns | max ns | CV |", "|---:|---|---:|---:|---:|---:|"]
    for t in timing:
        lines.append(
            f"| {t['sample_count']} | {t['scheme']} | {t['median_ns']:.0f} | "
            f"{t['min_ns']} | {t['max_ns']} | {t['cv']:.4f} |"
        )

    Path("pr32895_metric_tournament.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
