"""Analyze staged near-free searchsorted selector experiments.

The analyzer normalizes candidate timings against the median of same-runner
current-before/current-mid/current-after baselines and ranks candidates without
mixing profiles.
"""

from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path


KEYS = ("dtype", "n", "q", "shape", "layout", "side")
LOCAL = {"dense", "medium", "mostly_monotonic", "duplicates"}


def load(path: Path):
    return json.loads(path.read_text())["rows"]


def key(row):
    return tuple(row[k] for k in KEYS)


def pct(values, p):
    if not values:
        return float("nan")
    xs = sorted(values)
    idx = min(len(xs) - 1, max(0, round((len(xs) - 1) * p)))
    return xs[idx]


def summarize(directory: Path, suite: str):
    baselines = []
    for name in ("current-before", "current-mid", "current-after"):
        path = directory / f"{name}-{suite}.json"
        if path.exists():
            baselines.append({key(r): r["median_ns"] for r in load(path)})
    if not baselines:
        raise SystemExit("no baseline files")

    common = set.intersection(*(set(b) for b in baselines))
    base = {
        k: statistics.median([b[k] for b in baselines if k in b])
        for k in common
    }

    results = {}
    for path in sorted(directory.glob(f"*-{suite}.json")):
        if path.name.startswith("current-"):
            continue
        rows = load(path)
        ratios = []
        random_ratios = []
        local_ratios = []
        for r in rows:
            k = key(r)
            if k not in base:
                continue
            ratio = r["median_ns"] / base[k]
            ratios.append(ratio)
            if r["shape"] == "random":
                random_ratios.append(ratio)
            if r["shape"] in LOCAL:
                local_ratios.append(ratio)
        if not ratios:
            continue
        results[path.stem.removesuffix(f"-{suite}")] = {
            "random_median": statistics.median(random_ratios) if random_ratios else None,
            "random_p95": pct(random_ratios, .95) if random_ratios else None,
            "random_worst": max(random_ratios) if random_ratios else None,
            "random_gt_5pct": (
                sum(x > 1.05 for x in random_ratios) / len(random_ratios)
                if random_ratios else None
            ),
            "local_median": statistics.median(local_ratios) if local_ratios else None,
            "local_p95": pct(local_ratios, .95) if local_ratios else None,
            "overall_median": statistics.median(ratios),
            "cases": len(ratios),
        }
    return results


def score(item):
    _, s = item
    rm = s["random_median"] if s["random_median"] is not None else 9.0
    rp = s["random_p95"] if s["random_p95"] is not None else 9.0
    lm = s["local_median"] if s["local_median"] is not None else 9.0
    # Random tax dominates. Locality remains a hard usefulness signal.
    penalty = max(0.0, lm - 0.80) * 2.0
    return rm + 0.35 * max(0.0, rp - 1.0) + penalty


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("directory", type=Path)
    ap.add_argument("suite")
    ap.add_argument("--select", type=int, default=0)
    ap.add_argument("--gate", action="store_true")
    ap.add_argument("--out", type=Path)
    args = ap.parse_args()

    results = summarize(args.directory, args.suite)
    ranked = sorted(results.items(), key=score)

    payload = {"suite": args.suite, "results": results}
    if args.select:
        useful = [
            name for name, s in ranked
            if s["local_median"] is not None and s["local_median"] <= 0.85
        ]
        payload["selected"] = useful[:args.select] or [name for name, _ in ranked[:args.select]]

    if args.gate:
        passing = []
        for name, s in ranked:
            if (
                s["random_median"] is not None
                and s["random_p95"] is not None
                and s["local_median"] is not None
                and s["random_median"] <= 1.05
                and s["random_p95"] <= 1.10
                and s["local_median"] <= 0.75
            ):
                passing.append(name)
        payload["passing"] = passing
        payload["proceed"] = bool(passing)

    print(json.dumps(payload, indent=2))
    if args.out:
        args.out.write_text(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
