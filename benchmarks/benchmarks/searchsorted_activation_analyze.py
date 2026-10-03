"""Analyze searchsorted selector activation-threshold experiments."""

from __future__ import annotations

import argparse
import json
import re
import statistics
from pathlib import Path

KEYS = ("dtype", "n", "q", "shape", "layout", "side")
LOCAL = {"dense", "medium", "mostly_monotonic", "duplicates"}


def load(path: Path):
    return json.loads(path.read_text())["rows"]


def key(row):
    return tuple(row[k] for k in KEYS)


def pct(xs, p):
    if not xs:
        return float("nan")
    ys = sorted(xs)
    i = min(len(ys)-1, max(0, round((len(ys)-1)*p)))
    return ys[i]


def threshold_from_name(name: str) -> int:
    m = re.search(r"t(\d+)", name)
    if not m:
        raise ValueError(name)
    return int(m.group(1))


def summarize(directory: Path, suite: str):
    baselines = []
    for label in ("current-before", "current-mid", "current-after"):
        p = directory / f"{label}-{suite}.json"
        if p.exists():
            baselines.append({key(r): r["median_ns"] for r in load(p)})
    if not baselines:
        raise SystemExit("no baseline files")

    common = set.intersection(*(set(x) for x in baselines))
    base = {k: statistics.median([b[k] for b in baselines]) for k in common}

    out = {}
    for p in sorted(directory.glob(f"stage*-t*-{suite}.json")):
        rows = load(p)
        th = threshold_from_name(p.stem)
        rnd, loc_active, loc_all, below = [], [], [], []
        per_q = {}
        for r in rows:
            k = key(r)
            if k not in base:
                continue
            ratio = r["median_ns"] / base[k]
            q = int(r["q"])
            per_q.setdefault(q, []).append(ratio)
            if r["shape"] == "random":
                rnd.append(ratio)
            if r["shape"] in LOCAL:
                loc_all.append(ratio)
                if q >= th:
                    loc_active.append(ratio)
            if q < th:
                below.append(ratio)

        out[p.stem.removesuffix(f"-{suite}")] = {
            "threshold": th,
            "random_median": statistics.median(rnd) if rnd else None,
            "random_p95": pct(rnd, .95) if rnd else None,
            "random_worst": max(rnd) if rnd else None,
            "below_gate_median": statistics.median(below) if below else None,
            "local_active_median": statistics.median(loc_active) if loc_active else None,
            "local_all_median": statistics.median(loc_all) if loc_all else None,
            "per_q_median": {str(q): statistics.median(v) for q, v in sorted(per_q.items())},
        }
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("directory", type=Path)
    ap.add_argument("suite")
    ap.add_argument("--select", type=int, default=0)
    ap.add_argument("--gate", action="store_true")
    ap.add_argument("--out", type=Path)
    args = ap.parse_args()

    results = summarize(args.directory, args.suite)

    # Prefer the smallest threshold that is already safe and useful.
    safe = []
    for name, s in results.items():
        if (
            s["random_median"] is not None and
            s["random_p95"] is not None and
            s["local_active_median"] is not None and
            s["random_median"] <= 1.03 and
            s["random_p95"] <= 1.08 and
            s["local_active_median"] <= 0.80
        ):
            safe.append((s["threshold"], name))
    safe.sort()

    payload = {"suite": args.suite, "results": results}
    if args.select:
        selected = [name for _, name in safe[:args.select]]
        if not selected:
            fallback = sorted(
                results.items(),
                key=lambda kv: (
                    max(0, (kv[1]["random_median"] or 9) - 1.03),
                    kv[1]["threshold"],
                    kv[1]["local_active_median"] or 9,
                ),
            )
            selected = [name for name, _ in fallback[:args.select]]
        payload["selected"] = selected

    if args.gate:
        passing = []
        for name, s in results.items():
            if (
                s["random_median"] is not None and
                s["random_p95"] is not None and
                s["local_active_median"] is not None and
                s["random_median"] <= 1.03 and
                s["random_p95"] <= 1.06 and
                s["local_active_median"] <= 0.75
            ):
                passing.append((s["threshold"], name))
        passing.sort()
        payload["passing"] = [name for _, name in passing]
        payload["proceed"] = bool(passing)

    print(json.dumps(payload, indent=2))
    if args.out:
        args.out.write_text(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
