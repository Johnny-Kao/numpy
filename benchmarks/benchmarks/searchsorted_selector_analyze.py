"""Aggregate same-runner searchsorted selector tournament JSON files."""

from __future__ import annotations

import csv
import json
import statistics
import sys
from pathlib import Path


root = Path(sys.argv[1])
suite = sys.argv[2]


def load(name: str) -> list[dict]:
    return json.loads((root / name).read_text(encoding="utf-8"))["rows"]


def key(row: dict) -> tuple:
    return (row["dtype"], row["n"], row["q"], row["shape"], row["side"])


before = {key(r): r for r in load(f"current-before-{suite}.json")}
after = {key(r): r for r in load(f"current-after-{suite}.json")}

modes = (
    "prev_bound",
    "galloping",
    "hybrid_full",
    "hybrid_sample16",
    "hybrid_sample32",
    "hybrid_chunk16",
    "hybrid_chunk64",
)

detail: list[dict] = []
for mode in modes:
    for row in load(f"{mode}-{suite}.json"):
        k = key(row)
        b = before[k]["median_ns"]
        a = after[k]["median_ns"]
        baseline = (b + a) / 2.0
        ratio = row["median_ns"] / baseline
        detail.append(
            {
                "mode": mode,
                "suite": suite,
                "dtype": row["dtype"],
                "n": row["n"],
                "q": row["q"],
                "shape": row["shape"],
                "side": row["side"],
                "baseline_before_ns": b,
                "baseline_after_ns": a,
                "baseline_drift_ratio": a / b,
                "candidate_ns": row["median_ns"],
                "ratio": ratio,
            }
        )

with (root / f"detail-{suite}.csv").open("w", newline="", encoding="utf-8") as f:
    writer = csv.DictWriter(f, fieldnames=detail[0].keys())
    writer.writeheader()
    writer.writerows(detail)

lines = []
lines.append(f"# searchsorted selector tournament — {suite}")
lines.append("")
lines.append("| mode | median ratio | best | worst | >2% faster | >2% slower |")
lines.append("| --- | ---: | ---: | ---: | ---: | ---: |")
for mode in modes:
    ratios = [r["ratio"] for r in detail if r["mode"] == mode]
    faster = sum(x < 0.98 for x in ratios)
    slower = sum(x > 1.02 for x in ratios)
    lines.append(
        f"| {mode} | {statistics.median(ratios):.3f} | "
        f"{min(ratios):.3f} | {max(ratios):.3f} | "
        f"{faster}/{len(ratios)} | {slower}/{len(ratios)} |"
    )

lines.append("")
lines.append("## Per-shape median ratio")
lines.append("")
lines.append("| mode | shape | median ratio | worst |")
lines.append("| --- | --- | ---: | ---: |")
for mode in modes:
    shapes = sorted({r["shape"] for r in detail if r["mode"] == mode})
    for shape in shapes:
        ratios = [
            r["ratio"]
            for r in detail
            if r["mode"] == mode and r["shape"] == shape
        ]
        lines.append(
            f"| {mode} | {shape} | {statistics.median(ratios):.3f} | "
            f"{max(ratios):.3f} |"
        )

drifts = [r["baseline_drift_ratio"] for r in detail]
lines.append("")
lines.append(
    f"Baseline before/after drift across cases: "
    f"median={statistics.median(drifts):.3f}, "
    f"min={min(drifts):.3f}, max={max(drifts):.3f}."
)

(root / f"summary-{suite}.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
print("\n".join(lines))
