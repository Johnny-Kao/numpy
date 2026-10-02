"""Aggregate same-runner searchsorted attribution tournament JSON files."""

from __future__ import annotations

import csv
import json
import statistics
import sys
from pathlib import Path


root = Path(sys.argv[1])
suite = sys.argv[2]


def load(path: Path) -> list[dict]:
    return json.loads(path.read_text(encoding="utf-8"))["rows"]


def key(row: dict) -> tuple:
    return (row["dtype"], row["n"], row["q"], row["shape"], row["side"])


before_path = root / f"current-before-{suite}.json"
after_path = root / f"current-after-{suite}.json"
before = {key(r): r for r in load(before_path)}
after = {key(r): r for r in load(after_path)}

candidate_paths = sorted(
    p for p in root.glob(f"*-{suite}.json")
    if p.name not in {before_path.name, after_path.name}
)

detail: list[dict] = []
labels: list[str] = []
for path in candidate_paths:
    payload = json.loads(path.read_text(encoding="utf-8"))
    label = payload.get("label") or payload.get("mode") or path.stem
    labels.append(label)
    for row in payload["rows"]:
        k = key(row)
        b = before[k]["median_ns"]
        a = after[k]["median_ns"]
        baseline = (b + a) / 2.0
        detail.append(
            {
                "label": label,
                "mode": row.get("mode", label),
                "chunk": row.get("chunk", 0),
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
                "ratio": row["median_ns"] / baseline,
            }
        )

labels = sorted(set(labels))
with (root / f"detail-{suite}.csv").open("w", newline="", encoding="utf-8") as f:
    writer = csv.DictWriter(f, fieldnames=detail[0].keys())
    writer.writeheader()
    writer.writerows(detail)

lines = [f"# searchsorted attribution tournament — {suite}", ""]
lines += [
    "| label | median ratio | best | worst | >2% faster | >2% slower |",
    "| --- | ---: | ---: | ---: | ---: | ---: |",
]
for label in labels:
    ratios = [r["ratio"] for r in detail if r["label"] == label]
    faster = sum(x < 0.98 for x in ratios)
    slower = sum(x > 1.02 for x in ratios)
    lines.append(
        f"| {label} | {statistics.median(ratios):.3f} | "
        f"{min(ratios):.3f} | {max(ratios):.3f} | "
        f"{faster}/{len(ratios)} | {slower}/{len(ratios)} |"
    )

lines += [
    "",
    "## Per-shape median ratio",
    "",
    "| label | shape | median ratio | worst |",
    "| --- | --- | ---: | ---: |",
]
for label in labels:
    shapes = sorted({r["shape"] for r in detail if r["label"] == label})
    for shape in shapes:
        ratios = [
            r["ratio"] for r in detail
            if r["label"] == label and r["shape"] == shape
        ]
        lines.append(
            f"| {label} | {shape} | {statistics.median(ratios):.3f} | "
            f"{max(ratios):.3f} |"
        )

lines += [
    "",
    "## Per-Q median ratio",
    "",
    "| label | q | median ratio | worst |",
    "| --- | ---: | ---: | ---: |",
]
for label in labels:
    qs = sorted({r["q"] for r in detail if r["label"] == label})
    for q in qs:
        ratios = [
            r["ratio"] for r in detail
            if r["label"] == label and r["q"] == q
        ]
        lines.append(
            f"| {label} | {q} | {statistics.median(ratios):.3f} | "
            f"{max(ratios):.3f} |"
        )

drifts = [r["baseline_drift_ratio"] for r in detail]
lines += [
    "",
    f"Baseline before/after drift across cases: "
    f"median={statistics.median(drifts):.3f}, "
    f"min={min(drifts):.3f}, max={max(drifts):.3f}.",
]

(root / f"summary-{suite}.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
print("\n".join(lines))
