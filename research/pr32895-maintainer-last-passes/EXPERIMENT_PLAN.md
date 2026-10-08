# PR #32895 — maintainer last-pass experiments

## Isolation contract
- Starting point (immutable): `439b07aaf620495de332996e9324145828cf99d3` (PR #32895 head at 2026-10-08).
- Research branch: `research/pr32895-maintainer-last-passes`.
- **Do not push to** `perf/searchsorted-locality-selector`, `numpy/numpy`, or any staging branch.
- Changes and experiment artifacts stay on this research branch until reviewed.
- PR variant is a *reference*, not evidence of improvements from the alternative.

## Motivation (review comment 6050943943)
The maintainer suggests smaller interventions in batched binary search: merge the final 2 search passes, vary how many tail passes are fused (dtype-sensitive), and consider replacing those passes with a per-key linear scan. Measure the worst-case cost and pattern prevalence, not just headline speedup.

## Baselines and variants
B0 — upstream batched binary search at PR base commit `1365bae8111b4187748912d11234ad7fb32ac66d`.
B1 — exact PR-head implementation, including existing selector (this branch's initial tree).
V1 — *non-selector* batched search with final 2 passes handled per key.
V2 — final 3 or 4 passes handled per key; sweep cutover threshold, avoid hardcoding winner.
V3 — bounded linear scan for the last 2 / 3 / 4 levels, with explicit bounds and left/right semantics.

Do not conflate V1–V3 with locality selector activation; these are distinct hypotheses. Preserve an unmodified comparator baseline for each test.

## Measurement matrix
- Array N: 0, 1, 2, 3, 7, 16, 32, 256, 4096, 65536, 1048576.
- Query Q: 1, 8, 64, 1024, 65536, 1048576 where resource limits permit.
- Pattern: uniform random, monotone ascending, descending, all-equal, duplicates, clustered, alternating extreme, adversarial sparse-sample.
- Dtypes: at least int32, int64, float32, float64; signed zeros/NaNs/infinities in correctness validation.
- Layout: contiguous plus strided inputs and outputs where supported.
- Side: left and right.
- Repeat matched runs with alternating order and record warmup, compiler flags, CPU, OS, SHA and raw distributions.
- Report geometric mean only alongside per-cell min/median/max, worst slowdown and workload coverage; don't infer real-world workload prevalence from synthetic frequencies.

## Required correctness gates
- Compare against B0 for all cases (including empty array, non-power-of-two N, duplicates, left/right, irregular strides, boundary values and floating-point ordering).
- Avoid OOB accesses and preserve insertion index range 0..N.
- Run project-specific core tests and ASAN/UBSAN if supported before drawing conclusions.

## Decision gate
Retain alternative only if there is repeatable general-purpose benefit, bounded regressions, and materially simpler code; otherwise preserve negative result. **No PR changes without explicit approval.**
