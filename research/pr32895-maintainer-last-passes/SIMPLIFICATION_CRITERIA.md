# PR 32895 — simplification-first research criteria

Research branch: `research/pr32895-maintainer-last-passes`.
PR head must remain unchanged: `439b07aaf620495de332996e9324145828cf99d3`.

## Primary research question
Can we preserve the worthwhile throughput of the locality-based PR with **less maintenance cost and no meaningful extra overhead** by simplifying the selector and the search implementation?

This is *not* an instruction to rewrite the original PR. All experiments remain isolated.

## Priority order
1. **No selector if possible:** Can a small tail-local search change improve general workload throughput without pattern detection?
2. **Minimal selector:** If a specialized locality path remains worthwhile, measure whether a cheap key-count/stride gate alone suffices, and whether deterministic samples can be removed.
3. **One shared core:** Test whether existing batch-search control flow can share core logic with the special case, instead of duplicating independent search algorithms.
4. **Simpler search:** Compare bounded per-key binary tail vs galloping + residual binary search. Reduce mathematical special cases and invariants, not merely source lines.

## Constraints / acceptance
- Correctness against upstream for left/right, duplicates, NaN, edge lengths and strides must be demonstrated.
- No measurable material regression in common workloads; always report worst p95 slowdown and include raw data.
- Any retained complexity must pay for itself in repeated, diverse workloads and CPUs.
- Report impact on normal path, selector overhead, binary size, branches, state variables, duplicated code, exceptional cases and explanation burden.
- Prefer an explainable 1.1x broadly beneficial change over a large peak gain with unstable fallbacks, *if measured evidence supports it*. No preordained winners.
- Preserve original performance records and raw benchmark setup; do not assume 5x gain in unmeasured scenarios.
- No commit to `perf/searchsorted-locality-selector` or upstream; do not open/change PR without user approval.

## Decision matrix
S0 upstream original implementation
S1 current PR (frozen)
S2 simple tail reordering, no pattern selector
S3 minimal selector / cheap static gate, without sample-based detection
S4 current optimized branch simplified via replacing galloping with bounded binary search

For each: correctness; N/Q/pattern/dtype/stride x CPU benchmark; worst and typical regressions; code/maintenance complexity; CI evidence URL.

## Current state
Scope and research design recorded; algorithm implementation and CI experiments remain pending.
