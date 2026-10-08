# PR 32895 — first-pass technical assessment (2026-10-08)

Base: PR head `439b07aaf620495de332996e9324145828cf99d3`.
Review: https://github.com/numpy/numpy/pull/32895#issuecomment-6050943943
Scope: analysis only. No algorithm patch / benchmark has yet been run.

## Verified source behavior
- Upstream `numpy/_core/src/npysort/binsearch.cpp` uses batched queries, iterating over all keys for each binary-search level, then performs a final per-key comparison.
- PR head separates `binsearch_current` from `binsearch_locality`, with `LOCALITY_MIN_KEYS=1<<20` and contiguous-input gate.
- Reviewer proposes switching from batch-outer/keys-inner to a per-key tail after interval length falls below threshold; later tests should explore merging 2+ passes or linear scan.

## Hypotheses ranked
P0-1: batched prefix, per-key binary tail with threshold=2,4,8,16 (plus threshold=1 as reference).
P0-2: batched prefix, per-key bounded linear tail with threshold=2,4,8,16.
P0-3: compare loop-inversion alone, against any apparent additional benefit of linear scan.
P1: dtype-specific crossover, only if measured repeatably and after accounting for additional dispatch/maintenance.
P1: pivot reuse / compiled instruction sequence; inspect machine code for loads and register spills.
P2: SIMD late-stage alternative, only after low-complexity results.

## Important caveats
- Original batch formulation offers cross-query instruction-level parallelism; per-key completion adds a dependency chain. Fewer memory loads do not guarantee higher throughput.
- Linear scanning is not generally safe as a replacement for binary tail without respecting cmp semantics for left/right, NaN and duplicates.
- Cutover should be based on *remaining interval length*, not a constant number of outer iterations: N not power of two, N=0/1, and boundary behavior matter.
- Matched *upstream base*, *PR-head current path* and *PR-head gated selector* must be reported distinctly.
- A synthetic pattern matrix cannot establish real-world selector activation prevalence. That needs observed workload traces or representative benchmarks.

## Experiment order
1. Compile and validate upstream and PR head as isolated controls.
2. Introduce binary tail with thresholds 2/4/8/16 on research branch; correctness gate before timing.
3. Linear-tail variant with same thresholds; compare instruction counts and time distributions.
4. Survey dtype, N, Q, contiguous and strided, ascending/descending/random/duplicates/adversarial patterns; record worst regressions and runner URL.
5. Choose simple general-purpose path only if substantial, portable and repeatable improvement.

## Status
Analysis completed; no implementation, CI run or speedup confirmed. PR head remains untouched.
