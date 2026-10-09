# Q-free interval-selector information screen (2026-10-09)

Status: exploratory synthetic discrimination screen, **not** native NumPy performance or algorithmic correctness validation.

## Method
800 inputs across 8 synthetic families and 100 seeds; each input contains 4,097 keys. Failure proxy: an adjacent decrease in keys, potentially defeating monotone predecessor reuse. The coarse position proxy is one of eight bins. This test deliberately includes hidden reversals that leave sampled positions and coarse bins unchanged.

| Screening feature | Accepted | Unsafe accepted | Mean inspected positions (proxy) |
|---|---:|---:|---:|
| Original sparse anchors + predecessors | 300 | 100 | ~17 |
| Additional midpoint samples | 300 | 100 | ~23 |
| Additional quartile samples | 300 | 100 | ~35 |
| Eight-bin histogram alone | 600 | 400 | 4097 |
| Coarse-position total variation alone | 600 | 400 | 4097 |
| Full adjacent key reversal scan | 200 | 0 | 4097 |

CAUTION: histogram and TV were tested **alone**, not stacked atop the original selector; the acceptance counts are not an apples-to-apples end-to-end performance comparison. Synthetic failure proxy is **not** synonymous with an actual measured slowdown. Fixed-sample probes can miss reversals hidden in unsampled regions, including within the same coarse bucket.

## Conclusions
- Sampling a few extra fixed points cannot guarantee detecting interior inversions.
- Coarse bucket histograms and coarse-position TV cannot detect a reversal when all keys map to the same coarse bucket.
- Full key monotonicity scan can detect every adjacent reversal in this synthetic screen, at O(Q) read/compare cost; whether that is economically viable requires C++ timing of fused versus separated scans.
- Next controlled variants: original Q + selector; exactly same code without Q; no-Q + fused monotonicity check with early stop. Keep Q only as comparison control and retain exact official PR code untouched.
- Do not claim all accepted cases are fast. Must test accepted-path timings, fallback costs, distributions, dtype, N and multiple physical/virtual CPU generations.
- GitHub Actions API workflow_dispatch on registered workflow 378468289 returned HTTP 422 `Actions has been disabled for this repository`, although GET repository actions permissions reported enabled=true, allowed_actions=all. No run URL exists for this screen.
