# Searchsorted locality research record

Status: active research  
Branch: `research/searchsorted-selector-tournament`  
Scope: research-only prototypes and benchmark evidence for a possible follow-up optimization to NumPy `searchsorted`.

## 1. Why this work exists

NumPy's current typed `searchsorted` implementation, introduced by upstream PR #30517, batches binary-search steps across all query keys. That design is strong for general and random query order because it improves pivot cache locality and exposes query-level instruction-level parallelism.

The research question here is deliberately narrower:

> Can a follow-up path exploit query locality strongly enough to provide multi-x speedups while preserving the current batched implementation as the safe default?

The important variable is not simply whether query values are sorted. It is the displacement of consecutive insertion positions:

```
Delta_i = abs(position_i - position_{i-1})
```

When Delta is small, restarting a full O(log N) search for every query can be unnecessary. A locality-aware/galloping search can approach O(log Delta).

This work is therefore treated as a possible PR2 layered on top of the existing batched-search baseline, not as a replacement for PR #30517.

## 2. Baseline and historical context

Relevant historical signal:

- NumPy issue #10937 discussed sorted second-input behavior and reuse of the previous search bound.
- Upstream PR #30517 changed the primitive typed path to batched binary search.
- The typed path now favors same-depth batched work across queries; the generic/custom-comparator path still has previous-key narrowing behavior.

The current typed implementation is the baseline throughout this research.

Original research base:
`6acfb88f0b1a3606ec73a382dddf9696272a175f`

Research branches used during exploration:

- `research/searchsorted-monotonicity`
- `research/searchsorted-prev-bound`
- `research/searchsorted-galloping`
- `research/searchsorted-selector-tournament` (current)

## 3. Experimental discipline

The benchmark process evolved during the work. Current rules:

1. Compare candidates against the existing current path.
2. Prefer same-runner paired measurements.
3. Use the same commit, wheel, workload generator, timing method and runner for candidate/baseline comparisons.
4. Long sweeps use current-before / current-mid / current-after baselines to reduce drift bias.
5. GitHub-hosted logical resource profiles use CPU affinity and memory limits. These are simulated logical profiles, not claims of distinct physical cloud instance types.
6. Apple M5 local measurements are directional only; primary comparison uses GitHub Actions.
7. No paid testing.
8. Broad coverage is the goal; the research does not claim exhaustive coverage of every dtype, distribution, CPU or downstream use case.
9. Parameters are research knobs unless and until a production implementation and broader upstream benchmark evidence justify fixing them.

Logical resource profiles used:

- 1 vCPU / 2 GiB
- 2 vCPU / 4 GiB
- 4 vCPU / 8 GiB
- 4 vCPU / 14 GiB

Physical GitHub hosts observed across runs included AMD EPYC 7763, AMD EPYC 9V45 and Intel Xeon Platinum 8573C.

## 4. Phase A — prove that locality has real value

### Candidate A: current batched search

The existing typed implementation from PR #30517.

### Candidate B: previous-bound narrowing

Reuse a previous bound to reduce the next search domain.

Result:

- modest gains on monotonic/local inputs;
- random inputs can regress badly;
- useful mechanism evidence but not strong enough as the final specialized path.

### Candidate C: galloping/local search

Start from the previous insertion result, grow the search interval exponentially, then binary-search only the discovered local range.

Standalone mechanism run:
GitHub Actions run `37045851099`.

Representative results:

- N=1M, Q=1k: dense ~0.130x current; medium ~0.500x; sparse ~0.646x; clustered ~0.218x; mostly-monotonic ~0.251x.
- Random was ~6.7x slower.
- N=10M, Q=100k: dense ~0.120x; medium ~0.391x; clustered ~0.139x; mostly-monotonic ~0.249x.
- Random was ~15.3x slower.

Conclusion:

> Locality is independently valuable and can provide multi-x speedups. Galloping cannot replace the current path globally because random order is catastrophic.

This established the fundamental PR2 opportunity.

## 5. Phase B — understand the crossover

Controlled displacement run:
GitHub Actions run `37046276757`.

A full monotonic scan selected galloping only when the entire query was nondecreasing.

At N=10M, Q=1k, gallop/current moved from roughly 0.111x at Delta=1 toward ~1.0x around large displacement.

Important observation:

> There is no universal fixed Delta threshold. Cache behavior and the current batched algorithm make the crossover non-monotonic.

Therefore the problem is not simply "pick Delta < X".

## 6. Phase C — validate inside real NumPy

Benchmark:
`benchmarks/benchmarks/searchsorted_monotonicity_research.py`

Three same-machine paired jobs:
current->current, current->previous-bound, current->galloping.

Run:
`37048235707`

Representative galloping/current ratios:

- N=1M, Q=1k: dense ~0.19x, medium ~0.42x, sparse ~0.74x, clustered ~0.24x, mostly-monotonic ~0.25x.
- N=1M, Q=100k: dense ~0.14x, mostly-monotonic ~0.23x.
- N=10M, Q=1k: dense ~0.16x, medium ~0.35x, sparse ~0.99x, clustered ~0.21x, mostly-monotonic ~0.22x.
- N=10M, Q=100k: dense ~0.12x, medium ~0.38x, clustered ~0.15x, mostly-monotonic ~0.20x.
- Random remained materially slower, reaching roughly 9-10x in some large cases.

Concrete example:

- N=10M, Q=100k, dense:
  current ~2.67 ms
  galloping ~0.316 ms
  ~8.4x faster.

Conclusion:

> The multi-x gain survives integration into real NumPy. The remaining problem is safe routing, not the local-search mechanism.

## 7. Phase D — fixed chunk attribution

Run:
`37055796383`

Compared current-chunk, gallop-chunk and hybrid-chunk at widths:

`4, 8, 16, 32, 64, 128`

Findings:

1. Locality is independently valuable.
2. Batch width is independently valuable.
3. No fixed chunk width is universally optimal.
4. A chunk width that helps one dtype/workload can hurt another.
5. Current full-batch execution is a valuable safe fallback.
6. Tiny-Q selector overhead is first-order.

Examples:

- hybrid chunk 8: broad overall ~0.517x, random ~0.983x in one suite;
- hybrid chunk 16: broad overall ~0.652x, random ~0.985x;
- but int64 random could regress ~17-18%.

Conclusion:

> The optimization is an execution-policy/crossover problem, not a single best algorithm or a universal chunk size.

Toolkit attribution record commit:
`7a0d58a1814a2baf2fbe41f029d4509d4bbcb478`

## 8. Phase E — staged global adaptive selector

Eight representative policies P0-P7 varied:

- activation Q
- stage boundaries
- sample budgets
- allowed inversions

Implementation commit:
`3ae1248dfeb81b3c304913415412c235568750ec`

Benchmark expansion:
`274b4a4418a01eadc619da2060b2186d77b7a3c8`

Workflow:
`73fb1c9e57756edeaebdd6620f9db3cafd0f9479`

Run:
`37088313500`

High-level result:

- strong locality remained very fast;
- random still commonly paid ~10-25% or more;
- int64 random was especially stubborn;
- policies differed by workload and machine.

Conclusion:

> Sampling a query first and then routing the entire batch to either current or galloping is not a sufficiently safe production architecture.

## 9. Phase F — heavy resource-profile stress

Broad heavy suite:

- N=10M / 100M
- Q=100k / 1M
- dense / medium / sparse / mostly-monotonic / random / block-sorted / reversal-bursts
- int32
- left/right
- four logical resource profiles

Workflow commit:
`28955f8d9fcf2d40604c68c2888f6c2650d7929e`

Run:
`37090048307`

All four profiles completed successfully without OOM.

Representative outcome:

- dense up to ~8x faster;
- medium ~2.5-3x faster;
- mostly-monotonic ~4-6x faster;
- random still ~20-40% slower for otherwise attractive tolerant policies.

P4/P6 emerged as useful parameter regions, but neither removed the random-path problem.

Conclusion:

> The core limitation is architectural, not merely a bad parameter choice.

## 10. Phase G — 48-policy fine sweep

To test whether the previous conclusion was premature, the selector was parameterized dynamically and a 48-combination grid was tested.

Dynamic mode commit:
`0aed1009e8cec8ab119cd8f471572a775fef8692`

Midpoint baseline analyzer:
`d765cb174b22bf8d1e1870f8e2dfcc4ab4e1aff5`

Fine-suite setup:
`39567d80535618c224be743b085019d6848a3fb6`

Workflow:
`b27ccab581642ca7dc7400ec4f751816ee0414df`

Run:
`37093388977`

Grid dimensions:

- activation: 32 / 64 / 128 / 256
- stage2/stage3: four combinations
- sample budgets: 4/8/16, 8/16/32, 8/16/64, 8/24/64
- inversion tolerance: 0/0/0, 0/1/2, 0/1/3
- four logical resource profiles

Workload:

- N=10M / 100M
- Q=8192 / 100k / 1M
- seven query shapes

All four jobs completed successfully.

Result:

> No combination simultaneously preserved the locality gains and reduced random regression to the desired ~<5% range across profiles.

This is the point where broad global-sampling parameter exploration was stopped.

Key inference:

> Continuing to scan 100+ nearby parameter combinations would be low-value. The selector architecture had to change.

## 11. Metadata inspection

NumPy's `PyArray_SearchSorted` entry already knows or materializes, before typed binsearch dispatch:

- common dtype
- query size
- query ndim/shape via the PyArrayObject
- native byte order
- alignment
- C-contiguous query storage
- haystack size
- haystack stride
- query itemsize
- side
- whether a sorter is present

These are effectively free metadata for routing.

Important boundary:

> dtype/shape/contiguity cannot tell whether query values are random or locally ordered.

There is no ndarray metadata flag equivalent to `is_sorted`.

Therefore metadata should be used as a conservative eligibility gate, not as a complete locality classifier.

A useful additional observation is that multidimensional query shape is known for free even though typed binsearch receives a flattened contiguous stream. Future work could use row/axis boundaries as natural reset boundaries if justified by benchmarks.

## 12. Phase H — output-driven online locality routing

New idea:

> Do not pre-scan query values. Use insertion positions already produced by the current search as the locality signal.

This is attractive because insertion-position displacement is the actual quantity of interest.

Prototype commit:
`8660b00be235add9854d216bececbb4e6093b49a`

Broad dtype/shape benchmark:
`c4309e83d3f53b74ab5f43312c95df609975dbc7`

Multidimensional analyzer update:
`feb41a3a0448f0433fe46730f07766a56ee0fa56`

Workflow:
`c806009fab7d0fd411c8190aa35612671db051fb`

Run:
`37110326931`

Coverage:

- int32 / int64 / float64
- 1D / 2D query layout
- N=1M / 10M
- Q=1024 / 8192 / 100k
- dense / medium / mostly-monotonic / random / block-sorted / reversal-bursts / duplicates
- left/right
- four logical resource profiles
- block sizes 128 / 256 / 512 / 1024
- 4 / 8 / 12 output observations

Representative locality results:

- dense ~0.19-0.25x on many useful settings -> ~4-5x faster;
- duplicates ~0.15-0.23x -> ~4-6x faster;
- medium ~0.44-0.60x -> ~1.7-2.3x faster;
- mostly-monotonic ~0.23-0.30x on useful settings -> ~3-4x faster.

Random improved substantially relative to global sampling:

- 4 vCPU / 8 GiB: best useful settings ~1.03-1.04x;
- 2 vCPU / 4 GiB: ~1.09-1.13x;
- 4 vCPU / 14 GiB: ~1.11-1.15x;
- 1 vCPU / 2 GiB: ~1.18-1.20x.

Conclusion:

> Output-driven locality observation is a materially better selector architecture. However, repeatedly splitting current execution into small blocks damages the full-batch batching/ILP advantage of PR #30517, especially on smaller resource profiles.

This identified a much narrower remaining problem.

## 13. Phase I — early locality gate with full-batch fallback

Current experiment.

Prototype commit:
`415e27b7bea259af49ada728f87c87c171ea1db8`

Design:

```
small prefix -> current
              |
              +-- observe already-computed insertion positions
                    |
                    +-- strong locality -> galloping for the remainder
                    |
                    +-- not local -> one full-batch current call for the remainder
```

Unlike the previous online-block design, random inputs are no longer repeatedly broken into chunks.

Research knobs:

- early probe: 32 / 64 / 128 / 256
- observations: 4 / 8 / 12
- inversion allowance currently 0

Coverage remains the broad online suite:

- int32 / int64 / float64
- 1D / 2D
- seven query distributions
- Q=1024 / 8192 / 100k
- N=1M / 10M
- left/right
- four logical resource profiles
- before/mid/after baselines

Workflow:
`993d6b5c0d589cc83576e4833a287cd01e7c5036`

Run:
`37112628771`

Final state:

- 1 vCPU / 2 GiB: success
- 2 vCPU / 4 GiB: success
- 4 vCPU / 8 GiB: success
- 4 vCPU / 14 GiB: success

Representative results:

- dense: roughly 0.16-0.19x current on strong settings, about 5-6x faster;
- duplicates: roughly 0.13-0.15x, about 6-8x faster;
- medium: roughly 0.38-0.47x, about 2.1-2.6x faster;
- mostly-monotonic: strong only for some probe/observation settings, roughly 0.22-0.24x when correctly routed;
- random: still approximately 1.13-1.22x on median-like summaries across profiles, with noisy worst cases materially higher.

Baseline before/after drift medians were approximately 1.000, 1.001, 0.996 and 0.996 across the four profiles.

Conclusion:

> Preserving one full-batch current remainder is not enough. Splitting even a small prefix from the batch still imposes a measurable cost on random/general workloads, and the early probe can also miss or over-reject mostly-monotonic structure depending on observation placement.

This rules out the current early-prefix architecture as the final production selector.

## 14. Current technical hypothesis

The research has converged from a broad algorithm question to a narrow routing question.

The early-gate run materially tightened the conclusion:

- repeated chunking is too expensive for random workloads;
- one-time prefix splitting is also still too expensive;
- therefore the final selector cannot require a separate preliminary execution phase before the existing full-batch current path.

The remaining viable direction is to preserve the current full-batch execution intact and obtain routing information without splitting it. This likely means either:

1. piggybacking locality evidence inside the existing full-batch loop with near-zero incremental work, then applying it only to a future call/subregion where semantics permit; or
2. finding a production-safe structural signal that is already known before execution and is strong enough to route without touching the batch shape.

Any next prototype should explicitly measure selector tax in isolation before attempting another broad sweep.

Established:

1. The current batched search is the correct default baseline.
2. Galloping/local search can provide large gains when insertion positions are locally coherent.
3. The gain is real in integrated NumPy benchmarks.
4. No universal fixed chunk size exists.
5. Global pre-sampling has an overhead floor and is not the preferred production structure.
6. Output-driven observation is much cheaper and uses the right signal.
7. Repeatedly chunking the current path harms its batching/ILP advantage.
8. The remaining candidate is therefore a conservative early gate that preserves a full-batch current fallback.

Target steady state:

```
random/general:
    approximately current performance

strong locality:
    ~3-8x faster where the workload supports it

medium locality:
    meaningful ~1.5-3x improvement

tiny/special/unsupported cases:
    current path unchanged
```

A desirable random-path regression budget is approximately <=5%, with stronger preference for near-zero regression.

## 15. Production-scope philosophy

This benchmark campaign is intentionally broad but not exhaustive.

A possible upstream implementation should be described as a conservative first implementation, not as a globally optimal scheduler.

Reasons:

- NumPy supports many numeric and non-numeric dtypes.
- Real distributions are much richer than a synthetic benchmark matrix.
- CPU/cache behavior changes crossover points.
- Query layouts and downstream usage patterns vary.
- A parameter optimum observed here should not be presented as universal.

Therefore the expected upstream framing is:

> Introduce a narrowly gated locality-aware fast path that preserves the existing batched search as the default. The initial thresholds are conservative and benchmark-backed; additional dtype/distribution-specific tuning can be considered separately as evidence accumulates.

Do not overfit PR2 by trying to encode every possible data distribution in the first patch.

## 16. What not to repeat

The following avenues have enough evidence to avoid repeating unless new mechanism evidence appears:

- global full-query monotonic scan as the final production selector;
- global sampled-query then whole-batch binary routing as the final production selector;
- searching for one universal fixed chunk width;
- blind expansion of P0-P7/global-sampling parameter grids;
- OS/process-load checks in the hot path;
- claiming logical GitHub resource limits are distinct physical cloud instance types.

## 17. Next decision gate

Run `37112628771` completed and random remained materially slower.

Therefore broad parameter sweeps remain stopped.

Next work should isolate the remaining cost into:

1. prefix split cost;
2. loss of full-batch ILP from the prefix/remainder split;
3. false-positive or false-negative early routing;
4. dtype-specific effects;
5. benchmark noise.

The preferred next diagnostic is a selector-tax decomposition rather than another tuning sweep:

- current(full batch) baseline;
- current(prefix) + current(remainder) with no locality decision;
- current(full batch) plus equivalent integer-observation bookkeeping inside the loop if feasible;
- early-gate with routing disabled;
- early-gate with routing enabled.

This should identify exactly whether the residual regression comes from split execution, bookkeeping, or misrouting before another production prototype is attempted.

## 18. Evidence index

Major workflow runs:

- `37045851099` — standalone galloping mechanism screen
- `37046276757` — controlled displacement/crossover screen
- `37048235707` — real NumPy previous-bound/galloping validation
- `37055796383` — fixed chunk attribution tournament
- `37088313500` — staged adaptive P0-P7 tournament
- `37090048307` — four-profile heavy stress
- `37093388977` — 48-policy fine sweep
- `37110326931` — output-driven online locality matrix
- `37112628771` — early-gate/full-batch-fallback matrix, current

Important research commits:

- `3ae1248dfeb81b3c304913415412c235568750ec` — staged adaptive policies
- `274b4a4418a01eadc619da2060b2186d77b7a3c8` — transition suites
- `73fb1c9e57756edeaebdd6620f9db3cafd0f9479` — adaptive tournament workflow
- `9e94794aa40538e9360c80655441927a21181a16` — heavy stress suite
- `28955f8d9fcf2d40604c68c2888f6c2650d7929e` — four-profile stress workflow
- `0aed1009e8cec8ab119cd8f471572a775fef8692` — dynamic policy tuning mode
- `d765cb174b22bf8d1e1870f8e2dfcc4ab4e1aff5` — midpoint baseline normalization
- `b27ccab581642ca7dc7400ec4f751816ee0414df` — 48-policy fine workflow
- `8660b00be235add9854d216bececbb4e6093b49a` — output-driven selector
- `c806009fab7d0fd411c8190aa35612671db051fb` — broad online matrix
- `415e27b7bea259af49ada728f87c87c171ea1db8` — early locality gate
- `993d6b5c0d589cc83576e4833a287cd01e7c5036` — early-gate matrix

## 19. Record-maintenance rule

This file is the durable narrative record for the searchsorted locality/PR2 investigation.

After each meaningful experiment:

1. add the exact run ID and commit;
2. record the tested mechanism, not only the final number;
3. record representative ratios and baseline drift;
4. state what the result ruled in or ruled out;
5. preserve failed paths because they explain why the final architecture exists;
6. update the current hypothesis and next decision gate;
7. do not rewrite history to make the path look linear.

The goal is to retain enough evidence to reconstruct the reasoning later for an upstream PR, engineering retrospective, or technical article.


## 21. Selector-tax decomposition launched

After run `37112628771`, the remaining question was narrowed to selector overhead itself.

New diagnostic modes were added in commit:
`4d314d5e4250a5f5a4799bc81b58013a78c49f15`

The decomposition compares:

1. `current` — untouched full-batch baseline.
2. `split_current_control` — current(prefix) + current(remainder), no locality decision.
3. `signature_then_current` — inspect a tiny fixed set of query values, then execute one untouched full-batch current call.
4. `early_observe_current` — current(prefix), observe already-computed insertion positions, then always current(remainder), with routing disabled.

Purpose:

- isolate pure split/batching cost;
- isolate tiny pre-dispatch signature cost;
- isolate output-observation bookkeeping cost;
- separate those costs from actual galloping misrouting.

Workflow:
`.github/workflows/searchsorted-selector-tax-decomposition.yml`

Workflow creation commit:
`84d39443a232f9000385a288bbc4b636f63195b9`

Decision rule:

- if `signature_then_current` stays within roughly 1-3% of current across broad random workloads, a pre-dispatch fixed-signature selector remains viable;
- if split controls alone reproduce the ~10-20% regression, prefix execution is conclusively ruled out;
- if even signature-only inspection exceeds budget, hidden dynamic routing should be narrowed or abandoned in favor of a more explicit specialized path.


## 22. Coarse-base piggyback selector prototype

A new selector architecture was added in commit:
`338f2b408b087faeac910a583af02bdceaa5560f`

Hypothesis:

> The first few levels of the existing full-batch binary search already compute a coarse approximation of every query's insertion position in `ret/base`. Those values can serve as a locality signature with almost no extra memory traffic.

Mechanism:

1. Execute the first 2-4 binary-search levels exactly as the current implementation already does.
2. Keep the full query batch intact.
3. Read only a tiny fixed set of already-computed `ret/base` values.
4. Count coarse-position inversions.
5. If the coarse signature is not strongly local, continue the existing current algorithm from the already-computed state.
6. If the signature is strongly local, the research prototype restarts the specialized galloping path for the full batch.

Important distinction:

- random/general input is never prefix-split and never reruns current;
- no separate query-value pre-scan is required;
- selector tax on the random path is limited to a few integer reads/comparisons from state that already exists;
- the local branch currently duplicates the first few levels because this is a mechanism screen, not yet the production local-finisher design.

Research knobs:

- coarse levels: 2 / 3 / 4
- observations: 4 / 8 / 12
- inversion allowance: 0 for the first screen

Workflow:
`.github/workflows/searchsorted-coarse-base-piggyback.yml`

Workflow commit:
`65aa2d748e80204b0b2b7cef380786533027ed01`

Primary success criterion:

- random/general ~ current, target <= ~3-5% regression;
- dense/duplicates retain multi-x gains;
- medium locality retains meaningful gains;
- broad behavior stable across int32/int64/float64, 1D/2D, multiple Q/N and four logical resource profiles.

If this mechanism succeeds, the next engineering step is not further selector tuning. It is to implement a specialized local finisher that resumes from the coarse bounds instead of restarting galloping from scratch.
