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


## 23. Coarse-base piggyback result

Run:
`37126529277`

All four logical resource profiles completed successfully.

Representative normalized shape-level results:

### 1 vCPU / 2 GiB
- best random observed: ~1.045x at 4 coarse levels / 8 observations;
- dense: ~0.328x (~3.0x faster);
- duplicates: ~0.297x (~3.4x faster);
- medium: ~0.535x (~1.9x faster);
- mostly-monotonic: ~0.376x (~2.7x faster).

### 2 vCPU / 4 GiB
- random generally ~1.06-1.07x on useful settings;
- dense ~0.24-0.33x;
- duplicates ~0.21-0.30x;
- medium ~0.46-0.54x;
- mostly-monotonic ~0.29-0.38x.

### 4 vCPU / 8 GiB
- random generally ~1.06-1.08x;
- dense ~0.24-0.33x;
- duplicates ~0.21-0.30x;
- medium ~0.46-0.53x;
- mostly-monotonic ~0.30-0.38x.

### 4 vCPU / 14 GiB
- best random observed: ~1.026x at 3 coarse levels / 8 observations;
- dense ~0.282x (~3.5x faster);
- duplicates ~0.259x (~3.9x faster);
- medium ~0.511x (~2.0x faster);
- mostly-monotonic ~0.338x (~3.0x faster).

Baseline before/after median drift remained ~1.0 on all profiles, though individual-case tails were noisier.

Interpretation:

> Piggybacking on coarse `ret/base` state is the first architecture that gets the random path close to the target without splitting the full batch or pre-scanning query values.

Compared with previous architectures:

- global sampling: commonly +20-50% random regression;
- online chunking: roughly +3-20%, profile dependent;
- one-prefix early gate: roughly +13-22%;
- coarse-base piggyback: commonly about +3-8%, with best profile/settings near +2.6-4.5%.

The remaining gap is now small enough to treat as a selector-quality problem rather than an architecture problem.

### Why the current coarse classifier is still imperfect

The first piggyback classifier only counts inversions among a few coarse base samples.

At only 2-4 binary-search levels there are few coarse buckets, so random samples frequently contain repeated bucket values. A sequence with many ties can appear nondecreasing even when the underlying insertion positions are random. This creates false-positive local routing and explains part of the remaining random regression.

### Next hypothesis: path roughness from already-computed bases

Use the same sampled coarse bases, but accumulate a tiny arithmetic signature:

```
range = max(base) - min(base)
total_variation = sum(abs(base[i] - base[i-1]))
```

For a clean monotonic/local path:

```
total_variation ~= range
```

For random/scattered positions:

```
total_variation >> range
```

This needs only integer subtraction/absolute-value/addition over a handful of already-existing `base` values. It adds no extra query memory pass and reuses state already produced by the current algorithm.

The next selector should combine:

- inversion count;
- coarse path total variation;
- coarse range;
- conservative fallback to current when evidence is ambiguous.

The objective is not maximum recall. The objective is high precision: only strongly local inputs should route to the specialized path.


## 24. Coarse roughness result

Run:
`37127994426`

All four profiles completed successfully.

The roughness classifier tested:

```
range = max(base) - min(base)
total_variation = sum(abs(base[i] - base[i-1]))
```

with factors 1 and 2, coarse levels 3/4/5, and 4/8 sampled coarse bases.

### Main result

`roughness_factor = 1` is the only viable setting in this screen.

Representative median-like random ratios with factor 1 were often near current:

- 1 vCPU / 2 GiB: ~0.98-1.04x depending on levels/observations;
- 2 vCPU / 4 GiB: ~0.99-1.04x;
- 4 vCPU / 8 GiB: ~1.02-1.07x;
- 4 vCPU / 14 GiB: ~0.93-1.06x.

At the same time locality gains remained substantial:

- dense: roughly 0.27-0.38x;
- duplicates: roughly 0.23-0.35x;
- medium: roughly 0.52-0.68x;
- mostly-monotonic: roughly 0.32-0.46x.

However individual random-case tails remained materially worse (commonly ~1.2-3x in the worst summarized case, depending on profile/settings).

`roughness_factor = 2` was decisively unsafe:

- random medians frequently ~2.3-2.6x;
- worst cases reached ~5-13x.

This indicates that allowing even modest backtracking in a coarse sampled trajectory admits too many random patterns.

### Interpretation

For nonnegative path distance:

```
total_variation >= range
```

Therefore `factor=1` effectively requires the sampled coarse path to have no backtracking. This is a strong, cheap, platform-independent condition and is appropriate for a high-precision selector.

The remaining failures are not due to selector arithmetic cost. They are false positives from undersampling: with only 4-8 sampled coarse bases, a random sequence can occasionally look monotonic by chance.

### Next experiment

Keep the architecture and strict `factor=1` rule fixed.

Increase only the number of sampled already-computed coarse bases:

- observations: 8 / 16 / 32
- coarse levels: 3 / 4

No new query memory pass is introduced; these are extra integer reads from `ret/base` that current already produced.

Goal:

- preserve random median near 1.00;
- collapse worst random false-positive tails;
- retain most locality gains.

If 16/32 observations remove the tails with negligible median cost, freeze the selector architecture.


## 25. Current research status — focused strict roughness gate in progress

Current active branch:
`research/searchsorted-selector-tournament`

Current production hypothesis:

> Preserve the existing full-batch current search. Reuse the coarse insertion-position bases already computed by the first few binary-search levels, and classify only when the sampled coarse path is strictly non-backtracking.

Why this is now the leading architecture:

- global pre-scan/sampling imposed too much random-path overhead;
- fixed chunking damaged the existing full-batch ILP/cache behavior;
- one-prefix early gating still cost roughly 13-22% on random workloads;
- coarse-base piggybacking reduced random median overhead to roughly low-single-digit percentages while preserving large locality gains;
- adding path roughness (`total_variation / range`) further moved many random medians to approximately current performance;
- permissive roughness (`factor=2`) is unsafe and has been rejected;
- strict roughness (`factor=1`) is the only viable criterion from the latest sweep.

Current unresolved issue:

> With only 4-8 sampled coarse bases, a random trajectory can occasionally look monotonic by chance and be falsely routed to galloping, creating large tail regressions.

Current focused experiment:

Workflow:
`.github/workflows/searchsorted-coarse-roughness-focused.yml`

Workflow commit:
`b35cc79bfa0c26fd35ed73cd32548dec70962bf8`

Run:
`37129876085`

Status at 2026-10-03 23:47 JST:

- micro-1vcpu-2gb: in progress
- small-2vcpu-4gb: in progress
- standard-4vcpu-8gb: in progress
- standard-4vcpu-14gb: in progress
- no failures observed

Focused matrix:

- coarse levels: 3 / 4
- observations: 8 / 16 / 32
- roughness factor: fixed at 1
- broad workload suite unchanged

Decision gate after this run:

1. If 16/32 observations collapse random false-positive tails while median random stays approximately current and locality gains remain meaningful:
   - freeze selector architecture;
   - implement a production-shaped local finisher that resumes from the already-computed coarse bounds instead of restarting galloping;
   - remove research knobs;
   - run correctness/ASV/cross-platform CI.
2. If tails remain materially unsafe:
   - do not resume broad threshold sweeps;
   - inspect the exact false-positive distributions and add only one conservative structural discriminator, or narrow PR2 scope.

Current expected upstream positioning if successful:

> A conservative locality-aware fast path for primitive typed `searchsorted` that reuses state already produced by the existing batched binary search, leaves ambiguous/general workloads on the current implementation, and avoids any separate query pre-scan.

No upstream PR has been opened. User signoff remains required before submission.


## 26. Focused strict roughness result — freeze gate not yet met

Run:
`37129876085`

Status:
- micro-1vcpu-2gb: success
- small-2vcpu-4gb: success
- standard-4vcpu-8gb: success
- standard-4vcpu-14gb: success

Matrix:
- coarse levels: 3 / 4
- observations: 8 / 16 / 32
- roughness factor: fixed at 1
- broad workload suite unchanged

### Representative results

#### 1 vCPU / 2 GiB
Best balanced level-4 settings:
- l4/o16 random: median-like ~1.003x, worst summarized ~1.109x
- l4/o32 random: ~1.010x, worst ~1.145x
- dense: ~0.355-0.359x
- duplicates: ~0.320-0.330x
- medium: ~0.651-0.661x
- mostly-monotonic: ~0.412-0.413x

#### 2 vCPU / 4 GiB
- l4/o32 random: ~1.014x, worst ~1.103x
- l3/o16 random: ~1.007x, worst ~1.210x
- locality shapes remain strongly faster.

#### 4 vCPU / 8 GiB
- l4/o16 random: ~1.055x, worst ~1.080x
- l4/o32 random: ~1.058x, worst ~1.085x
- dense ~0.319x
- duplicates ~0.291-0.292x
- medium ~0.527-0.529x
- mostly-monotonic ~0.378-0.379x

#### 4 vCPU / 14 GiB
- l3/o16 random: ~1.033x, worst ~1.095x
- l4/o16 random: ~1.055x, worst ~1.357x
- l4/o32 random: ~1.059x, worst ~1.339x
- locality shapes remain strongly faster.

Baseline before/after median drift remained near 1.0 on all four profiles.

### Interpretation

Increasing observations from 8 to 16/32 helps some random false-positive tails, but it does **not** eliminate them consistently across profiles.

There is no single tested (levels, observations) pair that simultaneously delivers:

- random median near current;
- random worst/tails consistently inside the desired <=5-10% envelope;
- strong locality gains;
- stability across all four logical resource profiles.

Therefore the selector architecture is **not frozen yet**.

What this run rules out:

- simply increasing coarse sample count is not sufficient by itself;
- further blind observation-count sweeps are unlikely to solve the residual problem.

What remains true:

- coarse-state piggybacking is still the strongest architecture tested;
- selector arithmetic cost is low enough;
- the remaining issue is false-positive classification on specific random distributions / profiles, not the basic reuse-of-current-state idea.

### Next step

Do not broaden parameter search.

Instead identify the exact random cases that generate the worst tails, and compare their coarse-state signatures against correctly rejected random cases.

The next discriminator should be derived from those false positives only.

Candidate structural signals to inspect without another query pass:

1. number of distinct coarse buckets among sampled bases;
2. zero-step / tie fraction;
3. span occupancy: distinct buckets relative to reachable coarse buckets;
4. direction consistency plus bucket entropy / concentration;
5. possibly require both strict no-backtracking and a minimum amount of observed forward progress.

The goal is a conservative precision gate:

> if evidence is not strongly local, stay on current.

No upstream PR should be prepared until the false-positive tail mechanism is understood and bounded.


## 27. Experiment-tree batching rule for the next false-positive study

The next phase will not use another serial “run -> inspect -> design one more run” loop.

Because CI provisioning/build/benchmark cost is now material, the next experiment must be designed as a **precomputed decision tree** and collect evidence for the likely follow-up branches in one runner allocation whenever the measurements do not perturb each other.

### Primary question

Why do some random workloads still satisfy the strict coarse no-backtracking test and route incorrectly?

### Evidence to collect in one pass for every sampled random/local case

Using only coarse `ret/base` state already produced by current:

- sampled base sequence;
- number of distinct coarse buckets;
- zero-step / tie count and fraction;
- min/max/range;
- total variation;
- forward-step count;
- backward-step count;
- maximum single step;
- span occupancy / reachable-bucket occupancy;
- concentration of samples per bucket;
- minimum observed forward progress;
- routing result under the current strict roughness rule.

The same instrumentation should be applied to:
- false-positive random cases;
- correctly rejected random cases;
- dense/local positives;
- medium-locality positives;
- mostly-monotonic positives.

### Precomputed decision tree

```
A. Are false positives dominated by high tie / low-distinct-bucket trajectories?
   YES -> test tie/distinct-bucket gate in the same data.
   NO  -> B

B. Are false positives distinguishable by low span occupancy / low forward progress?
   YES -> test minimum-progress / occupancy gate in the same data.
   NO  -> C

C. Are they distinguishable by bucket concentration / entropy-like concentration?
   YES -> test concentration gate in the same data.
   NO  -> D

D. Do false positives remain statistically indistinguishable from true local cases at coarse levels 3/4?
   YES -> coarse-state-only selector is insufficient at this resolution; narrow scope or use one additional structural signal.
   NO  -> select the cheapest discriminator with highest false-positive rejection and lowest local false-negative rate.
```

### Validation layout

Once the discriminator candidates are computed from the diagnostic data, the same CI allocation should, where practical, evaluate all cheap candidate gates against the broad workload matrix instead of launching one workflow per candidate.

Rules:

- preserve same-machine current baselines;
- keep factor=1 fixed;
- do not resume broad levels/observation sweeps;
- prefer a small set of orthogonal candidate gates over threshold grids;
- record both median and tail behavior;
- only launch a later run if the prior evidence exposes a genuinely new unknown that could not reasonably have been anticipated.

This is now the active experimental-design rule for the searchsorted investigation.


## 28. Decision-tree convergence harness failures and fixes

Two consecutive workflow failures occurred before any selector timing evidence was collected.

Run `37132141827`:
- failure: `KeyError: NPY_SEARCHSORTED_RESEARCH_MODE`
- cause: the diagnostics script imported the shared tournament module, which reads the research mode at import time
- fix commit: `0f72abb06911c7aeff330ae041d60f8d6d699d1c`
- fix: diagnostics sets a harmless default mode before importing the shared workload generator

Run `37132551688`:
- failure: `ValueError: workload does not fit`
- cause: diagnostics did not mirror the benchmark harness behavior that skips invalid synthetic n/q/shape combinations
- fix commit: `18e9cae03688a7da2bede7776c48428b6e35c4d3`
- fix: catch `ValueError` during workload generation and skip the invalid case

These were harness-only failures; no algorithm/selector correctness or performance conclusion should be drawn from them.

Reusable rule:

> Diagnostic scripts that reuse benchmark generators must match the benchmark harness's environment defaults and case-validity filtering before expensive CI is launched.


## 29. Session handoff checkpoint — selector research convergence phase

Date: 2026-10-03 JST

### Current phase

The investigation has entered **selector convergence**, not algorithm discovery.

Established:

1. The upstream/current typed batched binary search is the correct general/random fallback.
2. Galloping/local search is a real specialized fast path and can deliver multi-x gains when insertion positions have locality.
3. The production problem is selector precision and selector tax, not whether the local algorithm works.
4. Selector architectures that pre-scan, split prefixes, or repeatedly chunk current have been tested and rejected because they materially regress random/general workloads.
5. The strongest architecture is **coarse-base piggybacking**:
   - execute the first few levels of the existing full-batch current search;
   - reuse the already-computed `ret/base` values as a coarse insertion-position signature;
   - do not add a separate query scan;
   - do not split the current batch;
   - ambiguous/general input continues current from the already-computed state;
   - only strongly local input may route to the specialized path.

### Best validated evidence so far

Run `37126529277` — coarse-base piggyback:
- 4/4 profiles PASS;
- random median-like behavior roughly ~1.03-1.08x current, best observed ~1.026x;
- dense/duplicates roughly ~3-4x faster;
- medium roughly ~1.9-2.2x faster;
- mostly-monotonic roughly ~2.7-3x faster.

Run `37127994426` — coarse roughness:
- 4/4 profiles PASS;
- `total_variation == range` / strict non-backtracking is the only viable roughness condition;
- permissive factor=2 was decisively unsafe;
- many random medians reached approximately current, but rare false-positive tails remained.

Run `37129876085` — focused observations:
- 4/4 profiles PASS;
- levels 3/4 × observations 8/16/32 × strict factor=1;
- increasing observations improved some false-positive tails but did not make them uniformly safe across all profiles;
- selector architecture therefore was **not frozen**.

Representative balanced random results from that run:
- 1 vCPU / 2 GiB, l4/o16: ~1.003x median-like, ~1.109x worst summarized;
- 2 vCPU / 4 GiB, l4/o32: ~1.014x, ~1.103x;
- 4 vCPU / 8 GiB, l4/o16: ~1.055x, ~1.080x;
- 4 vCPU / 14 GiB, l3/o16: ~1.033x, ~1.095x.

### Current research question

Why do a small number of random workloads still look strictly monotonic in sampled coarse `base` space and get falsely routed to galloping?

The next discriminator must come from **already-computed coarse state only**.

Candidate signals:
- distinct coarse bucket count;
- tie / zero-step fraction;
- forward/backward step count;
- range;
- total variation;
- max step;
- span/bucket occupancy;
- concentration;
- minimum forward progress.

### Experimental-method upgrade

The research process now uses a **precomputed decision tree** for expensive CI:

- mechanism-unknown stage: small sequential experiments are acceptable;
- once likely branches are known: collect evidence for all foreseeable next branches in one runner allocation;
- same-machine baselines are preserved;
- independent policy candidates are batched in the same workflow;
- do not run one workflow per small hypothesis unless a genuinely new unknown appears.

This rule was also promoted to the OSS Engineering Toolkit Master SOP.

### Current implementation for the convergence run

New research mode:
`coarse_decision_tree`

Commit:
`c168c73ff4de1bfc373898713af6d93171c79880`

Policies evaluated in one mode:
- p0: strict roughness baseline;
- p1: strict + minimum distinct buckets;
- p2: strict + minimum forward progress;
- p3: strict + tie-fraction guard;
- p4: combined conservative gate.

Diagnostic script:
`benchmarks/benchmarks/searchsorted_coarse_decision_tree_diagnostics.py`

Diagnostic commit:
`7da5390693bb040f7ae5f94adf7fdb5da3cbeba9`

Workflow:
`.github/workflows/searchsorted-decision-tree-convergence.yml`

Workflow commit:
`96a7049757e1bbe404e70bacad38f203073b014e`

Two harness-only failures occurred and were fixed:
- run `37132141827`: missing research-mode env during diagnostics import;
- run `37132551688`: diagnostics did not skip invalid synthetic workloads.

Fix commits:
- `0f72abb06911c7aeff330ae041d60f8d6d699d1c`
- `18e9cae03688a7da2bede7776c48428b6e35c4d3`

These failures contain no algorithm evidence.

### Active run at handoff

Primary run:
`37133052228`

Status at handoff:
- micro-1vcpu-2gb: in progress
- small-2vcpu-4gb: in progress
- standard-4vcpu-8gb: in progress
- standard-4vcpu-14gb: in progress
- no failure observed at the checkpoint

The run collects both:
1. coarse signature diagnostics for every broad workload case;
2. real timing for p0-p4 across levels 3/4 and observations 16/32 on the same runner allocation.

### Next-session procedure

1. Read this record first.
2. Check run `37133052228`.
3. If complete, extract:
   - diagnostic policy route counts;
   - exact false-positive random signatures;
   - normalized timing summaries for p0-p4;
   - baseline drift.
4. Compare false-positive random cases against correctly rejected random and true local cases.
5. Select only a conservative discriminator supported by the evidence.
6. If one policy bounds random tails across profiles while retaining useful locality gains, freeze the selector architecture.
7. If no coarse-state policy can separate the cases, conclude that this resolution is insufficient and narrow the selector or PR2 scope rather than restarting parameter sweeps.
8. Update this same research record after the result.
9. Do not open an upstream PR without explicit user signoff.

### Do not repeat

Do not restart:
- global pre-scan selector research;
- fixed chunk search;
- broad observation-count sweeps;
- prefix-splitting selectors;
- factor>1 roughness;
- blind parameter tuning.

The research question is now narrow: **can a near-zero-cost structural discriminator from existing coarse binary-search state eliminate the remaining random false positives?**


## 30. Decision-tree convergence result — run 37133052228

Date: 2026-10-04 JST

Run `37133052228` completed successfully on all four logical resource profiles:
- micro-1vcpu-2gb
- small-2vcpu-4gb
- standard-4vcpu-8gb
- standard-4vcpu-14gb

The workflow evaluated levels 3/4, observations 16/32, policies p0-p4, same-machine before/mid/after baselines, coarse-signature diagnostics, and the broad timing matrix in one runner allocation.

### Critical correction: timing tail is not evidence of false-positive routing

The diagnostic corpus contains 72 random cases for each levels/observations setting.

For every tested configuration:
- p0 strict roughness: random routes = **0/72**
- p1 strict + distinct buckets: **0/72**
- p2 strict + forward progress: **0/72**
- p3 strict + tie guard: **0/72**
- p4 combined conservative gate: **0/72**

Therefore no exact random false-positive case exists in this completed convergence run.

This changes the interpretation of the earlier random tails:

> A random timing regression cannot be labeled a routing false positive unless routing diagnostics prove that the case entered the local path.

In this run, random inputs were correctly rejected by the selector, yet random timing still regressed. The remaining problem is therefore selector/hot-path tax plus measurement noise, not insufficient structural discrimination.

### Policy comparison

Diagnostic true-local routing for dense / medium / mostly-monotonic / duplicates:

- p0: 276/276 = 100% at all tested levels/observations.
- p1:
  - l3: 228/276 = 82.6%
  - l4: 264/276 = 95.7%
- p2/p3/p4:
  - 228/276 = 82.6% across the tested settings.

Because random routing is already 0/72 for p0, the additional p1-p4 structural guards do not improve random classification in this corpus. They only reject useful local cases.

Decision:
- **p1 rejected**
- **p2 rejected**
- **p3 rejected**
- **p4 rejected**
- **p0 strict roughness retained only as the minimal research mechanism baseline**

### Timing evidence

Cross-profile random timing, selected configurations:

- l3/o32/p0:
  - median ratio ~1.052x
  - p95 ~1.089x
  - worst ~1.247x
  - ~50.7% of random cases exceed +5%
- l3/o32/p1:
  - median ~1.054x
  - p95 ~1.105x
  - worst ~1.203x
- l4/o32/p4:
  - median ~1.043x
  - p95 ~1.086x
  - worst ~1.287x
  - ~44.1% of random cases exceed +5%

No tested policy reliably keeps rejected-random execution within the desired ~<=5% budget across profiles.

Locality remains strong. For l3/o32/p0, across dense / medium / mostly-monotonic / duplicates:
- median ratio ~0.319x
- worst ~0.742x in the aggregated broad matrix
- diagnostics route all 276/276 target-local cases.

The more conservative policies sacrifice local routing without producing a corresponding random-path benefit.

### False-positive mechanism result

Exact mechanism result:

> **No false-positive mechanism was found because no random false positive occurred in the diagnostic matrix.**

The previous hypothesis that rare random timing tails were caused by sampled coarse trajectories accidentally satisfying strict roughness is not supported by this run and should not remain the working explanation.

The structural resolution is sufficient to reject the current synthetic random corpus. The unresolved issue is the cost of observing/evaluating the selector inside the existing batched search even when the selector ultimately falls back to current.

### Architecture freeze decision

Production selector architecture is **not frozen**.

Reason:
- classification is now sufficiently conservative on the tested random corpus;
- but rejected-random execution still carries material overhead/tails;
- therefore the remaining gate is implementation cost, not another classification threshold.

Do not add p5/p6 or restart observations/threshold sweeps without new mechanism evidence.

### Current hypothesis

The remaining random regression is produced by one or more of:
1. bookkeeping needed to collect sampled coarse-base state;
2. branches/guards in the hot binary-search loop;
3. loss of compiler/ILP/vectorization quality from the research instrumentation;
4. fixed selector setup cost, especially visible in smaller Q;
5. benchmark drift/noise in the largest tail cases.

Large worst-tail observations correlate with substantial before/after baseline drift in several cases, so worst values must not be interpreted as pure selector cost. Median/p95 evidence still shows a real residual tax.

### Next decision gate

The next experiment is no longer a selector-classification tournament.

It must isolate **rejected-random selector tax** using the same full-batch architecture:

1. current baseline;
2. current + equivalent coarse-state bookkeeping but no routing decision;
3. current + strict-roughness decision forced to remain on current;
4. current + production-shaped p0 routing;
5. compare generated code / branch structure if needed.

Goal:
- prove where the ~4-6% median random tax originates;
- remove that tax without weakening the 0/72 random rejection result;
- only then decide whether p0 can become the frozen production selector.

No upstream PR without contributor signoff.


## 31. Near-free P0 implementation convergence — active experiment

Date: 2026-10-04 JST

Objective:

> Keep the validated strict P0 classification mechanism, but reduce correctly-rejected random/general selector tax toward measurement noise by reusing already-produced coarse current-search state.

This experiment is intentionally not another selector-policy sweep. It is an implementation-cost tournament.

### Three-layer single-allocation decision tree

**Layer 1 — cheap mechanism/cost screen**

Nine foreseeable P0 implementations are tested in one runner allocation:

- A16: original total-variation/range P0, 16 observations
- A32: original total-variation/range P0, 32 observations
- B16E: direction/reversal equivalence, 16 observations, early abort
- B32E: direction/reversal equivalence, 32 observations, early abort
- B16FULL: reversal detector without early abort, isolating early-abort value
- C16: fixed 16-way shift-based sampling + reversal early abort
- C32: fixed 32-way shift-based sampling + reversal early abort
- C16-Q2K: C16 plus free metadata activation gate at Q>=2048
- C16-Q8K: C16 plus free metadata activation gate at Q>=8192

All candidates:
- reuse coarse bases produced by the first three current binary-search levels;
- do not pre-scan query values;
- do not split the batch;
- correctly rejected random/general continues current from already-computed bounds.

**Layer 2 — broad finalist validation**

Layer 1 automatically selects at most three useful finalists. Only those candidates run the full `online_broad` dtype/layout/N/Q/distribution matrix.

Proceed threshold:
- random median <= 1.05x current;
- random p95 <= 1.10x;
- local median <= 0.75x.

If none pass, the workflow stops and records the mechanism-cost conclusion.

**Layer 3 — production gate**

At most two passing finalists continue to:
- `adaptive_stress` for large N/Q;
- `tiny` for tiny-Q behavior.

No new GitHub Actions allocation is needed between stages.

### Implementation assets

- near-free research mode commit: `d73985adfbcf9ce37802c5724c3976f1ec47887f`
- staged benchmark suites: `0a5b5e89cd482bcb1064180c26cbaec33fd74259`, `08f1064c30fda88279cbd9f1e1e65af415b4480f`
- staged analyzer: `ea8771a9bdf3fccfd96c77b6ff3b557b442b7db4`
- workflow creation: `e323775f1d5ad13a5885e1d24170e252522919ec`
- push-trigger commit: `27a0edf7a16fa7f799d90086b37d3b483b2a959e`
- workflow: `.github/workflows/searchsorted-nearfree-convergence.yml`

### Next decision

Do not add P5/P6 classification policies.

The next conclusion must come from this implementation tournament:
1. identify whether TV/range arithmetic, early-abort behavior, sample-index arithmetic, or metadata gating dominates residual selector tax;
2. select the lowest-cost mechanism that preserves locality gain;
3. stop if the production tax remains outside the acceptable envelope;
4. if the tax reaches near-current behavior, freeze the selector implementation and move to the production-shaped local finisher that resumes from existing coarse bounds.

No upstream PR without contributor signoff.


## 32. Production activation decision — portable default frozen

Date: 2026-10-04 JST

Run: https://github.com/Johnny-Kao/numpy/actions/runs/37147552873

### Decision

For the first production-shaped implementation, use a conservative cross-platform activation gate:

```text
Q < 2048   -> untouched current batched binary search
Q >= 2048  -> near-free P0 coarse selector
```

The selector mechanism itself remains the validated strict P0 structural test implemented with the low-cost fixed-sample/reversal form. Random/general fallback continues from already-computed coarse bounds; strong-locality cases route to the specialized local finisher.

### Why 2048, not a hardware-specific threshold

The activation crossover is measurably hardware-dependent.

In this run, AMD EPYC profiles tolerated activation as low as Q=64 while preserving near-current random/general performance and strong locality speedups. The Intel Xeon Platinum 8573C profile required a materially larger Q before the selector tax approached the same envelope; Q=1024 was close but retained a somewhat higher p95 tail than the conservative production gate.

Therefore Q=2048 is intentionally a **portable conservative default**, not a claim of a globally optimal threshold.

### Explicit optimization note

There is further optimization headroom in this constant.

The AMD/Intel divergence indicates that the optimal activation point depends on lower-level hardware/code-generation characteristics. A future improvement may replace or refine the fixed Q threshold with an architecture-neutral, already-available cost signal, or may justify a lower portable threshold with broader validation.

Do **not** introduce CPU-model-specific tuning in the initial production patch.

### Research status

The selector-policy research is closed for this optimization:
- P0 strict classification is retained.
- P1-P4 are rejected as unnecessary complexity.
- near-free implementation tax is sufficiently small for large-enough Q.
- Q>=2048 is the conservative portable activation rule.
- remaining work is production shaping, correctness/CI validation, and the local finisher that resumes from existing coarse bounds.

No upstream PR without contributor signoff.


## 33. Clean PR candidate frozen

Date: 2026-10-04 JST

Clean branch:
- `perf/searchsorted-locality-pr-draft`
- production commit: `263f72551457c8f246b7e52f90cdb3a49842a85b`
- base: upstream NumPy main `2f1eca306857b641fb0fef0ab854a2d4137e1581`
- final diff: one commit, one source file (`numpy/_core/src/npysort/binsearch.cpp`)

The temporary validation workflow and harness were removed from the final branch history.

### Production behavior

- Q < 2048: existing batched binary-search implementation is executed directly.
- Q >= 2048: execute the first three existing batched levels, sample 16 already-computed coarse bases from `ret`, and detect direction reversals.
- reversal observed: continue the existing batched binary-search path from the already-computed coarse state.
- strong coarse locality: finish from the existing coarse bounds using the previous exact insertion position, exponential probing, and a final bounded binary search.

The selector controls strategy only. Each key's coarse interval remains the correctness envelope, so classifier accuracy is not a correctness dependency.

No extra query scan, heap allocation, persistent buffer, thread, cache, or sorter-path change is introduced.

### Final clean validation

Primary final revalidation:
https://github.com/Johnny-Kao/numpy/actions/runs/37153156389

Independent preceding clean A/B:
https://github.com/Johnny-Kao/numpy/actions/runs/37152299188

Final run:
- 4/4 jobs success
- Base and PR correctness fingerprints identical on all four jobs
- same-runner forward/reverse Base/PR measurement
- int32 / int64 / float64
- N = 1M / 10M
- Q = 1024 / 2048 / 4096 / 8192 / 100000
- left / right
- random / dense / medium / mostly-monotonic / duplicates

For active local cases (Q >= 2048):
- dense: median PR/Base 0.281x (~3.56x speedup), 192/192 faster
- medium: 0.447x (~2.24x), 168/168 faster
- mostly-monotonic: 0.299x (~3.34x), 192/192 faster
- duplicates: 0.235x (~4.25x), 192/192 faster
- combined: 0.285x (~3.51x), 744/744 faster

For random Q=2048/4096/8192:
- median PR/Base 0.991x
- p95 1.010x
- worst 1.029x across 144 comparisons
- interpret as near-current / within measurement noise, not as a claimed random-workload speedup

Q=100000 random showed a noisy tail on one constrained hosted runner while the other jobs remained near current; do not use that tail as either positive or negative performance evidence without a tighter interleaved harness.

### Portable threshold note

Activation run:
https://github.com/Johnny-Kao/numpy/actions/runs/37147552873

Observed hardware crossover differed:
- AMD EPYC 9V74 / 7763 accepted substantially lower activation thresholds in the tested matrix.
- Intel Xeon Platinum 8573C required a higher threshold; Q=1024 was close but missed the conservative p95 gate.

Therefore Q>=2048 is frozen as the initial portable default, not a globally optimal constant. Future work may lower or replace it with an architecture-neutral cheap signal. Do not add CPU-model-specific tuning to the initial PR.

### PR status

Research is closed. Clean production candidate and maintainer-facing draft are ready for review.

Do not open the upstream PR without explicit user signoff.


## 34. OSS Toolkit PR sign-off — final current-HEAD run found codegen/fallback cost

Date: 2026-10-04 JST

Production candidate:
- branch: `perf/searchsorted-locality-selector`
- candidate HEAD: `fb28dba5eb01bc97f4734ccb37921386db643890`
- upstream base: `2f1eca306857b641fb0fef0ab854a2d4137e1581`
- production activation gate: `Q >= 131072`
- final diff at sign-off: only `numpy/_core/src/npysort/binsearch.cpp`

Final Toolkit L4/L5 sign-off run:
https://github.com/Johnny-Kao/numpy/actions/runs/37172716011

Result:
- GitHub Actions workflow: **4/4 jobs PASS**
- Engineering sign-off decision: **REVISE**
- Reason: final current-HEAD validation exposed a reproducible performance regression in the historical path even when the locality gate is not active, plus a small number of active-local losing cases near the gate.

### Validation performed

Each job:
- pinned exact upstream base and exact candidate SHA;
- checked `git diff --check`;
- asserted the production diff contains only `numpy/_core/src/npysort/binsearch.cpp`;
- built separate base/candidate wheels;
- installed them into isolated environments;
- verified the installed candidate provenance;
- ran focused NumPy `searchsorted` tests;
- measured binary-size impact;
- ran interleaved Base-before -> Candidate -> Base-after performance validation;
- retained raw per-case results and summary artifacts.

Focused tests:
- **21 passed, 14848 deselected** on each of the four jobs.

Runner CPUs:
- Intel Xeon Platinum 8370C
- AMD EPYC 7763 (two logical resource profiles)
- AMD EPYC 9V74

Final production matrix:
- N = 1M / 10M
- Q = 100000 / 131071 / 131072 / 262144 / 1000000
- dtype = int32 / int64 / float64
- side = left / right
- shapes = dense / medium / mostly-monotonic / random / duplicates / reversal-bursts
- candidate compared against the median of same-runner Base-before and Base-after.

### Positive result — active locality gain remains very large

For `Q >= 131072`, active-local median Candidate/Base by runner:

| Runner profile | Median ratio | Median speedup | Median runtime reduction |
| --- | ---: | ---: | ---: |
| Intel micro | 0.236x | 4.25x | 76.45% |
| AMD EPYC 7763 small | 0.241x | 4.14x | 75.87% |
| AMD EPYC 7763 standard | 0.241x | 4.14x | 75.86% |
| AMD EPYC 9V74 standard | 0.214x | 4.67x | 78.60% |

At `Q=1M`, local medians reached roughly 4.8-5.7x speedup depending on runner.

So the locality mechanism itself still provides a very large production-shaped benefit.

### Active random/general cost

For active random cases (`Q >= 131072`):

| Runner profile | Median ratio | p95 | Worst |
| --- | ---: | ---: | ---: |
| Intel micro | 1.003x | 1.093x | 1.315x |
| AMD EPYC 7763 small | 0.936x | 1.018x | 1.025x |
| AMD EPYC 7763 standard | 0.984x | 1.065x | 1.140x |
| AMD EPYC 9V74 standard | 0.931x | 1.039x | 1.063x |

Some worst active-random tails coincide with substantial Base-before/Base-after drift and should not be interpreted as pure candidate overhead. However stable cases still show several ~5-9% regressions, so the non-target cost is not universally zero.

### Critical finding — below-gate path is not performance-identical

Although `Q < 131072` never enters the selector logically, the enlarged/restructured `binsearch` body changes compiler/code-generation behavior.

A particularly clean reproduced case:

- CPU: AMD EPYC 7763
- dtype: float64
- N = 1,000,000
- Q = 100,000 (< gate)
- shape: duplicates
- side: left
- Base-before: ~2.783 ms
- Candidate: ~3.576 ms
- Base-after: ~2.787 ms
- Candidate/Base: **1.284x**
- Base-after/Base-before drift: ~0.15%

This is a real ~28% regression with a stable baseline, not runner noise.

Similar ~27-28% below-gate regressions appear on related AMD float64/left cases at Q=100000 and Q=131071.

Therefore the statement "below-gate performance is untouched" is not currently valid at machine-code level even though the semantic branch does not activate.

### Active-local losing boundary cases

The final matrix also found two reproducible active-local losses near the gate:

- AMD EPYC 7763
- float64
- N = 1M
- Q = 131072
- duplicates
- side = left
- Candidate/Base ~= **1.103-1.104x**
- Base-before/Base-after drift < 0.5%

So strong locality is highly beneficial in aggregate, but the current production path does **not** yet satisfy a strict "routed local cases never regress" claim.

### Resource / binary-size cost

No new query pre-scan, O(Q) temporary allocation, persistent cache, thread, or shared mutable state is introduced by the production code.

Compiled `_multiarray_umath` size:
- base: 11,041,288 bytes
- candidate: 11,147,784 bytes
- delta: **+106,496 bytes (+0.965%)**

This is the concrete binary-size cost of the current implementation.

### Toolkit gate result

L4:
- correctness/focused tests: PASS
- build/provenance: PASS
- diff scope: PASS
- safety/state surface: PASS by design review (no new shared state/allocation/thread)
- performance/resource: **REVISE**
- future-risk note: threshold/crossover documentation present in production code

L5 final sign-off:
- **REVISE**
- do not create upstream Draft PR yet.

### Minimal next action

Do **not** reopen selector-policy or threshold research.

The shortest corrective path is to isolate the historical/current batched implementation from the locality implementation at the function/code-generation boundary so that below-gate/fallback execution does not inherit the enlarged locality body.

After that structural isolation:
1. rerun the same focused tests;
2. rerun the same pinned final sign-off matrix;
3. require below-gate/fallback performance to return to near-current behavior;
4. confirm active-local gain remains material;
5. only then re-enter L5 PR preparation.

No upstream PR without contributor signoff.
