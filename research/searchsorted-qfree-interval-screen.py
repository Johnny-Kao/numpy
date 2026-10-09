"""Q-free selector information sufficiency screen. Diagnostic, not a NumPy speed benchmark."""
import random
from collections import Counter
N=8192
Q=4097
S=16
ANCHORS=[(j*(Q-1))//S for j in range(S+1)]
def cases(seed):
    rng=random.Random(seed)
    base=[rng.randrange(N//8,N//8+N//32) for _ in range(Q)]
    asc=sorted(base)
    yield "sorted", asc
    yield "repeated", [N//8]*Q
    yield "clustered", [N//8+rng.randrange(8) for _ in range(Q)]
    yield "random", base
    yield "alternating", [N//8 if i%2==0 else N//8+N//16 for i in range(Q)]
    spike=asc[:]
    spike[128]=N-1
    yield "hidden_spike", spike
    spike=asc[:]
    spike[ANCHORS[5]-1]=N-1
    yield "sample_predecessor_spike", spike
    block=asc[:]
    for k in range(0,Q-1,256):
        a=block[k:k+256]
        rng.shuffle(a)
        block[k:k+len(a)]=a
    yield "block_shuffle",block
def classifier(xs,mode):
    # The real code computes three coarse batched passes; here we emulate
    # each key's exact high 3 bits of insertion position in a sorted range.
    coarse=[x//(N//8) for x in xs]
    sampled=[coarse[i] for i in ANCHORS]
    if not (min(sampled)==max(sampled) and all(sampled[i]>=sampled[i-1] for i in range(1,len(sampled)))):
        return False
    # Existing selector also checks predecessors of sample boundaries.
    checks=sorted(set(ANCHORS+[i-1 for i in ANCHORS if i>0]))
    if mode=="midpoint":
        checks=sorted(set(checks+[(ANCHORS[j]+ANCHORS[j+1])//2 for j in range(S)]))
    elif mode=="quartile":
        checks=sorted(set(checks+[ANCHORS[j]+(ANCHORS[j+1]-ANCHORS[j])*k//4 for j in range(S) for k in (1,2,3)]))
    elif mode=="all_coarse":
        checks=list(range(Q))
    for i in checks:
        if i>0 and coarse[i]<coarse[i-1]:
            return False
        if i>0 and xs[i]<xs[i-1]:
            return False
    return True
def hazard(xs):
    # Deterministic surrogate for unsafe monotone predecessor reuse.
    return any(xs[i]<xs[i-1] for i in range(1,len(xs)))
counts={mode:Counter() for mode in ("existing","midpoint","quartile","all_coarse")}
for seed in range(100):
    for pat,xs in cases(seed):
        unsafe=hazard(xs)
        for mode,c in counts.items():
            selected=classifier(xs,mode)
            c["total"]+=1
            c["unsafe"]+=unsafe
            c["accepted"]+=selected
            c["false_accept"]+=selected and unsafe
            c["missed_safe"]+=(not selected) and (not unsafe)
print("INPUTS",counts["existing"]["total"],"each across 8 input distributions and 100 seeds")
for mode,c in counts.items():
    print(mode,dict(c))
print("NOTE: information sufficiency proxy only; no wall-clock timing, no assertion that unsafe=slower.")
