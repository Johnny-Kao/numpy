import os, json, sys
import numpy as np
from acceptance_ledger import decision
rng=np.random.default_rng(260810)
rows=[]
for seed in (7,19,43):
 for dtype in ("int32","float64"):
  for n in (128,65536):
   a=np.arange(n,dtype=dtype)
   for q in (8,64,1024):
    for pattern in ("repeated","ordered_local","random","alternating","deceptive"):
     center=int(rng.integers(n//4,3*n//4))
     keys=np.full(q,center,dtype=dtype)
     if pattern=="ordered_local":keys=np.sort(rng.integers(center-2,center+3,size=q)).astype(dtype)
     if pattern=="random":keys=rng.integers(0,n,size=q).astype(dtype)
     if pattern=="alternating":keys[::2]=0;keys[1::2]=n-1
     if pattern=="deceptive":
      for j in range(1,17):
       idx=(j*(q-1))>>4
       if idx+1<q:keys[idx+1]=n-1 if j%2 else 0
     shadow=decision(a,keys,"left")
     output=np.searchsorted(a,keys,side="left")
     # Full correctness independent of gate and trace
     expected=np.searchsorted(a,keys,side="left",sorter=np.arange(n,dtype=np.intp))
     if not np.array_equal(output,expected):raise AssertionError("searchsorted mismatch")
     rows.append(dict(seed=seed,dtype=dtype,n=n,q=q,pattern=pattern,shadow=shadow))
print(json.dumps(rows))
