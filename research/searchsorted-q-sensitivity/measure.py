import os,json,time,statistics,resource,platform
import numpy as np
rng=np.random.default_rng(20261009)
rows=[]
for n in (128,4096,65536,1048576):
 a=np.arange(n,dtype=np.int64)
 for q in (4096,16384,65536,262144,1048576,4194304):
  for pattern in ("repeated","ordered_local","random","alternating","deceptive"):
   if pattern=="repeated":v=np.full(q,n//2,dtype=np.int64)
   elif pattern=="ordered_local":v=np.full(q,n//2,dtype=np.int64)+(np.arange(q,dtype=np.int64)*8//q)
   elif pattern=="random":v=rng.integers(0,n,size=q,dtype=np.int64)
   elif pattern=="alternating":v=np.where(np.arange(q)%2,n-1,0).astype(np.int64)
   else:
    v=np.full(q,n//2,dtype=np.int64)
    mask=np.arange(q)%64>2
    v[mask]=rng.integers(0,n,size=int(mask.sum()),dtype=np.int64)
   for side in ("left","right"):
    expected=np.searchsorted(a,v,side=side)
    assert np.array_equal(expected,np.searchsorted(a,v,side=side))
    # Measure work after data construction. Alternating pass directions are handled by outer workflow.
    loops=3 if q>=262144 else 7
    samples=[]
    for rep in range(5):
     t=time.perf_counter_ns()
     for k in range(loops):
      result=np.searchsorted(a,v,side=side)
      assert result.size==q
     samples.append((time.perf_counter_ns()-t)/loops)
    rows.append(dict(n=n,q=q,pattern=pattern,side=side,ns=statistics.median(samples)))
print(json.dumps(dict(rows=rows,cpu=platform.processor(),rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)))
