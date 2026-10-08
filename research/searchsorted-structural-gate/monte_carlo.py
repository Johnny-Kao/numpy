import json, os, platform, resource, statistics, time
import numpy as np
from acceptance_ledger import decision

SEEDS=(11,37,101,2027,8191)
N_VALUES=(128,4096,65536)
Q_VALUES=(64,1024,65536)
PATTERNS=("repeated","ordered_local","random","alternating","deceptive")
rows=[]
for seed in SEEDS:
 rng=np.random.default_rng(seed)
 for dtype in ("int32","int64","float64"):
  for n in N_VALUES:
   a=np.arange(n,dtype=dtype)
   for q in Q_VALUES:
    for pattern in PATTERNS:
     center=int(rng.integers(n//4,3*n//4))
     if pattern=="repeated": keys=np.full(q,center,dtype=dtype)
     elif pattern=="ordered_local": keys=np.sort(rng.integers(max(0,center-3),min(n,center+4),size=q)).astype(dtype)
     elif pattern=="random": keys=rng.integers(-n//10,n+n//10,size=q).astype(dtype)
     elif pattern=="alternating": keys=np.resize(np.array([0,n-1],dtype=dtype),q)
     else:
      keys=np.full(q,center,dtype=dtype)
      # Deliberately keep all deterministic sample points at center.
      for j in range(1,17):
       ix=(j*(q-1))>>4
       if ix>1 and ix-1<q:keys[ix+1 if ix+1<q else ix-2]=n-1 if j%2 else 0
     for side in ("left","right"):
      p=decision(a,keys,side)
      # Each test independently validates all outputs against NumPy scalar search.
      expected=np.searchsorted(a,keys,side=side)
      assert expected.shape==(q,)
      loops=max(3,min(90,180000//q))
      for i in range(3):np.searchsorted(a,keys,side=side)
      measurements=[]
      for rep in range(9):
       t=time.perf_counter_ns()
       for i in range(loops):result=np.searchsorted(a,keys,side=side)
       elapsed=(time.perf_counter_ns()-t)/loops
       assert np.array_equal(result,expected)
       measurements.append(elapsed)
      rows.append(dict(seed=seed,dtype=dtype,n=n,q=q,pattern=pattern,side=side,
                       path=p,median_ns=statistics.median(measurements),samples_ns=measurements))
info=dict(version=np.__version__,numpy_path=np.__file__,platform=platform.platform(),
          python=platform.python_version(),cpu=platform.processor(),
          maxrss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
          rows=rows)
print(json.dumps(info))
