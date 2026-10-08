import json,os,platform,resource,statistics,time
import numpy as np
from acceptance_ledger import decision
cases=[(11,'float64',65536,1024,'random','left'),(2027,'float64',65536,1024,'random','left'),(37,'float64',65536,1024,'random','left'),(101,'float64',65536,1024,'random','left'),(2027,'float64',128,1024,'alternating','right'),(2027,'float64',65536,1024,'alternating','right'),(11,'float64',128,1024,'alternating','right'),(8191,'float64',128,1024,'alternating','right')]
rows=[]
for seed,dtype,n,q,pattern,side in cases:
 # Test exact configuration, with independent deterministic data regeneration.
 rng=np.random.default_rng(seed)
 a=np.arange(n,dtype=dtype)
 center=int(rng.integers(n//4,3*n//4))
 if pattern=='random': keys=rng.integers(-n//10,n+n//10,size=q).astype(dtype)
 else: keys=np.resize(np.array([0,n-1],dtype=dtype),q)
 p=decision(a,keys,side)
 assert p=='structural_fallback',(seed,n,q,pattern,p)
 for _ in range(5):np.searchsorted(a,keys,side=side)
 for block in range(12):
  samples=[]
  for rep in range(11):
   t=time.perf_counter_ns()
   for _ in range(80): out=np.searchsorted(a,keys,side=side)
   samples.append((time.perf_counter_ns()-t)/80)
  rows.append(dict(case=[seed,dtype,n,q,pattern,side],block=block,path=p,median_ns=statistics.median(samples)))
print(json.dumps(dict(env=dict(np=np.__version__,path=np.__file__,platform=platform.platform(),rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,cpu=platform.processor()),rows=rows)))
