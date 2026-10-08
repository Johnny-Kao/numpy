"""Independent NumPy-main versus A0 searchsorted performance matrix."""
import json, os, time, platform
import numpy as np
rng = np.random.default_rng(32895)
rows=[]
for dtype in ("int32","int64","float64"):
 for n in (128,4096,65536,1048576):
  arr=np.arange(n,dtype=dtype)
  for q in (1,16,1024,65536):
   for pattern in ("random","sorted","repeated"):
    keys=rng.integers(-n//10,n+n//10,size=q).astype(dtype)
    if pattern=="sorted": keys.sort()
    if pattern=="repeated": keys[:]=keys[0]
    for side in ("left","right"):
     for sorter in (False,True) if n<=65536 else (False,):
      a=arr
      s=None
      if sorter:
       ix=rng.permutation(n)
       a=arr[ix]
       s=np.argsort(a,kind="stable")
      for _ in range(3): np.searchsorted(a,keys,side=side,sorter=s)
      samples=[]
      # Batch enough calls to reduce resolution effects, cap large workloads.
      loops=max(2,min(2000,round(0.012/max(1e-7,(q*max(1,n.bit_length()))*2e-9))))
      for rep in range(7):
       start=time.perf_counter_ns()
       for _ in range(loops): np.searchsorted(a,keys,side=side,sorter=s)
       samples.append((time.perf_counter_ns()-start)/loops)
      samples.sort()
      rows.append(dict(dtype=dtype,n=n,q=q,pattern=pattern,side=side,sorter=sorter,median_ns=samples[3],samples_ns=samples))
print(json.dumps(dict(version=np.__version__,path=np.__file__,platform=platform.platform(),rows=rows)))
