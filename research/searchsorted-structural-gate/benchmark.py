import numpy as np, time, json, statistics, os
rng=np.random.default_rng(32895)
rows=[]
for dtype in ("int32","int64","float64"):
 for n in (128,4096,65536,1048576):
  a=np.arange(n,dtype=dtype)
  for q in (1,16,64,1024,65536,1048576):
   for pattern in ("random","sorted","repeated","clustered","alternating"):
    if q==1048576 and pattern not in ("random","sorted","repeated","clustered"):continue
    keys=rng.integers(-n//10,n+n//10,size=q).astype(dtype)
    if pattern=="sorted": keys.sort()
    if pattern=="repeated": keys[:]=keys[0]
    if pattern=="clustered": keys[:]=n//2;keys[::7]=n//2+1
    if pattern=="alternating": keys[::2]=0;keys[1::2]=n-1
    for side in ("left","right"):
     for sorter in ((False,True) if n<=65536 and q<=65536 else (False,)):
      arr=a;s=None
      if sorter:
       ix=rng.permutation(n);arr=a[ix];s=np.argsort(arr,kind="stable")
      for _ in range(2):np.searchsorted(arr,keys,side=side,sorter=s)
      loops=max(2,min(350,int(1000000/max(1,q))))
      samples=[]
      for rep in range(7):
       start=time.perf_counter_ns()
       for _ in range(loops): np.searchsorted(arr,keys,side=side,sorter=s)
       samples.append((time.perf_counter_ns()-start)/loops)
      rows.append(dict(dtype=dtype,n=n,q=q,pattern=pattern,side=side,sorter=sorter,median_ns=statistics.median(samples)))
print(json.dumps(dict(path=np.__file__,rows=rows)))
