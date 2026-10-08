import json,statistics,time,platform,resource
import numpy as np

rng=np.random.default_rng(20261009)
rows=[]
for dtype in ("int32","int64","float64"):
 for n,q in ((128,1024),(4096,65536),(65536,65536),(1048576,65536)):
  a=np.arange(n,dtype=dtype)
  for pattern in ("random_interiors","reversed_blocks","full_alternating","single_spike","within_bucket_random","honest_repeated"):
   for side in ("left","right"):
    x=np.full(q,n//2,dtype=dtype)
    anchors={(j*(q-1))>>4 for j in range(17)}
    protect=anchors|{i-1 for i in anchors if i>0}
    free=np.asarray([i for i in range(q) if i not in protect],dtype=np.intp)
    if pattern=="random_interiors":x[free]=rng.integers(0,n,size=len(free))
    elif pattern=="reversed_blocks":x[free]=np.rint(np.linspace(n-1,0,len(free))).astype(dtype)
    elif pattern=="full_alternating":x[free[::2]]=0;x[free[1::2]]=n-1
    elif pattern=="single_spike":x[free[len(free)//2]]=n-1
    elif pattern=="within_bucket_random":
     width=max(2,n//8);x[free]=rng.integers(max(0,n//2-width//2),min(n,n//2+width//2),size=len(free))
    block_modes=[]
    for start in range(0,q,256):
     part=x[start:start+256]
     block_modes.append('locality' if len(part)>1 and bool(np.all(part[1:]>=part[:-1])) else 'batched')
    p='all_locality' if all(m=='locality' for m in block_modes) else ('all_batched' if all(m=='batched' for m in block_modes) else 'mixed')
    # Independent reference: the upstream baseline is checked separately by workflow ABBA;
    # basic insertion invariants are checked against a and x here.
    out=np.searchsorted(a,x,side=side)
    assert np.all((out>=0)&(out<=n))
    if side=="left":
     assert np.all((out==0)|(a[np.maximum(0,out-1)]<x))
     assert np.all((out==n)|(x<=a[np.minimum(n-1,out)]))
    else:
     assert np.all((out==0)|(a[np.maximum(0,out-1)]<=x))
     assert np.all((out==n)|(x<a[np.minimum(n-1,out)]))
    loops=max(3,min(100,300000//q))
    for _ in range(3):np.searchsorted(a,x,side=side)
    measurements=[]
    for rep in range(9):
     t=time.perf_counter_ns()
     for _ in range(loops):z=np.searchsorted(a,x,side=side)
     measurements.append((time.perf_counter_ns()-t)/loops)
    rows.append(dict(dtype=dtype,n=n,q=q,pattern=pattern,side=side,path=p,median_ns=statistics.median(measurements)))
print(json.dumps(dict(version=np.__version__,numpy_path=np.__file__,platform=platform.platform(),rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,rows=rows)))
