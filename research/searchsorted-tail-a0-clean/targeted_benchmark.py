import json, os, time, statistics, platform
import numpy as np
rng=np.random.default_rng(3289501)
cases=[
("float64",128,65536,"repeated","right",False),
("int32",128,1,"random","left",False),
("float64",1048576,65536,"repeated","right",False),
("float64",128,65536,"sorted","right",False),
("float64",1048576,1024,"repeated","right",False),
("float64",128,1024,"sorted","right",False),
("float64",128,1024,"random","right",False),
("float64",4096,65536,"repeated","left",True),
("int64",65536,65536,"repeated","left",True),
("int32",1048576,65536,"random","left",False),
]
out=[]
for dtype,n,q,pattern,side,sorter in cases:
 arr=np.arange(n,dtype=dtype)
 keys=rng.integers(-n//10,n+n//10,size=q).astype(dtype)
 if pattern=="sorted":keys.sort()
 if pattern=="repeated":keys[:]=keys[0]
 s=None
 if sorter:
  ix=rng.permutation(n);arr=arr[ix];s=np.argsort(arr,kind="stable")
 for i in range(5):np.searchsorted(arr,keys,side=side,sorter=s)
 loops=max(4,min(2000,round(0.025/max(1e-7,q*n.bit_length()*2e-9))))
 batches=[]
 for j in range(21):
  start=time.perf_counter_ns()
  for k in range(loops): np.searchsorted(arr,keys,side=side,sorter=s)
  batches.append((time.perf_counter_ns()-start)/loops)
 out.append(dict(dtype=dtype,n=n,q=q,pattern=pattern,side=side,sorter=sorter,median_ns=statistics.median(batches),samples_ns=batches))
print(json.dumps(dict(platform=platform.platform(),version=np.__version__,path=np.__file__,rows=out)))
