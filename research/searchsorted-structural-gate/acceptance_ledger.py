import numpy as np, json
from collections import Counter
# Shadow the exact existing coarse-bucket selector without instrumenting timed C++.
def decision(a,keys,side):
 n=len(a);q=len(keys)
 if not n or q<=16 or a.strides[0]!=a.itemsize or keys.strides[0]!=keys.itemsize:
  return "metadata_fallback"
 length=n;half=length>>1;length-=half
 bases=np.where(a[half]<=keys if side=="right" else a[half]<keys,half,0).astype(np.intp)
 completed=1
 while length>1 and completed<3:
  half=length>>1;length-=half
  vals=a[bases+half]
  bases+=(vals<=keys if side=="right" else vals<keys)*half
  completed+=1
 reversed_=False;direction=0;prev=int(bases[0]);first=prev;same=True;last=q-1
 for j in range(1,17):
  i=(j*last)>>4;pos=int(bases[i])
  if pos!=first:same=False
  if pos>prev:
   if direction<0:reversed_=True;break
   direction=1
  elif pos<prev:
   if direction>0:reversed_=True;break
   direction=-1
  prev=pos
 if not reversed_ and same and direction>=0:
  for j in range(17):
   i=(j*last)>>4
   if i>0 and keys[i]<keys[i-1]:reversed_=True;break
 return "accepted" if not reversed_ and same and direction>=0 and length>1 else "structural_fallback"

rng=np.random.default_rng(32895);out=[]
for dtype in ("int32","int64","float64"):
 for n in (128,4096,65536,1048576):
  a=np.arange(n,dtype=dtype)
  for q in (1,16,64,1024,65536,1048576):
   for pattern in ("random","sorted","repeated","clustered","alternating"):
    if q==1048576 and pattern=="alternating":continue
    keys=rng.integers(-n//10,n+n//10,size=q).astype(dtype)
    if pattern=="sorted":keys.sort()
    if pattern=="repeated":keys[:]=keys[0]
    if pattern=="clustered":keys[:]=n//2;keys[::7]=n//2+1
    if pattern=="alternating":keys[::2]=0;keys[1::2]=n-1
    for side in ("left","right"):
     for sorter in ((False,True) if n<=65536 and q<=65536 else (False,)):
      # Search with sorter uses argbinsearch, not the optimized binsearch.
      if sorter: rng.permutation(n)  # mirror benchmark RNG consumption
      path="sorter_fallback" if sorter else decision(a,keys,side)
      out.append(dict(dtype=dtype,n=n,q=q,pattern=pattern,side=side,sorter=sorter,path=path))
print(json.dumps(out))
