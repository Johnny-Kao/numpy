import numpy as np
import platform

rng=np.random.default_rng(32895)
cases=0
for dtype in (np.int32,np.int64,np.float32,np.float64,np.complex128,"datetime64[ns]"):
    for n in (0,1,2,3,4,5,7,8,16,31,32,64,256,1024):
        if dtype=="datetime64[ns]":
            arr=np.sort(rng.integers(-20,21,n).astype("datetime64[ns]"))
            queries=rng.integers(-30,31,131).astype("datetime64[ns]")
            arr=np.sort(np.r_[arr,np.datetime64("NaT")])
            queries=np.r_[queries,np.datetime64("NaT")]
        elif dtype==np.complex128:
            arr=np.sort((rng.integers(-10,11,n)+1j*rng.integers(-5,6,n)).astype(dtype))
            queries=(rng.integers(-12,13,131)+1j*rng.integers(-7,8,131)).astype(dtype)
            arr=np.sort(np.r_[arr,np.nan+0j])
            queries=np.r_[queries,np.nan+0j]
        else:
            arr=np.sort(rng.integers(-20,21,n).astype(dtype))
            queries=rng.integers(-30,31,131).astype(dtype)
            if np.issubdtype(np.dtype(dtype),np.floating):
                arr=np.sort(np.r_[arr,[-np.inf,np.inf,np.nan]].astype(dtype))
                queries=np.r_[queries,[-np.inf,-0.0,0.0,np.inf,np.nan]].astype(dtype)
        for keys in (queries,queries[::-1],np.sort(queries),np.repeat(queries[:3],10),queries[::2]):
            for a in (arr,arr[::2]):
                for side in ("left","right"):
                    actual=np.searchsorted(a,keys,side=side)
                    ref=np.array([np.searchsorted(a,v,side=side) for v in keys],dtype=np.intp)
                    np.testing.assert_array_equal(actual,ref)
                    cases+=1
                    if a.size:
                        indices=rng.permutation(len(a))
                        shuffled=a[indices]
                        sorter=np.argsort(shuffled,kind="stable")
                        np.testing.assert_array_equal(np.searchsorted(shuffled,keys,side=side,sorter=sorter),ref)
                        cases+=1
print("PASS cases=",cases,"numpy=",np.__version__,"path=",np.__file__,"CPU=",platform.processor())
