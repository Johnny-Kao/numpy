import sys,json,statistics,collections
files=sys.argv[1:]
assert len(files)==4
runs=[json.load(open(f)) for f in files]
def case(r):return tuple(r[k] for k in ("seed","dtype","n","q","pattern","side"))
by=collections.defaultdict(lambda:{"base":[],"candidate":[],"path":None})
for run,kind in zip(runs,("base","candidate","candidate","base")):
 for r in run["rows"]:
  z=by[case(r)];z[kind].append(r["median_ns"])
  if z["path"] is None:z["path"]=r["path"]
  assert z["path"]==r["path"]
print("environment",json.dumps([{k:r.get(k) for k in ("numpy_path","version","platform","maxrss_kib")} for r in runs]))
vals=[]
for k,r in by.items():
 assert len(r["base"])==len(r["candidate"])==2
 ratio=statistics.median(r["base"])/statistics.median(r["candidate"])
 vals.append(dict(case=k,path=r["path"],ratio=ratio))
print("matched",len(vals))
for path in sorted(set(v["path"] for v in vals)):
 x=[v for v in vals if v["path"]==path];rs=[v["ratio"] for v in x]
 print("path",path,"n",len(x),"median_speedup",round(statistics.median(rs),4),
       "regress_gt5pct",sum(r<.95 for r in rs),
       "regress_gt10pct",sum(r<.9 for r in rs),
       "improve_gt5pct",sum(r>1.05 for r in rs),
       "min",round(min(rs),4))
 for v in sorted(x,key=lambda x:x["ratio"])[:5]:print("worst",v)
for pat in ("repeated","ordered_local","random","alternating","deceptive"):
 x=[v for v in vals if v["case"][4]==pat]
 print("pattern",pat,"paths",dict(collections.Counter(v["path"] for v in x)),
       "median",round(statistics.median(v["ratio"] for v in x),4))
with open("/tmp/monte_carlo_summary.json","w") as f:json.dump(vals,f)
