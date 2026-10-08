import json,sys,statistics,collections
label=sys.argv[1]
filenames=sys.argv[2:]
assert len(filenames)==8,len(filenames)
runs=[json.load(open(f)) for f in filenames]
groups={"base":[runs[0],runs[7]],"a1":[runs[1],runs[5]],"a2":[runs[2],runs[6]],"original_q":[runs[3],runs[4]]}
def key(r):return tuple(r.get(k) for k in ("seed","dtype","n","q","pattern","side","sorter"))
data={}
for name,results in groups.items():
 d=collections.defaultdict(list)
 for run in results:
  for r in run["rows"]:d[key(r)].append(r["median_ns"])
 data[name]={k:statistics.median(v) for k,v in d.items() if len(v)==2}
keys=set.intersection(*(set(d) for d in data.values()))
print("matrix",label,"matched_cases",len(keys))
for name in ("a1","a2","original_q"):
 ratios=[(data["base"][k]/data[name][k],k) for k in keys]
 values=[v for v,k in ratios]
 print("candidate",name,"median",round(statistics.median(values),4),"min",round(min(values),4),
 "regress_gt5",sum(v<.95 for v in values),"regress_gt10",sum(v<.9 for v in values),"improve_gt5",sum(v>1.05 for v in values))
 for v,k in sorted(ratios)[:6]:print("worst",name,round(v,4),k,"base_ns",round(data["base"][k],1),"candidate_ns",round(data[name][k],1))
 for p in ("repeated","ordered_local","random","alternating","deceptive","honest_repeated","random_interiors","within_bucket_random","full_alternating","reversed_blocks","single_spike"):
  z=[v for v,k in ratios if k[4]==p]
  if z:print("pattern",name,p,"n",len(z),"median",round(statistics.median(z),4),"min",round(min(z),4))
