import json,sys,statistics,collections
paths=sys.argv[1:]
runs=[json.load(open(p)) for p in paths]
assert len(runs)==4
assert all(len(x["rows"])==len(runs[0]["rows"]) for x in runs)
def key(r): return tuple(r[k] for k in ("dtype","n","q","pattern","side","sorter"))
base=collections.defaultdict(list);a0=collections.defaultdict(list)
for run,kind in zip(runs,("base","a0","a0","base")):
 for row in run["rows"]:
  (base if kind=="base" else a0)[key(row)].append(row["median_ns"])
ratios={}
for k in base:
 ratios[k]=statistics.median(base[k])/statistics.median(a0[k])
vals=list(ratios.values())
print("matched_cases",len(vals),"median_speedup",round(statistics.median(vals),4),"regress_gt5pct",sum(x<.95 for x in vals),"improve_gt5pct",sum(x>1.05 for x in vals))
for field,ix in (("pattern",3),("q",2),("dtype",0),("sorter",5)):
 print(field, {str(v):round(statistics.median(r for k,r in ratios.items() if k[ix]==v),4) for v in sorted(set(k[ix] for k in ratios))})
for k,v in sorted(ratios.items(),key=lambda kv:kv[1])[:12]: print("worst",repr(k),round(v,4))
with open("comparison.json","w") as f:json.dump({"n":len(vals),"median_speedup":statistics.median(vals),"over5pct_regressions":sum(x<.95 for x in vals),"ratios":[{"case":k,"speedup":v} for k,v in ratios.items()]},f)
