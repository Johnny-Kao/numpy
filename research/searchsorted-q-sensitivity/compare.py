import json,sys,statistics,collections
items={}
for name in ("base","q12","q14","q16","q18","q20","q22"):
 runs=[json.load(open(f"/tmp/qstudy/{name}_{i}.json")) for i in (1,2)]
 d=collections.defaultdict(list)
 for run in runs:
  for r in run["rows"]:d[(r["n"],r["q"],r["pattern"],r["side"])].append(r["ns"])
 items[name]={k:statistics.median(v) for k,v in d.items() if len(v)==2}
assert all(len(v)==240 for v in items.values()),{k:len(v) for k,v in items.items()}
for name in list(items)[1:]:
 for q in (4096,16384,65536,262144,1048576,4194304):
  keys=[k for k in items["base"] if k[1]==q]
  ratios=[items["base"][k]/items[name][k] for k in keys]
  print(name,"Q",q,"cases",len(keys),"median",round(statistics.median(ratios),4),
    "min",round(min(ratios),4),"regress_gt5",sum(x<.95 for x in ratios),
    "gain_gt5",sum(x>1.05 for x in ratios))
  for p in ("repeated","ordered_local","random","alternating","deceptive"):
   zz=[items["base"][k]/items[name][k] for k in keys if k[2]==p]
   print("  ",p,"median",round(statistics.median(zz),4),"min",round(min(zz),4))
