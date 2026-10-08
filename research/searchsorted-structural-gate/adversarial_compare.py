import json,sys,statistics,collections
runs=[json.load(open(x)) for x in sys.argv[1:]]
assert len(runs)==4
d=collections.defaultdict(lambda:dict(base=[],cand=[],path=None))
for run,kind in zip(runs,("base","cand","cand","base")):
 for r in run["rows"]:
  k=tuple(r[x] for x in ("dtype","n","q","pattern","side"))
  e=d[k];e[kind].append(r["median_ns"])
  if e["path"] is None:e["path"]=r["path"]
  assert e["path"]==r["path"]
allrows=[]
for k,v in d.items():
 assert len(v["base"])==len(v["cand"])==2
 ratio=statistics.median(v["base"])/statistics.median(v["cand"])
 allrows.append((ratio,v["path"],k))
print("case_count",len(allrows),"accepted",sum(r[1]=="accepted" for r in allrows))
for path in sorted(set(r[1] for r in allrows)):
 rr=sorted((r for r in allrows if r[1]==path),key=lambda r:r[0])
 print("path",path,"cases",len(rr),"median",round(statistics.median(r[0] for r in rr),4),"regress_gt5",sum(r[0]<.95 for r in rr))
 for r in rr[:12]:print("worst",r)
for pattern in sorted(set(r[2][3] for r in allrows)):
 rr=[r for r in allrows if r[2][3]==pattern]
 print("pattern",pattern,"accepted",sum(r[1]=="accepted" for r in rr),"median",round(statistics.median(r[0] for r in rr),4),"worst",round(min(r[0] for r in rr),4))
