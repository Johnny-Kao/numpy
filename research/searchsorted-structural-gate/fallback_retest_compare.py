import json,sys,statistics,collections
runs=[json.load(open(f)) for f in sys.argv[1:]]
assert len(runs)==4
by=collections.defaultdict(lambda:dict(base=[],cand=[]))
for run,kind in zip(runs,("base","cand","cand","base")):
 for row in run['rows']:
  k=tuple(row['case'])+(row['block'],)
  by[k][kind].append(row['median_ns'])
assert len(by)==96
out=collections.defaultdict(list)
for k,v in by.items():
 assert len(v['base'])==len(v['cand'])==2
 out[k[:-1]].append(statistics.median(v['base'])/statistics.median(v['cand']))
for k,values in out.items():
 print('case',k,'median_speedup',round(statistics.median(values),4),'min_block',round(min(values),4),'below_95pct',sum(x<.95 for x in values),'blocks',len(values))
print('overall',len(out),'cases','regress_5pct',sum(statistics.median(v)<.95 for v in out.values()))
for i,r in enumerate(runs):print('run_environment',i,r['env'])
