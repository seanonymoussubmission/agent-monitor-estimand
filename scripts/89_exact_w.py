#!/usr/bin/env python3
"""Exact per-corpus same-unit pair weight w, and the Table-7 turn-cap statistic."""
import glob,json,collections,csv
import numpy as np, pyarrow.parquet as pq
def wstat(fails, within_pairs):
    P=sum(fails); N=sum(1-f for f in fails)
    return within_pairs/(P*N), P, N
# C1
f=[]; 
for p in sorted(glob.glob("data/swe_agent_traj/data/*.parquet")):
    f+= [1-int(bool(x)) for x in pq.read_table(p,columns=["target"]).column("target").to_pylist()]
w,P,N=wstat(np.array(f),220055); print(f"C1: P={P} N={N} w={w:.2e}")
# C2
f=[1-int(bool(x)) for x in pq.read_table("data/swe_rebench/trajectories.parquet",columns=["resolved"]).column("resolved").to_pylist()]
w,P,N=wstat(np.array(f),38576); print(f"C2: P={P} N={N} w={w:.2e}")
# C3
sc=pq.read_table("data/liveclaw/data/v0.2.1-00000-of-00001.parquet",columns=["score"]).column("score").to_pylist()
f=np.array([1 if (s is None or s!=1.0) else 0 for s in sc])
w,P,N=wstat(f,1000); print(f"C3: P={f.sum()} N={len(f)-f.sum()} w={w:.2e}")
# C4-L probe (k=1): recompute from npz
Z=np.load("FINAL/merged.npz",allow_pickle=False); sel=Z["k"]==1
task,y=Z["task"][sel],Z["y"][sel].astype(int)   # y=1 success? check: pass rate .588 => y is success
fails=1-y
wp=0
for u in set(task):
    m=task==u; wp+=fails[m].sum()*(y[m]).sum()
w=wp/(fails.sum()*y.sum()); print(f"C4-L: P={fails.sum()} N={y.sum()} within={wp} w={w:.2e}")
# turn-cap stat: fraction of SUCCESSFUL C4-L runs with <=10 commands
import re,os
outm={}
for r in csv.DictReader(open("data/runs_laguna.csv")):
    outm[(r["task"],int(r["run"]))]=int(r["success"])
RE=re.compile(r"^(.*)_run(\d+)\.json$"); tot=short=0
for fp in sorted(glob.glob("data/lph/swebench/laguna_xs2_full/*_run*.json")):
    m=RE.match(os.path.basename(fp)); 
    if not m: continue
    key=(m.group(1),int(m.group(2)))
    if key not in outm: key=(m.group(1),int(m.group(2))-1)
    if key not in outm or not outm[key]: continue
    try: ch=json.load(open(fp)).get("command_history") or []
    except Exception: continue
    tot+=1; short+= (len(ch)<=10)
print(f"C4-L successful runs with <=10 commands: {short}/{tot} = {100*short/tot:.1f}%")
