import json, numpy as np
SP='/private/tmp/claude-501/-Users-michalprusek-PycharmProjects-microtubules/7c791e01-8540-449c-8601-d781c8b046b6/scratchpad/temporal'
R=json.load(open(SP+'/res/measure.json')); L={o['vid']:o for o in json.load(open(SP+'/list.json'))}
vids=sorted(R['v16']); proj=[L[v]['project'] for v in vids]; P=sorted(set(proj))
keys=['n_per_frame','len_per_frame','S1','C_S1','S2','C_S2','T1','T2','T2abs','T3','E1_ratio','E1_below07','E1_cd','E1_cd_p90','drift_px']
def per(arm,k): return np.array([R[arm][v][k] for v in vids],float)
rng=np.random.default_rng(20261103)
def ci(d):
    m=[]
    for _ in range(10000):
        ps=rng.choice(P,len(P))
        x=np.concatenate([d[[i for i,p in enumerate(proj) if p==q]] for q in ps]); m.append(x.mean())
    return np.percentile(m,[2.5,97.5])
print(f"{'metric':12s} {'SPARSE35':>9s} {'V16':>9s} {'V16-S35':>9s} {'95% CI':>20s}")
for k in keys:
    a,b=per('sparse35',k),per('v16',k); d=b-a; lo,hi=ci(d)
    print(f"{k:12s} {a.mean():9.3f} {b.mean():9.3f} {d.mean():+9.3f}  [{lo:+8.3f},{hi:+8.3f}]")
print('\nreal pair vs same-scene control (mean over containers), paired by container')
for arm in ('sparse35','v16'):
    s,c=per(arm,'S1'),per(arm,'C_S1'); d=s-c; lo,hi=ci(d)
    print(arm,'S1 real',round(s.mean(),3),'ctrl',round(c.mean(),3),'real-ctrl',round(d.mean(),3),[round(lo,3),round(hi,3)])
    s,c=per(arm,'S2'),per(arm,'C_S2'); print(arm,'S2 real',round(s.mean(),3),'ctrl',round(c.mean(),3))
print('\nper container: n, S1, ctrl S1, T1, T2, T2abs, E1<0.7  (S35 / V16)')
for v in vids:
    a,b=R['sparse35'][v],R['v16'][v]
    print(v[:8],L[v]['project'][:6],f"{a['frames']}f", f"n {a['n_per_frame']:.0f}/{b['n_per_frame']:.0f}", f"S1 {a['S1']:.3f}/{b['S1']:.3f}", f"ctl {a['C_S1']:.3f}/{b['C_S1']:.3f}", f"T1 {a['T1']:.2f}/{b['T1']:.2f}", f"T2 {a['T2']:.2f}/{b['T2']:.2f}", f"T2abs {a['T2abs']:.0f}/{b['T2abs']:.0f}", f"E1 {a['E1_below07']:.2f}/{b['E1_below07']:.2f}", 'dec',b['decoders'])
