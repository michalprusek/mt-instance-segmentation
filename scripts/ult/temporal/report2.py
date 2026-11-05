import json, numpy as np
SP='/private/tmp/claude-501/-Users-michalprusek-PycharmProjects-microtubules/7c791e01-8540-449c-8601-d781c8b046b6/scratchpad/temporal'
R=json.load(open(SP+'/res/measure2.json')); L={o['vid']:o for o in json.load(open(SP+'/list.json'))}
vids=sorted(R['v16']); proj=[L[v]['project'] for v in vids]; P=sorted(set(proj)); rng=np.random.default_rng(20261103)
IDX={q:[i for i,p in enumerate(proj) if p==q] for q in P}
def ci(d):
    m=[d[np.concatenate([IDX[q] for q in rng.choice(P,len(P))])].mean() for _ in range(10000)]
    return np.percentile(m,[2.5,97.5])
def col(m,k): return np.array([R[m][v][k] for v in vids],float)
print('(a) S1 by shift            SPARSE35    V16    V16-S35   95% CI')
for k,n in (('S0','no shift'),('Se','estimator'),('Ss','search'),('C','same-scene control')):
    a,b=col('sparse35',k),col('v16',k); lo,hi=ci(b-a); print(f'  {n:20s} {a.mean():8.3f} {b.mean():7.3f} {(b-a).mean():+8.3f}  [{lo:+.3f},{hi:+.3f}]')
print('\n(b) estimator vs search, per pair, px')
for m in ('sparse35','v16'):
    e=np.concatenate([R[m][v]['err'] for v in vids]); s=np.concatenate([R[m][v]['search'] for v in vids]); x=np.concatenate([R[m][v]['est'] for v in vids])
    print(f'  {m:9s} pairs {len(e)} |err| med {np.median(e):.1f} p90 {np.percentile(e,90):.1f} max {e.max():.1f} share>3.33 {np.mean(e>5/1.5):.2f} | estimate med {np.median(x):.1f} p90 {np.percentile(x,90):.1f} max {x.max():.1f} | search med {np.median(s):.1f} p90 {np.percentile(s,90):.1f} max {s.max():.1f} share>3.33 {np.mean(s>5/1.5):.2f}')
print('\n(c) tracker arms')
K=['tracks','T1','T2','T2abs','T3','E1_ratio','E1_below07','E1_cd','E1_cd_p90']
for m in ('sparse35','v16'):
    print(' ',m)
    base={k:np.array([R[m][v]['arms']['estimator'][k] for v in vids],float) for k in K}
    for k in K:
        z=np.array([R[m][v]['arms']['zero'][k] for v in vids],float); s=np.array([R[m][v]['arms']['search'][k] for v in vids],float)
        l1,h1=ci(z-base[k]); l2,h2=ci(s-base[k])
        print(f'    {k:11s} deployed {base[k].mean():9.2f} | zero {z.mean():9.2f} d {(z-base[k]).mean():+8.2f} [{l1:+.2f},{h1:+.2f}] | search {s.mean():9.2f} d {(s-base[k]).mean():+8.2f} [{l2:+.2f},{h2:+.2f}]')
print('\nV16 - SPARSE35 under each tracker arm')
for a in ('estimator','zero','search'):
    for k in ('T1','T2','T2abs','T3','E1_below07'):
        d=np.array([R['v16'][v]['arms'][a][k]-R['sparse35'][v]['arms'][a][k] for v in vids]); lo,hi=ci(d); print(f'  {a:9s} {k:10s} {d.mean():+9.3f} [{lo:+.3f},{hi:+.3f}]')
print('\nper container, V16: S1 none/est/search/control | T1 est/zero/search | T2abs est/zero/search | median length-of-track frames est/zero/search')
for v in vids:
    r=R['v16'][v]; a=r['arms']
    print(v[:8],f"{r['S0']:.2f}/{r['Se']:.2f}/{r['Ss']:.2f}/{r['C']:.2f} | {a['estimator']['T1']:.2f}/{a['zero']['T1']:.2f}/{a['search']['T1']:.2f} | {a['estimator']['T2abs']:.0f}/{a['zero']['T2abs']:.0f}/{a['search']['T2abs']:.0f} | {a['estimator']['T3']:.0f}/{a['zero']['T3']:.0f}/{a['search']['T3']:.0f} | search shift med {np.median(r['search']):.1f} max {max(r['search']):.1f}")
