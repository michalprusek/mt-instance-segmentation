#!/bin/bash
cd /data2/prusek/mt_work/mt34_work
export PYTHONPATH=.:src:scripts:synth:third_party OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 TMPDIR=/data2/prusek/tmp
P=/scratch/prusek/envs/nnunet/bin/python; PD=/data2/prusek/mt_work/preds; O=/data2/prusek/mt_work/runs/ult/bridge
for mc in DEB_S3:0.995:0.99 DEB_S4:0.99:0.99 DEB_S5:0.99:0.99 DEB_S6:0.995:0.98; do IFS=: read m ch cs <<< "$mc"
  $P l_bridge.py data/real/roi303_v2 $PD/GAUSS_${m}_roiv2_ep40_native 0.995 $O/roiv2_$m.json 60,80,100 24 val all
  $P l_bridge.py data/real/mt34_htw_eval $PD/${m}_hb_ep40_native $ch $O/hball_$m.json 60,80,100 24 val all
  $P l_bridge.py data/synth_relink96b $PD/GAUSS_${m}_syn96b_ep40_native $cs $O/syn96b_$m.json 60,80,100 24 val all
done
/scratch/prusek/envs/mt/bin/python - <<PY
import json, sys, collections, numpy as np
sys.path.insert(0, "scripts")
from step1_read import source_of
U = "/data2/prusek/mt_work/runs/ult/bridge/"; MS = [f"DEB_S{k}" for k in (3, 4, 5, 6)]
gid = json.load(open("/data2/prusek/mt_work/tulen_ckpt/bench_v2/groups.json"))["gid"]
f1 = lambda t: 2 * t[0] / max(2 * t[0] + t[1] + t[2], 1)
def load(m, f, br, keep=lambda n: True):
    return {n: np.array((r[br]["tp"], r[br]["fp"], r[br]["fn"], r[br]["jn"], r[br]["jp"]), float) for n, _, r in json.load(open(f"{U}{f}_{m}.json"))["rows"] if keep(n)}
SETS = {"roi303": ("roiv2", lambda n: True), "htw": ("hball", lambda n: source_of(n) == "htw"), "MT-34": ("hball", lambda n: source_of(n) != "htw"), "synth": ("syn96b", lambda n: True)}
D = {}
for br in ("b60", "b80", "b100"):
    print(f"== {br}")
    for lab, (f, keep) in SETS.items():
        X = [load(m, f, "b60", keep) for m in MS]; Y = [load(m, f, br, keep) for m in MS]
        sx, sy = sum(sum(x.values()) for x in X), sum(sum(y.values()) for y in Y)
        key = (lambda n: gid.get(n, n)) if lab == "roi303" else (lambda n: n)
        names = sorted(X[0]); keys = sorted({key(n) for n in names}); ix = {k: i for i, k in enumerate(keys)}; A = np.zeros((len(keys), 3)); B = np.zeros((len(keys), 3))
        for x, y in zip(X, Y):
            for n in names: A[ix[key(n)]] += x[n][:3]; B[ix[key(n)]] += y[n][:3]
        rng = np.random.default_rng(0); o = []
        for _ in range(5000):
            s = rng.integers(0, len(keys), len(keys)); o.append(f1(B[s].sum(0)) - f1(A[s].sum(0)))
        lo, hi = np.percentile(o, [2.5, 97.5]); d = f1(sy) - f1(sx); D[br, lab] = d
        per = [f1(sum(y.values())) - f1(sum(x.values())) for x, y in zip(X, Y)]
        print(f"   {lab:7s} F1 {f1(sy):.4f} ({d:+.4f} [{lo:+.4f},{hi:+.4f}]) per model " + " ".join(f"{p:+.4f}" for p in per) + f" | dTP {(sy[0] - sx[0]) / 4:+.0f} dFP {(sy[1] - sx[1]) / 4:+.0f} | identity {sy[4] / max(sy[3], 1):.3f} (was {sx[4] / max(sx[3], 1):.3f})")
bm = (D["b80", "roi303"] + D["b80", "htw"]) / 2; ok = bm >= 0.010 and D["b80", "roi303"] >= -0.005 and D["b80", "htw"] >= -0.005 and D["b80", "synth"] >= -0.005
print(f"bridge 80: block mean {bm:+.4f}, line +0.010 ->", "ADOPT" if ok else "NOT ADOPTED"); print("BRIDGE_READ_DONE")
PY
