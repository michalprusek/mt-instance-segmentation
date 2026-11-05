#!/bin/bash
# L+H b80 for DEB_S4 at both of its own cuts on VAL (docs/TODO.md 2026-11-03 08:05). The own-cut cells already exist in bridge/; this adds the other cut of each block and prints the block mean per cut.
W=/data2/prusek/mt_work; U=$W/runs/ult; O=$U/deploy; mkdir -p $O; cd $W/mt34_work
export PYTHONPATH=.:src:scripts:synth:third_party OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 TMPDIR=/data2/prusek/tmp
P="nice -n 5 /scratch/prusek/envs/nnunet/bin/python"; m=DEB_S4
for c in 0.99 0.995; do
  $P l_bridge.py data/real/roi303_v2 $W/preds/GAUSS_${m}_roiv2_ep40_native $c $O/roiv2_t$c.json 80 16 val all > $O/roiv2_t$c.log 2>&1
  $P l_bridge.py data/real/mt34_htw_eval $W/preds/${m}_hb_ep40_native $c $O/hball_t$c.json 80 16 val all > $O/hball_t$c.log 2>&1
done
/scratch/prusek/envs/nnunet/bin/python - > $U/deploy_cut.txt <<'PY'
import json, sys
sys.path.insert(0, "scripts")
from step1_read import source_of
O = "/data2/prusek/mt_work/runs/ult/deploy/"; f1 = lambda s: 2 * s[0] / max(2 * s[0] + s[1] + s[2], 1); res = {}
for c in ("0.99", "0.995"):
    t = {"roi303": [0, 0, 0], "htw": [0, 0, 0], "mt34": [0, 0, 0]}
    for fn, split in (("roiv2", False), ("hball", True)):
        for n, _, r in json.load(open(f"{O}{fn}_t{c}.json"))["rows"]:
            q = r["b80"]; k = ("htw" if source_of(n) == "htw" else "mt34") if split else "roi303"
            for i, a in enumerate(("tp", "fp", "fn")): t[k][i] += q[a]
    F = {k: f1(v) for k, v in t.items()}; F["block"] = (F["htw"] + F["roi303"]) / 2; res[c] = F
    print(c, {k: round(v, 4) for k, v in F.items()}, {k: v for k, v in t.items()})
d = res["0.995"]["block"] - res["0.99"]["block"]; pick = "0.995" if d >= -0.001 else "0.99"
print(f"block mean 0.995 - 0.99 = {d:+.4f}; declared: higher block mean, ties within 0.001 to the higher cut -> DEPLOY CUT {pick}")
PY
cat $U/deploy_cut.txt; echo DEPLOY_CUT_DONE
