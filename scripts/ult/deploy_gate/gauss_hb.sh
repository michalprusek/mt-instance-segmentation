#!/bin/bash
# The hb block's dumps are uniform fusion (chain.sh: single tile for htw, but NOT for MT-34). The deployed path is gauss everywhere, so read DEB_S4 on hb under gauss on VAL: reference polylines for the gate and the MT-34 and htw numbers of the deployed path. docs/TODO.md 2026-11-03.
W=/data2/prusek/mt_work; G=$W/runs/ult/deploy/gate; C=0.995; N=/scratch/prusek/envs/nnunet/bin/python; D=$W/preds/GAUSS_DEB_S4_hb_ep40_native
cd $W/mt34_work; export PYTHONPATH=.:src:scripts:scripts/lpc:synth:third_party OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 TMPDIR=/data2/prusek/tmp
[ -d $D ] || CUDA_VISIBLE_DEVICES=0 $N scripts/eval_v5.py --ckpt $W/runs/DEB_S4/ep040.pth --arch nnunet --data data/real/mt34_htw_eval --split val --include-empty --no-fov --params src/instance/params_a_derived.json --cov 0.80 --thrs 0.98 --infer-scale 1.0 --fusion gauss --dump-pred $D --out-json $G/out/dump_hb_gauss.json > $G/out/dump_hb_gauss.log 2>&1
ls $D | wc -l
nice -n 5 $N $G/research_polys.py data/real/mt34_htw_eval $D $C $G/out/res_hb_gauss.json 16
nice -n 5 $N l_bridge.py data/real/mt34_htw_eval $D $C $W/runs/ult/deploy/hball_gauss_t$C.json 60,80 16 val all
nice -n 5 $N l_bridge.py data/real/mt34_htw_eval $D 0.99 $W/runs/ult/deploy/hball_gauss_t0.99.json 60,80 16 val all
$N - <<'PY'
import json, sys
sys.path.insert(0, "scripts")
from step1_read import source_of
f1 = lambda s: 2 * s[0] / max(2 * s[0] + s[1] + s[2], 1)
for tag, fn in (("uniform dump 0.995", "hball_t0.995"), ("gauss 0.995", "hball_gauss_t0.995"), ("uniform dump 0.99", "hball_t0.99"), ("gauss 0.99", "hball_gauss_t0.99")):
    t = {"htw": [0, 0, 0], "mt34": [0, 0, 0]}
    for n, _, r in json.load(open(f"/data2/prusek/mt_work/runs/ult/deploy/{fn}.json"))["rows"]:
        q = r["b80"]; k = "htw" if source_of(n) == "htw" else "mt34"
        for i, a in enumerate(("tp", "fp", "fn")): t[k][i] += q[a]
    print(tag, {k: (round(f1(v), 4), v) for k, v in t.items()})
PY
cd $G; unset PYTHONPATH
$N compare_polys.py out/res_hb_gauss.json out/e2e_hb_eps0.json; echo "e2e hb vs gauss research exit $?"
cd $W/mt34_work; export PYTHONPATH=.:src:scripts:scripts/lpc:synth:third_party
nice -n 5 $N $G/score_polys.py data/real/mt34_htw_eval $G/out/e2e_hb_rdp.json $W/runs/ult/deploy/hball_gauss_t$C.json
echo GAUSS_HB_DONE
