#!/bin/bash
# Deployment gate on kajman (docs/TODO.md 2026-11-03 08:05). usage: gate_run.sh [polys|fixture|e2e]
W=/data2/prusek/mt_work; G=$W/runs/ult/deploy/gate; C=0.995; PKG=$G/hub/backend/segmentation/models/microtubule; N=/scratch/prusek/envs/nnunet/bin/python; mkdir -p $G/out $G/weights
ln -sf $W/runs/DEB_S4/ep040.pth $G/weights/microtubule_v16_deb_s4_ep040.pth
R=$W/preds/GAUSS_DEB_S4_roiv2_ep40_native; H=$W/preds/DEB_S4_hb_ep40_native
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 TMPDIR=/data2/prusek/tmp
case "$1" in
polys)
  cd $W/mt34_work; export PYTHONPATH=.:src:scripts:scripts/lpc:synth:third_party
  nice -n 5 $N $G/research_polys.py data/real/roi303_v2 $R $C $G/out/res_roi.json 16
  nice -n 5 $N $G/research_polys.py data/real/mt34_htw_eval $H $C $G/out/res_hb.json 16
  unset PYTHONPATH; cd $G
  for b in roi hb; do $N -c "import json;json.dump(sorted(json.load(open('out/res_$b.json'))),open('out/names_$b.json','w'))"; done
  nice -n 5 $N hub_polys.py $PKG $R out/names_roi.json $C out/hub_roi.json 16; $N compare_polys.py out/res_roi.json out/hub_roi.json; echo "roi exit $?"
  nice -n 5 $N hub_polys.py $PKG $H out/names_hb.json $C out/hub_hb.json 16; $N compare_polys.py out/res_hb.json out/hub_hb.json; echo "hb exit $?"
  echo "== mutation: reach 60 in the hub vector must FAIL the comparison"
  rm -rf mut; mkdir -p mut; cp -r $PKG mut/microtubule; $N -c "import json;p='mut/microtubule/params_v16.json';d=json.load(open(p));d['match_reach']=60.0;json.dump(d,open(p,'w'))"
  nice -n 5 $N hub_polys.py mut/microtubule $R out/names_roi.json $C out/mut_roi.json 16; $N compare_polys.py out/res_roi.json out/mut_roi.json | tail -1; echo "mutation exit ${PIPESTATUS[0]}"
  echo "== mutation: float32 map instead of the stored float16 is not reachable here (dumps are float16); covered in e2e"
  echo POLYS_DONE;;
fixture)
  cd $W/mt34_work; export PYTHONPATH=.:src:scripts:scripts/lpc:synth:third_party CUDA_VISIBLE_DEVICES=0
  X=$G/hub/backend/segmentation/tests/fixtures
  FR=$($N - <<'PY'
import glob, numpy as np, sys
sys.path[:0] = ["src", "scripts"]
from mt_bench.cvat_import import read_frame_h5
from step1_read import source_of
rows = {"roi": [], "htw": [], "mt34": []}
for d, tag in (("data/real/roi303_v2", "roi"), ("data/real/mt34_htw_eval", "hb")):
    for p in sorted(glob.glob(d + "/*.h5")):
        f = read_frame_h5(p)
        if str(f["attrs"].get("split")) != "val": continue
        k = "roi" if tag == "roi" else ("htw" if source_of(p.split("/")[-1]) == "htw" else "mt34")
        rows[k].append((len(f["polylines"]), p))
out = []
for k, r in rows.items():
    r.sort(); out.append(f"{k}_median={r[len(r) // 2][1]}"); out.append(f"{k}_densest={r[-1][1]}")
print(" ".join(out))
PY
)
  echo "frames: $FR"
  $N $X/mt_v16/make_mt_fixture_v16.py --frame-png $X/mt_sparse35/mt_sparse35_synthetic_irm.png --gt-json $X/mt_sparse35/gt_polylines.json --thr $C --out-dir $G/fixture --frames $FR
  echo FIXTURE_DONE;;
e2e)
  cd $G; unset PYTHONPATH
  CUDA_VISIBLE_DEVICES=0 nice -n 5 $N hub_e2e.py $PKG $G/weights/microtubule_v16_deb_s4_ep040.pth $W/mt34_work/data/real/roi303_v2 $R out/names_roi.json $C out/e2e_roi 0/2 > out/e2e_roi0.log 2>&1 &
  CUDA_VISIBLE_DEVICES=1 nice -n 5 $N hub_e2e.py $PKG $G/weights/microtubule_v16_deb_s4_ep040.pth $W/mt34_work/data/real/roi303_v2 $R out/names_roi.json $C out/e2e_roi 1/2 > out/e2e_roi1.log 2>&1 &
  CUDA_VISIBLE_DEVICES=0 nice -n 5 $N hub_e2e.py $PKG $G/weights/microtubule_v16_deb_s4_ep040.pth $W/mt34_work/data/real/mt34_htw_eval $H out/names_hb.json $C out/e2e_hb 0/1 > out/e2e_hb0.log 2>&1 &
  wait
  $N - <<'PY'
import json, numpy as np
for b, K in (("roi", 2), ("hb", 1)):
    for t in ("eps0", "rdp"):
        d = {}
        for k in range(K): d.update(json.load(open(f"out/e2e_{b}_{t}_{k}.json")))
        json.dump(d, open(f"out/e2e_{b}_{t}.json", "w"))
    S = [r for k in range(K) for r in json.load(open(f"out/e2e_{b}_stats_{k}.json"))]
    print(b, "frames", len(S), "map max|dp| worst", max(r["maxdp"] for r in S), "flips total", sum(r["flips"] for r in S), "predict s median", round(float(np.median([r["t_predict"] for r in S])), 1), "max", round(max(r["t_predict"] for r in S), 1), "largest", max(tuple(r["shape"]) for r in S))
PY
  $N compare_polys.py out/res_roi.json out/e2e_roi_eps0.json; echo "e2e roi exit $?"
  $N compare_polys.py out/res_hb.json out/e2e_hb_eps0.json; echo "e2e hb exit $?"
  cd $W/mt34_work; export PYTHONPATH=.:src:scripts:scripts/lpc:synth:third_party
  nice -n 5 $N $G/score_polys.py data/real/roi303_v2 $G/out/e2e_roi_rdp.json $W/runs/ult/deploy/roiv2_t$C.json
  nice -n 5 $N $G/score_polys.py data/real/mt34_htw_eval $G/out/e2e_hb_rdp.json $W/runs/ult/deploy/hball_t$C.json
  echo E2E_GATE_DONE;;
esac
