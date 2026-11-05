#!/bin/bash
# Crossing diagnostic (scripts/diag_crossing_map.py) for one arm's cached maps, 8 CPU chunks,
# merged into one summary -- the "weak-arm fraction" mechanism check of docs/TODO.md 0.0-CAMPAIGN.
#
#   scripts/xmap_arm.sh NAME THR [CHANNEL]      e.g. scripts/xmap_arm.sh AUG_PHOTO 0.98
#
# Reads preds/NAME_roi_native (the 1.5x eval-frame maps eval_v5.py dumps at native inference),
# writes runs/diag_roi/xmap_NAME/out_chunk_k/crossing_diag.json and .../summary.json.
# Liveness: runs/diag_roi/xmap_NAME/chunk_k.pid + kill -0, never pgrep -f.
cd /disk2/prusek/mt_work/mt34_work
export PYTHONPATH=.:src:scripts:synth:third_party
PY=/scratch/prusek/envs/nnunet/bin/python
NAME=$1; THR=$2; CH=${3:-0}
P=/disk2/prusek/mt_work/preds/${NAME}_roi_native
O=runs/diag_roi/xmap_$NAME
[ -d $P ] || { echo "no maps at $P"; exit 2; }
mkdir -p $O
# VAL stems with >= 2 GT, split into 8 chunks
$PY - <<EOF
import glob, os, sys
sys.path.insert(0, "src")
from mt_bench.cvat_import import read_frame_h5
stems = []
for p in sorted(glob.glob("data/real/roi303_eval/*.h5")):
    fr = read_frame_h5(p)
    if str(fr["attrs"].get("split")) == "val" and len(fr["polylines"]) >= 2 \
            and os.path.exists("$P/" + os.path.basename(p)[:-3] + ".npz"):
        stems.append(os.path.basename(p)[:-3])
for k in range(8):
    open("$O/chunk_%02d.txt" % k, "w").write(",".join(stems[k::8]))
print(len(stems), "frames")
EOF
for k in 0 1 2 3 4 5 6 7; do
  kk=$(printf %02d $k)
  ( nice -n 10 $PY scripts/diag_crossing_map.py --pred-dir $P --thr $THR --channel $CH \
      --frames "$(cat $O/chunk_$kk.txt)" --out $O/out_chunk_$kk --montage 0 > $O/chunk_$kk.log 2>&1;
    rm -f $O/chunk_$kk.pid ) &
  echo $! > $O/chunk_$kk.pid
done
wait
$PY - <<EOF
import json, glob, numpy as np
rows = []
for f in sorted(glob.glob("$O/out_chunk_*/crossing_diag.json")):
    rows += json.load(open(f))["rows"]
fa = [r for r in rows if not r["identity"]]
weak = lambda r: min(r["f0"]["p_far_mean"], r["f1"]["p_far_mean"]) < 0.98
s = {"arm": "$NAME", "thr": $THR, "n": len(rows), "frames": len({r["frame"] for r in rows}),
     "identity": float(np.mean([r["identity"] for r in rows])),
     "oracle_identity": float(np.mean([r["oracle_identity"] for r in rows])),
     "mask_connected": float(np.mean([r["connected"] for r in rows])),
     "weak_arm_fraction_all": float(np.mean([weak(r) for r in rows])),
     "n_fail": len(fa),
     "fail_mask_disconnected": float(np.mean([not r["connected"] for r in fa])) if fa else None,
     "fail_weak_arm": float(np.mean([weak(r) for r in fa])) if fa else None,
     "fail_arm_missing": float(np.mean([not (r["f0"]["found"] and r["f1"]["found"]) for r in fa])) if fa else None,
     "identity_same_polarity": float(np.mean([r["identity"] for r in rows if r["same_polarity"]])),
     "identity_opposite_polarity": float(np.mean([r["identity"] for r in rows if not r["same_polarity"]])) if any(not r["same_polarity"] for r in rows) else None}
json.dump({"summary": s, "rows": rows}, open("$O/summary.json", "w"))
print(json.dumps(s, indent=1))
EOF
echo "XMAP_DONE $NAME"
