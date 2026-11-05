#!/bin/bash
# The declared HEAT decoder screen (docs/TODO.md 0.0-CAMPAIGN, HEAT decode declaration): the moment
# the fast-grid queue has dumped HEAT's 2-channel native maps, run decode_ridge.py's gated
# heat-ridge variants (G = dilate3(NMS ridge of the HEAT channel >= t & prob > tx), GU = the same
# united with the 0.98 skeleton) against B(0.98) / B(0.99) on the same maps, 8 CPU chunks, with
# the crossing metrics, then the paired report. Nothing here touches the GPU or TEST.
cd /disk2/prusek/mt_work/mt34_work
export PYTHONPATH=.:src:scripts:synth:third_party
PY=/scratch/prusek/envs/nnunet/bin/python
R=/disk2/prusek/mt_work/runs; P=/disk2/prusek/mt_work/preds; D=$P/HEAT_roi_ep40_native; O=runs/ridge_heat
mkdir -p $O; echo $$ > $R/heat_screen_waiter.pid
NEED=139
while :; do
  n=$(ls $D/*.npz 2>/dev/null | wc -l)
  recent=$(find $D -name '*.npz' -mmin -3 2>/dev/null | wc -l)
  if [ "$n" -ge $NEED ] && [ "$recent" = 0 ] && grep -q "HEAT_ep40_native dump rc=0" $R/eval_queue_fast.log; then break; fi
  sleep 300
done
echo "$(date) HEAT dump complete ($n maps); launching the G/GU screen"
VARS="B(0.98),B(0.99),G(0.3,0.3),G(0.5,0.3),G(0.7,0.3),G(0.3,0.5),G(0.5,0.5),G(0.7,0.5),GU(0.5,0.3),GU(0.5,0.5)"
$PY - <<PYEOF
import glob, os
st = sorted(os.path.basename(p)[:-4] for p in glob.glob("$D/*.npz"))
st = [s for s in st if s not in ("roi_img_8", "roi_img_149")]
K = 8
for k in range(K):
    open("$O/chunk_%d.txt" % k, "w").write(",".join(st[k::K]))
print(len(st), "frames in", K, "chunks")
PYEOF
for k in $(seq 0 7); do
  ( nice -n 10 $PY scripts/decode_ridge.py --pred-dir $D --channel 0 --variants "$VARS" --crossings \
      --frames "$(cat $O/chunk_$k.txt)" --out $O/chunk_$k.json > $O/chunk_$k.log 2>&1; echo "chunk $k rc=$?" ) &
  echo $! > $O/chunk_$k.pid
done
wait
$PY scripts/ridge_screen_report.py --inputs $O/chunk_*.json --baseline "B(0.98)" --out $O/report.json > $O/report.txt 2>&1
rm -f $R/heat_screen_waiter.pid
echo "$(date) HEAT_SCREEN_DONE"; cat $O/report.txt
