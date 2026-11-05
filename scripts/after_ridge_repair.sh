#!/bin/bash
# When the ridge-decoder screen (which reads the roi303 h5 files) has finished, apply the declared
# repair to the tulen block and re-sweep the SPARSE35 baseline on the repaired VAL from its cached
# maps (both scales, the trainer's pick) so the campaign has baseline rows on the same frame set.
cd /disk2/prusek/mt_work/mt34_work
export PYTHONPATH=.:src:scripts:synth:third_party
PY=/scratch/prusek/envs/nnunet/bin/python
R=/disk2/prusek/mt_work/runs; P=/disk2/prusek/mt_work/preds; S=$R/scale
echo $$ > $R/after_ridge_repair.pid
while :; do
  alive=0
  for p in runs/ridge/chunk_*.pid; do [ -f "$p" ] && kill -0 "$(cat "$p")" 2>/dev/null && alive=1; done
  [ $alive = 0 ] && break
  sleep 120
done
echo "$(date) ridge screen finished; repairing the block"
$PY scripts/repair_roi_dataset.py --data data/real/roi303_eval > $R/repair_tulen.log 2>&1
echo "repair rc=$?"; tail -3 $R/repair_tulen.log
GRID=0.30,0.40,0.50,0.60,0.70,0.80,0.90,0.94,0.96,0.97,0.98,0.985,0.99,0.993,0.995,0.997,0.998,0.999,0.9995
ROI="--data data/real/roi303_eval --split val --include-empty --no-fov --params src/instance/params_a_model_v5H.json --thrs $GRID"
( nice -n 10 $PY scripts/eval_v5.py --pred-dir $P/SPARSE35_roi_native $ROI --out-json $S/thr_SPARSE35_native_rep.json > $S/sweep_SPARSE35_native_rep.log 2>&1; echo "native rep rc=$?" ) &
( nice -n 10 $PY scripts/eval_v5.py --pred-dir $P/SPARSE35_roi $ROI --out-json $S/thr_SPARSE35_rep.json > $S/sweep_SPARSE35_rep.log 2>&1; echo "1.5x rep rc=$?" ) &
wait
rm -f $R/after_ridge_repair.pid
echo "$(date) REPAIR_AND_BASELINE_DONE"
