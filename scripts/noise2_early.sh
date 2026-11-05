#!/bin/bash
# 2026-09-06 02:55: NOISE2's native read pulled forward onto GPU 1 (the stage-3 queue is still on
# PW2S's remaining rows, ~06:30 before it reaches NOISE2). Same dump command as eval_arm2.sh's ev(),
# same dump dir (stage 3 will find it present and reuse it), fast-grid sweep into the same parts dir
# (reused too), then the pre-registered reads: anatomy + covered_split at 0.99 for the 16-bit
# falsifier. Nothing here touches TEST.
cd /disk2/prusek/mt_work/mt34_work
export PYTHONPATH=.:src:scripts:synth:third_party
PY=/scratch/prusek/envs/nnunet/bin/python
R=/disk2/prusek/mt_work/runs; P=/disk2/prusek/mt_work/preds; S=$R/scale; NAME=NOISE2
echo $$ > $R/noise2_early.pid
ROI="--data data/real/roi303_eval --split val --include-empty --no-fov --params src/instance/params_a_model_v5H.json"
D=$P/${NAME}_roi_ep40_native
if [ ! -d $D ] || [ $(ls $D/*.npz 2>/dev/null | wc -l) -lt 100 ]; then
  CUDA_VISIBLE_DEVICES=1 $PY scripts/eval_v5.py --ckpt $R/$NAME/ep040.pth --arch nnunet $ROI --thrs 0.98 --infer-scale 1.0 \
    --dump-pred $D --out-json $S/dump_${NAME}_ep40_native.json > $S/dump_${NAME}_ep40_native.log 2>&1
  echo "$(date +%H:%M) ${NAME}_ep40_native dump rc=$?"
fi
FAST=0.90,0.94,0.96,0.97,0.98,0.985,0.99,0.993,0.995,0.997,0.998,0.999,0.9995
GRID=$FAST NPAR=8 bash scripts/sweep_parallel.sh $D $S/thr_${NAME}_ep40_native.json > $S/sweep_${NAME}_ep40_native.log 2>&1
echo "$(date +%H:%M) ${NAME}_ep40_native sweep rc=$?  $(tail -1 $S/sweep_${NAME}_ep40_native.log)"
A=runs/diag_roi/anat; C=runs/diag_roi/csplit
for k in 0 1 2 3; do
  nice -n 15 $PY scripts/diag_fp_anatomy.py --pred-dir $D --thr 0.99 --exclude roi_img_8,roi_img_149 --frames "$(cat runs/diag_roi/fpnat/chunk_$k.txt)" --out $A/${NAME}_roi_ep40_native_$k.json > $A/${NAME}_roi_ep40_native_$k.log 2>&1 &
  nice -n 15 $PY scripts/diag_covered_split.py --pred-dir $D --thr 0.99 --frames "$(cat runs/diag_roi/fpnat/chunk_$k.txt)" --out $C/${NAME}_roi_ep40_native_$k.json > $C/${NAME}_roi_ep40_native_$k.log 2>&1 &
done
wait
rm -f $R/noise2_early.pid
echo "$(date +%H:%M) NOISE2_EARLY_DONE"
