#!/bin/bash
# Threshold sweep of cached probability maps, one eval_v5.py process per threshold, 8 at a time,
# merged into the SAME JSON layout eval_v5.py writes (results[thr] with _frames rows), so every
# reader (arm_report.py, factorial_report.py, ridge_screen_report.py) keeps working.
#
#   scripts/sweep_parallel.sh PRED_DIR OUT_JSON [extra eval_v5 flags...]
#
# Why: the 19-point grid's low thresholds (0.30-0.80) take ~30 min EACH single-threaded on the
# 139-frame VAL (large masks, many components); sequential sweeps were ~6 h per arm x scale.
cd /disk2/prusek/mt_work/mt34_work
export PYTHONPATH=.:src:scripts:synth:third_party
PY=/scratch/prusek/envs/nnunet/bin/python
PRED=$1; OUT=$2; shift 2
EXTRA=("$@")
GRID=${GRID:-0.30,0.40,0.50,0.60,0.70,0.80,0.90,0.94,0.96,0.97,0.98,0.985,0.99,0.993,0.995,0.997,0.998,0.999,0.9995}
ROI="--data data/real/roi303_eval --split val --include-empty --no-fov --params src/instance/params_a_model_v5H.json"
TMP=$OUT.parts; mkdir -p $TMP
# HIGH thresholds first: they are fast (small masks) and are where every arm's optimum lies, so a
# preliminary read (merge_sweep_parts.py on the parts present) is possible long before the slow
# 0.30-0.60 thresholds (~80 min each) finish; the final merge is identical either way.
echo "$GRID" | tr ',' '\n' | sort -rn | nice -n 10 xargs -P ${NPAR:-12} -I{} sh -c \
  "[ -f $TMP/thr_{}.json ] || $PY scripts/eval_v5.py --pred-dir $PRED $ROI --thrs {} ${EXTRA[*]} --out-json $TMP/thr_{}.json > $TMP/log_{}.txt 2>&1"
$PY scripts/merge_sweep_parts.py "$OUT" "$TMP"
