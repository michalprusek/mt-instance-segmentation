#!/bin/bash
# Score one trained arm exactly as the 2026-09-03 factorial was scored (run_factorial_eval2.sh):
#   roi303 VAL 141, ep040 AND the trainer's pick, at 1.5x AND native, 19-point grid, v5H params;
#   MT-34+HTW VAL, trainer's pick, native, the thtw grid + 0.9995.
# Extra arguments are passed to eval_v5.py (e.g. --prob-channel 0 for a --head heat arm).
#
#   scripts/eval_arm.sh NAME GPU [extra eval_v5 flags...]
#
# Refuses to start until the roi303 block carries the 2026-09-04 repair log, so every arm scored
# through here is scored on the repaired block.
cd /disk2/prusek/mt_work/mt34_work
export PYTHONPATH=.:src:scripts:synth:third_party
PY=/scratch/prusek/envs/nnunet/bin/python
R=/disk2/prusek/mt_work/runs; P=/disk2/prusek/mt_work/preds; S=$R/scale
NAME=$1; GPU=$2; shift 2
EXTRA=("$@")          # captured ONCE: inside a bash function "$@" would be the function's own args
[ -f data/real/roi303_eval/_repair_log.json ] || { echo "eval_arm: roi303 block not repaired yet -- refusing"; exit 4; }
echo $$ > $S/eval_$NAME.pid
GRID=0.30,0.40,0.50,0.60,0.70,0.80,0.90,0.94,0.96,0.97,0.98,0.985,0.99,0.993,0.995,0.997,0.998,0.999,0.9995
HGRID=0.10,0.30,0.50,0.70,0.80,0.90,0.95,0.97,0.98,0.99,0.995,0.997,0.999,0.9995
ROI="--data data/real/roi303_eval --split val --include-empty --no-fov --params src/instance/params_a_model_v5H.json --thrs $GRID"
ev () {  # ckpt scale tag
  [ -f "$1" ] || { echo "$(date +%H:%M) $NAME$3: checkpoint $1 missing -- skipped"; return 1; }
  CUDA_VISIBLE_DEVICES=$GPU $PY scripts/eval_v5.py --ckpt $1 --arch nnunet $ROI --infer-scale $2 "${EXTRA[@]}" \
    --dump-pred $P/${NAME}_roi$3 --out-json $S/thr_$NAME$3.json > $S/dump_$NAME$3.log 2>&1
  echo "$(date +%H:%M) $NAME$3 rc=$?"
}
ev $R/$NAME/ep040.pth 1.0 _ep40_native
ev $R/$NAME/ep040.pth 1.5 _ep40
ev $R/$NAME/model.pth 1.0 _native
ev $R/$NAME/model.pth 1.5 ""
# the secondary set: the trainer's pick if it exists (gate passed), else ep040
HCK=$R/$NAME/model.pth; [ -f $HCK ] || HCK=$R/$NAME/ep040.pth
CUDA_VISIBLE_DEVICES=$GPU $PY scripts/eval_v5.py --ckpt $HCK --arch nnunet "${EXTRA[@]}" \
  --data data/real/mt34_htw_eval --split val --include-empty --no-fov --params src/instance/params_a_model_v5H.json \
  --thrs $HGRID --infer-scale 1.0 --dump-pred $P/${NAME}_htw_native --out-json $S/thtw_${NAME}_native.json > $S/dump_${NAME}_htw_native.log 2>&1
echo "$(date +%H:%M) ${NAME}_htw_native rc=$?"
rm -f $S/eval_$NAME.pid
echo "EVAL_DONE $NAME"
