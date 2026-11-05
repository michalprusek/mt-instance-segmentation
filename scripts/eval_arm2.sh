#!/bin/bash
# Score one trained arm as the campaign declares (eval_arm.sh), but in two phases: a fast GPU dump
# (one threshold, --dump-pred) and a PARALLEL CPU sweep of the cached maps (sweep_parallel.sh).
#
#   scripts/eval_arm2.sh NAME GPU [extra eval_v5 flags...]     e.g. HEAT 1 --prob-channel 0
cd /disk2/prusek/mt_work/mt34_work
export PYTHONPATH=.:src:scripts:synth:third_party
PY=/scratch/prusek/envs/nnunet/bin/python
R=/disk2/prusek/mt_work/runs; P=/disk2/prusek/mt_work/preds; S=$R/scale
NAME=$1; GPU=$2; shift 2
EXTRA=("$@")
[ -f data/real/roi303_eval/_repair_log.json ] || { echo "eval_arm2: roi303 block not repaired -- refusing"; exit 4; }
echo $$ > $S/eval_$NAME.pid
ROI="--data data/real/roi303_eval --split val --include-empty --no-fov --params src/instance/params_a_model_v5H.json"
HGRID=0.10,0.30,0.50,0.70,0.80,0.90,0.95,0.97,0.98,0.99,0.995,0.997,0.999,0.9995
ev () {  # ckpt scale tag
  [ -f "$1" ] || { echo "$(date +%H:%M) $NAME$3: checkpoint $1 missing -- skipped"; return 1; }
  if [ ! -d $P/${NAME}_roi$3 ] || [ $(ls $P/${NAME}_roi$3/*.npz 2>/dev/null | wc -l) -lt 100 ]; then
    CUDA_VISIBLE_DEVICES=$GPU $PY scripts/eval_v5.py --ckpt $1 --arch nnunet $ROI --thrs 0.98 --infer-scale $2 "${EXTRA[@]}" \
      --dump-pred $P/${NAME}_roi$3 --out-json $S/dump_$NAME$3.json > $S/dump_$NAME$3.log 2>&1
    echo "$(date +%H:%M) $NAME$3 dump rc=$?"
  else
    echo "$(date +%H:%M) $NAME$3 dump present, reusing"
  fi
  bash scripts/sweep_parallel.sh $P/${NAME}_roi$3 $S/thr_$NAME$3.json "${EXTRA[@]}" > $S/sweep_$NAME$3.log 2>&1
  echo "$(date +%H:%M) $NAME$3 sweep rc=$?  $(tail -1 $S/sweep_$NAME$3.log)"
}
ev $R/$NAME/ep040.pth 1.0 _ep40_native
ev $R/$NAME/ep040.pth 1.5 _ep40
ev $R/$NAME/model.pth 1.0 _native
ev $R/$NAME/model.pth 1.5 ""
HCK=$R/$NAME/model.pth; [ -f $HCK ] || HCK=$R/$NAME/ep040.pth
CUDA_VISIBLE_DEVICES=$GPU nice -n 10 $PY scripts/eval_v5.py --ckpt $HCK --arch nnunet "${EXTRA[@]}" \
  --data data/real/mt34_htw_eval --split val --include-empty --no-fov --params src/instance/params_a_model_v5H.json \
  --thrs $HGRID --infer-scale 1.0 --dump-pred $P/${NAME}_htw_native --out-json $S/thtw_${NAME}_native.json > $S/dump_${NAME}_htw_native.log 2>&1
echo "$(date +%H:%M) ${NAME}_htw_native rc=$?"
rm -f $S/eval_$NAME.pid
echo "EVAL_DONE $NAME"
