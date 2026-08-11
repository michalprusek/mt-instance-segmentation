#!/bin/bash
# End-to-end: the GATED foreground -> instancer A -> numbers, intervals, pictures. RUNS ON TULEN.
#
# Run this after scripts/train_gated.py finishes. Order is the point:
#
#   1. re-derive the selection from the run report (the training job loaded select_checkpoint
#      BEFORE the min_frames partial-collapse guard existed, so its own pick is not trusted);
#   2. predict MT-34 with the selected checkpoint;
#   3. compare on **VAL** -- gated vs v4b, both as foreground quality and as instance F1. All
#      development comparison happens here;
#   4. score **TEST once** and attach a paired interval against v4b on the same frames;
#   5. render the five-panel pictures.
#
# TEST discipline: the checkpoint was selected on VAL, so step 4 is a first and only TEST
# evaluation of a new model. It must not be iterated on -- if the result disappoints, the next
# move is a different training run, not a second TEST score.
set -u
cd /home/prusek/mt_enc_exp/mt34_work
PY=$HOME/dinov3_env/bin/python
L=data/enc_sensitivity_testset
V4B=/home/prusek/mt_enc_exp/mt34_pred
GATED_W=/home/prusek/mt_enc_exp/dino_seg_ori_gated.pth
GATED_PRED=/home/prusek/mt_enc_exp/mt34_pred_gated
LOCK=/tmp/mt34_e2e.lock

if [ -e "$LOCK" ] && kill -0 "$(cat "$LOCK")" 2>/dev/null; then
  echo "another e2e run is live (pid $(cat "$LOCK")); refusing"; exit 1
fi
echo $$ > "$LOCK"
trap 'rm -f "$LOCK"' EXIT

echo "=== 1. re-derive checkpoint selection (with the partial-collapse guard) ==="
SEG_MODE=ori PYTHONPATH=src $PY scripts/select_from_report.py \
    --report "$L/train_gated.json" --out "$GATED_W" || exit 2

echo
echo "=== 2. predict MT-34 with the selected checkpoint ==="
mkdir -p "$GATED_PRED"
SEG_MODE=ori SEG_BACKBONE=dinov2 SEG_INPUT=raw SEG_ARCH=base \
SEG_WEIGHTS="$GATED_W" MT34_PRED="$GATED_PRED" PYTHONPATH=src \
    $PY scripts/predict_v4b_mt34.py > "$L/predict_gated.log" 2>&1
tail -2 "$L/predict_gated.log"

echo
echo "=== 3. VAL comparison (development happens here) ==="
for tag in v4b gated; do
  [ "$tag" = v4b ] && PD=$V4B || PD=$GATED_PRED
  $PY scripts/run_oracle_eval.py --split val --masks model --pred-dir "$PD" --methods a \
      --params-a src/instance/params_a_model_v2.json \
      --out-json "$L/VAL_${tag}_e2e.json" > "$L/VAL_${tag}_e2e.log" 2>&1
  echo "--- $tag VAL: $(grep -m1 POOLED "$L/VAL_${tag}_e2e.log")"
done
$PY scripts/bootstrap_report.py "$L/VAL_gated_e2e.json" "$L/VAL_v4b_e2e.json" \
    --label-a gated --label-b v4b

echo
echo "=== 4. TEST -- scored ONCE, with a paired interval vs v4b ==="
$PY scripts/run_oracle_eval.py --split test --masks model --pred-dir "$GATED_PRED" --methods a \
    --params-a src/instance/params_a_model_v2.json \
    --out-json "$L/TEST_gated_e2e.json" > "$L/TEST_gated_e2e.log" 2>&1
grep -E "task|POOLED|polarity|junction-id by" "$L/TEST_gated_e2e.log"
echo
$PY scripts/bootstrap_report.py "$L/TEST_gated_e2e.json" "$L/TESTv2_model_ci.json" \
    --label-a gated --label-b v4b

echo
echo "=== 5. pictures: oracle | v4b | gated, on TEST frames ==="
PYTHONPATH=src $PY scripts/viz_test_set.py \
    --pred-dir "$V4B" --pred-dir-2 "$GATED_PRED" --label v4b --label-2 gated \
    --out-dir "$L/test_viz_gated"

echo E2E_DONE
