#!/bin/bash
# Decide whether the temporal model ships. RUNS ON KAJMAN (or tulen, if its GPUs are free).
#
# Order matters and is not negotiable:
#
#   1. THE GATE first -- single-frame quality on MT-34 TEST against the deployed 0.457. If the
#      temporal model is worse on one frame, it does not ship whatever it does on video, and
#      the rest of this chain is diagnosis rather than a decision.
#   2. Sequences, all three foregrounds through the SAME evaluator: v4b (deployed), control
#      (same recipe, zero temporal information) and temporal.
#
# The control is what makes the comparison mean anything. Both new models were trained on a
# generator that did not exist yesterday, so temporal-vs-v4b alone cannot separate "temporal
# context helps" from "the new generator helps". Temporal-vs-control can.
set -u
cd /home/prusek/mt_enc_exp/mt34_work
PY=${PY:-/home/prusek/mt_env_kajman/bin/python}
L=data/enc_sensitivity_testset
E=/home/prusek/mt_enc_exp
LOCK=/tmp/mt34_temporal_eval.lock

if [ -e "$LOCK" ] && kill -0 "$(cat "$LOCK")" 2>/dev/null; then
  echo "another eval is live (pid $(cat "$LOCK")); refusing"; exit 1
fi
echo $$ > "$LOCK"; trap 'rm -f "$LOCK"' EXIT

export SEG_MODE=ori SEG_BACKBONE=dinov2 SEG_INPUT=raw SEG_ARCH=base PYTHONPATH=src:synth

echo "=== 1. THE GATE: single-frame quality on MT-34 ==="
for tag in control temporal; do
  W=$E/dino_seg_ori_${tag}.pth
  [ -f "$W" ] || { echo "  $tag: no weights at $W -- skipping"; continue; }
  SEG_WEIGHTS=$W $PY scripts/predict_seq_temporal.py --data data/real/mt34_eval \
      --out $E/mt34_pred_${tag} --mode single --weights "$W" \
      > "$L/predict_mt34_${tag}.log" 2>&1
  echo "  $tag: $(tail -1 "$L/predict_mt34_${tag}.log")"
  $PY scripts/run_oracle_eval.py --split test --masks model --pred-dir $E/mt34_pred_${tag} \
      --methods a --params-a src/instance/params_a_model_synthtuned.json \
      --out-json "$L/TEST_mt34_${tag}.json" > "$L/TEST_mt34_${tag}.log" 2>&1
  echo "  $tag MT-34 TEST: $(grep -m1 POOLED "$L/TEST_mt34_${tag}.log")"
done
echo "  deployed v4b reference: 0.457 [0.379, 0.533]"
echo
echo "  paired against the deployed model on the same frames:"
for tag in control temporal; do
  [ -f "$L/TEST_mt34_${tag}.json" ] || continue
  $PY scripts/bootstrap_report.py "$L/TEST_mt34_${tag}.json" "$L/TEST_mt34_synthtuned.json" \
      --label-a "$tag" --label-b v4b 2>/dev/null | grep -E "^  a:|^  a " || true
done

echo
echo "=== 2. sequences: v4b vs control vs temporal ==="
for tag in control temporal; do
  W=$E/dino_seg_ori_${tag}.pth
  [ -f "$W" ] || continue
  mode=temporal; [ "$tag" = control ] && mode=single
  SEG_WEIGHTS=$W $PY scripts/predict_seq_temporal.py --data data/synth_seq \
      --out $E/synth_seq_pred_${tag} --mode $mode --weights "$W" \
      > "$L/predict_seq_${tag}.log" 2>&1
  echo "  $tag ($mode): $(tail -1 "$L/predict_seq_${tag}.log")"
done

echo
echo "--- tracking, TEST split ---"
printf '%-10s ' "v4b"; $PY scripts/eval_tracking.py --data data/synth_seq --split test \
    --masks model --pred-dir $E/synth_seq_pred \
    --params src/instance/params_a_model_synthtuned.json 2>/dev/null | grep "^model" || true
for tag in control temporal; do
  [ -d "$E/synth_seq_pred_${tag}" ] || continue
  printf '%-10s ' "$tag"
  $PY scripts/eval_tracking.py --data data/synth_seq --split test --masks model \
      --pred-dir $E/synth_seq_pred_${tag} \
      --params src/instance/params_a_model_synthtuned.json \
      --out-json "$L/tracking_${tag}.json" 2>/dev/null | grep "^model" || true
done

echo
echo "Read it this way: tracks/object must fall towards the oracle 1.59 for TEMPORAL and stay"
echo "put for CONTROL. If both move, it is the generator and not the temporal context."
echo TEMPORAL_EVAL_DONE
