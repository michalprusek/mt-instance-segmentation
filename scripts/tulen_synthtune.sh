#!/bin/bash
# Can the instancer be tuned WITHOUT any real annotations? RUNS ON TULEN.
#
# The claim the project wants to make is "no real annotations anywhere". It is currently true
# of the semantic model -- trained purely on synthetic frames -- but NOT of the instancer:
# its 17 hyperparameters were fitted by Optuna against human polylines on the real MT-34 VAL
# split. That is a legitimate protocol, and it is also the weakest sentence in the paper.
#
# This tunes the same 17 knobs on SYNTHETIC data instead, where the ground truth is exact and
# free (the centerlines ARE the objects that were drawn), then scores MT-34 TEST once. If the
# synth-tuned parameters transfer, the whole pipeline becomes annotation-free end to end.
#
# Budget parity matters: the real-VAL params got 100 trials at n_jobs 12, so this gets the
# same. A win produced by a bigger search would prove nothing about the data source.
#
# TEST discipline: one scoring run, paired against the already-measured real-VAL-tuned result
# on the same 17 frames. Whatever it says is the answer -- no re-tuning against TEST.
set -u
cd /home/prusek/mt_enc_exp/mt34_work
PY=$HOME/dinov3_env/bin/python
L=data/enc_sensitivity_testset
SYNTH_PRED=/home/prusek/mt_enc_exp/synth_pred_v4b
MT34_PRED=/home/prusek/mt_enc_exp/mt34_pred
LOCK=/tmp/mt34_synthtune.lock

if [ -e "$LOCK" ] && kill -0 "$(cat "$LOCK")" 2>/dev/null; then
  echo "another synthtune run is live (pid $(cat "$LOCK")); refusing"; exit 1
fi
echo $$ > "$LOCK"
trap 'rm -f "$LOCK"' EXIT

echo "=== 1. tune A on SYNTHETIC VAL (exact GT, v4b foreground) -- 100 trials, n_jobs 12 ==="
$PY scripts/tune_instancer.py --method a --masks model \
    --data data/synth_eval --split val --pred-dir "$SYNTH_PRED" \
    --n-trials 100 --n-jobs 12 --tag synthtuned \
    > "$L/tune_a_synth.log" 2>&1
grep -E "BEST|wrote" "$L/tune_a_synth.log"

echo
echo "=== 2. sanity: the synth-tuned params on SYNTH TEST (in-domain) ==="
$PY scripts/run_oracle_eval.py --split test --data data/synth_eval --masks model \
    --pred-dir "$SYNTH_PRED" --methods a \
    --params-a src/instance/params_a_model_synthtuned.json \
    --out-json "$L/TEST_synth_synthtuned.json" > "$L/TEST_synth_synthtuned.log" 2>&1
grep -E "POOLED" "$L/TEST_synth_synthtuned.log"

echo
echo "=== 3. THE QUESTION: synth-tuned params on the REAL MT-34 VAL (development check) ==="
$PY scripts/run_oracle_eval.py --split val --masks model --pred-dir "$MT34_PRED" --methods a \
    --params-a src/instance/params_a_model_synthtuned.json \
    --out-json "$L/VAL_mt34_synthtuned.json" > "$L/VAL_mt34_synthtuned.log" 2>&1
grep -E "task|POOLED" "$L/VAL_mt34_synthtuned.log"
echo "  (real-VAL-tuned reference on the same split: see VAL_v4b_e2e.log)"
grep -m1 POOLED "$L/VAL_v4b_e2e.log"

echo
echo "=== 4. MT-34 TEST, scored once, paired against the real-VAL-tuned params ==="
$PY scripts/run_oracle_eval.py --split test --masks model --pred-dir "$MT34_PRED" --methods a \
    --params-a src/instance/params_a_model_synthtuned.json \
    --out-json "$L/TEST_mt34_synthtuned.json" > "$L/TEST_mt34_synthtuned.log" 2>&1
grep -E "task|POOLED|polarity|junction-id by" "$L/TEST_mt34_synthtuned.log"
echo
$PY scripts/bootstrap_report.py "$L/TEST_mt34_synthtuned.json" "$L/TESTv2_model_ci.json" \
    --label-a synth-tuned --label-b real-VAL-tuned

echo SYNTHTUNE_DONE
