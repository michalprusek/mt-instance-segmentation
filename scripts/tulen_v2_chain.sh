#!/bin/bash
# v2: re-tune instancer A with the three new levers, ablate each on VAL, then ONE TEST
# re-score. RUNS ON TULEN.
#
# TEST discipline for this batch: all development, tuning and ablation happen on VAL. TEST is
# scored exactly once at the end and reported next to v1 -- otherwise three iterations turn
# TEST into a development set.
set -u
cd /home/prusek/mt_enc_exp/mt34_work
PY=$HOME/dinov3_env/bin/python
L=data/enc_sensitivity_testset
LOCK=/tmp/mt34_v2.lock
PRED=/home/prusek/mt_enc_exp/mt34_pred          # whole-frame norm won the VAL selection
mkdir -p "$L"

if [ -e "$LOCK" ] && kill -0 "$(cat "$LOCK")" 2>/dev/null; then
  echo "another v2 run is live (pid $(cat "$LOCK")); refusing"; exit 1
fi
echo $$ > "$LOCK"
trap 'rm -f "$LOCK"' EXIT

echo "=== re-tune A on ORACLE VAL (new levers in the space) ==="
$PY scripts/tune_instancer.py --method a --masks oracle --n-trials 120 --n-jobs 12 \
    > "$L/tune_a_v2.log" 2>&1
cp src/instance/params_a.json src/instance/params_a_v2.json
echo "--- $(grep 'BEST oracle' "$L/tune_a_v2.log")"

echo "=== re-tune A on MODEL VAL ==="
$PY scripts/tune_instancer.py --method a --masks model --pred-dir "$PRED" \
    --n-trials 100 --n-jobs 12 > "$L/tune_a_model_v2.log" 2>&1
cp src/instance/params_a_model.json src/instance/params_a_model_v2.json
echo "--- $(grep 'BEST oracle' "$L/tune_a_model_v2.log")"

echo "=== LEVER ABLATION on VAL ==="
PYTHONPATH=src $PY scripts/ablate_levers.py --masks oracle \
    --params src/instance/params_a_v2.json --out "$L/ablation_oracle.json"
PYTHONPATH=src $PY scripts/ablate_levers.py --masks model --pred-dir "$PRED" \
    --params src/instance/params_a_model_v2.json --out "$L/ablation_model.json"

echo "=== TEST v2 (scored once) ==="
$PY scripts/run_oracle_eval.py --split test --masks oracle --methods a \
    --params-a src/instance/params_a_v2.json \
    --out-json "$L/TESTv2_oracle.json" > "$L/TESTv2_oracle.log" 2>&1
grep -E "task|POOLED|junction-id by" "$L/TESTv2_oracle.log"

$PY scripts/run_oracle_eval.py --split test --masks model --pred-dir "$PRED" --methods a \
    --params-a src/instance/params_a_model_v2.json \
    --out-json "$L/TESTv2_model.json" > "$L/TESTv2_model.log" 2>&1
grep -E "task|POOLED|polarity|junction-id by" "$L/TESTv2_model.log"

$PY scripts/run_oracle_eval.py --split test --data data/synth_eval --masks oracle \
    --methods a --params-a src/instance/params_a_v2.json \
    --out-json "$L/TESTv2_synth_oracle.json" > "$L/TESTv2_synth_oracle.log" 2>&1
grep -E "task|POOLED|junction-id by" "$L/TESTv2_synth_oracle.log"

$PY scripts/run_oracle_eval.py --split test --data data/synth_eval --masks model \
    --pred-dir /home/prusek/mt_enc_exp/synth_pred_v4b --methods a \
    --params-a src/instance/params_a_model_v2.json \
    --out-json "$L/TESTv2_synth_model.json" > "$L/TESTv2_synth_model.log" 2>&1
grep -E "task|POOLED" "$L/TESTv2_synth_model.log"

echo V2_DONE
