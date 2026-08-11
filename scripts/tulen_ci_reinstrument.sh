#!/bin/bash
# Attach confidence intervals to the numbers that are ALREADY published. RUNS ON TULEN.
#
# This is RE-INSTRUMENTATION, not a second TEST shot. Every invocation below uses the same
# frozen params files, the same prediction directories and the same splits as the runs that
# produced the reported numbers; the ONLY difference is that run_oracle_eval.py now writes its
# per-frame rows under `_frames` and prints paired bootstrap intervals. The gate is therefore:
#
#   if any point estimate here differs from the published one, STOP -- something changed, and
#   the run is a second TEST shot rather than an instrumentation pass.
#
# Published values to check against (docs/protocol.md 17k-l):
#   v1 oracle pooled 0.893 | v2 oracle pooled 0.920 | v1 model 0.379 | v2 model 0.416
#   v1 synth oracle 0.695  | v2 synth oracle 0.710  | v1 synth model 0.180 | v2 synth model 0.183
set -u
cd /home/prusek/mt_enc_exp/mt34_work
PY=$HOME/dinov3_env/bin/python
L=data/enc_sensitivity_testset
PRED=/home/prusek/mt_enc_exp/mt34_pred
LOCK=/tmp/mt34_ci.lock

if [ -e "$LOCK" ] && kill -0 "$(cat "$LOCK")" 2>/dev/null; then
  echo "another CI run is live (pid $(cat "$LOCK")); refusing"; exit 1
fi
echo $$ > "$LOCK"
trap 'rm -f "$LOCK"' EXIT

echo "=== v1 params, oracle TEST (all three methods -> A vs B vs PySOAX with intervals) ==="
$PY scripts/run_oracle_eval.py --split test --masks oracle --methods pysoax,a,b \
    --params-a src/instance/params_a.json \
    --params-b src/instance/params_b.json \
    --params-pysoax src/instance/params_pysoax.json \
    --out-json "$L/TESTv1_oracle_ci.json" > "$L/TESTv1_oracle_ci.log" 2>&1
grep -E "POOLED|paired bootstrap| - " "$L/TESTv1_oracle_ci.log"

echo "=== v2 params, oracle TEST ==="
$PY scripts/run_oracle_eval.py --split test --masks oracle --methods a \
    --params-a src/instance/params_a_v2.json \
    --out-json "$L/TESTv2_oracle_ci.json" > "$L/TESTv2_oracle_ci.log" 2>&1
grep -E "POOLED" "$L/TESTv2_oracle_ci.log"

echo "=== v1 params, model TEST ==="
$PY scripts/run_oracle_eval.py --split test --masks model --pred-dir "$PRED" --methods a,b \
    --params-a src/instance/params_a_model.json \
    --params-b src/instance/params_b_model.json \
    --out-json "$L/TESTv1_model_ci.json" > "$L/TESTv1_model_ci.log" 2>&1
grep -E "POOLED|paired bootstrap| - " "$L/TESTv1_model_ci.log"

echo "=== v2 params, model TEST ==="
$PY scripts/run_oracle_eval.py --split test --masks model --pred-dir "$PRED" --methods a \
    --params-a src/instance/params_a_model_v2.json \
    --out-json "$L/TESTv2_model_ci.json" > "$L/TESTv2_model_ci.log" 2>&1
grep -E "POOLED" "$L/TESTv2_model_ci.log"

echo
echo "=== v2 - v1, paired ACROSS reports (the 0.893 -> 0.920 claim) ==="
$PY scripts/bootstrap_report.py "$L/TESTv2_oracle_ci.json" "$L/TESTv1_oracle_ci.json" \
    --label-a v2 --label-b v1
echo
$PY scripts/bootstrap_report.py "$L/TESTv2_model_ci.json" "$L/TESTv1_model_ci.json" \
    --label-a v2 --label-b v1

echo CI_DONE
