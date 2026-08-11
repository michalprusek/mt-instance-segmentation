#!/bin/bash
# Everything that remains, in one process. RUNS ON TULEN.
#
# Consolidated deliberately: three separately-launched drivers ended up with two stale Optuna
# studies writing into the SAME log as the live one, which silently mixes results from
# different search configurations. A single driver with a PID lock cannot do that.
set -u
cd /home/prusek/mt_enc_exp/mt34_work
PY=$HOME/dinov3_env/bin/python
L=data/enc_sensitivity_testset
LOCK=/tmp/mt34_finish.lock
mkdir -p "$L"

if [ -e "$LOCK" ] && kill -0 "$(cat "$LOCK")" 2>/dev/null; then
  echo "another run is live (pid $(cat "$LOCK")); refusing to start"; exit 1
fi
echo $$ > "$LOCK"
trap 'rm -f "$LOCK"' EXIT

run_study () {   # method extra_flag masks n_trials n_jobs pred_dir tag
  local M=$1 FLAG=$2 MASKS=$3 NT=$4 NJ=$5 PD=$6 TAG=$7
  echo "=== $TAG (n_trials=$NT n_jobs=$NJ) ==="
  local args=(--method "$M" --masks "$MASKS" --n-trials "$NT" --n-jobs "$NJ")
  [ -n "$FLAG" ] && args+=("$FLAG")
  [ -n "$PD" ] && args+=(--pred-dir "$PD")
  $PY scripts/tune_instancer.py "${args[@]}" > "$L/tune_$TAG.log" 2>&1
  echo "--- $(grep 'BEST oracle' "$L/tune_$TAG.log" || echo "FAILED, see $L/tune_$TAG.log")"
}

P_FOV=/home/prusek/mt_enc_exp/mt34_pred_fovnorm

# Instancer B's tracing loop is pure Python, so Optuna's THREAD-based --n-jobs is GIL-bound
# (12 "parallel" trials measured 173% CPU across 75 threads). B therefore gets fewer trials at
# lower concurrency -- a comparable WALL-CLOCK budget, not a comparable trial count.
run_study b  ""             oracle 60  4  ""      b
run_study a  --tune-kappa   oracle 120 12 ""      a_kappatuned
run_study a  ""             model  100 12 "$P_FOV" a_model
run_study b  ""             model  50  4  "$P_FOV" b_model

echo "### VAL reproduce -- tuned params through the eval must match study.best_value ###"
$PY scripts/run_oracle_eval.py --split val --masks oracle --methods pysoax,a,b \
    --params-a src/instance/params_a.json --params-b src/instance/params_b.json \
    --params-pysoax src/instance/params_pysoax.json \
    --out-json "$L/val_oracle_tuned.json" > "$L/val_oracle_tuned.log" 2>&1
grep -E "task|POOLED" "$L/val_oracle_tuned.log"

echo "### FINAL TEST ###"
bash scripts/final_test_run.sh > final_test.log 2>&1
tail -60 final_test.log
echo ALL_DONE
