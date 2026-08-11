#!/bin/bash
# Full instancer tuning chain. RUNS ON TULEN (32 cores); all CPU work belongs on the server.
# Leaves 20 cores for nnU-Net's dataloaders, which are training on the same box.
set -u
cd /home/prusek/mt_enc_exp/mt34_work
PY=$HOME/dinov3_env/bin/python
L=data/enc_sensitivity_testset
PRED=/home/prusek/mt_enc_exp/mt34_pred_fovnorm     # FOV-normalised v4b channels
J=12
mkdir -p "$L"

echo "### ORACLE studies (same budget for every method -- W10 parity) ###"
for spec in ${ORACLE_SPECS:-"a:" "pysoax:" "b:" "a:--tune-kappa"}; do
  M="${spec%%:*}"; FLAG="${spec#*:}"
  TAG="$M${FLAG:+_kappatuned}"
  # Instancer B's tracing loop is pure Python, so Optuna's THREAD-based --n-jobs is
  # GIL-bound: 12 "parallel" trials measured 173% CPU across 75 threads, i.e. ~1.7 cores.
  # B therefore gets fewer trials at lower concurrency -- a comparable WALL-CLOCK budget
  # rather than a comparable trial count. Stated rather than hidden.
  if [ "$M" = "b" ]; then NT=60; NJ=4; else NT=120; NJ=$J; fi
  echo "=== oracle $TAG (n_trials=$NT n_jobs=$NJ) ==="
  $PY scripts/tune_instancer.py --method "$M" $FLAG --masks oracle \
      --n-trials $NT --n-jobs $NJ > "$L/tune_$TAG.log" 2>&1
  echo "--- $(grep 'BEST oracle' "$L/tune_$TAG.log")"
done

echo "### MODEL-MASK studies (prob_thr is in A's space here -- it dominates on noisy fg) ###"
for M in a b; do
  echo "=== model $M ==="
  if [ "$M" = "b" ]; then NT=50; NJ=4; else NT=100; NJ=$J; fi
  $PY scripts/tune_instancer.py --method "$M" --masks model --pred-dir "$PRED" \
      --n-trials $NT --n-jobs $NJ > "$L/tune_${M}_model.log" 2>&1
  echo "--- $(grep 'BEST oracle' "$L/tune_${M}_model.log")"
done

echo "### VAL reproduce check -- tuned params through the eval must match study.best_value ###"
$PY scripts/run_oracle_eval.py --split val --masks oracle --methods pysoax,a,b \
    --params-a src/instance/params_a.json --params-b src/instance/params_b.json \
    --out-json "$L/val_oracle_tuned.json" > "$L/val_oracle_tuned.log" 2>&1
$PY scripts/run_oracle_eval.py --split val --masks model --pred-dir "$PRED" --methods a,b \
    --params-a src/instance/params_a_model.json --params-b src/instance/params_b_model.json \
    --out-json "$L/val_model_tuned.json" > "$L/val_model_tuned.log" 2>&1

echo "CHAIN DONE"
