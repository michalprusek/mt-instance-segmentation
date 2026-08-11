#!/bin/bash
# The single TEST shot. RUNS ON TULEN, after every hyperparameter is frozen.
#
# Order matters: the VAL selection between the two prediction sets happens FIRST and on VAL,
# so nothing about the TEST split influences a choice. TEST is then scored exactly once.
set -eu
cd /home/prusek/mt_enc_exp/mt34_work
PY=$HOME/dinov3_env/bin/python
L=data/enc_sensitivity_testset
P_FOV=/home/prusek/mt_enc_exp/mt34_pred_fovnorm
P_STD=/home/prusek/mt_enc_exp/mt34_pred
P_NN=/home/prusek/mt_enc_exp/mt34_pred_nnunet_s15

echo "### VAL: choose the prediction set (whole-frame vs FOV-based input normalisation) ###"
for tag in std fov; do
  [ "$tag" = std ] && PD=$P_STD || PD=$P_FOV
  $PY scripts/run_oracle_eval.py --split val --masks model --pred-dir "$PD" --methods a,b \
      --params-a src/instance/params_a_model.json \
      --params-b src/instance/params_b_model.json \
      --out-json "$L/val_model_$tag.json" > "$L/val_model_$tag.log" 2>&1
  echo "--- $tag: $(grep -m1 POOLED "$L/val_model_$tag.log")"
done

echo
echo "### TEST (scored once) -- oracle foreground ###"
$PY scripts/run_oracle_eval.py --split test --masks oracle --methods pysoax,a,b \
    --params-a src/instance/params_a.json \
    --params-b src/instance/params_b.json \
    --params-pysoax src/instance/params_pysoax.json \
    --out-json "$L/TEST_oracle.json" > "$L/TEST_oracle.log" 2>&1
grep -E "^---|task|POOLED|junction-id by" "$L/TEST_oracle.log"

echo
echo "### TEST (scored once) -- v4b predicted foreground, winning normalisation ###"
# Pick the input normalisation on VAL, never on TEST.
F_STD=$($PY - <<PYEOF
import json
d = json.load(open("$L/val_model_std.json"))
print(d.get("a/pooled", {}).get("mean_f1", 0.0))
PYEOF
)
F_FOV=$($PY - <<PYEOF
import json
d = json.load(open("$L/val_model_fov.json"))
print(d.get("a/pooled", {}).get("mean_f1", 0.0))
PYEOF
)
if $PY -c "import sys; sys.exit(0 if float('$F_FOV') > float('$F_STD') else 1)"; then
  WIN=$P_FOV; echo "VAL picks FOV-based normalisation ($F_FOV > $F_STD)"
else
  WIN=$P_STD; echo "VAL picks whole-frame normalisation ($F_STD >= $F_FOV)"
fi
$PY scripts/run_oracle_eval.py --split test --masks model --pred-dir "$WIN" --methods a,b \
    --params-a src/instance/params_a_model.json \
    --params-b src/instance/params_b_model.json \
    --out-json "$L/TEST_model.json" > "$L/TEST_model.log" 2>&1
grep -E "^---|task|POOLED|polarity|junction-id by" "$L/TEST_model.log"

echo
echo "### TEST -- instancer A on the nnU-Net foreground (A only: nnU-Net has no amodal channels) ###"
$PY scripts/run_oracle_eval.py --split test --masks model --pred-dir "$P_NN" --methods a \
    --params-a src/instance/params_a_model.json \
    --out-json "$L/TEST_nnunet.json" > "$L/TEST_nnunet.log" 2>&1
grep -E "^---|task|POOLED" "$L/TEST_nnunet.log"

echo
echo "### TEST -- SYNTHETIC, exact ground truth (in-domain, isolates instancer quality) ###"
# MT-34's GT is human-corrected v7+PySOAX output, so it carries an agreement bias and is
# demonstrably incomplete on sparse frames. Synthetic GT has neither problem: the centerlines
# ARE the objects that were drawn. Together the two sets separate instancer error from
# annotation error + domain gap.
$PY scripts/run_oracle_eval.py --split test --data data/synth_eval --masks oracle \
    --methods pysoax,a,b \
    --params-a src/instance/params_a.json \
    --params-b src/instance/params_b.json \
    --params-pysoax src/instance/params_pysoax.json \
    --out-json "$L/TEST_synth_oracle.json" > "$L/TEST_synth_oracle.log" 2>&1
grep -E "^---|task|POOLED|junction-id by" "$L/TEST_synth_oracle.log"

echo
echo "### TEST -- SYNTHETIC, v4b predicted foreground (in-domain: no domain gap) ###"
$PY scripts/run_oracle_eval.py --split test --data data/synth_eval --masks model \
    --pred-dir /home/prusek/mt_enc_exp/synth_pred_v4b --methods a,b \
    --params-a src/instance/params_a_model.json \
    --params-b src/instance/params_b_model.json \
    --out-json "$L/TEST_synth_model.json" > "$L/TEST_synth_model.log" 2>&1
grep -E "^---|task|POOLED|junction-id by" "$L/TEST_synth_model.log"

echo "TEST DONE"
