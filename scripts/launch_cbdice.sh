#!/bin/bash
# cbDice against the measured failure: two weights, one per GPU. RUNS ON KAJMAN.
#
# The control is NOT launched here -- prodA and prodB on panda already are it: identical recipe
# (45 epochs, 6000 samples, p_single 0.35), cbdice_w = 0. Spending kajman's two cards on two
# WEIGHTS instead of on a duplicate control is the better use of them, because the weight is
# the thing nobody knows: the reference trainer uses lambda_cbdice = 2.0 against
# lambda_dice = 1.0, while this project's own history has clDice collapsing to all-foreground
# at 0.5. Guessing one number and reporting the result would be indistinguishable from tuning
# on the test set afterwards.
#
# Success is NOT F1 first. The measured failure is connectivity -- 2043 mask components and
# 4302 endpoints against an oracle's 294 and 968 -- so those are what must move. F1 is checked
# after, and only means something if they did.
set -u
cd /home/prusek/mt_enc_exp/mt34_work

PY=${PY:-/home/prusek/mt_env_kajman/bin/python}
EPOCHS=${EPOCHS:-45}
EPOCH_LEN=${EPOCH_LEN:-6000}
WORKERS=${WORKERS:-32}
L=data/enc_sensitivity_testset

export SEG_MODE=ori SEG_BACKBONE=dinov2 SEG_INPUT=raw SEG_ARCH=base
export CALIB=/home/prusek/mt_enc_exp/calib_reg418_morph.json
export MASK_HW=1.0 POS_W=8 CLDICE_W=0.1
export PYTHONPATH=src:synth:scripts

run() {   # run <gpu> <cbdice_w> <tag>
  local gpu=$1 w=$2 tag=$3
  mkdir -p "/home/prusek/mt_enc_exp/${tag}_ckpt"
  CUDA_VISIBLE_DEVICES=$gpu nohup "$PY" scripts/train_temporal.py \
      --epochs "$EPOCHS" --val-every 3 --epoch-len "$EPOCH_LEN" --workers "$WORKERS" \
      --p-single 0.35 --cbdice-w "$w" \
      --ckpt-dir "/home/prusek/mt_enc_exp/${tag}_ckpt" \
      --out "/home/prusek/mt_enc_exp/dino_seg_ori_${tag}.pth" \
      --report "$L/train_${tag}.json" \
      > "/home/prusek/mt_enc_exp/train_${tag}.log" 2>&1 &
  echo "  gpu$gpu  cbdice_w=$w  -> train_${tag}.log  (pid $!)"
}

echo "launching on $(hostname): ${EPOCHS} epochs x ${EPOCH_LEN}"
echo "control = prodA/prodB on panda (same recipe, cbdice_w=0)"
run 0 0.5 cbA
sleep 5
run 1 2.0 cbB
sleep 45
echo
nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv,noheader
for t in cbA cbB; do
  echo "[$t] $(grep -m1 backgrounds "/home/prusek/mt_enc_exp/train_${t}.log" 2>/dev/null)"
  echo "     $(grep -m1 '^epoch' "/home/prusek/mt_enc_exp/train_${t}.log" 2>/dev/null)"
done
