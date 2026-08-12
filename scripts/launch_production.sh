#!/bin/bash
# Production temporal foreground: two independent runs of the SAME recipe. RUNS ON PANDA.
#
# Two runs rather than one, on the two free L40S, for a reason this project has already paid
# for once: the per-epoch validation score scatters by about +-0.1, so a single run's selected
# checkpoint is one draw from a noisy process. Two independent runs of the identical recipe
# measure that scatter directly, and the gate picks between them on evidence rather than on
# whichever one happened to be trained.
#
# The recipe is the one that beat its matched control on fragmentation (protocol 19):
# p_single 0.35, so 65 % of batches carry (t-1, t, t+1) and the rest keep the single frame as
# the exact degenerate case. Longer than the 30-epoch pilot because the selected epochs there
# (28 temporal, 10 control) sat well inside the run, not at its end.
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
export PYTHONPATH=src:synth

run() {   # run <gpu> <tag>
  local gpu=$1 tag=$2
  mkdir -p "/home/prusek/mt_enc_exp/${tag}_ckpt"
  CUDA_VISIBLE_DEVICES=$gpu nohup "$PY" scripts/train_temporal.py \
      --epochs "$EPOCHS" --val-every 3 --epoch-len "$EPOCH_LEN" --workers "$WORKERS" \
      --p-single 0.35 \
      --ckpt-dir "/home/prusek/mt_enc_exp/${tag}_ckpt" \
      --out "/home/prusek/mt_enc_exp/dino_seg_ori_${tag}.pth" \
      --report "$L/train_${tag}.json" \
      > "/home/prusek/mt_enc_exp/train_${tag}.log" 2>&1 &
  echo "  gpu$gpu  $tag  (pid $!)"
}

echo "launching production runs on $(hostname): ${EPOCHS} epochs x ${EPOCH_LEN} samples"
run 0 prodA
sleep 5
run 1 prodB
sleep 40
echo
nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv,noheader
for t in prodA prodB; do
  echo "[$t] $(grep -m1 backgrounds "/home/prusek/mt_enc_exp/train_${t}.log" 2>/dev/null)"
done
