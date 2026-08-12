#!/bin/bash
# Launch the temporal foreground training AND its single-frame control. RUNS ON KAJMAN / PANDA.
#
# Two runs, one per GPU, identical in every respect except the one thing under test:
#
#   GPU 0  --p-single 0.35   65 % of batches see (t-1, t, t+1)
#   GPU 1  --p-single 1.00   every batch is (t, t, t) -- ZERO temporal information
#
# The control is not optional. Both runs use a generator that did not exist yesterday and a
# schedule nothing has been tuned for, so without it an improvement could be the new sequence
# generator, the motion priors, or the run-to-run scatter that already cost this project a
# false headline once (protocol 17p). With it, the difference is attributable to temporal
# context and nothing else.
#
# Written as a FILE rather than an ssh one-liner deliberately: a `pgrep -f <pattern>` typed
# into an ssh command matches that command's OWN command line and kills the shell running it,
# which happened twice while setting this up.
set -u
cd /home/prusek/mt_enc_exp/mt34_work

PY=${PY:-/home/prusek/mt_env_kajman/bin/python}
EPOCHS=${EPOCHS:-30}
EPOCH_LEN=${EPOCH_LEN:-5000}
WORKERS=${WORKERS:-24}
L=data/enc_sensitivity_testset

export SEG_MODE=ori SEG_BACKBONE=dinov2 SEG_INPUT=raw SEG_ARCH=base
export CALIB=/home/prusek/mt_enc_exp/calib_reg418_morph.json
export MASK_HW=1.0 POS_W=8 CLDICE_W=0.1
export PYTHONPATH=src:synth

run() {   # run <gpu> <p_single> <tag>
  local gpu=$1 p=$2 tag=$3
  mkdir -p "/home/prusek/mt_enc_exp/${tag}_ckpt"
  CUDA_VISIBLE_DEVICES=$gpu nohup "$PY" scripts/train_temporal.py \
      --epochs "$EPOCHS" --val-every 2 --epoch-len "$EPOCH_LEN" --workers "$WORKERS" \
      --p-single "$p" \
      --ckpt-dir "/home/prusek/mt_enc_exp/${tag}_ckpt" \
      --out "/home/prusek/mt_enc_exp/dino_seg_ori_${tag}.pth" \
      --report "$L/train_${tag}.json" \
      > "/home/prusek/mt_enc_exp/train_${tag}.log" 2>&1 &
  echo "  gpu$gpu  p_single=$p  -> train_${tag}.log  (pid $!)"
}

echo "launching on $(hostname):"
run 0 0.35 temporal
sleep 5
run 1 1.00 control
sleep 40

echo
echo "--- gpu ---"
nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv,noheader
echo "--- first lines ---"
for t in temporal control; do
  echo "[$t] $(grep -m1 backgrounds "/home/prusek/mt_enc_exp/train_${t}.log" 2>/dev/null)"
done
