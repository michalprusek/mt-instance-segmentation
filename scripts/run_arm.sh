#!/bin/bash
# One training arm on the SPARSE35 base recipe, plus whatever extra trainer flags follow.
#
#   scripts/run_arm.sh NAME GPU [extra train_v5.py flags...]
#
# Base recipe = SPARSE35 exactly (scripts/run_sparse_sweep.sh): nnU-Net ResEnc-M, binary head,
# 40 epochs x 6000 frames, batch 8, calib_V6_c070_narrow.json, seed 20260827, 35 % TIRF with the
# sparse renderer, 16-frame mt34_htw selection set. Every arm launched through this script differs
# from SPARSE35 ONLY by the extra flags, which is what makes the A/B attributable.
#
# Liveness: `kill -0 $(cat $R/NAME.pid)`, never `pgrep -f` on the script name.
cd /disk2/prusek/mt_work/mt34_work
P=/scratch/prusek/envs/nnunet/bin/python
R=/disk2/prusek/mt_work/runs
NAME=$1; GPU=$2; shift 2
if [ "$GPU" = "auto" ]; then
  # the GPU with the least memory in use at LAUNCH time, if that is under 1 GB; else the A5000 (1).
  # The A100 (0) is often the user's own job -- never assume it is free hours in advance.
  GPU=$(nvidia-smi --query-gpu=index,memory.used --format=csv,noheader,nounits | sort -t, -k2 -n | head -1 | awk -F, '{ if ($2+0 < 1000) print $1; else print 1 }')
fi
mkdir -p $R/$NAME
echo "GPU=$GPU" > $R/$NAME.gpu
echo $$ > $R/$NAME.pid
echo "extra flags: $*" > $R/$NAME.log
CUDA_VISIBLE_DEVICES=$GPU PYTHONPATH=.:src:scripts:synth:third_party \
$P scripts/train_v5.py \
  --arch nnunet --head binary --epochs 40 --epoch-len 6000 --batch 8 --workers 8 \
  --calib calib_V6_c070_narrow.json --seed 20260827 \
  --tirf-prob 0.35 --tirf-label-sparse \
  --val-data data/real/mt34_htw_eval --val-n 16 \
  --ckpt-dir $R/$NAME --out $R/$NAME/model.pth --report $R/$NAME/report.json \
  "$@" \
  >> $R/$NAME.log 2>&1
echo "$NAME done rc=$?" >> $R/$NAME.log
rm -f $R/$NAME.pid
