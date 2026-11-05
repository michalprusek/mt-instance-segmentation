#!/bin/bash
# TIRF fraction sweep with the SPARSE-LABELLING renderer.
#
# Replaces the killed continuous-renderer sweep. Everything is held fixed except --tirf-prob and
# the new --tirf-label-sparse: same calibration, same seed, same budget, same checkpoint-selection
# data as TIRF35, so both comparisons below are clean.
#
#   renderer A/B :  SPARSE35 vs the existing TIRF35   (same fraction, sparse vs continuous)
#   fraction curve: PHOTO_V6 (0.00) - SPARSE20 - SPARSE35 - SPARSE50
#
# Liveness: check the pidfile with `kill -0`, never `pgrep -f` on the script name -- a pgrep
# pattern also matches the ssh command line that asks about it, which has already reported
# "2 running" in this project when nothing was.
cd /disk2/prusek/mt_work/mt34_work
P=/scratch/prusek/envs/nnunet/bin/python
R=/disk2/prusek/mt_work/runs
echo $$ > $R/sparse_sweep.pid

train () {   # $1 = name, $2 = tirf_prob, $3 = gpu
  mkdir -p $R/$1
  CUDA_VISIBLE_DEVICES=$3 PYTHONPATH=.:src:scripts:synth:third_party \
  $P scripts/train_v5.py \
    --arch nnunet --head binary --epochs 40 --epoch-len 6000 --batch 8 --workers 8 \
    --calib calib_V6_c070_narrow.json --seed 20260827 \
    --tirf-prob $2 --tirf-label-sparse \
    --val-data data/real/mt34_htw_eval --val-n 16 \
    --ckpt-dir $R/$1 --out $R/$1/model.pth --report $R/$1/report.json \
    > $R/$1.log 2>&1
  echo "$1 done  rc=$?"
}

( train SPARSE20 0.20 0 ; train SPARSE50 0.50 0 ) &
( train SPARSE35 0.35 1 ) &
wait
rm -f $R/sparse_sweep.pid
echo ALL_DONE
