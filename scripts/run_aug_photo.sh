#!/bin/bash
# AUG_PHOTO: the SPARSE35 recipe with photometric nuisance augmentation, nothing else changed.
#
# Declared 2026-09-04 in docs/TODO.md §0.0-CAMPAIGN (Holm entries 15/16). Same calibration, seed,
# budget, TIRF fraction, sparse renderer and checkpoint-selection data as SPARSE35
# (scripts/run_sparse_sweep.sh), plus --photo-aug 1.0. The mechanism it targets is frame-dependent
# output calibration; the mechanism check is whether the per-frame optimal-threshold spread narrows.
#
# Liveness: check the pidfile with `kill -0`, never `pgrep -f` on the script name.
cd /disk2/prusek/mt_work/mt34_work
P=/scratch/prusek/envs/nnunet/bin/python
R=/disk2/prusek/mt_work/runs
NAME=${1:-AUG_PHOTO}
GPU=${2:-1}
mkdir -p $R/$NAME
echo $$ > $R/$NAME.pid
CUDA_VISIBLE_DEVICES=$GPU PYTHONPATH=.:src:scripts:synth:third_party \
$P scripts/train_v5.py \
  --arch nnunet --head binary --epochs 40 --epoch-len 6000 --batch 8 --workers 8 \
  --calib calib_V6_c070_narrow.json --seed 20260827 \
  --tirf-prob 0.35 --tirf-label-sparse --photo-aug 1.0 \
  --val-data data/real/mt34_htw_eval --val-n 16 \
  --ckpt-dir $R/$NAME --out $R/$NAME/model.pth --report $R/$NAME/report.json \
  > $R/$NAME.log 2>&1
echo "$NAME done rc=$?" >> $R/$NAME.log
rm -f $R/$NAME.pid
