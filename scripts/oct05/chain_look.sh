#!/bin/bash
# Second looks declared docs/TODO.md 2026-10-06 04:45: (A) ARM 56/57 on v1 roi303 VAL under gauss fusion,
# (B) SPARSE35 / S35_SEED2 / NMIX / NMIX_S2 on roi303 v2 VAL under gauss fusion. v2 TEST is never dumped.
set -u
cd /disk2/prusek/mt_work/mt34_work
P=/scratch/prusek/envs/nnunet/bin/python
export PYTHONPATH=.:src:scripts:synth:third_party OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2
export TMPDIR=/disk2/prusek/tmp
W=/disk2/prusek/mt_work; R=$W/runs/oct05; PD=$W/preds
DER=src/instance/params_a_derived.json; ROI=data/real/roi303_eval; V2=data/real/roi303_v2
L=$R/chain_look.log; echo $$ > $R/chain_look.pid; echo "$(date +%F_%T) start" >> $L
ck () { case $1 in SPARSE35) echo $W/runs/SPARSE35/ep040.pth;; *) echo $W/runs/$1/ep040.pth;; esac; }
dump () {  # gpu arm data outdir tag
  if [ -d $4 ] && [ "$(ls $4 | wc -l)" -gt 0 ]; then echo "$(date +%F_%T) SKIP existing $4" >> $L; return; fi
  CUDA_VISIBLE_DEVICES=$1 $P scripts/eval_v5.py --ckpt $(ck $2) --arch nnunet --data $3 --split val \
    --include-empty --no-fov --params $DER --cov 0.80 --thrs 0.98 --infer-scale 1.0 --fusion gauss \
    --dump-pred $4 --out-json $R/dump_$5.json > $R/dump_$5.log 2>&1
  echo "$(date +%F_%T) dump $5 rc=$? n=$(ls $4 2>/dev/null | wc -l)" >> $L
}
( dump 0 NMIX $ROI $PD/GAUSS_NMIX_roi_ep40_native gauss_NMIX; dump 0 NMIX_S2 $ROI $PD/GAUSS_NMIX_S2_roi_ep40_native gauss_NMIX_S2
  dump 0 NMIX $V2 $PD/GAUSS_NMIX_roiv2_ep40_native gaussv2_NMIX; dump 0 SPARSE35 $V2 $PD/GAUSS_SPARSE35_roiv2_ep40_native gaussv2_SPARSE35 ) &
( dump 1 NMIXFLAT $ROI $PD/GAUSS_NMIXFLAT_roi_ep40_native gauss_NMIXFLAT; dump 1 NMIXFLAT_S2 $ROI $PD/GAUSS_NMIXFLAT_S2_roi_ep40_native gauss_NMIXFLAT_S2
  dump 1 NMIX_S2 $V2 $PD/GAUSS_NMIX_S2_roiv2_ep40_native gaussv2_NMIX_S2; dump 1 S35_SEED2 $V2 $PD/GAUSS_S35_SEED2_roiv2_ep40_native gaussv2_S35_SEED2 ) &
wait; echo "$(date +%F_%T) DUMPS_DONE" >> $L
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
G9="0.999 0.995 0.99 0.98 0.97 0.96 0.94 0.92 0.90"
one () {
  [ -f $R/sw/sweep_$3_t$4.json ] && return
  nice -n 5 $P scripts/eval_v5.py --pred-dir $1 --arch nnunet --data $2 --split val --include-empty --no-fov \
    --params $DER --cov 0.80 --thrs $4 --out-json $R/sw/sweep_$3_t$4.json > $R/sw/sweep_$3_t$4.log 2>&1
  echo "$(date +%F_%T) node $3 t=$4 rc=$?" >> $L
}
export -f one; export P R DER L
J=$R/jobsLook.txt; : > $J
for A in NMIX NMIX_S2 NMIXFLAT NMIXFLAT_S2; do for T in $G9; do echo "$PD/GAUSS_${A}_roi_ep40_native $ROI roiVAL_GAUSS_${A}_der $T" >> $J; done; done
for A in SPARSE35 S35_SEED2 NMIX NMIX_S2; do for T in $G9; do echo "$PD/GAUSS_${A}_roiv2_ep40_native $V2 roiv2VAL_GAUSS_${A}_der $T" >> $J; done; done
xargs -P 28 -L 1 bash -c 'one "$@"' _ < $J
echo "$(date +%F_%T) LOOK_ALL_DONE" >> $L
