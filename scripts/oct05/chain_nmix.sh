#!/bin/bash
# ARM 56/57 read chain (declared docs/TODO.md 2026-10-05 19:50): waits for the four trainings, dumps ep040 natively
# (htw block VAL, roi303 VAL, synth_relink96), then the declared sweeps and the synth guard at m = 1.
set -u
cd /disk2/prusek/mt_work/mt34_work
P=/scratch/prusek/envs/nnunet/bin/python
export PYTHONPATH=.:src:scripts:synth:third_party OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2
export TMPDIR=/disk2/prusek/tmp
W=/disk2/prusek/mt_work; R=$W/runs/oct05; PD=$W/preds; mkdir -p $R/sw
DER=src/instance/params_a_derived.json; ROI=data/real/roi303_eval; HB=data/real/mt34_htw_eval
L=$R/chain_nmix.log; echo $$ > $R/chain_nmix.pid; echo "$(date +%F_%T) start" >> $L
ARMS="NMIX NMIX_S2 NMIXFLAT NMIXFLAT_S2"
for A in $ARMS; do
  while kill -0 $(cat $W/runs/$A.pid) 2>/dev/null; do sleep 120; done
  echo "$(date +%F_%T) training $A ended; ep040 $(ls $W/runs/$A/ep040.pth 2>/dev/null | wc -l)" >> $L
done
dump () {  # gpu ckpt data outdir tag
  if [ -d $4 ] && [ "$(ls $4 | wc -l)" -gt 0 ]; then echo "$(date +%F_%T) SKIP existing $4" >> $L; return; fi
  CUDA_VISIBLE_DEVICES=$1 $P scripts/eval_v5.py --ckpt $2 --arch nnunet --data $3 --split val \
    --include-empty --no-fov --params $DER --cov 0.80 --thrs 0.98 --infer-scale 1.0 \
    --dump-pred $4 --out-json $R/dump_$5.json > $R/dump_$5.log 2>&1
  echo "$(date +%F_%T) dump $5 rc=$? n=$(ls $4 2>/dev/null | wc -l)" >> $L
}
dump_arm () {  # gpu arm
  [ -f $W/runs/$2/ep040.pth ] || { echo "$(date +%F_%T) NO ep040 for $2" >> $L; return; }
  dump $1 $W/runs/$2/ep040.pth $HB $PD/${2}_hb_ep40_native hbVAL_$2
  dump $1 $W/runs/$2/ep040.pth data/synth_relink96 $PD/${2}_relink96_ep40_native relink96_$2
  dump $1 $W/runs/$2/ep040.pth $ROI $PD/${2}_roi_ep40_native roiVAL_$2
}
( dump_arm 0 NMIX; dump_arm 0 NMIX_S2 ) &
( dump_arm 1 NMIXFLAT; dump_arm 1 NMIXFLAT_S2 ) &
wait; echo "$(date +%F_%T) DUMPS_DONE" >> $L
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
G9="0.999 0.995 0.99 0.98 0.97 0.96 0.94 0.92 0.90"; G13="$G9 0.85 0.80 0.70 0.50"
one () {
  [ -f $R/sw/sweep_$3_t$4.json ] && return
  nice -n 5 $P scripts/eval_v5.py --pred-dir $1 --arch nnunet --data $2 --split val --include-empty --no-fov \
    --params $DER --cov 0.80 --thrs $4 --out-json $R/sw/sweep_$3_t$4.json > $R/sw/sweep_$3_t$4.log 2>&1
  echo "$(date +%F_%T) node $3 t=$4 rc=$?" >> $L
}
export -f one; export P R DER L
J=$R/jobsN.txt; : > $J
for A in $ARMS; do for T in $G13; do echo "$PD/${A}_hb_ep40_native $HB hbVAL_${A}_der $T" >> $J; done; done
for A in $ARMS; do for T in $G9; do echo "$PD/${A}_roi_ep40_native $ROI roiVAL_${A}_der $T" >> $J; done; done
xargs -P 28 -L 1 bash -c 'one "$@"' _ < $J
echo "$(date +%F_%T) SWEEPS_DONE" >> $L
for A in $ARMS; do
  nice -n 5 $P $R/relink_grid.py data/synth_relink96 $PD/${A}_relink96_ep40_native 0.98 $R/relink_synth96_${A}_0.98.json 24 val 1 > $R/relink_synth96_${A}_0.98.log 2>&1
  echo "$(date +%F_%T) synth guard $A rc=$?" >> $L
done
echo "$(date +%F_%T) NMIX_ALL_DONE" >> $L
