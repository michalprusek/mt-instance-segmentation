#!/bin/bash
# Declared docs/TODO.md 2026-10-06 07:55 (2): baseline dumps for the roi303 v2 TEST read (gauss fusion) + the third seed's
# v2 VAL dump and sweep. TEST is NOT swept: it is evaluated later at the one VAL-chosen threshold per seed.
set -u
cd /disk2/prusek/mt_work/mt34_work
P=/scratch/prusek/envs/nnunet/bin/python
export PYTHONPATH=.:src:scripts:synth:third_party OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 TMPDIR=/disk2/prusek/tmp
W=/disk2/prusek/mt_work; R=$W/runs/oct05; PD=$W/preds; DER=src/instance/params_a_derived.json; V2=data/real/roi303_v2
L=$R/chain_test_v2.log; echo $$ > $R/chain_test_v2.pid; echo "$(date +%F_%T) start" >> $L
dump () {  # arm split outdir tag
  if [ -d $3 ] && [ "$(ls $3 | wc -l)" -gt 0 ]; then echo "$(date +%F_%T) SKIP existing $3" >> $L; return; fi
  CUDA_VISIBLE_DEVICES=1 $P scripts/eval_v5.py --ckpt $W/runs/$1/ep040.pth --arch nnunet --data $V2 --split $2 \
    --include-empty --no-fov --params $DER --cov 0.80 --thrs 0.98 --infer-scale 1.0 --fusion gauss \
    --dump-pred $3 --out-json $R/dump_$4.unread.json > $R/dump_$4.log 2>&1
  echo "$(date +%F_%T) dump $4 rc=$? n=$(ls $3 2>/dev/null | wc -l)" >> $L
}
dump S35_SEED3 val $PD/GAUSS_S35_SEED3_roiv2_ep40_native gaussv2_S35_SEED3
for A in SPARSE35 S35_SEED2 S35_SEED3; do dump $A test $PD/GAUSS_${A}_roiv2TEST_ep40_native gaussv2TEST_$A; done
echo "$(date +%F_%T) DUMPS_DONE" >> $L
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
for T in 0.999 0.995 0.99 0.98 0.97 0.96 0.94 0.92 0.90; do
  ( [ -f $R/sw/sweep_roiv2VAL_GAUSS_S35_SEED3_der_t$T.json ] || nice -n 5 $P scripts/eval_v5.py --pred-dir $PD/GAUSS_S35_SEED3_roiv2_ep40_native --arch nnunet --data $V2 --split val \
    --include-empty --no-fov --params $DER --cov 0.80 --thrs $T --out-json $R/sw/sweep_roiv2VAL_GAUSS_S35_SEED3_der_t$T.json > $R/sw/sweep_roiv2VAL_GAUSS_S35_SEED3_der_t$T.log 2>&1 ) &
done; wait
echo "$(date +%F_%T) TESTV2_ALL_DONE" >> $L
