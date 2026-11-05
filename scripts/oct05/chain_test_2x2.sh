#!/bin/bash
# Declared docs/TODO.md 2026-10-06 17:20: the ONE TEST read (ii), 2x2 {SPARSE35 recipe, NMIX} x {A, L-hand}. Nothing is swept.
set -u
cd /disk2/prusek/mt_work/mt34_work
P=/scratch/prusek/envs/nnunet/bin/python
export PYTHONPATH=.:src:scripts:synth:third_party OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 TMPDIR=/disk2/prusek/tmp
W=/disk2/prusek/mt_work; R=$W/runs/oct05; PD=$W/preds; DER=src/instance/params_a_derived.json; V2=data/real/roi303_v2; HB=data/real/mt34_htw_eval
T=$R/test2x2; mkdir -p $T; L=$R/chain_test_2x2.log; echo $$ > $R/chain_test_2x2.pid; echo "$(date +%F_%T) start" >> $L
dump () {  # arm data fusion outdir tag   (the dump's own CPU sweep is not needed: killed once every npz exists)
  local n; n=$(ls $2/*.h5 | wc -l)
  if [ -d $4 ] && [ "$(ls $4 | wc -l)" -gt 0 ]; then echo "$(date +%F_%T) SKIP existing $4" >> $L; return; fi
  CUDA_VISIBLE_DEVICES=0 $P scripts/eval_v5.py --ckpt $W/runs/$1/ep040.pth --arch nnunet --data $2 --split test \
    --include-empty --no-fov --params $DER --cov 0.80 --thrs 0.98 --infer-scale 1.0 --fusion $3 \
    --dump-pred $4 --out-json $R/dump_$5.unread.json > $R/dump_$5.log 2>&1 &
  local pid=$!
  while kill -0 $pid 2>/dev/null; do grep -q "predicted; sweeping" $R/dump_$5.log 2>/dev/null && { sleep 20; kill $pid 2>/dev/null; break; }; sleep 10; done
  wait $pid 2>/dev/null; echo "$(date +%F_%T) dump $5 n=$(ls $4 2>/dev/null | wc -l)" >> $L
}
for A in NMIX NMIX_S2 NMIX_S3; do
  dump $A $V2 gauss $PD/GAUSS_${A}_roiv2TEST_ep40_native gaussv2TEST_$A
  dump $A $HB uniform $PD/${A}_hbTEST_ep40_native hbTEST_$A
done
echo "$(date +%F_%T) DUMPS_DONE" >> $L
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
evalA () {  # arm cut
  nice -n 5 $P scripts/eval_v5.py --pred-dir $PD/GAUSS_$1_roiv2TEST_ep40_native --arch nnunet --data $V2 --split test --include-empty --no-fov \
    --params $DER --cov 0.80 --thrs $2 --out-json $T/A_roiv2TEST_$1_t$2.json > $T/A_roiv2TEST_$1_t$2.log 2>&1
  local rc=$?; echo "$(date +%F_%T) A roiv2TEST $1 $2 rc=$rc" >> $L
}
gridL () {  # skipA data pred cut out
  LPC_SKIP_A=$1 nice -n 5 $P scripts/lpc/a_grid.py $2 $3 $4 $T/$5.json src/instance/params_l_hand.json 60 0 8 test > $T/$5.log 2>&1
  local rc=$?; echo "$(date +%F_%T) L $5 rc=$rc" >> $L
}
evalA NMIX 0.99 & evalA NMIX_S2 0.99 &
for A in SPARSE35 S35_SEED2 S35_SEED3 NMIX NMIX_S2; do gridL 1 $V2 $PD/GAUSS_${A}_roiv2TEST_ep40_native 0.995 L_roiv2TEST_${A}_t0.995 & done
gridL 0 $V2 $PD/GAUSS_NMIX_S3_roiv2TEST_ep40_native 0.995 AL_roiv2TEST_NMIX_S3_t0.995 &
wait
for q in "SPARSE35 0.995" "S35_SEED2 0.99" "S35_SEED3 0.99" "NMIX 0.995" "NMIX_S2 0.995" "NMIX_S3 0.995"; do set -- $q; gridL 0 $HB $PD/$1_hbTEST_ep40_native $2 AL_hbTEST_$1_t$2 & done
wait
echo "$(date +%F_%T) TEST2X2_DONE" >> $L
