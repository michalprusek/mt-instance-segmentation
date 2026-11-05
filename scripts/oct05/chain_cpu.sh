#!/bin/bash
# 2026-10-05 CPU chain: declared-instrument sweeps (derived params, cov 0.80, no-fov, include-empty), sharded per node.
set -u
cd /disk2/prusek/mt_work/mt34_work
P=/scratch/prusek/envs/nnunet/bin/python
export PYTHONPATH=.:src:scripts:synth:third_party OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
export TMPDIR=/disk2/prusek/tmp
W=/disk2/prusek/mt_work; R=$W/runs/oct05; PD=$W/preds; mkdir -p $R/sw
DER=src/instance/params_a_derived.json; ROI=data/real/roi303_eval; HB=data/real/mt34_htw_eval
L=$R/chain_cpu.log; echo $$ > $R/chain_cpu.pid; echo "$(date +%F_%T) start" >> $L
G9="0.999 0.995 0.99 0.98 0.97 0.96 0.94 0.92 0.90"; G13="$G9 0.85 0.80 0.70 0.50"
one () {  # pred data tag thr
  [ -f $R/sw/sweep_$3_t$4.json ] && return
  nice -n 5 $P scripts/eval_v5.py --pred-dir $1 --arch nnunet --data $2 --split val --include-empty --no-fov \
    --params $DER --cov 0.80 --thrs $4 --out-json $R/sw/sweep_$3_t$4.json > $R/sw/sweep_$3_t$4.log 2>&1
  echo "$(date +%F_%T) node $3 t=$4 rc=$?" >> $L
}
export -f one; export P R DER L
run () { xargs -P ${NPAR:-24} -L 1 bash -c 'one "$@"' _ < $1; }
# stage A: roi303 VAL of the arms whose ep040 native dumps already exist
J=$R/jobsA.txt; : > $J
for A in SCALE SCALE_S2 NOISE NOISE_S2 COMBO JWEIGHT; do for T in $G9; do echo "$PD/${A}_roi_ep40_native $ROI roiVAL_${A}_der $T" >> $J; done; done
NPAR=20 run $J; echo "$(date +%F_%T) STAGE_A_DONE" >> $L
# stage B: RELINK synth grid (needs the synth dumps)
while [ ! -f $R/SYNTH_DUMPS_DONE ]; do sleep 60; done
for A in SPARSE35 S35_SEED2 S35_SEED3; do for T in 0.98 0.99; do
  nice -n 5 $P $R/relink_grid.py data/synth_relink96 $PD/${A}_relink96_ep40_native $T $R/relink_synth96_${A}_$T.json 24 > $R/relink_synth96_${A}_$T.log 2>&1
  echo "$(date +%F_%T) relink synth96 $A $T rc=$?" >> $L
done; done
echo "$(date +%F_%T) STAGE_B_DONE" >> $L
# stage C: htw block (13 nodes) + the three re-dumped roi arms
while [ ! -f $R/ARM_DUMPS_DONE ]; do sleep 60; done
J=$R/jobsC.txt; : > $J
for A in SCALE SCALE_S2 NOISE NOISE_S2 COMBO JWEIGHT SPARSE20 SPARSE50 TIRF35; do for T in $G13; do echo "$PD/${A}_hb_ep40_native $HB hbVAL_${A}_der $T" >> $J; done; done
for A in SPARSE20 SPARSE50 TIRF35; do for T in $G9; do echo "$PD/${A}_roi_ep40_native $ROI roiVAL_${A}_der $T" >> $J; done; done
NPAR=26 run $J; echo "$(date +%F_%T) STAGE_C_DONE" >> $L
# stage D: fusion variants
while [ ! -f $R/FUSION_DUMPS_DONE ]; do sleep 60; done
J=$R/jobsD.txt; : > $J
for V in A_decl A_c UL387_c G256_a G256_c G128_c WHOLE_c; do
  for T in $G13; do echo "$PD/FUS_${V}_hb $HB hbVAL_FUS${V}_der $T" >> $J; done
  for T in $G9; do echo "$PD/FUS_${V}_roi $ROI roiVAL_FUS${V}_der $T" >> $J; done
done
NPAR=26 run $J; echo "$(date +%F_%T) STAGE_D_DONE" >> $L
echo "$(date +%F_%T) CPU_ALL_DONE" >> $L
