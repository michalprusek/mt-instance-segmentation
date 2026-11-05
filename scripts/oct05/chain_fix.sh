#!/bin/bash
# 2026-10-05 22:15 rulings (docs/TODO.md): FIX-FUSION identity checks + two-seed quantification, and D1b RELINK on a fresh set.
set -u
cd /disk2/prusek/mt_work/mt34_work
P=/scratch/prusek/envs/nnunet/bin/python
export PYTHONPATH=.:src:scripts:synth:third_party OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 TMPDIR=/disk2/prusek/tmp
W=/disk2/prusek/mt_work; R=$W/runs/oct05; PD=$W/preds; GPU=${GPU:-0}
DER=src/instance/params_a_derived.json; ROI=data/real/roi303_eval
L=$R/chain_fix.log; echo $$ > $R/chain_fix.pid; echo "$(date +%F_%T) start" >> $L
dump () {  # ckpt data outdir tag fusion
  if [ -d $3 ] && [ "$(ls $3 | wc -l)" -gt 0 ]; then echo "$(date +%F_%T) SKIP existing $3" >> $L; return; fi
  CUDA_VISIBLE_DEVICES=$GPU nice -n 5 $P scripts/eval_v5.py --ckpt $1 --arch nnunet --data $2 --split val \
    --include-empty --no-fov --params $DER --cov 0.80 --thrs 0.98 --infer-scale 1.0 --fusion $5 \
    --dump-pred $3 --out-json $R/dump_$4.json > $R/dump_$4.log 2>&1
  echo "$(date +%F_%T) dump $4 rc=$? n=$(ls $3 2>/dev/null | wc -l)" >> $L
}
# D1b first (short)
if [ ! -d data/synth_relink96b ]; then
  nice -n 5 $P scripts/build_synth_eval.py --n 96 --seed-base 940000 --all-val --crop 768 --calib calib_V6_c070_narrow.json \
     --min-len 5 --out data/synth_relink96b --png-dir $R/synth_relink96b_tif > $R/build_relink96b.log 2>&1
  echo "$(date +%F_%T) build synth_relink96b rc=$? n=$(ls data/synth_relink96b | wc -l)" >> $L
fi
for A in SPARSE35 S35_SEED2 S35_SEED3; do
  dump $W/runs/$A/ep040.pth data/synth_relink96b $PD/${A}_relink96b_ep40_native relink96b_$A uniform
  nice -n 5 $P $R/relink_grid.py data/synth_relink96b $PD/${A}_relink96b_ep40_native 0.98 $R/relink_synth96b_${A}_0.98.json 8 > $R/relink_synth96b_${A}_0.98.log 2>&1
  echo "$(date +%F_%T) relink synth96b $A rc=$?" >> $L
done
echo "$(date +%F_%T) D1B_DONE" >> $L
# FIX-FUSION
dump $W/runs/SPARSE35/ep040.pth $ROI $PD/FIXCHK_uniform_SPARSE35_roi fixchk_uniform uniform
dump $W/runs/SPARSE35/ep040.pth $ROI $PD/GAUSS_SPARSE35_roi_ep40_native gauss_SPARSE35 gauss
$P - > $R/fixchk.txt 2>&1 <<PY
import glob, os, numpy as np
def cmp(a, b):
    w = 0.0; n = 0
    for f in sorted(glob.glob(a + "/*.npz")):
        g = os.path.join(b, os.path.basename(f))
        if not os.path.exists(g): continue
        w = max(w, float(np.abs(np.load(f)["prob"].astype(np.float32) - np.load(g)["prob"].astype(np.float32)).max())); n += 1
    return n, w
print("uniform vs cached declared", cmp("$PD/FIXCHK_uniform_SPARSE35_roi", "$PD/SPARSE35_roi_ep40_native"))
print("gauss vs FUS_G256_a", cmp("$PD/GAUSS_SPARSE35_roi_ep40_native", "$PD/FUS_G256_a_roi"))
PY
echo "$(date +%F_%T) fixchk $(tr '\n' ' ' < $R/fixchk.txt)" >> $L
for A in S35_SEED2 S35_SEED3; do dump $W/runs/$A/ep040.pth $ROI $PD/GAUSS_${A}_roi_ep40_native gauss_$A gauss; done
one () {
  [ -f $R/sw/sweep_$3_t$4.json ] && return
  nice -n 5 $P scripts/eval_v5.py --pred-dir $1 --arch nnunet --data $2 --split val --include-empty --no-fov \
    --params $DER --cov 0.80 --thrs $4 --out-json $R/sw/sweep_$3_t$4.json > $R/sw/sweep_$3_t$4.log 2>&1
}
export -f one; export P R DER
J=$R/jobsFix.txt; : > $J
for A in SPARSE35 S35_SEED2 S35_SEED3; do for T in 0.999 0.995 0.99 0.98 0.97 0.96 0.94 0.92 0.90; do
  echo "$PD/GAUSS_${A}_roi_ep40_native $ROI roiVAL_GAUSS_${A}_der $T" >> $J; done; done
xargs -P 6 -L 1 bash -c 'one "$@"' _ < $J
echo "$(date +%F_%T) FIX_ALL_DONE" >> $L
