#!/bin/bash
# 2026-10-05 GPU chain (inference only): ep040 re-dumps of cached arms, fresh synth set dumps, fusion variants.
set -u
GPU=${1:-0}
cd /disk2/prusek/mt_work/mt34_work
P=/scratch/prusek/envs/nnunet/bin/python
export PYTHONPATH=.:src:scripts:synth:third_party OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2
export TMPDIR=/disk2/prusek/tmp
W=/disk2/prusek/mt_work; R=$W/runs/oct05; PD=$W/preds; mkdir -p $R
DER=src/instance/params_a_derived.json; ROI=data/real/roi303_eval; HB=data/real/mt34_htw_eval
L=$R/chain_gpu.log; echo $$ > $R/chain_gpu.pid; echo "$(date +%F_%T) start gpu=$GPU" >> $L
dump () {  # ckpt data split outdir tag
  if [ -d $4 ] && [ "$(ls $4 | wc -l)" -gt 0 ]; then echo "$(date +%F_%T) SKIP existing $4" >> $L; return; fi
  CUDA_VISIBLE_DEVICES=$GPU $P scripts/eval_v5.py --ckpt $1 --arch nnunet --data $2 --split $3 \
    --include-empty --no-fov --params $DER --cov 0.80 --thrs 0.98 --infer-scale 1.0 \
    --dump-pred $4 --out-json $R/dump_$5.json > $R/dump_$5.log 2>&1
  echo "$(date +%F_%T) dump $5 rc=$? n=$(ls $4 2>/dev/null | wc -l)" >> $L
}
# 1. fresh synth derivation set (CPU, exact GT) + the min-len-5 relabel of the old 48
if [ ! -d data/synth_relink96 ]; then
  $P scripts/build_synth_eval.py --n 96 --seed-base 920000 --all-val --crop 768 --calib calib_V6_c070_narrow.json \
     --min-len 5 --out data/synth_relink96 --png-dir $R/synth_relink96_tif > $R/build_relink96.log 2>&1
  echo "$(date +%F_%T) build synth_relink96 rc=$? n=$(ls data/synth_relink96 | wc -l)" >> $L
fi
if [ ! -d data/synth_eval_s35_ml5 ]; then
  $P scripts/build_synth_eval.py --n 48 --seed-base 910000 --all-val --crop 768 --calib calib_V6_c070_narrow.json \
     --min-len 5 --out data/synth_eval_s35_ml5 --png-dir $R/synth_s35_ml5_tif > $R/build_s35_ml5.log 2>&1
  echo "$(date +%F_%T) build synth_eval_s35_ml5 rc=$? n=$(ls data/synth_eval_s35_ml5 | wc -l)" >> $L
fi
for A in SPARSE35 S35_SEED2 S35_SEED3; do
  dump $W/runs/$A/ep040.pth data/synth_relink96 val $PD/${A}_relink96_ep40_native relink96_$A
done
touch $R/SYNTH_DUMPS_DONE
# 2. htw-block ep040 dumps of the cached arms (the *_htw_native dumps are trainer picks)
for A in SCALE SCALE_S2 NOISE NOISE_S2 COMBO JWEIGHT SPARSE20 SPARSE50 TIRF35; do
  dump $W/runs/$A/ep040.pth $HB val $PD/${A}_hb_ep40_native hbVAL_$A
done
for A in SPARSE20 SPARSE50 TIRF35; do
  dump $W/runs/$A/ep040.pth $ROI val $PD/${A}_roi_ep40_native roiVAL_$A
done
touch $R/ARM_DUMPS_DONE
# 3. tile-fusion / decode variants of SPARSE35 ep040
CUDA_VISIBLE_DEVICES=$GPU $P $R/fusion_dump.py $W/runs/SPARSE35/ep040.pth $HB val $PD hb $PD/SPARSE35_hb_ep40_native > $R/fusion_hb.log 2>&1
echo "$(date +%F_%T) fusion hb rc=$? $(tail -1 $R/fusion_hb.log)" >> $L
CUDA_VISIBLE_DEVICES=$GPU $P $R/fusion_dump.py $W/runs/SPARSE35/ep040.pth $ROI val $PD roi $PD/SPARSE35_roi_ep40_native > $R/fusion_roi.log 2>&1
echo "$(date +%F_%T) fusion roi rc=$? $(tail -1 $R/fusion_roi.log)" >> $L
touch $R/FUSION_DUMPS_DONE
echo "$(date +%F_%T) GPU_ALL_DONE" >> $L
