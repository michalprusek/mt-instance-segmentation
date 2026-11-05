#!/bin/bash
# Report only (docs/TODO.md 2026-11-02 14:30): what did the thin regime change? Tile-class survival on htw, L+H anatomy on both blocks, foreground per cut, width of MT-34 VAL.
W=/data2/prusek/mt_work; U=$W/runs/ult; cd $W/mt34_work
export PYTHONPATH=.:src:scripts:synth:third_party OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 TMPDIR=/data2/prusek/tmp; P="nice -n 5 /scratch/prusek/envs/nnunet/bin/python"
$P fg_cut.py > $U/fg_cut.txt 2>&1
$P fwhm_set.py data/real/mt34_htw_eval mt34 > $U/fwhm_mt34.txt 2>&1
$P offgt_survive_thin.py data/enc_sensitivity_testset/offgt_htw/labels.json > $U/anat/offgt_survive_thin.txt 2>&1
for mc in THIN_S3:0.999:0.995 THIN_S4:0.999:0.999 THIN_S5:0.995:0.999 THIN_S6:0.999:0.995; do IFS=: read m ch cr <<< "$mc"
  $P anat_lh.py data/real/mt34_htw_eval $W/preds/${m}_hb_ep40_native $ch $U/anat/lh_hb_$m.json H htw 16 > $U/anat/lh_hb_$m.log 2>&1
  $P anat_lh.py data/real/roi303_v2 $W/preds/GAUSS_${m}_roiv2_ep40_native $cr $U/anat/lh_roiv2_$m.json H all 16 > $U/anat/lh_roiv2_$m.log 2>&1
done
$P sum_lh_thin.py > $U/anat/sum_lh_thin.txt 2>&1
echo "MECH63_DONE $(date +%T)" >> $U/anat/sum_lh_thin.txt
