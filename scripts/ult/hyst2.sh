#!/bin/bash
# Crossing hysteresis v2 grid (docs/TODO.md 2026-11-01 07:55): L + 15 settings on roi303 v2 VAL, htw VAL, synth 96b for
# AA_S3..S6 at the own cuts; then sel_hyst2.py (SEL selects, CONF confirms). CPU only.
cd /data2/prusek/mt_work/mt34_work
export PYTHONPATH=.:src:scripts:synth:third_party OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 TMPDIR=/data2/prusek/tmp
P=/scratch/prusek/envs/nnunet/bin/python; PD=/data2/prusek/mt_work/preds; O=/data2/prusek/mt_work/runs/ult/hyst2; mkdir -p $O
G="L,disc:0.98:22.5,disc:0.98:45,disc:0.95:22.5,disc:0.95:45,disc:0.9:22.5,disc:0.9:45,bridge:0.98:22.5,bridge:0.98:45,bridge:0.98:inf,bridge:0.95:22.5,bridge:0.95:45,bridge:0.95:inf,bridge:0.9:22.5,bridge:0.9:45,bridge:0.9:inf"
for m in AA_S3 AA_S4 AA_S5 AA_S6; do
  $P l_hyst2.py data/real/roi303_v2 $PD/GAUSS_${m}_roiv2_ep40_native 0.995 $O/roiv2_$m.json $G 40 val all > $O/roiv2_$m.log 2>&1
  c=0.995; [ $m = AA_S4 ] && c=0.98
  $P l_hyst2.py data/real/mt34_htw_eval $PD/${m}_hb_ep40_native $c $O/hb_$m.json $G 33 val htw > $O/hb_$m.log 2>&1
  c=0.99; [ $m = AA_S4 ] && c=0.98; [ $m = AA_S5 ] && c=0.995; [ $m = AA_S6 ] && c=0.995
  $P l_hyst2.py data/synth_relink96b $PD/GAUSS_${m}_syn96b_ep40_native $c $O/syn96b_$m.json $G 40 val all > $O/syn96b_$m.log 2>&1
  echo "$(date +%F_%T) $m done"
done
$P sel_hyst2.py $O > $O/sel.txt 2>&1
echo "HYST2_DONE $(date +%T)" >> $O/sel.txt
