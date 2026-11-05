#!/bin/bash
# Fold-in read (docs/TODO.md 2026-11-02 06:45): V16 + L+H at the own L cuts, VAL + synth; anatomy of DEB_S3..S6 with L. CPU only.
cd /data2/prusek/mt_work/mt34_work
export PYTHONPATH=.:src:scripts:synth:third_party OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 LPC_SKIP_A=1 TMPDIR=/data2/prusek/tmp
P=/scratch/prusek/envs/nnunet/bin/python; PD=/data2/prusek/mt_work/preds; O=/data2/prusek/mt_work/runs/ult/fold; A=/data2/prusek/mt_work/runs/ult/anat; mkdir -p $O
G=L,bridge:0.98:22.5
for mc in DEB_S3:0.995:0.99 DEB_S4:0.99:0.99 DEB_S5:0.99:0.99 DEB_S6:0.995:0.98; do IFS=: read m ch cs <<< "$mc"
  $P l_hyst2.py data/real/roi303_v2 $PD/GAUSS_${m}_roiv2_ep40_native 0.995 $O/roiv2_$m.json $G 40 val all > $O/roiv2_$m.log 2>&1
  $P l_hyst2.py data/real/mt34_htw_eval $PD/${m}_hb_ep40_native $ch $O/hball_$m.json $G 40 val all > $O/hball_$m.log 2>&1
  $P l_hyst2.py data/synth_relink96b $PD/GAUSS_${m}_syn96b_ep40_native $cs $O/syn96b_$m.json $G 40 val all > $O/syn96b_$m.log 2>&1
  $P anat_l.py data/real/mt34_htw_eval $PD/${m}_hb_ep40_native $ch $A/hb_$m.json htw 33 > /dev/null 2>&1
  $P anat_l.py data/real/roi303_v2 $PD/GAUSS_${m}_roiv2_ep40_native 0.995 $A/roiv2_$m.json all 40 > /dev/null 2>&1
  echo "$(date +%F_%T) $m done"
done
$P sum_anat_deb.py > $A/sum_anat_deb.txt 2>&1
echo FOLD_DONE
