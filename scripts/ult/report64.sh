#!/bin/bash
# Report only, docs/TODO.md 2026-11-02 22:55. CPU, nice 5.
W=/data2/prusek/mt_work; U=$W/runs/ult; cd $W/mt34_work
export PYTHONPATH=.:src:scripts:scripts/lpc:synth:third_party OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 TMPDIR=/data2/prusek/tmp
P=/scratch/prusek/envs/mt/bin/python; N=/scratch/prusek/envs/nnunet/bin/python
for m in THIN05_S3 THIN05_S4 THIN05_S5 THIN05_S6; do
  c1=$($P -c "import json; print(json.load(open('$U/bridge/hball_$m.json'))['cut'])"); c2=$($P -c "import json; print(json.load(open('$U/bridge/roiv2_$m.json'))['cut'])")
  nice -n 5 $N anat_l.py data/real/mt34_htw_eval $W/preds/${m}_hb_ep40_native $c1 $U/anat/hb_$m.json htw 24 > $U/anat/hb_$m.log 2>&1
  nice -n 5 $N anat_l.py data/real/roi303_v2 $W/preds/GAUSS_${m}_roiv2_ep40_native $c2 $U/anat/roiv2_$m.json all 24 > $U/anat/roiv2_$m.log 2>&1
done
$P short_thin05.py > $U/anat/short_thin05.txt 2>&1
nice -n 5 $N offgt_survive_64.py $U/offgt_htw/labels.json > $U/anat/offgt_survive_64.txt 2>&1
echo "REPORT64_DONE $(date +%T)" >> $U/anat/short_thin05.txt
