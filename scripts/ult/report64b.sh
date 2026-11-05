#!/bin/bash
W=/data2/prusek/mt_work; U=$W/runs/ult; cd $W/mt34_work
export PYTHONPATH=.:src:scripts:scripts/lpc:synth:third_party OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 TMPDIR=/data2/prusek/tmp
P=/scratch/prusek/envs/mt/bin/python; N=/scratch/prusek/envs/nnunet/bin/python
for m in DEB_S1 DEB_S2 AA AA_S2; do
  [ -f $U/anat/roiv2_$m.json ] && [ -f $U/anat/hb_$m.json ] && { echo "have $m"; continue; }
  c1=$($P -c "import json; print(json.load(open('$U/bridge/hball_$m.json'))['cut'])"); c2=$($P -c "import json; print(json.load(open('$U/bridge/roiv2_$m.json'))['cut'])")
  echo "$m cuts $c1 $c2"
  nice -n 5 $N anat_l.py data/real/mt34_htw_eval $W/preds/${m}_hb_ep40_native $c1 $U/anat/hb_$m.json htw 24 > $U/anat/hb_$m.log 2>&1 || echo "FAIL hb $m"
  nice -n 5 $N anat_l.py data/real/roi303_v2 $W/preds/GAUSS_${m}_roiv2_ep40_native $c2 $U/anat/roiv2_$m.json all 24 > $U/anat/roiv2_$m.log 2>&1 || echo "FAIL roiv2 $m"
done
$P short_runlevel.py > $U/anat/short_runlevel.txt 2>&1; cat $U/anat/short_runlevel.txt
