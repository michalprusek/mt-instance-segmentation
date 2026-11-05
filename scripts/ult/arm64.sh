#!/bin/bash
# ARM 64 DOSE (V18) + ARM 63b THIN05 (V19), docs/TODO.md 2026-11-02 14:30. Seeds 20260906-09, four runs per GPU, launch_ult.sh verbatim, chains; cut 0.9995 added before the own-cut rule; L, L+H b60, b80 folds, L anatomy for the short-recall rule; declared reads.
set -u
W=/data2/prusek/mt_work; R=$W/runs; U=$R/ult; L=$U/arm64.log; A="DOSE_S3 DOSE_S4 DOSE_S5 DOSE_S6 THIN05_S3 THIN05_S4 THIN05_S5 THIN05_S6"
cd $W
echo "$(date +%F_%T) start; generator $(md5sum < mt34_work/synth/mt_generator.py | cut -c1-8) V18 $(md5sum < mt34_work/calib_V18_debris05.json | cut -c1-8) V19 $(md5sum < mt34_work/calib_V19_thin05.json | cut -c1-8)" >> $L
go () { # name calib seed gpu
  ./launch_ult.sh $1 $2 $3 $4 >> $L 2>&1; nohup $U/chain.sh $1 $R/$1/ep040.pth $4 $(cat $R/$1.pid) > $U/chain_$1.out 2>&1 < /dev/null &
}
go DOSE_S3 calib_V18_debris05.json 20260906 0; go DOSE_S4 calib_V18_debris05.json 20260907 0; go THIN05_S3 calib_V19_thin05.json 20260906 0; go THIN05_S4 calib_V19_thin05.json 20260907 0
go DOSE_S5 calib_V18_debris05.json 20260908 1; go DOSE_S6 calib_V18_debris05.json 20260909 1; go THIN05_S5 calib_V19_thin05.json 20260908 1; go THIN05_S6 calib_V19_thin05.json 20260909 1
for m in $A; do until grep -q ALL_DONE $U/chain_$m.log 2>/dev/null; do sleep 120; done; done
echo "$(date +%F_%T) chains done" >> $L
bash $U/sweep_t.sh 0.9995 $A >> $L 2>&1
for m in $A; do bash $U/fold_one.sh $m >> $L 2>&1; bash $U/fold80.sh $m >> $L 2>&1; done
cd $W/mt34_work; export PYTHONPATH=.:src:scripts:synth:third_party OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 TMPDIR=/data2/prusek/tmp; P=/scratch/prusek/envs/mt/bin/python
for m in DOSE_S3 DOSE_S4 DOSE_S5 DOSE_S6; do
  c1=$($P -c "import json; print(json.load(open('$U/bridge/hball_$m.json'))['cut'])"); c2=$($P -c "import json; print(json.load(open('$U/bridge/roiv2_$m.json'))['cut'])")
  /scratch/prusek/envs/nnunet/bin/python anat_l.py data/real/mt34_htw_eval $W/preds/${m}_hb_ep40_native $c1 $U/anat/hb_$m.json htw 24 > $U/anat/hb_$m.log 2>&1
  /scratch/prusek/envs/nnunet/bin/python anat_l.py data/real/roi303_v2 $W/preds/GAUSS_${m}_roiv2_ep40_native $c2 $U/anat/roiv2_$m.json all 24 > $U/anat/roiv2_$m.log 2>&1
done
RF=DEB_S1,DEB_S2,DEB_S3,DEB_S4,DEB_S5,DEB_S6
md5sum read_arm80.py read_dose.py > $U/read_64.txt
$P read_arm80.py DOSE_S3=DEB_S3,DOSE_S4=DEB_S4,DOSE_S5=DEB_S5,DOSE_S6=DEB_S6 $RF >> $U/read_64.txt 2>&1
$P read_dose.py >> $U/read_64.txt 2>&1
$P read_arm80.py DOSE_S3=AA_S3,DOSE_S4=AA_S4,DOSE_S5=AA_S5,DOSE_S6=AA_S6 > $U/read_64_vsAA.txt 2>&1
$P read_arm.py DOSE_S3=DEB_S3,DOSE_S4=DEB_S4,DOSE_S5=DEB_S5,DOSE_S6=DEB_S6 $RF > $U/read_64_b60.txt 2>&1
md5sum read_arm80.py > $U/read_63b.txt
$P read_arm80.py THIN05_S3=DEB_S3,THIN05_S4=DEB_S4,THIN05_S5=DEB_S5,THIN05_S6=DEB_S6 $RF >> $U/read_63b.txt 2>&1
$P read_arm.py THIN05_S3=DEB_S3,THIN05_S4=DEB_S4,THIN05_S5=DEB_S5,THIN05_S6=DEB_S6 $RF > $U/read_63b_b60.txt 2>&1
echo "ARM64_DONE $(date +%T)" >> $U/read_64.txt
