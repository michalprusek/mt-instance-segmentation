#!/bin/bash
# ARM 63 THIN + ARM 62 replication (docs/TODO.md 2026-11-02 07:15). Waits for the TEST read to be written, then THIN_S3..S6 (V17, seeds
# 20260906-09) and DEB_S1, DEB_S2 (V16, the seeds of AA, AA_S2), three per GPU, launch_ult.sh verbatim, chains; L+H folds; declared reads.
set -u
W=/data2/prusek/mt_work; R=$W/runs; U=$R/ult; L=$U/arm63.log
until grep -q TEST62_DONE $U/read_test62.txt 2>/dev/null; do sleep 60; done
cd $W
echo "$(date +%F_%T) start; generator $(md5sum < mt34_work/synth/mt_generator.py | cut -c1-8) V17 $(md5sum < mt34_work/calib_V17_thin.json | cut -c1-8) V16 $(md5sum < mt34_work/calib_V16_debris.json | cut -c1-8) seeds AA $SEED1 AA_S2 $SEED2" >> $L
go () { # name calib seed gpu
  ./launch_ult.sh $1 $2 $3 $4 >> $L 2>&1; nohup $U/chain.sh $1 $R/$1/ep040.pth $4 $(cat $R/$1.pid) > $U/chain_$1.out 2>&1 < /dev/null &
}
go THIN_S3 calib_V17_thin.json 20260906 0; go THIN_S4 calib_V17_thin.json 20260907 0; go DEB_S1 calib_V16_debris.json $SEED1 0
go THIN_S5 calib_V17_thin.json 20260908 1; go THIN_S6 calib_V17_thin.json 20260909 1; go DEB_S2 calib_V16_debris.json $SEED2 1
for m in THIN_S3 THIN_S4 THIN_S5 THIN_S6 DEB_S1 DEB_S2; do until grep -q ALL_DONE $U/chain_$m.log 2>/dev/null; do sleep 120; done; done
echo "$(date +%F_%T) chains done" >> $L
for m in THIN_S3 THIN_S4 THIN_S5 THIN_S6 DEB_S1 DEB_S2 AA AA_S2; do bash $U/fold_one.sh $m >> $L 2>&1; done
cd $W/mt34_work; export PYTHONPATH=.:src:scripts:synth:third_party
md5sum read_arm.py > $U/read_63.txt; nice -n 5 /scratch/prusek/envs/mt/bin/python read_arm.py THIN_S3=DEB_S3,THIN_S4=DEB_S4,THIN_S5=DEB_S5,THIN_S6=DEB_S6 DEB_S1,DEB_S2,DEB_S3,DEB_S4,DEB_S5,DEB_S6 >> $U/read_63.txt 2>&1
nice -n 5 /scratch/prusek/envs/mt/bin/python read_arm.py DEB_S1=AA,DEB_S2=AA_S2 AA,AA_S2,AA_S3,AA_S4,AA_S5,AA_S6 > $U/read_62rep.txt 2>&1
echo "ARM63_DONE $(date +%T)" >> $U/read_63.txt
