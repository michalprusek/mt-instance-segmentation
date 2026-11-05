#!/bin/bash
# After arm63.sh: b80 fold of the six new runs, then the declared reads at L+H b80 (docs/TODO.md 2026-11-02 09:15).
W=/data2/prusek/mt_work; U=$W/runs/ult
until grep -q ARM63_DONE $U/read_63.txt 2>/dev/null; do sleep 120; done
for m in THIN_S3 THIN_S4 THIN_S5 THIN_S6 DEB_S1 DEB_S2; do bash $U/fold80.sh $m >> $U/after63.log 2>&1; done
cd $W/mt34_work; export PYTHONPATH=.:src:scripts:synth:third_party
md5sum read_arm80.py > $U/read_63_b80.txt; nice -n 5 /scratch/prusek/envs/mt/bin/python read_arm80.py THIN_S3=DEB_S3,THIN_S4=DEB_S4,THIN_S5=DEB_S5,THIN_S6=DEB_S6 DEB_S1,DEB_S2,DEB_S3,DEB_S4,DEB_S5,DEB_S6 >> $U/read_63_b80.txt 2>&1
nice -n 5 /scratch/prusek/envs/mt/bin/python read_arm80.py DEB_S1=AA,DEB_S2=AA_S2 AA,AA_S2,AA_S3,AA_S4,AA_S5,AA_S6 > $U/read_62rep_b80.txt 2>&1
echo "ARM63_B80_DONE $(date +%T)" >> $U/read_63_b80.txt
