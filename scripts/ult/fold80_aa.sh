#!/bin/bash
# b80 fold of the six V14 reference runs (their maps exist); the smoke read of read_arm80.py follows on the four with folds. docs/TODO.md 2026-11-02 09:15.
U=/data2/prusek/mt_work/runs/ult
for m in AA_S3 AA_S4 AA_S5 AA_S6 AA AA_S2; do bash $U/fold80.sh $m >> $U/fold80_aa.log 2>&1; done
cd /data2/prusek/mt_work/mt34_work; export PYTHONPATH=.:src:scripts:synth:third_party
/scratch/prusek/envs/mt/bin/python read_arm80.py DEB_S3=AA_S3,DEB_S4=AA_S4,DEB_S5=AA_S5,DEB_S6=AA_S6 > $U/read_62_b80.txt 2>&1
echo "FOLD80_AA_DONE $(date +%T)" >> $U/fold80_aa.log
