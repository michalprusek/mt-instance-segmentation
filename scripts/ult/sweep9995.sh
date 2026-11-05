#!/bin/bash
# Grid-edge repair of the ARM 63 read (docs/TODO.md 2026-11-02 14:10): L at cut 0.9995 on the existing dumps, THIN_S3..S6 and every DEB run, same a_grid.py call as chain.sh.
cd /data2/prusek/mt_work/mt34_work
P=/scratch/prusek/envs/nnunet/bin/python
export PYTHONPATH=.:src:scripts:synth:third_party OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 TMPDIR=/data2/prusek/tmp LPC_SKIP_A=1
W=/data2/prusek/mt_work; R=$W/runs/ult; PD=$W/preds; V2=data/real/roi303_v2; HB=data/real/mt34_htw_eval; SY=data/synth_relink96b; T=0.9995
J=$R/jobs_sweep9995.txt; : > $J
for A in THIN_S3 THIN_S4 THIN_S5 THIN_S6 DEB_S1 DEB_S2 DEB_S3 DEB_S4 DEB_S5 DEB_S6; do
  echo "$PD/${A}_hb_ep40_native $HB hb_$A $T" >> $J
  echo "$PD/GAUSS_${A}_roiv2_ep40_native $V2 roiv2_$A $T" >> $J
  echo "$PD/GAUSS_${A}_syn96b_ep40_native $SY syn96b_$A $T" >> $J
done
one () { O=$R/lsw/L_$3_t$4; [ -f $O.json ] && return
  nice -n 5 $P scripts/lpc/a_grid.py $2 $1 $4 $O.json src/instance/params_l_hand.json 60 0 4 val > $O.log 2>&1
  echo "$(date +%F_%T) node $3 t=$4 exit $? $( [ -f $O.json ] && echo ok || echo MISSING)" >> $R/sweep9995.log; }
export -f one; export P R
xargs -P 12 -L 1 bash -c "one \"\$@\"" _ < $J
echo "SWEEP9995_DONE $(date +%T) n=$(ls $R/lsw/L_*_t0.9995.json | wc -l)" >> $R/sweep9995.log
