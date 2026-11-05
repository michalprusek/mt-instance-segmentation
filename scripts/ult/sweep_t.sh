#!/bin/bash
# L at one extra cut on existing dumps, same a_grid.py call as chain.sh. usage: sweep_t.sh CUT MODEL...
T=$1; shift
cd /data2/prusek/mt_work/mt34_work
P=/scratch/prusek/envs/nnunet/bin/python
export PYTHONPATH=.:src:scripts:synth:third_party OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 TMPDIR=/data2/prusek/tmp LPC_SKIP_A=1
W=/data2/prusek/mt_work; R=$W/runs/ult; PD=$W/preds; V2=data/real/roi303_v2; HB=data/real/mt34_htw_eval; SY=data/synth_relink96b
for A in "$@"; do
  for j in "$PD/${A}_hb_ep40_native $HB hb_$A" "$PD/GAUSS_${A}_roiv2_ep40_native $V2 roiv2_$A" "$PD/GAUSS_${A}_syn96b_ep40_native $SY syn96b_$A"; do echo "$j $T"; done
done > $R/jobs_sweep_$T.txt
one () { O=$R/lsw/L_$3_t$4; [ -f $O.json ] && return
  nice -n 5 $P scripts/lpc/a_grid.py $2 $1 $4 $O.json src/instance/params_l_hand.json 60 0 4 val > $O.log 2>&1; }
export -f one; export P R
xargs -P 12 -L 1 bash -c "one \"\$@\"" _ < $R/jobs_sweep_$T.txt
