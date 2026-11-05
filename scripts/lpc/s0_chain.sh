#!/bin/bash
# LPC S0 as declared (docs/TODO.md 2026-10-06 07:55 + addendum 08:20): choose (cut, bridge) on synth_relink96 by the
# written rule, freeze, then the three reads. Nothing here is tuned on real data.
set -u
cd /disk2/prusek/mt_work/mt34_work
P=/scratch/prusek/envs/nnunet/bin/python
export PYTHONPATH=.:src:scripts OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 TMPDIR=/disk2/prusek/tmp
W=/disk2/prusek/mt_work; R=$W/runs/lpc; PD=$W/preds; L=$R/s0_chain.log; echo $$ > $R/s0_chain.pid
echo "$(date +%F_%T) start" >> $L
nice -n 5 $P scripts/lpc/s0_ceiling.py data/synth_relink96 $PD/SPARSE35_relink96_ep40_native $R/s0_select.json all 0.3,0.5,0.7,0.9,0.98 12,24,36,60 12 > $R/s0_select.txt 2>&1
echo "$(date +%F_%T) select rc=$?" >> $L
$P - > $R/s0_choice.txt <<'PY'
import json, numpy as np
R = json.load(open("/disk2/prusek/mt_work/runs/lpc/s0_select.json"))["rows"]
f1 = lambda t: 2 * t[0] / max(2 * t[0] + t[1] + t[2], 1); T = {}
for vk in R[0][2]:
    for k in R[0][2][vk]:
        if k.startswith("br"): T[(float(vk[1:]), float(k[2:]))] = f1(np.sum([r[2][vk][k]["s"] for r in R], 0))
best = max(T.values()); ok = [k for k, v in T.items() if v >= best - 0.005]
cut = max(c for c, b in ok); br = min(b for c, b in ok if c == cut)
print("%g %g" % (cut, br)); print("best", best, "chosen F1", T[(cut, br)], "edge" if br == 60 else "")
PY
read CUT BR < $R/s0_choice.txt; echo "$(date +%F_%T) CHOICE cut=$CUT bridge=$BR" >> $L
nice -n 5 $P scripts/lpc/s0_ceiling.py data/synth_relink96b BAND $R/s0_band96b.json all 0 $BR 12 > $R/s0_band96b.txt 2>&1; echo "$(date +%F_%T) band96b rc=$?" >> $L
nice -n 5 $P scripts/lpc/s0_ceiling.py data/synth_relink96b $PD/SPARSE35_relink96b_ep40_native $R/s0_map96b.json all $CUT,0.98 $BR 12 > $R/s0_map96b.txt 2>&1; echo "$(date +%F_%T) map96b rc=$?" >> $L
nice -n 5 $P scripts/lpc/s0_ceiling.py $W/runs/oct05/htw_only $PD/SPARSE35_hb_ep40_native $R/s0_htwVAL.json val $CUT,0.995 $BR 12 > $R/s0_htwVAL.txt 2>&1; echo "$(date +%F_%T) htwVAL rc=$?" >> $L
nice -n 5 $P scripts/lpc/s0_ceiling.py data/real/roi303_v2 $PD/GAUSS_SPARSE35_roiv2_ep40_native $R/s0_roiv2VAL.json val $CUT,0.99 $BR 12 > $R/s0_roiv2VAL.txt 2>&1; echo "$(date +%F_%T) roiv2VAL rc=$?" >> $L
echo "$(date +%F_%T) S0_ALL_DONE" >> $L
