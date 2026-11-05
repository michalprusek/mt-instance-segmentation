#!/bin/bash
# L+H at match reach 60 and 80 (l_bridge.py) at a model's own VAL L cuts (runs/ult/lsw), on roi303 v2, the full hb block and synth 96b. usage: fold80.sh MODEL
m=$1; W=/data2/prusek/mt_work; U=$W/runs/ult; PD=$W/preds; O=$U/bridge; mkdir -p $O
cd $W/mt34_work
export PYTHONPATH=.:src:scripts:synth:third_party OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 TMPDIR=/data2/prusek/tmp
P="nice -n 5 /scratch/prusek/envs/nnunet/bin/python"
cut () { /scratch/prusek/envs/nnunet/bin/python -c "import json,glob,sys; sys.path.insert(0,'scripts'); from step1_read import source_of
best=None
for f in glob.glob('$U/lsw/L_$1_${m}_t*.json'):
    d=json.load(open(f)); s=[0,0,0]
    for n,g,res in d['rows']:
        if '$1'=='hb' and source_of(n)!='htw': continue
        q=res['b60_t0']; s[0]+=q['tp']; s[1]+=q['fp']; s[2]+=q['fn']
    v=2*s[0]/max(2*s[0]+s[1]+s[2],1)
    if best is None or v>best[1]+1e-12 or (abs(v-best[1])<=1e-12 and d['cut']>best[0]): best=(d['cut'],v)
print(best[0])"; }
$P l_bridge.py data/real/roi303_v2 $PD/GAUSS_${m}_roiv2_ep40_native $(cut roiv2) $O/roiv2_$m.json 60,80 16 val all > $O/roiv2_$m.log 2>&1
$P l_bridge.py data/real/mt34_htw_eval $PD/${m}_hb_ep40_native $(cut hb) $O/hball_$m.json 60,80 16 val all > $O/hball_$m.log 2>&1
$P l_bridge.py data/synth_relink96b $PD/GAUSS_${m}_syn96b_ep40_native $(cut syn96b) $O/syn96b_$m.json 60,80 16 val all > $O/syn96b_$m.log 2>&1
echo "$(date +%F_%T) fold80 $m done"
