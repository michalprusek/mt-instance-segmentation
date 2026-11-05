#!/bin/bash
# ARM 62 (docs/TODO.md 2026-11-01 07:55): compact debris, recipe calib_V16_debris.json, seeds 20260906-09
# -> DEB_S3..S6 on kajman, started when RING_S3..S6 exit, two per GPU (launch_ult.sh verbatim); eval chains as for every arm; then decomposition at the
# own htw cut, the debris probe on DEB_S3..S6 and AA_S3..S6, and the declared reader read_62.py.
set -u
W=/data2/prusek/mt_work; R=$W/runs; U=$R/ult; L=$U/arm62.log
cd $W
echo "$(date +%F_%T) start; generator $(md5sum < mt34_work/synth/mt_generator.py | cut -c1-8) calib $(md5sum < mt34_work/calib_V16_debris.json | cut -c1-8)" >> $L
wait_pids () { for q in "$@"; do while kill -0 $q 2>/dev/null; do sleep 60; done; done; }
wait_pids $(cat $R/RING_S3.pid) $(cat $R/RING_S4.pid) $(cat $R/RING_S5.pid) $(cat $R/RING_S6.pid)
echo "$(date +%F_%T) RING trainings exited" >> $L
for k in 3 4 5 6; do g=$(( k <= 4 ? 0 : 1 ))
  ./launch_ult.sh DEB_S$k calib_V16_debris.json $((20260903 + k)) $g >> $L 2>&1
  TP=$(cat $R/DEB_S$k.pid)
  nohup $U/chain.sh DEB_S$k $R/DEB_S$k/ep040.pth $g $TP > $U/chain_DEB_S$k.out 2>&1 < /dev/null &
done
for k in 3 4 5 6; do until grep -q ALL_DONE $U/chain_DEB_S$k.log 2>/dev/null; do sleep 120; done; done
echo "$(date +%F_%T) chains done" >> $L
cd $W/mt34_work
export PYTHONPATH=.:src:scripts:synth:third_party OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 TMPDIR=/data2/prusek/tmp
P=/scratch/prusek/envs/nnunet/bin/python
for k in 3 4 5 6; do m=DEB_S$k
  CUT=$($P -c "import json,glob,sys; sys.path.insert(0,'scripts'); from step1_read import source_of
best=None
for f in glob.glob('$U/lsw/L_hb_${m}_t*.json'):
    d=json.load(open(f)); s=[0,0,0]
    for n,g,res in d['rows']:
        if source_of(n)!='htw': continue
        q=res['b60_t0']; s[0]+=q['tp']; s[1]+=q['fp']; s[2]+=q['fn']
    v=2*s[0]/max(2*s[0]+s[1]+s[2],1)
    if best is None or v>best[1]+1e-12 or (abs(v-best[1])<=1e-12 and d['cut']>best[0]): best=(d['cut'],v)
print(best[0])")
  $P decomp_aa.py data/real/mt34_htw_eval $W/preds/${m}_hb_ep40_native $CUT $U/decomp/hb_$m.json htw 24 > $U/decomp/hb_$m.log 2>&1
done
echo "$(date +%F_%T) decomp done" >> $L
for m in DEB_S3 DEB_S4 DEB_S5 DEB_S6 AA_S3 AA_S4 AA_S5 AA_S6; do
  MT_BG_DIR=$W/IRM_backgrounds_v2 CUDA_VISIBLE_DEVICES=0 $P debris_probe.py $R/$m/ep040.pth $U/debrisprobe_$m.json > $U/debrisprobe_$m.log 2>&1
done
echo "$(date +%F_%T) debris probe done" >> $L
md5sum read_62.py > $U/read_62.txt
nice -n 5 /scratch/prusek/envs/mt/bin/python read_62.py >> $U/read_62.txt 2>&1
echo "ARM62_DONE $(date +%T)" >> $U/read_62.txt
