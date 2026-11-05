#!/bin/bash
# ARM 65 TEMPORAL INPUT, docs/TODO.md 2026-11-04 08:10. Seeds of DEB_S1..S6, three runs per GPU, launch_ult.sh verbatim + the flag, chains; cut 0.9995 and folds as ARM 64; then Gaussian dumps of the hb block for MT-34, TMP and DEB, at b80.
set -u
W=/data2/prusek/mt_work; R=$W/runs; U=$R/ult; L=$U/arm65.log; A="TMP_S1 TMP_S2 TMP_S3 TMP_S4 TMP_S5 TMP_S6"; C="DEB_S1 DEB_S2 DEB_S3 DEB_S4 DEB_S5 DEB_S6"
cd $W
echo "$(date +%F_%T) start; generator $(md5sum < mt34_work/synth/mt_generator.py | cut -c1-8) clip $(md5sum < mt34_work/synth/mt_clip.py | cut -c1-8) trainer $(md5sum < mt34_work/scripts/train_v5.py | cut -c1-8) V16 $(md5sum < mt34_work/calib_V16_debris.json | cut -c1-8)" >> $L
go () { # name seed gpu
  ./launch_ult.sh $1 calib_V16_debris.json $2 $3 --temporal-p-single 0.35 >> $L 2>&1; nohup $U/chain.sh $1 $R/$1/ep040.pth $3 $(cat $R/$1.pid) > $U/chain_$1.out 2>&1 < /dev/null &
}
go TMP_S1 20260827 0; go TMP_S2 20260905 0; go TMP_S3 20260906 0
go TMP_S4 20260907 1; go TMP_S5 20260908 1; go TMP_S6 20260909 1
for m in $A; do until grep -q ALL_DONE $U/chain_$m.log 2>/dev/null; do sleep 120; done; done
echo "$(date +%F_%T) chains done" >> $L
bash $U/sweep_t.sh 0.9995 $A >> $L 2>&1
for m in $A; do bash $U/fold_one.sh $m >> $L 2>&1; bash $U/fold80.sh $m >> $L 2>&1; done
cd $W/mt34_work; export PYTHONPATH=.:src:scripts:synth:third_party OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 TMPDIR=/data2/prusek/tmp; P=/scratch/prusek/envs/nnunet/bin/python; PD=$W/preds; HB=data/real/mt34_htw_eval; DER=src/instance/params_a_derived.json
mkdir -p $U/mt34g
for m in $C $A; do
  O=$PD/GAUSS_${m}_hb_ep40_native
  if [ ! -d $O ] || [ "$(ls $O | wc -l)" -eq 0 ]; then CUDA_VISIBLE_DEVICES=0 $P scripts/eval_v5.py --ckpt $R/$m/ep040.pth --arch nnunet --data $HB --split val --include-empty --no-fov --params $DER --cov 0.80 --thrs 0.98 --infer-scale 1.0 --fusion gauss --dump-pred $O --out-json $U/mt34g/dump_$m.json > $U/mt34g/dump_$m.log 2>&1; fi
  c=$($P -c "import json; print(json.load(open('$U/bridge/hball_$m.json'))['cut'])")
  nice -n 5 $P l_bridge.py $HB $O $c $U/mt34g/hball_$m.json 60,80 16 val all > $U/mt34g/hball_$m.log 2>&1
done
md5sum read_arm80.py > $U/read_65.txt
$P read_arm80.py TMP_S1=DEB_S1,TMP_S2=DEB_S2,TMP_S3=DEB_S3,TMP_S4=DEB_S4,TMP_S5=DEB_S5,TMP_S6=DEB_S6 $(echo $C | tr ' ' ',') >> $U/read_65.txt 2>&1
echo "ARM65_DONE $(date +%T)" >> $U/read_65.txt
