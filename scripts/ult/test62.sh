#!/bin/bash
# The declared single TEST read after ARM 62 ADOPT + the fold-in (docs/TODO.md 2026-11-02; user 06:36 "2. - ano"). DEB_S3..S6
# dumped on TEST with the VAL chain's flags (ep040, native, htw block uniform, roi303 v2 gauss); L and L+H (bridge 0.98, R
# 22.5) at each model's own VAL L cut, on DEB_S3..S6 and on the existing AA_S3..S6 TEST dumps; then read_test62.py.
# usage: test62.sh CAND   (CAND = H -> V16 + L+H is the candidate; L -> V16 + L)
set -u
CAND=$1
W=/data2/prusek/mt_work; U=$W/runs/ult; O=$U/test62; PD=$W/preds; mkdir -p $O; L=$O/driver.log
cd $W/mt34_work
P=/scratch/prusek/envs/nnunet/bin/python; DER=src/instance/params_a_derived.json; V2=data/real/roi303_v2; HB=data/real/mt34_htw_eval
export PYTHONPATH=.:src:scripts:synth:third_party TMPDIR=/data2/prusek/tmp
echo "$(date +%F_%T) start cand $CAND" >> $L
dump () {  # model gpu
  A=$1; G=$2; CK=$W/runs/$A/ep040.pth; md5sum $CK >> $L
  OMP_NUM_THREADS=2 CUDA_VISIBLE_DEVICES=$G $P scripts/eval_v5.py --ckpt $CK --arch nnunet --data $HB --split test --include-empty --no-fov --params $DER --cov 0.80 --thrs 0.98 --infer-scale 1.0 --fusion uniform --dump-pred $PD/TEST_${A}_hb_ep40_native --out-json $O/dump_hb_$A.json > $O/dump_hb_$A.log 2>&1
  echo "$(date +%F_%T) dump hb $A exit $? n=$(ls $PD/TEST_${A}_hb_ep40_native | wc -l)" >> $L
  OMP_NUM_THREADS=2 CUDA_VISIBLE_DEVICES=$G $P scripts/eval_v5.py --ckpt $CK --arch nnunet --data $V2 --split test --include-empty --no-fov --params $DER --cov 0.80 --thrs 0.98 --infer-scale 1.0 --fusion gauss --dump-pred $PD/TEST_GAUSS_${A}_roiv2_ep40_native --out-json $O/dump_roiv2_$A.json > $O/dump_roiv2_$A.log 2>&1
  echo "$(date +%F_%T) dump roiv2 $A exit $? n=$(ls $PD/TEST_GAUSS_${A}_roiv2_ep40_native | wc -l)" >> $L
}
dump DEB_S3 0 & dump DEB_S4 0 & dump DEB_S5 1 & dump DEB_S6 1 & wait
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 LPC_SKIP_A=1
G=L,bridge:0.98:22.5
for mc in AA_S3:0.995 AA_S4:0.98 AA_S5:0.995 AA_S6:0.995 DEB_S3:0.995 DEB_S4:0.99 DEB_S5:0.99 DEB_S6:0.995; do m=${mc%%:*}; c=${mc##*:}
  $P l_hyst2.py $HB $PD/TEST_${m}_hb_ep40_native $c $O/hb_$m.json $G 33 test htw > $O/hb_$m.log 2>&1
  $P l_hyst2.py $V2 $PD/TEST_GAUSS_${m}_roiv2_ep40_native 0.995 $O/roiv2_$m.json $G 40 test all > $O/roiv2_$m.log 2>&1
  echo "$(date +%F_%T) decoded $m" >> $L
done
md5sum read_test62.py > $U/read_test62.txt
nice -n 5 /scratch/prusek/envs/mt/bin/python read_test62.py $CAND >> $U/read_test62.txt 2>&1
echo "TEST62_DONE $(date +%T)" >> $U/read_test62.txt
