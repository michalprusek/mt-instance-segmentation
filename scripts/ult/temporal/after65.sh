#!/bin/bash
# R2 and R3 inference (docs/TODO.md 2026-11-04 08:10), once ARM 65's chains and folds are done: each model's cut, then real windows and clips. No read here.
W=/data2/prusek/mt_work; U=$W/runs/ult; T=$U/temporal; O=$T/r2; mkdir -p $O; P=/scratch/prusek/envs/nnunet/bin/python
A="TMP_S1 TMP_S2 TMP_S3 TMP_S4 TMP_S5 TMP_S6"; C="DEB_S1 DEB_S2 DEB_S3 DEB_S4 DEB_S5 DEB_S6"
until grep -q ARM65_DONE $U/read_65.txt 2>/dev/null; do sleep 300; done
cd $W/mt34_work; export PYTHONPATH=.:src:scripts:synth:third_party OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 TMPDIR=/data2/prusek/tmp
for m in $A $C; do $P $T/cut65.py $m $O/cuts >> $O/cuts.log 2>&1 || { echo CUT_FAILED $m > $T/out/r2_marker; exit 1; }; done
cd $T; unset PYTHONPATH; PKG=$T/hub/backend/segmentation/models/microtubule; : > $O/jobs.txt
for m in $A $C; do
  mkdir -p weights65/$m; ln -sfn $W/runs/$m/ep040.pth weights65/$m/microtubule_v16_deb_s4_ep040.pth; c=$(python3 -c "import json;print(json.load(open('$O/cuts/cut_$m.json'))['cut'])")
  case $m in TMP*) echo "$m $c temporal real frames list.json" >> $O/jobs.txt; echo "$m $c temporal synth clips" >> $O/jobs.txt; echo "$m $c single synth clips" >> $O/jobs.txt;;
             *)    echo "$m $c single real frames list.json" >> $O/jobs.txt;   echo "$m $c single synth clips" >> $O/jobs.txt;; esac
done
one () { m=$2; c=$3; mode=$4; set=$5; g=$1; shift 5; f=$O/${m}_${mode}_$set; [ -f $f.done ] && return
  CUDA_VISIBLE_DEVICES=$g OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 TMPDIR=/data2/prusek/tmp nice -n 5 $P tmp_infer.py $PKG weights65/$m/microtubule_v16_deb_s4_ep040.pth geom $f.json $c $mode $set "$@" > $f.log 2>&1 && grep -q TMP_INFER_DONE $f.log && touch $f.done; }
export -f one; export O P PKG
awk 'NR%2==0' $O/jobs.txt | xargs -P 4 -L 1 bash -c 'one 0 "$@"' _ &
awk 'NR%2==1' $O/jobs.txt | xargs -P 4 -L 1 bash -c 'one 1 "$@"' _ &
wait
n=$(wc -l < $O/jobs.txt); d=$(ls $O/*.done 2>/dev/null | wc -l); [ "$n" = "$d" ] && echo "R2_OK $d" > $T/out/r2_marker || echo "R2_FAILED $d of $n" > $T/out/r2_marker
