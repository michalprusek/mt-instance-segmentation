#!/bin/bash
# The NOISE replicate's pre-registered replication reads (docs/TODO.md 23:40 entry): the moment stage 4
# has dumped NOISE_S2's native maps, run (a) the per-filament coverage instrument, (b) the anatomy and
# covered_split at 0.99 (16-bit falsifier + fragments/missed), on the cached dump. The combined-VAL row
# comes from stage 4's HTW dump. CPU only; nothing touches TEST.
cd /disk2/prusek/mt_work/mt34_work
export PYTHONPATH=.:src:scripts:synth:third_party
PY=/scratch/prusek/envs/nnunet/bin/python
R=/disk2/prusek/mt_work/runs; P=/disk2/prusek/mt_work/preds; D=$P/NOISE_S2_roi_ep40_native
echo $$ > $R/noise_s2_reads.pid
while ! grep -q "NOISE_S2_ep40_native dump rc=0" $R/eval_queue_stage4.log 2>/dev/null; do sleep 300; done
echo "$(date +%H:%M) NOISE_S2 dump present ($(ls $D/*.npz | wc -l) maps); launching the replication reads"
A=runs/diag_roi/anat; C=runs/diag_roi/csplit
nice -n 15 $PY scripts/diag_filament_contrast.py --pred-dir $D --thr 0.98 --exclude roi_img_8,roi_img_149 --calib calib_V6_c070_narrow.json --n-synth 0 --out runs/diag_roi/contrast_NOISE_S2.json > runs/diag_roi/contrast_NOISE_S2.log 2>&1 &
for k in 0 1 2 3; do
  nice -n 15 $PY scripts/diag_fp_anatomy.py --pred-dir $D --thr 0.99 --exclude roi_img_8,roi_img_149 --frames "$(cat runs/diag_roi/fpnat/chunk_$k.txt)" --out $A/NOISE_S2_roi_ep40_native_$k.json > $A/NOISE_S2_roi_ep40_native_$k.log 2>&1 &
  nice -n 15 $PY scripts/diag_covered_split.py --pred-dir $D --thr 0.99 --frames "$(cat runs/diag_roi/fpnat/chunk_$k.txt)" --out $C/NOISE_S2_roi_ep40_native_$k.json > $C/NOISE_S2_roi_ep40_native_$k.log 2>&1 &
done
wait
rm -f $R/noise_s2_reads.pid
echo "$(date +%H:%M) NOISE_S2_READS_DONE"
