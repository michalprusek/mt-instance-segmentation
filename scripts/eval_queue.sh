#!/bin/bash
# The campaign's evaluation queue, sequential so the CPU is never flooded: baseline re-sweeps on
# the repaired block from cached maps, then every arm as its training finishes (polls the arm
# log for "NAME done rc=", accepts rc 0 or 2, 40 h cap each). Appends "EVAL_DONE NAME" lines to
# runs/eval_queue.log. Liveness: runs/eval_queue.pid + kill -0.
#
#   scripts/eval_queue.sh ARM[:extra flags] ...     e.g. AUG_PHOTO SCALE NOISE "HEAT:--prob-channel 0" DENSE
cd /disk2/prusek/mt_work/mt34_work
R=/disk2/prusek/mt_work/runs; P=/disk2/prusek/mt_work/preds; S=$R/scale
echo $$ > $R/eval_queue.pid
if [ ! -f $S/thr_SPARSE35_native_rep.json ]; then
  bash scripts/sweep_parallel.sh $P/SPARSE35_roi_native $S/thr_SPARSE35_native_rep.json > $S/sweep_SPARSE35_native_rep.log 2>&1
  echo "$(date +%H:%M) baseline native rep rc=$? $(tail -1 $S/sweep_SPARSE35_native_rep.log)"
fi
if [ ! -f $S/thr_SPARSE35_rep.json ]; then
  bash scripts/sweep_parallel.sh $P/SPARSE35_roi $S/thr_SPARSE35_rep.json > $S/sweep_SPARSE35_rep.log 2>&1
  echo "$(date +%H:%M) baseline 1.5x rep rc=$? $(tail -1 $S/sweep_SPARSE35_rep.log)"
fi
echo "BASELINE_REP_DONE"
for spec in "$@"; do
  NAME=${spec%%:*}; FLAGS=""; [ "$spec" != "$NAME" ] && FLAGS=${spec#*:}
  deadline=$(( $(date +%s) + 40 * 3600 ))
  while ! grep -q "^$NAME done rc=" $R/$NAME.log 2>/dev/null; do
    [ "$(date +%s)" -gt "$deadline" ] && { echo "eval_queue: $NAME not finished in 40 h -- skipping"; continue 2; }
    sleep 300
  done
  rc=$(grep "^$NAME done rc=" $R/$NAME.log | tail -1 | sed 's/.*rc=//')
  [ "$rc" = "0" ] || [ "$rc" = "2" ] || { echo "eval_queue: $NAME finished with rc=$rc -- not scoring"; continue; }
  bash scripts/eval_arm2.sh $NAME ${EVAL_GPU:-1} $FLAGS
done
rm -f $R/eval_queue.pid
echo "QUEUE_DONE"
