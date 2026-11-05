#!/bin/bash
# When the running evaluation queue has finished AUG_PHOTO, stop it and restart it in TWO PASSES:
# pass 1 sweeps only the 13 FAST thresholds (>= 0.90, where every arm's optimum lies; ~25 min per
# row instead of ~90) for SCALE, NOISE, HEAT, DENSE so each can be read hours earlier; pass 2 later
# re-runs the same queue with the full 19-point grid, which reuses every part already on disk and
# only adds the slow 0.30-0.80 thresholds (needed for the per-frame threshold-spread check).
cd /disk2/prusek/mt_work/mt34_work
R=/disk2/prusek/mt_work/runs
echo $$ > $R/requeue_fast.pid
killtree () { for c in $(pgrep -P $1); do killtree $c; done; kill $1 2>/dev/null; }
while ! grep -q "EVAL_DONE AUG_PHOTO" $R/eval_queue.log 2>/dev/null; do sleep 120; done
echo "$(date +%H:%M) AUG_PHOTO scored; stopping the full-grid queue"
q=$(cat $R/eval_queue.pid 2>/dev/null); [ -n "$q" ] && killtree $q; sleep 3
FAST=0.90,0.94,0.96,0.97,0.98,0.985,0.99,0.993,0.995,0.997,0.998,0.999,0.9995
GRID=$FAST NPAR=13 setsid nohup bash scripts/eval_queue.sh NOISE SCALE "HEAT:--prob-channel 0" DENSE > $R/eval_queue_fast.log 2>&1 &
echo $! > $R/eval_queue.pid
echo "$(date +%H:%M) fast-grid queue launched pid $(cat $R/eval_queue.pid)"
# pass 2: the full grid, after the fast pass has finished (QUEUE_DONE); parts are reused
while ! grep -q "QUEUE_DONE" $R/eval_queue_fast.log 2>/dev/null; do sleep 300; done
NPAR=12 setsid nohup bash scripts/eval_queue.sh NOISE SCALE "HEAT:--prob-channel 0" DENSE > $R/eval_queue_full.log 2>&1 &
echo $! > $R/eval_queue.pid
echo "$(date +%H:%M) full-grid pass launched pid $(cat $R/eval_queue.pid)"
rm -f $R/requeue_fast.pid
