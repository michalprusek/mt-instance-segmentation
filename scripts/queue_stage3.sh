#!/bin/bash
# Evaluation for the two arms that have no path yet (2026-09-05 18:50): the resolution row
# S35_SEED2 (training ends ~19:00) and NOISE2 (ends ~06:00 next day). Runs after the fast-grid
# queue has finished (QUEUE_DONE in eval_queue_fast.log) and REPLACES requeue_fast.sh's pass 2
# (which is killed here so the two passes cannot overlap), in this order:
#   A  fast grid  S35_SEED2   -> the campaign's seed-to-seed resolution, read first
#   C  fast grid  NOISE2      -> waits for its training inside eval_queue.sh
#   B  full grid  S35_SEED2   (other arms' full grids by hand, only where the fast verdict is not null)
cd /disk2/prusek/mt_work/mt34_work
R=/disk2/prusek/mt_work/runs
echo $$ > $R/queue_stage3.pid
killtree () { for c in $(pgrep -P $1); do killtree $c; done; kill $1 2>/dev/null; }
# 23:30: PW2S and NOISE2 are both trained and GPU 0 is idle, while the resumed fast queue still has
# SCALE / HEAT / DENSE (~9 h) ahead of QUEUE_DONE. NOWAIT=1 runs this stage NOW on GPU 0 alongside
# the fast queue (distinct arm names -> distinct files; NPAR lowered so the two sweeps share the CPU).
if [ "${NOWAIT:-0}" != 1 ]; then
  while ! grep -q "QUEUE_DONE" $R/eval_queue_fast.log 2>/dev/null; do sleep 300; done
fi
q=$(cat $R/requeue_fast.pid 2>/dev/null); [ -n "$q" ] && kill -0 $q 2>/dev/null && { killtree $q; echo "$(date +%H:%M) requeue_fast pass 2 cancelled (pid $q)"; }
sleep 5
FAST=0.90,0.94,0.96,0.97,0.98,0.985,0.99,0.993,0.995,0.997,0.998,0.999,0.9995
# Order revised 19:00 (advisor): the fast reads of the two new arms come BEFORE any full-grid pass,
# so NOISE2 (training ends ~06:00) is not queued behind ~20 h of slow thresholds whose only consumer
# is the threshold-spread check; the full grid runs last and only for the baseline replicate and
# any arm whose fast verdict is not null (decided then, by hand).
# 19:30: PW2S (loss arm, GPU 0 after S35_SEED2, ends ~02:00) inserted before NOISE2 (ends ~06:00);
# eval_queue.sh waits for each training in turn, so the order is the order the reads arrive in.
echo "$(date +%H:%M) stage A: fast grid PW2S NOISE2 (S35_SEED2 is in the fast queue) (each waits for its training)"
GRID=$FAST NPAR=${NPAR:-13} EVAL_GPU=${EVAL_GPU:-1} bash scripts/eval_queue.sh PW2S NOISE2 > $R/eval_queue_stageA.log 2>&1
echo "$(date +%H:%M) stage B: full grid S35_SEED2 only (other arms by hand if their fast verdict is not null)"
NPAR=${NPAR:-12} EVAL_GPU=${EVAL_GPU:-1} bash scripts/eval_queue.sh S35_SEED2 > $R/eval_queue_full.log 2>&1
rm -f $R/queue_stage3.pid
echo "$(date +%H:%M) STAGE3_DONE"
