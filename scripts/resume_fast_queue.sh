#!/bin/bash
# 2026-09-05 19:05: the fast-grid queue (eval_queue.sh pid 6305) was killed by mistake together with
# the idle requeue_fast waiter (it was that waiter's child). Its current job, eval_arm2.sh NOISE
# (pid $ORPHAN), survived as an orphan and finishes NOISE on its own. This script waits for that
# orphan, writes the EVAL_DONE line the dead queue would have written, and continues the queue for
# the remaining arms on the same fast grid, APPENDING to the same log so the waiters that grep it
# (heat_screen_waiter.sh, queue_stage3.sh) see the lines they wait for.
cd /disk2/prusek/mt_work/mt34_work
R=/disk2/prusek/mt_work/runs
ORPHAN=${1:?orphan pid}
echo $$ > $R/resume_fast_queue.pid
while kill -0 $ORPHAN 2>/dev/null; do sleep 120; done
echo "$(date +%H:%M) EVAL_DONE NOISE (orphaned eval_arm2 finished; queue resumed)" >> $R/eval_queue_fast.log
FAST=0.90,0.94,0.96,0.97,0.98,0.985,0.99,0.993,0.995,0.997,0.998,0.999,0.9995
GRID=$FAST NPAR=13 bash scripts/eval_queue.sh S35_SEED2 SCALE "HEAT:--prob-channel 0" DENSE >> $R/eval_queue_fast.log 2>&1
rm -f $R/resume_fast_queue.pid
