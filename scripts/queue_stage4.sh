#!/bin/bash
# Third seed of the baseline recipe (S35_SEED3, declared 2026-09-05 23:20): scored on the fast grid
# after stage 3 has finished, so the resolution of the primary metric rests on three seeds, not two.
cd /disk2/prusek/mt_work/mt34_work
R=/disk2/prusek/mt_work/runs
echo $$ > $R/queue_stage4.pid
# stage 3's log landed under mt34_work/runs (relative $R at its launch); accept either location
while ! grep -q "STAGE3_DONE" $R/queue_stage3.log /disk2/prusek/mt_work/mt34_work/runs/queue_stage3.log 2>/dev/null; do sleep 300; done
FAST=0.90,0.94,0.96,0.97,0.98,0.985,0.99,0.993,0.995,0.997,0.998,0.999,0.9995
# 23:40: NOISE_S2 (second seed of NOISE, GPU 0) scored after S35_SEED3; each waits for its training
echo "$(date +%H:%M) stage 4: fast grid S35_SEED3 NOISE_S2"
GRID=$FAST NPAR=${NPAR:-13} EVAL_GPU=${EVAL_GPU:-1} bash scripts/eval_queue.sh S35_SEED3 NOISE_S2 > $R/eval_queue_stage4.log 2>&1
rm -f $R/queue_stage4.pid
echo "$(date +%H:%M) STAGE4_DONE"
