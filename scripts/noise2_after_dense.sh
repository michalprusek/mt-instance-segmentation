#!/bin/bash
# Launch the declared NOISE2 arm (dose escalation, calib_V6_noise2d.json) on GPU 1 the moment the
# DENSE training there has finished (docs/TODO.md 0.0-CAMPAIGN, 2026-09-05). Nothing else changes:
# same recipe, same seed, same 40 x 6000 budget as every arm.
cd /disk2/prusek/mt_work/mt34_work
R=/disk2/prusek/mt_work/runs
echo $$ > $R/noise2_after_dense.pid
while ! grep -q "^DENSE done rc=" $R/DENSE.log 2>/dev/null; do sleep 300; done
sleep 60
echo "$(date) DENSE finished; launching NOISE2 on GPU 1"
setsid nohup bash scripts/run_arm.sh NOISE2 1 --calib calib_V6_noise2d.json > $R/NOISE2.launch.log 2>&1 &
sleep 30; kill -0 $(cat $R/NOISE2.pid) 2>/dev/null && echo "NOISE2 alive" || echo "NOISE2 FAILED TO START"
rm -f $R/noise2_after_dense.pid
