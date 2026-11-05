#!/bin/bash
# Launch the declared LOSS arm PW2S (an August run named PW2 already exists on tulen; docs/TODO.md 0.0-CAMPAIGN, 2026-09-05 19:30: the SPARSE35 recipe
# with --pos-weight 2 instead of the inherited 8, nothing else changed) on GPU 0 the moment the
# resolution row S35_SEED2 has finished there.
cd /disk2/prusek/mt_work/mt34_work
R=/disk2/prusek/mt_work/runs
echo $$ > $R/pw2_after_seed2.pid
while ! grep -q "^S35_SEED2 done rc=" $R/S35_SEED2.log 2>/dev/null; do sleep 300; done
sleep 60
echo "$(date) S35_SEED2 finished; launching PW2 on GPU 0"
setsid nohup bash scripts/run_arm.sh PW2S 0 --pos-weight 2 > $R/PW2S.launch.log 2>&1 &
sleep 30; kill -0 $(cat $R/PW2S.pid) 2>/dev/null && echo "PW2S alive" || echo "PW2S FAILED TO START"
rm -f $R/pw2_after_seed2.pid
