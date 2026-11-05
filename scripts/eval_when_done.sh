#!/bin/bash
# Wait until a training arm has written its "done rc=0" line, then score it with eval_arm.sh.
#
#   scripts/eval_when_done.sh NAME GPU MAX_HOURS [extra eval_v5 flags...]
#
# Polls the arm's log (the launcher appends "NAME done rc=N" when train_v5.py exits), never pgrep.
# A non-zero rc, or the deadline, means nothing is scored.
R=/disk2/prusek/mt_work/runs
NAME=$1; GPU=$2; MAXH=$3; shift 3
deadline=$(( $(date +%s) + MAXH * 3600 ))
while ! grep -q "^$NAME done rc=" $R/$NAME.log 2>/dev/null; do
  [ "$(date +%s)" -gt "$deadline" ] && { echo "eval_when_done: $NAME not finished after ${MAXH}h"; exit 3; }
  sleep 180
done
rc=$(grep "^$NAME done rc=" $R/$NAME.log | tail -1 | sed 's/.*rc=//')
# rc 2 = the trainer's over-firing gate rejected every checkpoint; ep040.pth (the campaign's PRIMARY
# checkpoint rule) exists regardless, so the arm is still scored. Anything else is a crash.
[ "$rc" = "0" ] || [ "$rc" = "2" ] || { echo "eval_when_done: $NAME finished with rc=$rc -- not scoring"; exit 5; }
while [ ! -f /disk2/prusek/mt_work/mt34_work/data/real/roi303_eval/_repair_log.json ]; do
  [ "$(date +%s)" -gt "$deadline" ] && { echo "eval_when_done: roi303 repair never landed"; exit 6; }
  sleep 180
done
exec bash /disk2/prusek/mt_work/mt34_work/scripts/eval_arm.sh $NAME $GPU "$@"
