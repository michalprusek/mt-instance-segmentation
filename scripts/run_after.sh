#!/bin/bash
# Run a command once the process in a PIDFILE has exited.
#
#   scripts/run_after.sh PIDFILE MAX_HOURS command args...
#
# Waits on the PID in the file with `kill -0` -- NOT `pgrep -f`, which matches the asking shell
# and has already stalled this project for 36 h once (docs/TODO.md 2026-09-02). A missing pidfile
# counts as "already finished". Gives up after MAX_HOURS and launches nothing.
PIDFILE=$1; MAXH=$2; shift 2
deadline=$(( $(date +%s) + MAXH * 3600 ))
while [ -f "$PIDFILE" ] && kill -0 "$(cat "$PIDFILE")" 2>/dev/null; do
  if [ "$(date +%s)" -gt "$deadline" ]; then
    echo "run_after: gave up waiting on $PIDFILE after ${MAXH}h; NOT launching: $*"
    exit 3
  fi
  sleep 120
done
echo "run_after: $PIDFILE finished at $(date); launching: $*"
exec "$@"
