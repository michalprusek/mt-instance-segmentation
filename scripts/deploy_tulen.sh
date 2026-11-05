#!/bin/bash
# Copy files to BOTH tulen trees and VERIFY by md5 that they arrived.
#
# There are two trees on tulen and they are NOT the same checkout:
#   /disk2/prusek/mt_work            -- 61 scripts, no third_party
#   /disk2/prusek/mt_work/mt34_work  -- 86 scripts, third_party, all of data/
# The training launcher does `cd mt34_work` and sets PYTHONPATH=.:src:scripts:synth:third_party
# RELATIVE TO THAT DIRECTORY, so mt34_work is the tree that actually trains, while several
# analysis scripts have been run from the outer one. Copying to only one of them silently trains
# or scores with stale code -- it has already cost one launch this project.
#
# Usage:  scripts/deploy_tulen.sh synth/mt_generator.py scripts/train_v5.py ...
set -euo pipefail
H=prusek@tulen.utia.cas.cz
T1=/disk2/prusek/mt_work
T2=/disk2/prusek/mt_work/mt34_work
[ $# -gt 0 ] || { echo "usage: $0 <repo-relative file> ..." >&2; exit 2; }

for f in "$@"; do
  [ -f "$f" ] || { echo "MISSING LOCALLY: $f" >&2; exit 1; }
  for T in "$T1" "$T2"; do
    ssh -o BatchMode=yes "$H" "mkdir -p $T/$(dirname "$f")"
    # copy to a temp name and rename: bash reads a running script by file OFFSET, so overwriting a
    # script in place while tulen executes it corrupts that run (it happened 2026-09-05 03:30 to the
    # baseline 1.5x sweep). A rename gives the running process the old inode and new runs the new one.
    scp -o BatchMode=yes -q "$f" "$H:$T/$f.deploying" && ssh -o BatchMode=yes "$H" "mv -f $T/$f.deploying $T/$f"
  done
done

# verify: a copy that silently did not land is the failure mode this exists to catch
LOCAL=$(md5sum "$@" | awk '{print $1" "$2}')
REMOTE=$(ssh -o BatchMode=yes "$H" "cd $T1 && md5sum $* ; cd $T2 && md5sum $*")
fail=0
while read -r sum name; do
  n=$(echo "$REMOTE" | grep -c "^$sum  $name$" || true)
  if [ "$n" -ne 2 ]; then echo "MISMATCH: $name landed correctly in $n/2 trees"; fail=1
  else echo "ok (both trees): $name  ${sum:0:10}"; fi
done <<< "$LOCAL"
exit $fail
