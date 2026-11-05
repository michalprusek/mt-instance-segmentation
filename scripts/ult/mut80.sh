#!/bin/bash
# Mutation test of read_arm80.py's gate: a copy of bridge/ with one frame changed or one cut changed must void the read; the unmutated copy must not.
W=/data2/prusek/mt_work; U=$W/runs/ult; T=/data2/prusek/tmp/mut80_20261102; rm -rf $T; mkdir -p $T/ult/bridge; cd $W/mt34_work
export PYTHONPATH=.:src:scripts:synth:third_party; PY=/scratch/prusek/envs/mt/bin/python
ln -s $U/fold $T/ult/fold; ln -s $U/hyst2 $T/ult/hyst2
for f in roiv2 hball syn96b; do for m in DEB_S3 AA_S3; do cp $U/bridge/${f}_$m.json $T/ult/bridge/; done; done
sed "s#U = W + \"runs/ult/\"#U = \"$T/ult/\"#" read_arm80.py > $T/read_arm80_mut.py
run () { $PY $T/read_arm80_mut.py DEB_S3=AA_S3 > $T/out.txt 2>&1; rc=$?; echo "$1: exit $rc | $(grep -c FAIL $T/out.txt) FAIL lines | $(grep -E 'GATE FAILED|ADOPT' $T/out.txt | cut -c1-60)"; }
run "unmutated"
$PY - <<PY
import json; p="$T/ult/bridge/hball_DEB_S3.json"; d=json.load(open(p)); d["rows"][0][2]["b60"]["fp"]+=1; json.dump(d,open(p,"w"))
PY
run "b60 fp +1 on one frame"
cp $U/bridge/hball_DEB_S3.json $T/ult/bridge/
$PY - <<PY
import json; p="$T/ult/bridge/syn96b_AA_S3.json"; d=json.load(open(p)); d["cut"]=0.5; json.dump(d,open(p,"w"))
PY
run "wrong cut"
cp $U/bridge/syn96b_AA_S3.json $T/ult/bridge/
$PY - <<PY
import json; p="$T/ult/bridge/roiv2_AA_S3.json"; d=json.load(open(p)); d["rows"]=d["rows"][1:]; json.dump(d,open(p,"w"))
PY
run "frame missing"
rm -rf $T
