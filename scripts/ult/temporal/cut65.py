#!/usr/bin/env python3
"""One model's single cut for the video read (docs/TODO.md 2026-11-04 08:10), by the deployment rule of 2026-11-03: its two own VAL cuts, L+H b80, higher block mean of htw and roi303, ties within 0.001 to the higher cut. cwd mt34_work.
usage: cut65.py MODEL OUT_DIR"""
import json, os, subprocess, sys
sys.path.insert(0, "scripts")
from step1_read import source_of
m, O = sys.argv[1:3]; W = "/data2/prusek/mt_work/"; U = W + "runs/ult/"; os.makedirs(O, exist_ok=True); P = "/scratch/prusek/envs/nnunet/bin/python"
own = {b: json.load(open(f"{U}bridge/{f}_{m}.json"))["cut"] for b, f in (("roi303", "roiv2"), ("htw", "hball"))}
f1 = lambda s: 2 * s[0] / max(2 * s[0] + s[1] + s[2], 1)
if own["roi303"] == own["htw"]:
    print(m, "own cuts equal", own, "CUT", own["roi303"]); json.dump(dict(cut=own["roi303"], own=own), open(f"{O}/cut_{m}.json", "w")); sys.exit()
F = {}
for c in sorted(set(own.values())):
    t = {"roi303": [0, 0, 0], "htw": [0, 0, 0]}
    for fn, data, pred in (("roiv2", "data/real/roi303_v2", f"{W}preds/GAUSS_{m}_roiv2_ep40_native"), ("hball", "data/real/mt34_htw_eval", f"{W}preds/{m}_hb_ep40_native")):
        out = f"{O}/{fn}_{m}_t{c}.json"
        subprocess.run([P, "l_bridge.py", data, pred, str(c), out, "80", "12", "val", "all"], check=True, stdout=subprocess.DEVNULL)
        for n, _, r in json.load(open(out))["rows"]:
            q = r["b80"]; k = "roi303" if fn == "roiv2" else ("htw" if source_of(n) == "htw" else None)
            if k: t[k] = [t[k][i] + q[a] for i, a in enumerate(("tp", "fp", "fn"))]
    F[c] = (f1(t["htw"]) + f1(t["roi303"])) / 2
lo, hi = sorted(F); cut = hi if F[hi] - F[lo] >= -0.001 else lo
print(m, "own", own, "block", {k: round(v, 4) for k, v in F.items()}, "CUT", cut); json.dump(dict(cut=cut, own=own, block=F), open(f"{O}/cut_{m}.json", "w"))
