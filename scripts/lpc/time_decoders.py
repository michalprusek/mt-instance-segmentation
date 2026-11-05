#!/usr/bin/env python3
"""Decode time per frame, instancer A vs instancer L-hand, one thread, cached maps (no metric is computed or read).
    time_decoders.py DATA PRED_DIR CUT [N]"""
import glob, json, os, sys, time
import numpy as np
sys.path[:0] = [".", "src", "scripts", "scripts/lpc"]
from lpc_lib import P, KAPPA, decode_links, link_edges
from instance.instancer_a import instance_a
from mt_bench.cvat_import import read_frame_h5
data, pred, cut = sys.argv[1], sys.argv[2], float(sys.argv[3]); N = int(sys.argv[4]) if len(sys.argv) > 4 else 24
M = json.load(open("src/instance/params_l_hand.json"))
paths = [p for p in sorted(glob.glob(os.path.join(data, "*.h5"))) if str(read_frame_h5(p)["attrs"].get("split")) == "val"
         and os.path.exists(os.path.join(pred, os.path.basename(p).replace(".h5", ".npz")))]
paths = paths[::max(1, len(paths) // N)][:N]; tA, tL, nA, nL, nE, px = [], [], [], [], [], []
for p in paths:
    prob = np.load(os.path.join(pred, os.path.basename(p).replace(".h5", ".npz")))["prob"].astype(np.float32)[0]
    t = time.perf_counter(); _, mA = instance_a(prob > cut, KAPPA, params=P, channels=prob[None], prob=prob); tA.append(time.perf_counter() - t)
    t = time.perf_counter(); mL = decode_links(prob, cut, M, 60.0, 0.0); tL.append(time.perf_counter() - t)
    nA.append(len(mA)); nL.append(len(mL)); nE.append(len(link_edges(prob, cut, M, 60.0)[1])); px.append(prob.size)
q = lambda v: "median %.2f s  mean %.2f s  max %.2f s" % (np.median(v), np.mean(v), np.max(v))
print(f"{data} cut {cut}: {len(paths)} frames, {np.mean(px)/1e6:.2f} Mpx (eval frame), instances/frame A {np.mean(nA):.0f} L {np.mean(nL):.0f}, candidate links/frame {np.mean(nE):.0f}")
print("  A:", q(tA)); print("  L:", q(tL)); print("  ratio of means L/A: %.2f" % (np.mean(tL) / np.mean(tA)))
