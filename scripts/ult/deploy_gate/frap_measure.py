#!/usr/bin/env python3
"""How often would a FRAP spot land where the map saw no filament, before and after split_unsupported? Real VAL frames, deployed polylines (RDP), map from the dump resampled to native by taking the eval frame. usage: frap_measure.py SEG_DIR POLYS.json DUMP_DIR UM_PER_PX"""
import json, os, sys
import numpy as np
SEG, PJ, DUMP, UM = sys.argv[1], sys.argv[2], sys.argv[3], float(sys.argv[4])
sys.path[:0] = [os.path.join(SEG, "models"), SEG]
import frap_select as FS
P = json.load(open(PJ)); UP = 1.5; tot = dict(frames=0, polys=0, pieces=0, cut=0, before=0, before_low=0, after=0, after_low=0); vals = []
for n, polys in sorted(P.items()):
    prob_e = np.load(os.path.join(DUMP, n.replace(".h5", ".npz")))["prob"].astype(np.float32)[0]; H, W = (int(round(s / UP)) for s in prob_e.shape)
    r = (np.arange(H) * UP).round().astype(int).clip(0, prob_e.shape[0] - 1); c = (np.arange(W) * UP).round().astype(int).clip(0, prob_e.shape[1] - 1); prob = prob_e[np.ix_(r, c)]
    pts = [np.asarray(p, float) / UP for p in polys if len(p) >= 2]
    a = FS.select_spots(pts, (H, W), UM); pieces, owners = FS.split_unsupported(pts, prob); b = FS.select_spots(pieces, (H, W), UM)
    p_at = lambda s: float(prob[min(H - 1, int(round(s.y))), min(W - 1, int(round(s.x)))])
    tot["frames"] += 1; tot["polys"] += len(pts); tot["pieces"] += len(pieces); tot["cut"] += len(set(i for i in owners if owners.count(i) > 1 or False))
    tot["before"] += len(a.spots); tot["after"] += len(b.spots); tot["before_low"] += sum(p_at(s) < 0.5 for s in a.spots); tot["after_low"] += sum(p_at(s) < 0.5 for s in b.spots)
print(tot)
