#!/usr/bin/env python3
"""What did the debris arm remove? (docs/TODO.md 2026-11-02 07:00 (C)). The 197 AA_S3 htw off-annotation predictions labelled by
eye on 2026-10-31 (labels.json; same rule and order as offgt_montage.py, gate: frame + length). A tile SURVIVES in a model if
>= 50 % of its 1-px samples lie within 5 eval px of that model's predictions. usage: offgt_survive.py LABELS.json"""
import glob, json, os, sys, collections
import numpy as np
from multiprocessing import Pool
from scipy.spatial import cKDTree
sys.path[:0] = [".", "src", "scripts", "scripts/lpc"]
from lpc_lib import UP, P, graph, to_logit, resample, rasterise_polyline
from l_dup import match, walk_all, arclen
from l_hyst2 import hmask
from instance.metrics import _polyline_pixels, centerline_f1
from mt_bench.cvat_import import read_frame_h5
from step1_read import source_of
from diag_fp_anatomy import resample as rs1
LAB = json.load(open(sys.argv[1]))["tiles"]; W = "/data2/prusek/mt_work/preds/"; DATA = "data/real/mt34_htw_eval"
TOL, COV, BORDER = 5.0, 0.80, 8.0
MODELS = {"AA_S3": 0.995, "AA_S4": 0.98, "AA_S5": 0.995, "AA_S6": 0.995, "DEB_S3": 0.995, "DEB_S4": 0.99, "DEB_S5": 0.99, "DEB_S6": 0.995, "THIN_S3": 0.999, "THIN_S4": 0.999, "THIN_S5": 0.995, "THIN_S6": 0.999, "DOSE_S3": 0.995, "THIN05_S3": 0.995, "DOSE_S4": 0.995, "THIN05_S4": 0.99, "DOSE_S5": 0.995, "THIN05_S5": 0.995, "DOSE_S6": 0.995, "THIN05_S6": 0.995}
def decode(prob, cut, dec):
    arcs, _, arms = graph(hmask(prob, cut, "bridge", 0.98, 22.5) if dec == "H" else prob > cut)
    if not arcs: return []
    polys = []
    for ch in walk_all(arcs, match(arms, {}, to_logit(prob))):
        p = np.concatenate([arcs[k] if w == "start" else arcs[k][::-1] for k, w in ch], 0)
        if arclen(p) >= P["min_length"]: polys.append(p)
    return polys
def off(path):
    name = os.path.basename(path); fr = read_frame_h5(path)
    if str(fr["attrs"].get("split", "")) != "val" or source_of(name) != "htw": return []
    prob = np.load(W + "AA_S3_hb_ep40_native/" + name.replace(".h5", ".npz"))["prob"].astype(np.float32)[0]; H, Wd = prob.shape
    gt = [np.asarray(q, float) * UP for q in fr["polylines"]]; polys = decode(prob, 0.995, "L")
    masks = [rasterise_polyline(resample(p, ds=0.5), prob.shape) for p in polys]
    r = centerline_f1(masks, gt, tol=TOL, length_coverage=COV, precision_coverage=COV, end_tol=0.0); mp = {q["pred"] for q in r["pairs"] if q["tp"]}
    gpx = [x for x in (_polyline_pixels(g, (H, Wd)) for g in gt if len(g) >= 2) if len(x)]; tr = cKDTree(np.concatenate(gpx, 0)) if gpx else None; out = []
    for i, p in enumerate(polys):
        if i in mp: continue
        c = rs1(np.asarray(p, float))[:, ::-1]
        if len(c) < 2: continue
        on = float((tr.query(c)[0] <= TOL).mean()) if tr is not None else 0.0
        e = np.minimum(np.minimum(c[:, 0], H - 1 - c[:, 0]), np.minimum(c[:, 1], Wd - 1 - c[:, 1]))
        if on >= 0.5 or float((e <= BORDER).mean()) >= 0.7: continue
        out.append({"frame": name, "len": len(c) / UP, "c": c.tolist()})
    return out
def surv(a):
    m, dec, name, tiles = a
    d = f"{W}{m}_hb_ep40_native/" + name.replace(".h5", ".npz"); prob = np.load(d)["prob"].astype(np.float32)[0]; polys = decode(prob, MODELS[m], dec)
    if not polys: return m, dec, [(t, False) for t, _ in tiles]
    tr = cKDTree(np.concatenate([rs1(np.asarray(p, float))[:, ::-1] for p in polys], 0))
    return m, dec, [(t, bool((tr.query(np.asarray(c))[0] <= TOL).mean() >= 0.5)) for t, c in tiles]
if __name__ == "__main__":
    with Pool(33) as pl:
        items = [x for o in pl.map(off, sorted(glob.glob(DATA + "/*.h5")), chunksize=1) for x in o]
        items.sort(key=lambda z: z["len"])
        assert len(items) == len(LAB), (len(items), len(LAB))
        bad = sum(z["frame"] != t["frame"] or abs(round(z["len"], 1) - t["len_native"]) > 0.11 for z, t in zip(items, LAB))
        print("alignment gate: tiles", len(items), "mismatches", bad, flush=True); assert bad == 0
        byf = collections.defaultdict(list)
        for i, z in enumerate(items): byf[z["frame"]].append((i, z["c"]))
        jobs = [(m, dec, f, byf[f]) for m, dec in [(m, "L") for m in MODELS if m != "AA_S3"] + [(m, "H") for m in MODELS if m.startswith(("DEB", "THIN", "DOSE"))] + [("AA_S3", "H")] for f in byf]
        res = pl.map(surv, jobs, chunksize=1)
    S = collections.defaultdict(dict)
    for m, dec, rows in res:
        for t, ok in rows: S[(m, dec)][t] = ok
    cls = [t["cls"] for t in LAB]; L = np.array([t["len_native"] for t in LAB])
    G = {"other AA + L": [("AA_S4", "L"), ("AA_S5", "L"), ("AA_S6", "L")], "DEB + L": [(f"DEB_S{k}", "L") for k in (3, 4, 5, 6)], "DEB + L+H": [(f"DEB_S{k}", "H") for k in (3, 4, 5, 6)], "same seed DEB_S3 + L": [("DEB_S3", "L")]}
    G = {"DEB + L+H": G["DEB + L+H"]}; G["DOSE + L+H"] = [(f"DOSE_S{k}", "H") for k in (3, 4, 5, 6)]; G["THIN05 + L+H"] = [(f"THIN05_S{k}", "H") for k in (3, 4, 5, 6)]; G["THIN + L+H"] = [(f"THIN_S{k}", "H") for k in (3, 4, 5, 6)]
    print("class   n  " + "  ".join(f"{g:>20s}" for g in G))
    out = {}
    for c in "UEDFN?":
        ix = [i for i, x in enumerate(cls) if x == c]; row = []
        for g, ms in G.items():
            v = float(np.mean([np.mean([S[k][i] for i in ix]) for k in ms])); row.append(v); out[f"{c}|{g}"] = v
        print(f"  {c}   {len(ix):3d} " + "  ".join(f"{v:20.3f}" for v in row))
    ix = list(range(len(cls))); print(" all  %3d " % len(ix) + "  ".join(f"{float(np.mean([np.mean([S[k][i] for i in ix]) for k in ms])):20.3f}" for ms in G.values()))
    json.dump(out, open("/data2/prusek/mt_work/runs/ult/anat/offgt_survive_64.json", "w"), indent=1); print("SURVIVE_DONE")
