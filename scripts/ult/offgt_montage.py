#!/usr/bin/env python3
"""Eye pass on htw off-annotation predictions (docs/TODO.md 2026-10-31 20:00 (1)). Decoding = anat_l.py (L-hand, bridge
60, t 0); an off_other prediction = unmatched, < 50 % of its 1-px samples within 5 eval px of any GT, not a border line.
Tile = 96 x 96 eval px around the prediction midpoint, raw image zoomed to the eval frame, 1-99 % stretch of the crop,
bare | overlay (prediction red, GT green), 2x. Sheets of 24 ordered by prediction length.
    python offgt_montage.py DATA PRED_DIR CUT OUTDIR"""
import glob, json, os, sys
import numpy as np
from multiprocessing import Pool
from PIL import Image, ImageDraw
from scipy.ndimage import zoom
from scipy.spatial import cKDTree
sys.path[:0] = [".", "src", "scripts", "scripts/lpc"]
from lpc_lib import UP, chain, link_edges
import lpc_lib as LL
from instance.metrics import _polyline_pixels, centerline_f1
from mt_bench.cvat_import import read_frame_h5
from step1_read import source_of
from diag_fp_anatomy import resample
data, pred, cut, od = sys.argv[1], sys.argv[2], float(sys.argv[3]), sys.argv[4]
MODEL = json.load(open("src/instance/params_l_hand.json")); TOL, COV, BORDER, HALF = 5.0, 0.80, 8.0, 48

def one(path):
    name = os.path.basename(path); fr = read_frame_h5(path)
    if str(fr["attrs"].get("split", "")) != "val" or source_of(name) != "htw": return []
    f = os.path.join(pred, name.replace(".h5", ".npz"))
    if not os.path.exists(f): return []
    prob = np.load(f)["prob"].astype(np.float32)[0]; H, W = prob.shape
    gt = [np.asarray(q, float) * UP for q in fr["polylines"]]
    arcs, edges = link_edges(prob, cut, MODEL, 60.0)
    polys = chain(arcs, edges, 60.0, 0.0, None) if arcs else []
    masks = [LL.rasterise_polyline(LL.resample(p, ds=0.5), prob.shape) for p in polys]
    r = centerline_f1(masks, gt, tol=TOL, length_coverage=COV, precision_coverage=COV, end_tol=0.0)
    mp = {q["pred"] for q in r["pairs"] if q["tp"]}
    gpx = [_polyline_pixels(g, (H, W)) for g in gt if len(g) >= 2]; gpx = [g for g in gpx if len(g)]
    gtree = cKDTree(np.concatenate(gpx, 0)) if gpx else None
    img = np.asarray(fr["image"], np.float32); img = zoom(img, (H / img.shape[0], W / img.shape[1]), order=1)
    out = []
    for i, p in enumerate(polys):
        if i in mp: continue
        c = resample(np.asarray(p, float))[:, ::-1]                      # (y, x)
        if len(c) < 2: continue
        on_gt = float((gtree.query(c)[0] <= TOL).mean()) if gtree is not None else 0.0
        edge = np.minimum(np.minimum(c[:, 0], H - 1 - c[:, 0]), np.minimum(c[:, 1], W - 1 - c[:, 1]))
        if on_gt >= 0.5 or float((edge <= BORDER).mean()) >= 0.7: continue
        my, mx = c[len(c) // 2].astype(int); y0 = int(np.clip(my - HALF, 0, H - 2 * HALF)); x0 = int(np.clip(mx - HALF, 0, W - 2 * HALF))
        crop = img[y0:y0 + 2 * HALF, x0:x0 + 2 * HALF]; lo, hi = np.percentile(crop, [1, 99])
        g = (np.clip((crop - lo) / (hi - lo + 1e-9), 0, 1) * 255).astype(np.uint8)
        gl = [(np.asarray(q, float) - [x0, y0]).tolist() for q in gt if len(q) >= 2]
        out.append({"frame": name, "len_native": len(c) / UP, "y0": y0, "x0": x0, "tile": g,
                    "pred": (np.asarray(p, float) - [x0, y0]).tolist(), "gts": gl, "mean_p": float(prob[np.clip(c[:, 0].astype(int), 0, H - 1), np.clip(c[:, 1].astype(int), 0, W - 1)].mean())})
    return out

def panel(t, pred=None, gts=()):
    im = Image.fromarray(t).convert("RGB").resize((4 * HALF, 4 * HALF), Image.NEAREST); d = ImageDraw.Draw(im)
    for q in gts:
        if len(q) >= 2: d.line([(2 * x, 2 * y) for x, y in q], fill=(0, 220, 0), width=1)
    if pred is not None: d.line([(2 * x, 2 * y) for x, y in pred], fill=(255, 40, 40), width=1)
    return im

if __name__ == "__main__":
    os.makedirs(od, exist_ok=True)
    with Pool(33) as pl: items = [x for o in pl.map(one, sorted(glob.glob(os.path.join(data, "*.h5"))), chunksize=1) for x in o]
    items.sort(key=lambda z: z["len_native"]); idx = []
    S, PW = 24, 4 * HALF
    for s in range(0, len(items), S):
        sheet = Image.new("RGB", (4 * (2 * PW + 12), 6 * (PW + 22)), (255, 255, 255)); d = ImageDraw.Draw(sheet)
        for k, z in enumerate(items[s:s + S]):
            n = s + k; R, C = divmod(k, 4); X, Y = C * (2 * PW + 12), R * (PW + 22)
            sheet.paste(panel(z["tile"]), (X, Y + 18)); sheet.paste(panel(z["tile"], z["pred"], z["gts"]), (X + PW + 2, Y + 18))
            d.text((X + 2, Y + 2), f"#{n}  {z['len_native']:.0f} px  p {z['mean_p']:.3f}", fill=(0, 0, 0))
            idx.append({"n": n, "frame": z["frame"], "len_native": round(z["len_native"], 1), "y0": z["y0"], "x0": z["x0"], "mean_p": round(z["mean_p"], 4)})
        sheet.save(os.path.join(od, f"sheet_{s // S:02d}.png"))
    json.dump(idx, open(os.path.join(od, "index.json"), "w"), indent=1)
    print("off_other", len(items), "sheets", (len(items) + S - 1) // S, "->", od)
