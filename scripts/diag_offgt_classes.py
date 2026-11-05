#!/usr/bin/env python3
"""What the model's OFF-ANNOTATION predictions are, by count and by length, machine-classifiable part.

The montage of 2026-09-16 (`viz_offgt_fp.py`) showed three recurring things among the off_other
class: the FIELD-STOP EDGE (the hard bright/dark boundary of the illuminated field), short BRIGHT
RODS (unannotated short filaments / seeds), and defocused PARTICLES. Only the first is machine-
classifiable without a judgement call; this script measures it, and for the remainder records the
length and the contrast so the eye's classes can be sized.

Per off-annotation prediction (centerline < 50 % within 5 eval px of any GT, instancer at --thr):
  fov_edge   >= 50 % of its points OUTSIDE the field of view or within --edge-px of its boundary
             (`mt_bench.fov.fov_mask`, the benchmark's own detector, erode_px 0 so the boundary is
             the transition itself)
  length     eval px
  snr        median perpendicular-max |contrast| along it / the frame's robust noise (MAD of a
             high-passed frame), so "as bright as a filament" is a number
Also, for scale, the same snr statistic on the annotated GT curves of the frame.

Shards like diag_line_integral.py; --merge pools shard JSONs and prints the table.

    PYTHONPATH=src python scripts/diag_offgt_classes.py --pred-dir <dump> --data data/real/roi303_eval \\
        --split val --thr 0.98 --shard 0/6 --out offgt_0.json
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import sys

import numpy as np
from scipy.ndimage import distance_transform_edt, gaussian_filter
from scipy.spatial import cKDTree

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))

from instance.geometry import resample  # noqa: E402
from instance.instancer_a import instance_a  # noqa: E402
from mt_bench.cvat_import import read_frame_h5  # noqa: E402
from mt_bench.fov import fov_mask  # noqa: E402

UP, KAPPA_MAX = 1.5, 0.25


def normals(pts):
    d = np.gradient(pts, axis=0)
    n = np.stack([-d[:, 1], d[:, 0]], axis=1)
    return n / (np.linalg.norm(n, axis=1, keepdims=True) + 1e-9)


def contrast_along(img, pts, perp=2.0, bg_lo=6.0, bg_hi=12.0):
    H, W = img.shape
    nrm = normals(pts)
    offs_sig = np.arange(-perp, perp + 1e-9, 0.5)
    offs_bg = np.concatenate([np.arange(-bg_hi, -bg_lo + 1e-9, 1.0), np.arange(bg_lo, bg_hi + 1e-9, 1.0)])

    def sample(off):
        p = pts[:, None, :] + nrm[:, None, :] * off[None, :, None]
        cc = np.clip(np.rint(p[..., 0]).astype(int), 0, W - 1)
        rr = np.clip(np.rint(p[..., 1]).astype(int), 0, H - 1)
        return img[rr, cc]

    bg = np.median(sample(offs_bg), axis=1)
    return np.abs(sample(offs_sig) - bg[:, None]).max(axis=1)


def frame_noise(img):
    hp = img - gaussian_filter(img, 3.0)
    return 1.4826 * np.median(np.abs(hp - np.median(hp))) + 1e-9


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pred-dir", required=True)
    ap.add_argument("--data", default="data/real/roi303_eval")
    ap.add_argument("--split", default="val")
    ap.add_argument("--thr", type=float, required=True)
    ap.add_argument("--params", default="src/instance/params_a_derived.json")
    ap.add_argument("--edge-px", type=float, default=10.0, help="native px from the fov boundary")
    ap.add_argument("--shard", default="0/1")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    si, sn = (int(x) for x in args.shard.split("/"))
    params = json.load(open(args.params))

    rows = []
    gt_snr = []
    files = sorted(glob.glob(os.path.join(args.data, "*.h5")))[si::sn]
    for path in files:
        name = os.path.basename(path)
        fr = read_frame_h5(path)
        if str(fr["attrs"].get("split")) != args.split or not fr["polylines"]:
            continue
        npz = os.path.join(args.pred_dir, name.replace(".h5", ".npz"))
        if not os.path.exists(npz):
            continue
        ch = np.load(npz)["prob"].astype(np.float32)
        prob = ch.max(axis=0)
        img = np.asarray(fr["image"], float)
        H, W = img.shape
        sigma = frame_noise(img)
        gt = [resample(np.asarray(q, float), ds=1.0) for q in fr["polylines"] if len(q) >= 2]
        tree = cKDTree(np.concatenate(gt, 0))
        for g in gt:
            if len(g) >= 4:
                gt_snr.append(float(np.median(contrast_along(img, g)) / sigma))
        fov = fov_mask(img, erode_px=0)
        has_stop = bool((~fov).mean() > 0.005)
        # distance to the fov boundary: inside distance to the outside, outside = 0
        d_in = distance_transform_edt(fov) if has_stop else np.full((H, W), np.inf)
        polys, _ = instance_a(prob > args.thr, KAPPA_MAX, params=params, channels=ch, prob=prob)
        for pl in polys:
            pl = np.asarray(pl, float)
            if len(pl) < 2:
                continue
            nat = resample(pl / UP, ds=1.0)
            if len(nat) < 3:
                continue
            on = float((tree.query(nat)[0] <= 5.0 / UP).mean())
            if on >= 0.5:
                continue
            cc = np.clip(np.rint(nat[:, 0]).astype(int), 0, W - 1)
            rr = np.clip(np.rint(nat[:, 1]).astype(int), 0, H - 1)
            near_edge = float((d_in[rr, cc] <= args.edge_px).mean()) if has_stop else 0.0
            snr = float(np.median(contrast_along(img, nat)) / sigma)
            rows.append({"frame": name, "len": float(len(nat) * UP), "on_gt": on,
                         "fov_edge_frac": near_edge, "fov_edge": near_edge >= 0.5,
                         "snr": snr, "frame_has_stop": has_stop})
    with open(args.out, "w") as fh:
        json.dump({"rows": rows, "gt_snr": gt_snr, "thr": args.thr, "shard": args.shard}, fh)
    report(rows, gt_snr)


def report(rows, gt_snr):
    if not rows:
        print("no off-annotation predictions"); return
    L = np.array([r["len"] for r in rows]); E = np.array([r["fov_edge"] for r in rows])
    S = np.array([r["snr"] for r in rows]); G = np.array(gt_snr)
    tot = L.sum()
    print(f"off-annotation predictions: {len(rows)}  total length {tot:.0f} eval px  "
          f"(frames with a field stop hold {np.mean([r['frame_has_stop'] for r in rows]):.2f} of them)")
    print(f"  FIELD-STOP EDGE: {E.mean():.3f} by count, {L[E].sum() / tot:.3f} by length")
    rest = ~E
    for lo, hi, lab in ((0, 30, "<30 px"), (30, 60, "30-60"), (60, 120, "60-120"), (120, 1e9, ">=120")):
        m = rest & (L >= lo) & (L < hi)
        if m.any():
            print(f"  not edge, {lab:<7}: {m.mean():.3f} by count, {L[m].sum() / tot:.3f} by length, "
                  f"snr p25/p50/p75 {np.percentile(S[m], 25):.1f}/{np.percentile(S[m], 50):.1f}/{np.percentile(S[m], 75):.1f}")
    print(f"  annotated GT curves, snr p25/p50/p75: {np.percentile(G, 25):.1f}/{np.percentile(G, 50):.1f}/{np.percentile(G, 75):.1f}  (n={G.size})")
    thr_snr = np.percentile(G, 25)
    m = rest & (S >= thr_snr)
    print(f"  not edge AND at least as bright as the 25th-percentile annotated curve (snr >= {thr_snr:.1f}): "
          f"{m.mean():.3f} by count, {L[m].sum() / tot:.3f} by length")


def merge_main():
    rows, gt = [], []
    for f in sys.argv[2:]:
        d = json.load(open(f)); rows += d["rows"]; gt += d["gt_snr"]
    report(rows, gt)


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "--merge":
        merge_main()
    else:
        main()
