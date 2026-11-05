#!/usr/bin/env python3
"""Rotation probe: is the model's per-orientation weakness tied to the filament's angle to the
PIXEL GRID (model / renderer anisotropy) or to the image content (it follows the filament)?

Declared in docs/TODO.md 2026-09-16 20:55 before any read. For each VAL frame the normalised image
is rotated by 0, 90 (lossless), +45 and -45 degrees (bilinear, reflect padding), the model is run
exactly as eval_v5.py runs it (norm01, predict, native scale), the probability map is rotated back,
cropped to the frame, resampled to the eval frame, and the weak share (p < thr within +-2 eval px)
on plain stretches is read per ORIGINAL tangent bin (diag_angle_weak.py's sampling and bins).

Grid-orientation hypothesis: former-horizontal filaments GAIN p at +-45 (weak share falls toward the
oblique level), former-oblique filaments LOSE p (a swap); nothing changes at 90.
Content hypothesis: the weakness follows the filaments through every rotation (no swap).

    PYTHONPATH=.:src:scripts python scripts/diag_rotation_probe.py --ckpt runs/SPARSE35/ep040.pth \\
        --data data/real/mt34_htw_eval --split val --thr 0.995 --prefix htw_ --out rot_htw.json
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import sys

import numpy as np
from scipy.ndimage import rotate as nd_rotate
from scipy.spatial import cKDTree

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src")); sys.path.insert(0, os.path.join(ROOT, "scripts"))

from diag_angle_weak import BINS, END, FAR, normals, prob_at  # noqa: E402
from eval_v5 import eval_frame_shape, resample_to_eval  # noqa: E402
from instance.geometry import resample  # noqa: E402
from mt_bench.cvat_import import read_frame_h5  # noqa: E402
from mt_bench.gt_stats import count_crossings  # noqa: E402

ANGLES = (0.0, 90.0, 45.0, -45.0)


def rotate_fwd(a, deg):
    if deg == 0:
        return a
    if deg == 90:
        return np.rot90(a, 1)
    return nd_rotate(a, deg, reshape=True, order=1, mode="reflect")


def rotate_back(a, deg, shape):
    if deg == 0:
        return a
    if deg == 90:
        return np.rot90(a, -1)
    b = nd_rotate(a, -deg, reshape=True, order=1, mode="constant", cval=0.0)
    H, W = shape
    y0 = (b.shape[0] - H) // 2; x0 = (b.shape[1] - W) // 2
    return b[y0:y0 + H, x0:x0 + W]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--arch", default="nnunet")
    ap.add_argument("--data", required=True)
    ap.add_argument("--split", default="val")
    ap.add_argument("--thr", type=float, required=True)
    ap.add_argument("--prefix", default=None)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    if args.split == "test":
        raise SystemExit("VAL only")
    import torch
    from train_v5 import TILE_FOR, build_model, norm01, predict
    state = torch.load(args.ckpt, map_location="cuda")
    seg = [v for k, v in state.items() if "decoder.seg_layers." in k and k.endswith(".weight")]
    out_ch = int(seg[0].shape[0]) if seg else 1
    model = build_model(args.arch, out_ch=out_ch, in_ch=3).to("cuda").eval()
    model.load_state_dict(state, strict=False)

    per_angle = {str(a): {"wk": np.zeros(len(BINS) - 1), "tot": np.zeros(len(BINS) - 1), "frames": []} for a in ANGLES}
    n_frames = 0
    for path in sorted(glob.glob(os.path.join(args.data, "*.h5"))):
        name = os.path.basename(path)
        if args.prefix and not name.startswith(args.prefix):
            continue
        fr = read_frame_h5(path)
        if str(fr["attrs"].get("split")) != args.split or not fr["polylines"]:
            continue
        img = np.asarray(fr["image"], float)
        im01 = norm01(img)
        polys = [np.asarray(q, float) for q in fr["polylines"] if len(q) >= 2]
        cr = count_crossings(polys, tol=2.0, ds=1.0, endpoint_margin=8.0)
        cr_tree = cKDTree(np.array([[c["x"], c["y"]] for c in cr])) if cr else None
        samples = []   # (pts, ok, bin index) per polyline, angle-independent
        for q in polys:
            pts = resample(q, ds=1.0)
            if len(pts) < 2 * END + 5:
                continue
            _, d = normals(pts)
            ang = np.degrees(np.arctan2(d[:, 1], d[:, 0])) % 180.0
            ang = np.where(ang > 90.0, 180.0 - ang, ang)
            ok = np.ones(len(pts), bool); ok[: int(END)] = False; ok[-int(END):] = False
            if cr_tree is not None:
                ok &= cr_tree.query(pts)[0] >= FAR
            if ok.any():
                samples.append((pts, ok, np.clip(np.digitize(ang, BINS) - 1, 0, len(BINS) - 2)))
        if not samples:
            continue
        for a in ANGLES:
            x = np.ascontiguousarray(rotate_fwd(im01, a)).astype(np.float32)
            ch = predict(model, x, TILE_FOR[args.arch], bank=None)
            pn = np.asarray(ch)[:6].max(axis=0) if np.asarray(ch).ndim == 3 else np.asarray(ch)
            pn = rotate_back(pn, a, im01.shape)
            pe = resample_to_eval(pn[None].astype(np.float32), eval_frame_shape(im01.shape))[0]
            wk = np.zeros(len(BINS) - 1); tot = np.zeros(len(BINS) - 1)
            for pts, ok, b in samples:
                weak = prob_at(pe, pts) < args.thr
                for i in range(len(BINS) - 1):
                    m = ok & (b == i)
                    tot[i] += m.sum(); wk[i] += (weak & m).sum()
            per_angle[str(a)]["wk"] += wk; per_angle[str(a)]["tot"] += tot
            per_angle[str(a)]["frames"].append((wk.tolist(), tot.tolist()))
        n_frames += 1
        if args.limit and n_frames >= args.limit:
            break
    out = {"ckpt": args.ckpt, "data": args.data, "thr": args.thr, "prefix": args.prefix, "n_frames": n_frames, "bins": BINS.tolist(), "angles": {}}
    print(f"{n_frames} frames, thr {args.thr}; weak share per 15-deg bin of the ORIGINAL tangent (0..90) | overall")
    for a in ANGLES:
        d = per_angle[str(a)]
        share = (d["wk"] / np.maximum(d["tot"], 1)).tolist()
        overall = float(d["wk"].sum() / max(d["tot"].sum(), 1))
        out["angles"][str(a)] = {"weak_share_per_bin": share, "n_per_bin": d["tot"].tolist(), "overall": overall, "frames": d["frames"]}
        print(f"  rot {a:+6.1f}: " + " ".join(f"{v:.4f}" for v in share) + f" | {overall:.4f}")
    with open(args.out, "w") as fh:
        json.dump(out, fh, indent=1)


if __name__ == "__main__":
    main()
