#!/usr/bin/env python3
"""Can the network localise CROSSINGS at all? The early gate for campaign arm 34.

Declared in docs/TODO.md 0.0-CAMPAIGN (2026-09-08). The arm adds a second logit meaning "two or
more instances cover this pixel". Before any F1 is read, the question that decides whether the
direction is worth building on is whether that logit carries signal. This measures average
precision of the overlap channel against two references:

  in domain  -- the generator's own multiplicity channel on synthetic frames, i.e. EXACT labels;
  on real    -- discs of radius ``--disc`` at the crossing points of the human GT polylines,
                which is an APPROXIMATION (a human polyline crossing is a point, the true overlap
                is a small region whose size depends on filament width), so the real AP is a
                lower bound and must never be compared to the in-domain number as if the two
                references were the same thing.

Chance level is printed next to every AP, because the positive class is ~0.1 % of pixels and a
bare AP is meaningless without it.

    PYTHONPATH=.:src:scripts:synth python scripts/diag_overlap_ap.py \
        --ckpt runs/OVERLAP/ep004.pth --synth data/synth_eval_s35 --real data/real/roi303_eval
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import sys

import numpy as np
import torch

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(_ROOT, "src"))
sys.path.insert(0, os.path.join(_ROOT, "scripts"))

from mt_bench.cvat_import import read_frame_h5  # noqa: E402
from mt_bench.gt_stats import count_crossings  # noqa: E402
import train_v5 as T  # noqa: E402


def average_precision(score: np.ndarray, target: np.ndarray) -> float:
    """AP as the area under the precision-recall curve, computed exactly from sorted scores."""
    order = np.argsort(-score)
    t = target[order].astype(np.float64)
    tp = np.cumsum(t)
    fp = np.cumsum(1.0 - t)
    n_pos = t.sum()
    if n_pos == 0:
        return float("nan")
    recall = tp / n_pos
    precision = tp / np.maximum(tp + fp, 1e-9)
    return float(np.sum(np.diff(np.concatenate([[0.0], recall])) * precision))


def overlap_target_synth(fr, shape):
    """Exact multiplicity from the stored polylines: a pixel covered by >= 2 rasterised centrelines."""
    from scipy.ndimage import binary_dilation
    cnt = np.zeros(shape, np.int16)
    for q in fr["polylines"]:
        q = np.asarray(q, float)
        seg = np.linalg.norm(np.diff(q, axis=0), axis=1)
        n = max(int(seg.sum()), 2)
        cum = np.concatenate([[0.0], np.cumsum(seg)])
        cum = cum / max(cum[-1], 1e-9)
        t = np.linspace(0, 1, n)
        pts = np.stack([np.interp(t, cum, q[:, 0]), np.interp(t, cum, q[:, 1])], 1)
        one = np.zeros(shape, bool)
        one[np.clip(pts[:, 1].astype(int), 0, shape[0] - 1),
            np.clip(pts[:, 0].astype(int), 0, shape[1] - 1)] = True
        cnt += binary_dilation(one, np.ones((3, 3))).astype(np.int16)
    return cnt >= 2


def overlap_target_real(fr, shape, disc):
    """Discs at GT polyline crossings. An APPROXIMATION -- see the module docstring."""
    from scipy.ndimage import distance_transform_edt
    pls = [np.asarray(q, float) for q in fr["polylines"]]
    seed = np.zeros(shape, bool)
    for c in count_crossings(pls):
        y, x = int(round(c["y"])), int(round(c["x"]))
        if 0 <= y < shape[0] and 0 <= x < shape[1]:
            seed[y, x] = True
    if not seed.any():
        return None
    return distance_transform_edt(~seed) <= disc


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--synth", default="data/synth_eval_s35")
    ap.add_argument("--real", default="data/real/roi303_eval")
    ap.add_argument("--split", default="val")
    ap.add_argument("--n", type=int, default=16, help="frames per source")
    ap.add_argument("--disc", type=float, default=3.0, help="real overlap disc radius, native px")
    ap.add_argument("--arch", default="nnunet")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    model = T.build_model(args.arch, 2).to(T.DEV)
    sd = torch.load(args.ckpt, map_location=T.DEV, weights_only=False)
    model.load_state_dict(sd, strict=False)
    model.eval()

    out = {"ckpt": args.ckpt, "disc": args.disc}
    for tag, d in (("in_domain", args.synth), ("real", args.real)):
        if not d or not os.path.isdir(d):
            continue
        scores, targets, chance = [], [], []
        for p in sorted(glob.glob(os.path.join(d, "*.h5")))[: args.n]:
            fr = read_frame_h5(p)
            if tag == "real" and str(fr["attrs"].get("split")) != args.split:
                continue
            img = T.norm01(np.asarray(fr["image"], float))
            ch = T.predict(model, img, T.TILE_FOR[args.arch])
            ov_logit = ch[1]
            tgt = (overlap_target_synth(fr, img.shape) if tag == "in_domain"
                   else overlap_target_real(fr, img.shape, args.disc))
            if tgt is None or not tgt.any():
                continue
            # subsample the negatives 1:200 so the AP is computable in memory; positives all kept
            pos = tgt.ravel()
            sc = ov_logit.ravel()
            idx_pos = np.flatnonzero(pos)
            idx_neg = np.flatnonzero(~pos)
            rng = np.random.default_rng(0)
            keep_neg = rng.choice(idx_neg, size=min(len(idx_neg), 200 * len(idx_pos)), replace=False)
            idx = np.concatenate([idx_pos, keep_neg])
            scores.append(sc[idx])
            targets.append(pos[idx])
            chance.append(float(pos.mean()))
        if not scores:
            continue
        s = np.concatenate(scores)
        t = np.concatenate(targets)
        # the subsample changes the base rate, so report AP against the SUBSAMPLED chance too
        ap_v = average_precision(s, t)
        out[tag] = {"ap": ap_v, "chance_full_frame": float(np.mean(chance)),
                    "chance_subsampled": float(t.mean()), "n_frames": len(scores),
                    "n_pos": int(t.sum())}
        print(f"{tag:10s} AP {ap_v:.4f}  (chance in the scored sample {t.mean():.4f}, "
              f"in a full frame {np.mean(chance):.5f})  frames {len(scores)}  pos {int(t.sum())}")
    if args.out:
        json.dump(out, open(args.out, "w"), indent=1)
        print("wrote", args.out)


if __name__ == "__main__":
    main()
