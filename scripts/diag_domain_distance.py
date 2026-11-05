#!/usr/bin/env python3
"""Does feature-level domain alignment have a TARGET, or is the trunk already domain-invariant?

The 13-agent review ranked feature-level unsupervised alignment on the 522-frame unlabelled real
corpus as the one lane nobody has ruled on, and deliberately gave it no predicted number because
no measurement in this project supports one. This is that measurement, and it is a DIAGNOSTIC: it
trains nothing on real data and creates no label, so it does not touch the standing rules against
self-training or against MMD-FITTING morphology to the corpus. It only answers whether the trunk's
features are separable by domain at all.

Method: hook the activation the segmentation head reads, average-pool it per tile, and fit a
cross-validated linear domain classifier (render vs real). Reported as the PROXY A-DISTANCE
2*(2*acc - 1), the standard summary in the domain-adaptation literature.

    ~1.0  the domains are trivially separable  -> alignment has a large target
    ~0.0  the trunk is already domain-invariant -> alignment has nothing to align, and the gap
          measured at the OUTPUT (pixel AP 0.996 in domain vs 0.945 on real) lives in the head or
          in the decision, not in the representation.

    PYTHONPATH=src:scripts:synth python scripts/diag_domain_distance.py --ckpt <pth> \
        --real data/real/roi303_eval --synth data/synth_eval_s35 --n 40
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import sys

import numpy as np
import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
for sub in ("src", "scripts"):
    sys.path.insert(0, os.path.join(ROOT, sub))

from eval_v5 import checkpoint_in_channels  # noqa: E402
from mt_bench.cvat_import import read_frame_h5  # noqa: E402
from eval_v5 import local_norm  # noqa: E402
from train_v5 import build_model, norm01  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--real", required=True)
    ap.add_argument("--synth", required=True)
    ap.add_argument("--arch", default="nnunet")
    ap.add_argument("--tile", type=int, default=512)
    ap.add_argument("--n", type=int, default=40, help="tiles per domain")
    ap.add_argument("--norm-support", type=int, default=0,
                    help="normalise both domains at this percentile support (768 = the render's "
                         "own size). 0 reproduces the whole-frame behaviour and is CONFOUNDED.")
    ap.add_argument("--out", default="")
    args = ap.parse_args()

    state = torch.load(args.ckpt, map_location="cuda")
    state = state.get("model", state) if isinstance(state, dict) and "model" in state else state
    hw = [v for k, v in state.items() if k.endswith("head.weight") and v.ndim == 4]
    model = build_model(args.arch, out_ch=int(hw[0].shape[0]) if hw else 1,
                        in_ch=checkpoint_in_channels(state)).to("cuda").eval()
    model.load_state_dict(state, strict=False)

    feats = {}
    def hook(_m, inp, _out):
        feats["x"] = inp[0].detach()
    head = None
    for m in model.modules():
        if isinstance(m, torch.nn.Conv2d) and m.kernel_size == (1, 1):
            head = m
    if head is None:
        raise SystemExit("no 1x1 head conv found to hook")
    head.register_forward_hook(hook)
    print(f"hooked head conv with {head.in_channels} input channels", flush=True)

    def tiles(pattern, n):
        out = []
        for p in sorted(glob.glob(pattern)):
            fr = read_frame_h5(p)
            raw = np.asarray(fr["image"], float)
            # CONTROL for trap 6: the render is 768 px and real frames are 1024-2048, so
            # normalising each whole frame makes the SAME pixels arrive 3.2x weaker on the real
            # side. Without this the domain classifier would be reading the normalisation support,
            # not the imagery, and this measurement would repeat the error it is meant to avoid.
            img = local_norm(raw, args.norm_support) if args.norm_support > 0 else norm01(raw)
            h, w = img.shape
            if min(h, w) < args.tile:
                continue
            y, x = (h - args.tile) // 2, (w - args.tile) // 2
            out.append(img[y:y + args.tile, x:x + args.tile].astype(np.float32))
            if len(out) >= n:
                break
        return out

    X, y = [], []
    for dom, pat in ((0, os.path.join(args.synth, "*.h5")), (1, os.path.join(args.real, "*.h5"))):
        for t in tiles(pat, args.n):
            with torch.no_grad():
                x = torch.from_numpy(t)[None, None].repeat(1, checkpoint_in_channels(state), 1, 1).cuda()
                model(x)
            f = feats["x"].mean(dim=(2, 3))[0].float().cpu().numpy()
            X.append(f); y.append(dom)
        print(f"  domain {dom}: {sum(1 for v in y if v == dom)} tiles", flush=True)

    X = np.asarray(X); y = np.asarray(y)
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import cross_val_score
    from sklearn.preprocessing import StandardScaler
    from sklearn.pipeline import make_pipeline
    clf = make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000, C=1.0))
    acc = float(cross_val_score(clf, X, y, cv=5, scoring="accuracy").mean())
    summ = {"n_per_domain": int((y == 0).sum()), "feat_dim": int(X.shape[1]),
            "domain_classifier_accuracy": round(acc, 4),
            "proxy_A_distance": round(2 * (2 * acc - 1), 4),
            "reading": ("trunk is ALREADY domain-invariant; alignment has no target"
                        if acc < 0.65 else
                        "domains are trivially separable; alignment has a large target"
                        if acc > 0.90 else "partially separable")}
    if args.out:
        json.dump(summ, open(args.out, "w"), indent=1)
    print(json.dumps(summ, indent=1))


if __name__ == "__main__":
    main()
