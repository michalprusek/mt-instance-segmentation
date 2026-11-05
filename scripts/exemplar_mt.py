#!/usr/bin/env python3
"""Fit Exemplar (AutoSeg) on K real microtubule frames and dump its foreground probability maps.

WHY THIS ARM EXISTS. The failure decomposition says the FOREGROUND carries about 62 % of all
failures, and swapping it for an oracle removes them; the instancer is not the bottleneck. Every
foreground this project has built was trained on synthetic data, and the synthetic-to-real gap is
the documented wall. Exemplar attacks that gap from the other side: a frozen DINOv3 backbone plus a
small head fitted on K REAL annotated frames, no synthetic data at all.

WHAT CHANGES ABOUT THE CLAIM, stated here so it cannot be lost downstream: this arm sees real human
annotation, so it is NOT comparable to the synth-only arms or to TARDIS zero-shot without saying so.
It answers a different question -- "what do 8 real annotated frames buy?" -- and must be reported as
its own track. It does not violate the project's no-self-training rule, which is about pseudo-labels.

WHAT THIS SCRIPT DELIBERATELY DOES NOT DO. It does not score anything. It writes ``<out>/<stem>.npz``
with key ``prob`` as ``(1, H, W)`` float16 at the 1.5x evaluation shape -- byte-identical IN FORM to
what `eval_v5.py --dump-pred` writes for the binary-head arms, which are also single-channel -- so
the existing, already-verified threshold sweep, instancer and paired bootstrap run over it UNCHANGED,
and the instancer receives the same kind of input from both arms. Every line of scoring code not
written here is a coordinate bug that cannot be introduced; the (x, y) vs (row, col) transpose has
already cost this project three times.

PROTOCOL, fixed before the first score:
  * A seeded support POOL of 20 frames is drawn from roi303 VAL. K are subsampled from the pool per
    seed. The pool is excluded from prediction ENTIRELY, so the scored set is the same 121 frames for
    every seed and every arm -- a test slice that shifts with the support draw is the exact mistake
    Exemplar's own protocol section warns about.
  * TEST (140 frames) is never touched here.
  * Support masks are the polylines rasterised at the eval scale and dilated to half-width 1.0 --
    the SAME convention the current model was trained against, so "foreground" means the same thing
    in both arms. Declared once; sweeping it would be a new fitted knob.
  * Images are 1.5x-upscaled and 1/99-percentile stretched, identically for support and scored
    frames. roi303 mixes raw-tif and 8-bit-export frames, so a support/test normalisation mismatch
    would be invisible and fatal.

TWO STEPS, and the split is deliberate. The `seg` environment that carries DINOv3 has no h5py, and
the user has an AutoSeg campaign running against that very environment -- installing into it could
pull a different numpy and take their run down. So the h5 reading happens in the `nnunet`
environment (--export) and the fitting reads the exported cache (--fit). The eval-frame conversion
therefore happens exactly ONCE, which also removes any chance of the two steps normalising
differently.

    # 1. export the eval-frame cache, and the oracle ceiling, in an env that has h5py
    python scripts/exemplar_mt.py --export --oracle-out preds/ORACLE_roi
    # 2a. one head for the whole corpus -- MEASURED 2026-08-31 and it does not work (see below)
    python scripts/exemplar_mt.py --fit --out preds/EXEMPLAR_roi --seed 0
    # 2b. one head PER ACQUISITION, which is what the method is designed for
    python scripts/exemplar_mt.py --fit --per-session --k 4 --out preds/EXEMPLAR_SESS_roi

MEASURED, and it decides which of the two modes is right. A single head fitted on 8 frames drawn
from across the whole corpus reaches 88-98 % ground-truth coverage ON ITS OWN SUPPORT FRAMES -- the
fit is fine, it ran for 21 minutes and learned microtubules -- and then covers ~10 % of the 121
held-out ones. That is not a broken fit and not a broken integration; it is the method's documented
scope: it personalises to ONE acquisition ("the support and the query images come from the same
acquisition, the same modality"), and roi303 is 23 distinct acquisitions in one directory, from
730x732 to 2048x2048. Asking one head to span them asks for exactly what the method says it does not
do. --per-session fits one head per acquisition instead, which is also the deployment shape: a user
uploads their own acquisition and annotates a few frames of it.
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import sys

import numpy as np
from scipy.ndimage import binary_dilation, zoom
from skimage.morphology import disk

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))

from instance.metrics import rasterise_polyline  # noqa: E402

UP = 1.5
MASK_HALF_WIDTH = 1.0      # eval-frame pixels; the training target's convention, not a knob
METHOD = "head_fusion_best_nobank_flat_es_ep500_wd4_do5_mix"


def eval_image(img):
    """1.5x upscale then a 1/99 percentile stretch -- the project's eval convention."""
    big = zoom(np.asarray(img, np.float64), UP, order=1)
    lo, hi = np.percentile(big, [1, 99])
    return np.clip((big - lo) / (hi - lo + 1e-6), 0, 1).astype(np.float32)


def eval_mask(polys, shape):
    m = np.zeros(shape, bool)
    for q in polys:
        if len(q) >= 2:
            m |= rasterise_polyline(np.asarray(q, float) * UP, shape)
    r = int(round(MASK_HALF_WIDTH))
    return binary_dilation(m, disk(r)) if r >= 1 else m


def dump_prob(path, prob):
    """One writer for both arms, so the two can never differ in shape or dtype."""
    np.savez_compressed(path, prob=np.asarray(prob, np.float32)[None].astype(np.float16))


def split_indices(n, pool_size, pool_seed, limit=0):
    """The pool is drawn ONCE with its own seed, so every arm and every support draw is scored on
    the identical remaining frames."""
    pool = sorted(np.random.default_rng(pool_seed).permutation(n)[:pool_size].tolist())
    rest = [i for i in range(n) if i not in set(pool)]
    return pool, (rest[:limit] if limit else rest)


def do_export(args):
    import h5py
    paths = []
    for p in sorted(glob.glob(os.path.join(args.data_dir, "*.h5"))):
        with h5py.File(p, "r") as f:
            if str(f.attrs.get("split")) == args.split:
                paths.append(p)
    if not paths:
        raise SystemExit(f"no {args.split} frames under {args.data_dir}")
    os.makedirs(args.export_dir, exist_ok=True)
    if args.oracle_out:
        os.makedirs(args.oracle_out, exist_ok=True)
    pool, scored = split_indices(len(paths), args.pool, args.pool_seed, args.limit)
    names = []
    for i, p in enumerate(paths):
        with h5py.File(p, "r") as f:
            img = f["image"][()]
            polys = [np.asarray(f["polylines"][k][()], float) for k in sorted(f["polylines"])]
        im = eval_image(img)
        gt = eval_mask(polys, im.shape)
        stem = os.path.basename(p).replace(".h5", "")
        np.savez_compressed(os.path.join(args.export_dir, stem + ".npz"),
                            image=im.astype(np.float16), mask=gt)
        names.append(stem + ".npz")
        # the oracle: the ground truth ITSELF as the probability map, through the identical
        # downstream path. It is the denominator that turns "arm X scored Y" into "the foreground
        # gap is Z % closed", and on this clean benchmark it has never been measured.
        if args.oracle_out and i in set(scored):
            dump_prob(os.path.join(args.oracle_out, stem + ".npz"), gt.astype(np.float32))
    json.dump(dict(frames=names, data_dir=args.data_dir, split=args.split, up=UP,
                   mask_half_width=MASK_HALF_WIDTH),
              open(os.path.join(args.export_dir, "frames.json"), "w"), indent=1)
    print(f"exported {len(names)} frames -> {args.export_dir}")
    print(f"  support pool {len(pool)}, scored {len(scored)}")
    if args.oracle_out:
        print(f"  oracle probability maps -> {args.oracle_out}")


def sessions(export_dir, names):
    """Group frames by image shape, a proxy for acquisition session.

    It is the same proxy the split declaration used, and it is a proxy: frames of one session share
    the camera ROI, so equal shape is strong evidence of one acquisition and unequal shape is proof
    of more than one. Stated rather than assumed."""
    from collections import defaultdict
    g = defaultdict(list)
    for i, n in enumerate(names):
        shp = np.load(os.path.join(export_dir, n))["image"].shape
        g[tuple(int(v) for v in shp)].append(i)
    return g


def do_fit(args):
    idx = json.load(open(os.path.join(args.export_dir, "frames.json")))
    names = idx["frames"]
    pool, scored = split_indices(len(names), args.pool, args.pool_seed, args.limit)
    print(f"{len(names)} frames -> support pool {len(pool)}, scored {len(scored)}")

    sys.path.insert(0, args.autoseg)
    sys.path.insert(0, os.path.join(args.autoseg, "scripts"))
    from active_segmenter.config import EncoderConfig, RunConfig       # noqa: E402
    from active_segmenter.encoder.cached import CachedEncoder          # noqa: E402
    from active_segmenter.segment.base import LabeledExample           # noqa: E402
    from al_testbed import make_backend                                # noqa: E402

    cfg = RunConfig(device="auto", cache_dir=args.cache, encoder=EncoderConfig())
    dev = cfg.device_resolved()
    enc = CachedEncoder(cfg, dev, args.cache)
    be = make_backend(args.method, cfg, dev, enc=enc)

    def load(i):
        z = np.load(os.path.join(args.export_dir, names[i]))
        return z["image"].astype(np.float32), z["mask"]

    os.makedirs(args.out, exist_ok=True)
    rng = np.random.default_rng(args.seed)

    if args.per_session:
        groups = sessions(args.export_dir, names)
        keep = {s: ix for s, ix in groups.items() if len(ix) >= args.k + args.min_test}
        dropped = sum(len(ix) for s, ix in groups.items() if s not in keep)
        print(f"{len(groups)} acquisitions; {len(keep)} have >= {args.k}+{args.min_test} frames "
              f"and are used; {dropped} frames in sessions too small to support a fit are DROPPED "
              f"(named, not silently skipped)")
        for s, ix in sorted(groups.items(), key=lambda kv: -len(kv[1])):
            if s not in keep:
                print(f"    dropped {str(s):>14s}  {len(ix)} frames")
        plan = []
        for s, ix in sorted(keep.items(), key=lambda kv: -len(kv[1])):
            perm = rng.permutation(len(ix))
            sup = [ix[j] for j in perm[:args.k]]
            tst = [ix[j] for j in perm[args.k:]]
            plan.append((s, sup, tst))
    else:
        sup = [pool[j] for j in rng.permutation(len(pool))[:args.k]]
        plan = [(None, sup, scored)]

    all_sup, all_scored = [], []
    for s, sup, tst in plan:
        tag = f"session {s}: " if s is not None else ""
        print(f"\n{tag}fitting on {len(sup)} support frames, predicting {len(tst)}", flush=True)
        print("  support: " + ", ".join(names[i] for i in sup))
        if s is not None:
            be.reset_support_state()          # a head per acquisition; nothing carries over
        support = [LabeledExample(load(i)[0], enc.extract(load(i)[0]), load(i)[1].astype(int))
                   for i in sup]
        be.fit(support)
        for n, i in enumerate(tst, 1):
            im, _ = load(i)
            prob = np.asarray(be.foreground_prob(im, enc.extract(im)), np.float32)
            if prob.shape != im.shape:
                raise SystemExit(f"{names[i]}: probability map {prob.shape} != image {im.shape}; "
                                 "refusing to write a mismatched frame")
            dump_prob(os.path.join(args.out, names[i]), prob)
            if n % 10 == 0 or n == len(tst):
                print(f"    {n}/{len(tst)}", flush=True)
        all_sup += [names[i] for i in sup]
        all_scored += [names[i] for i in tst]

    json.dump(dict(method=args.method, seed=args.seed, pool_seed=args.pool_seed, k=args.k,
                   per_session=args.per_session, up=UP, mask_half_width=MASK_HALF_WIDTH,
                   support=all_sup, scored=all_scored),
              open(os.path.join(args.out, "manifest.json"), "w"), indent=1)
    print(f"\nwrote {len(all_scored)} maps + manifest.json -> {args.out}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", default="data/real/roi303_eval")
    ap.add_argument("--split", default="val", choices=["val"],
                    help="val only; TEST carries a declared one-shot exposure budget")
    ap.add_argument("--export", action="store_true",
                    help="h5 -> eval-frame cache (+ the oracle maps); run in the nnunet env")
    ap.add_argument("--fit", action="store_true",
                    help="cache -> probability maps; run in the seg env, which has DINOv3")
    ap.add_argument("--export-dir", default="/disk2/prusek/mt_work/.exemplar_frames")
    ap.add_argument("--oracle-out", default="", help="export mode: where to write the ceiling maps")
    ap.add_argument("--out", default="", help="fit mode: where to write the probability maps")
    ap.add_argument("--pool", type=int, default=20, help="support POOL size, held out from scoring")
    ap.add_argument("--k", type=int, default=8, help="support masks subsampled from the pool")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--pool-seed", type=int, default=20260830,
                    help="fixes WHICH frames form the pool; the same for every seed and every arm")
    ap.add_argument("--autoseg", default="/disk1/prusek/active-segmenter")
    ap.add_argument("--cache", default="/disk2/prusek/mt_work/.exemplar_cache")
    ap.add_argument("--method", default=METHOD)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--per-session", action="store_true",
                    help="fit ONE HEAD PER ACQUISITION (frames grouped by image shape) instead of "
                         "one head for the whole corpus. This is the setting the method is designed "
                         "for, and the one the deployment has.")
    ap.add_argument("--min-test", type=int, default=6,
                    help="per-session: a session must have this many frames left after the support "
                         "or it is dropped, and named in the output")
    args = ap.parse_args()
    if args.export == args.fit:
        raise SystemExit("give exactly one of --export (nnunet env) or --fit (seg env)")
    if args.fit and not args.out:
        raise SystemExit("--fit needs --out")
    (do_export if args.export else do_fit)(args)


if __name__ == "__main__":
    main()
