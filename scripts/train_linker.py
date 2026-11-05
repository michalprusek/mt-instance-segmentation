#!/usr/bin/env python3
"""Train the learned arm-pairing cost (``instance.linker``). RUNS ON TULEN / KAJMAN.

Training data is SYNTHETIC ONLY and the labels are exact: two skeleton arcs are a positive
pair when they belong to the same generated microtubule. That is a supervision signal real
data cannot give -- human polylines do not say which skeleton fragment continues which -- and
it is the reason this step can be learned without any annotation.

The masks are deliberately drawn from a MIXTURE of qualities:

* the predicted foreground of the current segmenter (what the linker will actually see),
* rasterised ground truth (clean junctions, so the geometric signal is learnable at all),
* ground truth DEGRADED with the defects measured on real data -- random spurs, punched holes
  and dilation noise.

Training on the predicted mask alone would tie the linker to one checkpoint's failure modes;
the mask factorial (docs/F1_PUSH_PLAN.md) shows those failure modes are exactly what is being
fixed elsewhere in the pipeline, so the linker must not overfit to them.

    PYTHONPATH=src:third_party python scripts/train_linker.py \
        --pred-dir /home/prusek/mt_enc_exp/synth_pred_v4b --out src/instance/linker_a.npz
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import sys

import numpy as np
from scipy.ndimage import binary_dilation, gaussian_filter
from scipy.spatial import cKDTree

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))
sys.path.insert(0, os.path.join(ROOT, "third_party"))

from instance.geometry import resample, total_length  # noqa: E402
from instance.instancer_a import KAPPA_DS, _ori_profile  # noqa: E402
from instance.linker import N_FEATURES, edge_features  # noqa: E402
from instance.matching import ArmEnd, pair_cost  # noqa: E402
from instance.oracle import oracle_mask  # noqa: E402
from instance.skeleton_graph import build_arc_graph  # noqa: E402
from mt_bench.cvat_import import read_frame_h5  # noqa: E402
from instance.geometry import window_tangent  # noqa: E402

UP = 1.5
KAPPA_MAX = 0.25


def degrade(mask: np.ndarray, rng) -> np.ndarray:
    """Add the defects the real predicted mask actually has: spurs, holes, ragged width.

    Sizes come from the measured comparison against the oracle -- the real predicted mask
    carries ~7x the connected components and 4.4x the loose ends at the same median width.
    """
    m = mask.copy()
    h, w = m.shape
    ys, xs = np.nonzero(m)
    if len(ys) == 0:
        return m
    # spurs: short protrusions off the filament, which is what creates spurious junction arms
    for _ in range(int(rng.integers(20, 80))):
        k = int(rng.integers(len(ys)))
        y, x = ys[k], xs[k]
        ang = rng.uniform(0, 2 * np.pi)
        L = rng.integers(3, 12)
        for t in range(L):
            yy = int(y + t * np.sin(ang))
            xx = int(x + t * np.cos(ang))
            if 0 <= yy < h and 0 <= xx < w:
                m[yy, xx] = True
    # holes: one-pixel dropouts that split a component
    for _ in range(int(rng.integers(10, 60))):
        k = int(rng.integers(len(ys)))
        y, x = ys[k], xs[k]
        r = int(rng.integers(1, 3))
        m[max(0, y - r):y + r + 1, max(0, x - r):x + r + 1] = False
    if rng.random() < 0.5:
        m = binary_dilation(m, iterations=1)
    return m


def arc_labels(arcs, gt_polys, tol: float = 3.0) -> np.ndarray:
    """Index of the ground-truth microtubule each arc belongs to, or -1.

    An arc is assigned to the microtubule that covers the largest share of its points; an arc
    whose best share is under half is left unassigned, because a spur that touches two real
    filaments must not manufacture a positive pair between them.
    """
    trees = [cKDTree(resample(np.asarray(p, float), ds=1.0)) for p in gt_polys if len(p) > 1]
    out = np.full(len(arcs), -1, dtype=int)
    if not trees:
        return out
    for i, a in enumerate(arcs):
        if len(a) < 2:
            continue
        share = [float((t.query(a, k=1)[0] <= tol).mean()) for t in trees]
        b = int(np.argmax(share))
        if share[b] >= 0.5:
            out[i] = b
    return out


def collect(frames, params, thr, rng, want_pred=True):
    """Feature/label pairs over junction candidates and free-end candidates."""
    X, Y = [], []
    for fr in frames:
        gt = [np.asarray(q, float) * UP for q in fr["polylines"] if len(q) > 1]
        if not gt:
            continue
        variants = []
        if want_pred and fr.get("prob") is not None:
            variants.append((fr["prob"].max(0) > thr, fr["prob"]))
        omask = oracle_mask(fr["polylines"], fr["image"].shape, half_width=1.0, up=UP)
        variants.append((omask, None))
        variants.append((degrade(omask, rng), None))
        for mask, prob in variants:
            g = build_arc_graph(mask, merge_radius=params["merge_radius"],
                                min_arc_len=int(params["min_arc_len"]),
                                bridge_max_len=params["bridge_max_len"])
            if not g.arcs:
                continue
            arcs = [resample(a, ds=params["ds"]) for a in g.arcs]
            lab = arc_labels(arcs, gt)
            # The bridge features read a CONTINUOUS probability field at inference time. Handing
            # them a hard 0/1 mask here would train three of the fourteen features on a
            # distribution that never occurs in service -- which is exactly what made the first
            # linker cut 111 true positives on real VAL. For the ground-truth-derived variants
            # there is no probability map, so one is synthesised with the soft falloff a real
            # probability field has around a filament.
            if prob is not None:
                p2 = prob.max(0)
            else:
                p2 = gaussian_filter(mask.astype(np.float32), 1.2)
                p2 = np.clip(p2 / max(p2.max(), 1e-6), 0, 1)
            chans = prob if prob is not None else None
            arms_by_j, free = {}, []
            for ai, (j0, j1) in enumerate(g.arc_ends):
                if len(arcs[ai]) < 2:
                    continue
                for which, jid in (("start", j0), ("end", j1)):
                    th, kp = window_tangent(arcs[ai], which, window=params["window"])
                    pos = arcs[ai][0] if which == "start" else arcs[ai][-1]
                    arm = ArmEnd(arc_idx=ai, which=which, theta=th, kappa=kp,
                                 pos=np.asarray(pos, float),
                                 ori=_ori_profile(chans, arcs[ai], which, params["window"]),
                                 arc_len=total_length(arcs[ai]))
                    if jid is None:
                        free.append(arm)
                    else:
                        arms_by_j.setdefault(jid, []).append(arm)

            groups = [(v, False) for v in arms_by_j.values()] + [(free, True)]
            for arms, is_gap in groups:
                if len(arms) < 2 or len(arms) > 40:
                    continue
                for i in range(len(arms)):
                    for j in range(i + 1, len(arms)):
                        a, b = arms[i], arms[j]
                        if a.arc_idx == b.arc_idx:
                            continue
                        d = float(np.linalg.norm(a.pos - b.pos))
                        if is_gap and d > params["link_max_gap"]:
                            continue
                        _, kap = pair_cost(a, b, None, gap_floor=params["gap_floor"])
                        if kap > KAPPA_MAX:
                            continue        # the hard bound also filters the training set
                        X.append(edge_features(a, b, prob=p2, gap_floor=params["gap_floor"],
                                               n_arms=len(arms), is_gap_link=is_gap))
                        Y.append(1 if (lab[a.arc_idx] >= 0
                                       and lab[a.arc_idx] == lab[b.arc_idx]) else 0)
    return (np.asarray(X, np.float32).reshape(-1, N_FEATURES),
            np.asarray(Y, np.float32))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/synth_eval")
    ap.add_argument("--split", default="val")
    ap.add_argument("--pred-dir", default=None)
    ap.add_argument("--params", default="src/instance/params_a_model_synthtuned.json")
    ap.add_argument("--out", default="src/instance/linker_a.npz")
    ap.add_argument("--epochs", type=int, default=300)
    ap.add_argument("--hidden", type=int, default=48)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    params = json.load(open(args.params))
    thr = params.get("prob_thr", 0.35)
    frames = []
    for p in sorted(glob.glob(os.path.join(args.data, "*.h5"))):
        fr = read_frame_h5(p)
        if str(fr["attrs"].get("split")) != args.split or not fr["polylines"]:
            continue
        if args.pred_dir:
            f = os.path.join(args.pred_dir, os.path.basename(p).replace(".h5", ".npz"))
            fr["prob"] = np.load(f)["prob"].astype(np.float32) if os.path.exists(f) else None
        frames.append(fr)
    print(f"{len(frames)} frames", flush=True)

    rng = np.random.default_rng(args.seed)
    X, Y = collect(frames, params, thr, rng, want_pred=bool(args.pred_dir))
    print(f"candidates: {len(Y)}  positives: {int(Y.sum())} ({100 * Y.mean():.1f}%)", flush=True)
    if len(Y) < 200:
        raise SystemExit("too few candidates to train on")

    # Frame-disjoint would be ideal; the pool is already frame-shuffled and this split is only
    # used to report generalisation, not to select anything that reaches the benchmark.
    idx = rng.permutation(len(Y))
    X, Y = X[idx], Y[idx]
    ntr = int(0.8 * len(Y))
    mu, sd = X[:ntr].mean(0), X[:ntr].std(0) + 1e-6

    import torch
    import torch.nn as nn
    torch.manual_seed(args.seed)
    xt = torch.tensor((X - mu) / sd)
    yt = torch.tensor(Y)[:, None]
    h = args.hidden
    net = nn.Sequential(nn.Linear(N_FEATURES, h), nn.ReLU(), nn.Linear(h, h), nn.ReLU(),
                        nn.Linear(h, 1))
    opt = torch.optim.AdamW(net.parameters(), 3e-3, weight_decay=1e-4)
    pos_w = torch.tensor([(1 - Y[:ntr].mean()) / max(Y[:ntr].mean(), 1e-6)])
    lossf = nn.BCEWithLogitsLoss(pos_weight=pos_w)
    for ep in range(args.epochs):
        net.train()
        perm = torch.randperm(ntr)
        for k in range(0, ntr, 4096):
            b = perm[k:k + 4096]
            opt.zero_grad()
            loss = lossf(net(xt[b]), yt[b])
            loss.backward()
            opt.step()
        if (ep + 1) % 60 == 0:
            net.eval()
            with torch.no_grad():
                pv = torch.sigmoid(net(xt[ntr:])).numpy().ravel()
            yv = Y[ntr:]
            acc = ((pv > 0.5) == (yv > 0.5)).mean()
            tp = ((pv > .5) & (yv > .5)).sum()
            prec = tp / max((pv > .5).sum(), 1)
            rec = tp / max((yv > .5).sum(), 1)
            print(f"  epoch {ep + 1}: holdout acc={acc:.3f} precision={prec:.3f} "
                  f"recall={rec:.3f}", flush=True)

    # Baseline: how well does the SHIPPED geometric cost separate the same candidates? Without
    # this the learned number has nothing to be better than.
    from instance.matching import pair_cost as _pc  # noqa: F401
    geo = -X[ntr:, 0] - 10.0 * X[ntr:, 2] - 0.02 * X[ntr:, 4]
    order = np.argsort(-geo)
    yv = Y[ntr:]
    k = int(yv.sum())
    print(f"\ngeometric-cost ranking: precision@{k} = {yv[order[:k]].mean():.3f}", flush=True)
    net.eval()
    with torch.no_grad():
        pv = torch.sigmoid(net(xt[ntr:])).numpy().ravel()
    order = np.argsort(-pv)
    print(f"learned ranking:        precision@{k} = {yv[order[:k]].mean():.3f}", flush=True)

    W = [net[0].weight.detach().numpy().T, net[2].weight.detach().numpy().T,
         net[4].weight.detach().numpy().T]
    B = [net[0].bias.detach().numpy(), net[2].bias.detach().numpy(),
         net[4].bias.detach().numpy()]
    np.savez(args.out, W0=W[0], W1=W[1], W2=W[2], b0=B[0], b1=B[1], b2=B[2],
             mu=mu.astype(np.float32), sd=sd.astype(np.float32),
             n_features=np.array([N_FEATURES]))
    print(f"wrote {args.out}", flush=True)


if __name__ == "__main__":
    main()
