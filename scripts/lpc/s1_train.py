#!/usr/bin/env python3
"""LPC screen S1, step 2: train the strip verifier on synthetic links only and read it against the comparators.

    s1_train.py TRAIN.npz VALSYN.npz OUT.json NAME=FILE.npz [NAME=FILE.npz ...]

Variants: full (raw + map logit), map (logit only), raw (raw only); seeds 0 and 1. C1 = mean map logit along the
bridge; C2 = logistic regression on the five hand features, fitted on TRAIN.
"""
import json, sys
import numpy as np, torch, torch.nn as nn
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
dev = "cuda" if torch.cuda.is_available() else "cpu"
train_f, val_f, out = sys.argv[1:4]; tests = dict(a.split("=", 1) for a in sys.argv[4:])


def load(f):
    z = np.load(f, allow_pickle=True); n = z["lens"]
    s = np.arange(z["raw"].shape[2])[None]
    valid = (s < n[:, 2:3]).astype(np.float32); bridge = ((s >= n[:, 0:1]) & (s < n[:, 0:1] + n[:, 1:2])).astype(np.float32)
    return dict(raw=z["raw"].astype(np.float32), logit=z["logit"].astype(np.float32) / 12.0, valid=valid, bridge=bridge,
                y=z["y"].astype(np.float32), feats=z["feats"], frame=z["frame"])


def batch(d, idx, variant, aug):
    raw = torch.from_numpy(d["raw"][idx]); lg = torch.from_numpy(d["logit"][idx]); H = raw.shape[1]
    v = torch.from_numpy(d["valid"][idx])[:, None, :].expand(-1, H, -1); b = torch.from_numpy(d["bridge"][idx])[:, None, :].expand(-1, H, -1)
    if variant == "map": raw = torch.zeros_like(raw)
    if variant == "raw": lg = torch.zeros_like(lg)
    x = torch.stack([raw * v, lg * v, v, b], 1)
    if aug and np.random.rand() < 0.5: x = x.flip(2)                    # mirror across the path (valid: the strip has no handedness)
    return x.to(dev), torch.from_numpy(d["y"][idx]).to(dev)


class Net(nn.Module):
    def __init__(s):
        super().__init__()
        def blk(i, o): return nn.Sequential(nn.Conv2d(i, o, 3, padding=1), nn.BatchNorm2d(o), nn.ReLU(), nn.Conv2d(o, o, 3, padding=1), nn.BatchNorm2d(o), nn.ReLU())
        s.b1, s.b2, s.b3 = blk(4, 32), blk(32, 64), blk(64, 96); s.pool = nn.MaxPool2d(2); s.fc = nn.Sequential(nn.Linear(192, 64), nn.ReLU(), nn.Linear(64, 1))
    def forward(s, x):
        h = s.b3(s.pool(s.b2(s.pool(s.b1(x)))))
        return s.fc(torch.cat([h.mean((2, 3)), h.amax((2, 3))], 1))[:, 0]


@torch.no_grad()
def predict(net, d, variant):
    net.eval(); o = []
    for i in range(0, len(d["y"]), 1024):
        x, _ = batch(d, np.arange(i, min(i + 1024, len(d["y"]))), variant, False); o.append(net(x).float().cpu().numpy())
    return np.concatenate(o) if o else np.zeros(0)


def fit(tr, va, variant, seed, epochs=12, bs=256):
    torch.manual_seed(seed); np.random.seed(seed); net = Net().to(dev)
    opt = torch.optim.AdamW(net.parameters(), 1e-3, weight_decay=1e-4); n = len(tr["y"]); steps = epochs * (n // bs)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, 2e-3, total_steps=steps)
    pw = torch.tensor(float((1 - tr["y"]).sum() / max(tr["y"].sum(), 1)), device=dev); best, state = -1, None
    for ep in range(epochs):
        net.train(); perm = np.random.permutation(n)
        for i in range(0, n - bs + 1, bs):
            x, y = batch(tr, np.sort(perm[i:i + bs]), variant, True)
            loss = nn.functional.binary_cross_entropy_with_logits(net(x), y, pos_weight=pw)
            opt.zero_grad(); loss.backward(); opt.step(); sched.step()
        auc = roc_auc_score(va["y"], predict(net, va, variant))
        if auc > best: best, state = auc, {k: v.clone() for k, v in net.state_dict().items()}
        print(f"  {variant} seed {seed} ep {ep + 1} val AUC {auc:.4f}", flush=True)
    net.load_state_dict(state); return net, best


def boot(y, a, b, frame, nb=2000):
    rng = np.random.default_rng(0); fr = np.unique(frame); idx = {f: np.where(frame == f)[0] for f in fr}; o = []
    for _ in range(nb):
        s = np.concatenate([idx[f] for f in rng.choice(fr, len(fr))])
        if y[s].min() == y[s].max(): continue
        o.append(roc_auc_score(y[s], a[s]) - roc_auc_score(y[s], b[s]))
    return [float(v) for v in np.percentile(o, [2.5, 97.5])]


tr, va = load(train_f), load(val_f); T = {k: load(v) for k, v in tests.items()}
print("train", len(tr["y"]), "pos", int(tr["y"].sum()), "| val", len(va["y"]), "pos", int(va["y"].sum()), {k: (len(d["y"]), int(d["y"].sum())) for k, d in T.items()}, flush=True)
mu, sd = tr["feats"].mean(0), tr["feats"].std(0) + 1e-6
lr = LogisticRegression(C=1.0, max_iter=2000, class_weight="balanced").fit((tr["feats"] - mu) / sd, tr["y"])
res = {k: dict(n=len(d["y"]), pos=int(d["y"].sum()), C1=float(roc_auc_score(d["y"], d["feats"][:, 0])),
               C2=float(roc_auc_score(d["y"], lr.decision_function((d["feats"] - mu) / sd)))) for k, d in T.items()}
scores = {k: {} for k in T}
for variant in ("full", "map", "raw"):
    for seed in (0, 1):
        net, best = fit(tr, va, variant, seed)
        for k, d in T.items():
            s = predict(net, d, variant); scores[k][(variant, seed)] = s; res[k][f"{variant}_s{seed}"] = float(roc_auc_score(d["y"], s))
for k, d in T.items():
    r = res[k]; c2 = lr.decision_function((d["feats"] - mu) / sd)
    for v in ("full", "map", "raw"): r[v] = (r[f"{v}_s0"] + r[f"{v}_s1"]) / 2
    ens = (scores[k][("full", 0)] + scores[k][("full", 1)]) / 2
    r["full_minus_C1"] = r["full"] - r["C1"]; r["full_minus_C2"] = r["full"] - r["C2"]
    r["ci_full_minus_C1"] = boot(d["y"], ens, d["feats"][:, 0], d["frame"]); r["ci_full_minus_C2"] = boot(d["y"], ens, c2, d["frame"])
    gap = d["feats"][:, 2]; r["by_gap"] = {}
    for a, b in ((4, 12), (12, 24), (24, 40), (40, 61)):
        m = (gap >= a) & (gap < b)
        if m.sum() > 20 and 0 < d["y"][m].sum() < m.sum():
            r["by_gap"][f"{a}-{b}"] = dict(n=int(m.sum()), pos=int(d["y"][m].sum()), C1=float(roc_auc_score(d["y"][m], d["feats"][m, 0])),
                                           C2=float(roc_auc_score(d["y"][m], c2[m])), full=float(roc_auc_score(d["y"][m], ens[m])))
    print(f"== {k}: n {r['n']} pos {r['pos']} | C1 {r['C1']:.4f} C2 {r['C2']:.4f} | full {r['full']:.4f} ({r['full_s0']:.4f}/{r['full_s1']:.4f}) map-only {r['map']:.4f} raw-only {r['raw']:.4f}"
          f" | full-C1 {r['full_minus_C1']:+.4f} {r['ci_full_minus_C1']} full-C2 {r['full_minus_C2']:+.4f} {r['ci_full_minus_C2']}")
    for g, v in r["by_gap"].items(): print(f"     gap {g}: n {v['n']} pos {v['pos']} C1 {v['C1']:.3f} C2 {v['C2']:.3f} full {v['full']:.3f}")
json.dump(res, open(out, "w"), indent=1)
