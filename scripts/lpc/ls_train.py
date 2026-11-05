#!/usr/bin/env python3
"""L-strip, step 2: train the map-only strip scorer on synthetic links (seeds 0 and 1), write the weights once, refit
the hand logistic on the same links as a control, and print the sanity numbers and the score quantiles the t grid is
declared from.

    ls_train.py TRAIN.npz VAL.npz OUT_DIR [NAME=FILE.npz ...]
"""
import json, os, sys
import numpy as np, torch, torch.nn as nn
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score
sys.path[:0] = [".", "src", "scripts", "scripts/lpc"]
from lpc_lib import strip_net, strip_scores, strip_x
dev = "cuda" if torch.cuda.is_available() else "cpu"
train_f, val_f, out = sys.argv[1:4]; tests = dict(a.split("=", 1) for a in sys.argv[4:])
load = lambda f: {k: v for k, v in np.load(f, allow_pickle=True).items()}


def fit(tr, va, seed, epochs=12, bs=256):
    torch.manual_seed(seed); np.random.seed(seed); net = strip_net().to(dev); y = tr["y"].astype(np.float32); n = len(y)
    opt = torch.optim.AdamW(net.parameters(), 1e-3, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, 2e-3, total_steps=epochs * (n // bs))
    pw = torch.tensor(float((1 - y).sum() / max(y.sum(), 1)), device=dev); best, state = -1, None
    for ep in range(epochs):
        net.train(); perm = np.random.permutation(n)
        for i in range(0, n - bs + 1, bs):
            idx = np.sort(perm[i:i + bs]); x = torch.from_numpy(strip_x(tr["logit"][idx], tr["lens"][idx])).to(dev)
            if np.random.rand() < 0.5: x = x.flip(2)                   # mirror across the path, as in S1
            loss = nn.functional.binary_cross_entropy_with_logits(net(x), torch.from_numpy(y[idx]).to(dev), pos_weight=pw)
            opt.zero_grad(); loss.backward(); opt.step(); sched.step()
        net.eval(); auc = roc_auc_score(va["y"], strip_scores([net], va["logit"], va["lens"], dev))
        if auc > best: best, state = auc, {k: v.detach().cpu().clone() for k, v in net.state_dict().items()}
        print(f"  seed {seed} ep {ep + 1} val AUC {auc:.4f}", flush=True)
    net.load_state_dict(state); net.eval(); return net, state


tr, va = load(train_f), load(val_f); T = {"val": va, **{k: load(v) for k, v in tests.items()}}
print("train", len(tr["y"]), "pos", int(tr["y"].sum()), {k: (len(d["y"]), int(d["y"].sum())) for k, d in T.items()}, flush=True)
X = tr["feats"].astype(float); mu, sd = X.mean(0), X.std(0) + 1e-6
lr = LogisticRegression(C=1.0, max_iter=2000, class_weight="balanced").fit((X - mu) / sd, tr["y"].astype(int))
h0 = os.path.join(out, "params_l_hand0.json"); assert not os.path.exists(h0), "write-once"
json.dump(dict(features=["mean_logit", "min_logit", "gap", "turn", "pair_cost"], mu=mu.tolist(), sd=sd.tolist(), coef=lr.coef_[0].tolist(),
               intercept=float(lr.intercept_[0]), _fit=f"CONTROL hand0: synthetic links from gap 0, {train_f}, 2026-10-06; write-once"), open(h0, "w"), indent=1)
old = json.load(open("src/instance/params_l_hand.json")); nets = []
for seed in (0, 1):
    f = os.path.join(out, f"l_strip_s{seed}.pt"); assert not os.path.exists(f), "write-once"
    net, state = fit(tr, va, seed); torch.save(state, f); nets.append(net)
for k, d in T.items():
    y = d["y"]; g = d["feats"][:, 2]; F = d["feats"].astype(float)
    sc = {"strip": strip_scores(nets, d["logit"], d["lens"], dev), "hand": ((F - np.asarray(old["mu"])) / np.asarray(old["sd"])) @ np.asarray(old["coef"]) + old["intercept"],
          "hand0": lr.decision_function((F - mu) / sd), "C1": F[:, 0]}
    for tag, m in (("all", np.ones(len(y), bool)), ("gap>=4", g >= 4), ("gap<4", g < 4)):
        if 0 < y[m].sum() < m.sum():
            print(f"== {k} {tag}: n {int(m.sum())} pos {int(y[m].sum())} | " + "  ".join(f"{n} AUC {roc_auc_score(y[m], s[m]):.4f} AP {average_precision_score(y[m], s[m]):.4f}" for n, s in sc.items()))
    for n in ("strip", "hand"):
        q = np.percentile(sc[n], [1, 5, 25, 50, 75, 95, 99]); qp = np.percentile(sc[n][y == 1], [5, 25, 50, 75]) if y.sum() else []
        print(f"   {n} log-odds quantiles 1/5/25/50/75/95/99: {np.round(q, 2)}  positives 5/25/50/75: {np.round(qp, 2)}")
