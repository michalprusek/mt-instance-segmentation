#!/usr/bin/env python3
"""LPC screen S2, step 2: junction identity under four matchers (M0 instancer, M1 hand-feature logistic, M2 strip CNN,
M3 strip CNN with permuted labels) + map-only strip; also average precision on the S1 link sets.

    s2_eval.py TRAIN.npz VALSYN.npz OUT.json  J:NAME=PREFIX ...  L:NAME=FILE.npz ...
"""
import json, sys
import networkx as nx, numpy as np, torch, torch.nn as nn
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score
sys.argv, ARGS = sys.argv[:1], sys.argv[1:]
exec(open(__file__.replace("s2_eval.py", "s1_train.py")).read().split("tr, va = load(train_f), load(val_f)")[0].replace(
    "train_f, val_f, out = sys.argv[1:4]; tests = dict(a.split(\"=\", 1) for a in sys.argv[4:])", ""))
train_f, val_f, out = ARGS[:3]
JS = dict(a[2:].split("=", 1) for a in ARGS[3:] if a.startswith("J:")); LS = dict(a[2:].split("=", 1) for a in ARGS[3:] if a.startswith("L:"))
tr, va = load(train_f), load(val_f)
mu, sd = tr["feats"].mean(0), tr["feats"].std(0) + 1e-6
lr = LogisticRegression(C=1.0, max_iter=2000, class_weight="balanced").fit((tr["feats"] - mu) / sd, tr["y"])
nets = {}
for variant in ("full", "map"):
    nets[variant] = [fit(tr, va, variant, s)[0] for s in (0, 1)]
perm = dict(tr); rng = np.random.default_rng(0); perm["y"] = rng.permutation(tr["y"])
nets["perm"] = [fit(perm, va, "full", 0, epochs=6)[0]]


def score(d, key):
    if key == "C2": return lr.decision_function((d["feats"] - mu) / sd)
    v = "full" if key == "perm" else key
    return np.mean([predict(n, d, v) for n in nets[key]], 0)


def identity(J, s, sel=None):
    ok = n = 0
    for j in J:
        if sel is not None and not sel(j): continue
        g = nx.Graph()
        for q, (a, b) in enumerate(j["pairs"]):
            w = float(s[j["offset"] + q])
            if w > 0: g.add_edge(a, b, weight=w)
        m = {tuple(sorted(p)) for p in nx.max_weight_matching(g)} if g.number_of_edges() else set()
        ok += all(tuple(sorted(t)) in m for t in j["truth"]); n += 1
    return ok / max(n, 1), n


res = {}
for name, pref in JS.items():
    J = json.load(open(pref + ".json")); d = load(pref + ".npz"); r = {"n": len(J)}
    m0 = lambda sel=None: (sum(all(tuple(sorted(t)) in {tuple(sorted(m)) for m in j["m0"]} for t in j["truth"]) for j in J if sel is None or sel(j)),
                           sum(1 for j in J if sel is None or sel(j)))
    four = lambda j: j["n_arms"] == 4; other = lambda j: j["n_arms"] != 4
    a, n = m0(); r["M0"] = a / max(n, 1); a4, n4 = m0(four); ao, no = m0(other); r["M0_4arm"] = a4 / max(n4, 1); r["M0_other"] = ao / max(no, 1); r["n4"], r["nother"] = n4, no
    for key, lab in (("C2", "M1"), ("full", "M2"), ("perm", "M3"), ("map", "M2map")):
        s = score(d, key); r[lab] = identity(J, s)[0]; r[lab + "_4arm"] = identity(J, s, four)[0]; r[lab + "_other"] = identity(J, s, other)[0]
        r[lab + "_pairAUC"] = float(roc_auc_score(d["y"], s)) if 0 < d["y"].sum() < len(d["y"]) else None
    res["J:" + name] = r
    print(f"== junctions {name}: n {r['n']} (4-arm {r['n4']}, other {r['nother']}) identity M0 {r['M0']:.4f} | M1 hand {r['M1']:.4f} | M2 strip {r['M2']:.4f} | M2 map-only {r['M2map']:.4f} | M3 permuted {r['M3']:.4f}"
          f"\n     4-arm: M0 {r['M0_4arm']:.3f} M1 {r['M1_4arm']:.3f} M2 {r['M2_4arm']:.3f} map {r['M2map_4arm']:.3f} | other: M0 {r['M0_other']:.3f} M1 {r['M1_other']:.3f} M2 {r['M2_other']:.3f} map {r['M2map_other']:.3f}"
          f"\n     pair AUC: M1 {r['M1_pairAUC']} M2 {r['M2_pairAUC']} map {r['M2map_pairAUC']} perm {r['M3_pairAUC']}", flush=True)
for name, f in LS.items():
    d = load(f); r = {"prevalence": float(d["y"].mean())}
    for key in ("C2", "full", "map"): r["AP_" + key] = float(average_precision_score(d["y"], score(d, key)))
    r["AP_C1"] = float(average_precision_score(d["y"], d["feats"][:, 0])); res["L:" + name] = r
    print(f"== links {name}: prevalence {r['prevalence']:.3f} AP  C1 {r['AP_C1']:.3f}  C2 {r['AP_C2']:.3f}  strip {r['AP_full']:.3f}  map-only {r['AP_map']:.3f}", flush=True)
json.dump(res, open(out, "w"), indent=1)
