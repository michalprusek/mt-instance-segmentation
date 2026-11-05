import json, sys, numpy as np
res = json.load(open(sys.argv[1])); rows = []
fsat = {}
for name, rs, cnt, pairs, opairs, ocnt in res:
    m = {g for g, tp in pairs if tp}; om = {g for g, tp in (opairs or []) if tp}
    for r in rs:
        r["hit"] = r["k"] in m; r["ohit"] = (r["k"] in om) if opairs is not None else None; rows.append(r)
    fsat[name] = np.mean([r["sat"] for r in rs]) if rs else 0
R = rows; A = lambda k: np.array([r[k] for r in R], float)
hit = A("hit").astype(bool); L = A("len"); W = A("fwhm"); S = A("snr"); sat = A("sat")
print("curves", len(R), "hit", hit.sum(), "miss", (~hit).sum(), "recall %.3f" % hit.mean(), "| FWHM nan", np.isnan(W).sum())
def tab(name, x, edges, mask=None):
    mask = np.ones(len(R), bool) if mask is None else mask
    out = []
    for a, b in zip(edges[:-1], edges[1:]):
        s = mask & (x >= a) & (x < b)
        out.append("%g-%g: n %d miss %.2f" % (a, b, s.sum(), (~hit[s]).mean() if s.sum() else float("nan")))
    print(" ", name, "|", " | ".join(out))
LE = [0, 12, 20, 35, 60, 120, 1e9]; WE = [0, 2.0, 2.5, 3.0, 3.5, 4.5, 99]
tab("miss by length (native px)", L, LE); tab("miss by FWHM (native px)", W, WE); tab("miss by SNR", S, [0, 2, 4, 6, 10, 20, 1e9])
print("median FWHM hit %.2f miss %.2f | median len hit %.1f miss %.1f | median snr hit %.1f miss %.1f" % (np.nanmedian(W[hit]), np.nanmedian(W[~hit]), np.median(L[hit]), np.median(L[~hit]), np.median(S[hit]), np.median(S[~hit])))
print("== FWHM within LENGTH bins (is width a signal after length?)")
for a, b in zip(LE[:-1], LE[1:]):
    s = (L >= a) & (L < b) & ~np.isnan(W)
    if s.sum() < 20: continue
    med = np.median(W[s]); lo = s & (W < med); hi_ = s & (W >= med)
    print("  len %g-%g n %d  thin half (FWHM<%.2f) miss %.2f | wide half miss %.2f | FWHM-miss spearman %+.2f" % (a, b, s.sum(), med, (~hit[lo]).mean(), (~hit[hi_]).mean(), __import__("scipy.stats").stats.spearmanr(W[s], (~hit[s]).astype(float))[0]))
fs = np.array(sorted(fsat.values())); gap = np.argmax(np.diff(fs)); cut = (fs[gap] + fs[gap + 1]) / 2
cl = np.array([fsat[r["frame"]] >= cut for r in R])
print("== saturation clusters: frame-mean centerline saturation sorted", np.round(fs, 2), "cut %.2f" % cut)
for nm, s in (("low-sat frames", ~cl), ("high-sat frames", cl)):
    print(" ", nm, "frames", len({r["frame"] for r, c in zip(R, s) if c}), "curves", s.sum(), "miss %.3f" % (~hit[s]).mean(), "median FWHM %.2f len %.1f" % (np.nanmedian(W[s]), np.median(L[s])))
    tab("  miss by FWHM", W, WE, s); tab("  miss by length", L, LE, s)
# logistic regression, standardized
from sklearn.linear_model import LogisticRegression
ok = ~np.isnan(W); X = np.stack([np.log(L[ok]), W[ok], np.log(S[ok] + 1e-3), sat[ok]], 1); X = (X - X.mean(0)) / X.std(0); y = (~hit[ok]).astype(int)
fr = np.array([r["frame"] for r in R])[ok]; uf = sorted(set(fr)); rng = np.random.default_rng(0); B = []
for _ in range(500):
    pick = rng.choice(uf, len(uf)); idx = np.concatenate([np.where(fr == f)[0] for f in pick])
    B.append(LogisticRegression(C=10).fit(X[idx], y[idx]).coef_[0])
B = np.array(B); c = LogisticRegression(C=10).fit(X, y).coef_[0]
print("== logistic miss ~ log len, FWHM, log SNR, saturation (standardized; frame-cluster bootstrap 95 % CI)")
for n, v, lo, hi_ in zip(["log_len", "fwhm", "log_snr", "sat"], c, *np.percentile(B, [2.5, 97.5], 0)): print("  %-8s %+.2f [%+.2f, %+.2f]" % (n, v, lo, hi_))
oh = np.array([r["ohit"] for r in R]); 
if oh[0] is not None:
    oh = oh.astype(bool); print("== ORACLE band: missed", (~oh).sum(), "of", len(oh)); tab("oracle-miss by length", L, LE, None) if False else None
    for a, b in zip(LE[:-1], LE[1:]):
        s = (L >= a) & (L < b); print("  len %g-%g n %d oracle-miss %.2f model-miss %.2f" % (a, b, s.sum(), (~oh[s]).mean(), (~hit[s]).mean()))
    print("  model misses that the oracle ALSO misses: %d of %d" % ((~hit & ~oh).sum(), (~hit).sum()))
