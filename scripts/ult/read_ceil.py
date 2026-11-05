import json, numpy as np
U = "/data2/prusek/mt_work/runs/ult/ceil/"; MS = [f"DEB_S{k}" for k in (3, 4, 5, 6)]
f1 = lambda t: 2 * t[0] / max(2 * t[0] + t[1] + t[2], 1)
for blk in ("roiv2", "hb", "syn96b"):
    D = [json.load(open(f"{U}{blk}_{m}.json")) for m in MS]; T = {k: np.sum([d["total"][k] for d in D], 0) for k in ("L", "o60", "o100")}
    print(f"== {blk}: " + "  ".join(f"{k} F1 {f1(T[k]):.4f} (TP {T[k][0] / 4:.0f} FP {T[k][1] / 4:.0f} FN {T[k][2] / 4:.0f} id {T[k][4] / max(T[k][3], 1):.3f})" for k in T))
    print(f"   per model o60 - L: " + " ".join(f"{f1(d['total']['o60']) - f1(d['total']['L']):+.4f}" for d in D) + f" | ceiling gap pooled o60 {f1(T['o60']) - f1(T['L']):+.4f}  o100 {f1(T['o100']) - f1(T['L']):+.4f}")
    P = [p for d in D for p in d["pairs"]]; print(f"   right link lost to another GT's arc: {len(P) / 4:.0f} per model")
    if P:
        for k, sign in (("score", 1), ("gap", -1), ("turn", -1), ("turn30", -1), ("dlogit", -1), ("dwidth", -1), ("minlen", 1)):
            a = np.array([sign * (r[k] - w[k]) for r, w in P]); print(f"     {k:7s} right {np.median([r[k] for r, _ in P]):7.2f}  winner {np.median([w[k] for _, w in P]):7.2f}  P(right better) {np.mean(a > 0) + 0.5 * np.mean(a == 0):.3f}")
