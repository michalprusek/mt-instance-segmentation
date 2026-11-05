#!/usr/bin/env python3
"""Read instancer L grids.  a_read.py select FILE...   |   a_read.py read KEY [htw] FILE...   (pooled, then per file)"""
import json, sys
sys.path[:0] = ["scripts"]
mode = sys.argv[1]; f1 = lambda t: 2 * t["tp"] / max(2 * t["tp"] + t["fp"] + t["fn"], 1)


def pool(files, htw=False):
    T = {}
    for f in files:
        for name, ngt, res in json.load(open(f))["rows"]:
            if htw:
                from step1_read import source_of
                if source_of(name) != "htw": continue
            for k, r in res.items():
                t = T.setdefault(k, dict(tp=0, fp=0, fn=0, jn=0, jp=0, npar=0, bund=0.0))
                for q in ("tp", "fp", "fn", "jn", "jp"): t[q] += r[q]
                if r["bund"] is not None: t["npar"] += r["npar"]; t["bund"] += r["bund"]
    return T


def line(k, t, a):
    i, ia = t["jp"] / max(t["jn"], 1), a["jp"] / max(a["jn"], 1); b, ba = t["bund"] / max(t["npar"], 1), a["bund"] / max(a["npar"], 1)
    return "%-9s F1 %.4f (%+.4f) tp/fp/fn %d/%d/%d  ident %.4f (%+.1f%%)  bundle %.4f (%+.1f%%)" % (
        k, f1(t), f1(t) - f1(a), t["tp"], t["fp"], t["fn"], i, 100 * (i / max(ia, 1e-9) - 1), b, 100 * (b / max(ba, 1e-9) - 1))


if mode == "select":
    T = pool(sys.argv[2:]); a = T["A"]; ia, ba = a["jp"] / a["jn"], a["bund"] / a["npar"]; adm = {}
    print(line("A", a, a))
    for k in sorted((k for k in T if k != "A"), key=lambda k: tuple(float(x[1:]) for x in k.split("_"))):
        t = T[k]; ok = (t["jp"] / t["jn"]) / ia - 1 >= -0.02 and (t["bund"] / t["npar"]) / ba - 1 >= -0.02
        if ok: adm[k] = f1(t)
        print(line(k, t, a), "ADM" if ok else "fail")
    if not adm: print("NO ADMISSIBLE PAIR"); sys.exit()
    best = max(adm.values()); cand = [k for k, v in adm.items() if v >= best - 0.005]
    pick = min(cand, key=lambda k: (float(k.split("_")[0][1:]), -float(k.split("_")[1][1:])))
    print("best admissible %.4f; PICK %s F1 %.4f; beats A: %s" % (best, pick, adm[pick], adm[pick] > f1(a)))
else:
    key = sys.argv[2]; htw = sys.argv[3] == "htw"; files = sys.argv[4:] if htw else sys.argv[3:]
    T = pool(files, htw); print("POOLED   ", line(key, T[key], T["A"]), "| A F1 %.4f" % f1(T["A"]))
    for f in files:
        T = pool([f], htw); print(f.split("/")[-1][:34].ljust(34), line(key, T[key], T["A"]), "| A F1 %.4f" % f1(T["A"]))
