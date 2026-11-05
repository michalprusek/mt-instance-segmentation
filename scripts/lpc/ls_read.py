#!/usr/bin/env python3
"""Read ls_grid.py outputs.  ls_read.py select FILE...  (the 10:10 selection rule, per scorer, pooled over files; guards vs "A")
                               ls_read.py show KEY[,KEY] [htw] FILE...   (pooled then per file; reference = "A" if present else first key)"""
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
    return "%-12s F1 %.4f (%+.4f) tp/fp/fn %d/%d/%d  ident %.4f (%+.1f%%)  bundle %.4f (%+.1f%%)" % (
        k, f1(t), f1(t) - f1(a), t["tp"], t["fp"], t["fn"], i, 100 * (i / max(ia, 1e-9) - 1), b, 100 * (b / max(ba, 1e-9) - 1))


bt = lambda k: tuple(float(x[1:]) for x in k.split("_")[1:])
if mode == "select":
    T = pool(sys.argv[2:]); a = T["A"]; ia, ba = a["jp"] / a["jn"], a["bund"] / a["npar"]; print(line("A", a, a))
    for tag in sorted({k.split("_")[0] for k in T if k != "A"}):
        adm = {}
        for k in sorted((k for k in T if k.startswith(tag + "_")), key=bt):
            t = T[k]; ok = (t["jp"] / t["jn"]) / ia - 1 >= -0.02 and (t["bund"] / t["npar"]) / ba - 1 >= -0.02
            if ok: adm[k] = f1(t)
            print(line(k, t, a), "ADM" if ok else "fail")
        if not adm: print(tag, "NO ADMISSIBLE PAIR"); continue
        best = max(adm.values()); cand = [k for k, v in adm.items() if v >= best - 0.005]
        pick = min(cand, key=lambda k: (bt(k)[0], -bt(k)[1]))
        print("%s: best admissible %.4f; PICK %s F1 %.4f; beats A: %s" % (tag, best, pick, adm[pick], adm[pick] > f1(a)))
else:
    keys = sys.argv[2].split(","); htw = sys.argv[3] == "htw"; files = sys.argv[4:] if htw else sys.argv[3:]
    for fs in [files] + [[f] for f in files]:
        T = pool(fs, htw); ref = T.get("A", T[keys[0]])
        print("POOLED" if len(fs) > 1 else fs[0].split("/")[-1][:40])
        for k in (["A"] if "A" in T else []) + keys: print("   ", line(k, T[k], ref))
