#!/usr/bin/env python3
"""D4 (declared 2026-10-05): model-blind field-of-view groups of roi303 from runs/bench_v2/H.npy (high-pass image
correlation), threshold 0.20, connected components. Writes groups.json, the exclusion list, the STRUCTURE of the
re-split options (frame / curve counts only, NO scores), and the blind annotation sample."""
import json, numpy as np
from collections import defaultdict, Counter
B = "/disk2/prusek/mt_work/runs/bench_v2/"
H = np.load(B + "H.npy"); F = json.load(open(B + "frames.json")); N = len(F); THR = 0.20
par = list(range(N))
def find(x):
    while par[x] != x: par[x] = par[par[x]]; x = par[x]
    return x
for a in range(N):
    for b in range(a + 1, N):
        if H[a, b] >= THR: par[find(a)] = find(b)
g = defaultdict(list)
for a in range(N): g[find(a)].append(a)
groups = sorted(g.values(), key=lambda v: (-len(v), F[v[0]]["i"]))
gid = {}
for k, v in enumerate(groups):
    for a in v: gid[F[a]["name"]] = k
md = defaultdict(list)
for f in F: md[f["md5"]].append(f["name"])
dups = [sorted(v, key=lambda n: int(n.split("_")[-1][:-3])) for v in md.values() if len(v) > 1]
drop = sorted(v[-1] for v in dups)      # keep the lower index of each byte-identical pair
keep = [f for f in F if f["name"] not in drop]
sp = {f["name"]: f["split"] for f in F}; ngt = {f["name"]: f["n"] for f in F}
def tot(names): return len(names), sum(ngt[n] for n in names)
print("frames", N, "groups", len(groups), "singletons", sum(len(v) == 1 for v in groups), "sizes", [len(v) for v in groups[:12]])
print("byte-identical sets", dups, "-> excluded", drop)
for k, v in enumerate(groups[:8]):
    ix = sorted(F[a]["i"] for a in v); c = Counter(F[a]["split"] for a in v)
    print(f"  group {k}: {len(v)} frames, index {ix[0]}..{ix[-1]}, shape {F[v[0]]['shape']}, val {c['val']} test {c['test']}, curves {sum(F[a]['n'] for a in v)}")
names = [f["name"] for f in keep]
strad = {k for k, v in enumerate(groups) if len({F[a]["split"] for a in v if F[a]["name"] not in drop}) > 1}
cur_v = [n for n in names if sp[n] == "val"]; cur_t = [n for n in names if sp[n] == "test"]
print("current (after exclusion): VAL %d frames / %d curves, TEST %d / %d; TEST frames whose field also has a VAL frame: %d" % (*tot(cur_v), *tot(cur_t), sum(gid[n] in strad for n in cur_t)))
b1_t = [n for n in cur_t if gid[n] not in strad]; b1_v = [n for n in names if n not in b1_t]
print("B1 (straddling fields -> VAL): VAL %d / %d, TEST %d / %d, TEST groups %d" % (*tot(b1_v), *tot(b1_t), len({gid[n] for n in b1_t})))
# B2: deterministic group-wise split, large groups alternated by curve count, singletons keep their current side
order = sorted(strad, key=lambda k: -sum(ngt[F[a]["name"]] for a in groups[k]))
side = {}; acc = {"val": 0, "test": 0}
for k in order:
    s = "val" if acc["val"] <= acc["test"] else "test"; side[k] = s; acc[s] += sum(ngt[F[a]["name"]] for a in groups[k])
b2 = {n: (side[gid[n]] if gid[n] in side else sp[n]) for n in names}
b2_v = [n for n in names if b2[n] == "val"]; b2_t = [n for n in names if b2[n] == "test"]
moved = sum(1 for n in b2_t if sp[n] == "val")
print("B2 (group-wise, straddling fields assigned greedily by curve count, others stay): VAL %d / %d, TEST %d / %d, TEST groups %d, TEST frames that were VAL before: %d" % (*tot(b2_v), *tot(b2_t), len({gid[n] for n in b2_t}), moved))
print("B3 (keep split): effective clusters TEST %d, VAL %d" % (len({gid[n] for n in cur_t}), len({gid[n] for n in cur_v})))
json.dump(dict(threshold=THR, gid=gid, excluded=drop, b1={n: ("test" if n in b1_t else "val") for n in names}, b2=b2,
               current={n: sp[n] for n in names}), open(B + "groups.json", "w"), indent=0)
# blind annotation sample: 24 current-VAL frames with >= 10 curves, seeded; 2 per large field, the rest singletons
rng = np.random.default_rng(20261005); pick = []
for k, v in enumerate(groups):
    c = sorted(F[a]["name"] for a in v if F[a]["split"] == "val" and F[a]["n"] >= 10 and F[a]["name"] not in drop)
    if len(v) >= 6 and c: pick += list(rng.choice(c, size=min(2, len(c)), replace=False))
single = sorted(F[v[0]]["name"] for v in groups if len(v) < 6 for a in v[:1] if F[a]["split"] == "val" and F[a]["n"] >= 10)
pick += list(rng.choice(single, size=24 - len(pick), replace=False))
json.dump(sorted(pick, key=lambda n: int(n.split("_")[-1][:-3])), open(B + "annotation_sample_24.json", "w"))
print("annotation sample:", len(pick), sorted(int(n.split("_")[-1][:-3]) for n in pick), "curves", sum(ngt[n] for n in pick))
