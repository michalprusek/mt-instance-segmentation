"""In-domain ceiling of the declared instancer: a PERFECT band painted from the exact synthetic GT, decoded with
params_a_derived.json and scored with the declared metric (tol 5, cov 0.80/0.80). EXPLORATORY (docs/TODO.md 2026-10-05).
usage: synth_oracle.py DATA OUT.json [NPROC]   -- band half-widths 2.0 and 3.3 px at the 1.5x eval scale."""
import sys, os, glob, json, numpy as np
from multiprocessing import Pool
from scipy.ndimage import distance_transform_edt
sys.path[:0] = [".", "src", "scripts"]
from mt_bench.cvat_import import read_frame_h5
from mt_bench.gt_stats import count_crossings
from instance.instancer_a import instance_a
from instance.metrics import centerline_f1, junction_identity
P = json.load(open("src/instance/params_a_derived.json")); P = P.get("params", P)
UP, KAPPA = 1.5, 0.25
data, out = sys.argv[1], sys.argv[2]; NP = int(sys.argv[3]) if len(sys.argv) > 3 else 16


def one(path):
    fr = read_frame_h5(path); H, W = np.asarray(fr["image"]).shape
    h, w = int(round(H * UP)), int(round(W * UP)); gt = [np.asarray(q, float) * UP for q in fr["polylines"] if len(q) >= 2]
    line = np.zeros((h, w), bool)
    for q in gt:
        seg = np.linalg.norm(np.diff(q, axis=0), axis=1); n = max(2, int(seg.sum() * 4))
        t = np.concatenate([[0], np.cumsum(seg)]); s = np.linspace(0, t[-1], n)
        x = np.interp(s, t, q[:, 0]); y = np.interp(s, t, q[:, 1])
        ok = (x >= 0) & (x <= w - 1) & (y >= 0) & (y <= h - 1)
        line[np.round(y[ok]).astype(int), np.round(x[ok]).astype(int)] = True
    d = distance_transform_edt(~line); res = {}
    cr = count_crossings(gt, tol=2.0 * UP, endpoint_margin=8.0 * UP) if len(gt) > 1 else []
    for hw in (2.0, 3.3):
        mask = d <= hw; prob = mask.astype(np.float32)
        _, masks = instance_a(mask, KAPPA, params=P, channels=prob[None], prob=prob)
        r = centerline_f1(masks, gt, tol=5.0, length_coverage=0.8, precision_coverage=0.8, end_tol=0.0)
        ji = junction_identity(masks, gt, cr, tol=5.0, arm_offset=15.0 * UP) if cr else {}
        L = [float(np.linalg.norm(np.diff(q, axis=0), axis=1).sum()) / UP for q in gt]
        hit = {p["gt"] for p in r.get("pairs", []) if p["tp"]}
        res["hw%g" % hw] = dict(tp=r["tp"], fp=r["fp"], fn=r["fn"], jn=ji.get("n_crossings", 0), jp=ji.get("n_preserved", 0),
                                miss_len=[L[k] for k in range(len(gt)) if k not in hit], all_len=L)
    return os.path.basename(path), res


if __name__ == "__main__":
    paths = sorted(glob.glob(os.path.join(data, "*.h5")))
    with Pool(NP) as pool: R = pool.map(one, paths, chunksize=1)
    json.dump(R, open(out, "w"))
    E = [0, 12, 20, 35, 60, 120, 1e9]
    for k in ("hw2", "hw3.3"):
        tp = sum(r[1][k]["tp"] for r in R); fp = sum(r[1][k]["fp"] for r in R); fn = sum(r[1][k]["fn"] for r in R)
        jn = sum(r[1][k]["jn"] for r in R); jp = sum(r[1][k]["jp"] for r in R)
        al = np.concatenate([r[1][k]["all_len"] for r in R]); ml = np.concatenate([r[1][k]["miss_len"] for r in R])
        print(k, "F1 %.4f P %.3f R %.3f tp %d fp %d fn %d ident %.3f (%d/%d)" % (2 * tp / (2 * tp + fp + fn), tp / (tp + fp), tp / (tp + fn), tp, fp, fn, jp / max(jn, 1), jp, jn))
        print("   miss by length:", " | ".join("%g-%g n %d miss %.2f" % (a, b, ((al >= a) & (al < b)).sum(), ((ml >= a) & (ml < b)).sum() / max(((al >= a) & (al < b)).sum(), 1)) for a, b in zip(E[:-1], E[1:])))
