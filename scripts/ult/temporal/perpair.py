import json, sys
import numpy as np
from scipy.spatial import cKDTree
sys.path.insert(0, SP := sys.argv[1] + '/hubcode/backend/segmentation')
from api.mt_geometry_cost import estimate_drift, resample
TOL = 5 / 1.5
D = {}
for k in (0, 1): D.update(json.load(open(f'{sys.argv[1]}/res/v16_{k}.json')))
def pts(cs):
    a = [resample(c, 1.0) for c in cs]
    return np.vstack(a) if a else np.zeros((0, 2))
def S(a, b):
    return ((cKDTree(b).query(a)[0] <= TOL).mean() + (cKDTree(a).query(b)[0] <= TOL).mean()) / 2
print('vid       S1 per pair: min p25 med p75 max | same with no drift step: med | drift px med max')
for v, x in sorted(D.items()):
    fr = sorted(x['frames']); C = [[np.asarray(c, float) for c in x['frames'][f]['polys'] if len(c) > 1] for f in fr]
    s, s0, dr = [], [], []
    for i in range(len(fr) - 1):
        d = estimate_drift(C[i], C[i + 1]); A, B = pts(C[i]), pts(C[i + 1])
        s.append(S(A + d, B)); s0.append(S(A, B)); dr.append(float(np.linalg.norm(d)))
    s = np.array(s)
    print(v[:8], ' '.join(f'{q:.2f}' for q in np.percentile(s, [0, 25, 50, 75, 100])), '|', f'{np.median(s0):.2f}', '|', f'{np.median(dr):.1f} {max(dr):.1f}', '| series', ' '.join(f'{q:.2f}' for q in s[:8]))
