#!/usr/bin/env python3
"""Deployment gate, map level + whole wrapper: MicrotubuleModel on real VAL frames vs the research dumps. Prints map max|dp| and flips at the cut; writes the wrapper's polylines at eps 0 (1.5x (x, y), identity gate) and at the shipped eps (scoring). usage: hub_e2e.py HUB_PKG WEIGHTS DATA DUMP NAMES.json CUT OUT_PREFIX [SHARD/NSHARDS]"""
import json, os, sys, time
import numpy as np
PKG, W, DATA, DUMP, NAMES, CUT, OUT = sys.argv[1:8]; CUT = float(CUT); SH = sys.argv[8] if len(sys.argv) > 8 else "0/1"; k, K = (int(x) for x in SH.split("/"))
sys.path[:0] = [os.path.dirname(PKG), PKG, os.path.join(PKG, "vendor")]
from wrapper import MicrotubuleModel, _simplify_polyline, UP
import h5py
names = json.load(open(NAMES))[k::K]; m = MicrotubuleModel().load_weights(W); print("params", m._params_path.name, "fusion", m.params.get("fusion"), "decoder", m.params.get("decoder"), "device", m._device, flush=True)
def read(p):
    with h5py.File(p, "r") as f: return np.asarray(f["image"])
R = {}; E = {}; S = []
for i, n in enumerate(names):
    img = read(os.path.join(DATA, n)); t0 = time.perf_counter(); maps = m.infer_maps(img); t1 = time.perf_counter()
    ref = np.load(os.path.join(DUMP, n.replace(".h5", ".npz")))["prob"].astype(np.float32)[0]; pe = maps["prob_eval"]
    assert pe.shape == ref.shape, (n, pe.shape, ref.shape)
    d = np.abs(pe.astype(np.float16).astype(np.float32) - ref); fl = int(((pe.astype(np.float16).astype(np.float32) > CUT) != (ref > CUT)).sum())
    t2 = time.perf_counter(); out = m.predict(img, seed_threshold=CUT, params={"polyline_eps_px": 0.0}); t3 = time.perf_counter()
    R[n] = [(np.asarray(c)[:, ::-1] * UP).tolist() for c in out["centerlines_rc"]]
    E[n] = [(_simplify_polyline(np.asarray(c), float(m.params["polyline_eps_px"]))[:, ::-1] * UP).tolist() for c in out["centerlines_rc"]]
    S.append(dict(name=n, shape=list(img.shape), maxdp=float(d.max()), flips=fl, t_maps=t1 - t0, t_predict=t3 - t2, n=len(R[n])))
    print(n, img.shape, f"max|dp| {d.max():.2e} flips {fl} maps {t1 - t0:.1f}s predict {t3 - t2:.1f}s n {len(R[n])}", flush=True)
json.dump(R, open(f"{OUT}_eps0_{k}.json", "w")); json.dump(E, open(f"{OUT}_rdp_{k}.json", "w")); json.dump(S, open(f"{OUT}_stats_{k}.json", "w")); print("E2E_DONE", k)
