#!/usr/bin/env python3
"""Shared plumbing for the `tune_diag_*` knob-sensitivity diagnostics. RUNS ON TULEN.

Everything here reuses `scripts/tune_instancer.py` -- the same `prepare()` (npz<->h5 pairing,
FOV masking, 1.5x frame) and the same metric calls (`oracle_instance_masks(half_width=1.0)`,
`centerline_f1(tol=5, length_coverage=0.95, precision_coverage=0.95)`, `aggregate_benchmark`).
A diagnostic that measured a different objective from the tuner it is meant to inform would be
worthless, so the objective is imported rather than re-implemented.

Two deliberate deviations from a tuning run:

* ``prob_thr`` is HELD FIXED at the shipped value. It is a decoding threshold, not an assembly
  knob, and it is the one dimension already known to dominate everything else -- leaving it in
  would swamp the ranking of the 13 knobs under test. The binary mask is therefore thresholded
  ONCE per frame at load time and the instancer is handed that mask directly.
* per-frame rows are kept for every configuration, so the paired bootstrap in
  ``tune_diag_bootstrap.py`` costs nothing extra.
"""
from __future__ import annotations

import json
import multiprocessing as mp
import os
import signal
import sys
import time

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))
sys.path.insert(0, os.path.join(ROOT, "scripts"))

from instance.instancer_a import instance_a            # noqa: E402
from instance.metrics import aggregate_benchmark, centerline_f1  # noqa: E402
from instance.oracle import oracle_instance_masks      # noqa: E402
from tune_instancer import prepare                     # noqa: E402

KAPPA_MAX = 0.25
SHIPPED = os.path.join(ROOT, "src", "instance", "params_a_model_synthtuned.json")
CALL_TIMEOUT_S = 90.0     # a single frame; pathological configs exist (tune_instancer: 400s/trial)

_FRAMES: list[dict] = []


def load_frames(data_dir: str, pred_dir: str, split: str, prob_thr: float) -> list[dict]:
    fr = prepare(data_dir, split, masks="model", pred_dir=pred_dir, prob_thr=prob_thr)
    for f in fr:
        f["prob2d"] = f["chans"].max(axis=0)
        f["fg_px"] = int(f["mask"].sum())
    return fr


def shipped_params() -> dict:
    with open(SHIPPED) as fh:
        p = json.load(fh)
    p.pop("kappa_max", None)
    return p


class _Timeout(Exception):
    pass


def _alarm(signum, frame):        # noqa: ARG001
    raise _Timeout()


def score_frame(f: dict, params: dict) -> dict:
    """One instancer call + the strict centerline-F1 row for one frame."""
    p = dict(params)
    p.pop("prob_thr", None)       # mask is pre-thresholded; see module docstring
    t0 = time.time()
    signal.signal(signal.SIGALRM, _alarm)
    signal.setitimer(signal.ITIMER_REAL, CALL_TIMEOUT_S)
    try:
        pls, _ = instance_a(f["mask"], KAPPA_MAX, p,
                            channels=f["chans"], prob=f["prob2d"])
        timeout = False
    except _Timeout:
        pls, timeout = [], True
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)
    masks = oracle_instance_masks(pls, f["hi_shape"], half_width=1.0, up=1.0)
    r = centerline_f1(masks, f["gt"], tol=5.0,
                      length_coverage=0.95, precision_coverage=0.95)
    return {**r, "n_gt": len(f["gt"]), "frame": f["name"],
            "secs": time.time() - t0, "timeout": timeout}


def _init(frames):
    global _FRAMES
    _FRAMES = frames
    os.environ["OMP_NUM_THREADS"] = "1"


def _work(job):
    cfg_id, frame_idx, params = job
    r = score_frame(_FRAMES[frame_idx], params)
    return cfg_id, r


def run_configs(frames: list[dict], configs: list[tuple[str, dict]],
                n_jobs: int = 24) -> dict[str, dict]:
    """Score every (config, frame) pair in parallel; return per-config rows + aggregate."""
    jobs = [(cid, i, prm) for cid, prm in configs for i in range(len(frames))]
    out: dict[str, list] = {cid: [] for cid, _ in configs}
    with mp.get_context("fork").Pool(n_jobs, initializer=_init, initargs=(frames,)) as pool:
        for cid, r in pool.imap_unordered(_work, jobs, chunksize=1):
            out[cid].append(r)
    res = {}
    for cid, rows in out.items():
        rows.sort(key=lambda r: r["frame"])
        agg = aggregate_benchmark(rows)
        res[cid] = {"mean_f1": float(agg["mean_f1"]), "rows": rows,
                    "secs": float(sum(r["secs"] for r in rows)),
                    "n_timeout": int(sum(r["timeout"] for r in rows))}
    return res


def save(path: str, obj) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as fh:
        json.dump(obj, fh, indent=1)
    print("wrote", path, flush=True)
