"""Is any SpheroSeg-verified frame already in a block we train-calibrate or score on?

The project has leaked twice (corpus <-> benchmark, roi303 VAL <-> TEST), both times through
anonymous file names, so this compares PIXELS: each query frame against every frame of the
calibration corpus, the benchmarks and the background set, by normalised cross-correlation of
background-flattened, 4x-reduced images. A central crop of one is searched inside the other, in
both directions, so a shifted field or an ROI cut out of a larger frame is still found.

Two things keep the reading honest. A positive control runs with it (each query against a
shifted, cropped, 8-bit, re-noised copy of itself; and MT-34's Alice frames against
``alice_eval``, the same twelve scenes exported twice) -- without one, "no match" could not be
told from "the test cannot fire". And flat references are set aside and counted: a saturated
frame has no variance, and normalised correlation against it returns 1.000 for EVERY query
(the first run of this script "found" all nine frames in one empty roi303 frame that way).

Run:  python scripts/diag_spheroseg_verified_leak.py [--root data/real]
"""
from __future__ import annotations

import argparse
import glob
import os

import cv2
import h5py
import numpy as np
import tifffile

REDUCE = 4          # work at 1/4 scale: a microtubule is still ~1 px wide, a frame is ~256-480 px
FLAT_SIGMA = 12.0   # px at the reduced scale; removes illumination, keeps filaments and dirt


MIN_LOCAL_STD = 0.25  # of the frame's interior std: below this a window is flat (saturated, masked)


def prep(img: np.ndarray) -> np.ndarray | None:
    """Reduced, flattened image in units of its INTERIOR std; ``None`` for a frame with no content.

    The scale comes from the central 60 %, not the whole frame: several benchmark frames have a
    saturated surround outside an octagonal field stop, whose edge would otherwise set the
    scale and make every window of real content look flat.
    """
    a = np.asarray(img, dtype=np.float32)
    if a.ndim == 3:
        a = a[..., :3].mean(axis=-1) if a.shape[-1] in (3, 4) else a[0]
    a = cv2.resize(a, (a.shape[1] // REDUCE, a.shape[0] // REDUCE), interpolation=cv2.INTER_AREA)
    a = a - cv2.GaussianBlur(a, (0, 0), FLAT_SIGMA)
    h, w = a.shape
    core = float(a[int(0.2 * h):int(0.8 * h), int(0.2 * w):int(0.8 * w)].std())
    if core < 1e-6:
        return None
    return a / core


def center(a: np.ndarray, frac: float = 0.5) -> np.ndarray:
    h, w = a.shape
    dh, dw = int(h * frac / 2), int(w * frac / 2)
    return a[h // 2 - dh:h // 2 + dh, w // 2 - dw:w // 2 + dw]


def ncc(a: np.ndarray, b: np.ndarray) -> float:
    """Best match of the central half of either image anywhere inside the other.

    Windows of the searched image that are flat are excluded: their normalised correlation is
    0/0, which OpenCV reports as a perfect match.
    """
    best = -1.0
    for big, small in ((a, center(b)), (b, center(a))):
        th, tw = small.shape
        if th > big.shape[0] or tw > big.shape[1] or min(th, tw) < 16 or float(small.std()) < MIN_LOCAL_STD:
            continue
        res = cv2.matchTemplate(big, small, cv2.TM_CCOEFF_NORMED)
        mean = cv2.boxFilter(big, -1, (tw, th), anchor=(0, 0), borderType=cv2.BORDER_CONSTANT)
        sq = cv2.boxFilter(big * big, -1, (tw, th), anchor=(0, 0), borderType=cv2.BORDER_CONSTANT)
        local_std = np.sqrt(np.maximum(sq - mean * mean, 0.0))[:res.shape[0], :res.shape[1]]
        res = np.where((local_std >= MIN_LOCAL_STD) & np.isfinite(res), res, -1.0)
        best = max(best, float(res.max()))
    return best


def load_h5(path: str) -> np.ndarray:
    with h5py.File(path, "r") as h:
        return h["image"][:]


def load_any(path: str) -> np.ndarray:
    if path.endswith(".h5"):
        return load_h5(path)
    if path.lower().endswith((".tif", ".tiff")):
        return tifffile.imread(path)
    return cv2.imread(path, cv2.IMREAD_UNCHANGED)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="data/real")
    args = ap.parse_args()
    r = args.root
    blocks = {
        "mt_corpus": sorted(glob.glob(os.path.join(r, "mt_corpus/images/*"))),
        "roi303": sorted(glob.glob(os.path.join(r, "roi303_eval/*.h5"))
                         + glob.glob(os.path.join(r, "roi303_eval/_excluded/*.h5"))),
        "mt34": sorted(glob.glob(os.path.join(r, "mt34_eval/*.h5"))),
        "htw": sorted(glob.glob(os.path.join(r, "htw_eval/h5/*.h5"))),
        "backgrounds": sorted(glob.glob(os.path.join(r, "irm_backgrounds_v2/*.tif"))),
    }
    queries = sorted(glob.glob(os.path.join(r, "spheroseg_verified/h5/*.h5")))
    q = [(os.path.basename(p)[:-3], prep(load_h5(p))) for p in queries]

    print("query frames against each other (adjacent stage positions could overlap):")
    for i in range(len(q)):
        for j in range(i + 1, len(q)):
            s = ncc(q[i][1], q[j][1])
            if s > 0.2:
                print(f"   {s:.3f}  {q[i][0]}  <->  {q[j][0]}")

    rng = np.random.default_rng(0)
    self_ctrl = []
    for p in queries:
        raw = load_h5(p).astype(np.float32)
        h, w = raw.shape
        lo, hi = np.percentile(raw, (0.5, 99.5))
        crop = raw[int(0.15 * h) + 37:int(0.75 * h) + 37, int(0.2 * w) - 23:int(0.8 * w) - 23]
        crop = np.clip((crop - lo) / (hi - lo) * 255.0, 0, 255)
        crop = np.round(crop + rng.normal(0.0, 6.0, crop.shape)).clip(0, 255).astype(np.uint8)
        self_ctrl.append(ncc(prep(raw), prep(crop)))
    print(f"\npositive control, each query vs a shifted 60 % crop of itself, 8-bit, noise added: "
          f"min {min(self_ctrl):.3f}  median {np.median(self_ctrl):.3f}")

    alice = sorted(glob.glob(os.path.join(r, "alice_eval/*.h5")))
    ctrl = [ncc(prep(load_h5(a)), prep(load_h5(os.path.join(r, "mt34_eval", os.path.basename(a)))))
            for a in alice if os.path.isfile(os.path.join(r, "mt34_eval", os.path.basename(a)))]
    print(f"positive control, same scene exported twice (alice_eval vs mt34_eval, n={len(ctrl)}): "
          f"min {min(ctrl):.3f}  median {np.median(ctrl):.3f}")

    print(f"\n{'block':12s} {'n':>5s} {'null p50':>8s} {'p99':>6s} {'max':>6s}   best match per query (score, query, reference)")
    for name, paths in blocks.items():
        refs, flat = [], []
        for p in paths:
            try:
                a = prep(load_any(p))
            except Exception as e:  # an unreadable reference must be reported, never skipped silently
                print(f"   UNREADABLE {p}: {e}")
                continue
            (refs if a is not None else flat).append((os.path.basename(p), a))
        scores = np.array([[ncc(qi, ri) for _, ri in refs] for _, qi in q])
        blind = sorted({refs[j][0] for j in np.where((scores < 0).any(axis=0))[0]})
        print(f"{name:12s} {len(refs):5d} {np.median(scores):8.3f} {np.percentile(scores, 99):6.3f} {scores.max():6.3f}"
              + (f"   ({len(flat)} flat frames set aside: {', '.join(n for n, _ in flat)[:90]})" if flat else "")
              + (f"   NOT COMPARABLE ({len(blind)}): {', '.join(blind)[:120]}" if blind else ""))
        for i, (qn, _) in enumerate(q):
            j = int(scores[i].argmax())
            print(f"   {scores[i, j]:.3f}  {qn[:44]:44s} {refs[j][0]}")


if __name__ == "__main__":
    main()
