#!/usr/bin/env python3
"""D3 items 5-6: test-time augmentation and multi-scale prediction with v4b. RUNS ON TULEN.

Saves ALTERNATIVE max-over-channel probability maps in the same 1.5x eval frame as
``mt34_pred`` / ``synth_pred_v4b``. It deliberately does NOT save alternative K-channel
stacks for the instancer: the orientation channels are hard 30-degree bins, so a dihedral
transform PERMUTES them, and averaging un-permuted stacks would be wrong. Averaging the
max-over-channels map needs no permutation and is enough to build a mask, which is the only
thing the sweep changes (``channels``/``prob`` stay the raw identity maps).

The permuted-channel average is computed too, but only as a diagnostic + an extra
binarisation map, and the permutation is verified empirically before it is trusted.

    CUDA_VISIBLE_DEVICES=1 ~/dinov3_env/bin/python scripts/tta_predict.py \
        --data data/real/mt34_eval --split val --out /home/prusek/mt_enc_exp/mt34_pred_tta
"""
import argparse
import glob
import os
import sys
import time

os.environ.setdefault("SEG_MODE", "ori")
os.environ.setdefault("SEG_BACKBONE", "dinov2")
os.environ.setdefault("SEG_INPUT", "raw")
os.environ.setdefault("SEG_ARCH", "base")
os.environ.setdefault("SEG_WEIGHTS", "/home/prusek/mt_enc_exp/dino_seg_ori_v4b.pth")

import h5py  # noqa: E402
import numpy as np  # noqa: E402
from scipy.ndimage import zoom  # noqa: E402

sys.path.insert(0, "/home/prusek/mt_enc_exp/scripts")
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                                "src"))
import torch  # noqa: E402

import learn_amodal  # noqa: E402
from learn_amodal import norm01, prob_channels  # noqa: E402
from mt_bench.cvat_import import read_frame_h5  # noqa: E402

UP = 1.5
K = 6
TILE, STRIDE = learn_amodal.TILE, learn_amodal.STRIDE


@torch.no_grad()
def prob_channels_batched(img01, bs: int = 8):
    """Identical tiling/averaging to ``learn_amodal.prob_channels``, several tiles per forward.

    Verified against the original with ``--check-batch``: the maximum absolute difference must
    be at float32 round-off, otherwise every TTA number would be measuring a different model.
    """
    dev, m = learn_amodal.DEV, learn_amodal.m
    H, W = img01.shape
    acc = np.zeros((K, H, W)); cnt = np.zeros((H, W))
    ys = list(range(0, max(1, H - TILE + 1), STRIDE))
    xs = list(range(0, max(1, W - TILE + 1), STRIDE))
    if ys[-1] != H - TILE:
        ys.append(max(0, H - TILE))
    if xs[-1] != W - TILE:
        xs.append(max(0, W - TILE))
    coords = [(y, x) for y in ys for x in xs]
    full = [c for c in coords if img01[c[0]:c[0] + TILE, c[1]:c[1] + TILE].shape == (TILE, TILE)]
    part = [c for c in coords if c not in full]
    for i in range(0, len(full), bs):
        blk = full[i:i + bs]
        t = np.stack([img01[y:y + TILE, x:x + TILE] for y, x in blk]).astype(np.float32)
        tt = torch.from_numpy(t)[:, None].repeat(1, 3, 1, 1)
        tt = ((tt - learn_amodal.IMA_M) / learn_amodal.IMA_S).to(dev)
        o = torch.sigmoid(m(tt)).cpu().numpy()
        for j, (y, x) in enumerate(blk):
            acc[:, y:y + TILE, x:x + TILE] += o[j]
            cnt[y:y + TILE, x:x + TILE] += 1
    for y, x in part:
        t = img01[y:y + TILE, x:x + TILE]
        th, tw = t.shape
        tt = torch.from_numpy(t.astype(np.float32))[None].repeat(3, 1, 1)
        tt = ((tt - learn_amodal.IMA_M) / learn_amodal.IMA_S)[None].to(dev)
        o = torch.sigmoid(m(tt))[0].cpu().numpy()
        acc[:, y:y + th, x:x + tw] += o[:, :th, :tw]
        cnt[y:y + th, x:x + tw] += 1
    return acc / np.maximum(cnt, 1)[None]
#: The 8 dihedral transforms as (number of rot90s, horizontal flip).
D4 = [(k, f) for k in range(4) for f in (False, True)]


def fwd(a, k, fl):
    x = np.rot90(a, k, axes=(-2, -1))
    if fl:
        x = x[..., :, ::-1]
    return np.ascontiguousarray(x)


def inv(a, k, fl):
    x = a[..., :, ::-1] if fl else a
    return np.ascontiguousarray(np.rot90(x, -k, axes=(-2, -1)))


def chan_perm(k, fl):
    """Channel index permutation induced by the transform: out[b] = in[perm[b]].

    Bin b covers tangent angles [30b, 30b+30) with theta = atan2(dr, dc). A rot90 maps
    (dr, dc) -> (-dc, dr), i.e. theta -> theta - 90 deg = -3 bins; a horizontal flip maps
    (dr, dc) -> (dr, -dc), i.e. theta -> -theta, b -> (5 - b).
    """
    idx = np.arange(K)
    b = (idx - 3 * k) % K                       # undo the rotation
    if fl:
        b = (5 - b) % K
    return b


def predict_max(img):
    return prob_channels_batched(img).max(axis=0).astype(np.float32)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--split", default="val")
    ap.add_argument("--out", required=True)
    ap.add_argument("--ref-dir", default=None, help="saved baseline npz dir, for parity")
    ap.add_argument("--diag-only", action="store_true")
    ap.add_argument("--check-batch", action="store_true")
    ap.add_argument("--bs", type=int, default=8)
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)
    paths = []
    for p in sorted(glob.glob(os.path.join(args.data, "*.h5"))):
        with h5py.File(p) as f:
            if args.split == "all" or str(f.attrs.get("split")) == args.split:
                paths.append(p)
    print(f"{len(paths)} frames -> {args.out}", flush=True)

    for i, p in enumerate(paths):
        t0 = time.time()
        name = os.path.basename(p)
        raw = read_frame_h5(p)["image"].astype(float)
        img = zoom(norm01(raw), UP, order=1)

        ch_id = prob_channels_batched(img, args.bs).astype(np.float32)
        pmax_id = ch_id.max(axis=0)

        if args.check_batch and i == 0:
            ref0 = prob_channels(img).astype(np.float32)
            print(f"  BATCH-CHECK: max|batched-original| = "
                  f"{np.abs(ch_id - ref0).max():.3e}", flush=True)

        if args.ref_dir:
            ref = np.load(os.path.join(args.ref_dir,
                                       name.replace(".h5", ".npz")))["prob"].astype(np.float32)
            print(f"  PARITY {name}: max|ch_id-ref| = {np.abs(ch_id - ref).max():.5f} "
                  f"shape {ch_id.shape} vs {ref.shape}", flush=True)

        # ---- channel-permutation check (rot90, no flip) -------------------------------
        if i == 0:
            sel = pmax_id > 0.35
            a = ch_id.argmax(axis=0)[sel]
            for k, fl in [(1, False), (0, True), (1, True), (2, False)]:
                ch_t = prob_channels_batched(fwd(img, k, fl), args.bs).astype(np.float32)
                back = inv(ch_t, k, fl)[chan_perm(k, fl)]
                b = back.argmax(axis=0)[sel]
                b0 = inv(ch_t, k, fl).argmax(axis=0)[sel]     # unpermuted control
                print(f"  PERM-CHECK rot{90 * k}{'+flip' if fl else '':5s}: "
                      f"permuted {float((a == b).mean()):.3f}  "
                      f"unpermuted-control {float((a == b0).mean()):.3f}", flush=True)
        if args.diag_only:
            continue

        # ---- 8-fold dihedral, both polarities ------------------------------------------
        acc_d = np.zeros_like(pmax_id)
        acc_ch = np.zeros_like(ch_id)
        for k, fl in D4:
            if (k, fl) == (0, False):
                ct = ch_id
            else:
                ct = prob_channels_batched(fwd(img, k, fl), args.bs).astype(np.float32)
            acc_d += inv(ct.max(axis=0), k, fl)
            acc_ch += inv(ct, k, fl)[chan_perm(k, fl)]
        pmax_tta8 = acc_d / len(D4)
        pmax_tta8_perm = (acc_ch / len(D4)).max(axis=0)
        # polarity inversion: the identity transform on the inverted frame, and its average
        # with the identity prediction (the model has an explicit inversion augmentation)
        pmax_inv = prob_channels_batched(1.0 - img, args.bs).max(axis=0).astype(np.float32)
        pmax_tta8pol = 0.5 * (pmax_tta8 + pmax_inv)   # dihedral TTA + inverted identity
        pmax_polavg = 0.5 * (pmax_id + pmax_inv)      # inversion only

        # ---- multi-scale ---------------------------------------------------------------
        out = {"pmax_id": pmax_id, "pmax_tta8": pmax_tta8, "pmax_tta8_perm": pmax_tta8_perm,
               "pmax_inv": pmax_inv, "pmax_tta8pol": pmax_tta8pol, "pmax_polavg": pmax_polavg}
        hi = pmax_id.shape
        scale_maps = {}
        for s in (1.25, 2.0):
            m = predict_max(zoom(norm01(raw), s, order=1))
            m = zoom(m, UP / s, order=1)
            m = np.pad(m, ((0, max(0, hi[0] - m.shape[0])),
                           (0, max(0, hi[1] - m.shape[1]))))[:hi[0], :hi[1]]
            scale_maps[s] = m
            out[f"pmax_s{int(s * 100)}"] = m
        out["pmax_ms"] = (pmax_id + scale_maps[1.25] + scale_maps[2.0]) / 3.0

        np.savez_compressed(os.path.join(args.out, name.replace(".h5", ".npz")),
                            **{k2: v.astype(np.float16) for k2, v in out.items()})
        print(f"  {name:52s} [{time.time() - t0:.0f}s] "
              + " ".join(f"{k2}={100 * float((v > 0.4418).mean()):.2f}%"
                         for k2, v in out.items()), flush=True)


if __name__ == "__main__":
    main()
